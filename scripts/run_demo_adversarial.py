#!/usr/bin/env python3
"""batch 敵対学習デモ（identifiability, docs/09）。

GRL による batch 敵対学習が encoder 表現を batch 不変にすることを、batch 線形 probe の
精度（chance まで下がれば不変）で確認する。個人差予測（indiv_R2）が損なわれないことも見る。

使い方:
    python scripts/run_demo_adversarial.py
    python scripts/run_demo_adversarial.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biomodel.adversarial import AdvConfig, batch_probe_accuracy, train_supervised_adversarial  # noqa: E402
from biomodel.evaluate import evaluate_predictions, population_mean_baseline, true_delta_test  # noqa: E402
from biomodel.model import PerturbationResponseModel  # noqa: E402
from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402
from biomodel.train import predict_delta, pretrain_encoder  # noqa: E402


def build_train(data, tr, adv_weight, epochs, pre, seed):
    torch.manual_seed(seed)
    model = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
        interaction="film", use_genotype=True)
    cfg = AdvConfig(epochs=epochs, pretrain_epochs=pre, verbose=False, seed=seed,
                    adv_weight=adv_weight)
    pretrain_encoder(model, data, cfg)
    train_supervised_adversarial(model, data, tr, cfg)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--pretrain-epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    nd = 80 if args.quick else 120
    if args.quick:
        args.epochs = min(args.epochs, 80)
    # batch effect を明瞭に与え、個人差は genotype 由来にする（baseline は batch を含む）。
    # これで batch（z_base 経由）と個人差（genotype 経由）が分離可能な状況になる。
    scfg = SimConfig(n_donors=nd, n_genes=48, n_perts=10, batch_effect=0.6,
                     n_batches=4, genotype_drives_baseline=0.0, seed=args.seed)
    data = simulate(scfg)
    tr, te = donor_split(data, n_test=max(16, nd // 8), seed=args.seed)
    n_batches = scfg.n_batches
    y = true_delta_test(data, te)

    print("=" * 74)
    print("batch 敵対学習デモ — identifiability（真の生物差 vs batch effect）")
    print("=" * 74)
    print(f"ドナー train={len(tr)} test={len(te)}  batch数={n_batches} "
          f"(chance精度={1/n_batches:.2f})  batch_effect={scfg.batch_effect}\n")

    rows = []
    for label, w in [("標準学習 (adv=0)", 0.0), ("敵対学習 (adv=1.0)", 1.0)]:
        model = build_train(data, tr, w, args.epochs, args.pretrain_epochs, args.seed)
        probe = batch_probe_accuracy(model, data, tr, seed=args.seed)
        res = evaluate_predictions(label, y, predict_delta(model, data, te))
        rows.append((label, probe, res))

    base = evaluate_predictions("population-mean", y, population_mean_baseline(data, tr, te))
    print("結果（leave-one-donor-out）")
    print("-" * 74)
    print(f"  {'手法':<18}{'batch_probe精度':>16}{'indiv_R2':>12}{'rank_rho':>10}")
    for label, probe, res in rows:
        print(f"  {label:<18}{probe:>16.3f}{res.indiv_r2:>+12.3f}{res.ranking_spearman:>+10.3f}")
    print(f"  {'population-mean':<18}{'-':>16}{base.indiv_r2:>+12.3f}{base.ranking_spearman:>+10.3f}")
    print("=" * 74)
    std_probe = rows[0][1]; adv_probe = rows[1][1]
    print(f"判定: batch_probe 精度 {std_probe:.2f}(標準) -> {adv_probe:.2f}(敵対)  "
          f"chance={1/n_batches:.2f}")
    if adv_probe < std_probe - 0.05:
        print("  => 敵対学習で表現が batch 不変に近づいた ✅（identifiability の改善）")
    else:
        print("  => 効果が小さい。adv_weight/grl_lambda を調整。")


if __name__ == "__main__":
    main()
