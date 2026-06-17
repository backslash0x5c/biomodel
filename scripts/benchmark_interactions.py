#!/usr/bin/env python3
"""相互作用モジュール Φ の比較ベンチマーク（docs/08）。

線形 / 非線形の 2 レジームで、additive(=CPA相当) / FiLM / cross-attn / hypernet と
線形 ridge・population-mean を leave-one-donor-out で比較する。

  - 線形レジーム: gain = 1 + a_p·g_d → 線形 ridge が強い
  - 非線形レジーム: gain = 固定ランダム MLP(g_d, u_p) → ridge が崩れ、相互作用の差が出る

使い方:
    python scripts/benchmark_interactions.py
    python scripts/benchmark_interactions.py --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biomodel.evaluate import evaluate_predictions, population_mean_baseline, true_delta_test  # noqa: E402
from biomodel.model import PerturbationResponseModel  # noqa: E402
from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402
from biomodel.train import TrainConfig, predict_delta, pretrain_encoder, train_supervised  # noqa: E402


def ridge_predict(data, tr, te, alpha=1.0):
    Xtr = np.concatenate([np.ones((len(tr), 1), np.float32), data.geno[tr]], 1)
    Xte = np.concatenate([np.ones((len(te), 1), np.float32), data.geno[te]], 1)
    pred = np.zeros((len(te), data.n_perts, data.n_genes), np.float32)
    for p in range(data.n_perts):
        W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ data.delta[tr, p])
        pred[:, p] = Xte @ W
    return pred


def run_regime(name, scfg, epochs, pre, interactions, seed):
    data = simulate(scfg)
    tr, te = donor_split(data, n_test=max(20, scfg.n_donors // 8), seed=seed)
    y = true_delta_test(data, te)
    print(f"\n### regime = {name}  (donors train={len(tr)} test={len(te)}, "
          f"nonlinear={scfg.nonlinear})")
    rows = []
    for inter in interactions:
        t = time.time()
        torch.manual_seed(seed)
        model = PerturbationResponseModel(
            n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
            interaction=inter, use_genotype=True)
        cfg = TrainConfig(epochs=epochs, pretrain_epochs=pre, verbose=False, seed=seed)
        pretrain_encoder(model, data, cfg)
        train_supervised(model, data, tr, cfg)
        r = evaluate_predictions(inter, y, predict_delta(model, data, te))
        rows.append((r, time.time() - t))
    # 参照ベースライン
    rid = evaluate_predictions("ridge(linear)", y, ridge_predict(data, tr, te))
    pm = evaluate_predictions("population-mean", y, population_mean_baseline(data, tr, te))

    print(f"  {'module':<18}{'indiv_R2':>10}{'overall_r':>11}{'rank_rho':>10}{'sec':>7}")
    for r, dt in rows:
        print(f"  {r.name:<18}{r.indiv_r2:>+10.3f}{r.delta_pearson:>+11.3f}"
              f"{r.ranking_spearman:>+10.3f}{dt:>7.1f}")
    for r in (rid, pm):
        print(f"  {r.name:<18}{r.indiv_r2:>+10.3f}{r.delta_pearson:>+11.3f}"
              f"{r.ranking_spearman:>+10.3f}{'-':>7}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    interactions = ["additive", "film", "crossattn", "hypernet"]
    nd, ng, npn = (80, 40, 10) if args.quick else (200, 64, 12)
    epochs = min(args.epochs, 100) if args.quick else args.epochs

    print("=" * 74)
    print("相互作用モジュール Φ の比較（leave-one-donor-out, indiv_R2 が本命指標）")
    print("=" * 74)
    run_regime("linear", SimConfig(n_donors=nd, n_genes=ng, n_perts=npn, nonlinear=False,
                                   gain_scale=1.0, seed=args.seed),
               epochs, args.pretrain_epochs, interactions, args.seed)
    run_regime("nonlinear", SimConfig(n_donors=nd, n_genes=ng, n_perts=npn, nonlinear=True,
                                      gain_scale=1.0, seed=args.seed),
               epochs, args.pretrain_epochs, interactions, args.seed)
    print("\n" + "=" * 74)
    print("読み: 線形では ridge が上限に近い。非線形では ridge が崩れ、")
    print("      表現力の高い相互作用（cross-attn/hypernet）が FiLM/additive を上回りうる。")


if __name__ == "__main__":
    main()
