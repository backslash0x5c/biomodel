#!/usr/bin/env python3
"""cell-level + 分布マッチング（MMD）デモ（docs/06）。

pseudobulk ではなく細胞分布を予測し、予測細胞群と観測細胞群（unpaired）の MMD を
最小化して学習する。leave-one-donor-out で (a) 個人差 pseudobulk 精度（indiv_R2）と
(b) 分布レベル精度（energy distance）を population-mean 対照と比較する。

使い方:
    python scripts/run_demo_celllevel.py
    python scripts/run_demo_celllevel.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biomodel.cell_level import (  # noqa: E402
    CellTrainConfig,
    distribution_energy,
    predict_pseudobulk_delta,
    pretrain_encoder_cells,
    train_cell_level,
)
from biomodel.evaluate import evaluate_predictions, population_mean_baseline, true_delta_test  # noqa: E402
from biomodel.model import CellLevelResponseModel  # noqa: E402
from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--pretrain-epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["mmd", "sinkhorn"], default="mmd",
                    help="分布マッチング損失（mmd or sinkhorn=最適輸送, docs/06）")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        scfg = SimConfig(n_donors=50, n_genes=32, n_perts=5, n_control_cells=96, seed=args.seed)
        args.epochs = min(args.epochs, 15)
    else:
        scfg = SimConfig(n_donors=80, n_genes=40, n_perts=6, n_control_cells=128, seed=args.seed)

    print("=" * 74)
    print("cell-level + MMD デモ — unpaired 分布マッチング（leave-one-donor-out）")
    print("=" * 74)
    data = simulate(scfg)
    tr, te = donor_split(data, n_test=max(12, scfg.n_donors // 6), seed=args.seed)
    print(f"ドナー train={len(tr)} test={len(te)}  遺伝子={data.n_genes} 摂動={data.n_perts} "
          f"control細胞/ドナー={scfg.n_control_cells}\n")

    torch.manual_seed(args.seed)
    model = CellLevelResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
        interaction="film", use_genotype=True)
    cfg = CellTrainConfig(epochs=args.epochs, pretrain_epochs=args.pretrain_epochs,
                          seed=args.seed, loss_type=args.loss)
    pretrain_encoder_cells(model, data, cfg)
    train_cell_level(model, data, tr, cfg)

    y = true_delta_test(data, te)
    pred = predict_pseudobulk_delta(model, data, te, seed=args.seed)
    res = evaluate_predictions(f"cell-{args.loss}(FiLM+geno)", y, pred)
    base = evaluate_predictions("population-mean", y, population_mean_baseline(data, tr, te))
    energy = distribution_energy(model, data, te, seed=args.seed)

    print("\n" + "=" * 74)
    print("結果（leave-one-donor-out）")
    print("-" * 74)
    print("  " + res.row())
    print("  " + base.row())
    print(f"\n  分布レベル精度（energy distance, 小さいほど良い）: {energy:.4f}")
    print("=" * 74)
    if res.indiv_r2 > base.indiv_r2 + 0.05:
        print("  => cell-level MMD 学習でも個人差を捉えている ✅")
    else:
        print("  => epochs/細胞数を増やすことを検討。")


if __name__ == "__main__":
    main()
