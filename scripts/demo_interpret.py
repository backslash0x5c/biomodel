#!/usr/bin/env python3
"""解釈性・バイオマーカー発見デモ（docs/11 §4）。

「なぜこの薬がこの患者に効くか」を、効果スコアの genotype 特徴に対する勾配で説明する。
合成データは真の駆動因子（geno_gain_weights）が分かるので、抽出した重要度がそれを
復元できるかで検証する。実データなら、これが候補バイオマーカー（安価な検査）になる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biomodel.interpret import (  # noqa: E402
    genotype_importance,
    validate_attribution,
)
from biomodel.model import PerturbationResponseModel  # noqa: E402
from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402
from biomodel.train import TrainConfig, pretrain_encoder, train_supervised  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    scfg = SimConfig(n_donors=120, n_genes=40, n_perts=6, geno_dim=6, seed=0) if args.quick \
        else SimConfig(seed=0)
    if args.quick:
        args.epochs = min(args.epochs, 100)

    print("=" * 74)
    print(" 解釈性・バイオマーカー発見デモ")
    print("=" * 74)
    data = simulate(scfg)               # 線形モード（真の駆動因子が既知）
    tr, te = donor_split(data, n_test=20, seed=0)
    torch.manual_seed(0)
    model = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
        interaction="film", use_genotype=True)
    cfg = TrainConfig(epochs=args.epochs, pretrain_epochs=20, verbose=False)
    print("学習中...")
    pretrain_encoder(model, data, cfg); train_supervised(model, data, tr, cfg)

    drug = 0
    imp = genotype_importance(model, data, te, drug)
    true_imp = np.abs(data.geno_gain_weights[drug])
    print(f"\n【薬 D{drug+1} の応答を駆動する genotype 特徴】（勾配で抽出した重要度）")
    print("   特徴   抽出重要度   真の重要度")
    order = np.argsort(imp)[::-1]
    for j in order:
        print(f"   g{j:<5}{imp[j]:>10.3f}{true_imp[j]:>12.3f}")
    top_pred = set(order[:2].tolist())
    top_true = set(np.argsort(true_imp)[::-1][:2].tolist())
    print(f"   抽出 top2={sorted(top_pred)} / 真の top2={sorted(top_true)} "
          f"-> 一致 {len(top_pred & top_true)}/2")

    rho = validate_attribution(model, data, te)
    print(f"\n【検証】全薬で『抽出重要度 vs 真の駆動因子』の順位相関 平均 = {rho:+.2f}")
    print("   (1 に近いほど、モデルが正しい生物学的駆動因子を学んでいる)")
    print("   => 実データでは、この上位特徴が候補バイオマーカー（コンパニオン診断）になる。")
    print("=" * 74)


if __name__ == "__main__":
    main()
