#!/usr/bin/env python3
"""一気通貫デモ: 合成データ -> MGM事前学習 -> 摂動学習 -> leave-one-donor 評価。

「個人差を予測できるか」を population-mean ベースライン（=個人差を使わない）
および ablation（加法 Φ / genotype なし）との比較で定量化する（docs/04）。

使い方:
    python scripts/run_demo.py --epochs 60
    python scripts/run_demo.py --quick           # 小さめ・高速
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biomodel.evaluate import (  # noqa: E402
    evaluate_predictions,
    population_mean_baseline,
    true_delta_test,
)
from biomodel.model import PerturbationResponseModel  # noqa: E402
from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402
from biomodel.train import (  # noqa: E402
    TrainConfig,
    predict_delta,
    pretrain_encoder,
    train_supervised,
)


def build_and_train(data, train_idx, *, interaction, use_genotype, tcfg, seed):
    torch.manual_seed(seed)
    model = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
        interaction=interaction, use_genotype=use_genotype,
    )
    pretrain_encoder(model, data, tcfg)
    train_supervised(model, data, train_idx, tcfg)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="小さめ設定で高速に動作確認")
    ap.add_argument("--no-ablations", action="store_true")
    args = ap.parse_args()

    if args.quick:
        # 注意: ドナー数が少ないと個人差の学習は不十分になりやすい（docs/04 §4）。
        scfg = SimConfig(n_donors=80, n_genes=40, n_perts=10, seed=args.seed)
        args.epochs = min(args.epochs, 120)
        args.pretrain_epochs = min(args.pretrain_epochs, 15)
        args.n_test = 12
    else:
        scfg = SimConfig(seed=args.seed)

    print("=" * 78)
    print("個人差を予測する摂動応答モデル — leave-one-donor-out デモ")
    print("=" * 78)
    data = simulate(scfg)
    train_idx, test_idx = donor_split(data, n_test=args.n_test, seed=args.seed)
    print(f"ドナー: train={len(train_idx)}  test={len(test_idx)}  "
          f"遺伝子={data.n_genes}  摂動={data.n_perts}  genotype次元={data.geno.shape[1]}")
    print(f"真の個人差ゲイン gain の範囲: [{data.gain.min():.2f}, {data.gain.max():.2f}] "
          f"(1.0 から離れるほど個人差が大きい)\n")

    tcfg = TrainConfig(epochs=args.epochs, pretrain_epochs=args.pretrain_epochs,
                       seed=args.seed, verbose=True)
    y_true = true_delta_test(data, test_idx)
    results = []

    # --- 本命: FiLM + genotype ---
    print("\n--- [1] FiLM + genotype（提案モデル）---")
    model = build_and_train(data, train_idx, interaction="film", use_genotype=True,
                            tcfg=tcfg, seed=args.seed)
    pred = predict_delta(model, data, test_idx)
    results.append(evaluate_predictions("FiLM+genotype(提案)", y_true, pred))

    # --- ベースライン: population-mean（個人差を使わない）---
    pm = population_mean_baseline(data, train_idx, test_idx)
    results.append(evaluate_predictions("population-mean(対照)", y_true, pm))

    if not args.no_ablations:
        # --- ablation: 加法 Φ（CPA 相当, 個人差を表現できない）---
        print("\n--- [2] 加法 Φ（ablation: CPA 相当）---")
        m_add = build_and_train(data, train_idx, interaction="additive", use_genotype=True,
                                tcfg=TrainConfig(epochs=args.epochs,
                                                 pretrain_epochs=args.pretrain_epochs,
                                                 seed=args.seed, verbose=False),
                                seed=args.seed)
        results.append(evaluate_predictions(
            "additive(ablation)", y_true, predict_delta(m_add, data, test_idx)))

        # --- ablation: genotype なし ---
        print("--- [3] FiLM, genotype なし（ablation）---")
        m_nog = build_and_train(data, train_idx, interaction="film", use_genotype=False,
                                tcfg=TrainConfig(epochs=args.epochs,
                                                 pretrain_epochs=args.pretrain_epochs,
                                                 seed=args.seed, verbose=False),
                                seed=args.seed)
        results.append(evaluate_predictions(
            "FiLM,no-genotype(abl.)", y_true, predict_delta(m_nog, data, test_idx)))

    # --- 結果表 ---
    print("\n" + "=" * 78)
    print("結果（leave-one-donor-out, テストドナーで評価）")
    print("-" * 78)
    print("  overall_r : 全体の Δ̂ vs Δ 相関（集団平均も含む）")
    print("  indiv_R2  : ★個人差成分（集団平均を引いた残差）の決定係数 — 本命指標")
    print("  rank_rho  : 薬ごと『どのドナーで効くか』のランキング相関")
    print("-" * 78)
    for r in results:
        print("  " + r.row())
    print("=" * 78)

    prop = next(r for r in results if r.name.startswith("FiLM+genotype"))
    base = next(r for r in results if r.name.startswith("population-mean"))
    print("\n判定:")
    print(f"  提案モデルの個人差説明率 indiv_R2 = {prop.indiv_r2:+.3f}")
    print(f"  population-mean 対照     indiv_R2 = {base.indiv_r2:+.3f} (定義上ほぼ 0)")
    if prop.indiv_r2 > 0.05 and prop.indiv_r2 > base.indiv_r2 + 0.05:
        print("  => 提案モデルは集団平均を超えて『個人差』を捉えている ✅")
    else:
        print("  => 個人差の捕捉は不十分。epochs 増加やモデル拡張を検討。")


if __name__ == "__main__":
    main()
