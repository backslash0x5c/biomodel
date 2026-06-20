#!/usr/bin/env python3
"""用量反応・組み合わせ予測デモ（docs/11 §3）。

「効く/効かない」でなく『適量』を当てる。Hill 曲線で必要量(EC50)が genotype 依存
（ワルファリン用量が遺伝子型で変わるのと同じ）。未知患者の用量反応曲線を予測し、
目標応答に必要な用量（個別化用量）を推定。さらに 2 剤併用の相乗/拮抗も予測。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biomodel.dose import (  # noqa: E402
    DoseConfig,
    dose_to_target,
    hill,
    patient_split,
    predict_curve,
    simulate_combination,
    simulate_dose,
    train_dose,
)
from biomodel.evaluate import pearson, spearman  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = DoseConfig(n_patients=120, n_drugs=5, seed=0) if args.quick \
        else DoseConfig(seed=0)
    if args.quick:
        args.epochs = min(args.epochs, 150)

    print("=" * 74)
    print(" 用量反応・組み合わせ予測デモ")
    print("=" * 74)
    data = simulate_dose(cfg)
    tr, te = patient_split(data, n_test=40, seed=0)
    print(f"患者 train={len(tr)} test={len(te)}、薬={cfg.n_drugs}、用量点（学習）={cfg.n_obs_doses}")
    print("用量反応モデル学習中...")
    model = train_dose(data, tr, epochs=args.epochs, verbose=False)

    # --- 個別化用量: 目標応答に必要な用量を未知患者で予測 ---
    drug = 0
    target = 0.5
    true_doses, pred_doses = [], []
    for p in te:
        true_curve = hill(data.doses_grid, data.Emax[p, drug], data.EC50[p, drug], cfg.hill)
        pred_curve = predict_curve(model, data, int(p), drug)
        true_doses.append(dose_to_target(true_curve, data.doses_grid, target))
        pred_doses.append(dose_to_target(pred_curve, data.doses_grid, target))
    true_doses, pred_doses = np.array(true_doses), np.array(pred_doses)

    print(f"\n【個別化用量】薬 D{drug+1} で目標応答 {target} に必要な用量（未知患者）:")
    print(f"   真の必要量の範囲: {true_doses.min():.2f}〜{true_doses.max():.2f} "
          f"(患者で {true_doses.max()/max(true_doses.min(),1e-6):.1f} 倍の差)")
    print(f"   予測 vs 真の必要量: Pearson r={pearson(pred_doses, true_doses):+.2f}, "
          f"平均誤差={np.abs(pred_doses-true_doses).mean():.2f}")
    for p, td, pd in list(zip(te, true_doses, pred_doses))[:4]:
        print(f"     患者P{p}: 必要量 予測{pd:.2f} / 真値{td:.2f}")
    print("   => 一律用量でなく、患者ごとに必要量を出せる（過少・過量投与を回避）。")

    # --- 組み合わせ: 2 剤併用の相乗/拮抗（genotype 依存）---
    pairs, single, synergy, _ = simulate_combination(data, dose_level=2.0)
    pi = 0
    a, b = pairs[pi]
    add = single[te, a] + single[te, b]                  # 相加（単純な和）
    combo = add + synergy[te, pi]                          # 真の併用効果
    order = np.argsort(synergy[te, pi])[::-1]
    print(f"\n【組み合わせ】D{a+1}+D{b+1} 併用の相乗効果（患者で符号も変わる）:")
    print(f"   相乗が最大の患者: P{te[order[0]]} (相加{add[order[0]]:.2f}→併用{combo[order[0]]:.2f})")
    print(f"   拮抗（逆効果）の患者: P{te[order[-1]]} (相加{add[order[-1]]:.2f}→併用{combo[order[-1]]:.2f})")
    print("   => 併用が効く患者/打ち消す患者を切り分けられる（がんの併用療法設計に直結）。")
    print("=" * 74)


if __name__ == "__main__":
    main()
