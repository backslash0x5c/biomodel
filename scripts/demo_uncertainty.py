#!/usr/bin/env python3
"""不確実性の定量化デモ（deep ensemble ＋ 較正, docs/11 §1）。

複数モデルのばらつきを「予測の信頼度」とし、予測区間の被覆率で較正を確認する。
臨床では『この予測をどれだけ信じてよいか』が必須。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.evaluate import benefit_scores  # noqa: E402
from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402
from biomodel.train import TrainConfig  # noqa: E402
from biomodel.uncertainty import (  # noqa: E402
    calibration_error,
    coverage_at_levels,
    ensemble_predict,
    fit_variance_scale,
    train_ensemble_models,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    scfg = SimConfig(n_donors=120, n_genes=40, n_perts=6, seed=0) if args.quick \
        else SimConfig(n_donors=180, n_genes=48, n_perts=8, seed=0)
    if args.quick:
        args.epochs = min(args.epochs, 80); args.n_models = min(args.n_models, 3)

    print("=" * 74)
    print(" 不確実性の定量化デモ（deep ensemble ＋ 較正）")
    print("=" * 74)
    data = simulate(scfg)
    # train / calibration / test に分割（較正は held-out で行う）
    tr_all, te = donor_split(data, n_test=24, seed=0)
    cal = tr_all[:20]; tr = tr_all[20:]
    print(f"患者 train={len(tr)} calib={len(cal)} test={len(te)}、"
          f"ensemble={args.n_models} モデル学習中...")
    cfg = TrainConfig(epochs=args.epochs, pretrain_epochs=15, verbose=False)
    models = train_ensemble_models(data, tr, n_models=args.n_models, cfg=cfg)

    # 較正集合で分散スケールを推定（生のアンサンブル分散は過信になりがち）
    cal_pred = ensemble_predict(models, data, cal)
    scale = fit_variance_scale(data.true_delta[cal] - cal_pred.mean, cal_pred.std)

    raw = ensemble_predict(models, data, te)
    calibrated = ensemble_predict(models, data, te, std_scale=scale)
    cov_raw = coverage_at_levels(data.true_delta[te], raw.mean, raw.std)
    cov_cal = coverage_at_levels(data.true_delta[te], calibrated.mean, calibrated.std)
    ens = calibrated

    print(f"\n【較正】予測区間の被覆率（nominal に近いほど信頼度が正しい）  分散スケール={scale:.2f}")
    print(f"   {'区間':<8}{'生ensemble':>12}{'較正後':>10}")
    for k in cov_raw:
        print(f"   {int(k*100)}%{'':<5}{cov_raw[k]*100:>10.1f}%{cov_cal[k]*100:>9.1f}%")
    print(f"   平均較正誤差: 生={calibration_error(cov_raw):.3f} -> 較正後={calibration_error(cov_cal):.3f}")

    # 患者ごとの不確実性（効果スコアの std）
    score_mean = benefit_scores(ens.mean, data.base_effect)        # (n_te, n_drugs)
    score_std = np.stack([benefit_scores(m, data.base_effect) for m in ens.members]).std(0)
    drug = 0
    print(f"\n【患者ごとの信頼度】薬 D{drug+1} の予測効果 ± 不確実性:")
    order = np.argsort(score_std[:, drug])
    print("   自信あり(低std) 3例: " +
          ", ".join(f"P{te[i]}:{score_mean[i,drug]:+.1f}±{score_std[i,drug]:.1f}" for i in order[:3]))
    print("   自信なし(高std) 3例: " +
          ", ".join(f"P{te[i]}:{score_mean[i,drug]:+.1f}±{score_std[i,drug]:.1f}" for i in order[-3:]))
    print("\n => 不確実性が高い患者は『追加検査や慎重判断が必要』と機械的に切り分けられる。")
    print("=" * 74)


if __name__ == "__main__":
    main()
