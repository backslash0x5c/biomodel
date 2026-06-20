#!/usr/bin/env python3
"""docs 用の結果図を生成する（precision-medicine デモの可視化）。

出力: docs/figures/precision_medicine.png
  左: 未知患者 × 候補薬 の「予測される効きの強さ」ヒートマップ（行ごとの推薦薬に枠）
  右: ある薬での 予測 vs 真の効き の散布図（responder を当てられているか）

ラベルは英数字（matplotlib の日本語フォント不在による文字化けを避けるため）。
キャプションでの説明は docs/00 側で日本語にて行う。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import torch  # noqa: E402

from biomodel.evaluate import spearman  # noqa: E402
from biomodel.model import PerturbationResponseModel  # noqa: E402
from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402
from biomodel.train import TrainConfig, predict_delta, pretrain_encoder, train_supervised  # noqa: E402


def benefit_scores(delta, base_effect):
    unit = base_effect / (np.linalg.norm(base_effect, axis=1, keepdims=True) + 1e-8)
    return np.einsum("pdg,dg->pd", delta, unit)


def main():
    seed = 0
    data = simulate(SimConfig(seed=seed))
    tr, te = donor_split(data, n_test=20, seed=seed)
    torch.manual_seed(seed)
    model = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
        interaction="film", use_genotype=True)
    cfg = TrainConfig(epochs=150, pretrain_epochs=20, seed=seed, verbose=False)
    print("training ...")
    pretrain_encoder(model, data, cfg)
    train_supervised(model, data, tr, cfg)

    pred = predict_delta(model, data, te)
    pred_score = benefit_scores(pred, data.base_effect)
    true_score = benefit_scores(data.true_delta[te], data.base_effect)

    n_show = 12                       # 見やすさのため患者 12 人だけ表示
    drugs = [f"D{i+1}" for i in range(data.n_perts)]
    P = pred_score[:n_show]
    patients = [f"P{int(te[i])}" for i in range(n_show)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # --- 左: ヒートマップ（患者 × 薬の予測効果）---
    vmax = np.abs(P).max()
    im = axL.imshow(P, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axL.set_xticks(range(len(drugs))); axL.set_xticklabels(drugs)
    axL.set_yticks(range(n_show)); axL.set_yticklabels(patients)
    axL.set_xlabel("Candidate drug"); axL.set_ylabel("Unseen patient")
    axL.set_title("Predicted drug effect per patient\n(red = effective, blue = no/adverse; "
                  "box = recommended)")
    for i in range(n_show):                       # 行ごとの推薦薬（argmax）に枠
        j = int(np.argmax(P[i]))
        axL.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                edgecolor="black", lw=2.2))
    fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04, label="predicted benefit score")

    # --- 右: ある薬での 予測 vs 真値（responder を当てているか）---
    drug = 0
    x, y = true_score[:, drug], pred_score[:, drug]
    rho = spearman(x, y)
    axR.axhline(0, color="gray", lw=0.8, ls="--")
    axR.axvline(0, color="gray", lw=0.8, ls="--")
    axR.scatter(x, y, c=y, cmap="RdBu_r", vmin=-np.abs(y).max(), vmax=np.abs(y).max(),
                edgecolor="k", s=70)
    axR.set_xlabel(f"True effect of drug {drugs[drug]}")
    axR.set_ylabel(f"Predicted effect of drug {drugs[drug]}")
    axR.set_title(f"Responder prediction for {drugs[drug]}\nSpearman rho = {rho:+.2f}")

    fig.suptitle("Predicting who responds to which drug (synthetic data, leave-one-patient-out)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "precision_medicine.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
