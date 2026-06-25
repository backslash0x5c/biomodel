#!/usr/bin/env python3
"""「精密医療（precision medicine）」デモ — 何の役に立つかを具体例で見せる。

合成データを「患者 × 候補薬」に見立て、学習済みモデルが **未知の患者** に対して
「どの薬がどれくらい効くか」を予測し、患者ごとに最適な薬を推薦する流れを示す。

重要: データはすべて合成（架空）で、実在薬の効果を主張するものではありません。
個人差（薬の効きが患者の genotype で変わる）を **予測できる** ことの実演です。
現実の動機づけ例（ワルファリン/CYP2C9, クロピドグレル/CYP2C19, COVID のデキサメタゾン等）は
docs/00_overview_for_beginners.md を参照。

使い方:
    python scripts/demo_precision_medicine.py
    python scripts/demo_precision_medicine.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biomodel.evaluate import spearman  # noqa: E402
from biomodel.model import PerturbationResponseModel  # noqa: E402
from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402
from biomodel.train import TrainConfig, predict_delta, pretrain_encoder, train_supervised  # noqa: E402


def benefit_scores(delta: np.ndarray, base_effect: np.ndarray) -> np.ndarray:
    """各 (患者, 薬) の「治療効果スコア」= 予測応答を薬の意図する方向へ射影した量。

    delta: (n_patients, n_drugs, n_genes)、base_effect: (n_drugs, n_genes)。
    スコアが大きいほど「その薬が効く（responder）」、0 近傍/負は「効かない/逆効果」。
    現実では『抑えたい疾患遺伝子プログラム』の方向で定義する（ここでは薬の平均効果方向）。
    """
    unit = base_effect / (np.linalg.norm(base_effect, axis=1, keepdims=True) + 1e-8)
    return np.einsum("pdg,dg->pd", delta, unit)  # (n_patients, n_drugs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    scfg = (SimConfig(n_donors=80, n_genes=40, n_perts=6, seed=args.seed)
            if args.quick else SimConfig(seed=args.seed))
    if args.quick:
        args.epochs = min(args.epochs, 100)

    print("=" * 78)
    print(" 精密医療デモ：未知の患者に『どの薬が効くか』を予測する（合成データ）")
    print("=" * 78)
    data = simulate(scfg)
    tr, te = donor_split(data, n_test=min(20, scfg.n_donors // 4), seed=args.seed)
    drugs = [f"D{i+1}" for i in range(data.n_perts)]
    print(f"設定: 学習に使う患者={len(tr)}人 / 予測対象の未知患者={len(te)}人、"
          f"候補薬={data.n_perts}種、遺伝子={data.n_genes}")
    print("（D1..Dn は合成データ上の架空の候補薬。患者ごとに genotype が異なる）\n")

    # --- 学習（既存パイプライン）---
    torch.manual_seed(args.seed)
    model = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
        interaction="film", use_genotype=True)
    cfg = TrainConfig(epochs=args.epochs, pretrain_epochs=args.pretrain_epochs,
                      seed=args.seed, verbose=False)
    print("学習中（事前学習→摂動応答学習）...")
    pretrain_encoder(model, data, cfg)
    train_supervised(model, data, tr, cfg)

    # --- 未知患者への予測 ---
    pred = predict_delta(model, data, te)             # (n_te, n_drugs, n_genes)
    true = data.true_delta[te]
    pred_score = benefit_scores(pred, data.base_effect)   # (n_te, n_drugs)
    true_score = benefit_scores(true, data.base_effect)
    pop_score = benefit_scores(data.true_delta[tr], data.base_effect).mean(0)  # 集団平均

    # === (1) 患者ごとの薬の推薦 ===
    print("\n" + "-" * 78)
    print("【1】患者ごとの『最も効く薬』の推薦（未知患者・モデル予測）")
    print("-" * 78)
    pop_best = drugs[int(np.argmax(pop_score))]
    print(f"  集団平均だけで決めると、全員に同じ薬『{pop_best}』を薦めることになる。")
    print("  → モデルは患者の genotype を見て、患者ごとに違う薬を薦められる:\n")
    print(f"    {'患者':<8}{'推薦薬(予測)':<14}{'予測効果':<10}{'本当の最適薬':<14}{'当たり?'}")
    n_show = min(8, len(te))
    hit = 0
    for i in range(len(te)):
        rec = int(np.argmax(pred_score[i]))
        truth = int(np.argmax(true_score[i]))
        if rec == truth:
            hit += 1
        if i < n_show:
            ok = "○" if rec == truth else "×"
            print(f"    患者{te[i]:<5}{drugs[rec]:<14}{pred_score[i, rec]:<10.2f}"
                  f"{drugs[truth]:<14}{ok}")
    print(f"\n  推薦が本当の最適薬と一致: {hit}/{len(te)} 人 "
          f"(集団平均の一律推薦より個別化できている)")

    # === (2) ある薬の responder / non-responder の振り分け ===
    drug = 0
    print("\n" + "-" * 78)
    print(f"【2】薬『{drugs[drug]}』に対する responder（効く人）の見分け")
    print("-" * 78)
    order = np.argsort(pred_score[:, drug])[::-1]
    print(f"  予測効果スコアで未知患者を並べ替え（高い=効く, 低い/負=効かない・逆効果）:")
    print(f"    効くと予測 TOP3 : " +
          ", ".join(f"患者{te[j]}({pred_score[j, drug]:+.2f})" for j in order[:3]))
    print(f"    効かない予測 LOW3: " +
          ", ".join(f"患者{te[j]}({pred_score[j, drug]:+.2f})" for j in order[-3:]))
    rho = spearman(pred_score[:, drug], true_score[:, drug])
    print(f"  予測スコア vs 真の効きの順位相関 (Spearman) = {rho:+.2f} "
          f"(1 に近いほど responder を正しく当てている)")
    print("  → 臨床試験なら『効くと予測された患者』に絞って試せる（無駄な投薬を減らせる）。")

    print("\n" + "=" * 78)
    print(" まとめ: 同じ薬でも患者ごとに効果が違う、を genotype から予測できた。")
    print(" 用途: ①患者に合った薬選び ②臨床試験の responder 濃縮 ③創薬の効きそうな層の特定")
    print(" ※合成データの実演です。現実の動機例は docs/00_overview_for_beginners.md。")
    print("=" * 78)


if __name__ == "__main__":
    main()
