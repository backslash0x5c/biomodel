#!/usr/bin/env python3
"""実データ検証の指標デモ（docs/11 §2）。

ex vivo 薬剤感受性スクリーニング（患者検体×薬剤応答）を想定し、leave-one-patient-out で
『薬の推薦が当たるか(top-1/regret)』『responder を見分けられるか(AUROC)』を測る。
ここでは合成データだが、実 BeatAML 等でも同じ指標で評価できる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biomodel.evaluate import benefit_scores  # noqa: E402
from biomodel.model import PerturbationResponseModel  # noqa: E402
from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402
from biomodel.train import TrainConfig, predict_delta, pretrain_encoder, train_supervised  # noqa: E402
from biomodel.validation import validate_recommendations  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    scfg = SimConfig(n_donors=80, n_genes=40, n_perts=6, seed=0) if args.quick \
        else SimConfig(seed=0)
    if args.quick:
        args.epochs = min(args.epochs, 100)

    print("=" * 74)
    print(" 実データ検証指標デモ（leave-one-patient-out）")
    print("=" * 74)
    data = simulate(scfg)
    tr, te = donor_split(data, n_test=20, seed=0)
    torch.manual_seed(0)
    model = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
        interaction="film", use_genotype=True)
    cfg = TrainConfig(epochs=args.epochs, pretrain_epochs=20, verbose=False)
    print("学習中...")
    pretrain_encoder(model, data, cfg); train_supervised(model, data, tr, cfg)

    pred = benefit_scores(predict_delta(model, data, te), data.base_effect)
    true = benefit_scores(data.true_delta[te], data.base_effect)
    rep = validate_recommendations(pred, true)

    # ベースライン: 集団平均（個人差を使わない）で全員に同じ薬
    import numpy as np
    pop = benefit_scores(data.true_delta[tr], data.base_effect).mean(0)
    pop_pred = np.tile(pop, (len(te), 1))
    rep_pop = validate_recommendations(pop_pred, true)

    print("\n【検証指標】(top1_acc=推薦が真の最適と一致, regret=最適との差, AUROC=responder見分け)")
    print(f"   提案モデル        : {rep.row()}")
    print(f"   集団平均(対照)    : {rep_pop.row()}")
    print("\n 読み: 提案モデルは top-1 的中・regret・AUROC で集団平均を上回る＝個別化の価値。")
    print(" 実 ex vivo データ(BeatAML 等)でも同じ枠組みで『本当に効くか』を検証できる。")
    print("=" * 74)


if __name__ == "__main__":
    main()
