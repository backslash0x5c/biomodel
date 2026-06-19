#!/usr/bin/env python3
"""実データ前処理パイプラインのデモ（docs/07）。

実データが無くても流せるよう、OneK1K 風の合成生データ（load_fake_onek1k）を
正規化 -> HVG -> pseudobulk/delta -> genotype 特徴 と処理し、ProcessedDataset を作る。
最後に SimData 互換へ変換し、既存モデルで leave-one-donor-out 学習・評価まで通す
（pipeline -> model が連結することの確認）。

実 OneK1K を使うときは load_fake_onek1k を load_anndata + genotype 読込に差し替える。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

import argparse  # noqa: E402

from biomodel.data_pipeline import (  # noqa: E402
    GenotypeFeaturizer,
    GReXFeaturizer,
    build_processed,
    load_fake_onek1k,
    processed_to_simdata,
    synthesize_grex_model,
)
from biomodel.evaluate import evaluate_predictions, population_mean_baseline, true_delta_test  # noqa: E402
from biomodel.model import PerturbationResponseModel  # noqa: E402
from biomodel.simulate import donor_split  # noqa: E402
from biomodel.train import TrainConfig, predict_delta, pretrain_encoder, train_supervised  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genotype-features", choices=["pca", "grex"], default="pca",
                    help="genotype 特徴: pca フォールバック or grex(PrediXcan風, docs/02 §2)")
    args = ap.parse_args()

    print("=" * 74)
    print("実データ前処理パイプライン デモ（fake OneK1K -> ProcessedDataset -> model）")
    print("=" * 74)

    raw, geno = load_fake_onek1k(n_donors=100, n_genes=200, n_perts=6,
                                 cells_per_cond=40, n_variants=500, seed=0)
    print(f"[raw] cells={raw.counts.shape[0]} genes={raw.counts.shape[1]} "
          f"donors={len(geno.donor_ids)} variants={len(geno.variant_ids)}")
    print(f"      control細胞={(raw.perturbation == -1).sum()} "
          f"摂動細胞={(raw.perturbation >= 0).sum()}")

    if args.genotype_features == "grex":
        # PrediXcan 風の合成重みモデルで GReX 特徴を作る
        grex_model = synthesize_grex_model(
            geno.variant_ids, [f"GREXgene_{i}" for i in range(64)], snps_per_gene=8, seed=0)
        featurizer = GReXFeaturizer(grex_model)
    else:
        featurizer = GenotypeFeaturizer("pca", 16)
    proc = build_processed(raw, geno, n_hvg=64, n_cells=64, featurizer=featurizer, seed=0)
    print(f"\n[processed] donors={proc.n_donors} genes(HVG)={proc.n_genes} "
          f"perts={proc.n_perts} geno_feat={proc.geno_features.shape[1]}")
    print(f"            観測率(donor×pert)={proc.observed.mean():.2f}  "
          f"batch種類={len(set(proc.batch.tolist()))}")
    print(f"            meta={proc.meta}")

    # SimData 互換へ変換して既存モデルに接続
    data = processed_to_simdata(proc)
    tr, te = donor_split(data, n_test=20, seed=0)
    torch.manual_seed(0)
    model = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
        interaction="film", use_genotype=True)
    cfg = TrainConfig(epochs=120, pretrain_epochs=15, verbose=False, seed=0)
    pretrain_encoder(model, data, cfg)
    train_supervised(model, data, tr, cfg)

    y = true_delta_test(data, te)   # 実データでは観測 delta が評価の基準
    res = evaluate_predictions("FiLM+genotype", y, predict_delta(model, data, te))
    base = evaluate_predictions("population-mean", y, population_mean_baseline(data, tr, te))
    print("\n[leave-one-donor-out on processed data]")
    print("  " + res.row())
    print("  " + base.row())
    print("=" * 74)
    print("  pipeline -> model が連結し、前処理済み実データ形式で個人差予測まで通ることを確認。")


if __name__ == "__main__":
    main()
