"""実データ前処理パイプラインのテスト（numpy のみ; SimData 変換のみ simulate 依存）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.data_pipeline import (
    GenotypeFeaturizer,
    align_genotype,
    build_processed,
    load_anndata,
    load_fake_onek1k,
    normalize_log1p,
    processed_to_simdata,
    select_hvgs,
)


def test_normalize_log1p_nonneg_and_finite():
    counts = np.random.default_rng(0).integers(0, 30, size=(50, 20)).astype(np.float32)
    x = normalize_log1p(counts)
    assert x.shape == counts.shape
    assert np.all(np.isfinite(x))
    assert np.all(x >= 0)


def test_select_hvgs():
    rng = np.random.default_rng(0)
    expr = rng.standard_normal((40, 30))
    expr[:, 5] *= 10  # 高分散遺伝子
    sub, ids, idx = select_hvgs(expr, [f"g{i}" for i in range(30)], n_top=10)
    assert sub.shape == (40, 10)
    assert "g5" in ids


def test_fake_loader_and_build_processed():
    raw, geno = load_fake_onek1k(n_donors=20, n_genes=60, n_perts=4,
                                 cells_per_cond=15, n_variants=80, seed=0)
    assert raw.counts.shape[1] == 60
    assert len(geno.donor_ids) == 20
    proc = build_processed(raw, geno, n_hvg=32, n_cells=20,
                           featurizer=GenotypeFeaturizer("pca", 8), seed=0)
    assert proc.n_donors == 20
    assert proc.n_genes == 32
    assert proc.n_perts == 4
    assert proc.geno_features.shape == (20, 8)
    assert proc.control_cells.shape == (20, 20, 32)
    assert proc.delta.shape == (20, 4, 32)
    # fake データは全 (donor, pert) を観測
    assert proc.observed.mean() == 1.0


def test_processed_to_simdata_roundtrip():
    raw, geno = load_fake_onek1k(n_donors=16, n_genes=50, n_perts=3,
                                 cells_per_cond=12, n_variants=60, seed=1)
    proc = build_processed(raw, geno, n_hvg=24, n_cells=16, seed=1)
    data = processed_to_simdata(proc)
    assert data.n_donors == 16
    assert data.n_genes == 24
    assert data.n_perts == 3
    assert data.control_cells.shape == (16, 16, 24)
    assert data.delta.shape == (16, 3, 24)
    assert np.array_equal(data.delta, data.true_delta)


def _make_fake_adata():
    ad = pytest.importorskip("anndata")
    import pandas as pd
    rng = np.random.default_rng(0)
    n_donors, n_perts, cpc, n_genes = 6, 3, 10, 40
    rows, donor, batch, pert = [], [], [], []
    for d in range(n_donors):
        for cond in ["control"] + [f"drugP{p}" for p in range(n_perts)]:
            rows.append(rng.poisson(5, size=(cpc, n_genes)))
            donor += [f"D{d}"] * cpc
            batch += [f"B{d % 2}"] * cpc
            pert += [cond] * cpc
    X = np.concatenate(rows).astype(np.float32)
    obs = pd.DataFrame({"donor": donor, "batch": batch, "pert": pert})
    var = pd.DataFrame(index=[f"ENSG_{i:04d}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_load_anndata_roundtrip():
    pytest.importorskip("anndata")
    adata = _make_fake_adata()
    raw, meta = load_anndata(adata, donor_key="donor", batch_key="batch",
                             perturbation_key="pert", control_value="control")
    # control は -1、薬剤摂動は 0..2
    assert set(np.unique(raw.perturbation).tolist()) == {-1, 0, 1, 2}
    assert (raw.perturbation == -1).sum() == 6 * 10        # 6 donor × 10 control cells
    assert len(meta["donor_ids"]) == 6
    assert meta["pert_names"] == ["drugP0", "drugP1", "drugP2"]
    assert raw.counts.shape == (6 * 4 * 10, 40)


def test_anndata_to_model_arrays():
    pytest.importorskip("anndata")
    adata = _make_fake_adata()
    raw, meta = load_anndata(adata, donor_key="donor", batch_key="batch",
                             perturbation_key="pert")
    # genotype を donor_ids 順に整列
    rng = np.random.default_rng(1)
    dosage = {d: rng.integers(0, 3, size=50).astype(np.float32) for d in meta["donor_ids"]}
    geno = align_genotype(meta["donor_ids"], dosage, [f"rs{i}" for i in range(50)])
    assert geno.dosage.shape == (6, 50)
    proc = build_processed(raw, geno, n_hvg=24, n_cells=8,
                           featurizer=GenotypeFeaturizer("pca", 6), seed=0)
    assert proc.n_donors == 6 and proc.n_perts == 3 and proc.n_genes == 24
    data = processed_to_simdata(proc)
    assert data.control_cells.shape == (6, 8, 24)
    assert data.observed.shape == (6, 3)


def test_grex_featurizer():
    from biomodel.data_pipeline import (
        GReXFeaturizer,
        RawGenotype,
        synthesize_grex_model,
    )
    rng = np.random.default_rng(0)
    vids = [f"rs{i}" for i in range(120)]
    geno = RawGenotype(dosage=rng.integers(0, 3, size=(25, 120)).astype(np.float32),
                       variant_ids=vids, donor_ids=[f"d{i}" for i in range(25)])
    model = synthesize_grex_model(vids, [f"ENSG{i}" for i in range(40)],
                                  snps_per_gene=6, seed=1)
    feat = GReXFeaturizer(model)(geno)
    assert feat.shape == (25, 40)                 # donor × GReX 遺伝子
    assert np.all(np.isfinite(feat))
    assert abs(feat.mean()) < 1e-3                 # 標準化済み
    assert len(GReXFeaturizer(model).method) > 0   # method 属性で build_processed と互換


def test_grex_in_build_processed():
    from biomodel.data_pipeline import GReXFeaturizer, synthesize_grex_model
    raw, geno = load_fake_onek1k(n_donors=18, n_genes=60, n_perts=3,
                                 cells_per_cond=12, n_variants=100, seed=0)
    model = synthesize_grex_model(geno.variant_ids,
                                  [f"GREX_{i}" for i in range(20)], snps_per_gene=5, seed=0)
    proc = build_processed(raw, geno, n_hvg=24, n_cells=10,
                           featurizer=GReXFeaturizer(model), seed=0)
    assert proc.geno_features.shape == (18, 20)
    assert proc.meta["geno_method"] == "grex"
