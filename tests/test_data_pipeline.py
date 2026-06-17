"""実データ前処理パイプラインのテスト（numpy のみ; SimData 変換のみ simulate 依存）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.data_pipeline import (
    GenotypeFeaturizer,
    build_processed,
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
