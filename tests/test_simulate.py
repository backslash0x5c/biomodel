"""合成データ生成器のテスト（torch 不要）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.simulate import SimConfig, donor_split, sample_treated_cells, simulate


def test_shapes():
    cfg = SimConfig(n_donors=12, n_genes=20, n_perts=5, n_control_cells=16)
    d = simulate(cfg)
    assert d.geno.shape == (12, cfg.geno_dim)
    assert d.control_cells.shape == (12, 16, 20)
    assert d.delta.shape == (12, 5, 20)
    assert d.true_delta.shape == (12, 5, 20)
    assert d.gain.shape == (12, 5)
    assert d.base_effect.shape == (5, 20)


def test_individual_variability_exists():
    """gain がドナー間でばらつく = 個人差が実在することを確認。"""
    d = simulate(SimConfig(n_donors=40, seed=1))
    per_pert_std = d.gain.std(axis=0)        # 各摂動のドナー間ばらつき
    assert np.all(per_pert_std > 0.05), "個人差ゲインのばらつきが小さすぎる"


def test_gain_relates_delta_to_base_effect():
    """delta ≈ gain * base_effect（生成過程の整合性）。"""
    d = simulate(SimConfig(n_donors=8, n_genes=15, n_perts=4, delta_noise=0.0))
    recon = d.gain[:, :, None] * d.base_effect[None, :, :]
    assert np.allclose(recon, d.true_delta, atol=1e-5)


def test_donor_split_disjoint():
    d = simulate(SimConfig(n_donors=20))
    tr, te = donor_split(d, n_test=6, seed=3)
    assert len(te) == 6
    assert len(tr) == 14
    assert set(tr).isdisjoint(set(te))
    assert set(tr) | set(te) == set(range(20))


def test_reproducible():
    a = simulate(SimConfig(seed=7))
    b = simulate(SimConfig(seed=7))
    assert np.array_equal(a.delta, b.delta)


def test_nonlinear_gain_differs_from_linear():
    """非線形ゲインは線形ゲインと異なる個人差構造を持つ。"""
    lin = simulate(SimConfig(n_donors=60, nonlinear=False, seed=2))
    non = simulate(SimConfig(n_donors=60, nonlinear=True, seed=2))
    assert lin.gain.shape == non.gain.shape
    assert not np.allclose(lin.gain, non.gain)
    # どちらも個人差（ドナー間ばらつき）を持つ
    assert np.all(non.gain.std(axis=0) > 0.05)


def test_sample_treated_cells():
    d = simulate(SimConfig(n_donors=8, n_genes=20, n_perts=4, n_control_cells=16))
    rng = np.random.default_rng(0)
    cells = sample_treated_cells(d, donor=0, pert=1, n_cells=32, rng=rng)
    assert cells.shape == (32, 20)
    # 処置後細胞の平均は baseline + batch_shift + true_delta に近い
    expected = (d.baseline[0] + d._batch_shift[d.batch[0]] + d.true_delta[0, 1])
    assert np.allclose(cells.mean(axis=0), expected, atol=0.2)
