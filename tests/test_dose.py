"""用量反応・組み合わせのテスト（torch が無ければ skip）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.dose import (
    DoseConfig,
    dose_to_target,
    hill,
    patient_split,
    simulate_combination,
    simulate_dose,
)


def test_hill_monotonic_and_bounded():
    d = np.linspace(0, 10, 50)
    r = hill(d, Emax=1.0, EC50=2.0, h=1.5)
    assert np.all(np.diff(r) >= -1e-9)      # 単調増加
    assert r[0] == 0.0 and r[-1] < 1.0      # 0 から Emax 未満


def test_simulate_dose_shapes_and_individual_variation():
    data = simulate_dose(DoseConfig(n_patients=50, n_drugs=4, seed=0))
    assert data.geno.shape == (50, data.config.geno_dim)
    assert data.Emax.shape == (50, 4) and data.EC50.shape == (50, 4)
    assert len(data.obs) == 50 * 4
    # EC50（必要量）に個人差がある
    assert data.EC50[:, 0].std() > 0.05


def test_dose_to_target():
    grid = np.linspace(0, 5, 11)
    curve = hill(grid, 1.0, 1.0, 1.5)
    d = dose_to_target(curve, grid, 0.5)
    assert 0.5 < d < 1.5                     # EC50=1 付近で応答 0.5 に到達


def test_simulate_combination_has_synergy_variation():
    data = simulate_dose(DoseConfig(n_patients=40, n_drugs=4, seed=1))
    pairs, single, synergy, wS = simulate_combination(data, dose_level=2.0)
    assert single.shape == (40, 4)
    assert synergy.shape == (40, len(pairs))
    assert synergy[:, 0].std() > 0.05        # 相乗効果に個人差（符号も変わりうる）


def test_dose_model_trains():
    pytest.importorskip("torch")
    from biomodel.dose import predict_curve, train_dose
    data = simulate_dose(DoseConfig(n_patients=60, n_drugs=3, seed=0))
    tr, te = patient_split(data, n_test=15, seed=0)
    model = train_dose(data, tr, epochs=60, verbose=False)
    curve = predict_curve(model, data, int(te[0]), 0)
    assert curve.shape == data.doses_grid.shape
    # 予測曲線はおおむね単調増加（Hill 形）
    assert curve[-1] > curve[0]
