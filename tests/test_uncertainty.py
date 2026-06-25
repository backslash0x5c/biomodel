"""不確実性・較正のテスト（torch が無ければ skip）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("torch")

from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402
from biomodel.train import TrainConfig  # noqa: E402
from biomodel.uncertainty import (  # noqa: E402
    coverage_at_levels,
    ensemble_predict,
    fit_variance_scale,
    train_ensemble_models,
)


def test_coverage_well_calibrated_gaussian():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal((2000,))
    sd = np.abs(rng.standard_normal((2000,))) + 0.5
    y = mu + sd * rng.standard_normal((2000,))         # 正しく較正された予測
    cov = coverage_at_levels(y, mu, sd, levels=(0.5, 0.9))
    assert abs(cov[0.9] - 0.9) < 0.04
    assert abs(cov[0.5] - 0.5) < 0.05


def test_fit_variance_scale_recovers_factor():
    rng = np.random.default_rng(0)
    std = np.abs(rng.standard_normal((5000,))) + 0.3
    residual = 2.0 * std * rng.standard_normal((5000,))  # 残差は std の 2 倍の広がり
    scale = fit_variance_scale(residual, std)
    assert 1.7 < scale < 2.3


def test_ensemble_predict_shapes_and_recalibration_helps():
    data = simulate(SimConfig(n_donors=90, n_genes=24, n_perts=4, seed=0))
    tr_all, te = donor_split(data, n_test=16, seed=0)
    cal, tr = tr_all[:14], tr_all[14:]
    models = train_ensemble_models(
        data, tr, n_models=3, cfg=TrainConfig(epochs=40, pretrain_epochs=8, verbose=False))
    assert len(models) == 3
    cal_pred = ensemble_predict(models, data, cal)
    assert cal_pred.mean.shape == (len(cal), 4, 24)
    assert cal_pred.std.shape == cal_pred.mean.shape
    scale = fit_variance_scale(data.true_delta[cal] - cal_pred.mean, cal_pred.std)
    raw = ensemble_predict(models, data, te)
    cal_t = ensemble_predict(models, data, te, std_scale=scale)
    c_raw = coverage_at_levels(data.true_delta[te], raw.mean, raw.std, levels=(0.9,))
    c_cal = coverage_at_levels(data.true_delta[te], cal_t.mean, cal_t.std, levels=(0.9,))
    # 生 ensemble は過信（被覆率不足）になりがちで、較正で 90% に近づく
    assert c_cal[0.9] >= c_raw[0.9]
