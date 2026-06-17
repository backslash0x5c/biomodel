"""cell-level + MMD 学習のテスト（torch が無ければ skip）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.simulate import SimConfig, donor_split, simulate

torch = pytest.importorskip("torch")

from biomodel.cell_level import (  # noqa: E402
    CellTrainConfig,
    distribution_energy,
    predict_pseudobulk_delta,
    pretrain_encoder_cells,
    train_cell_level,
)
from biomodel.model import CellLevelResponseModel  # noqa: E402


def _small():
    data = simulate(SimConfig(n_donors=24, n_genes=24, n_perts=4, n_control_cells=64, seed=0))
    tr, te = donor_split(data, n_test=6, seed=0)
    model = CellLevelResponseModel(
        n_genes=24, n_perts=4, geno_dim=data.geno.shape[1], interaction="film")
    return data, tr, te, model


def test_cell_forward_shapes():
    data, _, _, model = _small()
    cells = torch.tensor(data.control_cells[0, :10])
    base = torch.tensor(data.control_cells[0].mean(0)).unsqueeze(0)
    geno = torch.tensor(data.geno[0]).unsqueeze(0)
    pid = torch.zeros(1, dtype=torch.long)
    effect = model.effect_for(base, geno, pid).expand(10, -1)
    out = model(cells, effect)
    assert out.shape == (10, 24)


def test_cell_level_training_runs():
    data, tr, te, model = _small()
    cfg = CellTrainConfig(epochs=6, pretrain_epochs=3, pairs_per_epoch=40,
                          n_cells=32, verbose=False, seed=0)
    pretrain_encoder_cells(model, data, cfg)
    losses = train_cell_level(model, data, tr, cfg)
    assert len(losses) == 6
    pred = predict_pseudobulk_delta(model, data, te, n_cells=48, seed=0)
    assert pred.shape == (len(te), 4, 24)
    energy = distribution_energy(model, data, te, n_cells=48, seed=0)
    assert energy >= 0
