"""モデル・学習・評価のテスト（torch が無ければ自動 skip）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.evaluate import (
    evaluate_predictions,
    pearson,
    population_mean_baseline,
    r2_score,
    spearman,
    true_delta_test,
)
from biomodel.simulate import SimConfig, donor_split, simulate

torch = pytest.importorskip("torch")

from biomodel.model import INTERACTIONS, PerturbationResponseModel  # noqa: E402
from biomodel.train import (  # noqa: E402
    TrainConfig,
    predict_delta,
    pretrain_encoder,
    train_supervised,
)


def test_metrics_numpy():
    x = np.array([1.0, 2, 3, 4, 5])
    assert pearson(x, 2 * x + 1) == pytest.approx(1.0, abs=1e-6)
    assert spearman(x, np.exp(x)) == pytest.approx(1.0, abs=1e-6)
    assert r2_score(x, x) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("interaction", list(INTERACTIONS))
def test_forward_shapes(interaction):
    data = simulate(SimConfig(n_donors=8, n_genes=16, n_perts=4, n_control_cells=8))
    model = PerturbationResponseModel(
        n_genes=16, n_perts=4, geno_dim=data.geno.shape[1], interaction=interaction)
    baseline = torch.tensor(data.control_cells.mean(axis=1))
    geno = torch.tensor(data.geno)
    pid = torch.zeros(8, dtype=torch.long)
    out = model(baseline, geno, pid)
    assert out.shape == (8, 16)


def test_pretrain_reduces_mgm_loss():
    data = simulate(SimConfig(n_donors=10, n_genes=24, n_perts=4, n_control_cells=32))
    model = PerturbationResponseModel(n_genes=24, n_perts=4, geno_dim=data.geno.shape[1])
    losses = pretrain_encoder(model, data, TrainConfig(pretrain_epochs=15, verbose=False))
    assert losses[-1] < losses[0]


def test_training_reduces_loss():
    data = simulate(SimConfig(n_donors=20, n_genes=24, n_perts=6, n_control_cells=16))
    tr, _ = donor_split(data, n_test=6)
    model = PerturbationResponseModel(n_genes=24, n_perts=6, geno_dim=data.geno.shape[1])
    losses = train_supervised(model, data, tr,
                              TrainConfig(epochs=40, pretrain_epochs=0, verbose=False))
    assert losses[-1] < losses[0]


def test_film_beats_population_mean_on_individual_variation():
    """提案モデル(FiLM+genotype)が個人差成分で population-mean 対照を上回る。

    ドナー数が十分なら神経モデルは genotype×摂動の相互作用を学習できる（docs/04 §4）。
    """
    data = simulate(SimConfig(seed=0, n_donors=120, genotype_drives_baseline=0.3))
    tr, te = donor_split(data, n_test=20, seed=0)
    torch.manual_seed(0)
    model = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
        interaction="film", use_genotype=True)
    cfg = TrainConfig(epochs=120, pretrain_epochs=15, verbose=False, seed=0)
    pretrain_encoder(model, data, cfg)
    train_supervised(model, data, tr, cfg)

    y_true = true_delta_test(data, te)
    pred = predict_delta(model, data, te)
    res_model = evaluate_predictions("film", y_true, pred)
    res_base = evaluate_predictions("pop-mean",
                                    y_true, population_mean_baseline(data, tr, te))

    # 個人差成分で対照を明確に上回る
    assert res_model.indiv_r2 > 0.3
    assert res_model.indiv_r2 > res_base.indiv_r2 + 0.3
