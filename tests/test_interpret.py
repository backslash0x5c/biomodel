"""解釈性・バイオマーカーのテスト（torch が無ければ skip）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.simulate import SimConfig, donor_split, simulate

pytest.importorskip("torch")

import torch  # noqa: E402

from biomodel.interpret import (  # noqa: E402
    genotype_attribution,
    genotype_importance,
    top_response_genes,
    validate_attribution,
)
from biomodel.model import PerturbationResponseModel  # noqa: E402
from biomodel.train import TrainConfig, pretrain_encoder, train_supervised  # noqa: E402


def _train_small():
    data = simulate(SimConfig(n_donors=120, n_genes=30, n_perts=5, geno_dim=6, seed=0))
    tr, te = donor_split(data, n_test=20, seed=0)
    torch.manual_seed(0)
    model = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1])
    cfg = TrainConfig(epochs=120, pretrain_epochs=15, verbose=False)
    pretrain_encoder(model, data, cfg)
    train_supervised(model, data, tr, cfg)
    return data, tr, te, model


def test_attribution_shapes():
    data, _, te, model = _train_small()
    attr = genotype_attribution(model, data, te, pert=0)
    assert attr.shape == (len(te), data.geno.shape[1])
    imp = genotype_importance(model, data, te, pert=0)
    assert imp.shape == (data.geno.shape[1],)
    genes = top_response_genes(model, data, te, pert=0, k=5)
    assert len(genes) == 5


def test_attribution_recovers_true_drivers():
    """抽出した genotype 重要度が真の駆動因子と正に相関する。"""
    data, _, te, model = _train_small()
    rho = validate_attribution(model, data, te)
    assert rho > 0.3
