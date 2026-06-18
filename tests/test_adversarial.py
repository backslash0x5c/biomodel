"""batch 敵対学習（identifiability）のテスト（torch が無ければ skip）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.simulate import SimConfig, donor_split, simulate

torch = pytest.importorskip("torch")

from biomodel.adversarial import (  # noqa: E402
    AdvConfig,
    batch_probe_accuracy,
    train_supervised_adversarial,
)
from biomodel.model import PerturbationResponseModel, grad_reverse  # noqa: E402
from biomodel.train import pretrain_encoder  # noqa: E402


def test_grad_reverse_flips_gradient():
    x = torch.randn(4, 3, requires_grad=True)
    (grad_reverse(x, 2.0).sum()).backward()
    # 順伝播は恒等、逆伝播は -lambda 倍
    assert torch.allclose(x.grad, torch.full_like(x, -2.0))


def test_adversarial_training_runs_and_probe_valid():
    """敵対学習が動き、batch probe 診断が妥当な値を返すことを検証。

    注: 敵対学習による probe 低下は小スケールでは不安定なので、ここでは機構の健全性
    （学習が走る・probe が妥当・標準学習では batch が分離可能）のみを検証する。
    probe 低下のデモは scripts/run_demo_adversarial.py（docs/09）で正直に提示する。
    """
    data = simulate(SimConfig(n_donors=60, n_genes=32, n_perts=6, batch_effect=0.8,
                              genotype_drives_baseline=0.0, n_control_cells=64, seed=0))
    tr, _ = donor_split(data, n_test=12, seed=0)
    torch.manual_seed(0)
    m = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1])
    cfg = AdvConfig(epochs=40, pretrain_epochs=10, verbose=False, seed=0, adv_weight=1.0)
    pretrain_encoder(m, data, cfg)
    hist = train_supervised_adversarial(m, data, tr, cfg)
    assert len(hist["delta"]) == 40 and len(hist["adv"]) == 40
    probe = batch_probe_accuracy(m, data, tr, seed=0)
    n_batches = int(data.batch.max()) + 1
    assert 1.0 / n_batches - 0.2 <= probe <= 1.0        # 妥当な精度範囲
    # 標準学習（adv=0）では batch が線形分離可能（強い entanglement）
    torch.manual_seed(0)
    m0 = PerturbationResponseModel(
        n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1])
    cfg0 = AdvConfig(epochs=40, pretrain_epochs=10, verbose=False, seed=0, adv_weight=0.0)
    pretrain_encoder(m0, data, cfg0)
    train_supervised_adversarial(m0, data, tr, cfg0)
    assert batch_probe_accuracy(m0, data, tr, seed=0) > 0.5
