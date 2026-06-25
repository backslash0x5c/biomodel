"""不確実性の定量化（deep ensemble / MC dropout）と較正（calibration）。

臨床で使うには「予測をどれだけ信じてよいか」が必要（docs/11 §1）。複数モデルの
ばらつきを不確実性とし、予測区間の被覆率で較正を評価する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import PerturbationResponseModel
from .simulate import SimData
from .train import TrainConfig, predict_delta, pretrain_encoder, train_supervised


@dataclass
class EnsemblePrediction:
    mean: np.ndarray   # (n_test, n_perts, n_genes) 予測平均
    std: np.ndarray    # (n_test, n_perts, n_genes) モデル間ばらつき（不確実性）
    members: np.ndarray  # (n_models, n_test, n_perts, n_genes)


def train_ensemble_models(data: SimData, train_idx: np.ndarray, n_models: int = 5,
                          cfg: TrainConfig | None = None, interaction: str = "film",
                          use_genotype: bool = True) -> list:
    """異なる初期値で n_models 個のモデルを学習して返す（deep ensemble の構成要素）。"""
    cfg = cfg or TrainConfig(verbose=False)
    models = []
    for k in range(n_models):
        torch.manual_seed(1000 + k)
        model = PerturbationResponseModel(
            n_genes=data.n_genes, n_perts=data.n_perts, geno_dim=data.geno.shape[1],
            interaction=interaction, use_genotype=use_genotype)
        kcfg = TrainConfig(**{**cfg.__dict__, "seed": cfg.seed + k})
        pretrain_encoder(model, data, kcfg)
        train_supervised(model, data, train_idx, kcfg)
        models.append(model)
    return models


def ensemble_predict(models: list, data: SimData, idx: np.ndarray,
                     std_scale: float = 1.0) -> EnsemblePrediction:
    """学習済みアンサンブルで予測。std_scale で分散を較正（recalibration）できる。"""
    members = np.stack([predict_delta(m, data, idx) for m in models], axis=0)
    return EnsemblePrediction(members.mean(0), members.std(0) * std_scale, members)


def fit_variance_scale(residual: np.ndarray, std: np.ndarray) -> float:
    """較正集合の残差と予測 std から分散スケール係数を推定（平均分散を残差に合わせる）。

    scale = sqrt(mean(residual^2) / mean(std^2))。test に掛けると被覆率が改善する。
    """
    r2 = float((residual.reshape(-1) ** 2).mean())
    s2 = float((std.reshape(-1) ** 2).mean()) + 1e-12
    return float(np.sqrt(r2 / s2))


def train_ensemble(data: SimData, train_idx: np.ndarray, test_idx: np.ndarray,
                   n_models: int = 5, cfg: TrainConfig | None = None,
                   interaction: str = "film", use_genotype: bool = True) -> EnsemblePrediction:
    """学習＋予測をまとめて行う簡易版（較正なし）。"""
    models = train_ensemble_models(data, train_idx, n_models, cfg, interaction, use_genotype)
    return ensemble_predict(models, data, test_idx)


def mc_dropout_predict(model: PerturbationResponseModel, data: SimData,
                       test_idx: np.ndarray, n_samples: int = 20,
                       device: str = "cpu") -> EnsemblePrediction:
    """学習済みモデルで dropout を推論時も有効にし、複数サンプルから不確実性を出す。

    モデルに dropout 層がある場合に有効（dropout=0 なら std≈0）。
    """
    samples = []
    model.train()  # dropout を有効化
    for _ in range(n_samples):
        samples.append(predict_delta(model, data, test_idx, device=device))
    model.eval()
    arr = np.stack(samples, axis=0)
    return EnsemblePrediction(arr.mean(0), arr.std(0), arr)


def coverage_at_levels(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray,
                       levels=(0.5, 0.8, 0.9, 0.95)) -> dict:
    """ガウス近似の予測区間 mean ± z·std の経験的被覆率。

    返り値: {nominal: empirical}。較正が良いと nominal ≈ empirical。
    """
    from statistics import NormalDist
    nd = NormalDist()
    yt = y_true.reshape(-1)
    mu = mean.reshape(-1)
    sd = std.reshape(-1) + 1e-8
    out = {}
    for lv in levels:
        z = nd.inv_cdf((1.0 + lv) / 2.0)      # 両側 lv 区間の z 値
        lo, hi = mu - z * sd, mu + z * sd
        out[lv] = float(np.mean((yt >= lo) & (yt <= hi)))
    return out


def calibration_error(coverage: dict) -> float:
    """|nominal - empirical| の平均（小さいほど較正が良い）。"""
    return float(np.mean([abs(k - v) for k, v in coverage.items()]))
