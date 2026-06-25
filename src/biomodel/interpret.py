"""解釈性・バイオマーカー発見（docs/11 §4）。

「なぜこの患者がこの薬に反応するか」を、予測効果スコアの genotype 特徴に対する勾配
（saliency）で説明する。合成データでは真の駆動因子（geno_gain_weights）が分かるので、
抽出した重要度がそれを復元できるかで検証できる。
"""

from __future__ import annotations

import numpy as np
import torch

from .evaluate import spearman
from .model import PerturbationResponseModel
from .simulate import SimData
from .train import _donor_baseline


def _unit_base_effect(data: SimData, pert: int, device) -> torch.Tensor:
    v = data.base_effect[pert]
    v = v / (np.linalg.norm(v) + 1e-8)
    return torch.tensor(v, device=device)


def genotype_attribution(model: PerturbationResponseModel, data: SimData,
                         donor_idx: np.ndarray, pert: int, device: str = "cpu") -> np.ndarray:
    """薬 pert の効果スコアの、各 genotype 特徴に対する勾配（患者ごと）。

    返り値: (len(donor_idx), geno_dim)。各患者で「どの遺伝的特徴が反応を駆動するか」。
    """
    dev = torch.device(device)
    model.eval()
    baseline = _donor_baseline(data)
    Xb = torch.tensor(baseline[donor_idx], device=dev)
    Xg = torch.tensor(data.geno[donor_idx], device=dev, requires_grad=True)
    pid = torch.full((len(donor_idx),), pert, dtype=torch.long, device=dev)
    unit = _unit_base_effect(data, pert, dev)
    delta = model(Xb, Xg, pid)                       # (n, n_genes)
    score = (delta * unit).sum(dim=1)                # (n,) 効果スコア
    score.sum().backward()
    return Xg.grad.detach().cpu().numpy()


def genotype_importance(model: PerturbationResponseModel, data: SimData,
                        donor_idx: np.ndarray, pert: int, device: str = "cpu") -> np.ndarray:
    """薬 pert について、患者横断の genotype 特徴 重要度 = mean |attribution|。(geno_dim,)"""
    return np.abs(genotype_attribution(model, data, donor_idx, pert, device)).mean(axis=0)


def top_response_genes(model: PerturbationResponseModel, data: SimData,
                       donor_idx: np.ndarray, pert: int, k: int = 10,
                       device: str = "cpu") -> np.ndarray:
    """薬 pert で最も動くと予測される遺伝子のインデックス上位 k（|平均予測Δ|）。"""
    from .train import predict_delta
    pred = predict_delta(model, data, donor_idx, device=device)   # (n, n_perts, n_genes)
    mag = np.abs(pred[:, pert, :].mean(axis=0))
    return np.argsort(mag)[::-1][:k]


def validate_attribution(model: PerturbationResponseModel, data: SimData,
                         donor_idx: np.ndarray, device: str = "cpu") -> float:
    """抽出した genotype 重要度が、真の駆動因子 |geno_gain_weights| を復元するか。

    各薬で Spearman(importance, |true_weight|) を取り平均（線形モードのみ）。1 に近いほど良い。
    """
    if data.geno_gain_weights is None:
        raise ValueError("geno_gain_weights が無い（非線形モード）。線形モードで検証してください。")
    rhos = []
    for p in range(data.n_perts):
        imp = genotype_importance(model, data, donor_idx, p, device)
        true_imp = np.abs(data.geno_gain_weights[p])
        if true_imp.std() > 1e-8:
            rhos.append(spearman(imp, true_imp))
    return float(np.mean(rhos)) if rhos else float("nan")
