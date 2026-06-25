"""leave-one-donor-out 評価（docs/04）。

核心は「集団平均（ATE）が当たっているだけ」を「個人差が当たった」と誤認しないこと。
そのため (a) 全体精度に加えて (b) 集団平均を引いた個人差成分、(c) ドナー横断の
応答ランキング、を測り、population-mean ベースラインと比較する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulate import SimData


# --- numpy のみの相関・決定係数（scipy 非依存）---
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else 0.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均順位（同順位タイ対応）。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # タイの平均化
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_rankdata(a.reshape(-1)), _rankdata(b.reshape(-1)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


@dataclass
class EvalResult:
    name: str
    delta_pearson: float        # 全体: Δ̂ vs Δ の相関
    indiv_r2: float             # ★個人差成分（集団平均を引いた残差）の R²
    indiv_pearson: float        # 個人差成分の相関
    ranking_spearman: float     # 薬ごと「どのドナーで効くか」のランキング相関（平均）

    def row(self) -> str:
        return (f"{self.name:<22} "
                f"overall_r={self.delta_pearson:+.3f}  "
                f"indiv_R2={self.indiv_r2:+.3f}  "
                f"indiv_r={self.indiv_pearson:+.3f}  "
                f"rank_rho={self.ranking_spearman:+.3f}")


def _ranking_spearman(true_dp: np.ndarray, pred_dp: np.ndarray) -> float:
    """薬ごとに「ドナー横断の効果の大きさ」のランキング相関を測り平均する。

    true_dp, pred_dp: (n_test_donors, n_perts, n_genes)
    効果の大きさは delta の L2 ノルム（|Δ|）で代表させる。
    """
    true_mag = np.linalg.norm(true_dp, axis=2)   # (n_donors, n_perts)
    pred_mag = np.linalg.norm(pred_dp, axis=2)
    rhos = []
    for p in range(true_dp.shape[1]):
        if np.std(true_mag[:, p]) > 1e-8:
            rhos.append(spearman(true_mag[:, p], pred_mag[:, p]))
    return float(np.mean(rhos)) if rhos else 0.0


def evaluate_predictions(name: str, true_dp: np.ndarray, pred_dp: np.ndarray) -> EvalResult:
    """真値 / 予測（ともに (n_test_donors, n_perts, n_genes)）から指標を計算。"""
    # 全体
    overall = pearson(pred_dp, true_dp)
    # 個人差成分: 摂動ごとの集団平均（テストドナー横断）を引いた残差
    true_mean = true_dp.mean(axis=0, keepdims=True)   # (1, n_perts, n_genes)
    pred_mean = pred_dp.mean(axis=0, keepdims=True)
    true_resid = true_dp - true_mean
    pred_resid = pred_dp - pred_mean
    indiv_r2 = r2_score(true_resid, pred_resid)
    indiv_r = pearson(pred_resid, true_resid)
    rank = _ranking_spearman(true_dp, pred_dp)
    return EvalResult(name, overall, indiv_r2, indiv_r, rank)


def population_mean_baseline(data: SimData, train_idx: np.ndarray,
                            test_idx: np.ndarray) -> np.ndarray:
    """ベースライン: train ドナーの摂動別平均 delta を全 test ドナーに適用。

    = ATE のみ。個人差を一切使わない本命の対照（docs/04 §3）。
    """
    train_mean = data.delta[train_idx].mean(axis=0)        # (n_perts, n_genes)
    return np.broadcast_to(train_mean[None], (len(test_idx),) + train_mean.shape).copy()


def true_delta_test(data: SimData, test_idx: np.ndarray) -> np.ndarray:
    """評価の真値（ノイズなし true_delta を用いる）。"""
    return data.true_delta[test_idx]
