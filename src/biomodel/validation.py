"""実データ検証のための評価指標（docs/11 §2）。

ex vivo 薬剤感受性スクリーニング（例: 患者検体 × 薬剤応答）での「臨床的に効くか」を測る:
薬の推薦が当たるか（top-1/regret）、responder を見分けられるか（AUROC）など。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluate import spearman


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """二値ラベルに対する AUROC（Mann–Whitney U に基づく、numpy のみ）。"""
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # タイの平均化
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = sums[inv] / counts[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class ValidationReport:
    top1_accuracy: float       # 推薦薬 = 真の最適薬 の割合
    mean_regret: float         # 推薦薬の効きが真の最適からどれだけ劣るか（小さいほど良い）
    score_spearman: float      # 予測 vs 真の効きスコアの順位相関
    mean_responder_auroc: float  # 薬ごと responder 見分けの AUROC 平均

    def row(self) -> str:
        return (f"top1_acc={self.top1_accuracy:.3f}  regret={self.mean_regret:.3f}  "
                f"score_rho={self.score_spearman:+.3f}  responder_AUROC={self.mean_responder_auroc:.3f}")


def validate_recommendations(pred_score: np.ndarray, true_score: np.ndarray,
                             responder_quantile: float = 0.6) -> ValidationReport:
    """予測/真の効きスコア (n_patients, n_drugs) から検証指標を計算。

    responder_quantile: 各薬で真のスコアが上位 (1-q) を responder とする閾値の分位。
    """
    n_pat, n_drug = pred_score.shape
    rec = pred_score.argmax(axis=1)
    best = true_score.argmax(axis=1)
    top1 = float(np.mean(rec == best))
    # regret: 真の最適スコア - 推薦薬の真のスコア（患者ごと、最適=0）
    best_val = true_score[np.arange(n_pat), best]
    rec_val = true_score[np.arange(n_pat), rec]
    rng = (true_score.max() - true_score.min()) + 1e-8
    regret = float(np.mean((best_val - rec_val) / rng))
    rho = spearman(pred_score, true_score)
    # responder AUROC（薬ごと）
    aurocs = []
    for d in range(n_drug):
        thr = np.quantile(true_score[:, d], responder_quantile)
        labels = true_score[:, d] >= thr
        a = auroc(pred_score[:, d], labels)
        if not np.isnan(a):
            aurocs.append(a)
    mean_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
    return ValidationReport(top1, regret, rho, mean_auroc)
