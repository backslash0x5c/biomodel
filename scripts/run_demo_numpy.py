#!/usr/bin/env python3
"""numpy のみのフォールバックデモ（torch 不要）。

設計思想の最小核 —「genotype に条件づけて個人差を予測する」— を、摂動ごとの
リッジ回帰 Δ ~ [1, genotype] で示す。合成データの個人差は
Δ_{d,p} = (1 + a_p·g_d) v_p と genotype に線形なので、回帰の傾き項が個人差を、
切片項が集団平均（= population-mean ベースライン）を表す。

これにより torch が無い環境でも、評価プロトコル（docs/04）と「個人差を捉える/
捉えない」の差を確認できる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.evaluate import evaluate_predictions  # noqa: E402
from biomodel.simulate import SimConfig, donor_split, simulate  # noqa: E402


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """(X^T X + alpha I)^-1 X^T Y。X: (n, d), Y: (n, g) -> W: (d, g)。"""
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d)
    return np.linalg.solve(A, X.T @ Y)


def main() -> None:
    data = simulate(SimConfig())
    train_idx, test_idx = donor_split(data, n_test=20, seed=0)
    y_true = data.true_delta[test_idx]                 # (n_test, n_perts, n_genes)

    n_test = len(test_idx)
    pred_geno = np.zeros_like(y_true)                  # genotype 条件づけ
    pred_mean = np.zeros_like(y_true)                  # population-mean（切片のみ）

    g_train = data.geno[train_idx]
    g_test = data.geno[test_idx]
    ones_tr = np.ones((len(train_idx), 1), dtype=np.float32)
    ones_te = np.ones((n_test, 1), dtype=np.float32)
    X_tr = np.concatenate([ones_tr, g_train], axis=1)  # [1, genotype]
    X_te = np.concatenate([ones_te, g_test], axis=1)

    for p in range(data.n_perts):
        Y_tr = data.delta[train_idx, p]                # (n_train, n_genes)
        W = ridge_fit(X_tr, Y_tr, alpha=1.0)           # (1+geno_dim, n_genes)
        pred_geno[:, p, :] = X_te @ W
        # population-mean: 切片のみ（= train 平均）
        pred_mean[:, p, :] = ones_te @ W[:1]

    print("=" * 70)
    print("numpy フォールバックデモ（torch 不要） — leave-one-donor-out")
    print("=" * 70)
    print(f"ドナー train={len(train_idx)} test={n_test}  "
          f"遺伝子={data.n_genes} 摂動={data.n_perts}\n")
    for r in (evaluate_predictions("ridge(+genotype)", y_true, pred_geno),
              evaluate_predictions("population-mean", y_true, pred_mean)):
        print("  " + r.row())
    print("=" * 70)
    print("  indiv_R2（個人差成分の説明率）が genotype 条件づけで上がれば成功。")


if __name__ == "__main__":
    main()
