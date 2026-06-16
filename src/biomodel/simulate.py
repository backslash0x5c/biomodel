"""個人差（genotype × 摂動の相互作用）を埋め込んだ合成データ生成器。

実データ（多ドナー × 薬剤 × single-cell）が希少なため、概念実証として
「真の個人差」を既知の生成過程で埋め込んだ合成データを作る。これにより
モデル（docs/01）と評価プロトコル（docs/04）を実データなしで検証できる。

生成過程（要点）:
    baseline_d            = mu0 + B @ s_d                     # ドナー固有ベースライン
    gain_{d,p}            = 1 + a_p · g_d                      # ★個人差の本体（genotype×摂動の相互作用）
    delta_{d,p}           = gain_{d,p} * v_p                   # 個別化された処置効果（CATE）
    x_ctrl_{d,i}          = baseline_d + batch_b + noise
    x_pert_{d,i,p}        = baseline_d + delta_{d,p} + batch_b + noise

個人差は gain（薬の効きの強度・符号）が genotype に依存することとして表現される。
これは加法モデル（CPA 相当）では表現できず、FiLM 等の相互作用が必要になる
（docs/01 §1, docs/02 §3 参照）。numpy のみに依存。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimConfig:
    # ドナー数は「個人差（genotype×摂動の相互作用）」を学習するのに十分な数が要る。
    # 少なすぎる（~30）と神経モデルは未学習で線形 ridge にも負ける（docs/04 §4 の注記参照）。
    n_donors: int = 160
    n_genes: int = 64
    n_perts: int = 12
    geno_dim: int = 6            # 潜在 genotype 次元（GReX 特徴の抽象化）
    state_dim: int = 8           # ドナー状態（ベースラインを駆動）の次元
    pert_dim: int = 5            # 摂動潜在次元
    n_control_cells: int = 64    # ドナーあたり control 細胞数（MGM 事前学習用）
    cell_noise: float = 0.30     # 細胞レベル観測ノイズ
    delta_noise: float = 0.05    # pseudobulk delta の測定ノイズ
    batch_effect: float = 0.40   # batch effect の強さ（identifiability を難しくする）
    n_batches: int = 4
    # baseline が genotype をどれだけ反映するか。0 なら個人差は genotype 入力でしか
    # 説明できない（ablation が最も明快）。実際の baseline は genotype を一部反映するので
    # 既定は小さめの正値にして、genotype が「baseline を超えて」効くことを示す。
    genotype_drives_baseline: float = 0.3
    seed: int = 0


@dataclass
class SimData:
    config: SimConfig
    geno: np.ndarray            # (n_donors, geno_dim)  観測 genotype 特徴
    control_cells: np.ndarray   # (n_donors, n_control_cells, n_genes)
    baseline: np.ndarray        # (n_donors, n_genes)   真のベースライン pseudobulk
    delta: np.ndarray           # (n_donors, n_perts, n_genes)  観測処置効果（ノイズ付き）
    true_delta: np.ndarray      # (n_donors, n_perts, n_genes)  ノイズなし真値
    gain: np.ndarray            # (n_donors, n_perts)   真の個人差ゲイン
    base_effect: np.ndarray     # (n_perts, n_genes)    集団平均的な摂動効果 v_p
    batch: np.ndarray           # (n_donors,)           batch ラベル

    @property
    def n_donors(self) -> int:
        return self.config.n_donors

    @property
    def n_genes(self) -> int:
        return self.config.n_genes

    @property
    def n_perts(self) -> int:
        return self.config.n_perts


def simulate(config: SimConfig | None = None) -> SimData:
    """個人差を持つ合成 single-cell 摂動データを生成する。"""
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    # --- ドナーの潜在変数 ---
    g = rng.standard_normal((cfg.n_donors, cfg.geno_dim))            # 潜在 genotype
    # ドナー状態 s_d: 一部は genotype 由来（baseline が genotype を反映）、一部は独立
    g_to_state = rng.standard_normal((cfg.geno_dim, cfg.state_dim))
    s = (cfg.genotype_drives_baseline * (g @ g_to_state)
         + rng.standard_normal((cfg.n_donors, cfg.state_dim)))

    # --- ベースライン発現 ---
    mu0 = rng.standard_normal(cfg.n_genes) * 0.5
    B = rng.standard_normal((cfg.state_dim, cfg.n_genes)) / np.sqrt(cfg.state_dim)
    baseline = mu0[None, :] + s @ B                                  # (n_donors, n_genes)

    # --- batch effect（交絡）---
    batch = rng.integers(0, cfg.n_batches, size=cfg.n_donors)
    batch_shift = rng.standard_normal((cfg.n_batches, cfg.n_genes)) * cfg.batch_effect

    # --- 摂動の集団平均効果 v_p ---
    u = rng.standard_normal((cfg.n_perts, cfg.pert_dim))             # 摂動潜在
    P = rng.standard_normal((cfg.pert_dim, cfg.n_genes)) / np.sqrt(cfg.pert_dim)
    base_effect = u @ P                                             # (n_perts, n_genes) = v_p

    # --- ★個人差: gain_{d,p} = 1 + a_p · g_d （genotype × 摂動の相互作用）---
    a = rng.standard_normal((cfg.n_perts, cfg.geno_dim)) / np.sqrt(cfg.geno_dim)
    gain = 1.0 + g @ a.T                                            # (n_donors, n_perts)

    # --- 個別化処置効果 delta_{d,p} = gain * v_p ---
    true_delta = gain[:, :, None] * base_effect[None, :, :]         # (n_donors, n_perts, n_genes)
    delta = true_delta + rng.standard_normal(true_delta.shape) * cfg.delta_noise

    # --- control 細胞（MGM 事前学習・baseline 推定用）---
    cell_base = (baseline[:, None, :]
                 + batch_shift[batch][:, None, :])                  # (n_donors, 1, n_genes)
    control_cells = (cell_base
                     + rng.standard_normal(
                         (cfg.n_donors, cfg.n_control_cells, cfg.n_genes)) * cfg.cell_noise)

    return SimData(
        config=cfg,
        geno=g.astype(np.float32),
        control_cells=control_cells.astype(np.float32),
        baseline=baseline.astype(np.float32),
        delta=delta.astype(np.float32),
        true_delta=true_delta.astype(np.float32),
        gain=gain.astype(np.float32),
        base_effect=base_effect.astype(np.float32),
        batch=batch.astype(np.int64),
    )


def donor_split(data: SimData, n_test: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """leave-one-donor-out 評価のためのドナー分割（train / test）。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.n_donors)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx
