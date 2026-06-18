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
    # 個人差ゲインを非線形（genotype×摂動の固定ランダム MLP）にする。線形 ridge を
    # 崩し、相互作用モジュール（FiLM/cross-attn/hypernet）の比較に使う（docs/08）。
    nonlinear: bool = False
    nonlinear_hidden: int = 16
    gain_scale: float = 1.0      # 個人差ゲインの広がり（大きいほど個人差が強い）
    # 観測率: 各 (donor, pert) が観測される確率。<1 にすると疎な観測（実データを模す）。
    observed_rate: float = 1.0
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
    _batch_shift: np.ndarray    # (n_batches, n_genes)  batch effect（処置後細胞の生成に使用）
    observed: np.ndarray | None = None  # (n_donors, n_perts) 観測フラグ（None=全観測）

    def observed_mask(self) -> np.ndarray:
        """観測フラグを返す（None の場合は全 1）。"""
        if self.observed is None:
            return np.ones((self.n_donors, self.n_perts), dtype=np.float32)
        return self.observed

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

    # --- ★個人差: gain_{d,p}（genotype × 摂動の相互作用）---
    if cfg.nonlinear:
        # 非線形ゲイン: 飽和的な 2 層 tanh MLP ＋ genotype の二次項を [g_d, u_p] に適用。
        # genotype に強く非線形・摂動と非自明に結合するため、摂動ごとの線形 ridge
        # （Δ ~ [1, g]）では捉えきれない。FiLM/cross-attn/hypernet の比較に使う。
        h = cfg.nonlinear_hidden
        inp_dim = cfg.geno_dim + cfg.pert_dim
        W1 = rng.standard_normal((h, inp_dim))
        b1 = rng.standard_normal(h) * 0.3
        W2 = rng.standard_normal((h, h)) / np.sqrt(h)
        b2 = rng.standard_normal(h) * 0.3
        w3 = rng.standard_normal(h) / np.sqrt(h)
        # genotype 二次相互作用（線形 ridge では表現不可）
        Q = rng.standard_normal((cfg.n_perts, cfg.geno_dim, cfg.geno_dim)) / cfg.geno_dim
        gg = np.broadcast_to(g[:, None, :], (cfg.n_donors, cfg.n_perts, cfg.geno_dim))
        uu = np.broadcast_to(u[None, :, :], (cfg.n_donors, cfg.n_perts, cfg.pert_dim))
        inp = np.concatenate([gg, uu], axis=2)
        # 飽和させて強い非線形性を与える（pre-activation を拡大）
        h1 = np.tanh(2.5 * (inp @ W1.T / np.sqrt(inp_dim) + b1))
        h2 = np.tanh(2.0 * (h1 @ W2.T + b2))
        quad = np.einsum("di,pij,dj->dp", g, Q, g)                 # (n_donors, n_perts)
        raw = h2 @ w3 + 0.7 * quad                                 # (n_donors, n_perts)
        raw = (raw - raw.mean()) / (raw.std() + 1e-8)
        gain = 1.0 + cfg.gain_scale * raw
    else:
        # 線形ゲイン: gain_{d,p} = 1 + a_p · g_d
        a = rng.standard_normal((cfg.n_perts, cfg.geno_dim)) / np.sqrt(cfg.geno_dim)
        gain = 1.0 + cfg.gain_scale * (g @ a.T)                    # (n_donors, n_perts)

    # --- 個別化処置効果 delta_{d,p} = gain * v_p ---
    true_delta = gain[:, :, None] * base_effect[None, :, :]         # (n_donors, n_perts, n_genes)
    delta = true_delta + rng.standard_normal(true_delta.shape) * cfg.delta_noise

    # --- 観測マスク（疎な (donor, pert) を模す）---
    if cfg.observed_rate < 1.0:
        observed = (rng.random((cfg.n_donors, cfg.n_perts)) < cfg.observed_rate).astype(np.float32)
    else:
        observed = np.ones((cfg.n_donors, cfg.n_perts), dtype=np.float32)

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
        _batch_shift=batch_shift.astype(np.float32),
        observed=observed,
    )


def sample_treated_cells(data: SimData, donor: int, pert: int, n_cells: int,
                         rng: np.random.Generator) -> np.ndarray:
    """指定ドナー×摂動の「処置後細胞」を生成（cell-level / MMD 学習用, docs/06）。

    control 細胞と対応はつかない（unpaired）。生成過程:
        x_pert = baseline + batch_shift + true_delta_{d,p} + cell_noise
    """
    cfg = data.config
    mean = (data.baseline[donor] + data._batch_shift[data.batch[donor]]
            + data.true_delta[donor, pert])
    noise = rng.standard_normal((n_cells, cfg.n_genes)) * cfg.cell_noise
    return (mean[None, :] + noise).astype(np.float32)


def donor_split(data: SimData, n_test: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """leave-one-donor-out 評価のためのドナー分割（train / test）。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.n_donors)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx
