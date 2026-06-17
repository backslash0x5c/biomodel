"""実データ前処理パイプライン雛形（OneK1K 等の多ドナー scRNA ＋ genotype, docs/07）。

実データは取得・倫理（genotype は機微情報）・ストレージの制約が大きいので、本モジュールは
**インターフェースと処理の骨組み**を提供する。scanpy/anndata はオプショナル import とし、
実データが無くても `load_fake_onek1k()` で end-to-end に流せる。出力は既存のモデル・学習
コード（train.py / evaluate.py）がそのまま使える形（SimData 互換）に変換する。

実 OneK1K を使う場合の注意:
  - scRNA: ~125万 PBMC × ~1,000 ドナー（GEO/ArrayExpress 等で公開）。
  - genotype: SNP アレイ。dbGaP/EGA 等の **アクセス制御**・IRB 承認が必要。
  - 遺伝子空間は Ensembl ID に統一、cis-eQTL/GReX 特徴は別途算出（PrediXcan 等）。
  - batch/donor メタデータを必ず保持（identifiability, docs/02 §4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# 生データのインターフェース
# ---------------------------------------------------------------------------
@dataclass
class RawScRNA:
    """単一細胞発現（生カウント）。実データは AnnData から詰め替える。"""
    counts: np.ndarray          # (n_cells, n_genes) 生カウント
    gene_ids: list[str]         # 長さ n_genes（Ensembl ID 推奨）
    donor: np.ndarray           # (n_cells,) ドナー ID（整数 index）
    batch: np.ndarray           # (n_cells,) batch ラベル
    perturbation: np.ndarray    # (n_cells,) 摂動 ID（control は -1）
    cell_type: np.ndarray | None = None


@dataclass
class RawGenotype:
    """ドナー genotype（0/1/2 ドーズ）。実データは VCF/plink から詰め替える。"""
    dosage: np.ndarray          # (n_donors, n_variants)
    variant_ids: list[str]
    donor_ids: list[str]


@dataclass
class ProcessedDataset:
    """モデルが消費する前処理済みデータ。"""
    geno_features: np.ndarray   # (n_donors, geno_feat_dim)
    control_cells: np.ndarray   # (n_donors, n_cells, n_genes) 正規化済み control
    delta: np.ndarray           # (n_donors, n_perts, n_genes) 観測 pseudobulk 処置効果
    observed: np.ndarray        # (n_donors, n_perts) 観測フラグ（1=観測あり）
    gene_ids: list[str]
    pert_names: list[str]
    donor_ids: list[str]
    batch: np.ndarray           # (n_donors,)
    meta: dict = field(default_factory=dict)

    @property
    def n_donors(self) -> int:
        return self.control_cells.shape[0]

    @property
    def n_genes(self) -> int:
        return self.control_cells.shape[2]

    @property
    def n_perts(self) -> int:
        return self.delta.shape[1]


# ---------------------------------------------------------------------------
# 前処理ステップ
# ---------------------------------------------------------------------------
def normalize_log1p(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """library-size 正規化 ＋ log1p（scanpy の標準前処理に対応）。"""
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(counts / lib * target_sum).astype(np.float32)


def select_hvgs(expr: np.ndarray, gene_ids: list[str], n_top: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """高変動遺伝子（HVG）を分散で選択。共通遺伝子パネルの簡易版。"""
    var = expr.var(axis=0)
    idx = np.argsort(var)[::-1][:n_top]
    idx = np.sort(idx)
    return expr[:, idx], [gene_ids[i] for i in idx], idx


def harmonize_genes(expr: np.ndarray, gene_ids: list[str],
                    panel: list[str]) -> np.ndarray:
    """共通遺伝子パネル（Ensembl ID）に整列。欠損遺伝子は 0 埋め。"""
    pos = {g: i for i, g in enumerate(gene_ids)}
    out = np.zeros((expr.shape[0], len(panel)), dtype=expr.dtype)
    for j, g in enumerate(panel):
        if g in pos:
            out[:, j] = expr[:, pos[g]]
    return out


def genotype_pca_features(dosage: np.ndarray, n_components: int = 16) -> np.ndarray:
    """genotype 特徴のフォールバック: 標準化 ＋ PCA（numpy SVD）。

    実運用では cis-eQTL/GReX（PrediXcan 等）や PGx star-allele 特徴に差し替える
    （docs/02 §2）。ここは依存を増やさない簡易版。
    """
    x = dosage.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x = x / std
    k = min(n_components, min(x.shape) - 1) if min(x.shape) > 1 else 1
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return (u[:, :k] * s[:k]).astype(np.float32)


class GenotypeFeaturizer:
    """genotype -> 特徴。既定は PCA フォールバック。GReX/eQTL に差し替え可能。"""

    def __init__(self, method: str = "pca", n_components: int = 16):
        self.method = method
        self.n_components = n_components

    def __call__(self, geno: RawGenotype) -> np.ndarray:
        if self.method == "pca":
            return genotype_pca_features(geno.dosage, self.n_components)
        raise NotImplementedError(
            f"method={self.method!r} は雛形では未実装。GReX/eQTL/PGx を実装して差し替える"
            "（docs/02 §2）。")


def pseudobulk_and_delta(expr: np.ndarray, raw: RawScRNA, n_donors: int, n_perts: int,
                         n_cells: int, rng: np.random.Generator):
    """ドナーごとに control 細胞をサブサンプルし、(donor,pert) の pseudobulk delta を作る。

    delta_{d,p} = mean(treated 細胞) - mean(control 細胞)。観測が無い組は observed=0。
    """
    n_genes = expr.shape[1]
    control_cells = np.zeros((n_donors, n_cells, n_genes), dtype=np.float32)
    delta = np.zeros((n_donors, n_perts, n_genes), dtype=np.float32)
    observed = np.zeros((n_donors, n_perts), dtype=np.float32)
    batch = np.zeros(n_donors, dtype=np.int64)
    for d in range(n_donors):
        dm = raw.donor == d
        ctrl_mask = dm & (raw.perturbation == -1)
        ctrl_idx = np.where(ctrl_mask)[0]
        if len(ctrl_idx) == 0:
            continue
        pick = rng.choice(ctrl_idx, size=n_cells, replace=len(ctrl_idx) < n_cells)
        control_cells[d] = expr[pick]
        ctrl_mean = expr[ctrl_idx].mean(axis=0)
        batch[d] = raw.batch[ctrl_idx[0]]
        for p in range(n_perts):
            pidx = np.where(dm & (raw.perturbation == p))[0]
            if len(pidx) > 0:
                delta[d, p] = expr[pidx].mean(axis=0) - ctrl_mean
                observed[d, p] = 1.0
    return control_cells, delta, observed, batch


def build_processed(raw: RawScRNA, geno: RawGenotype, *, n_hvg: int = 64,
                    n_cells: int = 64, featurizer: GenotypeFeaturizer | None = None,
                    seed: int = 0) -> ProcessedDataset:
    """生データ -> ProcessedDataset（正規化 -> HVG -> pseudobulk/delta -> genotype 特徴）。"""
    rng = np.random.default_rng(seed)
    featurizer = featurizer or GenotypeFeaturizer()
    expr = normalize_log1p(raw.counts)
    expr, gene_ids, _ = select_hvgs(expr, raw.gene_ids, n_hvg)
    n_donors = len(geno.donor_ids)
    pert_ids = sorted({int(p) for p in np.unique(raw.perturbation) if p >= 0})
    n_perts = len(pert_ids)
    control_cells, delta, observed, batch = pseudobulk_and_delta(
        expr, raw, n_donors, n_perts, n_cells, rng)
    geno_features = featurizer(geno)
    return ProcessedDataset(
        geno_features=geno_features,
        control_cells=control_cells,
        delta=delta,
        observed=observed,
        gene_ids=gene_ids,
        pert_names=[f"pert_{p}" for p in pert_ids],
        donor_ids=geno.donor_ids,
        batch=batch,
        meta={"n_hvg": n_hvg, "n_cells": n_cells, "geno_method": featurizer.method},
    )


def processed_to_simdata(proc: ProcessedDataset):
    """ProcessedDataset を SimData 互換に変換し、既存の train.py/evaluate.py で使えるようにする。

    実データには「ノイズなし真値」が無いため true_delta には観測 delta を流用する
    （評価は観測 delta に対して行われる点に注意, docs/07）。
    """
    from .simulate import SimConfig, SimData

    cfg = SimConfig(
        n_donors=proc.n_donors, n_genes=proc.n_genes, n_perts=proc.n_perts,
        geno_dim=proc.geno_features.shape[1], n_control_cells=proc.control_cells.shape[1],
    )
    base_effect = np.nan_to_num(
        np.where(proc.observed.sum(0)[:, None] > 0,
                 proc.delta.sum(0) / np.clip(proc.observed.sum(0)[:, None], 1, None),
                 0.0)).astype(np.float32)
    return SimData(
        config=cfg,
        geno=proc.geno_features.astype(np.float32),
        control_cells=proc.control_cells.astype(np.float32),
        baseline=proc.control_cells.mean(axis=1).astype(np.float32),
        delta=proc.delta.astype(np.float32),
        true_delta=proc.delta.astype(np.float32),     # 実データは観測 delta を真値扱い
        gain=np.zeros((proc.n_donors, proc.n_perts), dtype=np.float32),
        base_effect=base_effect,
        batch=proc.batch.astype(np.int64),
        _batch_shift=np.zeros((int(proc.batch.max()) + 1, proc.n_genes), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# fake ローダ（実データ無しで pipeline を流すため）
# ---------------------------------------------------------------------------
def load_fake_onek1k(n_donors: int = 80, n_genes: int = 200, n_perts: int = 6,
                     cells_per_cond: int = 40, n_variants: int = 500,
                     seed: int = 0) -> tuple[RawScRNA, RawGenotype]:
    """OneK1K 風の生データを合成（生カウント＋genotype）。pipeline の動作確認用。

    実データの代わりに、ドナーごとに control と各摂動の細胞を生成する。摂動効果は
    genotype に依存（個人差）。raw counts は Poisson でそれらしく作る。
    """
    rng = np.random.default_rng(seed)
    dosage = rng.integers(0, 3, size=(n_donors, n_variants)).astype(np.float32)
    # genotype -> 低次元状態 -> ベースライン log 平均
    proj = rng.standard_normal((n_variants, 8)) / np.sqrt(n_variants)
    g = dosage @ proj
    base_log = rng.standard_normal((n_donors, n_genes)) * 0.3 + (g @ rng.standard_normal((8, n_genes))) * 0.2
    pert_eff = rng.standard_normal((n_perts, n_genes)) * 0.5
    pert_geno = rng.standard_normal((n_perts, 8)) / np.sqrt(8)   # 個人差: 効きが genotype 依存

    counts, donor, batch, pert = [], [], [], []
    for d in range(n_donors):
        b = d % 4
        # control
        rate = np.exp(base_log[d])
        c = rng.poisson(np.clip(rate, 0, 50), size=(cells_per_cond, n_genes))
        counts.append(c); donor += [d] * cells_per_cond
        batch += [b] * cells_per_cond; pert += [-1] * cells_per_cond
        # 各摂動
        for p in range(n_perts):
            gain = 1.0 + float(g[d] @ pert_geno[p])
            rate_p = np.exp(base_log[d] + gain * pert_eff[p])
            cp = rng.poisson(np.clip(rate_p, 0, 50), size=(cells_per_cond, n_genes))
            counts.append(cp); donor += [d] * cells_per_cond
            batch += [b] * cells_per_cond; pert += [p] * cells_per_cond

    raw = RawScRNA(
        counts=np.concatenate(counts).astype(np.float32),
        gene_ids=[f"ENSG_{i:05d}" for i in range(n_genes)],
        donor=np.array(donor), batch=np.array(batch), perturbation=np.array(pert),
    )
    geno = RawGenotype(
        dosage=dosage, variant_ids=[f"rs{i}" for i in range(n_variants)],
        donor_ids=[f"donor_{d}" for d in range(n_donors)],
    )
    return raw, geno


def load_anndata(path: str):  # pragma: no cover - 実データ用
    """AnnData(.h5ad) から RawScRNA を作る（scanpy/anndata が必要）。"""
    try:
        import anndata as ad
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "load_anndata には anndata が必要: pip install anndata scanpy") from e
    adata = ad.read_h5ad(path)
    raise NotImplementedError(
        "実データごとに obs のカラム名（donor/batch/perturbation）を割り当てて RawScRNA を"
        f"構築してください。読み込んだ adata: {adata.shape}")
