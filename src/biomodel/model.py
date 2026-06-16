"""個人差を予測する摂動応答モデル（PyTorch）。

三段構え（docs/01）:
    段1  ExpressionEncoder      : scRNA -> z_cell（MGM で自己教師あり事前学習）
    段2  PerturbationEncoder    : 摂動 -> z_pert
    段3  IndividualEncoder + Φ  : 個人条件づけ ＋ 摂動×個人の相互作用（個人差の本体）
    出力 ResponseDecoder        : delta（処置効果）を予測

相互作用 Φ は FiLM / CrossAttn / Additive(=CPA相当の対照) を選べる（docs/02 §3, docs/04 §4）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 段1: 発現 encoder（マスク発現予測 = MGM で事前学習可能）
# ---------------------------------------------------------------------------
class ExpressionEncoder(nn.Module):
    """遺伝子発現ベクトル -> 潜在表現 z。

    MGM 事前学習用の reconstruct ヘッドを内蔵。入力にマスク指示を結合できる
    （マスク位置は値 0 ＋ マスクフラグ 1）。細胞単位でも pseudobulk でも使える。
    """

    def __init__(self, n_genes: int, z_dim: int = 64, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.n_genes = n_genes
        self.z_dim = z_dim
        # 入力は [発現値(n_genes), マスクフラグ(n_genes)]
        self.encoder = mlp(2 * n_genes, hidden, z_dim, depth=3, dropout=dropout)
        self.recon_head = mlp(z_dim, hidden, n_genes, depth=2, dropout=dropout)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros_like(x)
        x_in = torch.cat([x * (1.0 - mask), mask], dim=-1)
        return self.encoder(x_in)

    def reconstruct(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, mask)
        return self.recon_head(z)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode(x, mask)


# ---------------------------------------------------------------------------
# 段2: 摂動 encoder
# ---------------------------------------------------------------------------
class PerturbationEncoder(nn.Module):
    """摂動 ID（または特徴）-> z_pert。

    プロトタイプでは学習埋め込み。実データでは化合物構造 encoder（chemCPA 流）や
    遺伝子グラフ事前知識（GEARS 流）に差し替える（docs/03 B）。
    """

    def __init__(self, n_perts: int, p_dim: int = 32, feat_dim: int | None = None, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(n_perts, p_dim)
        self.feat_proj = nn.Linear(feat_dim, p_dim) if feat_dim else None
        self.proj = mlp(p_dim, hidden, p_dim, depth=2)

    def forward(self, pert_id: torch.Tensor, pert_feat: torch.Tensor | None = None) -> torch.Tensor:
        z = self.embed(pert_id)
        if self.feat_proj is not None and pert_feat is not None:
            z = z + self.feat_proj(pert_feat)
        return self.proj(z)


# ---------------------------------------------------------------------------
# 段3: 個人 encoder（genotype + baseline + 共変量 -> z_indiv）
# ---------------------------------------------------------------------------
class IndividualEncoder(nn.Module):
    """個人 embedding を amortized に推論（free embedding ではない -> 未知ドナーに汎化）。

    use_genotype=False で genotype を落とすと ablation（docs/04 §4）。
    """

    def __init__(self, geno_dim: int, baseline_dim: int, cov_dim: int = 0,
                 i_dim: int = 32, hidden: int = 64, use_genotype: bool = True):
        super().__init__()
        self.use_genotype = use_genotype
        in_dim = baseline_dim + cov_dim + (geno_dim if use_genotype else 0)
        self.net = mlp(in_dim, hidden, i_dim, depth=2)
        self.geno_dim = geno_dim

    def forward(self, baseline_z: torch.Tensor, geno: torch.Tensor,
                cov: torch.Tensor | None = None) -> torch.Tensor:
        parts = [baseline_z]
        if self.use_genotype:
            parts.append(geno)
        if cov is not None:
            parts.append(cov)
        return self.net(torch.cat(parts, dim=-1))


# ---------------------------------------------------------------------------
# 相互作用 Φ（摂動 × 個人）— 個人差の本体
# ---------------------------------------------------------------------------
class FiLMInteraction(nn.Module):
    """既定。z_indiv から (gamma, beta) を生成し摂動効果を変調: gamma*e + beta。"""

    def __init__(self, p_dim: int, i_dim: int, effect_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.effect = mlp(p_dim, hidden, effect_dim, depth=2)
        self.film = mlp(i_dim, hidden, 2 * effect_dim, depth=2)
        self.effect_dim = effect_dim

    def forward(self, z_pert: torch.Tensor, z_indiv: torch.Tensor) -> torch.Tensor:
        e = self.effect(z_pert)
        gamma, beta = self.film(z_indiv).chunk(2, dim=-1)
        return (1.0 + gamma) * e + beta


class CrossAttnInteraction(nn.Module):
    """摂動トークンと個人トークンの cross-attention（組み合わせ汎化に強い）。"""

    def __init__(self, p_dim: int, i_dim: int, effect_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.q = nn.Linear(p_dim, effect_dim)
        self.kv = nn.Linear(i_dim, effect_dim)
        self.attn = nn.MultiheadAttention(effect_dim, n_heads, batch_first=True)
        self.out = nn.Linear(effect_dim, effect_dim)
        self.effect_dim = effect_dim

    def forward(self, z_pert: torch.Tensor, z_indiv: torch.Tensor) -> torch.Tensor:
        q = self.q(z_pert).unsqueeze(1)
        kv = self.kv(z_indiv).unsqueeze(1)
        a, _ = self.attn(q, kv, kv)
        return self.out(a.squeeze(1))


class AdditiveInteraction(nn.Module):
    """CPA 相当の対照（加法）。z_indiv を使わない -> 個人差を表現できない。

    docs/04 §4 の ablation 用。これに勝てて初めて「個人差を予測できた」と言える。
    """

    def __init__(self, p_dim: int, i_dim: int, effect_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.effect = mlp(p_dim, hidden, effect_dim, depth=2)
        self.effect_dim = effect_dim

    def forward(self, z_pert: torch.Tensor, z_indiv: torch.Tensor) -> torch.Tensor:
        return self.effect(z_pert)


INTERACTIONS = {
    "film": FiLMInteraction,
    "crossattn": CrossAttnInteraction,
    "additive": AdditiveInteraction,
}


# ---------------------------------------------------------------------------
# デコーダ（delta = 処置効果 を予測）
# ---------------------------------------------------------------------------
class ResponseDecoder(nn.Module):
    """(baseline_z, effect) -> delta（残差/処置効果）を予測（docs/01 §2 段3）。"""

    def __init__(self, baseline_dim: int, effect_dim: int, n_genes: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(baseline_dim + effect_dim, hidden, n_genes, depth=3)

    def forward(self, baseline_z: torch.Tensor, effect: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([baseline_z, effect], dim=-1))


# ---------------------------------------------------------------------------
# 統合モデル
# ---------------------------------------------------------------------------
class PerturbationResponseModel(nn.Module):
    """個人差を予測する摂動応答モデル（段1〜段3 ＋ デコーダ）。

    forward は pseudobulk delta（処置効果 Δ_{d,p}, = CATE）を予測する。
    """

    def __init__(self, n_genes: int, n_perts: int, geno_dim: int,
                 z_dim: int = 64, p_dim: int = 32, i_dim: int = 32, effect_dim: int = 64,
                 interaction: str = "film", use_genotype: bool = True, cov_dim: int = 0,
                 pert_feat_dim: int | None = None):
        super().__init__()
        if interaction not in INTERACTIONS:
            raise ValueError(f"unknown interaction {interaction!r}; choose from {list(INTERACTIONS)}")
        self.encoder = ExpressionEncoder(n_genes, z_dim=z_dim)
        self.pert_encoder = PerturbationEncoder(n_perts, p_dim=p_dim, feat_dim=pert_feat_dim)
        self.indiv_encoder = IndividualEncoder(
            geno_dim, baseline_dim=z_dim, cov_dim=cov_dim, i_dim=i_dim, use_genotype=use_genotype)
        self.interaction = INTERACTIONS[interaction](p_dim, i_dim, effect_dim=effect_dim)
        self.decoder = ResponseDecoder(z_dim, effect_dim, n_genes)
        self.interaction_name = interaction
        self.use_genotype = use_genotype

    def baseline_embedding(self, baseline_expr: torch.Tensor) -> torch.Tensor:
        """control pseudobulk -> z_donor_baseline（マスクなし encode）。"""
        return self.encoder.encode(baseline_expr)

    def forward(self, baseline_expr: torch.Tensor, geno: torch.Tensor, pert_id: torch.Tensor,
                cov: torch.Tensor | None = None, pert_feat: torch.Tensor | None = None) -> torch.Tensor:
        z_base = self.baseline_embedding(baseline_expr)
        z_pert = self.pert_encoder(pert_id, pert_feat)
        z_indiv = self.indiv_encoder(z_base, geno, cov)
        effect = self.interaction(z_pert, z_indiv)
        return self.decoder(z_base, effect)


def mgm_loss(encoder: ExpressionEncoder, cells: torch.Tensor, mask_rate: float = 0.25,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """マスク発現予測（MGM）損失。control 細胞での自己教師あり事前学習（docs/01 段1）。"""
    mask = (torch.rand(cells.shape, generator=generator, device=cells.device) < mask_rate).float()
    recon = encoder.reconstruct(cells, mask)
    diff = (recon - cells) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom
