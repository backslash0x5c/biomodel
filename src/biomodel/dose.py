"""用量反応（dose-response）と薬の組み合わせ（combination）の予測（docs/11 §3）。

yes/no でなく「適量」を当てる。Hill/Emax 曲線で、効きやすさ(EC50)・最大効果(Emax)が
genotype に依存（＝必要量の個人差。ワルファリン用量が遺伝子型で変わるのと同じ発想）。
自己完結（独自のミニ simulator と小さなモデル）で、本体モデルには手を入れない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .model import mlp


# ---------------------------------------------------------------------------
# 合成データ: genotype 依存の Hill 用量反応
# ---------------------------------------------------------------------------
@dataclass
class DoseConfig:
    n_patients: int = 200
    n_drugs: int = 6
    geno_dim: int = 6
    hill: float = 1.5                       # Hill 係数
    doses: tuple = field(default_factory=lambda: tuple(np.linspace(0.0, 5.0, 21)))
    n_obs_doses: int = 5                    # 学習で観測する用量点数（患者×薬あたり）
    noise: float = 0.05
    seed: int = 0


@dataclass
class DoseData:
    config: DoseConfig
    geno: np.ndarray         # (n_patients, geno_dim)
    Emax: np.ndarray         # (n_patients, n_drugs)  最大効果（個人差）
    EC50: np.ndarray         # (n_patients, n_drugs)  半数効果濃度（個人差＝必要量）
    obs: dict                # (patient,drug)->(doses, responses) 観測点
    doses_grid: np.ndarray   # (n_grid,) 評価用の用量グリッド


def hill(dose: np.ndarray, Emax, EC50, h) -> np.ndarray:
    d = np.asarray(dose) ** h
    return Emax * d / (EC50 ** h + d)


def simulate_dose(cfg: DoseConfig | None = None) -> DoseData:
    cfg = cfg or DoseConfig()
    rng = np.random.default_rng(cfg.seed)
    g = rng.standard_normal((cfg.n_patients, cfg.geno_dim)).astype(np.float32)
    # Emax / EC50 を genotype 依存に（効きの強さ・必要量の個人差）
    baseE = rng.uniform(0.6, 1.0, cfg.n_drugs)
    wE = rng.standard_normal((cfg.n_drugs, cfg.geno_dim)) / np.sqrt(cfg.geno_dim)
    baseK = rng.uniform(0.5, 2.0, cfg.n_drugs)
    wK = rng.standard_normal((cfg.n_drugs, cfg.geno_dim)) / np.sqrt(cfg.geno_dim)
    Emax = np.clip(baseE[None] + 0.3 * (g @ wE.T), 0.1, None)
    EC50 = np.exp(np.log(baseK)[None] + 0.5 * (g @ wK.T))     # 正値・対数線形
    grid = np.array(cfg.doses, dtype=np.float64)

    obs = {}
    for p in range(cfg.n_patients):
        for d in range(cfg.n_drugs):
            ds = rng.choice(grid[grid > 0], size=cfg.n_obs_doses, replace=False)
            r = hill(ds, Emax[p, d], EC50[p, d], cfg.hill)
            r = r + rng.standard_normal(len(ds)) * cfg.noise
            obs[(p, d)] = (ds.astype(np.float32), r.astype(np.float32))
    return DoseData(cfg, g, Emax.astype(np.float32), EC50.astype(np.float32), obs, grid)


def patient_split(data: DoseData, n_test: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.config.n_patients)
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


# ---------------------------------------------------------------------------
# モデル: (genotype, drug, dose) -> response
# ---------------------------------------------------------------------------
class DoseResponseModel(nn.Module):
    def __init__(self, geno_dim: int, n_drugs: int, d_embed: int = 16, hidden: int = 64):
        super().__init__()
        self.drug = nn.Embedding(n_drugs, d_embed)
        self.net = mlp(geno_dim + d_embed + 1, hidden, 1, depth=3)

    def forward(self, geno, drug_id, dose):
        de = self.drug(drug_id)
        x = torch.cat([geno, de, dose.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


def _build_dose_examples(data: DoseData, idx: np.ndarray):
    G, D, S, Y = [], [], [], []
    for p in idx:
        for d in range(data.config.n_drugs):
            ds, r = data.obs[(int(p), d)]
            for dose, resp in zip(ds, r):
                G.append(data.geno[int(p)]); D.append(d); S.append(dose); Y.append(resp)
    return (torch.tensor(np.array(G)), torch.tensor(np.array(D), dtype=torch.long),
            torch.tensor(np.array(S), dtype=torch.float32), torch.tensor(np.array(Y)))


def train_dose(data: DoseData, train_idx: np.ndarray, epochs: int = 200, lr: float = 1e-3,
               seed: int = 0, verbose: bool = False) -> DoseResponseModel:
    torch.manual_seed(seed)
    model = DoseResponseModel(data.config.geno_dim, data.config.n_drugs)
    G, D, S, Y = _build_dose_examples(data, train_idx)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = G.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 512):
            j = perm[i:i + 512]
            loss = torch.mean((model(G[j], D[j], S[j]) - Y[j]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ep % 50 == 0:
            print(f"[dose] epoch {ep} mse {loss.item():.4f}")
    return model


@torch.no_grad()
def predict_curve(model: DoseResponseModel, data: DoseData, patient: int, drug: int) -> np.ndarray:
    model.eval()
    grid = torch.tensor(data.doses_grid, dtype=torch.float32)
    geno = torch.tensor(data.geno[patient]).repeat(len(grid), 1)
    did = torch.full((len(grid),), drug, dtype=torch.long)
    return model(geno, did, grid).cpu().numpy()


def dose_to_target(curve: np.ndarray, grid: np.ndarray, target: float) -> float:
    """応答が target に最初に到達する用量（線形補間）。到達しなければ最大用量。"""
    above = np.where(curve >= target)[0]
    if len(above) == 0:
        return float(grid[-1])
    i = above[0]
    if i == 0:
        return float(grid[0])
    x0, x1, y0, y1 = grid[i - 1], grid[i], curve[i - 1], curve[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


# ---------------------------------------------------------------------------
# 組み合わせ（combination）: 2 剤併用の相乗/拮抗を genotype 依存で
# ---------------------------------------------------------------------------
def simulate_combination(data: DoseData, dose_level: float = 2.0, seed: int = 0):
    """固定用量で 2 剤併用したときの効果と相乗項（genotype 依存）を作る。

    combo_{p,(a,b)} = R_a + R_b + synergy,  synergy = s0 + g·wS（+で相乗・-で拮抗）。
    返り値: pairs, single (n_pat,n_drug), synergy (n_pat,n_pairs), wS（真の相乗駆動因子）。
    """
    cfg = data.config
    rng = np.random.default_rng(seed + 99)
    single = hill(dose_level, data.Emax, data.EC50, cfg.hill)        # (n_pat, n_drug)
    pairs = [(a, b) for a in range(cfg.n_drugs) for b in range(a + 1, cfg.n_drugs)]
    wS = rng.standard_normal((len(pairs), cfg.geno_dim)) / np.sqrt(cfg.geno_dim)
    s0 = rng.standard_normal(len(pairs)) * 0.1
    synergy = s0[None] + 0.4 * (data.geno @ wS.T)                    # (n_pat, n_pairs)
    return pairs, single.astype(np.float32), synergy.astype(np.float32), wS.astype(np.float32)
