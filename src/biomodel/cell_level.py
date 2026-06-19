"""cell-level 学習（unpaired・分布マッチング, docs/06）。

pseudobulk ではなく個々の細胞分布を予測し、予測細胞群と観測細胞群の MMD を最小化する。
control 細胞と処置後細胞に対応はつかない（unpaired）ことを前提にする。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .losses import energy_distance, gaussian_mmd2, sinkhorn_divergence
from .model import CellLevelResponseModel, mgm_loss
from .simulate import SimData, sample_treated_cells


@dataclass
class CellTrainConfig:
    epochs: int = 25
    pretrain_epochs: int = 15
    lr: float = 1e-3
    n_cells: int = 48          # 1 ステップあたりの細胞数（予測側・観測側それぞれ）
    pairs_per_epoch: int = 160  # 1 epoch で見る (donor, pert) ペア数の上限（速度のため）
    mask_rate: float = 0.25
    loss_type: str = "mmd"     # "mmd" or "sinkhorn"（最適輸送, docs/06）
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 50
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True


def _dist_loss(cfg: CellTrainConfig, pred, treated):
    """分布マッチング損失（MMD or Sinkhorn）を返す。"""
    if cfg.loss_type == "sinkhorn":
        return sinkhorn_divergence(pred, treated, cfg.sinkhorn_eps, cfg.sinkhorn_iters)
    return gaussian_mmd2(pred, treated)


def _baselines(data: SimData) -> np.ndarray:
    return data.control_cells.mean(axis=1)  # (n_donors, n_genes)


def pretrain_encoder_cells(model: CellLevelResponseModel, data: SimData,
                           cfg: CellTrainConfig) -> None:
    """control 細胞で MGM 事前学習（cell-level encoder の初期化）。"""
    dev = torch.device(cfg.device)
    model.to(dev)
    cells = torch.tensor(data.control_cells.reshape(-1, data.n_genes), device=dev)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.encoder.parameters(), lr=cfg.lr)
    model.train()
    for ep in range(cfg.pretrain_epochs):
        perm = torch.randperm(cells.shape[0], generator=gen, device=dev)
        for i in range(0, cells.shape[0], 512):
            loss = mgm_loss(model.encoder, cells[perm[i:i + 512]], cfg.mask_rate, gen)
            opt.zero_grad(); loss.backward(); opt.step()


def train_cell_level(model: CellLevelResponseModel, data: SimData,
                     train_idx: np.ndarray, cfg: CellTrainConfig) -> list[float]:
    """予測細胞分布と観測細胞分布の MMD を最小化（unpaired 学習）。"""
    dev = torch.device(cfg.device)
    model.to(dev)
    rng = np.random.default_rng(cfg.seed + 1)
    baselines = _baselines(data)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pairs = [(int(d), p) for d in train_idx for p in range(data.n_perts)]
    losses = []
    model.train()
    for ep in range(cfg.epochs):
        rng.shuffle(pairs)
        use = pairs[:cfg.pairs_per_epoch]
        total = 0.0
        for d, p in use:
            # control 細胞（観測）と処置後細胞（観測）を unpaired にサンプル
            ci = rng.integers(0, data.config.n_control_cells, size=cfg.n_cells)
            ctrl = torch.tensor(data.control_cells[d, ci], device=dev)
            treated = torch.tensor(
                sample_treated_cells(data, d, p, cfg.n_cells, rng), device=dev)
            base = torch.tensor(baselines[d], device=dev).unsqueeze(0)
            geno = torch.tensor(data.geno[d], device=dev).unsqueeze(0)
            pid = torch.tensor([p], dtype=torch.long, device=dev)
            effect = model.effect_for(base, geno, pid).expand(cfg.n_cells, -1)
            pred = model(ctrl, effect)
            loss = _dist_loss(cfg, pred, treated)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        losses.append(total / max(len(use), 1))
        if cfg.verbose and (ep % 5 == 0 or ep == cfg.epochs - 1):
            tag = "Sinkhorn" if cfg.loss_type == "sinkhorn" else "MMD^2"
            print(f"[cell-{cfg.loss_type}] epoch {ep:3d}  {tag} {losses[-1]:.4f}")
    return losses


@torch.no_grad()
def predict_pseudobulk_delta(model: CellLevelResponseModel, data: SimData,
                             donor_idx: np.ndarray, n_cells: int = 128,
                             device: str = "cpu", seed: int = 0) -> np.ndarray:
    """cell-level 予測から pseudobulk delta を集約（評価で既存指標に接続）。"""
    dev = torch.device(device)
    model.eval()
    rng = np.random.default_rng(seed)
    baselines = _baselines(data)
    out = np.zeros((len(donor_idx), data.n_perts, data.n_genes), dtype=np.float32)
    for di, d in enumerate(donor_idx):
        ci = rng.integers(0, data.config.n_control_cells, size=n_cells)
        ctrl = torch.tensor(data.control_cells[d, ci], device=dev)
        base = torch.tensor(baselines[d], device=dev).unsqueeze(0)
        geno = torch.tensor(data.geno[d], device=dev).unsqueeze(0)
        for p in range(data.n_perts):
            pid = torch.tensor([p], dtype=torch.long, device=dev)
            effect = model.effect_for(base, geno, pid).expand(n_cells, -1)
            pred = model(ctrl, effect)
            out[di, p] = (pred.mean(0) - ctrl.mean(0)).cpu().numpy()
    return out


@torch.no_grad()
def distribution_energy(model: CellLevelResponseModel, data: SimData,
                        donor_idx: np.ndarray, n_cells: int = 128,
                        device: str = "cpu", seed: int = 0) -> float:
    """予測細胞分布 vs 観測細胞分布のエネルギー距離（分布レベル精度・小さいほど良い）。"""
    dev = torch.device(device)
    model.eval()
    rng = np.random.default_rng(seed + 7)
    baselines = _baselines(data)
    vals = []
    for d in donor_idx:
        base = torch.tensor(baselines[int(d)], device=dev).unsqueeze(0)
        geno = torch.tensor(data.geno[int(d)], device=dev).unsqueeze(0)
        for p in range(data.n_perts):
            ci = rng.integers(0, data.config.n_control_cells, size=n_cells)
            ctrl = torch.tensor(data.control_cells[int(d), ci], device=dev)
            obs = torch.tensor(sample_treated_cells(data, int(d), p, n_cells, rng), device=dev)
            pid = torch.tensor([p], dtype=torch.long, device=dev)
            effect = model.effect_for(base, geno, pid).expand(n_cells, -1)
            pred = model(ctrl, effect)
            vals.append(energy_distance(pred, obs).item())
    return float(np.mean(vals))
