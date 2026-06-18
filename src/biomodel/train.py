"""学習ループ: MGM 事前学習（段1）＋ 摂動応答の教師あり学習（段3）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import PerturbationResponseModel, mgm_loss
from .simulate import SimData


@dataclass
class TrainConfig:
    pretrain_epochs: int = 20
    epochs: int = 60
    lr: float = 1e-3
    pretrain_lr: float = 1e-3
    weight_decay: float = 1e-5
    mask_rate: float = 0.25
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True


def _donor_baseline(data: SimData) -> np.ndarray:
    """control 細胞の pseudobulk（ドナーごとの平均発現）。"""
    return data.control_cells.mean(axis=1)  # (n_donors, n_genes)


def pretrain_encoder(model: PerturbationResponseModel, data: SimData,
                     cfg: TrainConfig) -> list[float]:
    """control 細胞でマスク発現予測（MGM）事前学習（docs/01 段1）。"""
    device = torch.device(cfg.device)
    model.to(device)
    cells = torch.tensor(data.control_cells.reshape(-1, data.n_genes), device=device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.encoder.parameters(), lr=cfg.pretrain_lr,
                           weight_decay=cfg.weight_decay)
    losses = []
    model.train()
    for ep in range(cfg.pretrain_epochs):
        perm = torch.randperm(cells.shape[0], generator=gen, device=device)
        total, nb = 0.0, 0
        for i in range(0, cells.shape[0], 512):
            batch = cells[perm[i:i + 512]]
            loss = mgm_loss(model.encoder, batch, mask_rate=cfg.mask_rate, generator=gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        losses.append(total / max(nb, 1))
        if cfg.verbose and (ep % 5 == 0 or ep == cfg.pretrain_epochs - 1):
            print(f"[pretrain] epoch {ep:3d}  MGM loss {losses[-1]:.4f}")
    return losses


def _build_examples(data: SimData, donor_idx: np.ndarray):
    """(donor, pert) の観測済みの組を平坦化して教師ありデータを作る。

    観測マスク（data.observed）が <1 の場合、未観測ペアは学習から除外する
    （疎な実データに対応, docs/07）。
    """
    baseline = _donor_baseline(data)              # (n_donors, n_genes)
    d_list, p_list = np.meshgrid(donor_idx, np.arange(data.n_perts), indexing="ij")
    d_flat = d_list.reshape(-1)
    p_flat = p_list.reshape(-1)
    keep = data.observed_mask()[d_flat, p_flat] > 0
    d_flat, p_flat = d_flat[keep], p_flat[keep]
    X_base = baseline[d_flat]                     # (N, n_genes)
    X_geno = data.geno[d_flat]                    # (N, geno_dim)
    Y = data.delta[d_flat, p_flat]                # (N, n_genes)
    return (torch.tensor(X_base), torch.tensor(X_geno),
            torch.tensor(p_flat, dtype=torch.long), torch.tensor(Y))


def train_supervised(model: PerturbationResponseModel, data: SimData,
                     train_idx: np.ndarray, cfg: TrainConfig) -> list[float]:
    """摂動応答（pseudobulk delta = 処置効果）の教師あり学習。"""
    device = torch.device(cfg.device)
    model.to(device)
    Xb, Xg, P, Y = (t.to(device) for t in _build_examples(data, train_idx))
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    losses = []
    model.train()
    n = Xb.shape[0]
    for ep in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen, device=device)
        total, nb = 0.0, 0
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            pred = model(Xb[idx], Xg[idx], P[idx])
            loss = torch.mean((pred - Y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        losses.append(total / max(nb, 1))
        if cfg.verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(f"[train]    epoch {ep:3d}  delta MSE {losses[-1]:.4f}")
    return losses


@torch.no_grad()
def predict_delta(model: PerturbationResponseModel, data: SimData,
                  donor_idx: np.ndarray, device: str = "cpu") -> np.ndarray:
    """指定ドナー × 全摂動の delta 予測を (len(donor_idx), n_perts, n_genes) で返す。"""
    model.eval()
    dev = torch.device(device)
    baseline = _donor_baseline(data)
    out = np.zeros((len(donor_idx), data.n_perts, data.n_genes), dtype=np.float32)
    Xb_all = torch.tensor(baseline[donor_idx], device=dev)
    Xg_all = torch.tensor(data.geno[donor_idx], device=dev)
    for p in range(data.n_perts):
        pid = torch.full((len(donor_idx),), p, dtype=torch.long, device=dev)
        out[:, p, :] = model(Xb_all, Xg_all, pid).cpu().numpy()
    return out
