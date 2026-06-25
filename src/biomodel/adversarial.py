"""batch 敵対学習（identifiability: 真の生物差と batch effect の分離, docs/09）。

ドナー差には「真の生物学的個人差」と「技術的 batch effect」が交絡する。これを分離しないと
「個人差を当てた」が実は batch を当てているだけになりうる（docs/02 §4）。勾配反転層（GRL）で
encoder 表現が batch を予測できないように学習し、batch 不変な表現を得る。genotype は batch と
独立なので、genotype 由来の差は batch ではないと解釈できる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .model import BatchDiscriminator, PerturbationResponseModel, grad_reverse, mlp
from .simulate import SimData
from .train import TrainConfig, _donor_baseline


@dataclass
class AdvConfig(TrainConfig):
    adv_weight: float = 0.5       # 敵対損失の重み（0 で通常学習）
    grl_lambda: float = 1.0       # 勾配反転の強さ（学習進行でランプ）
    disc_lr_mult: float = 5.0     # 判別器の学習率倍率（DANN 安定化）
    grad_clip: float = 1.0        # encoder/モデル側の勾配クリップ（DANN 安定化）


def _examples_with_batch(data: SimData, donor_idx: np.ndarray):
    """観測済み (donor, pert) 例 ＋ 各例の batch ラベルを作る。"""
    baseline = _donor_baseline(data)
    d_list, p_list = np.meshgrid(donor_idx, np.arange(data.n_perts), indexing="ij")
    d_flat = d_list.reshape(-1)
    p_flat = p_list.reshape(-1)
    keep = data.observed_mask()[d_flat, p_flat] > 0
    d_flat, p_flat = d_flat[keep], p_flat[keep]
    return (torch.tensor(baseline[d_flat]), torch.tensor(data.geno[d_flat]),
            torch.tensor(p_flat, dtype=torch.long), torch.tensor(data.delta[d_flat, p_flat]),
            torch.tensor(data.batch[d_flat], dtype=torch.long))


def train_supervised_adversarial(model: PerturbationResponseModel, data: SimData,
                                 train_idx: np.ndarray, cfg: AdvConfig) -> dict:
    """delta 回帰 ＋ **cell-level** batch 敵対損失（DANN 方式, GRL）を同時学習。

    encoder の細胞表現 encode(cell) に GRL を噛ませて判別器に通し、batch 不変にする。
    cell-level にするのは probe（細胞表現で batch を当てる）と整合させ、強い batch
    信号にも十分な勾配を与えるため。grl_lambda は学習進行に応じてランプ（DANN 流）。
    adv_weight=0 なら通常学習と等価。
    """
    dev = torch.device(cfg.device)
    model.to(dev)
    n_batches = int(data.batch.max()) + 1
    disc = BatchDiscriminator(model.encoder.z_dim, n_batches).to(dev)
    # delta 用の (donor,pert) 例
    Xb, Xg, P, Y, _ = (t.to(dev) for t in _examples_with_batch(data, train_idx))
    # 敵対用の control 細胞プール（細胞ごとに batch ラベル）
    ncc = data.config.n_control_cells
    cells = torch.tensor(data.control_cells[train_idx].reshape(-1, data.n_genes), device=dev)
    cell_batch = torch.tensor(
        np.repeat(data.batch[train_idx], ncc), dtype=torch.long, device=dev)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed + 1)
    # DANN 安定化: モデルと判別器で別オプティマイザ（判別器は高 lr）＋勾配クリップ
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    dopt = torch.optim.Adam(disc.parameters(), lr=cfg.lr * cfg.disc_lr_mult)
    hist = {"delta": [], "adv": []}
    model.train(); disc.train()
    n = Xb.shape[0]
    for ep in range(cfg.epochs):
        lambd = cfg.grl_lambda * min(1.0, (ep + 1) / max(1, cfg.epochs * 0.5))  # ランプ
        perm = torch.randperm(n, generator=gen, device=dev)
        td, ta, nb = 0.0, 0.0, 0
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            z_base = model.baseline_embedding(Xb[idx])
            z_pert = model.pert_encoder(P[idx])
            z_indiv = model.indiv_encoder(z_base, Xg[idx])
            pred = model.decoder(z_base, model.interaction(z_pert, z_indiv))
            delta_loss = torch.mean((pred - Y[idx]) ** 2)
            # cell-level 敵対: ランダムな control 細胞で batch を当てさせる（GRL）
            ci = torch.randint(0, cells.shape[0], (256,), generator=gen, device=dev)
            z_cell = model.encoder.encode(cells[ci])
            adv_loss = F.cross_entropy(disc(grad_reverse(z_cell, lambd)), cell_batch[ci])
            loss = delta_loss + cfg.adv_weight * adv_loss
            opt.zero_grad(); dopt.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); dopt.step()
            td += delta_loss.item(); ta += adv_loss.item(); nb += 1
        hist["delta"].append(td / max(nb, 1))
        hist["adv"].append(ta / max(nb, 1))
        if cfg.verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(f"[adv] epoch {ep:3d}  delta {hist['delta'][-1]:.4f}  "
                  f"adv_ce {hist['adv'][-1]:.4f}  lambda {lambd:.2f}")
    return hist


@torch.no_grad()
def _encode_cells(model: PerturbationResponseModel, data: SimData,
                  donor_idx: np.ndarray, device: str = "cpu"):
    """donor の control 細胞を encode し、(z, batch_label) を返す（probe 用）。"""
    dev = torch.device(device)
    model.eval()
    zs, bs = [], []
    for d in donor_idx:
        cells = torch.tensor(data.control_cells[int(d)], device=dev)
        zs.append(model.encoder.encode(cells).cpu().numpy())
        bs.append(np.full(cells.shape[0], data.batch[int(d)], dtype=np.int64))
    return np.concatenate(zs), np.concatenate(bs)


def batch_probe_accuracy(model: PerturbationResponseModel, data: SimData,
                         donor_idx: np.ndarray, epochs: int = 100, seed: int = 0,
                         device: str = "cpu", linear: bool = True) -> float:
    """凍結した encoder の細胞表現から batch を当てる probe の検証精度。

    既定は **線形 probe**（線形分離性の標準的な測り方）。高いほど表現が batch と
    絡んでいる（entangled）。chance = 1/n_batches に近いほど batch 不変。
    敵対学習の効果を定量化する（docs/09）。linear=False で MLP probe。
    """
    z, b = _encode_cells(model, data, donor_idx, device)
    n_batches = int(data.batch.max()) + 1
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(z))
    z, b = z[perm], b[perm]
    n_tr = int(len(z) * 0.7)
    dev = torch.device(device)
    Ztr = torch.tensor(z[:n_tr], device=dev); Btr = torch.tensor(b[:n_tr], device=dev)
    Zva = torch.tensor(z[n_tr:], device=dev); Bva = torch.tensor(b[n_tr:], device=dev)
    probe = (torch.nn.Linear(z.shape[1], n_batches) if linear
             else mlp(z.shape[1], 64, n_batches, depth=2)).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    g = torch.Generator(device=dev).manual_seed(seed)
    probe.train()
    for _ in range(epochs):
        idx = torch.randperm(Ztr.shape[0], generator=g, device=dev)
        for i in range(0, Ztr.shape[0], 256):
            j = idx[i:i + 256]
            loss = F.cross_entropy(probe(Ztr[j]), Btr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        acc = (probe(Zva).argmax(1) == Bva).float().mean().item()
    return acc
