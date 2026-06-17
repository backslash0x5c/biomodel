"""分布マッチング損失（unpaired な細胞分布の比較）。

cell-level 学習では、予測した処置後細胞と観測した処置後細胞の間に対応がつかない
（unpaired）。そこで点ごとの距離ではなく **分布間距離** を最小化する（docs/06）。
"""

from __future__ import annotations

import torch


def _pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """二乗ユークリッド距離行列 (n, m)。x:(n,d), y:(m,d)。"""
    x2 = (x * x).sum(1, keepdim=True)          # (n,1)
    y2 = (y * y).sum(1, keepdim=True).t()      # (1,m)
    return (x2 + y2 - 2.0 * x @ y.t()).clamp_min(0.0)


def gaussian_mmd2(x: torch.Tensor, y: torch.Tensor,
                  bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """多帯域ガウシアンカーネルによる MMD^2 の不偏推定（unpaired 分布距離）。

    x:(n,d) 予測細胞, y:(m,d) 観測細胞。距離スケールは中央値ヒューリスティックで正規化。
    """
    n, m = x.shape[0], y.shape[0]
    dxx = _pdist2(x, x)
    dyy = _pdist2(y, y)
    dxy = _pdist2(x, y)
    # 中央値ヒューリスティックでスケール基準を決める
    with torch.no_grad():
        med = torch.median(dxy.detach()).clamp_min(1e-6)

    kxx = torch.zeros((), device=x.device)
    kyy = torch.zeros((), device=x.device)
    kxy = torch.zeros((), device=x.device)
    for b in bandwidths:
        gamma = 1.0 / (2.0 * b * med)
        kxx = kxx + torch.exp(-gamma * dxx)
        kyy = kyy + torch.exp(-gamma * dyy)
        kxy = kxy + torch.exp(-gamma * dxy)

    # 対角（自己項）を除いた不偏推定
    if n > 1:
        kxx = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
    else:
        kxx = kxx.sum() / max(n * n, 1)
    if m > 1:
        kyy = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
    else:
        kyy = kyy.sum() / max(m * m, 1)
    kxy = kxy.sum() / (n * m)
    return kxx + kyy - 2.0 * kxy


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """エネルギー距離（分布間距離・評価用）。小さいほど分布が近い。"""
    dxy = _pdist2(x, y).clamp_min(0).sqrt().mean()
    dxx = _pdist2(x, x).clamp_min(0).sqrt().mean()
    dyy = _pdist2(y, y).clamp_min(0).sqrt().mean()
    return 2.0 * dxy - dxx - dyy
