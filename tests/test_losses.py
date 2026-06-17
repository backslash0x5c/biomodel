"""分布マッチング損失のテスト（torch が無ければ skip）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from biomodel.losses import energy_distance, gaussian_mmd2  # noqa: E402


def test_mmd_zero_for_same_distribution():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 8, generator=g)
    y = torch.randn(256, 8, generator=g)
    # 同分布なら MMD^2 はほぼ 0（不偏推定なので僅かに負もあり得る）
    assert abs(gaussian_mmd2(x, y).item()) < 0.05


def test_mmd_positive_for_shifted_distribution():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 8, generator=g)
    y = torch.randn(256, 8, generator=g) + 3.0
    assert gaussian_mmd2(x, y).item() > 0.1


def test_energy_distance_ordering():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(200, 6, generator=g)
    near = torch.randn(200, 6, generator=g) + 0.2
    far = torch.randn(200, 6, generator=g) + 3.0
    assert energy_distance(x, near).item() < energy_distance(x, far).item()
