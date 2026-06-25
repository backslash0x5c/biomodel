"""実データ検証指標のテスト（numpy のみ）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomodel.validation import auroc, validate_recommendations


def test_auroc_perfect_and_random():
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert auroc(scores, labels) == 1.0
    # 完全に逆なら 0
    assert auroc(-scores, labels) == 0.0
    # 単一クラスは nan
    assert np.isnan(auroc(scores, np.ones_like(labels)))


def test_validate_recommendations_perfect():
    rng = np.random.default_rng(0)
    true = rng.standard_normal((30, 5))
    rep = validate_recommendations(true, true)        # 予測=真値
    assert rep.top1_accuracy == 1.0
    assert rep.mean_regret == 0.0
    assert rep.score_spearman > 0.99
    assert rep.mean_responder_auroc > 0.99


def test_validate_recommendations_beats_constant():
    rng = np.random.default_rng(1)
    true = rng.standard_normal((40, 6))
    good = true + 0.1 * rng.standard_normal((40, 6))   # 真値に近い予測
    const = np.tile(true.mean(0), (40, 1))             # 全員一律（個人差なし）
    r_good = validate_recommendations(good, true)
    r_const = validate_recommendations(const, true)
    assert r_good.top1_accuracy > r_const.top1_accuracy
    assert r_good.mean_responder_auroc > r_const.mean_responder_auroc
