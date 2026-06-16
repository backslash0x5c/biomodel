"""biomodel — 個人差を予測する摂動応答基盤モデル（プロトタイプ）。

詳細は docs/ を参照。主要な公開 API:
    simulate / SimConfig / SimData / donor_split   : 個人差を持つ合成データ
    PerturbationResponseModel                       : 段1〜段3 ＋ デコーダの統合モデル
    TrainConfig / pretrain_encoder / train_supervised / predict_delta
    evaluate_predictions / population_mean_baseline : leave-one-donor-out 評価
"""

from .simulate import SimConfig, SimData, donor_split, simulate

__all__ = [
    "SimConfig",
    "SimData",
    "simulate",
    "donor_split",
]

# torch 依存部はオプショナル import（torch 不在でも simulate と numpy デモは動く）
try:  # pragma: no cover
    import torch  # noqa: F401

    from .evaluate import (
        EvalResult,
        evaluate_predictions,
        population_mean_baseline,
        true_delta_test,
    )
    from .model import PerturbationResponseModel
    from .train import (
        TrainConfig,
        predict_delta,
        pretrain_encoder,
        train_supervised,
    )

    __all__ += [
        "PerturbationResponseModel",
        "TrainConfig",
        "pretrain_encoder",
        "train_supervised",
        "predict_delta",
        "evaluate_predictions",
        "population_mean_baseline",
        "true_delta_test",
        "EvalResult",
    ]
    _HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_TORCH = False
