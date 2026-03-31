"""Evaluation — CV strategies, metrics, baselines, reporting, statistical testing."""

from offshore_dl.evaluation.cv import (
    BaseCVStrategy,
    ExpandingWindowCV,
    FoldNormalizer,
    HoldoutSplitter,
    LeakageGuard,
    StratifiedGroupKFoldCV,
    StratifiedGroupKFoldSKLearn,
    SlidingWindowCV,
    TemporalSplitCV,
)

__all__ = [
    "BaseCVStrategy",
    "ExpandingWindowCV",
    "FoldNormalizer",
    "HoldoutSplitter",
    "LeakageGuard",
    "SlidingWindowCV",
    "StratifiedGroupKFoldCV",
    "StratifiedGroupKFoldSKLearn",
    "TemporalSplitCV",
]
