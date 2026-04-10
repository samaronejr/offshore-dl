"""Evaluation — CV strategies, metrics, baselines, reporting, statistical testing."""

from offshore_dl.evaluation.cv import (
    BaseCVStrategy,
    ExpandingWindowCV,
    FoldNormalizer,
    GroupedExpandingWindowCV,
    GroupedTemporalHoldoutSplitter,
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
    "GroupedExpandingWindowCV",
    "GroupedTemporalHoldoutSplitter",
    "HoldoutSplitter",
    "LeakageGuard",
    "SlidingWindowCV",
    "StratifiedGroupKFoldCV",
    "StratifiedGroupKFoldSKLearn",
    "TemporalSplitCV",
]
