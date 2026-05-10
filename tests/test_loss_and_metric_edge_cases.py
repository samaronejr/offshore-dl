"""Focused regressions for loss/classifier edge cases."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from offshore_dl.models.base import FocalLoss
from offshore_dl.models.lstm import LSTMModel


def test_weighted_focal_loss_uses_unweighted_true_class_probability() -> None:
    logits = torch.tensor([[2.0, -1.0], [0.25, 1.25]], dtype=torch.float32)
    targets = torch.tensor([0, 1])
    weights = torch.tensor([10.0, 0.5])
    gamma = 2.0

    actual = FocalLoss(gamma=gamma, weight=weights, reduction="none")(logits, targets)

    unweighted_ce = F.cross_entropy(logits, targets, reduction="none")
    weighted_ce = F.cross_entropy(logits, targets, weight=weights, reduction="none")
    expected = (1.0 - torch.exp(-unweighted_ce)).pow(gamma) * weighted_ce

    torch.testing.assert_close(actual, expected)


def test_label_smoothing_reaches_base_classifier_loss() -> None:
    model = LSTMModel(
        task="classification",
        n_vars=3,
        n_classes=4,
        hidden_size=8,
        num_layers=1,
        bidirectional=False,
        label_smoothing=0.2,
    )

    assert isinstance(model._loss_fn, nn.CrossEntropyLoss)
    assert model._loss_fn.label_smoothing == 0.2
