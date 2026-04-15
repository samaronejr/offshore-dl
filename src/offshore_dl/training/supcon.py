"""Supervised Contrastive Learning for time-series classification.

Implements SupCon loss (Khosla et al., 2020) and a pre-training utility
that trains a model's encoder with contrastive loss before downstream
fine-tuning.

References:
    Khosla et al. (2020). Supervised Contrastive Learning.
    https://arxiv.org/abs/2004.11362
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Given a batch of L2-normalized embeddings and integer class labels,
    treats samples sharing the same label as positive pairs and all
    others as negatives.

    Args:
        temperature: Scaling temperature for cosine similarity. Default 0.07.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            embeddings: L2-normalized embeddings of shape ``(B, D)``.
            labels: Integer class labels of shape ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Pairwise cosine similarities scaled by temperature: (B, B)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Positive pair mask: mask[i,j] = 1 if labels[i] == labels[j]
        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

        # Self-pair mask (diagonal): exclude i==j from both numerator and denominator
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = pos_mask.masked_fill(self_mask, 0.0)

        # For numerical stability subtract the row-wise max before exp
        # (standard log-sum-exp trick); self pairs are excluded from denominator
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Exponentiate, zero out self pairs from denominator
        exp_sim = torch.exp(sim)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)

        # log-probability for each pair: log(exp(sim_ij) / sum_k exp(sim_ik))
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-9))

        # For each anchor i, average log-prob over its positives
        n_positives = pos_mask.sum(dim=1)  # (B,)

        # Anchors with no positives (singleton classes) are excluded
        has_positive = n_positives > 0
        if not has_positive.any():
            logger.warning(
                "SupConLoss: no positive pairs found in this batch — "
                "returning zero loss. Consider larger batch size or "
                "ensuring multiple samples per class."
            )
            return embeddings.sum() * 0.0  # differentiable zero

        # Mean log-prob of positives per anchor, then mean over valid anchors
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / n_positives.clamp(min=1)
        loss = -mean_log_prob_pos[has_positive].mean()
        return loss


class _ProjectionHead(nn.Module):
    """Two-layer MLP projection head with L2 normalization.

    Takes the model's output (logits or feature vector of size ``in_dim``)
    and projects to a unit-sphere embedding of size ``proj_dim``.

    Args:
        in_dim: Input dimensionality (e.g. number of classes for logit-based
            models, or the penultimate feature dimension).
        proj_dim: Output embedding dimension.
    """

    def __init__(self, in_dim: int, proj_dim: int) -> None:
        super().__init__()
        hidden = max(in_dim, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize embeddings.

        Args:
            x: Input tensor ``(B, in_dim)``.

        Returns:
            L2-normalized embeddings ``(B, proj_dim)``.
        """
        return F.normalize(self.net(x), dim=1)


class SupConPreTrainer:
    """Pre-train a model's encoder with supervised contrastive learning.

    The approach wraps the full model's output (logits of shape
    ``(B, n_classes)``) with a two-layer MLP projection head that maps to
    a unit-sphere embedding, then trains with :class:`SupConLoss`.  After
    pre-training the encoder weights (i.e. the base model's state dict,
    excluding the projection head) are returned for loading into a
    classification model before fine-tuning.

    Usage::

        pretrainer = SupConPreTrainer(LSTMModel, model_kwargs, device="cuda")
        encoder_state = pretrainer.pretrain(dataset, train_indices, epochs=50)
        model = LSTMModel(**model_kwargs)
        model.load_state_dict(encoder_state, strict=False)
        # fine-tune model normally ...

    Args:
        model_class: Class (not instance) of the model to pre-train.
        model_kwargs: Keyword arguments passed to ``model_class(**model_kwargs)``.
        device: Torch device string. Default ``"cpu"``.
        projection_dim: Output dimensionality of the projection head.
        temperature: SupCon temperature. Default 0.07 (Khosla et al.).
    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        device: str = "cpu",
        projection_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.device = torch.device(device)
        self.projection_dim = projection_dim
        self.temperature = temperature

    def pretrain(
        self,
        dataset,
        train_indices: list[int],
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> dict:
        """Pre-train with SupCon loss and return the encoder state dict.

        Args:
            dataset: A PyTorch Dataset whose ``__getitem__`` returns
                ``(features, label, metadata)`` tuples — the standard
                offshore_dl dataset contract.
            train_indices: Indices into ``dataset`` for pre-training.
            epochs: Number of pre-training epochs.
            batch_size: DataLoader batch size.
            lr: AdamW learning rate.

        Returns:
            ``model.state_dict()`` of the base encoder (without the
            projection head).
        """
        # Build base model + projection head
        model = self.model_class(**self.model_kwargs).to(self.device)
        model.train()

        # Infer output dim from a dummy forward pass
        n_vars = getattr(model, "n_vars", None)
        if n_vars is None:
            raise AttributeError(
                "model must expose .n_vars (number of input variables)"
            )

        # Determine logit/feature dimension from a single sample
        sample_features, _, _ = dataset[train_indices[0]]
        dummy = sample_features.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            dummy_out = model(dummy)
        in_dim = dummy_out.shape[-1]

        proj_head = _ProjectionHead(in_dim, self.projection_dim).to(self.device)

        criterion = SupConLoss(temperature=self.temperature)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(proj_head.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )

        subset = Subset(dataset, train_indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # avoids singleton batches at end
            num_workers=0,
        )

        logger.info(
            "SupConPreTrainer: starting pre-training — epochs=%d, "
            "batch_size=%d, lr=%.2e, projection_dim=%d, temperature=%.4f",
            epochs,
            batch_size,
            lr,
            self.projection_dim,
            self.temperature,
        )

        for epoch in range(epochs):
            epoch_losses: list[float] = []

            for batch in loader:
                features, targets, _metadata = batch
                features = features.to(self.device, dtype=torch.float32)
                if isinstance(targets, torch.Tensor):
                    labels = targets.long().to(self.device)
                else:
                    labels = torch.tensor(targets, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = model(features)              # (B, n_classes)
                embeddings = proj_head(logits)        # (B, projection_dim), L2-normed
                loss = criterion(embeddings, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        "Epoch %d: NaN/Inf loss encountered — skipping batch",
                        epoch + 1,
                    )
                    optimizer.zero_grad()
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(proj_head.parameters()), 1.0
                )
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = (
                sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
            )
            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                logger.info(
                    "SupCon pre-train epoch %d/%d — loss=%.6f",
                    epoch + 1,
                    epochs,
                    avg_loss,
                )

        # Return only the base encoder weights (no projection head)
        encoder_state = model.state_dict()
        logger.info(
            "SupConPreTrainer: pre-training complete — returning encoder "
            "state dict (%d parameter tensors)",
            len(encoder_state),
        )
        return encoder_state
