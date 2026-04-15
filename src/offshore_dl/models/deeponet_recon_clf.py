"""Per-class reconstruction DeepONet classifier (Architecture B).

Trains class-specific reconstruction operators using a shared S-DeepONet
backbone with LoRA adapters. Classification by minimum reconstruction error.

Architecture:
  - LSTMBranch (S-DeepONet): processes raw sensor windows temporally
  - Fourier-encoded trunk: maps (timestep, sensor) positions to basis vectors
  - Shared pretraining on all classes (reconstruction loss)
  - Per-class LoRA fine-tuning (low-rank adapters, rank=4)
  - Classify by argmin(reconstruction_error) across class operators

References:
  - Lu et al. (2021) DeepONet
  - He et al. (2024) S-DeepONet (LSTM branch)
  - Zhang et al. (2026) D2NO + LoRA
  - Choi et al. (2026) Fourier trunk features
  - Heinlein & Taraz (2026) Branch dominates error
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    """Fourier feature encoding for trunk input positions.

    Maps 2D positions (t, sensor) to high-dimensional sinusoidal features:
    [t, s, sin(2pi*f1*t), cos(2pi*f1*t), sin(2pi*f1*s), cos(2pi*f1*s), ...]

    Reference: Choi, Liu & Macedo (2026), arXiv:2507.10368.
    """

    def __init__(self, n_frequencies: int = 16):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(n_frequencies) * 2.0)
        self.output_dim = 2 + 4 * n_frequencies  # raw (t,s) + sin/cos for each freq x 2 dims

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode positions (N, 2) -> (N, output_dim)."""
        t = positions[:, 0:1]  # (N, 1)
        s = positions[:, 1:2]  # (N, 1)
        freqs = self.frequencies.unsqueeze(0) * 2 * math.pi  # (1, F)
        # Encode each dimension with sin/cos at each frequency
        t_enc = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)  # (N, 2F)
        s_enc = torch.cat([torch.sin(s * freqs), torch.cos(s * freqs)], dim=-1)  # (N, 2F)
        return torch.cat([positions, t_enc, s_enc], dim=-1)  # (N, 2+4F)


class LSTMBranch(nn.Module):
    """Bidirectional LSTM branch for temporal sensor data (S-DeepONet).

    Processes (B, W, n_vars) sequentially, capturing temporal dependencies.
    Uses concatenated forward+backward final hidden states.

    Reference: He et al. (2024), S-DeepONet, Eng. Applications of AI.
    """

    def __init__(
        self,
        n_vars: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.proj = nn.Linear(hidden_size * 2, rank)
        self.norm = nn.LayerNorm(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers*2, B, hidden)
        h_fwd = h_n[-2]  # last forward
        h_bwd = h_n[-1]  # last backward
        h = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden*2)
        return self.norm(self.proj(h))  # (B, rank)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Adds a trainable low-rank decomposition to a frozen base linear layer.
    output = base(x) + (x @ A @ B) * scale

    Reference: Hu et al. (2021) LoRA; Zhang et al. (2026) D2NO.
    """

    def __init__(self, base_layer: nn.Linear, lora_rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base_layer = base_layer
        # Freeze base
        for p in self.base_layer.parameters():
            p.requires_grad = False
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features
        self.lora_A = nn.Parameter(torch.randn(in_dim, lora_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(lora_rank, out_dim))
        self.scale = alpha / lora_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out


class DeepONetReconClassifier:
    """Per-class reconstruction DeepONet classifier.

    Trains class-specific reconstruction operators using a shared S-DeepONet
    backbone with LoRA adapters. Classification by minimum reconstruction error.

    Architecture:
      - LSTMBranch (S-DeepONet): processes raw sensor windows temporally
      - Fourier-encoded trunk: maps (timestep, sensor) positions to basis vectors
      - Shared pretraining on all classes (reconstruction loss)
      - Per-class LoRA fine-tuning (low-rank adapters, rank=4)
      - Classify by argmin(reconstruction_error) across 10 class operators

    Not a BaseModel subclass -- standalone pipeline like TiRexClassifier.

    References:
      - Lu et al. (2021) DeepONet
      - He et al. (2024) S-DeepONet (LSTM branch)
      - Zhang et al. (2026) D2NO + LoRA
      - Choi et al. (2026) Fourier trunk features
      - Heinlein & Taraz (2026) Branch dominates error

    Args:
        n_classes: Number of fault classes.
        n_vars: Number of sensor variables.
        window_size: Raw window length.
        rank: DeepONet embedding dimension.
        branch_hidden: LSTM hidden size.
        branch_layers: Number of LSTM layers.
        trunk_hidden: Trunk MLP hidden dimensions.
        n_frequencies: Number of Fourier frequencies for trunk encoding.
        lora_rank: LoRA adapter rank per class.
        pretrain_epochs: Epochs for shared pretraining.
        finetune_epochs: Epochs for per-class LoRA fine-tuning.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_classes=10,
        n_vars=27,
        window_size=720,
        rank=64,
        branch_hidden=128,
        branch_layers=2,
        trunk_hidden=None,
        n_frequencies=16,
        lora_rank=4,
        pretrain_epochs=30,
        finetune_epochs=10,
        dropout=0.1,
    ):
        self.n_classes = n_classes
        self.n_vars = n_vars
        self.window_size = window_size
        self.rank = rank
        self.branch_hidden = branch_hidden
        self.branch_layers = branch_layers
        self.trunk_hidden = trunk_hidden or [128, 128]
        self.n_frequencies = n_frequencies
        self.lora_rank = lora_rank
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.dropout = dropout

    def _build_base_model(self):
        """Build the shared reconstruction DeepONet."""
        branch = LSTMBranch(
            n_vars=self.n_vars,
            hidden_size=self.branch_hidden,
            num_layers=self.branch_layers,
            rank=self.rank,
            dropout=self.dropout,
        )

        pos_enc = FourierPositionalEncoding(n_frequencies=self.n_frequencies)

        trunk_layers = []
        prev_dim = pos_enc.output_dim
        for h_dim in self.trunk_hidden:
            trunk_layers.extend([nn.Linear(prev_dim, h_dim), nn.GELU(), nn.Dropout(self.dropout)])
            prev_dim = h_dim
        trunk_layers.append(nn.Linear(prev_dim, self.rank))
        trunk = nn.Sequential(*trunk_layers)

        output_bias = nn.Parameter(torch.zeros(self.window_size * self.n_vars))

        return branch, pos_enc, trunk, output_bias

    def _get_positions(self, device):
        """Generate normalized (timestep, sensor) query positions."""
        t = torch.linspace(0, 1, self.window_size, device=device)
        s = torch.linspace(0, 1, self.n_vars, device=device)
        grid_t, grid_s = torch.meshgrid(t, s, indexing="ij")
        return torch.stack([grid_t.flatten(), grid_s.flatten()], dim=-1)

    def _forward(self, x, branch, pos_enc, trunk, output_bias):
        """Single forward pass: reconstruct input window."""
        branch_emb = branch(x)  # (B, rank)
        positions = self._get_positions(x.device)  # (W*V, 2)
        encoded_pos = pos_enc(positions)  # (W*V, 2+4F)
        trunk_emb = trunk(encoded_pos)  # (W*V, rank)
        output = torch.matmul(branch_emb, trunk_emb.T) + output_bias  # (B, W*V)
        return output.view(-1, self.window_size, self.n_vars)

    def _add_lora(self, trunk):
        """Add LoRA adapters to trunk MLP linear layers."""
        new_trunk = nn.Sequential()
        for i, layer in enumerate(trunk):
            if isinstance(layer, nn.Linear):
                new_trunk.add_module(str(i), LoRALinear(layer, lora_rank=self.lora_rank))
            else:
                new_trunk.add_module(str(i), layer)
        return new_trunk

    def run(
        self,
        dataset,
        train_indices,
        val_indices,
        device="cpu",
        pretrain_lr=1e-3,
        finetune_lr=5e-4,
        batch_size=32,
    ):
        """Full training and evaluation pipeline.

        Args:
            dataset: ThreeWDataset (raw windows, NOT feature-extracted).
            train_indices: Training sample indices.
            val_indices: Validation/test sample indices.
            device: Device string.
            pretrain_lr: Learning rate for shared pretraining.
            finetune_lr: Learning rate for LoRA fine-tuning.
            batch_size: Batch size.

        Returns:
            Dict with predictions, targets, metrics, per_class_errors.
        """
        import copy
        import logging

        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Subset

        from offshore_dl.evaluation.metrics import MetricRegistry
        from offshore_dl.utils.reproducibility import set_global_seed

        logger = logging.getLogger(__name__)
        set_global_seed(42)

        # -- Phase 1: Shared pretraining (reconstruction on ALL classes) --
        logger.info(
            "Phase 1: Pretraining shared reconstruction operator (%d epochs)",
            self.pretrain_epochs,
        )

        branch, pos_enc, trunk, output_bias = self._build_base_model()
        branch = branch.to(device)
        pos_enc = pos_enc.to(device)
        trunk = trunk.to(device)
        output_bias = output_bias.to(device)

        all_params = (
            list(branch.parameters())
            + list(pos_enc.parameters())
            + list(trunk.parameters())
            + [output_bias]
        )
        optimizer = torch.optim.AdamW(all_params, lr=pretrain_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.pretrain_epochs)
        criterion = nn.MSELoss()

        train_subset = Subset(dataset, train_indices)
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=0,
        )

        for epoch in range(self.pretrain_epochs):
            branch.train()
            trunk.train()
            pos_enc.train()
            epoch_loss = 0.0
            n_batches = 0
            for features, _target, _meta in train_loader:
                features = features.to(device)
                optimizer.zero_grad()
                recon = self._forward(features, branch, pos_enc, trunk, output_bias)
                loss = criterion(recon, features)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(
                    "  Pretrain epoch %d/%d: loss=%.6f",
                    epoch + 1, self.pretrain_epochs, avg_loss,
                )

        # Save shared state
        shared_branch_state = copy.deepcopy(branch.state_dict())
        shared_pos_enc_state = copy.deepcopy(pos_enc.state_dict())
        shared_trunk_state = copy.deepcopy(trunk.state_dict())
        shared_bias = output_bias.data.clone()

        # -- Phase 2: Per-class LoRA fine-tuning --
        logger.info(
            "Phase 2: Per-class LoRA fine-tuning (%d classes, %d epochs each)",
            self.n_classes, self.finetune_epochs,
        )

        # Collect per-class train indices
        class_indices = {}
        for idx in train_indices:
            _, label, _ = dataset[idx]
            label = int(label) if not isinstance(label, int) else label
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        class_models = {}
        for k in range(self.n_classes):
            if k not in class_indices or len(class_indices[k]) < 2:
                logger.warning(
                    "  Class %d: insufficient samples (%d), skipping LoRA",
                    k, len(class_indices.get(k, [])),
                )
                # Use shared model as fallback
                class_models[k] = (
                    copy.deepcopy(branch),
                    copy.deepcopy(pos_enc),
                    copy.deepcopy(trunk),
                    output_bias.clone(),
                )
                continue

            logger.info("  Class %d: LoRA fine-tuning on %d samples", k, len(class_indices[k]))
            set_global_seed(42 + k)

            # Restore shared weights
            k_branch = copy.deepcopy(branch)
            k_branch.load_state_dict(shared_branch_state)
            k_pos_enc = copy.deepcopy(pos_enc)
            k_pos_enc.load_state_dict(shared_pos_enc_state)
            k_trunk_base = copy.deepcopy(trunk)
            k_trunk_base.load_state_dict(shared_trunk_state)
            k_bias = shared_bias.clone().to(device)
            k_bias = nn.Parameter(k_bias)

            # Freeze branch and pos_enc, add LoRA to trunk
            for p in k_branch.parameters():
                p.requires_grad = False
            for p in k_pos_enc.parameters():
                p.requires_grad = False

            k_trunk = self._add_lora(k_trunk_base).to(device)
            k_branch = k_branch.to(device)
            k_pos_enc = k_pos_enc.to(device)

            # Only train LoRA params + bias
            lora_params = [p for p in k_trunk.parameters() if p.requires_grad] + [k_bias]
            k_optimizer = torch.optim.AdamW(lora_params, lr=finetune_lr, weight_decay=1e-4)

            k_subset = Subset(dataset, class_indices[k])
            k_loader = DataLoader(
                k_subset,
                batch_size=min(batch_size, len(class_indices[k])),
                shuffle=True,
                num_workers=0,
            )

            for epoch in range(self.finetune_epochs):
                k_branch.eval()
                k_trunk.train()
                k_pos_enc.eval()
                for features, _target, _meta in k_loader:
                    features = features.to(device)
                    k_optimizer.zero_grad()
                    recon = self._forward(features, k_branch, k_pos_enc, k_trunk, k_bias)
                    loss = criterion(recon, features)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    loss.backward()
                    k_optimizer.step()

            class_models[k] = (k_branch, k_pos_enc, k_trunk, k_bias)

        # -- Phase 3: Inference -- classify by minimum reconstruction error --
        logger.info(
            "Phase 3: Classifying %d samples by minimum reconstruction error",
            len(val_indices),
        )

        val_subset = Subset(dataset, val_indices)
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=0,
        )

        all_predictions = []
        all_targets = []
        all_errors = []  # (N, K) per-class errors

        with torch.no_grad():
            for features, targets, _meta in val_loader:
                features = features.to(device)
                batch_size_actual = features.shape[0]
                batch_errors = torch.zeros(batch_size_actual, self.n_classes, device=device)

                for k in range(self.n_classes):
                    k_branch, k_pos_enc, k_trunk, k_bias = class_models[k]
                    k_branch.eval()
                    k_trunk.eval()
                    k_pos_enc.eval()

                    recon = self._forward(features, k_branch, k_pos_enc, k_trunk, k_bias)
                    # Per-sample MSE
                    error = ((recon - features) ** 2).mean(dim=(1, 2))  # (B,)
                    batch_errors[:, k] = error

                preds = batch_errors.argmin(dim=1)  # (B,)
                all_predictions.append(preds.cpu())
                all_targets.append(targets)
                all_errors.append(batch_errors.cpu())

        predictions = torch.cat(all_predictions).numpy()
        targets_np = torch.cat(all_targets).numpy()
        errors_np = torch.cat(all_errors).numpy()

        metrics = MetricRegistry.compute("classification", predictions, targets_np)

        logger.info(
            "Results: F1-macro=%.4f, Accuracy=%.4f",
            metrics.get("f1_macro", 0),
            metrics.get("accuracy", 0),
        )

        return {
            "predictions": predictions,
            "targets": targets_np,
            "per_class_errors": errors_np,
            "metrics": metrics,
            "n_classes": self.n_classes,
            "pretrain_epochs": self.pretrain_epochs,
            "finetune_epochs": self.finetune_epochs,
            "lora_rank": self.lora_rank,
            "rank": self.rank,
        }
