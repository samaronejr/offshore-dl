"""Experiment orchestration: CV folds × Trainer × MLflow × Optuna.

``ExperimentRunner.run()`` is the single entry point downstream model
slices call. Wires model + dataset + CV + metrics + Trainer + MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from offshore_dl.evaluation.cv import FoldNormalizer
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.models.base import model_summary
from offshore_dl.training.trainer import Trainer

logger = logging.getLogger(__name__)


class NormalizedSubset(Dataset):
    """Subset with per-feature z-score normalization.

    Computes mean/std from training indices, then normalizes both
    train and val subsets with the same statistics.  Optionally applies
    training-time augmentation (jitter + magnitude scaling).

    When ``target_mean`` and ``target_std`` are provided, targets are
    also z-score normalized.  Use ``denormalize_targets()`` to invert
    before computing metrics.
    """

    def __init__(
        self,
        dataset,
        indices: list[int],
        mean: torch.Tensor,
        std: torch.Tensor,
        augment: bool = False,
        target_mean: float | None = None,
        target_std: float | None = None,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.mean = mean
        self.std = std
        self.augment = augment
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        features, target, metadata = self.dataset[self.indices[idx]]
        # z-score normalize features: (x - mean) / std
        # For near-constant features (std was ~0), mean and std are set to
        # 0 and 1 respectively, so the raw value passes through unchanged.
        features = (features - self.mean) / self.std

        # z-score normalize targets (forecasting)
        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std

        # For anomaly task, target is the same window as input —
        # normalize with the same per-sensor stats so the reconstruction
        # error is scale-invariant.
        if (
            isinstance(target, torch.Tensor)
            and target.shape == features.shape
            and self.target_mean is None
        ):
            target = (target - self.mean) / self.std

        # Training-time augmentation
        if self.augment:
            features = self._apply_augmentation(features)

        return features, target, metadata

    @staticmethod
    def _apply_augmentation(x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic augmentations to a single sample.

        Three augmentations, each applied with 50% probability:
        - Gaussian jitter: ``x += N(0, 0.05)``
        - Magnitude scaling: ``x *= uniform(0.8, 1.2)``
        - Channel dropout: zero out 1-2 random channels entirely

        Args:
            x: Feature tensor ``(window, n_vars)``.

        Returns:
            Augmented tensor (same shape).
        """
        # Gaussian jitter
        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * 0.05

        # Magnitude scaling (per-channel)
        if torch.rand(1).item() < 0.5:
            n_vars = x.shape[1]
            scale = 0.8 + 0.4 * torch.rand(1, n_vars)
            x = x * scale

        # Channel dropout (zero 1-2 channels)
        if torch.rand(1).item() < 0.3:
            n_vars = x.shape[1]
            n_drop = min(2, max(1, int(torch.randint(1, 3, (1,)).item())))
            drop_idx = torch.randperm(n_vars)[:n_drop]
            x[:, drop_idx] = 0.0

        return x

    @staticmethod
    def compute_target_stats(
        dataset, indices: list[int], max_samples: int = 10000,
    ) -> tuple[float, float]:
        """Compute target mean/std from training samples (for forecasting).

        Excludes all-zero targets (shutdown windows) from statistics to
        prevent diluting the scale with non-productive periods.

        Returns:
            Tuple of (mean, std) as floats.
        """
        rng = np.random.RandomState(42)
        if len(indices) > max_samples:
            sample_idx = rng.choice(len(indices), max_samples, replace=False)
        else:
            sample_idx = np.arange(len(indices))

        targets = []
        for i in sample_idx:
            _, target, _ = dataset[indices[i]]
            targets.append(target)

        stacked = torch.stack(targets)  # [N, horizon]
        flat = stacked.ravel()

        # Exclude zeros from stats (shutdown periods)
        nonzero = flat[flat.abs() > 1e-8]
        if len(nonzero) < 10:
            # Almost all zeros — no normalization
            return 0.0, 1.0

        mean = float(nonzero.mean())
        std = float(nonzero.std())
        if std < 1e-8:
            std = 1.0
        return mean, std

    @staticmethod
    def denormalize_targets(
        predictions: np.ndarray,
        target_mean: float,
        target_std: float,
    ) -> np.ndarray:
        """Invert target z-score normalization."""
        return predictions * target_std + target_mean

    @staticmethod
    def compute_stats(dataset, indices: list[int], max_samples: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalization statistics from a subset of training samples.

        For raw windows ``(W, n_vars)`` where W is large (e.g. 720): computes
        per-sensor mean/std collapsed across both samples and timesteps →
        returns shape ``(n_vars,)``.

        For feature matrices ``(F, n_vars)`` where F is small (e.g. 14):
        computes per-feature-per-sensor mean/std collapsed across samples
        only → returns shape ``(F, n_vars)``.  This prevents averaging
        different-scale features (mean, std, energy, …) together.

        The threshold ``F ≤ 30`` distinguishes feature matrices from raw
        windows.
        """
        # Sample a subset for efficiency on large datasets
        rng = np.random.RandomState(42)
        if len(indices) > max_samples:
            sample_idx = rng.choice(len(indices), max_samples, replace=False)
        else:
            sample_idx = np.arange(len(indices))

        # Collect features
        all_features = []
        for i in sample_idx:
            feat, _, _ = dataset[indices[i]]
            all_features.append(feat)

        stacked = torch.stack(all_features)  # [N, window, n_vars]
        _, seq_len, _ = stacked.shape

        if seq_len <= 30:
            # Feature matrix: normalize per (feature_row, sensor) across samples
            mean = stacked.mean(dim=0)  # [seq_len, n_vars]
            std = stacked.std(dim=0)    # [seq_len, n_vars]
        else:
            # Raw window: normalize per sensor across samples + timesteps
            mean = stacked.mean(dim=(0, 1))  # [n_vars]
            std = stacked.std(dim=(0, 1))    # [n_vars]

        # For near-constant features (std ≈ 0), disable normalization by
        # setting mean=0, std=1 so raw values pass through unchanged.
        # This prevents dividing sparse non-zero values by tiny std.
        near_constant = std < 1e-3
        mean[near_constant] = 0.0
        std[near_constant] = 1.0
        return mean, std


def _setup_mlflow(cfg: DictConfig) -> bool:
    """Configure MLflow tracking. Returns True if available."""
    try:
        import mlflow
        import os

        # Prefer MLFLOW_TRACKING_URI env var (set by Docker Compose)
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            tracking_uri = cfg.mlflow.tracking_uri if hasattr(cfg, "mlflow") else "mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)
        return True
    except ImportError:
        logger.warning("mlflow not installed — runs will not be tracked")
        return False


class ExperimentRunner:
    """Orchestrates model training across CV folds with tracking.

    Usage::

        runner = ExperimentRunner(
            model_class=LSTMModel,
            dataset=my_dataset,
            cv_strategy=my_cv,
            cfg=cfg,
        )
        results = runner.run()

    Args:
        model_class: BaseModel subclass (not instance).
        dataset: Dataset instance.
        cv_strategy: BaseCVStrategy instance.
        cfg: Merged OmegaConf config.
        model_kwargs: Extra kwargs passed to model_class constructor.
    """

    def __init__(
        self,
        model_class: type,
        dataset: Any,
        cv_strategy: Any,
        cfg: DictConfig,
        model_kwargs: dict | None = None,
    ) -> None:
        self.model_class = model_class
        self.dataset = dataset
        self.cv_strategy = cv_strategy
        self.cfg = cfg
        self.model_kwargs = model_kwargs or {}

    def run(
        self,
        experiment_name: str | None = None,
        use_mlflow: bool = True,
    ) -> dict:
        """Run the full experiment across CV folds.

        Args:
            experiment_name: MLflow experiment name override.
            use_mlflow: Enable/disable MLflow logging.

        Returns:
            Dict with 'fold_results' (list of per-fold dicts),
            'aggregate' (mean±std metrics), and 'cost' (aggregated costs).
        """
        mlflow_available = False
        if use_mlflow:
            mlflow_available = _setup_mlflow(self.cfg)

        splits = self.cv_strategy.get_splits(len(self.dataset))

        if len(splits) == 0:
            logger.error("CV strategy produced 0 splits — check fold/data compatibility")
            return {
                "fold_results": [],
                "aggregate": {},
                "cost": {},
                "experiment_name": experiment_name or "unknown",
                "n_folds": 0,
            }

        task = self.model_kwargs.get("task", "classification")

        # Determine experiment name
        if experiment_name is None:
            prefix = self.cfg.mlflow.experiment_prefix if hasattr(self.cfg, "mlflow") else "offshore-dl"
            model_name = self.model_class.__name__
            ds_name = type(self.dataset).__name__
            experiment_name = f"{prefix}/{model_name}/{ds_name}"

        fold_results = []
        all_costs = []

        parent_run = None
        mlflow = None
        if mlflow_available:
            import mlflow as _mlflow
            mlflow = _mlflow
            mlflow.set_experiment(experiment_name)
            parent_run = mlflow.start_run(run_name="experiment")

        try:
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                logger.info("═══ Fold %d/%d ═══", fold_idx + 1, len(splits))

                fold_result = self._run_fold(
                    fold_idx, train_idx, val_idx, task,
                    mlflow=mlflow, parent_run=parent_run,
                )
                fold_results.append(fold_result)
                if "cost" in fold_result:
                    all_costs.append(fold_result["cost"])

        finally:
            if parent_run and mlflow:
                # Log aggregate metrics to parent
                agg = self._aggregate_folds(fold_results)
                if mlflow:
                    for key, val in agg.items():
                        try:
                            mlflow.log_metric(key, val)
                        except Exception:
                            pass
                mlflow.end_run()

        aggregate = self._aggregate_folds(fold_results)

        return {
            "fold_results": fold_results,
            "aggregate": aggregate,
            "cost": self._aggregate_costs(all_costs),
            "experiment_name": experiment_name,
            "n_folds": len(splits),
        }

    def run_nested(
        self,
        train_pool: np.ndarray,
        test_indices: np.ndarray,
        experiment_name: str | None = None,
        use_mlflow: bool = False,
    ) -> dict:
        """Nested evaluation: inner CV on training pool → retrain → test.

        Protocol:
          1. Run inner K-fold CV within ``train_pool`` (for variance
             estimates and model-selection diagnostics).
          2. Retrain a fresh model on the **entire** ``train_pool``.
          3. Evaluate the retrained model on the held-out ``test_indices``.

        The **test metrics** are the primary reported numbers. Inner CV
        metrics are retained for diagnostics (fold variance, convergence).

        Args:
            train_pool: Indices into the dataset for training + inner CV.
            test_indices: Held-out test indices (never used during training).
            experiment_name: Optional name for tracking.
            use_mlflow: Enable/disable MLflow logging.

        Returns:
            Dict with 'test_metrics', 'cv_aggregate' (inner CV mean±std),
            'cv_fold_results', 'retrain_history', and metadata.
        """
        task = self.model_kwargs.get("task", "classification")

        # ── MLflow setup ──
        mlflow = None
        if use_mlflow:
            mlflow_available = _setup_mlflow(self.cfg)
            if mlflow_available:
                import mlflow as _mlflow
                mlflow = _mlflow
                exp_name = experiment_name or f"offshore-dl-{task}-nested"
                mlflow.set_experiment(exp_name)
                logger.info("MLflow tracking enabled: experiment=%s", exp_name)

        # ── Step 1: Inner CV on training pool ──
        # Remap global indices to local [0, len(train_pool)) for the CV
        # strategy, then map back to global for _run_fold.
        from torch.utils.data import Subset

        train_pool_list = train_pool.tolist()
        pool_dataset = Subset(self.dataset, train_pool_list)

        inner_splits = self.cv_strategy.get_splits(len(pool_dataset))
        logger.info(
            "═══ Nested CV: %d inner folds on %d training samples, "
            "%d held-out test samples ═══",
            len(inner_splits), len(train_pool), len(test_indices),
        )

        cv_fold_results = []
        parent_run_ctx = None
        parent_run = None

        if mlflow is not None:
            parent_run_ctx = mlflow.start_run(run_name=f"{task}-nested")
            parent_run_ctx.__enter__()
            parent_run = mlflow.active_run()
            mlflow.log_params({
                f"model_{k}": v for k, v in self.model_kwargs.items()
                if isinstance(v, (str, int, float, bool))
            })
            mlflow.log_param("n_train", len(train_pool))
            mlflow.log_param("n_test", len(test_indices))
            mlflow.log_param("n_cv_folds", len(inner_splits))

        for fold_idx, (local_train, local_val) in enumerate(inner_splits):
            logger.info("── Inner fold %d/%d ──", fold_idx + 1, len(inner_splits))
            # Map local indices back to global dataset indices
            global_train = train_pool[local_train]
            global_val = train_pool[local_val]
            fold_result = self._run_fold(
                fold_idx, global_train, global_val, task,
                mlflow=mlflow, parent_run=parent_run,
            )
            cv_fold_results.append(fold_result)

        cv_aggregate = self._aggregate_folds(cv_fold_results)
        logger.info("Inner CV aggregate: %s", {
            k: f"{v:.4f}" for k, v in cv_aggregate.items()
            if k.endswith("_mean")
        })

        # ── Step 2: Retrain on full training pool ──
        logger.info("═══ Retraining on full training pool (%d samples) ═══",
                     len(train_pool))

        mean, std = NormalizedSubset.compute_stats(
            self.dataset, train_pool_list,
        )
        use_augment = task == "classification" and mean.dim() == 1

        target_mean, target_std = None, None
        if task == "forecasting":
            target_mean, target_std = NormalizedSubset.compute_target_stats(
                self.dataset, train_pool_list,
            )

        train_subset = NormalizedSubset(
            self.dataset, train_pool_list, mean, std,
            augment=use_augment,
            target_mean=target_mean, target_std=target_std,
        )

        batch_size = (
            self.cfg.training.batch_size
            if hasattr(self.cfg, "training") else 32
        )
        train_loader = DataLoader(
            train_subset, batch_size=batch_size,
            shuffle=True, num_workers=0, pin_memory=True,
        )

        # Use a small validation split from the training pool for early
        # stopping during retraining (last 10% of training pool).
        n_retrain_val = max(1, int(len(train_pool) * 0.1))
        retrain_train_idx = train_pool_list[:-n_retrain_val]
        retrain_val_idx = train_pool_list[-n_retrain_val:]

        retrain_val_subset = NormalizedSubset(
            self.dataset, retrain_val_idx, mean, std,
            target_mean=target_mean, target_std=target_std,
        )
        retrain_val_loader = DataLoader(
            retrain_val_subset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )

        model = self.model_class(**self.model_kwargs)
        if task == "classification":
            n_classes = self.model_kwargs.get("n_classes", None)
            class_weights = self._compute_class_weights(
                self.dataset, train_pool_list, n_classes=n_classes,
            )
            if class_weights is not None:
                model.set_class_weights(class_weights)

        device = getattr(self.cfg, "device", "cpu")
        trainer = Trainer(cfg=self.cfg, device=device)
        retrain_history = trainer.fit(
            model, train_loader, retrain_val_loader,
            max_epochs=(
                self.cfg.training.max_epochs
                if hasattr(self.cfg, "training") else 5
            ),
        )

        # Log retrain per-epoch curves to MLflow
        if mlflow is not None:
            for epoch, (tl, vl) in enumerate(zip(
                retrain_history["train_loss"], retrain_history["val_loss"]
            )):
                mlflow.log_metrics({
                    "retrain_train_loss": tl,
                    "retrain_val_loss": vl,
                }, step=epoch)

        # ── Step 3: Evaluate on held-out test set ──
        logger.info("═══ Evaluating on held-out test set (%d samples) ═══",
                     len(test_indices))

        test_subset = NormalizedSubset(
            self.dataset, test_indices.tolist(), mean, std,
            target_mean=target_mean, target_std=target_std,
        )
        test_loader = DataLoader(
            test_subset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )

        model.eval()
        all_preds, all_targets = [], []
        for batch in test_loader:
            batch = tuple(
                t.to(trainer.device) if isinstance(t, torch.Tensor) else t
                for t in batch
            )
            preds = model.predict(batch)
            all_preds.append(preds.cpu())
            _, targets, _ = batch
            if isinstance(targets, torch.Tensor):
                all_targets.append(targets.cpu())
            else:
                all_targets.append(torch.tensor(targets))

        predictions = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()

        if target_mean is not None and target_std is not None:
            predictions = NormalizedSubset.denormalize_targets(
                predictions, target_mean, target_std,
            )
            targets = NormalizedSubset.denormalize_targets(
                targets, target_mean, target_std,
            )

        test_metrics = MetricRegistry.compute(task, predictions, targets)

        logger.info("Test metrics: %s", {
            k: f"{v:.4f}" if isinstance(v, float) else v
            for k, v in test_metrics.items()
        })

        # ── Log final test metrics + close parent run ──
        if mlflow is not None:
            for k, v in cv_aggregate.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"cv_{k}", v)
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"test_{k}", v)
            logger.info("MLflow run logged successfully")
            if parent_run_ctx is not None:
                parent_run_ctx.__exit__(None, None, None)

        return {
            "test_metrics": test_metrics,
            "cv_aggregate": cv_aggregate,
            "cv_fold_results": cv_fold_results,
            "retrain_history": {
                "epochs_run": retrain_history["epochs_run"],
                "best_epoch": retrain_history["best_epoch"],
                "stopped_early": retrain_history["stopped_early"],
                "final_train_loss": (
                    retrain_history["train_loss"][-1]
                    if retrain_history["train_loss"] else None
                ),
                "final_val_loss": (
                    retrain_history["val_loss"][-1]
                    if retrain_history["val_loss"] else None
                ),
            },
            "n_train": len(train_pool),
            "n_test": len(test_indices),
            "n_cv_folds": len(inner_splits),
            "experiment_name": experiment_name or "unknown",
        }

    def _run_fold(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        task: str,
        mlflow: Any = None,
        parent_run: Any = None,
    ) -> dict:
        """Train and evaluate on a single fold."""
        # Compute normalization stats from training data only
        train_indices = train_idx.tolist()
        val_indices = val_idx.tolist()

        # Validate fold: no index overlap between train and val
        overlap = set(train_indices) & set(val_indices)
        if overlap:
            raise ValueError(f"LeakageGuard: {len(overlap)} indices overlap between train and val")

        mean, std = NormalizedSubset.compute_stats(self.dataset, train_indices)
        # Disable augmentation for feature-based datasets (short sequences)
        # — jitter on statistical features would corrupt the signal
        use_augment = task == "classification" and mean.dim() == 1

        # ── Target normalization for forecasting ──
        target_mean, target_std = None, None
        if task == "forecasting":
            target_mean, target_std = NormalizedSubset.compute_target_stats(
                self.dataset, train_indices,
            )
            logger.info(
                "  Target normalization: mean=%.4f, std=%.4f",
                target_mean, target_std,
            )

        train_subset = NormalizedSubset(
            self.dataset, train_indices, mean, std,
            augment=use_augment,
            target_mean=target_mean, target_std=target_std,
        )
        val_subset = NormalizedSubset(
            self.dataset, val_indices, mean, std,
            target_mean=target_mean, target_std=target_std,
        )

        batch_size = self.cfg.training.batch_size if hasattr(self.cfg, "training") else 32
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Create model
        model = self.model_class(**self.model_kwargs)

        # ── Class weights for imbalanced classification ──
        if task == "classification":
            n_classes = self.model_kwargs.get("n_classes", None)
            class_weights = self._compute_class_weights(
                self.dataset, train_indices, n_classes=n_classes,
            )
            if class_weights is not None:
                model.set_class_weights(class_weights)
                logger.info("Applied class weights: %s", [f"{w:.3f}" for w in class_weights.tolist()])

        # Start MLflow child run
        child_run = None
        if mlflow and parent_run:
            child_run = mlflow.start_run(
                run_name=f"fold_{fold_idx}",
                nested=True,
            )
            # Log params
            try:
                flat = OmegaConf.to_container(self.cfg, resolve=True)
                flat_params = _flatten_dict(flat)
                # MLflow has a param length limit — truncate keys
                for k, v in list(flat_params.items())[:100]:
                    try:
                        mlflow.log_param(k[:250], str(v)[:250])
                    except Exception:
                        pass
            except Exception:
                pass
            mlflow.log_param("fold_idx", fold_idx)
            summary = model_summary(model)
            for k, v in summary.items():
                mlflow.log_param(f"model_{k}", v)

        try:
            # Train
            device = getattr(self.cfg, "device", "cpu")
            trainer = Trainer(cfg=self.cfg, device=device)
            history = trainer.fit(
                model, train_loader, val_loader,
                max_epochs=self.cfg.training.max_epochs if hasattr(self.cfg, "training") else 5,
            )

            # Log per-epoch metrics to MLflow
            if mlflow and child_run:
                for epoch, (tl, vl) in enumerate(zip(history["train_loss"], history["val_loss"])):
                    mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)
                mlflow.log_metric("best_epoch", history.get("best_epoch", 0))
                mlflow.log_metric("epochs_run", history.get("epochs_run", 0))
                mlflow.log_metric("stopped_early", int(history.get("stopped_early", False)))

            # Collect predictions on val set
            model.eval()
            all_preds = []
            all_targets = []
            for batch in val_loader:
                batch = tuple(
                    t.to(trainer.device) if isinstance(t, torch.Tensor) else t
                    for t in batch
                )
                preds = model.predict(batch)
                all_preds.append(preds.cpu())
                _, targets, _ = batch
                if isinstance(targets, torch.Tensor):
                    all_targets.append(targets.cpu())
                else:
                    all_targets.append(torch.tensor(targets))

            predictions = torch.cat(all_preds).numpy()
            targets = torch.cat(all_targets).numpy()

            # ── Denormalize targets and predictions (forecasting) ──
            if target_mean is not None and target_std is not None:
                predictions = NormalizedSubset.denormalize_targets(
                    predictions, target_mean, target_std,
                )
                targets = NormalizedSubset.denormalize_targets(
                    targets, target_mean, target_std,
                )

            # Compute metrics
            metrics = MetricRegistry.compute(task, predictions, targets)

            # Log fold metrics to MLflow
            if mlflow and child_run:
                for k, v in metrics.items():
                    try:
                        mlflow.log_metric(f"val_{k}", v)
                    except Exception:
                        pass

            fold_result = {
                "fold_idx": fold_idx,
                "metrics": metrics,
                "history": {
                    "epochs_run": history["epochs_run"],
                    "best_epoch": history["best_epoch"],
                    "stopped_early": history["stopped_early"],
                    "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                    "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
                },
                "cost": history.get("cost", {}),
            }

        finally:
            if mlflow and child_run:
                mlflow.end_run()

        return fold_result

    @staticmethod
    def _aggregate_folds(fold_results: list[dict]) -> dict:
        """Aggregate per-fold metrics into mean±std."""
        if not fold_results:
            return {}

        all_keys = set()
        for fr in fold_results:
            if "metrics" in fr:
                all_keys.update(fr["metrics"].keys())

        agg = {}
        for key in sorted(all_keys):
            values = []
            for fr in fold_results:
                v = fr.get("metrics", {}).get(key)
                if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                    values.append(v)
            if values:
                agg[f"{key}_mean"] = float(np.mean(values))
                agg[f"{key}_std"] = float(np.std(values))

        return agg

    @staticmethod
    def _aggregate_costs(costs: list[dict]) -> dict:
        """Aggregate cost tracker results."""
        if not costs:
            return {}
        agg = {}
        for key in costs[0]:
            values = [c.get(key, 0) for c in costs]
            if isinstance(values[0], (int, float)):
                agg[f"{key}_total"] = sum(values)
                agg[f"{key}_mean"] = float(np.mean(values))
        return agg

    @staticmethod
    def _compute_class_weights(
        dataset: Any, train_indices: list[int], max_samples: int = 50000,
        n_classes: int | None = None,
    ) -> torch.Tensor | None:
        """Compute inverse-frequency class weights from training labels.

        Samples up to ``max_samples`` training examples to count class
        frequencies, then returns ``1 / freq`` weights normalized so they
        sum to ``n_classes``.

        Args:
            dataset: The full dataset.
            train_indices: Indices of training samples.
            max_samples: Max samples to scan for efficiency.
            n_classes: Expected number of classes (uses max label + 1 if None).

        Returns:
            Weight tensor ``(n_classes,)`` or None on error.
        """
        try:
            from collections import Counter

            rng = np.random.RandomState(42)
            indices = train_indices
            if len(indices) > max_samples:
                sample_idx = rng.choice(len(indices), max_samples, replace=False)
                indices = [train_indices[i] for i in sample_idx]

            labels = []
            for idx in indices:
                _, target, _ = dataset[idx]
                if isinstance(target, torch.Tensor):
                    labels.append(int(target.item()))
                else:
                    labels.append(int(target))

            counts = Counter(labels)
            if not counts:
                return None

            if n_classes is None:
                n_classes = max(counts.keys()) + 1
            total = sum(counts.values())

            weights = torch.ones(n_classes, dtype=torch.float32)
            for cls, count in counts.items():
                if cls < n_classes:
                    weights[cls] = total / (n_classes * count)

            logger.info(
                "Class distribution (train): %s",
                {k: counts.get(k, 0) for k in range(n_classes)},
            )
            return weights
        except Exception:
            logger.warning("Could not compute class weights — using uniform", exc_info=True)
            return None


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict for MLflow param logging."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items
