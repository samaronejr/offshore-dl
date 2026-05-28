"""Experiment orchestration: CV folds × Trainer × MLflow × Optuna.

``ExperimentRunner.run()`` is the single entry point downstream model
slices call. Wires model + dataset + CV + metrics + Trainer + MLflow.
"""

from __future__ import annotations

import logging
import numbers
import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

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
        dataset,
        indices: list[int],
        max_samples: int = 10000,
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
    def compute_stats(
        dataset, indices: list[int], max_samples: int = 10000
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            std = stacked.std(dim=0)  # [seq_len, n_vars]
        else:
            # Raw window: normalize per sensor across samples + timesteps
            mean = stacked.mean(dim=(0, 1))  # [n_vars]
            std = stacked.std(dim=(0, 1))  # [n_vars]

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
            tracking_uri = (
                cfg.mlflow.tracking_uri if hasattr(cfg, "mlflow") else "mlruns"
            )
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
        runtime_adjustments: dict | None = None,
    ) -> None:
        self.model_class = model_class
        self.dataset = dataset
        self.cv_strategy = cv_strategy
        self.cfg = cfg
        self.model_kwargs = model_kwargs or {}
        self.runtime_adjustments = runtime_adjustments or {}

    @staticmethod
    def _is_mlflow_metric_value(value: Any) -> bool:
        """MLflow metrics must be finite, real, non-bool scalars."""
        if isinstance(value, (bool, np.bool_)):
            return False
        if not isinstance(value, numbers.Real):
            return False
        return bool(np.isfinite(value))

    @staticmethod
    def _log_mlflow_metric_or_param(mlflow: Any, key: str, value: Any) -> None:
        """Log numeric metrics separately from string/bool provenance params."""
        if ExperimentRunner._is_mlflow_metric_value(value):
            try:
                mlflow.log_metric(key[:250], float(value))
            except Exception as exc:  # pragma: no cover - defensive tracking path
                logger.debug("MLflow metric logging skipped for %s: %s", key, exc)
            return

        if isinstance(value, (str, bool, np.bool_)):
            try:
                mlflow.log_param(key[:250], str(value)[:250])
            except Exception as exc:  # pragma: no cover - defensive tracking path
                logger.debug("MLflow param logging skipped for %s: %s", key, exc)

    @staticmethod
    def _log_mlflow_values(mlflow: Any, prefix: str, values: dict) -> None:
        """Log a metric/provenance dictionary to MLflow with an optional prefix."""
        for key, value in values.items():
            full_key = f"{prefix}_{key}" if prefix else key
            ExperimentRunner._log_mlflow_metric_or_param(mlflow, full_key, value)

    @staticmethod
    def _zero_epoch_history() -> dict:
        """History shape used when a zero-shot model intentionally skips fit."""
        return {
            "train_loss": [],
            "val_loss": [],
            "epochs_run": 0,
            "best_epoch": None,
            "stopped_early": False,
            "cost": {},
        }

    def run(
        self,
        experiment_name: str | None = None,
        use_mlflow: bool = True,
        fold_callback: Any | None = None,
    ) -> dict:
        """Run the full experiment across CV folds.

        Args:
            experiment_name: MLflow experiment name override.
            use_mlflow: Enable/disable MLflow logging.
            fold_callback: Optional callback invoked after each completed fold
                with ``(fold_idx, cumulative_fold_results)``. HPO uses this to
                report intermediate values and prune weak trials.

        Returns:
            Dict with 'fold_results' (list of per-fold dicts),
            'aggregate' (mean±std metrics), and 'cost' (aggregated costs).
        """
        mlflow_available = False
        if use_mlflow:
            mlflow_available = _setup_mlflow(self.cfg)

        splits = self.cv_strategy.get_splits(len(self.dataset))

        if len(splits) == 0:
            logger.error(
                "CV strategy produced 0 splits — check fold/data compatibility"
            )
            return {
                "fold_results": [],
                "aggregate": {},
                "cost": {},
                "experiment_name": experiment_name or "unknown",
                "n_folds": 0,
                "runtime_adjustments": self.runtime_adjustments,
            }

        task = self.model_kwargs.get("task", "classification")

        # Determine experiment name
        if experiment_name is None:
            prefix = (
                self.cfg.mlflow.experiment_prefix
                if hasattr(self.cfg, "mlflow")
                else "offshore-dl"
            )
            model_name = self.model_class.__name__
            ds_name = type(self.dataset).__name__
            experiment_name = f"{prefix}/{model_name}/{ds_name}"

        fold_results = []
        all_costs = []

        parent_run = None
        mlflow = None
        if mlflow_available:
            import mlflow as _mlflow

            try:
                mlflow = _mlflow
                mlflow.set_experiment(experiment_name)
                parent_run = mlflow.start_run(run_name="experiment")
            except Exception as _mlflow_err:
                logger.warning(
                    "MLflow unavailable (%s) — proceeding without tracking", _mlflow_err
                )
                mlflow = None

        try:
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                logger.info("═══ Fold %d/%d ═══", fold_idx + 1, len(splits))

                fold_result = self._run_fold(
                    fold_idx,
                    train_idx,
                    val_idx,
                    task,
                    mlflow=mlflow,
                    parent_run=parent_run,
                )
                fold_results.append(fold_result)
                if "cost" in fold_result:
                    all_costs.append(fold_result["cost"])
                if fold_callback is not None:
                    fold_callback(fold_idx, fold_results)

        finally:
            if parent_run and mlflow:
                # Log aggregate metrics to parent
                agg = self._aggregate_folds(fold_results)
                if mlflow:
                    self._log_mlflow_values(mlflow, "", agg)
                mlflow.end_run()

        aggregate = self._aggregate_folds(fold_results)

        return {
            "fold_results": fold_results,
            "aggregate": aggregate,
            "cost": self._aggregate_costs(all_costs),
            "experiment_name": experiment_name,
            "n_folds": len(splits),
            "runtime_adjustments": self.runtime_adjustments,
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
          2. Fit a fresh model on a retraining subset from ``train_pool``
             while reserving a disjoint randomly-sampled validation subset
             for checkpoint selection (random split avoids temporal bias
             that caused best_epoch=0 on non-stationary data).
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

                try:
                    mlflow = _mlflow
                    exp_name = experiment_name or f"offshore-dl-{task}-nested"
                    mlflow.set_experiment(exp_name)
                    logger.info("MLflow tracking enabled: experiment=%s", exp_name)
                except Exception as _mlflow_err:
                    logger.warning(
                        "MLflow unavailable (%s) — proceeding without tracking",
                        _mlflow_err,
                    )
                    mlflow = None

        # ── Step 1: Inner CV on training pool ──
        # Remap global indices to local [0, len(train_pool)) for the CV
        # strategy, then map back to global for _run_fold.
        train_pool_list = train_pool.tolist()
        pool_dataset = Subset(self.dataset, train_pool_list)

        cv_strategy = self.cv_strategy
        if hasattr(cv_strategy, "subset"):
            cv_strategy = cv_strategy.subset(train_pool)

        inner_splits = cv_strategy.get_splits(len(pool_dataset))
        logger.info(
            "═══ Nested CV: %d inner folds on %d training samples, "
            "%d held-out test samples ═══",
            len(inner_splits),
            len(train_pool),
            len(test_indices),
        )

        cv_fold_results = []
        parent_run_ctx = None
        parent_run = None

        if mlflow is not None:
            parent_run_ctx = mlflow.start_run(run_name=f"{task}-nested")
            parent_run_ctx.__enter__()
            parent_run = mlflow.active_run()
            mlflow.log_params(
                {
                    f"model_{k}": v
                    for k, v in self.model_kwargs.items()
                    if isinstance(v, (str, int, float, bool))
                }
            )
            mlflow.log_param("n_train", len(train_pool))
            mlflow.log_param("n_test", len(test_indices))
            mlflow.log_param("n_cv_folds", len(inner_splits))

        for fold_idx, (local_train, local_val) in enumerate(inner_splits):
            logger.info("── Inner fold %d/%d ──", fold_idx + 1, len(inner_splits))
            # Map local indices back to global dataset indices
            global_train = train_pool[local_train]
            global_val = train_pool[local_val]
            fold_result = self._run_fold(
                fold_idx,
                global_train,
                global_val,
                task,
                mlflow=mlflow,
                parent_run=parent_run,
            )
            cv_fold_results.append(fold_result)

        cv_aggregate = self._aggregate_folds(cv_fold_results)
        logger.info(
            "Inner CV aggregate: %s",
            {k: f"{v:.4f}" for k, v in cv_aggregate.items() if k.endswith("_mean")},
        )

        # ── Step 2: Retrain on full training pool ──
        logger.info(
            "═══ Retraining with disjoint train/val split from training pool (%d samples) ═══",
            len(train_pool),
        )

        retrain_train_idx, retrain_val_idx = self._split_retrain_train_val(
            train_pool_list,
        )
        mean, std = NormalizedSubset.compute_stats(
            self.dataset,
            retrain_train_idx,
        )
        use_augment = task == "classification" and mean.dim() == 1

        target_mean, target_std = None, None
        if task == "forecasting":
            target_mean, target_std = NormalizedSubset.compute_target_stats(
                self.dataset,
                retrain_train_idx,
            )

        train_subset = NormalizedSubset(
            self.dataset,
            retrain_train_idx,
            mean,
            std,
            augment=use_augment,
            target_mean=target_mean,
            target_std=target_std,
        )

        batch_size = (
            self.cfg.training.batch_size if hasattr(self.cfg, "training") else 32
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        retrain_val_subset = NormalizedSubset(
            self.dataset,
            retrain_val_idx,
            mean,
            std,
            target_mean=target_mean,
            target_std=target_std,
        )
        retrain_val_loader = DataLoader(
            retrain_val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        model = self.model_class(**self.model_kwargs)
        if task == "classification":
            n_classes = self.model_kwargs.get("n_classes", None)
            class_weights = self._compute_class_weights(
                self.dataset,
                retrain_train_idx,
                n_classes=n_classes,
            )
            if class_weights is not None:
                model.set_class_weights(class_weights)

        device = getattr(self.cfg, "device", "cpu")
        trainer = Trainer(cfg=self.cfg, device=device)
        model.to(trainer.device)
        if getattr(model, "is_zero_shot", False):
            retrain_history = self._zero_epoch_history()
        else:
            retrain_history = trainer.fit(
                model,
                train_loader,
                retrain_val_loader,
                max_epochs=(
                    self.cfg.training.max_epochs if hasattr(self.cfg, "training") else 5
                ),
            )

        # Log retrain per-epoch curves to MLflow
        if mlflow is not None:
            for epoch, (tl, vl) in enumerate(
                zip(retrain_history["train_loss"], retrain_history["val_loss"])
            ):
                mlflow.log_metrics(
                    {
                        "retrain_train_loss": tl,
                        "retrain_val_loss": vl,
                    },
                    step=epoch,
                )

        # ── Step 3: Evaluate on held-out test set ──
        logger.info(
            "═══ Evaluating on held-out test set (%d samples) ═══", len(test_indices)
        )

        test_subset = NormalizedSubset(
            self.dataset,
            test_indices.tolist(),
            mean,
            std,
            target_mean=target_mean,
            target_std=target_std,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        eval_device = trainer.device
        final_eval_device = os.environ.get("OFFSHORE_DL_FINAL_EVAL_DEVICE")
        if final_eval_device:
            eval_device = torch.device(final_eval_device)
            logger.info("Final held-out evaluation device override: %s", eval_device)
            model.to(eval_device)

        eval_progress_every = int(os.environ.get("OFFSHORE_DL_EVAL_PROGRESS_EVERY", "0") or 0)

        model.eval()
        all_preds, all_targets = [], []
        all_scores = []
        all_instance_ids = []
        all_groups = []
        all_orders = []
        for batch_idx, batch in enumerate(test_loader, start=1):
            batch = tuple(
                t.to(eval_device) if isinstance(t, torch.Tensor) else t
                for t in batch
            )
            features, targets, metadata = batch
            with torch.no_grad():
                outputs = model.forward(features)
            preds = model._extract_predictions(outputs)
            all_preds.append(preds.cpu())
            if task == "classification":
                all_scores.append(model._extract_prediction_scores(outputs).cpu())
                instance_ids = self._extract_instance_ids(metadata)
                if instance_ids is not None:
                    all_instance_ids.append(instance_ids)
            if task == "forecasting":
                group_ids = self._extract_group_ids(metadata)
                if group_ids is not None:
                    all_groups.append(group_ids)
                order_ids = self._extract_order_ids(metadata)
                if order_ids is not None:
                    all_orders.append(order_ids)
            if isinstance(targets, torch.Tensor):
                all_targets.append(targets.cpu())
            else:
                all_targets.append(torch.tensor(targets))
            if eval_progress_every and (
                batch_idx % eval_progress_every == 0
                or batch_idx == len(test_loader)
            ):
                logger.info(
                    "Held-out evaluation progress: %d/%d batches",
                    batch_idx,
                    len(test_loader),
                )

        predictions = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        logger.info("Held-out prediction loop complete: %d samples", len(predictions))
        prediction_scores = torch.cat(all_scores).numpy() if all_scores else None
        instance_ids = (
            np.concatenate(all_instance_ids)
            if all_instance_ids
            and sum(len(x) for x in all_instance_ids) == len(predictions)
            else None
        )
        groups = (
            np.concatenate(all_groups)
            if all_groups
            and sum(len(x) for x in all_groups) == len(predictions)
            else None
        )
        orders = (
            np.concatenate(all_orders)
            if all_orders
            and sum(len(x) for x in all_orders) == len(predictions)
            else None
        )

        if target_mean is not None and target_std is not None:
            predictions = NormalizedSubset.denormalize_targets(
                predictions,
                target_mean,
                target_std,
            )
            targets = NormalizedSubset.denormalize_targets(
                targets,
                target_mean,
                target_std,
            )

        # Collect training targets for MASE denominator deterministically.
        y_train_for_mase = None
        y_train_groups_for_mase = None
        y_train_order_for_mase = None
        if task == "forecasting":
            try:
                logger.info(
                    "Collecting forecasting MASE context from %d training samples",
                    len(train_pool),
                )
                (
                    y_train_for_mase,
                    y_train_groups_for_mase,
                    y_train_order_for_mase,
                ) = self._collect_forecasting_mase_context(
                    train_pool,
                    target_mean=target_mean,
                    target_std=target_std,
                )
            except (IndexError, TypeError, AttributeError) as exc:
                logger.warning(
                    "Could not collect deterministic y_train for MASE denominator: %s; MASE will be marked unavailable if metadata is insufficient.",
                    exc,
                )
            else:
                logger.info("Forecasting MASE context collection complete")

        test_metrics = MetricRegistry.compute(
            task,
            predictions,
            targets,
            prediction_scores=prediction_scores,
            instance_ids=instance_ids,
            y_train=y_train_for_mase,
            groups=groups,
            y_train_groups=y_train_groups_for_mase,
            order=orders,
            y_train_order=y_train_order_for_mase,
        )

        logger.info(
            "Test metrics: %s",
            {
                k: f"{v:.4f}" if isinstance(v, float) else v
                for k, v in test_metrics.items()
            },
        )

        # ── Log final test metrics + close parent run ──
        if mlflow is not None:
            self._log_mlflow_values(mlflow, "cv", cv_aggregate)
            self._log_mlflow_values(mlflow, "test", test_metrics)
            logger.info("MLflow run logged successfully")
            if parent_run_ctx is not None:
                parent_run_ctx.__exit__(None, None, None)

        return {
            "test_metrics": test_metrics,
            "test_indices": test_indices.copy(),
            "test_predictions": predictions,
            "test_probabilities": prediction_scores,
            "test_targets": targets,
            "cv_aggregate": cv_aggregate,
            "cv_fold_results": cv_fold_results,
            "retrain_history": {
                "epochs_run": retrain_history["epochs_run"],
                "best_epoch": retrain_history["best_epoch"],
                "stopped_early": retrain_history["stopped_early"],
                "final_train_loss": (
                    retrain_history["train_loss"][-1]
                    if retrain_history["train_loss"]
                    else None
                ),
                "final_val_loss": (
                    retrain_history["val_loss"][-1]
                    if retrain_history["val_loss"]
                    else None
                ),
            },
            "n_train": len(train_pool),
            "n_test": len(test_indices),
            "n_retrain_train": len(retrain_train_idx),
            "n_retrain_val": len(retrain_val_idx),
            "n_cv_folds": len(inner_splits),
            "experiment_name": experiment_name or "unknown",
            "runtime_adjustments": self.runtime_adjustments,
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
            raise ValueError(
                f"LeakageGuard: {len(overlap)} indices overlap between train and val"
            )

        mean, std = NormalizedSubset.compute_stats(self.dataset, train_indices)
        # Disable augmentation for feature-based datasets (short sequences)
        # — jitter on statistical features would corrupt the signal
        use_augment = task == "classification" and mean.dim() == 1

        # ── Target normalization for forecasting ──
        target_mean, target_std = None, None
        if task == "forecasting":
            target_mean, target_std = NormalizedSubset.compute_target_stats(
                self.dataset,
                train_indices,
            )
            logger.info(
                "  Target normalization: mean=%.4f, std=%.4f",
                target_mean,
                target_std,
            )

        train_subset = NormalizedSubset(
            self.dataset,
            train_indices,
            mean,
            std,
            augment=use_augment,
            target_mean=target_mean,
            target_std=target_std,
        )
        val_subset = NormalizedSubset(
            self.dataset,
            val_indices,
            mean,
            std,
            target_mean=target_mean,
            target_std=target_std,
        )

        batch_size = (
            self.cfg.training.batch_size if hasattr(self.cfg, "training") else 32
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Create model
        model = self.model_class(**self.model_kwargs)

        # ── Class weights for imbalanced classification ──
        if task == "classification":
            n_classes = self.model_kwargs.get("n_classes", None)
            class_weights = self._compute_class_weights(
                self.dataset,
                train_indices,
                n_classes=n_classes,
            )
            if class_weights is not None:
                model.set_class_weights(class_weights)
                logger.info(
                    "Applied class weights: %s",
                    [f"{w:.3f}" for w in class_weights.tolist()],
                )

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
            model.to(trainer.device)
            if getattr(model, "is_zero_shot", False):
                history = self._zero_epoch_history()
            else:
                history = trainer.fit(
                    model,
                    train_loader,
                    val_loader,
                    max_epochs=self.cfg.training.max_epochs
                    if hasattr(self.cfg, "training")
                    else 5,
                )

            # Log per-epoch metrics to MLflow
            if mlflow and child_run:
                for epoch, (tl, vl) in enumerate(
                    zip(history["train_loss"], history["val_loss"])
                ):
                    mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)
                mlflow.log_metric("best_epoch", history.get("best_epoch", 0))
                mlflow.log_metric("epochs_run", history.get("epochs_run", 0))
                mlflow.log_metric(
                    "stopped_early", int(history.get("stopped_early", False))
                )

            # Collect predictions on val set
            model.eval()
            all_preds = []
            all_targets = []
            all_scores = []
            all_instance_ids = []
            all_groups = []
            all_orders = []
            for batch in val_loader:
                batch = tuple(
                    t.to(trainer.device) if isinstance(t, torch.Tensor) else t
                    for t in batch
                )
                features, targets, metadata = batch
                with torch.no_grad():
                    outputs = model.forward(features)
                preds = model._extract_predictions(outputs)
                all_preds.append(preds.cpu())
                if task == "classification":
                    all_scores.append(model._extract_prediction_scores(outputs).cpu())
                    instance_ids = self._extract_instance_ids(metadata)
                    if instance_ids is not None:
                        all_instance_ids.append(instance_ids)
                if task == "forecasting":
                    group_ids = self._extract_group_ids(metadata)
                    if group_ids is not None:
                        all_groups.append(group_ids)
                    order_ids = self._extract_order_ids(metadata)
                    if order_ids is not None:
                        all_orders.append(order_ids)
                if isinstance(targets, torch.Tensor):
                    all_targets.append(targets.cpu())
                else:
                    all_targets.append(torch.tensor(targets))

            predictions = torch.cat(all_preds).numpy()
            targets = torch.cat(all_targets).numpy()
            prediction_scores = torch.cat(all_scores).numpy() if all_scores else None
            instance_ids = (
                np.concatenate(all_instance_ids)
                if all_instance_ids
                and sum(len(x) for x in all_instance_ids) == len(predictions)
                else None
            )
            groups = (
                np.concatenate(all_groups)
                if all_groups
                and sum(len(x) for x in all_groups) == len(predictions)
                else None
            )
            orders = (
                np.concatenate(all_orders)
                if all_orders
                and sum(len(x) for x in all_orders) == len(predictions)
                else None
            )

            # ── Denormalize targets and predictions (forecasting) ──
            if target_mean is not None and target_std is not None:
                predictions = NormalizedSubset.denormalize_targets(
                    predictions,
                    target_mean,
                    target_std,
                )
                targets = NormalizedSubset.denormalize_targets(
                    targets,
                    target_mean,
                    target_std,
                )

            y_train_for_mase = None
            y_train_groups_for_mase = None
            y_train_order_for_mase = None
            if task == "forecasting":
                try:
                    (
                        y_train_for_mase,
                        y_train_groups_for_mase,
                        y_train_order_for_mase,
                    ) = self._collect_forecasting_mase_context(
                        train_idx,
                        target_mean=target_mean,
                        target_std=target_std,
                    )
                except (IndexError, TypeError, AttributeError) as exc:
                    logger.warning(
                        "Could not collect deterministic y_train for MASE: %s; MASE will be marked unavailable if metadata is insufficient.",
                        exc,
                    )

            # Compute metrics
            metrics = MetricRegistry.compute(
                task,
                predictions,
                targets,
                prediction_scores=prediction_scores,
                instance_ids=instance_ids,
                y_train=y_train_for_mase,
                groups=groups,
                y_train_groups=y_train_groups_for_mase,
                order=orders,
                y_train_order=y_train_order_for_mase,
            )

            # Log fold metrics to MLflow
            if mlflow and child_run:
                self._log_mlflow_values(mlflow, "val", metrics)

            fold_result = {
                "fold_idx": fold_idx,
                "metrics": metrics,
                "sample_indices": val_idx.copy(),
                "predictions": predictions,
                "targets": targets,
                "history": {
                    "epochs_run": history["epochs_run"],
                    "best_epoch": history["best_epoch"],
                    "stopped_early": history["stopped_early"],
                    "final_train_loss": history["train_loss"][-1]
                    if history["train_loss"]
                    else None,
                    "final_val_loss": history["val_loss"][-1]
                    if history["val_loss"]
                    else None,
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
            invalid_count = 0
            for fr in fold_results:
                metrics = fr.get("metrics", {})
                if key not in metrics:
                    continue
                v = metrics.get(key)
                if ExperimentRunner._is_mlflow_metric_value(v):
                    values.append(v)
                elif isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)):
                    invalid_count += 1
            if values:
                agg[f"{key}_mean"] = float(np.mean(values))
                agg[f"{key}_std"] = float(np.std(values))
            if values or invalid_count:
                agg[f"{key}_valid_count"] = int(len(values))
                agg[f"{key}_invalid_count"] = int(invalid_count)

        return agg

    @staticmethod
    def _split_retrain_train_val(
        train_pool_indices: list[int],
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[int], list[int]]:
        """Split a training pool into disjoint retrain-train / retrain-val subsets.

        Uses a seeded random shuffle instead of a temporal tail split.
        The previous temporal-tail approach caused the retrain validation set
        to be maximally distribution-shifted from the training portion in
        non-stationary forecasting datasets (e.g., Ganymede), leading to
        best_epoch=0 failures where the model never improved beyond
        random initialization.
        """
        if len(train_pool_indices) < 2:
            return train_pool_indices[:], train_pool_indices[:]

        rng = np.random.RandomState(seed)
        n_retrain_val = max(1, int(len(train_pool_indices) * val_ratio))

        shuffled = list(train_pool_indices)
        rng.shuffle(shuffled)

        retrain_val_idx = shuffled[:n_retrain_val]
        retrain_train_idx = shuffled[n_retrain_val:]

        if not retrain_train_idx:
            retrain_train_idx = shuffled[:-1]
            retrain_val_idx = shuffled[-1:]

        return retrain_train_idx, retrain_val_idx

    def _collect_forecasting_mase_context(
        self,
        indices: np.ndarray | list[int],
        *,
        target_mean: float | None = None,
        target_std: float | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Collect deterministic ordered training targets for MASE.

        Previous code used a random subsample.  Benchmark artifacts now derive
        the denominator from the exact split's training indices in dataset order
        so repeated runs produce the same MASE provenance.
        """
        train_targets = []
        train_groups = []
        train_orders = []
        for idx in np.asarray(indices, dtype=np.int64):
            _features, target, metadata = self.dataset[int(idx)]
            if isinstance(target, torch.Tensor):
                train_targets.append(target.detach().cpu().numpy())
            else:
                train_targets.append(np.asarray(target))
            group_id = self._extract_single_group_id(metadata)
            if group_id is not None:
                train_groups.append(group_id)
            order_id = self._extract_single_order_id(metadata)
            if order_id is not None:
                train_orders.append(order_id)

        if not train_targets:
            return None, None, None

        y_train = np.asarray(train_targets)
        if target_mean is not None and target_std is not None:
            y_train = NormalizedSubset.denormalize_targets(
                y_train,
                target_mean,
                target_std,
            )

        y_train_groups = (
            np.asarray(train_groups, dtype=object)
            if len(train_groups) == len(train_targets)
            else None
        )
        y_train_order = (
            np.asarray(train_orders, dtype=object)
            if len(train_orders) == len(train_targets)
            else None
        )
        return y_train, y_train_groups, y_train_order

    @staticmethod
    def _extract_instance_ids(metadata: Any) -> np.ndarray | None:
        """Extract per-sample instance IDs from collated batch metadata."""
        if isinstance(metadata, dict):
            instance_ids = metadata.get("instance_id")
        elif isinstance(metadata, list) and metadata and isinstance(metadata[0], dict):
            instance_ids = [m.get("instance_id") for m in metadata]
        else:
            return None

        if instance_ids is None:
            return None
        if isinstance(instance_ids, torch.Tensor):
            instance_ids = instance_ids.cpu().numpy()
        elif isinstance(instance_ids, tuple):
            instance_ids = list(instance_ids)

        instance_ids = np.asarray(instance_ids, dtype=object)
        if instance_ids.ndim == 0:
            instance_ids = instance_ids.reshape(1)
        return instance_ids

    @staticmethod
    def _extract_group_ids(metadata: Any) -> np.ndarray | None:
        """Extract per-sample forecasting group IDs from collated metadata."""
        if isinstance(metadata, dict):
            for key in ("well_id", "well_idx", "well", "group", "instance_id"):
                if key in metadata:
                    values = metadata[key]
                    break
            else:
                return None
        elif isinstance(metadata, list) and metadata and isinstance(metadata[0], dict):
            values = [
                ExperimentRunner._extract_single_group_id(m)
                for m in metadata
            ]
        else:
            return None

        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        elif isinstance(values, tuple):
            values = list(values)

        values = np.asarray(values, dtype=object)
        if values.ndim == 0:
            values = values.reshape(1)
        return values

    @staticmethod
    def _extract_single_group_id(metadata: Any) -> object | None:
        """Extract one group ID from uncollated dataset metadata."""
        if not isinstance(metadata, dict):
            return None
        for key in ("well_id", "well_idx", "well", "group", "instance_id"):
            if key in metadata:
                value = metadata[key]
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        return value.detach().cpu().item()
                    return tuple(value.detach().cpu().numpy().tolist())
                return value
        return None

    @staticmethod
    def _extract_order_ids(metadata: Any) -> np.ndarray | None:
        """Extract per-sample temporal order from collated forecasting metadata."""
        if isinstance(metadata, dict):
            for key in ("target_start", "start_idx", "timestamp"):
                if key in metadata:
                    values = metadata[key]
                    break
            else:
                return None
        elif isinstance(metadata, list) and metadata and isinstance(metadata[0], dict):
            values = [
                ExperimentRunner._extract_single_order_id(m)
                for m in metadata
            ]
        else:
            return None

        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        elif isinstance(values, tuple):
            values = list(values)

        values = np.asarray(values, dtype=object)
        if values.ndim == 0:
            values = values.reshape(1)
        return values

    @staticmethod
    def _extract_single_order_id(metadata: Any) -> object | None:
        """Extract one temporal order value from uncollated metadata."""
        if not isinstance(metadata, dict):
            return None
        for key in ("target_start", "start_idx", "timestamp"):
            if key in metadata:
                value = metadata[key]
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        return value.detach().cpu().item()
                    return tuple(value.detach().cpu().numpy().tolist())
                return value
        return None

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
        dataset: Any,
        train_indices: list[int],
        max_samples: int = 50000,
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
            logger.warning(
                "Could not compute class weights — using uniform", exc_info=True
            )
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
