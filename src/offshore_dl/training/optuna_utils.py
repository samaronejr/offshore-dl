"""Optuna integration utilities for hyperparameter optimization.

Provides study creation with MedianPruner, convergence callback,
and an objective wrapper around ExperimentRunner.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def create_study(
    cfg: DictConfig,
    study_name: str,
    direction: str = "minimize",
) -> Any:
    """Create an Optuna study with configured pruner and storage.

    Args:
        cfg: Config with optuna.storage and optuna.pruner settings.
        study_name: Name for the study.
        direction: "minimize" or "maximize".

    Returns:
        optuna.Study instance.
    """
    import optuna

    storage = cfg.optuna.storage if hasattr(cfg, "optuna") else None
    # Fall back to in-memory if SQLite path isn't writable (e.g. Singularity)
    if storage and storage.startswith("sqlite:///"):
        db_path = storage.replace("sqlite:///", "")
        import os
        db_dir = os.path.dirname(db_path) or "."
        if not os.access(db_dir, os.W_OK):
            logger.warning("Optuna storage dir not writable (%s), using in-memory", db_dir)
            storage = None
    pruner_type = cfg.optuna.pruner if hasattr(cfg, "optuna") else "median"

    if pruner_type == "median":
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    logger.info("Optuna study created: %s (storage: %s)", study_name, storage)
    return study


def convergence_callback(patience: int = 20, threshold: float = 0.005) -> Any:
    """Create a callback that stops the study when improvement plateaus.

    Args:
        patience: Number of trials without improvement before stopping.
        threshold: Minimum improvement to count as progress.

    Returns:
        Callable for ``study.optimize(callbacks=[...])``.
    """

    def callback(study, trial) -> None:
        """Check for convergence and stop if plateaued."""
        if len(study.trials) < patience:
            return

        best_values = []
        for t in study.trials[-patience:]:
            if t.value is not None:
                best_values.append(t.value)

        if len(best_values) < 2:
            return

        # Check if improvement is below threshold
        improvement = abs(max(best_values) - min(best_values))
        if improvement < threshold:
            logger.info(
                "Convergence callback: stopping after %d trials "
                "(improvement=%.6f < threshold=%.6f)",
                len(study.trials), improvement, threshold,
            )
            study.stop()

    return callback


class OptunaObjective:
    """Wraps ExperimentRunner as an Optuna objective function.

    Samples hyperparameters from the search space config, creates a
    modified config, runs the experiment, and returns the primary metric.

    Args:
        model_class: BaseModel subclass.
        dataset: Dataset instance.
        cv_strategy: CV strategy instance.
        base_cfg: Base configuration.
        model_kwargs: Base model kwargs (task, n_vars, etc.).
        primary_metric: Metric key to optimize (e.g., "f1_macro", "mae").
        search_space: Dict mapping param names to suggest configs.
    """

    def __init__(
        self,
        model_class: type,
        dataset: Any,
        cv_strategy: Any,
        base_cfg: DictConfig,
        model_kwargs: dict,
        primary_metric: str = "val_loss",
        search_space: dict | None = None,
    ) -> None:
        self.model_class = model_class
        self.dataset = dataset
        self.cv_strategy = cv_strategy
        self.base_cfg = base_cfg
        self.model_kwargs = model_kwargs
        self.primary_metric = primary_metric
        self.search_space = search_space or {}

    # Training-level hyperparams that belong in cfg, not model kwargs
    TRAINING_PARAMS = {"lr", "batch_size", "max_epochs", "scheduler"}

    def __call__(self, trial) -> float:
        """Run one trial of HPO."""
        # Sample hyperparameters
        kwargs = dict(self.model_kwargs)
        cfg = OmegaConf.create(OmegaConf.to_container(self.base_cfg, resolve=True))

        for param_name, spec in self.search_space.items():
            value = self._suggest(trial, param_name, spec)
            if param_name in self.TRAINING_PARAMS:
                OmegaConf.update(cfg, f"training.{param_name}", value)
            else:
                kwargs[param_name] = value

        # Translate branch_width → branch_hidden (list of 2)
        if "branch_width" in kwargs:
            w = kwargs.pop("branch_width")
            kwargs["branch_hidden"] = [w, w]

        from offshore_dl.training.experiment import ExperimentRunner

        runner = ExperimentRunner(
            model_class=self.model_class,
            dataset=self.dataset,
            cv_strategy=self.cv_strategy,
            cfg=cfg,
            model_kwargs=kwargs,
        )

        results = runner.run(use_mlflow=False)
        aggregate = results.get("aggregate", {})

        # Return primary metric
        metric_key = f"{self.primary_metric}_mean"
        if metric_key in aggregate:
            return aggregate[metric_key]

        # Fallback: use mean val_loss from fold histories
        val_losses = []
        for fr in results.get("fold_results", []):
            vl = fr.get("history", {}).get("final_val_loss")
            if vl is not None:
                val_losses.append(vl)

        return float(sum(val_losses) / max(len(val_losses), 1))

    @staticmethod
    def _suggest(trial, name: str, spec: dict) -> Any:
        """Sample a hyperparameter value from its spec."""
        suggest_type = spec.get("type", "float")

        if suggest_type == "float":
            return trial.suggest_float(
                name, spec.get("low", 0.0001), spec.get("high", 0.1),
                log=spec.get("log", False),
            )
        elif suggest_type == "int":
            return trial.suggest_int(name, spec.get("low", 1), spec.get("high", 100))
        elif suggest_type == "categorical":
            return trial.suggest_categorical(name, spec.get("choices", []))
        else:
            msg = f"Unknown suggest type: {suggest_type!r}"
            raise ValueError(msg)


def run_hpo(
    model_class: type,
    dataset: Any,
    cv_strategy: Any,
    cfg: DictConfig,
    model_kwargs: dict,
    primary_metric: str = "val_loss",
    search_space: dict | None = None,
    n_trials: int | None = None,
    study_name: str | None = None,
    direction: str = "minimize",
) -> dict:
    """Run hyperparameter optimization.

    Args:
        model_class: BaseModel subclass.
        dataset: Dataset instance.
        cv_strategy: CV strategy.
        cfg: Configuration.
        model_kwargs: Base model kwargs.
        primary_metric: Metric to optimize.
        search_space: Param search space.
        n_trials: Number of trials override.
        study_name: Study name override.

    Returns:
        Dict with best_trial, best_params, best_value, study.
    """
    if study_name is None:
        study_name = f"{model_class.__name__}_{type(dataset).__name__}"

    n_trials = n_trials or (cfg.optuna.n_trials_min if hasattr(cfg, "optuna") else 10)
    patience = cfg.optuna.convergence_patience if hasattr(cfg, "optuna") else 20
    threshold = cfg.optuna.convergence_threshold if hasattr(cfg, "optuna") else 0.005

    study = create_study(cfg, study_name, direction=direction)
    objective = OptunaObjective(
        model_class=model_class,
        dataset=dataset,
        cv_strategy=cv_strategy,
        base_cfg=cfg,
        model_kwargs=model_kwargs,
        primary_metric=primary_metric,
        search_space=search_space,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[convergence_callback(patience, threshold)],
    )

    return {
        "best_trial": study.best_trial.number,
        "best_params": study.best_trial.params,
        "best_value": study.best_value,
        "n_trials_completed": len(study.trials),
        "study": study,
    }
