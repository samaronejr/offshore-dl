"""Optuna HPO for trained models on 3W, Ganymede, and SPE BERG.

Runs hyperparameter optimization using search spaces from YAML configs.
Each trial runs inner K-fold CV; the best params are then used for a
final nested evaluation (retrain on full pool → held-out test).

Usage::

    # 3W classification (all models, 30 trials each)
    python scripts/run_optuna_hpo.py --dataset 3w --n-trials 30 --device cuda

    # Ganymede forecasting (specific model, specific horizon)
    python scripts/run_optuna_hpo.py --dataset ganymede --models patchtst --horizon h7 --device cuda

    # Smoke test
    python scripts/run_optuna_hpo.py --dataset 3w --models lstm --n-trials 2 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import optuna
from omegaconf import OmegaConf

from offshore_dl.utils.config import load_merged_config
from offshore_dl.data.datasets import ThreeWFeatureDataset, GanymedeDataset, SPEBergDataset, VolveDataset
from offshore_dl.evaluation.cv import (
    GroupedExpandingWindowCV,
    GroupedTemporalHoldoutSplitter,
    HoldoutSplitter,
    StratifiedGroupKFoldSKLearn,
    resolve_cv_gap,
)
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.utils.reproducibility import set_global_seed
from offshore_dl.utils.results import resolve_results_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_HPO_OUTPUT_DIR = Path("results/hpo")
STAGE1_3W_MANIFEST = Path("scripts/hpo_3w_models.txt")


def _utc_campaign_id(prefix: str = "hpo") -> str:
    """Return a filesystem-safe UTC campaign id."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def _load_model_manifest(path: Path = STAGE1_3W_MANIFEST) -> list[str]:
    """Load model names from a manifest, ignoring blanks and comments."""
    if not path.exists():
        return []
    models = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            models.append(line)
    return models


def _load_3w_feature_dataset_for_hpo() -> ThreeWFeatureDataset:
    """Load 3W statistical features with an HPO-safe memory profile.

    The default 3W config keeps raw instance DataFrames in memory.  That is
    convenient for interactive training but too large for the lab GPU
    partition: Stage-1 HPO array tasks were killed while pre-computing
    descriptors.  For HPO we cache only raw float arrays during descriptor
    extraction, then release those raw caches because the feature dataset serves
    pre-computed ``(14, 27)`` matrices afterwards.
    """
    dataset = ThreeWFeatureDataset("configs/data/3w.yaml", cache_in_memory=False)
    dataset.release_inner_cache()
    return dataset


def _resolve_output_dir(output_dir: str | Path) -> Path:
    """Resolve the HPO output root."""
    return Path(output_dir)


def _result_dir(output_dir: Path, dataset: str, campaign_id: str | None) -> Path:
    """Directory for per-model HPO artifacts."""
    base = output_dir / dataset
    return base / campaign_id if campaign_id else base


def _result_path(
    output_dir: Path,
    dataset: str,
    campaign_id: str | None,
    model_name: str,
    horizon: str | None = None,
) -> Path:
    """Path to a per-model result JSON."""
    suffix = f"_{horizon}" if dataset in {"ganymede", "spe_berg", "volve"} else ""
    return _result_dir(output_dir, dataset, campaign_id) / f"{model_name}{suffix}.json"


def _summary_path(output_dir: Path, dataset: str, campaign_id: str | None) -> Path:
    """Path to a campaign or legacy summary JSON."""
    if campaign_id:
        return _result_dir(output_dir, dataset, campaign_id) / "summary.json"
    return output_dir / f"summary_{dataset}.json"


def _has_nonempty_mapping(value: object) -> bool:
    return isinstance(value, dict) and bool(value)


def _is_complete_hpo_result(data: dict, dataset: str = "3w") -> bool:
    """Return True only for final, benchmark-usable HPO JSON artifacts."""
    if not isinstance(data, dict):
        return False
    hpo = data.get("hpo")
    if not _has_nonempty_mapping(hpo):
        return False
    if not _has_nonempty_mapping(hpo.get("best_params")):
        return False
    if hpo.get("best_value") is None or hpo.get("n_trials") is None:
        return False

    test_metrics = data.get("test_metrics")
    cv_aggregate = data.get("cv_aggregate")
    if not _has_nonempty_mapping(test_metrics) or not _has_nonempty_mapping(cv_aggregate):
        return False
    required_metric = "f1_macro" if dataset == "3w" else None
    if required_metric and test_metrics.get(required_metric) is None:
        return False

    split_meta = data.get("split_metadata", {})
    if not _has_nonempty_mapping(split_meta):
        split_meta = {
            k: data.get(k)
            for k in ("n_train", "n_test", "n_cv_folds")
            if data.get(k) is not None
        }
    return all(split_meta.get(k) is not None for k in ("n_train", "n_test", "n_cv_folds"))


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _should_skip_existing(path: Path, dataset: str, skip_existing: bool) -> bool:
    """Decide whether an existing artifact is good enough to skip."""
    if not skip_existing or not path.exists():
        return False
    data = _load_json(path)
    if data is not None and _is_complete_hpo_result(data, dataset=dataset):
        logger.info("Skipping valid complete artifact: %s", path)
        return True
    if os.environ.get("ALLOW_INCOMPLETE") == "1":
        logger.warning("ALLOW_INCOMPLETE=1 — skipping incomplete artifact: %s", path)
        return True
    logger.warning("Existing artifact is incomplete/invalid; rerunning: %s", path)
    return False


def _configure_optuna_storage(
    cfg,
    storage_dir: str | Path | None,
    storage_key: str,
    *,
    resume: bool,
    force: bool,
) -> None:
    """Set a campaign-scoped SQLite storage path on the config."""
    if storage_dir is None:
        return
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)
    db_path = storage_path / f"{storage_key}.db"
    if db_path.exists() and not resume and force:
        sqlite_paths = (
            db_path,
            db_path.with_suffix(".db-wal"),
            db_path.with_suffix(".db-shm"),
        )
        for path in sqlite_paths:
            if path.exists():
                path.unlink()
    elif db_path.exists() and not resume:
        raise FileExistsError(
            f"Optuna storage exists for {storage_key}: {db_path}. "
            "Pass --resume to continue it or --force to replace it."
        )
    OmegaConf.update(cfg, "optuna.storage", f"sqlite:///{db_path}", merge=True)


def _split_metadata(
    *,
    n_train: int,
    n_test: int,
    n_cv_folds: int,
    n_trial_folds: int | None = None,
) -> dict:
    """Build common split metadata for result artifacts."""
    meta = {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_cv_folds": int(n_cv_folds),
    }
    if n_trial_folds is not None:
        meta["n_trial_folds"] = int(n_trial_folds)
    return meta


def _cfg_get(node, key: str, default=None):
    """Read a key from an OmegaConf/dict/object without assuming shape."""
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    try:
        if key in node:
            return node[key]
    except (TypeError, KeyError, AttributeError):
        pass
    return getattr(node, key, default)


def _forecast_cv_gap(cfg, dataset) -> int:
    """Resolve forecasting HPO embargo from the active dataset horizon."""
    data_cfg = _cfg_get(cfg, "data")
    forecasting_cfg = _cfg_get(data_cfg, "forecasting")
    policy = _cfg_get(data_cfg, "cv_gap_policy", "causal_horizon")
    explicit_gap = _cfg_get(data_cfg, "cv_gap", None)
    return int(
        resolve_cv_gap(
            policy,
            task="forecasting",
            input_window=int(getattr(dataset, "input_window", dataset[0][0].shape[0])),
            horizon=int(getattr(dataset, "horizon", _cfg_get(forecasting_cfg, "default_horizon", 1))),
            dataset_gap=int(getattr(dataset, "gap", _cfg_get(forecasting_cfg, "gap", 0))),
            explicit_gap=explicit_gap,
        )
    )


def _forecast_model_kwargs(entry: dict, dataset, horizon_days: int) -> dict:
    """Derive shape-sensitive forecasting model kwargs from the dataset."""
    kwargs = dict(entry["kwargs"])
    sample_x, _sample_y, _meta = dataset[0]
    kwargs.update(
        {
            "task": "forecasting",
            "n_vars": int(sample_x.shape[-1]),
            "window_size": int(sample_x.shape[0]),
            "horizon": int(horizon_days),
        }
    )
    if hasattr(dataset, "_target_col_idx"):
        kwargs["target_channel"] = int(dataset._target_col_idx)
    return kwargs


def _forecast_split_metadata(cfg, dataset, *, n_train: int, n_test: int, n_cv_folds: int, n_trial_folds: int | None = None) -> dict:
    """Build forecasting split metadata with active horizon/gap policy."""
    data_cfg = _cfg_get(cfg, "data")
    policy = _cfg_get(data_cfg, "cv_gap_policy", "causal_horizon")
    gap = _forecast_cv_gap(cfg, dataset)
    meta = _split_metadata(
        n_train=n_train,
        n_test=n_test,
        n_cv_folds=n_cv_folds,
        n_trial_folds=n_trial_folds,
    )
    meta.update(
        {
            "cv_gap_policy": str(policy),
            "cv_gap": int(gap),
            "outer_gap": int(gap),
            "inner_gap": int(gap),
            "raw_row_embargo_mode": "target_rows_only" if str(policy) == "causal_horizon" else str(policy),
            "input_window": int(getattr(dataset, "input_window", dataset[0][0].shape[0])),
            "horizon": int(getattr(dataset, "horizon", 0) or 0),
            "dataset_gap": int(getattr(dataset, "gap", 0) or 0),
        }
    )
    return meta


def _apply_best_params_to_final_eval(
    *,
    best_params: dict,
    base_kwargs: dict,
    cfg,
) -> dict:
    """Apply selected HPO params consistently for final retraining/eval.

    ``BaseModel.configure_optimizers`` reads optimizer settings from config
    before falling back to model attributes, so final evaluation must mirror
    the trial-time routing in ``OptunaObjective`` for ``lr`` and
    ``weight_decay``.
    """
    from offshore_dl.training.optuna_utils import OptunaObjective

    final_kwargs = dict(base_kwargs)
    for param_name, value in best_params.items():
        if param_name in OptunaObjective.OPTIMIZER_PARAMS:
            final_kwargs[param_name] = value
            OmegaConf.update(cfg, f"training.{param_name}", value, merge=True)
            OmegaConf.update(cfg, f"model.training.{param_name}", value, merge=True)
        elif param_name in OptunaObjective.TRAINING_PARAMS:
            OmegaConf.update(cfg, f"training.{param_name}", value, merge=True)
            if param_name == "scheduler":
                OmegaConf.update(
                    cfg,
                    "model.training.scheduler",
                    value,
                    merge=True,
                )
        else:
            final_kwargs[param_name] = value

    # Translate branch_width → branch_hidden (list of 2)
    if "branch_width" in final_kwargs:
        width = final_kwargs.pop("branch_width")
        final_kwargs["branch_hidden"] = [width, width]

    return final_kwargs


def _load_pytorch_deps():
    """Lazy-load PyTorch model classes and training utilities.

    Deferred to avoid importing transformers/torchvision at module level,
    which can fail when torchvision version is incompatible.
    """
    from offshore_dl.models.deeponet import DeepONetModel
    from offshore_dl.models.lstm import LSTMModel
    from offshore_dl.models.patchtst import PatchTSTModel
    from offshore_dl.models.tcn import TCNModel
    from offshore_dl.training.experiment import ExperimentRunner
    from offshore_dl.training.optuna_utils import run_hpo
    try:
        from offshore_dl.models.fkmad import FKMADModel
    except (ImportError, ModuleNotFoundError, RuntimeError):
        FKMADModel = None
    try:
        from offshore_dl.models.mambasl import MambaSLModel
    except (ImportError, ModuleNotFoundError, RuntimeError):
        MambaSLModel = None
    try:
        from offshore_dl.models.convtimenet import ConvTimeNetModel
    except (ImportError, ModuleNotFoundError, RuntimeError):
        ConvTimeNetModel = None
    return {
        "LSTMModel": LSTMModel,
        "DeepONetModel": DeepONetModel,
        "PatchTSTModel": PatchTSTModel,
        "TCNModel": TCNModel,
        "FKMADModel": FKMADModel,
        "MambaSLModel": MambaSLModel,
        "ConvTimeNetModel": ConvTimeNetModel,
        "ExperimentRunner": ExperimentRunner,
        "run_hpo": run_hpo,
    }


def _get_3w_models():
    """Build 3W model registry (lazy, imports PyTorch models)."""
    deps = _load_pytorch_deps()
    models = {
        "lstm": {
            "class": deps["LSTMModel"],
            "config": "configs/models/lstm.yaml",
            "kwargs": {
                "task": "classification",
                "n_vars": 27,
                "window_size": 14,
                "n_classes": 10,
                "hidden_size": 256,
                "num_layers": 2,
                "dropout": 0.3,
                "bidirectional": True,
            },
        },
        "deeponet": {
            "class": deps["DeepONetModel"],
            "config": "configs/models/deeponet.yaml",
            "kwargs": {
                "task": "classification",
                "n_vars": 27,
                "window_size": 14,
                "n_classes": 10,
                "rank": 128,
                "branch_hidden": [128, 128],
                "dropout": 0.2,
            },
        },
        "patchtst": {
            "class": deps["PatchTSTModel"],
            "config": "configs/models/patchtst.yaml",
            "kwargs": {
                "task": "classification",
                "n_vars": 27,
                "window_size": 14,
                "n_classes": 10,
                "pretrained": False,
                "d_model": 256,
                "d_ff": 512,
                "n_heads": 8,
                "n_layers": 3,
                "patch_len": 7,
                "stride": 4,
                "dropout": 0.15,
            },
        },
    }
    if deps["FKMADModel"] is not None:
        models["fkmad"] = {
            "class": deps["FKMADModel"],
            "config": "configs/models/fkmad.yaml",
            "kwargs": {
                "task": "classification",
                "n_vars": 27,
                "window_size": 14,
                "n_classes": 10,
                "d_model": 128,
                "n_mamba_layers": 2,
                "dropout": 0.2,
            },
        }
    if deps["MambaSLModel"] is not None:
        models["mambasl"] = {
            "class": deps["MambaSLModel"],
            "config": "configs/models/mambasl.yaml",
            "kwargs": {
                "task": "classification",
                "n_vars": 27,
                "window_size": 14,
                "n_classes": 10,
                "d_model": 128,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "d_ff": 256,
                "n_heads": 4,
                "dropout": 0.1,
            },
        }
    if deps["ConvTimeNetModel"] is not None:
        models["convtimenet"] = {
            "class": deps["ConvTimeNetModel"],
            "config": "configs/models/convtimenet.yaml",
            "kwargs": {
                "task": "classification",
                "n_vars": 27,
                "window_size": 14,
                "n_classes": 10,
                "d_model": 128,
                "d_ff": 256,
                "patch_size": 8,
                "patch_stride": 4,
                "dw_ks": [7, 13, 19],
                "dropout": 0.1,
                "pooling_tp": "max",
            },
        }
    return models


def _get_ganymede_models():
    """Build Ganymede model registry (lazy, imports PyTorch models)."""
    deps = _load_pytorch_deps()
    return {
        "lstm": {
            "class": deps["LSTMModel"],
            "config": "configs/models/lstm.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 63,
                "window_size": 30,
                "hidden_size": 256,
                "num_layers": 2,
                "dropout": 0.3,
                "bidirectional": True,
            },
        },
        "deeponet": {
            "class": deps["DeepONetModel"],
            "config": "configs/models/deeponet.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 63,
                "window_size": 30,
                "rank": 128,
                "branch_hidden": [128, 128],
                "dropout": 0.2,
            },
        },
        "patchtst": {
            "class": deps["PatchTSTModel"],
            "config": "configs/models/patchtst.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 63,
                "window_size": 30,
                "pretrained": False,
                "d_model": 256,
                "d_ff": 512,
                "n_heads": 8,
                "n_layers": 3,
                "patch_len": 8,
                "stride": 4,
                "dropout": 0.15,
            },
        },
        "tcn": {
            "class": deps["TCNModel"],
            "config": "configs/models/tcn.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 63,
                "window_size": 90,
                "n_channels": 128,
                "n_layers": 6,
                "kernel_size": 3,
                "dropout": 0.2,
            },
        },
    }


def _get_spe_berg_models():
    """Build SPE BERG model registry (lazy, imports PyTorch models)."""
    deps = _load_pytorch_deps()
    return {
        "lstm": {
            "class": deps["LSTMModel"],
            "config": "configs/models/lstm.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 67,
                "window_size": 90,
                "hidden_size": 256,
                "num_layers": 2,
                "dropout": 0.3,
                "bidirectional": True,
            },
        },
        "deeponet": {
            "class": deps["DeepONetModel"],
            "config": "configs/models/deeponet.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 67,
                "window_size": 90,
                "rank": 128,
                "branch_hidden": [128, 128],
                "dropout": 0.2,
            },
        },
        "patchtst": {
            "class": deps["PatchTSTModel"],
            "config": "configs/models/patchtst.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 67,
                "window_size": 90,
                "pretrained": False,
                "d_model": 256,
                "d_ff": 512,
                "n_heads": 8,
                "n_layers": 3,
                "patch_len": 8,
                "stride": 4,
                "dropout": 0.15,
            },
        },
        "tcn": {
            "class": deps["TCNModel"],
            "config": "configs/models/tcn.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 67,
                "window_size": 90,
                "n_channels": 128,
                "n_layers": 6,
                "kernel_size": 3,
                "dropout": 0.2,
            },
        },
    }


def _get_volve_models():
    """Build Volve model registry (lazy, imports PyTorch models).

    n_vars=73 (Volve has 73 feature columns), window_size=90.
    PatchTST requires target_channel=48 because BORE_OIL_VOL is at column index 48.
    """
    deps = _load_pytorch_deps()
    return {
        "lstm": {
            "class": deps["LSTMModel"],
            "config": "configs/models/lstm.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 73,
                "window_size": 90,
                "hidden_size": 256,
                "num_layers": 2,
                "dropout": 0.3,
                "bidirectional": True,
            },
        },
        "deeponet": {
            "class": deps["DeepONetModel"],
            "config": "configs/models/deeponet.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 73,
                "window_size": 90,
                "rank": 128,
                "branch_hidden": [128, 128],
                "dropout": 0.2,
            },
        },
        "patchtst": {
            "class": deps["PatchTSTModel"],
            "config": "configs/models/patchtst.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 73,
                "window_size": 90,
                "pretrained": False,
                "d_model": 256,
                "d_ff": 512,
                "n_heads": 8,
                "n_layers": 3,
                "patch_len": 8,
                "stride": 4,
                "dropout": 0.15,
                "target_channel": 48,
            },
        },
        "tcn": {
            "class": deps["TCNModel"],
            "config": "configs/models/tcn.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 73,
                "window_size": 90,
                "n_channels": 128,
                "n_layers": 6,
                "kernel_size": 3,
                "dropout": 0.2,
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════
# 3W CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════


def run_3w_hpo(
    model_name: str,
    n_trials: int,
    device: str,
    *,
    campaign_id: str | None = None,
    storage_dir: str | Path | None = None,
    resume: bool = False,
    force: bool = False,
    trial_folds: int | None = None,
    final_eval: bool = True,
) -> dict:
    """Run HPO for a 3W model."""
    from offshore_dl.training.experiment import ExperimentRunner
    from offshore_dl.training.optuna_utils import run_hpo

    set_global_seed(42)

    THREE_W_MODELS = _get_3w_models()
    entry = THREE_W_MODELS[model_name]
    model_cfg = OmegaConf.load(entry["config"])
    search_space = OmegaConf.to_container(
        model_cfg.model.optuna_search_space, resolve=True
    )
    logger.info("  Search space: %s", list(search_space.keys()))

    cfg = load_merged_config("configs/base.yaml", "configs/data/3w.yaml", entry["config"])
    cfg.training.max_epochs = 50  # shorter per trial for HPO
    cfg.training.batch_size = 64
    cfg.device = device
    cfg.training.scheduler = "cosine"
    cfg.training.checkpoint_metric = "f1_macro"
    cfg.training.checkpoint_mode = "max"
    study_name = f"3w_{model_name}_{campaign_id}" if campaign_id else f"3w_{model_name}"
    _configure_optuna_storage(
        cfg,
        storage_dir,
        study_name,
        resume=resume,
        force=force,
    )

    dataset = _load_3w_feature_dataset_for_hpo()
    logger.info("  3W loaded: %d samples", len(dataset))

    # Holdout split for final evaluation
    labels = np.array([int(dataset[i][1]) for i in range(len(dataset))])
    groups = np.array([dataset[i][2].get("instance_id", i) for i in range(len(dataset))])
    holdout = HoldoutSplitter(
        test_ratio=0.2, mode="stratified_group",
        labels=labels, groups=groups,
    )
    train_pool, test_indices = holdout.split(len(dataset))
    logger.info("  Holdout: train=%d, test=%d", len(train_pool), len(test_indices))

    # Inner CV for HPO trials — needs labels/groups for stratified split
    pool_labels = labels[train_pool]
    pool_groups = groups[train_pool]

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_pool.tolist())

    n_trial_folds = trial_folds or 5
    cv = StratifiedGroupKFoldSKLearn(
        n_folds=n_trial_folds, labels=pool_labels, groups=pool_groups, seed=42,
    )

    hpo_result = run_hpo(
        model_class=entry["class"],
        dataset=train_dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs=entry["kwargs"],
        primary_metric="f1_macro",
        search_space=search_space,
        n_trials=n_trials,
        study_name=study_name,
        direction="maximize",
        resume=resume,
    )

    logger.info("  Best params: %s (value=%.4f, trials=%d)",
                hpo_result["best_params"], hpo_result["best_value"],
                hpo_result["n_trials_completed"])

    # ── Final evaluation with best params ──
    best_kwargs = dict(entry["kwargs"])
    best_cfg = load_merged_config("configs/base.yaml", "configs/data/3w.yaml", entry["config"])
    best_cfg.training.max_epochs = 50  # same as HPO trials — lr schedule must match
    best_cfg.training.batch_size = 64
    best_cfg.device = device
    best_cfg.training.scheduler = "cosine"
    best_cfg.training.checkpoint_metric = "f1_macro"
    best_cfg.training.checkpoint_mode = "max"

    best_kwargs = _apply_best_params_to_final_eval(
        best_params=hpo_result["best_params"],
        base_kwargs=best_kwargs,
        cfg=best_cfg,
    )

    split_metadata = _split_metadata(
        n_train=len(train_pool),
        n_test=len(test_indices),
        n_cv_folds=5,
        n_trial_folds=n_trial_folds,
    )
    if final_eval:
        final_cv = StratifiedGroupKFoldSKLearn(
            n_folds=5, labels=pool_labels, groups=pool_groups, seed=42,
        )
        runner = ExperimentRunner(
            model_class=entry["class"],
            dataset=dataset,
            cv_strategy=final_cv,
            cfg=best_cfg,
            model_kwargs=best_kwargs,
        )
        nested_result = runner.run_nested(
            train_pool=train_pool,
            test_indices=test_indices,
            use_mlflow=True,
        )
        test_metrics = nested_result["test_metrics"]
        cv_aggregate = nested_result["cv_aggregate"]
        retrain_history = nested_result.get("retrain_history", {})
    else:
        logger.warning("  Final evaluation disabled; artifact will not validate as complete")
        test_metrics = {}
        cv_aggregate = {}
        retrain_history = {}

    return {
        "hpo": {
            "best_params": hpo_result["best_params"],
            "best_value": hpo_result["best_value"],
            "n_trials": hpo_result["n_trials_completed"],
        },
        "selection_metric": "f1_macro",
        "checkpoint_metric": "f1_macro",
        "test_metrics": test_metrics,
        "cv_aggregate": cv_aggregate,
        "split_metadata": split_metadata,
        "retrain_history": retrain_history,
        "baseline_kwargs": entry["kwargs"],
        "best_kwargs": best_kwargs,
    }


# ═══════════════════════════════════════════════════════════════════
# 3W CLASSIFICATION — SKLEARN MODELS (Random Forest, etc.)
# ═══════════════════════════════════════════════════════════════════


def _suggest_param(trial, name: str, spec: dict):
    """Sample a hyperparameter from its YAML spec (replicates OptunaObjective._suggest)."""
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


def run_3w_rf_hpo(
    n_trials: int,
    device: str,
    *,
    campaign_id: str | None = None,
    storage_dir: str | Path | None = None,
    resume: bool = False,
    force: bool = False,
    trial_folds: int | None = None,
    final_eval: bool = True,
) -> dict:
    """Run sklearn Random Forest HPO for 3W classification.

    Custom Optuna objective: each trial samples RF hyperparameters from
    the YAML search space, runs 5-fold StratifiedGroupKFoldSKLearn inner CV,
    and maximizes mean f1_macro.  After the study completes, the best params
    are used to retrain on the full training pool and evaluate on held-out test.
    """
    from sklearn.ensemble import RandomForestClassifier

    set_global_seed(42)

    model_cfg = OmegaConf.load("configs/models/random_forest.yaml")
    search_space = OmegaConf.to_container(
        model_cfg.model.optuna_search_space, resolve=True,
    )
    base_arch = OmegaConf.to_container(
        model_cfg.model.architecture, resolve=True,
    )
    logger.info("  RF search space: %s", list(search_space.keys()))

    # Load dataset and extract numpy arrays
    dataset = _load_3w_feature_dataset_for_hpo()
    n = len(dataset)
    logger.info("  3W loaded: %d samples", n)

    n_features = dataset[0][0].shape[0]  # window_size (14)
    n_sensors = dataset[0][0].shape[1]   # 27
    flat_dim = n_features * n_sensors

    X_all = np.empty((n, flat_dim), dtype=np.float32)
    Y_all = np.empty(n, dtype=np.int64)
    for i in range(n):
        x, y, _ = dataset[i]
        X_all[i] = x.numpy().reshape(-1)
        Y_all[i] = int(y)

    labels = np.array([int(dataset[i][1]) for i in range(n)])
    groups = np.array([dataset[i][2].get("instance_id", i) for i in range(n)])

    # Holdout split
    holdout = HoldoutSplitter(
        test_ratio=0.2, mode="stratified_group",
        labels=labels, groups=groups,
    )
    train_pool, test_indices = holdout.split(n)
    logger.info("  Holdout: train=%d, test=%d", len(train_pool), len(test_indices))

    X_train_pool, Y_train_pool = X_all[train_pool], Y_all[train_pool]
    X_test, Y_test = X_all[test_indices], Y_all[test_indices]

    pool_labels = labels[train_pool]
    pool_groups = groups[train_pool]

    # Inner CV for trials
    n_trial_folds = trial_folds or 5
    inner_cv = StratifiedGroupKFoldSKLearn(
        n_folds=n_trial_folds, labels=pool_labels, groups=pool_groups, seed=42,
    )
    inner_splits = inner_cv.get_splits(len(train_pool))
    final_cv = StratifiedGroupKFoldSKLearn(
        n_folds=5, labels=pool_labels, groups=pool_groups, seed=42,
    )
    final_splits = final_cv.get_splits(len(train_pool))

    # Optuna objective
    def objective(trial):
        params = {name: _suggest_param(trial, name, spec)
                  for name, spec in search_space.items()}
        # Merge sampled params with base architecture defaults
        rf_kwargs = dict(base_arch)
        rf_kwargs.update(params)

        fold_f1s = []
        for fold_idx, (local_train, local_val) in enumerate(inner_splits):
            X_tr, Y_tr = X_train_pool[local_train], Y_train_pool[local_train]
            X_va, Y_va = X_train_pool[local_val], Y_train_pool[local_val]

            clf = RandomForestClassifier(**rf_kwargs)
            clf.fit(X_tr, Y_tr)
            preds = clf.predict(X_va)
            probs = clf.predict_proba(X_va)

            metrics = MetricRegistry.compute(
                "classification", preds, Y_va, prediction_scores=probs,
            )
            fold_f1s.append(metrics["f1_macro"])
            trial.report(float(np.mean(fold_f1s)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"RF trial pruned after fold {fold_idx + 1}"
                )

        return float(np.mean(fold_f1s))

    # Create study and run optimization
    cfg = load_merged_config(
        "configs/base.yaml", "configs/data/3w.yaml",
        "configs/models/random_forest.yaml",
    )
    study_name = (
        f"3w_random_forest_{campaign_id}" if campaign_id else "3w_random_forest"
    )
    _configure_optuna_storage(
        cfg,
        storage_dir,
        study_name,
        resume=resume,
        force=force,
    )
    from offshore_dl.training.optuna_utils import create_study

    study = create_study(
        cfg,
        study_name=study_name,
        direction="maximize",
        load_if_exists=resume,
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_value = study.best_value
    n_completed = len(study.trials)
    logger.info("  Best params: %s (f1_macro=%.4f, trials=%d)",
                best_params, best_value, n_completed)

    # ── Final evaluation with best params ──
    final_kwargs = dict(base_arch)
    final_kwargs.update(best_params)
    split_metadata = _split_metadata(
        n_train=len(train_pool),
        n_test=len(test_indices),
        n_cv_folds=len(final_splits),
        n_trial_folds=n_trial_folds,
    )
    if not final_eval:
        logger.warning("  Final evaluation disabled; artifact will not validate as complete")
        return {
            "hpo": {
                "best_params": best_params,
                "best_value": best_value,
                "n_trials": n_completed,
            },
            "selection_metric": "f1_macro",
            "checkpoint_metric": "not_applicable_sklearn",
            "test_metrics": {},
            "cv_aggregate": {},
            "cv_fold_results": [],
            "split_metadata": split_metadata,
            "baseline_kwargs": base_arch,
            "best_kwargs": final_kwargs,
        }

    # Setup MLflow
    mlflow = None
    try:
        import mlflow as _mlflow
        _mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
        _mlflow.set_experiment("3w-random-forest-hpo")
        mlflow = _mlflow
    except Exception as exc:
        logger.warning("  MLflow unavailable for RF final eval (%s); continuing without tracking", exc)

    if mlflow:
        mlflow.start_run(run_name="rf_hpo_best_retrain")
        mlflow.log_params({k: str(v) for k, v in best_params.items()})
        mlflow.log_metric("hpo_best_f1_macro", best_value)

    # Inner CV with best params for cv_aggregate
    cv_fold_results = []
    for fold_idx, (local_train, local_val) in enumerate(final_splits):
        X_tr, Y_tr = X_train_pool[local_train], Y_train_pool[local_train]
        X_va, Y_va = X_train_pool[local_val], Y_train_pool[local_val]

        clf = RandomForestClassifier(**final_kwargs)
        clf.fit(X_tr, Y_tr)
        preds = clf.predict(X_va)
        probs = clf.predict_proba(X_va)
        metrics = MetricRegistry.compute(
            "classification", preds, Y_va, prediction_scores=probs,
        )
        cv_fold_results.append({"fold_idx": fold_idx, "metrics": metrics})
        if mlflow:
            mlflow.log_metric(f"cv_fold_{fold_idx}_f1_macro", metrics["f1_macro"])

    cv_agg = {}
    if cv_fold_results:
        metric_keys = [
            k for k in cv_fold_results[0]["metrics"]
            if isinstance(cv_fold_results[0]["metrics"][k], (int, float))
        ]
        for k in metric_keys:
            vals = [f["metrics"][k] for f in cv_fold_results]
            cv_agg[f"{k}_mean"] = float(np.mean(vals))
            cv_agg[f"{k}_std"] = float(np.std(vals))

    # Retrain on full pool → held-out test
    final_clf = RandomForestClassifier(**final_kwargs)
    final_clf.fit(X_train_pool, Y_train_pool)
    test_preds = final_clf.predict(X_test)
    test_probs = final_clf.predict_proba(X_test)
    test_metrics = MetricRegistry.compute(
        "classification", test_preds, Y_test, prediction_scores=test_probs,
    )

    if mlflow:
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", v)
        mlflow.end_run()

    metric_str = ", ".join(
        f"{k}={v:.4f}" for k, v in sorted(test_metrics.items())
        if isinstance(v, (int, float))
    )
    logger.info("  RF HPO final — TEST: %s", metric_str)

    return {
        "hpo": {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": n_completed,
        },
        "test_metrics": test_metrics,
        "cv_aggregate": cv_agg,
        "cv_fold_results": cv_fold_results,
        "selection_metric": "f1_macro",
        "checkpoint_metric": "not_applicable_sklearn",
        "split_metadata": split_metadata,
        "baseline_kwargs": base_arch,
        "best_kwargs": final_kwargs,
    }


# ═══════════════════════════════════════════════════════════════════
# GANYMEDE FORECASTING (PyTorch models)
# ═══════════════════════════════════════════════════════════════════


def run_ganymede_hpo(
    model_name: str, horizon: str, n_trials: int, device: str,
) -> dict:
    """Run HPO for a Ganymede model at a specific horizon."""
    from offshore_dl.training.experiment import ExperimentRunner
    from offshore_dl.training.optuna_utils import run_hpo

    set_global_seed(42)

    GANYMEDE_MODELS = _get_ganymede_models()
    entry = GANYMEDE_MODELS[model_name]
    model_cfg = OmegaConf.load(entry["config"])
    search_space = OmegaConf.to_container(
        model_cfg.model.optuna_search_space, resolve=True
    )

    cfg = load_merged_config("configs/base.yaml", "configs/data/ganymede.yaml", entry["config"])
    cfg.training.max_epochs = 50
    cfg.training.batch_size = 32
    cfg.device = device
    cfg.training.scheduler = "cosine"

    horizon_days = int(horizon.replace("h", ""))
    dataset = GanymedeDataset("configs/data/ganymede.yaml", horizon=horizon_days, filter_shutdowns=False)
    logger.info("  Ganymede %s loaded: %d samples", horizon, len(dataset))
    groups = np.array([well_idx for well_idx, _ in dataset._samples], dtype=np.int32)
    cv_gap = _forecast_cv_gap(cfg, dataset)
    model_kwargs = _forecast_model_kwargs(entry, dataset, horizon_days)

    # Temporal holdout
    holdout = GroupedTemporalHoldoutSplitter(test_ratio=0.2, groups=groups, gap=cv_gap)
    train_pool, test_indices = holdout.split(len(dataset))

    # Inner CV for HPO
    cv = GroupedExpandingWindowCV(groups=groups[train_pool], n_splits=3, gap=cv_gap)

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_pool.tolist())

    hpo_result = run_hpo(
        model_class=entry["class"],
        dataset=train_dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs=model_kwargs,
        primary_metric="mae",
        search_space=search_space,
        n_trials=n_trials,
        study_name=f"ganymede_{model_name}_{horizon}",
    )

    logger.info("  Best params: %s (value=%.4f)", hpo_result["best_params"], hpo_result["best_value"])

    # Final evaluation with best params
    best_kwargs = dict(model_kwargs)
    best_cfg = load_merged_config("configs/base.yaml", "configs/data/ganymede.yaml", entry["config"])
    best_cfg.training.max_epochs = 50  # same as HPO trials — lr schedule must match
    best_cfg.training.batch_size = 32
    best_cfg.device = device
    best_cfg.training.scheduler = "cosine"

    best_kwargs = _apply_best_params_to_final_eval(
        best_params=hpo_result["best_params"],
        base_kwargs=best_kwargs,
        cfg=best_cfg,
    )

    final_cv = GroupedExpandingWindowCV(groups=groups, n_splits=3, gap=cv_gap)
    runner = ExperimentRunner(
        model_class=entry["class"],
        dataset=dataset,
        cv_strategy=final_cv,
        cfg=best_cfg,
        model_kwargs=best_kwargs,
    )
    nested_result = runner.run_nested(
        train_pool=train_pool,
        test_indices=test_indices,
        use_mlflow=True,
    )

    return {
        "hpo": {
            "best_params": hpo_result["best_params"],
            "best_value": hpo_result["best_value"],
            "n_trials": hpo_result["n_trials_completed"],
        },
        "selection_metric": "mae",
        "checkpoint_metric": str(_cfg_get(best_cfg.training, "checkpoint_metric", "val_loss")),
        "test_metrics": nested_result["test_metrics"],
        "cv_aggregate": nested_result["cv_aggregate"],
        "split_metadata": _forecast_split_metadata(
            cfg,
            dataset,
            n_train=len(train_pool),
            n_test=len(test_indices),
            n_cv_folds=nested_result.get("n_cv_folds", 0),
            n_trial_folds=3,
        ),
        "baseline_kwargs": model_kwargs,
        "best_kwargs": best_kwargs,
    }


# ═══════════════════════════════════════════════════════════════════
# SPE BERG FORECASTING (PyTorch models)
# ═══════════════════════════════════════════════════════════════════


def run_spe_berg_hpo(
    model_name: str, horizon: str, n_trials: int, device: str,
) -> dict:
    """Run HPO for a SPE BERG model at a specific horizon."""
    from offshore_dl.training.experiment import ExperimentRunner
    from offshore_dl.training.optuna_utils import run_hpo

    set_global_seed(42)

    SPE_BERG_MODELS = _get_spe_berg_models()
    entry = SPE_BERG_MODELS[model_name]
    model_cfg = OmegaConf.load(entry["config"])
    search_space = OmegaConf.to_container(
        model_cfg.model.optuna_search_space, resolve=True
    )

    cfg = load_merged_config("configs/base.yaml", "configs/data/spe_berg.yaml", entry["config"])
    cfg.training.max_epochs = 50
    cfg.training.batch_size = 32
    cfg.device = device
    cfg.training.scheduler = "cosine"

    horizon_days = int(horizon.replace("h", ""))
    dataset = SPEBergDataset("configs/data/spe_berg.yaml", horizon=horizon_days, filter_shutdowns=False)
    logger.info("  SPE BERG %s loaded: %d samples", horizon, len(dataset))
    groups = np.array([well_idx for well_idx, _ in dataset._samples], dtype=np.int32)
    cv_gap = _forecast_cv_gap(cfg, dataset)
    model_kwargs = _forecast_model_kwargs(entry, dataset, horizon_days)

    # Temporal holdout
    holdout = GroupedTemporalHoldoutSplitter(test_ratio=0.2, groups=groups, gap=cv_gap)
    train_pool, test_indices = holdout.split(len(dataset))

    # Inner CV for HPO
    cv = GroupedExpandingWindowCV(groups=groups[train_pool], n_splits=3, gap=cv_gap)

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_pool.tolist())

    hpo_result = run_hpo(
        model_class=entry["class"],
        dataset=train_dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs=model_kwargs,
        primary_metric="mae",
        search_space=search_space,
        n_trials=n_trials,
        study_name=f"spe_berg_{model_name}_{horizon}",
    )

    logger.info("  Best params: %s (value=%.4f)", hpo_result["best_params"], hpo_result["best_value"])

    # Final evaluation with best params
    best_kwargs = dict(model_kwargs)
    best_cfg = load_merged_config("configs/base.yaml", "configs/data/spe_berg.yaml", entry["config"])
    best_cfg.training.max_epochs = 50  # same as HPO trials — lr schedule must match
    best_cfg.training.batch_size = 32
    best_cfg.device = device
    best_cfg.training.scheduler = "cosine"

    best_kwargs = _apply_best_params_to_final_eval(
        best_params=hpo_result["best_params"],
        base_kwargs=best_kwargs,
        cfg=best_cfg,
    )

    final_cv = GroupedExpandingWindowCV(groups=groups, n_splits=3, gap=cv_gap)
    runner = ExperimentRunner(
        model_class=entry["class"],
        dataset=dataset,
        cv_strategy=final_cv,
        cfg=best_cfg,
        model_kwargs=best_kwargs,
    )
    nested_result = runner.run_nested(
        train_pool=train_pool,
        test_indices=test_indices,
        use_mlflow=True,
    )

    return {
        "hpo": {
            "best_params": hpo_result["best_params"],
            "best_value": hpo_result["best_value"],
            "n_trials": hpo_result["n_trials_completed"],
        },
        "selection_metric": "mae",
        "checkpoint_metric": str(_cfg_get(best_cfg.training, "checkpoint_metric", "val_loss")),
        "test_metrics": nested_result["test_metrics"],
        "cv_aggregate": nested_result["cv_aggregate"],
        "split_metadata": _forecast_split_metadata(
            cfg,
            dataset,
            n_train=len(train_pool),
            n_test=len(test_indices),
            n_cv_folds=nested_result.get("n_cv_folds", 0),
            n_trial_folds=3,
        ),
        "baseline_kwargs": model_kwargs,
        "best_kwargs": best_kwargs,
    }


# ═══════════════════════════════════════════════════════════════════
# VOLVE FORECASTING (PyTorch models)
# ═══════════════════════════════════════════════════════════════════


def run_volve_hpo(
    model_name: str, horizon: str, n_trials: int, device: str,
) -> dict:
    """Run HPO for a Volve model at a specific horizon."""
    from offshore_dl.training.experiment import ExperimentRunner
    from offshore_dl.training.optuna_utils import run_hpo

    set_global_seed(42)

    VOLVE_MODELS = _get_volve_models()
    entry = VOLVE_MODELS[model_name]
    model_cfg = OmegaConf.load(entry["config"])
    search_space = OmegaConf.to_container(
        model_cfg.model.optuna_search_space, resolve=True
    )

    cfg = load_merged_config("configs/base.yaml", "configs/data/volve.yaml", entry["config"])
    cfg.training.max_epochs = 50
    cfg.training.batch_size = 32
    cfg.device = device
    cfg.training.scheduler = "cosine"

    horizon_days = int(horizon.replace("h", ""))
    dataset = VolveDataset("configs/data/volve.yaml", horizon=horizon_days, filter_shutdowns=False)
    logger.info("  Volve %s loaded: %d samples", horizon, len(dataset))
    groups = np.array([well_idx for well_idx, _ in dataset._samples], dtype=np.int32)
    cv_gap = _forecast_cv_gap(cfg, dataset)
    model_kwargs = _forecast_model_kwargs(entry, dataset, horizon_days)

    # Temporal holdout
    holdout = GroupedTemporalHoldoutSplitter(test_ratio=0.2, groups=groups, gap=cv_gap)
    train_pool, test_indices = holdout.split(len(dataset))

    # Inner CV for HPO
    cv = GroupedExpandingWindowCV(groups=groups[train_pool], n_splits=3, gap=cv_gap)

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_pool.tolist())

    hpo_result = run_hpo(
        model_class=entry["class"],
        dataset=train_dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs=model_kwargs,
        primary_metric="mae",
        search_space=search_space,
        n_trials=n_trials,
        study_name=f"volve_{model_name}_{horizon}",
    )

    logger.info("  Best params: %s (value=%.4f)", hpo_result["best_params"], hpo_result["best_value"])

    # Final evaluation with best params
    best_kwargs = dict(model_kwargs)
    best_cfg = load_merged_config("configs/base.yaml", "configs/data/volve.yaml", entry["config"])
    best_cfg.training.max_epochs = 50  # same as HPO trials — lr schedule must match
    best_cfg.training.batch_size = 32
    best_cfg.device = device
    best_cfg.training.scheduler = "cosine"

    best_kwargs = _apply_best_params_to_final_eval(
        best_params=hpo_result["best_params"],
        base_kwargs=best_kwargs,
        cfg=best_cfg,
    )

    final_cv = GroupedExpandingWindowCV(groups=groups, n_splits=3, gap=cv_gap)
    runner = ExperimentRunner(
        model_class=entry["class"],
        dataset=dataset,
        cv_strategy=final_cv,
        cfg=best_cfg,
        model_kwargs=best_kwargs,
    )
    nested_result = runner.run_nested(
        train_pool=train_pool,
        test_indices=test_indices,
        use_mlflow=True,
    )

    return {
        "hpo": {
            "best_params": hpo_result["best_params"],
            "best_value": hpo_result["best_value"],
            "n_trials": hpo_result["n_trials_completed"],
        },
        "selection_metric": "mae",
        "checkpoint_metric": str(_cfg_get(best_cfg.training, "checkpoint_metric", "val_loss")),
        "test_metrics": nested_result["test_metrics"],
        "cv_aggregate": nested_result["cv_aggregate"],
        "split_metadata": _forecast_split_metadata(
            cfg,
            dataset,
            n_train=len(train_pool),
            n_test=len(test_indices),
            n_cv_folds=nested_result.get("n_cv_folds", 0),
            n_trial_folds=3,
        ),
        "baseline_kwargs": model_kwargs,
        "best_kwargs": best_kwargs,
    }


# ═══════════════════════════════════════════════════════════════════

from offshore_dl.utils.serialization import make_serializable as _make_serializable


def _merge_summary(
    *,
    output_dir: Path,
    dataset: str,
    campaign_id: str | None,
    models: list[str],
    horizon: str,
) -> dict:
    """Merge per-model result JSONs into a compact status summary."""
    summary = {}
    for model_name in models:
        path = _result_path(output_dir, dataset, campaign_id, model_name, horizon=horizon)
        data = _load_json(path)
        if data is None:
            summary[model_name] = {
                "status": "error",
                "error": "missing_or_invalid_json",
                "path": str(path),
            }
            continue
        if not _is_complete_hpo_result(data, dataset=dataset):
            summary[model_name] = {
                "status": "error",
                "error": "incomplete_hpo_result",
                "path": str(path),
                "hpo": data.get("hpo", {}),
                "test_metrics": data.get("test_metrics", {}),
            }
            continue
        summary[model_name] = {
            "status": "ok",
            "path": str(path),
            "best_params": data.get("hpo", {}).get("best_params", {}),
            "best_value": data.get("hpo", {}).get("best_value"),
            "n_trials": data.get("hpo", {}).get("n_trials"),
            "test_metrics": data.get("test_metrics", {}),
            "cv_aggregate": data.get("cv_aggregate", {}),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for trained models")
    parser.add_argument("--dataset", required=True, choices=["3w", "ganymede", "spe_berg", "volve"],
                        help="Dataset to optimize on")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to optimize (default: all trained)")
    parser.add_argument("--horizon", default="h7",
                        help="Ganymede horizon (h7/h14/h30/h90)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials per model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--campaign-id",
        default=None,
        help=(
            "Campaign identifier. For 3W, omitted means a fresh UTC id; "
            "for forecasting datasets, omitted preserves legacy output paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_HPO_OUTPUT_DIR),
        help="HPO output root (default: results/hpo; not final benchmark output unless final eval is present)",
    )
    parser.add_argument(
        "--storage-dir",
        default=None,
        help="Optuna SQLite directory (default: <output-dir>/<dataset>/<campaign-id>/optuna)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip only complete, validator-acceptable result JSONs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing artifacts and replace campaign storage when not resuming",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write a summary JSON (use this in SLURM array tasks)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Merge existing per-model campaign outputs into a summary and exit",
    )
    parser.add_argument(
        "--trial-folds",
        type=int,
        default=None,
        help="Use fewer inner folds during HPO trials; final evaluation remains full-fold",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing campaign Optuna studies instead of requiring fresh storage",
    )
    final_eval_group = parser.add_mutually_exclusive_group()
    final_eval_group.add_argument(
        "--final-eval",
        dest="final_eval",
        action="store_true",
        help="Run final nested evaluation with best params (default)",
    )
    final_eval_group.add_argument(
        "--no-final-eval",
        dest="final_eval",
        action="store_false",
        help="Skip final nested evaluation; useful only for smoke/preflight runs",
    )
    parser.set_defaults(final_eval=True)
    args = parser.parse_args()

    if args.dataset == "3w":
        # Valid 3W models — names only for validation (lazy import of PyTorch deps)
        valid_models = {"lstm", "deeponet", "patchtst", "random_forest", "fkmad", "mambasl", "convtimenet"}
        default_models = _load_model_manifest() or [
            "lstm", "deeponet", "patchtst", "random_forest",
            "fkmad", "mambasl", "convtimenet",
        ]
    elif args.dataset == "spe_berg":
        # Valid SPE BERG models
        valid_models = {"lstm", "deeponet", "patchtst", "tcn"}
        default_models = ["lstm", "deeponet", "patchtst", "tcn"]
    elif args.dataset == "volve":
        # Valid Volve models
        valid_models = {"lstm", "deeponet", "patchtst", "tcn"}
        default_models = ["lstm", "deeponet", "patchtst", "tcn"]
    else:
        # Valid Ganymede models
        valid_models = {"lstm", "deeponet", "patchtst", "tcn"}
        default_models = ["lstm", "deeponet", "patchtst", "tcn"]

    models = args.models or default_models
    if args.trial_folds is not None and args.trial_folds < 2:
        parser.error("--trial-folds must be >= 2")

    output_dir = _resolve_output_dir(args.output_dir)
    campaign_id = args.campaign_id
    if args.dataset == "3w" and campaign_id is None:
        campaign_id = _utc_campaign_id("3w-hpo")
    storage_dir = args.storage_dir
    if storage_dir is None and campaign_id:
        storage_dir = output_dir / args.dataset / campaign_id / "optuna"

    if args.summary_only:
        summary = _merge_summary(
            output_dir=output_dir,
            dataset=args.dataset,
            campaign_id=campaign_id,
            models=models,
            horizon=args.horizon,
        )
        path = _summary_path(output_dir, args.dataset, campaign_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_make_serializable(summary), indent=2))
        logger.info("HPO summary saved: %s", path)
        if any(row.get("status") != "ok" for row in summary.values()):
            raise SystemExit(1)
        return

    logger.info("═" * 70)
    logger.info("OPTUNA HPO — %s", args.dataset.upper())
    logger.info(
        "  models=%s  n_trials=%d  device=%s  campaign=%s",
        models,
        args.n_trials,
        args.device,
        campaign_id or "legacy",
    )
    logger.info("  output_dir=%s  storage_dir=%s", output_dir, storage_dir or "config/default")
    logger.info("═" * 70)

    summary = {}
    had_errors = False
    for model_name in models:
        if model_name not in valid_models:
            logger.warning("Unknown model: %s (skipping)", model_name)
            continue

        logger.info("─" * 60)
        logger.info("HPO: %s", model_name)
        logger.info("─" * 60)
        start = time.time()
        out_path = _result_path(
            output_dir,
            args.dataset,
            campaign_id,
            model_name,
            horizon=args.horizon,
        )
        if not args.force and _should_skip_existing(
            out_path,
            args.dataset,
            args.skip_existing,
        ):
            data = _load_json(out_path) or {}
            summary[model_name] = {
                "status": "ok",
                "elapsed": 0.0,
                "skipped": True,
                "path": str(out_path),
                "best_params": data.get("hpo", {}).get("best_params", {}),
                "test_metrics": data.get("test_metrics", {}),
            }
            continue

        try:
            if args.dataset == "3w":
                if model_name == "random_forest":
                    result = run_3w_rf_hpo(
                        args.n_trials,
                        args.device,
                        campaign_id=campaign_id,
                        storage_dir=storage_dir,
                        resume=args.resume,
                        force=args.force,
                        trial_folds=args.trial_folds,
                        final_eval=args.final_eval,
                    )
                else:
                    result = run_3w_hpo(
                        model_name,
                        args.n_trials,
                        args.device,
                        campaign_id=campaign_id,
                        storage_dir=storage_dir,
                        resume=args.resume,
                        force=args.force,
                        trial_folds=args.trial_folds,
                        final_eval=args.final_eval,
                    )
            elif args.dataset == "spe_berg":
                result = run_spe_berg_hpo(model_name, args.horizon, args.n_trials, args.device)
            elif args.dataset == "volve":
                result = run_volve_hpo(model_name, args.horizon, args.n_trials, args.device)
            else:
                result = run_ganymede_hpo(model_name, args.horizon, args.n_trials, args.device)

            elapsed = time.time() - start

            # Save per-model result
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(_make_serializable(result), indent=2))

            tm = result.get("test_metrics", {})
            if args.dataset == "3w":
                logger.info("✓ %s: test acc=%.4f, F1m=%.4f (best_trial_value=%.4f, %d trials, %.0fs)",
                            model_name, tm.get("accuracy", 0), tm.get("f1_macro", 0),
                            result["hpo"]["best_value"], result["hpo"]["n_trials"], elapsed)
            else:
                logger.info("✓ %s: test MAE=%.4f, R2p=%.4f (best_trial_value=%.4f, %d trials, %.0fs)",
                            model_name, tm.get("mae", 0), tm.get("r2_prod", 0),
                            result["hpo"]["best_value"], result["hpo"]["n_trials"], elapsed)

            summary[model_name] = {
                "status": "ok", "elapsed": round(elapsed, 1),
                "path": str(out_path),
                "best_params": result["hpo"]["best_params"],
                "test_metrics": tm,
            }

        except Exception as e:
            had_errors = True
            elapsed = time.time() - start
            logger.error("✗ %s failed: %s (%.1fs)", model_name, e, elapsed)
            traceback.print_exc()
            error_result = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "elapsed": round(elapsed, 1),
                "path": str(out_path),
            }
            summary[model_name] = error_result
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(_make_serializable(error_result), indent=2))
            except OSError as write_err:
                logger.warning("Could not write error artifact %s: %s", out_path, write_err)

    # Save summary unless this is an array task.
    if args.no_summary:
        logger.info("HPO summary skipped (--no-summary)")
    else:
        summary_path = _summary_path(output_dir, args.dataset, campaign_id)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
        logger.info("HPO summary saved: %s", summary_path)

    if had_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
