"""CLI entrypoint for running experiments.

Usage::

    python -m offshore_dl.run_experiment --model lstm --dataset 3w
    python -m offshore_dl.run_experiment --model lstm --dataset ganymede --max-epochs 5
    python -m offshore_dl.run_experiment --model lstm --dataset cdf --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from offshore_dl.utils.serialization import make_serializable as _make_serializable
from offshore_dl.data.datasets import (
    CDFDataset,
    GanymedeDataset,
    InnerMongoliaDataset,
    ThreeWFeatureDataset,
    SPEBergDataset,
    ThreeWMultiScaleDataset,
    ThreeWDataset,
    ThreeWWaveletDataset,
    VolveDataset,
)
from offshore_dl.evaluation.cv import (
    GroupedExpandingWindowCV,
    SlidingWindowCV,
    StratifiedGroupKFoldSKLearn,
    resolve_cv_gap,
)
from offshore_dl.models.chronos_wrapper import ChronosWrapper
from offshore_dl.models.convtran import ConvTranModel
from offshore_dl.models.deeponet import DeepONetModel
from offshore_dl.models.inception_time import InceptionTimeModel
from offshore_dl.models.lstm import LSTMModel
from offshore_dl.models.tcn import TCNModel

try:
    from offshore_dl.models.patchtst import PatchTSTModel
except (ImportError, ModuleNotFoundError, RuntimeError):
    PatchTSTModel = None

try:
    from offshore_dl.models.timesfm_wrapper import TimesFMWrapper
except ImportError:
    TimesFMWrapper = None
try:
    from offshore_dl.models.tirex_wrapper import TiRexWrapper
except ImportError:
    TiRexWrapper = None
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
try:
    from sklearn.ensemble import RandomForestClassifier as RandomForestModel
except ImportError:
    RandomForestModel = None
from offshore_dl.training.experiment import ExperimentRunner
from offshore_dl.utils.config import load_merged_config
from offshore_dl.utils.reproducibility import set_global_seed
from offshore_dl.utils.results import resolve_results_dir


def _cfg_get(node, key: str, default=None):
    """Return ``node.key`` for OmegaConf/dict-like configs without requiring it."""
    if node is None:
        return default
    if hasattr(node, key):
        return getattr(node, key)
    if isinstance(node, dict):
        return node.get(key, default)
    return default


def _cv_setting(data_cfg, legacy_cv_cfg, top_level_key: str, legacy_key: str, default=None):
    """Read new top-level CV fields while preserving older nested test shape."""
    value = _cfg_get(data_cfg, top_level_key)
    if value is not None:
        return value
    return _cfg_get(legacy_cv_cfg, legacy_key, default)


def _forecast_cv_gap(cfg, ds) -> int:
    """Resolve forecasting CV embargo gap from explicit policy config.

    Preferred config shape is ``data.cv_gap_policy`` / ``data.cv_gap``.
    The older nested ``data.cv.gap_policy`` / ``data.cv.gap`` shape is accepted
    for compatibility with tests and external overrides.
    """
    data_cfg = _cfg_get(cfg, "data")
    legacy_cv_cfg = _cfg_get(data_cfg, "cv")
    explicit_gap = _cv_setting(data_cfg, legacy_cv_cfg, "cv_gap", "gap")
    policy = _cv_setting(
        data_cfg, legacy_cv_cfg, "cv_gap_policy", "gap_policy", "causal_horizon"
    )
    if explicit_gap is not None and explicit_gap != "auto":
        return resolve_cv_gap(policy, task="forecasting", explicit_gap=explicit_gap)

    horizon = int(getattr(ds, "horizon", _cfg_get(data_cfg, "horizon", 0)))
    input_window = int(
        getattr(ds, "input_window", getattr(ds, "window_size", ds[0][0].shape[0]))
    )
    dataset_gap = int(
        getattr(ds, "gap", _cfg_get(_cfg_get(data_cfg, "forecasting"), "gap", 0))
    )
    return resolve_cv_gap(
        policy,
        task="forecasting",
        input_window=input_window,
        horizon=horizon,
        dataset_gap=dataset_gap,
        explicit_gap=explicit_gap,
    )


def _cdf_cv_gap(cfg) -> int:
    """Resolve CDF/anomaly CV embargo gap from explicit policy config."""
    data_cfg = _cfg_get(cfg, "data")
    legacy_cv_cfg = _cfg_get(data_cfg, "cv")
    explicit_gap = _cv_setting(data_cfg, legacy_cv_cfg, "cv_gap", "gap")
    policy = _cv_setting(
        data_cfg, legacy_cv_cfg, "cv_gap_policy", "gap_policy", "strict_raw_row"
    )
    if explicit_gap is not None and explicit_gap != "auto":
        return resolve_cv_gap(policy, task="anomaly", explicit_gap=explicit_gap)

    window_size = int(_cfg_get(_cfg_get(data_cfg, "preprocessing"), "window_size", 1))
    return resolve_cv_gap(
        policy,
        task="anomaly",
        window_size=window_size,
        explicit_gap=explicit_gap,
    )

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, type] = {
    "lstm": LSTMModel,
    "deeponet": DeepONetModel,
    "deeponet_trunk_clf": DeepONetModel,
    "tcn": TCNModel,
    "inception_time": InceptionTimeModel,
    "convtran": ConvTranModel,
    "chronos": ChronosWrapper,
}

if PatchTSTModel is not None:
    MODEL_REGISTRY["patchtst"] = PatchTSTModel
if TimesFMWrapper is not None:
    MODEL_REGISTRY["timesfm"] = TimesFMWrapper
if TiRexWrapper is not None:
    MODEL_REGISTRY["tirex"] = TiRexWrapper
if FKMADModel is not None:
    MODEL_REGISTRY["fkmad"] = FKMADModel
if RandomForestModel is not None:
    MODEL_REGISTRY["random_forest"] = RandomForestModel
if MambaSLModel is not None:
    MODEL_REGISTRY["mambasl"] = MambaSLModel
if ConvTimeNetModel is not None:
    MODEL_REGISTRY["convtimenet"] = ConvTimeNetModel

DATASET_REGISTRY: dict[str, dict] = {
    "3w": {
        "class": ThreeWDataset,
        "config": "configs/data/3w.yaml",
        "task": "classification",
        "cv_factory": lambda cfg, ds: StratifiedGroupKFoldSKLearn(
            n_folds=5,
            labels=np.array([w["class_id"] for w in ds._windows]),
            groups=np.array([w["instance_id"] for w in ds._windows]),
            seed=42,
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "classification",
            "n_vars": ds.n_vars,
            "n_classes": cfg.data.n_classes,
            "window_size": cfg.data.preprocessing.window_size,
        },
    },
    "3w_multiscale": {
        "class": ThreeWMultiScaleDataset,
        "config": "configs/data/3w_multiscale.yaml",
        "task": "classification",
        "cv_factory": lambda cfg, ds: StratifiedGroupKFoldSKLearn(
            n_folds=5,
            labels=np.array([w["class_id"] for w in ds._inner._windows]),
            groups=np.array([w["instance_id"] for w in ds._inner._windows]),
            seed=42,
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "classification",
            "n_vars": ds.n_vars,
            "n_classes": cfg.data.n_classes,
            "window_size": ds.window_size,
        },
    },
    "3w_features": {
        "class": ThreeWFeatureDataset,
        "config": "configs/data/3w.yaml",
        "task": "classification",
        "cv_factory": lambda cfg, ds: StratifiedGroupKFoldSKLearn(
            n_folds=5,
            labels=np.array([w["class_id"] for w in ds._inner._windows]),
            groups=np.array([w["instance_id"] for w in ds._inner._windows]),
            seed=42,
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "classification",
            "n_vars": ds.n_vars,
            "n_classes": cfg.data.n_classes,
            "window_size": ds.window_size,
        },
    },
    "3w_wavelet": {
        "class": ThreeWWaveletDataset,
        "config": "configs/data/3w_wavelet.yaml",
        "task": "classification",
        "cv_factory": lambda cfg, ds: StratifiedGroupKFoldSKLearn(
            n_folds=5,
            labels=np.array([w["class_id"] for w in ds._inner._windows]),
            groups=np.array([w["instance_id"] for w in ds._inner._windows]),
            seed=42,
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "classification",
            "n_vars": ds.n_vars,
            "n_classes": cfg.data.n_classes,
            "window_size": ds.window_size,
        },
    },
    "ganymede": {
        "class": GanymedeDataset,
        "config": "configs/data/ganymede.yaml",
        "task": "forecasting",
        "cv_factory": lambda cfg, ds: GroupedExpandingWindowCV(
            groups=np.array([well_idx for well_idx, _ in ds._samples], dtype=np.int32),
            n_splits=3,
            min_train_ratio=0.5,
            gap=_forecast_cv_gap(cfg, ds),
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "forecasting",
            "n_vars": ds[0][0].shape[-1],  # infer from actual data
            "horizon": ds.horizon,
            "window_size": ds[0][0].shape[0],  # actual input window length
            "target_channel": ds._target_col_idx,
        },
    },
    "spe_berg": {
        "class": SPEBergDataset,
        "config": "configs/data/spe_berg.yaml",
        "task": "forecasting",
        "cv_factory": lambda cfg, ds: GroupedExpandingWindowCV(
            groups=np.array([well_idx for well_idx, _ in ds._samples], dtype=np.int32),
            n_splits=3,
            min_train_ratio=0.5,
            gap=_forecast_cv_gap(cfg, ds),
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "forecasting",
            "n_vars": ds[0][0].shape[-1],
            "horizon": ds.horizon,
            "window_size": ds[0][0].shape[0],
            "target_channel": ds._target_col_idx,
        },
    },
    "volve": {
        "class": VolveDataset,
        "config": "configs/data/volve.yaml",
        "task": "forecasting",
        "cv_factory": lambda cfg, ds: GroupedExpandingWindowCV(
            groups=np.array([well_idx for well_idx, _ in ds._samples], dtype=np.int32),
            n_splits=3,
            min_train_ratio=0.5,
            gap=_forecast_cv_gap(cfg, ds),
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "forecasting",
            "n_vars": ds[0][0].shape[-1],
            "horizon": ds.horizon,
            "window_size": ds[0][0].shape[0],
            "target_channel": ds._target_col_idx,
        },
    },
    "inner_mongolia": {
        "class": InnerMongoliaDataset,
        "config": "configs/data/inner_mongolia.yaml",
        "task": "forecasting",
        "cv_factory": lambda cfg, ds: GroupedExpandingWindowCV(
            groups=np.array([well_idx for well_idx, _ in ds._samples], dtype=np.int32),
            n_splits=3,
            min_train_ratio=0.5,
            gap=_forecast_cv_gap(cfg, ds),
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "forecasting",
            "n_vars": ds[0][0].shape[-1],
            "horizon": ds.horizon,
            "window_size": ds[0][0].shape[0],
            "target_channel": ds._target_col_idx,
        },
    },
    "cdf": {
        "class": CDFDataset,
        "config": "configs/data/cdf.yaml",
        "task": "anomaly",
        "cv_factory": lambda cfg, ds: SlidingWindowCV(
            n_splits=3,
            train_ratio=cfg.data.preprocessing.train_ratio,
            gap=_cdf_cv_gap(cfg),
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "anomaly",
            "n_vars": ds.n_vars,
            "window_size": cfg.data.preprocessing.window_size,
        },
    },
}


def _sanitize_patchtst_short_window(model_kwargs: dict) -> dict:
    """Clamp generic PatchTST kwargs that are invalid only for short windows.

    Direct ``PatchTSTModel`` construction remains fail-fast; this helper is
    applied only by ``build_experiment()`` after dataset/config kwargs are
    merged so short-window datasets do not inherit unsafe default patch config.
    Returns explicit runtime adjustment metadata when it mutates kwargs.
    """
    adjustments: dict[str, dict[str, int]] = {}
    window_size = model_kwargs.get("window_size")
    patch_len = model_kwargs.get("patch_len")
    stride = model_kwargs.get("stride")

    if window_size is None or patch_len is None:
        return adjustments

    if patch_len > window_size and window_size > 0:
        original = patch_len
        patch_len = max(1, window_size)
        model_kwargs["patch_len"] = patch_len
        adjustments["patch_len"] = {"from": int(original), "to": int(patch_len)}

    if stride is not None and stride > patch_len and patch_len > 0:
        original = stride
        model_kwargs["stride"] = patch_len
        adjustments["stride"] = {"from": int(original), "to": int(patch_len)}

    if adjustments:
        logger.warning(
            "Adjusted PatchTST short-window kwargs: %s",
            adjustments,
        )

    return adjustments


def build_experiment(
    model_name: str,
    dataset_name: str,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    device: str = "cpu",
    dataset_kwargs: dict | None = None,
) -> tuple[ExperimentRunner, dict]:
    """Build an ExperimentRunner from model and dataset names.

    Args:
        model_name: Key in MODEL_REGISTRY.
        dataset_name: Key in DATASET_REGISTRY.
        max_epochs: Override max epochs.
        batch_size: Override batch size.
        device: Device string.

    Returns:
        Tuple of (ExperimentRunner, merged_config).
    """
    if model_name not in MODEL_REGISTRY:
        msg = f"Unknown model: {model_name!r}. Available: {list(MODEL_REGISTRY.keys())}"
        raise ValueError(msg)
    if dataset_name not in DATASET_REGISTRY:
        msg = f"Unknown dataset: {dataset_name!r}. Available: {list(DATASET_REGISTRY.keys())}"
        raise ValueError(msg)

    ds_entry = DATASET_REGISTRY[dataset_name]
    model_class = MODEL_REGISTRY[model_name]

    # Load and merge configs
    model_config_path = f"configs/models/{model_name}.yaml"
    cfg = load_merged_config("configs/base.yaml", ds_entry["config"], model_config_path)

    # Apply overrides
    if max_epochs is not None:
        cfg.training.max_epochs = max_epochs
    if batch_size is not None:
        cfg.training.batch_size = batch_size
    cfg.device = device

    # Set seed
    set_global_seed(cfg.seed)

    # Instantiate dataset
    ds_class = ds_entry["class"]
    extra_ds_kwargs = dataset_kwargs or {}
    dataset = ds_class(ds_entry["config"], **extra_ds_kwargs)

    # Create CV strategy (some strategies need the dataset, e.g. StratifiedGroupKFoldSKLearn)
    cv_strategy = ds_entry["cv_factory"](cfg, dataset)

    # Get model kwargs
    model_kwargs = ds_entry["model_kwargs"](dataset, cfg)

    # Add architecture params from model config
    if hasattr(cfg, "model") and hasattr(cfg.model, "architecture"):
        arch = OmegaConf.to_container(cfg.model.architecture, resolve=True)
        if isinstance(arch, dict):
            model_kwargs.update(arch)
        else:
            flat_model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
            for reserved_key in (
                "architecture",
                "training",
                "optuna_search_space",
                "name",
            ):
                flat_model_cfg.pop(reserved_key, None)
            model_kwargs.update(flat_model_cfg)

    for key in ("loss_type", "focal_gamma", "label_smoothing"):
        if hasattr(cfg, "model") and hasattr(cfg.model, key):
            model_kwargs[key] = getattr(cfg.model, key)

    # Add training params
    if hasattr(cfg, "model") and hasattr(cfg.model, "training"):
        if hasattr(cfg.model.training, "lr"):
            model_kwargs["lr"] = cfg.model.training.lr
        if hasattr(cfg.model.training, "weight_decay"):
            model_kwargs["weight_decay"] = cfg.model.training.weight_decay

    runtime_adjustments = {}
    if model_name == "patchtst":
        patchtst_adjustments = _sanitize_patchtst_short_window(model_kwargs)
        if patchtst_adjustments:
            runtime_adjustments["patchtst_short_window"] = patchtst_adjustments

    runner = ExperimentRunner(
        model_class=model_class,
        dataset=dataset,
        cv_strategy=cv_strategy,
        cfg=cfg,
        model_kwargs=model_kwargs,
        runtime_adjustments=runtime_adjustments,
    )

    return runner, cfg


def run_and_save(
    model_name: str,
    dataset_name: str,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    device: str = "cpu",
    use_mlflow: bool = True,
    output_dir: str | Path | None = None,
    dataset_kwargs: dict | None = None,
) -> dict:
    """Run an experiment and save results to JSON.

    Args:
        model_name: Model name.
        dataset_name: Dataset name.
        max_epochs: Epoch limit.
        batch_size: Batch size override.
        device: Compute device.
        use_mlflow: Enable MLflow logging.
        output_dir: Base output directory.

    Returns:
        Results dict.
    """
    runner, cfg = build_experiment(
        model_name, dataset_name, max_epochs, batch_size, device, dataset_kwargs
    )

    logger.info(
        "Running %s on %s (max_epochs=%s, device=%s)",
        model_name,
        dataset_name,
        cfg.training.max_epochs,
        device,
    )

    results = runner.run(use_mlflow=use_mlflow)

    # Save results
    output_root = resolve_results_dir(output_dir, for_write=True)
    out_path = output_root / model_name / f"{dataset_name}.json"

    # Encode horizon/mode in filename for ganymede multi-horizon runs
    if dataset_name == "ganymede" and dataset_kwargs:
        horizon = dataset_kwargs.get("horizon", 30)
        mode = dataset_kwargs.get("mode", "multi_well")
        well_name = dataset_kwargs.get("well_name")
        if well_name:
            safe_name = well_name.replace("/", "_")
            out_path = (
                output_root
                / model_name
                / f"ganymede_h{horizon}_{mode}_{safe_name}.json"
            )
        else:
            out_path = (
                output_root / model_name / f"ganymede_h{horizon}_{mode}.json"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Make results JSON-serializable
    serializable = _make_serializable(results)
    out_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Results saved to %s", out_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  {model_name.upper()} on {dataset_name.upper()}")
    print(f"{'=' * 60}")
    print(f"  Folds: {results['n_folds']}")
    print("  Aggregate metrics:")
    for key, val in sorted(results.get("aggregate", {}).items()):
        print(f"    {key}: {val:.6f}")
    if results.get("cost"):
        print("  Cost:")
        for key, val in sorted(results["cost"].items()):
            print(f"    {key}: {val:.4f}")
    print(f"  Results saved: {out_path}")
    print(f"{'=' * 60}\n")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run offshore DL experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="Max training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow logging"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(resolve_results_dir(for_write=True)),
        help="Output directory for repaired result JSONs",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Max instances per class (3W only)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forecast horizon in days (Ganymede only)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["multi_well", "per_well"],
        help="Ganymede well mode",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    # Build dataset kwargs from CLI
    ds_kwargs = {}
    if args.max_instances and args.dataset == "3w":
        ds_kwargs["max_instances_per_class"] = args.max_instances
    if args.dataset == "ganymede":
        if args.horizon is not None:
            ds_kwargs["horizon"] = args.horizon
        if args.mode is not None:
            ds_kwargs["mode"] = args.mode

    run_and_save(
        model_name=args.model,
        dataset_name=args.dataset,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        device=args.device,
        use_mlflow=not args.no_mlflow,
        output_dir=args.output_dir,
        dataset_kwargs=ds_kwargs if ds_kwargs else None,
    )


if __name__ == "__main__":
    main()
