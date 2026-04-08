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
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from offshore_dl.data.datasets import CDFDataset, GanymedeDataset, SPEBergDataset, ThreeWDataset, VolveDataset
from offshore_dl.evaluation.cv import (
    ExpandingWindowCV,
    StratifiedGroupKFoldSKLearn,
    TemporalSplitCV,
)
from offshore_dl.models.chronos_wrapper import ChronosWrapper
from offshore_dl.models.deeponet import DeepONetModel
from offshore_dl.models.lstm import LSTMModel
from offshore_dl.models.patchtst import PatchTSTModel
from offshore_dl.models.tcn import TCNModel

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
from offshore_dl.training.experiment import ExperimentRunner
from offshore_dl.utils.config import load_merged_config
from offshore_dl.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, type] = {
    "lstm": LSTMModel,
    "deeponet": DeepONetModel,
    "patchtst": PatchTSTModel,
    "tcn": TCNModel,
    "chronos": ChronosWrapper,
}

if TimesFMWrapper is not None:
    MODEL_REGISTRY["timesfm"] = TimesFMWrapper
if TiRexWrapper is not None:
    MODEL_REGISTRY["tirex"] = TiRexWrapper
if FKMADModel is not None:
    MODEL_REGISTRY["fkmad"] = FKMADModel

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
            "n_vars": 27,
            "n_classes": cfg.data.n_classes,
            "window_size": cfg.data.preprocessing.window_size,
        },
    },
    "ganymede": {
        "class": GanymedeDataset,
        "config": "configs/data/ganymede.yaml",
        "task": "forecasting",
        "cv_factory": lambda cfg, ds: ExpandingWindowCV(
            n_splits=3,
            min_train_ratio=0.5,
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "forecasting",
            "n_vars": ds[0][0].shape[-1],  # infer from actual data
            "horizon": ds.horizon,
            "window_size": ds[0][0].shape[0],  # actual input window length
        },
    },
    "spe_berg": {
        "class": SPEBergDataset,
        "config": "configs/data/spe_berg.yaml",
        "task": "forecasting",
        "cv_factory": lambda cfg, ds: ExpandingWindowCV(
            n_splits=3,
            min_train_ratio=0.5,
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "forecasting",
            "n_vars": ds[0][0].shape[-1],
            "horizon": ds.horizon,
            "window_size": ds[0][0].shape[0],
        },
    },
    "volve": {
        "class": VolveDataset,
        "config": "configs/data/volve.yaml",
        "task": "forecasting",
        "cv_factory": lambda cfg, ds: ExpandingWindowCV(
            n_splits=3,
            min_train_ratio=0.5,
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
        "cv_factory": lambda cfg, ds: TemporalSplitCV(
            train_ratio=cfg.data.preprocessing.train_ratio,
        ),
        "model_kwargs": lambda ds, cfg: {
            "task": "anomaly",
            "n_vars": 11,
            "window_size": cfg.data.preprocessing.window_size,
        },
    },
}


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
        model_kwargs.update(arch)

    # Add training params
    if hasattr(cfg, "model") and hasattr(cfg.model, "training"):
        model_kwargs["lr"] = cfg.model.training.lr
        model_kwargs["weight_decay"] = cfg.model.training.weight_decay

    runner = ExperimentRunner(
        model_class=model_class,
        dataset=dataset,
        cv_strategy=cv_strategy,
        cfg=cfg,
        model_kwargs=model_kwargs,
    )

    return runner, cfg


def run_and_save(
    model_name: str,
    dataset_name: str,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    device: str = "cpu",
    use_mlflow: bool = True,
    output_dir: str | Path = "results",
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
    runner, cfg = build_experiment(model_name, dataset_name, max_epochs, batch_size, device, dataset_kwargs)

    logger.info("Running %s on %s (max_epochs=%s, device=%s)", model_name, dataset_name, cfg.training.max_epochs, device)

    results = runner.run(use_mlflow=use_mlflow)

    # Save results
    out_path = Path(output_dir) / model_name / f"{dataset_name}.json"

    # Encode horizon/mode in filename for ganymede multi-horizon runs
    if dataset_name == "ganymede" and dataset_kwargs:
        horizon = dataset_kwargs.get("horizon", 30)
        mode = dataset_kwargs.get("mode", "multi_well")
        well_name = dataset_kwargs.get("well_name")
        if well_name:
            safe_name = well_name.replace("/", "_")
            out_path = Path(output_dir) / model_name / f"ganymede_h{horizon}_{mode}_{safe_name}.json"
        else:
            out_path = Path(output_dir) / model_name / f"ganymede_h{horizon}_{mode}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Make results JSON-serializable
    serializable = _make_serializable(results)
    out_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Results saved to %s", out_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} on {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"  Folds: {results['n_folds']}")
    print(f"  Aggregate metrics:")
    for key, val in sorted(results.get("aggregate", {}).items()):
        print(f"    {key}: {val:.6f}")
    if results.get("cost"):
        print(f"  Cost:")
        for key, val in sorted(results["cost"].items()):
            print(f"    {key}: {val:.4f}")
    print(f"  Results saved: {out_path}")
    print(f"{'='*60}\n")

    return results


def _make_serializable(obj):
    """Convert non-serializable types for JSON output."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items() if k != "study"}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run offshore DL experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()), help="Model name")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_REGISTRY.keys()), help="Dataset name")
    parser.add_argument("--max-epochs", type=int, default=None, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--max-instances", type=int, default=None, help="Max instances per class (3W only)")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in days (Ganymede only)")
    parser.add_argument("--mode", type=str, default=None, choices=["multi_well", "per_well"], help="Ganymede well mode")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

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
