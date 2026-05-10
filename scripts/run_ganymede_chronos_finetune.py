"""Fine-tune Chronos-2 on Ganymede multi-well forecasting.

Uses the same grouped 80/20 temporal holdout protocol as the existing
Ganymede benchmark, but fine-tunes Chronos-2 on the pre-holdout history
of each well before evaluating on the held-out windows.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from offshore_dl.data.datasets import GanymedeDataset
from offshore_dl.evaluation.cv import GroupedTemporalHoldoutSplitter, resolve_cv_gap
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.utils.config import load_merged_config
from offshore_dl.utils.reproducibility import set_global_seed
from offshore_dl.utils.results import resolve_results_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = resolve_results_dir(for_write=True) / "chronos_finetuned"
DEFAULT_MODEL_NAME = "amazon/chronos-2"


def _module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _sample_groups(dataset: GanymedeDataset) -> np.ndarray:
    return np.array([well_idx for well_idx, _ in dataset._samples], dtype=np.int32)


def _make_holdout(dataset: GanymedeDataset) -> GroupedTemporalHoldoutSplitter:
    data_cfg = dataset.cfg.data
    gap = resolve_cv_gap(
        data_cfg.get("cv_gap_policy", "causal_horizon"),
        task="forecasting",
        input_window=int(dataset.input_window),
        horizon=int(dataset.horizon),
        dataset_gap=int(dataset.gap),
        explicit_gap=data_cfg.get("cv_gap", None),
    )
    return GroupedTemporalHoldoutSplitter(
        test_ratio=0.2, groups=_sample_groups(dataset), gap=gap
    )


from offshore_dl.utils.serialization import make_serializable as _make_serializable


def _get_support_status() -> dict[str, object]:
    chronos_version = _module_version("chronos")
    autogluon_version = _module_version("autogluon")
    peft_version = _module_version("peft")
    return {
        "chronos_version": chronos_version,
        "autogluon_version": autogluon_version,
        "peft_version": peft_version,
        "chronos2_pipeline_available": _has_module("chronos.chronos2.pipeline"),
        "lora_available": _has_module("peft"),
    }


def _check_ganymede_data_available() -> None:
    cfg = load_merged_config("configs/base.yaml", "configs/data/ganymede.yaml")
    processed_dir = Path(cfg.data.paths.processed)
    raw_path = Path(cfg.data.paths.raw)
    has_processed = processed_dir.exists() and any(processed_dir.glob("*.parquet"))
    if has_processed or raw_path.exists():
        return

    raise FileNotFoundError(
        "Ganymede data not found. Expected preprocessed parquets under "
        f"{processed_dir} or the raw CSV at {raw_path}."
    )


def _derive_train_series_end(
    dataset: GanymedeDataset, test_ratio: float = 0.2
) -> dict[int, int]:
    """Row-exclusive end index for each well's fine-tuning history.

    The first held-out forecast target for a well begins at:
        split_point + input_window + gap
    where split_point is the grouped temporal holdout boundary in sample space.

    We fine-tune only on observations strictly before that target boundary so
    the model never sees held-out target values during adaptation.
    """

    cutoffs: dict[int, int] = {}
    total_needed = dataset.input_window + dataset.gap + dataset.horizon
    for well_idx, (_well_name, _df) in enumerate(dataset._well_data):
        n_rows = len(dataset._arrays[well_idx])
        n_samples = max(0, n_rows - total_needed + 1)
        if n_samples < 2:
            continue

        split_point = int(n_samples * (1.0 - test_ratio))
        split_point = max(1, min(split_point, n_samples - 1))
        cutoffs[well_idx] = min(
            n_rows, split_point + dataset.input_window + dataset.gap
        )

    return cutoffs


def _series_to_chronos_input(
    values: np.ndarray,
    columns: list[str],
    target_idx: int,
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    target = values[:, target_idx].astype(np.float32, copy=False)
    item: dict[str, np.ndarray | dict[str, np.ndarray]] = {"target": target}

    past_covariates = {
        name: values[:, idx].astype(np.float32, copy=False)
        for idx, name in enumerate(columns)
        if idx != target_idx
    }
    if past_covariates:
        item["past_covariates"] = past_covariates

    return item


def _build_finetune_inputs(
    dataset: GanymedeDataset,
    validation_ratio: float,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]] | None, list[dict[str, object]]
]:
    train_end_by_well = _derive_train_series_end(dataset)
    train_inputs: list[dict[str, object]] = []
    val_inputs: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    min_series_len = max(dataset.input_window, dataset.horizon + 1)

    for well_idx, (well_name, _df) in enumerate(dataset._well_data):
        train_end = train_end_by_well.get(well_idx)
        if train_end is None or train_end < min_series_len:
            logger.warning(
                "Skipping %s: insufficient pre-holdout history (%s)",
                well_name,
                train_end,
            )
            continue

        series = dataset._arrays[well_idx][:train_end]
        train_series = series
        val_series = None

        if validation_ratio > 0.0:
            val_len = max(dataset.horizon, int(round(train_end * validation_ratio)))
            split_at = train_end - val_len
            if split_at >= min_series_len:
                train_series = series[:split_at]
                val_series = series[split_at - dataset.input_window : train_end]

        train_inputs.append(
            _series_to_chronos_input(
                train_series, dataset._common_columns, dataset._target_col_idx
            )
        )
        if val_series is not None and len(val_series) >= min_series_len:
            val_inputs.append(
                _series_to_chronos_input(
                    val_series, dataset._common_columns, dataset._target_col_idx
                )
            )

        summaries.append(
            {
                "well_name": well_name,
                "train_end_row": train_end,
                "train_rows": len(train_series),
                "val_rows": 0 if val_series is None else len(val_series),
            }
        )

    return train_inputs, (val_inputs or None), summaries


def _predict_test_windows(
    pipeline,
    dataset: GanymedeDataset,
    indices: np.ndarray,
    batch_size: int,
    quantile_level: float,
    cross_learning: bool,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_inputs = []
        batch_targets = []

        for idx in batch_idx:
            features, target, _metadata = dataset[int(idx)]
            batch_inputs.append(
                _series_to_chronos_input(
                    features.numpy(),
                    dataset._common_columns,
                    dataset._target_col_idx,
                )
            )
            batch_targets.append(target.numpy())

        quantiles, _means = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=dataset.horizon,
            quantile_levels=[quantile_level],
            batch_size=len(batch_inputs),
            context_length=dataset.input_window,
            cross_learning=cross_learning,
        )

        batch_preds = np.stack(
            [forecast[0, :, 0].detach().cpu().numpy() for forecast in quantiles],
            axis=0,
        )
        predictions.append(batch_preds)
        targets.append(np.stack(batch_targets, axis=0))

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def _fit_direct_backend(
    model_name: str,
    train_inputs: list[dict[str, object]],
    val_inputs: list[dict[str, object]] | None,
    horizon: int,
    device: str,
    finetune_mode: str,
    learning_rate: float,
    max_steps: int,
    train_batch_size: int,
    output_dir: Path,
    logging_steps: int,
):
    from chronos.chronos2.pipeline import Chronos2Pipeline

    if finetune_mode == "lora" and not _has_module("peft"):
        raise RuntimeError(
            "LoRA fine-tuning requires `peft`, but it is not installed. "
            "Install the updated optional FM dependencies (including peft) and rerun."
        )

    logger.info("Loading %s on %s", model_name, device)
    pipeline = Chronos2Pipeline.from_pretrained(model_name, device_map=device)
    logger.info("Fine-tuning Chronos-2 (%s) for %d steps", finetune_mode, max_steps)

    return pipeline.fit(
        inputs=train_inputs,
        prediction_length=horizon,
        validation_inputs=val_inputs,
        finetune_mode=finetune_mode,
        context_length=2048,
        learning_rate=learning_rate,
        num_steps=max_steps,
        batch_size=train_batch_size,
        output_dir=output_dir,
        finetuned_ckpt_name="fine-tuned-ckpt",
        remove_printer_callback=True,
        disable_data_parallel=True,
        logging_steps=logging_steps,
    )


def run_horizon(args: argparse.Namespace, horizon: int) -> dict[str, object]:
    set_global_seed(42)
    _check_ganymede_data_available()
    dataset = GanymedeDataset(
        "configs/data/ganymede.yaml",
        horizon=horizon,
        mode="multi_well",
        input_window=90,
        filter_shutdowns=False,
    )
    holdout = _make_holdout(dataset)
    train_pool, test_indices = holdout.split(len(dataset))
    support = _get_support_status()

    train_inputs, val_inputs, train_summary = _build_finetune_inputs(
        dataset,
        validation_ratio=args.validation_ratio,
    )
    if not train_inputs:
        raise RuntimeError(
            "No valid Ganymede well histories were available for fine-tuning"
        )

    run_dir = RESULTS_DIR / f"checkpoints_h{horizon}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = _fit_direct_backend(
        model_name=args.model_name,
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        horizon=horizon,
        device=args.device,
        finetune_mode=args.finetune_mode,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        train_batch_size=args.train_batch_size,
        output_dir=run_dir,
        logging_steps=args.logging_steps,
    )

    eval_indices = test_indices
    if args.max_test_samples is not None:
        eval_indices = eval_indices[: args.max_test_samples]

    predictions, targets = _predict_test_windows(
        model,
        dataset,
        eval_indices,
        batch_size=args.eval_batch_size,
        quantile_level=args.quantile_level,
        cross_learning=args.cross_learning,
    )
    test_metrics = MetricRegistry.compute("forecasting", predictions, targets)

    result: dict[str, object] = {
        "backend": "chronos_direct",
        "model_name": args.model_name,
        "dataset": "ganymede",
        "mode": "multi_well",
        "horizon": horizon,
        "support": support,
        "config": {
            "seed": 42,
            "device": args.device,
            "finetune_mode": args.finetune_mode,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "quantile_level": args.quantile_level,
            "cross_learning": args.cross_learning,
            "validation_ratio": args.validation_ratio,
        },
        "n_train_series": len(train_inputs),
        "n_val_series": 0 if val_inputs is None else len(val_inputs),
        "n_train_windows": len(train_pool),
        "n_test_windows": len(test_indices),
        "n_eval_windows": len(eval_indices),
        "train_series_summary": train_summary,
        "test_metrics": test_metrics,
    }

    out_path = RESULTS_DIR / f"ganymede_h{horizon}_multi_well.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_make_serializable(result), indent=2))
    logger.info("Saved %s", out_path)
    logger.info("h=%d test metrics: %s", horizon, test_metrics)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, default=[7])
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--finetune-mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--validation-ratio", type=float, default=0.0)
    parser.add_argument("--quantile-level", type=float, default=0.5)
    parser.add_argument("--cross-learning", action="store_true")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Output root for repaired Chronos fine-tune artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    global RESULTS_DIR
    args = parse_args()
    RESULTS_DIR = resolve_results_dir(args.results_dir, for_write=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = [run_horizon(args, horizon) for horizon in args.horizons]
    summary = {result["horizon"]: result["test_metrics"] for result in all_results}
    print(json.dumps(_make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
