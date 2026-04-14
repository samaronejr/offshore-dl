"""Compute 3W ensemble metrics from per-sample production outputs.

Usage::

    python scripts/ensemble_3w.py
    python scripts/ensemble_3w.py --models random_forest convtimenet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from offshore_dl.evaluation.metrics import MetricRegistry

DEFAULT_MODELS = ["random_forest", "convtimenet", "fkmad", "mambasl"]
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "ensemble" / "3w_ensemble.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _to_numpy_1d(values: list | None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    return arr.astype(np.int64).reshape(-1)


def _to_numpy_2d(values: list | None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or arr.ndim != 2:
        return None
    return arr


def _load_model_outputs(model_name: str) -> tuple[dict | None, str | None]:
    result_path = RESULTS_DIR / model_name / "3w.json"
    if not result_path.exists():
        return None, f"missing file: {result_path}"

    payload = _load_json(result_path)
    predictions = _to_numpy_1d(payload.get("test_predictions"))
    probabilities = _to_numpy_2d(payload.get("test_probabilities"))
    targets = _to_numpy_1d(payload.get("test_targets"))

    if predictions is None and probabilities is not None:
        predictions = probabilities.argmax(axis=1).astype(np.int64)

    if targets is None:
        return None, f"missing test_targets in {result_path}"
    if predictions is None:
        return None, f"missing test_predictions/test_probabilities in {result_path}"
    if len(predictions) != len(targets):
        return None, f"prediction/target length mismatch in {result_path}"
    if probabilities is not None and len(probabilities) != len(targets):
        return None, f"probability/target length mismatch in {result_path}"

    return {
        "model_name": model_name,
        "result_path": str(result_path),
        "predictions": predictions,
        "probabilities": probabilities,
        "targets": targets,
        "test_metrics": payload.get("test_metrics", {}),
    }, None


def _resolve_class_labels(model_outputs: list[dict], targets: np.ndarray) -> np.ndarray:
    n_classes = int(targets.max()) + 1
    for output in model_outputs:
        probs = output["probabilities"]
        if probs is not None:
            n_classes = max(n_classes, probs.shape[1])
    return np.arange(n_classes, dtype=np.int64)


def _one_hot(predictions: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
    scores = np.zeros((len(predictions), len(class_labels)), dtype=np.float64)
    scores[np.arange(len(predictions)), predictions.astype(np.int64)] = 1.0
    return scores


def _vote_scores(model_outputs: list[dict], class_labels: np.ndarray) -> np.ndarray:
    score_mats = [
        _one_hot(output["predictions"], class_labels) for output in model_outputs
    ]
    return np.mean(score_mats, axis=0)


def _majority_vote(
    model_outputs: list[dict], class_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vote_scores = _vote_scores(model_outputs, class_labels)
    predictions = vote_scores.argmax(axis=1).astype(np.int64)
    return predictions, vote_scores


def _soft_vote(model_outputs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    prob_outputs = [
        output["probabilities"]
        for output in model_outputs
        if output["probabilities"] is not None
    ]
    mean_probs = np.mean(prob_outputs, axis=0)
    predictions = mean_probs.argmax(axis=1).astype(np.int64)
    return predictions, mean_probs


def _stacking_vote(
    model_outputs: list[dict],
    targets: np.ndarray,
    class_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    meta_features = np.concatenate(
        [output["probabilities"] for output in model_outputs], axis=1
    )
    class_counts = np.unique(targets.astype(np.int64), return_counts=True)[1]
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError("stacking requires at least 2 samples per class")
    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(targets), len(class_labels)), dtype=np.float64)

    meta_params = {
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced_subsample",
    }

    for train_idx, val_idx in cv.split(meta_features, targets):
        clf = RandomForestClassifier(**meta_params)
        clf.fit(meta_features[train_idx], targets[train_idx])
        fold_probs = clf.predict_proba(meta_features[val_idx])
        oof_probs[val_idx, clf.classes_.astype(int)] = fold_probs

    predictions = oof_probs.argmax(axis=1).astype(np.int64)
    return predictions, oof_probs, {"cv_splits": n_splits, "meta_model": meta_params}


def _compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict:
    return MetricRegistry.compute(
        "classification",
        predictions,
        targets,
        prediction_scores=scores,
    )


def _print_summary(method_name: str, metrics: dict) -> None:
    print(
        f"{method_name:>14}: "
        f"f1_macro={metrics['f1_macro']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"auc_pr={metrics['auc_pr']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    loaded_outputs: list[dict] = []
    skipped: list[dict[str, str]] = []
    reference_targets: np.ndarray | None = None

    for model_name in args.models:
        output, error = _load_model_outputs(model_name)
        if error is not None:
            skipped.append({"model_name": model_name, "reason": error})
            continue

        targets = output["targets"]
        if reference_targets is None:
            reference_targets = targets
        elif not np.array_equal(reference_targets, targets):
            skipped.append(
                {
                    "model_name": model_name,
                    "reason": "test_targets mismatch vs reference",
                }
            )
            continue

        loaded_outputs.append(output)

    if reference_targets is None or not loaded_outputs:
        raise SystemExit("No valid 3W model outputs found.")

    class_labels = _resolve_class_labels(loaded_outputs, reference_targets)
    probability_outputs = [o for o in loaded_outputs if o["probabilities"] is not None]
    methods: dict[str, dict] = {}

    results: dict[str, object] = {
        "dataset": "3w",
        "requested_models": args.models,
        "models_loaded": [o["model_name"] for o in loaded_outputs],
        "models_with_probabilities": [o["model_name"] for o in probability_outputs],
        "skipped_models": skipped,
        "n_samples": int(len(reference_targets)),
        "class_labels": class_labels.tolist(),
        "methods": methods,
    }

    majority_preds, majority_scores = _majority_vote(loaded_outputs, class_labels)
    majority_metrics = _compute_metrics(
        majority_preds, reference_targets, majority_scores
    )
    methods["majority_vote"] = {
        "source_models": [o["model_name"] for o in loaded_outputs],
        "test_metrics": majority_metrics,
    }

    if probability_outputs:
        soft_preds, soft_scores = _soft_vote(probability_outputs)
        soft_metrics = _compute_metrics(soft_preds, reference_targets, soft_scores)
        methods["soft_vote"] = {
            "source_models": [o["model_name"] for o in probability_outputs],
            "test_metrics": soft_metrics,
        }

    if len(probability_outputs) >= 2:
        stack_preds, stack_scores, stack_meta = _stacking_vote(
            probability_outputs,
            reference_targets,
            class_labels,
        )
        stack_metrics = _compute_metrics(stack_preds, reference_targets, stack_scores)
        methods["stacking_rf"] = {
            "source_models": [o["model_name"] for o in probability_outputs],
            "test_metrics": stack_metrics,
            "meta": stack_meta,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Loaded models: {', '.join(results['models_loaded'])}")
    if skipped:
        print("Skipped models:")
        for item in skipped:
            print(f"  - {item['model_name']}: {item['reason']}")
    print(f"Saved ensemble results to {OUTPUT_PATH}")

    for method_name, payload in methods.items():
        _print_summary(method_name, payload["test_metrics"])


if __name__ == "__main__":
    main()
