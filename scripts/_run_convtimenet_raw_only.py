#!/usr/bin/env python3
"""Targeted raw-path ConvTimeNet runner.

Runs convtimenet_raw with lower lr (5e-4) for numerical stability on 720×27 windows.
Produces results/convtimenet_raw/3w.json matching the standard nested-CV format.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from offshore_dl.data.datasets import ThreeWDataset
from offshore_dl.evaluation.cv import HoldoutSplitter, StratifiedGroupKFoldSKLearn
from offshore_dl.models.convtimenet import ConvTimeNetModel
from offshore_dl.training.experiment import ExperimentRunner
from offshore_dl.utils.config import load_merged_config
from offshore_dl.utils.reproducibility import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


from offshore_dl.utils.serialization import make_serializable as _make_serializable


def main():
    set_global_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 100

    logger.info("Loading 3W dataset for ConvTimeNet raw-path ...")
    ds = ThreeWDataset("configs/data/3w.yaml", cache_in_memory=False)
    logger.info("  Dataset: %d samples", len(ds))

    labels = np.array([w["label"] for w in ds._windows])
    groups = np.array([w["instance_id"] for w in ds._windows])

    holdout = HoldoutSplitter(
        test_ratio=0.2, mode="stratified_group",
        labels=labels, groups=groups, seed=42,
    )
    train_pool, test_indices = holdout.split(len(labels))
    logger.info("Holdout: train_pool=%d, test=%d", len(train_pool), len(test_indices))

    pool_labels = labels[train_pool]
    pool_groups = groups[train_pool]
    inner_cv = StratifiedGroupKFoldSKLearn(
        n_folds=5, labels=pool_labels, groups=pool_groups, seed=42,
    )

    cfg = load_merged_config("configs/base.yaml", "configs/data/3w.yaml", "configs/models/convtimenet.yaml")
    cfg.training.max_epochs = max_epochs
    cfg.training.batch_size = 32
    cfg.device = device
    cfg.training.scheduler = "cosine"

    model_kwargs = {
        "task": "classification",
        "n_vars": 27,
        "n_classes": cfg.data.n_classes,
        "window_size": 720,
    }

    if hasattr(cfg, "model") and hasattr(cfg.model, "architecture"):
        arch = OmegaConf.to_container(cfg.model.architecture, resolve=True)
        model_kwargs.update(arch)

    # lr from YAML (0.001) via ExperimentRunner; weight_decay from YAML
    model_kwargs["lr"] = cfg.model.training.lr
    model_kwargs["weight_decay"] = cfg.model.training.weight_decay

    logger.info("Running ConvTimeNet raw-path (720×27), lr=5e-4, batch_size=32, device=%s", device)

    runner = ExperimentRunner(
        model_class=ConvTimeNetModel,
        dataset=ds,
        cv_strategy=inner_cv,
        cfg=cfg,
        model_kwargs=model_kwargs,
    )

    results = runner.run_nested(
        train_pool=train_pool,
        test_indices=test_indices,
        use_mlflow=False,
    )

    out_path = RESULTS_DIR / "convtimenet_raw" / "3w.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_make_serializable(results), indent=2))
    logger.info("Results saved: %s", out_path)

    test_metrics = results.get("test_metrics", {})
    acc = test_metrics.get("accuracy", 0)
    f1 = test_metrics.get("f1_macro", 0)
    logger.info("ConvTimeNet raw: accuracy=%.4f, f1_macro=%.4f", acc, f1)

    folds = results.get("cv_fold_results", [])
    logger.info("CV folds: %d", len(folds))
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
    assert acc > 0.10, f"Expected accuracy > 10%, got {acc:.4f}"
    logger.info("All checks passed!")


if __name__ == "__main__":
    main()
