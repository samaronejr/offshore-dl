# src/offshore_dl/

Core Python package for offshore production-monitoring benchmarks.

## Subpackages

### `models/` — architectures and wrappers

| Module | Model | Type |
|--------|-------|------|
| `base.py` | `BaseModel` ABC | Interface: `forward()`, `training_step()`, `predict()`, `configure_optimizers()` |
| `lstm.py` | Bidirectional LSTM + attention | Trained model for classification, forecasting, anomaly reconstruction |
| `deeponet.py` | Time-Dependent DeepONet | Branch-trunk forecasting/anomaly model plus classification heads/variants |
| `patchtst.py` | PatchTST | Trained channel-independent transformer |
| `tcn.py` | Temporal Convolutional Network | Trained dilated causal convolution model |
| `convtimenet.py` | ConvTimeNet | Trained multi-scale convolution classifier |
| `convtran.py` | ConvTran | Trained convolution + attention classifier |
| `inception_time.py` | InceptionTime | Trained inception-block classifier |
| `mambasl.py` | MambaSL | Trained selective state-space model |
| `fkmad.py` | FKMAD | Frequency-kernel multi-scale model |
| `hydra_rocket.py` | Hydra + ROCKET | Training-free/random-kernel feature pipeline |
| `chronos_wrapper.py` | Chronos-2 | Zero-shot foundation-model wrapper |
| `timesfm_wrapper.py` | TimesFM 2.5 | Zero-shot foundation-model wrapper |
| `tirex_wrapper.py` | TiRex | Zero-shot xLSTM foundation-model wrapper |
| `tirex_classifier.py` | TiRex + Random Forest | Embedding classifier for 3W classification |
| `moment_wrapper.py` | MOMENT | Optional foundation-model wrapper |
| `mantis_wrapper.py` | MANTIS | Optional foundation-model wrapper |
| `dummy.py` | DummyModel | Testing utility |

Trained models inherit `BaseModel`. Zero-shot FMs use wrapper classes that expose the same high-level prediction contract but do not perform gradient training. Random Forest and Hydra/ROCKET use sklearn-style dispatch in production scripts.

### `data/` — data loading and preprocessing

| Module | Description |
|--------|-------------|
| `datasets.py` | `ThreeWDataset`, `ThreeWFeatureDataset`, `GanymedeDataset`, `CDFDataset`, `SPEBergDataset`, `VolveDataset`, `InnerMongoliaDataset` |
| `dataloaders.py` | DataLoader factory plus `BalancedBatchSampler` for class imbalance |
| `transforms.py` | Frozen-value handling, z-score, EMA lag features |
| `feature_extractor.py` | Statistical descriptors per sensor for 3W feature matrices |
| `preprocess_3w.py` | 3W preprocessing pipeline |
| `preprocess_ganymede.py` | Ganymede preprocessing with shutdown filtering and EMA features |
| `preprocess_cdf.py` | CDF preprocessing |
| `preprocess_spe_berg.py` | SPE Berg preprocessing |
| `preprocess_volve.py` | Volve preprocessing |
| `preprocess_inner_mongolia.py` | Inner Mongolia preprocessing |
| `base.py` | Base dataset class |
| `check.py` | Data integrity checks |

### `evaluation/` — metrics and cross-validation

| Module | Description |
|--------|-------------|
| `metrics.py` | `MetricRegistry` — F1, MAE, RMSE, grouped MASE, R², R²_prod, AUC-PR, EDR |
| `cv.py` | Temporal, grouped, stratified-group, expanding-window, sliding-window, holdout, and leakage-guard splitters |
| `baselines.py` | Naive baselines: seasonal naive, majority classifier, mean reconstruction |
| `check.py` | Evaluation integrity checks |

Metric caveats for current reports:

- 3W classification uses macro-F1 as the headline HPO metric.
- Ganymede forecasting should report MAE/RMSE and MASE separately because foundation models lead by absolute error while trained LSTM/TCN lead by grouped MASE.
- R² and R²_prod are useful diagnostics but unstable across wells.

### `training/` — training and HPO

| Module | Description |
|--------|-------------|
| `trainer.py` | Training loop with early stopping, gradient clipping, and cost tracking |
| `experiment.py` | `ExperimentRunner` — nested holdout + inner CV + MLflow logging + JSON output |
| `optuna_utils.py` | `OptunaObjective` and convergence callbacks for HPO |
| `supcon.py` | Supervised contrastive loss for imbalanced classification |

The current 3W HPO workflow requires final held-out evaluation before a model result is accepted into `summary.json`; see `scripts/validate_hpo_3w_results.py`.

### `analysis/` — result analysis

| Module | Description |
|--------|-------------|
| `compare.py` | LaTeX table generation, multi-horizon comparison, statistical-test integration |

When regenerating reports, keep 3W Stage 1 standard models separate from Stage 2 feature/window variants.

### `utils/` — shared utilities

| Module | Description |
|--------|-------------|
| `config.py` | OmegaConf hierarchical YAML loader with `load_merged_config` |
| `reproducibility.py` | `set_global_seed()` — locks Python, NumPy, PyTorch CPU/CUDA/cuDNN/CUBLAS sources |
| `serialization.py` | JSON serialization helpers for result persistence |

## Entry points

| Module | Description |
|--------|-------------|
| `run_experiment.py` | CLI entry point: `python -m offshore_dl.run_experiment --model lstm --dataset ganymede` |
| `__main__.py` | Package runner: `python -m offshore_dl` |

## Config loading

```python
from offshore_dl.utils.config import load_merged_config
cfg = load_merged_config(model="lstm", dataset="3w")
```

## Result-output contract

- Production runs write JSON artifacts with test metrics, CV aggregates, split metadata, predictions/targets where applicable, and runtime adjustments.
- Fixed-code production reruns should go under `results/post_fix/` unless a command intentionally overrides the output directory.
- HPO campaigns write to `results/hpo/<dataset>/<campaign-id>/`; a campaign is not a final benchmark result unless final evaluation metrics are present and validation passes.
