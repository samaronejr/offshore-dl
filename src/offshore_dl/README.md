# src/offshore_dl/

Core Python package.

## Subpackages

### `models/` — All 8 architectures

| Module | Model | Type |
|--------|-------|------|
| `base.py` | `BaseModel` ABC | Interface: `forward()`, `training_step()`, `predict()`, `configure_optimizers()` |
| `lstm.py` | Bidirectional LSTM + attention | Trained (classification, forecasting, anomaly) |
| `deeponet.py` | Time-Dependent DeepONet | Trained (branch-trunk operator learning) |
| `patchtst.py` | PatchTST | Trained (channel-independent transformer) |
| `mlp.py` | MLP | Trained (Flatten→FC→BN→GELU→Dropout baseline) |
| `chronos_wrapper.py` | Chronos-2 | Zero-shot FM (forecasting, anomaly) |
| `timesfm_wrapper.py` | TimesFM 2.5 | Zero-shot FM (forecasting, anomaly) |
| `tirex_wrapper.py` | TiRex | Zero-shot FM (forecasting, anomaly) |
| `tirex_classifier.py` | TiRex + Random Forest | Embedding classifier (3W classification) |
| `dummy.py` | DummyModel | Testing utility |

### `data/` — Data loading and preprocessing

| Module | Description |
|--------|-------------|
| `datasets.py` | `ThreeWDataset`, `ThreeWFeatureDataset`, `GanymedeDataset`, `CDFDataset` |
| `dataloaders.py` | DataLoader factory + `BalancedBatchSampler` for class imbalance |
| `transforms.py` | Pure transforms: frozen-value handling, z-score, EMA lag features |
| `feature_extractor.py` | 14 statistical descriptors per sensor (vectorized) — compresses (720,27) → (14,27) |
| `preprocess_3w.py` | 3W preprocessing pipeline |
| `preprocess_ganymede.py` | Ganymede preprocessing (shutdown filtering, EMA features) |
| `preprocess_cdf.py` | CDF preprocessing |
| `base.py` | Base dataset class |
| `check.py` | Data integrity checks |

### `evaluation/` — Metrics and cross-validation

| Module | Description |
|--------|-------------|
| `metrics.py` | `MetricRegistry` — F1, MAE, RMSE, MASE, R², R²_prod, AUC-PR, EDR |
| `cv.py` | `ExpandingWindowCV`, `StratifiedGroupKFoldCV`, `TemporalSplitCV` + `LeakageGuard` |
| `baselines.py` | Naive baselines: seasonal naive, majority classifier, mean reconstruction |
| `check.py` | Evaluation integrity checks |

### `training/` — Training engine

| Module | Description |
|--------|-------------|
| `trainer.py` | Training loop with EarlyStopping, gradient clipping, CostTracker |
| `experiment.py` | `ExperimentRunner` — nested holdout + inner CV + MLflow logging |
| `optuna_utils.py` | `OptunaObjective` + convergence callback for HPO |

### `analysis/` — Result analysis

| Module | Description |
|--------|-------------|
| `compare.py` | LaTeX table generation, multi-horizon comparison, statistical test integration |

### `utils/` — Shared utilities

| Module | Description |
|--------|-------------|
| `config.py` | OmegaConf hierarchical YAML loader |
| `reproducibility.py` | `set_global_seed()` — locks 7 RNG sources (Python, NumPy, PyTorch CPU/CUDA/cuDNN/CUBLAS) |

## Entry Points

| Module | Description |
|--------|-------------|
| `run_experiment.py` | CLI entrypoint: `python -m offshore_dl.run_experiment --model lstm --dataset ganymede` |
| `__main__.py` | Package runner: `python -m offshore_dl` |
