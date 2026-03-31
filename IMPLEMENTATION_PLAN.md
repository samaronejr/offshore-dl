# Implementation Plan — MSc Dissertation UFRJ/COPPE PEE

## Production Forecasting and Anomaly Detection on Offshore Platforms Using Deep Learning

**Author:** Samarone Lima Santos Júnior  
**Advisor:** Prof. Natanael Nunes de Moura Júnior, D.Sc.  
**Program:** PEE — COPPE/UFRJ

---

## Project Overview

This project implements an MLOps-driven experimental pipeline to benchmark six deep learning architectures across two tasks in offshore hydrocarbon production monitoring:

| Task | Dataset | Applicable Models | Primary Metric |
|------|---------|-------------------|----------------|
| Anomaly detection/classification | 3W v2.0.0 (Petrobras) + PI System via CDF | LSTM, PatchTST-FM, DeepONet (TD) | F1-macro |
| Gas production forecasting | NSTA Ganymede (UKCS Daily Production Data) | LSTM, TimesFM 2.5, Chronos-2, TiRex, PatchTST-FM, DeepONet (TD) | MAE (scaled) |

The six models span four architectural paradigms:

- **Recurrent baseline** (task-specific, trained from scratch): LSTM
- **Foundation models — zero-shot** (pretrained, no fine-tuning): TimesFM 2.5, Chronos-2, TiRex
- **Foundation model — fine-tuned** (pretrained + supervised adaptation): PatchTST-FM
- **Neural operator** (trained from scratch): Time-Dependent DeepONet

---

## Target Directory Structure

```
offshore-dl-thesis/
├── configs/                        # Hierarchical YAML configs
│   ├── base.yaml                   # Global: seeds, paths, device, MLflow, Optuna
│   ├── data/
│   │   ├── 3w.yaml                 # 3W dataset config
│   │   └── ganymede.yaml           # Ganymede dataset config
│   └── models/
│       ├── lstm.yaml               # Hyperparams + Optuna search space
│       ├── timesfm.yaml
│       ├── chronos2.yaml
│       ├── tirex.yaml
│       ├── patchtst_fm.yaml
│       └── deeponet_td.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest_3w.py            # Download + DVC versioning for 3W
│   │   ├── ingest_ganymede.py      # Download + DVC versioning for NSTA
│   │   ├── preprocess_3w.py        # Anomaly detection preprocessing pipeline
│   │   ├── preprocess_ganymede.py  # Gas forecasting preprocessing pipeline
│   │   ├── datasets.py             # PyTorch Dataset classes
│   │   ├── dataloaders.py          # DataLoader factory + BalancedBatchSampler
│   │   └── transforms.py           # Pure (stateless) transformation functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract BaseModel interface
│   │   ├── lstm.py                 # LSTM classifier + forecaster
│   │   ├── timesfm_wrapper.py      # TimesFM 2.5 zero-shot wrapper
│   │   ├── chronos2_wrapper.py     # Chronos-2 zero-shot wrapper
│   │   ├── tirex_wrapper.py        # TiRex zero-shot wrapper
│   │   ├── patchtst_fm.py          # PatchTST-FM fine-tuning
│   │   └── deeponet_td.py          # Time-Dependent DeepONet
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Generic training loop
│   │   ├── optuna_objective.py     # Optuna objective function
│   │   └── cross_validation.py     # ExpandingWindowCV + StratifiedGroupKFoldCV
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics_anomaly.py      # F1-macro, AUC-PR, EDR, confusion matrix
│   │   ├── metrics_forecast.py     # MAE, MASE, CRPS, R²
│   │   └── reporting.py            # Report generation + publication-quality plots
│   └── utils/
│       ├── __init__.py
│       ├── reproducibility.py      # Global seed fixing, deterministic config
│       ├── mlflow_utils.py         # MLflow helper functions
│       └── config.py               # Config loader (OmegaConf / YAML)
├── tests/
│   ├── test_transforms.py
│   ├── test_datasets.py
│   ├── test_cross_validation.py
│   └── test_metrics.py
├── scripts/
│   ├── run_pipeline.py             # Main entry point (full pipeline)
│   ├── run_optuna_study.py         # Launch Optuna study for a single model
│   ├── run_final_eval.py           # Final holdout evaluation
│   └── generate_report.py          # Comparative report generation
├── docker/
│   ├── Dockerfile                  # Multi-stage image
│   ├── docker-compose.yml          # train + mlflow-server + optuna-db
│   └── .env.template
├── data/
│   ├── raw/                        # Raw data (gitignored, DVC-tracked)
│   ├── processed/                  # Preprocessed Parquet files
│   └── splits/                     # Persisted CV split indices
├── mlruns/                         # MLflow tracking (Docker volume)
├── requirements.txt                # Pinned dependencies
├── pyproject.toml
├── .dvc/
├── .gitignore
├── dvc.yaml                        # DVC pipeline definition
└── README.md
```

---

## PHASE 0 — Infrastructure and Reproducibility

**Objective:** Establish the project foundation — Docker, MLflow, deterministic seeds, DVC — before touching any data or model code.

### Step 0.1 — Repository Scaffolding

```
ACTIONS:
1. Create the full directory structure above (all __init__.py, .gitignore, pyproject.toml).
2. Initialize Git + DVC.
3. Create requirements.txt with pinned versions:
   - python==3.11
   - torch>=2.3,<2.5
   - numpy>=1.26,<2.0
   - pandas>=2.2
   - pyarrow>=15.0
   - scikit-learn>=1.4
   - mlflow>=2.14
   - optuna>=3.6
   - properscoring>=0.1
   - omegaconf>=2.3
   - pytest>=8.0
   - matplotlib>=3.9
   - seaborn>=0.13
   # Foundation model deps (install in Docker model layer):
   - timesfm>=2.5          # Google TimesFM
   - chronos-forecasting>=2.0   # Amazon Chronos-2
   - tirex                  # TiRex (xLSTM backbone)
   - transformers>=4.40     # PatchTST via HuggingFace
```

### Step 0.2 — Reproducibility Module (`src/utils/reproducibility.py`)

```python
# Implement set_global_seed(seed: int) that locks down:
#   - random.seed(seed)
#   - np.random.seed(seed)
#   - torch.manual_seed(seed)
#   - torch.cuda.manual_seed_all(seed)
#   - torch.backends.cudnn.deterministic = True
#   - torch.backends.cudnn.benchmark = False
#   - os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#   - torch.use_deterministic_algorithms(True)
# Default seed: 42 (overridable via YAML config)
```

### Step 0.3 — Hierarchical Configuration (`src/utils/config.py`)

```
ACTIONS:
1. Use OmegaConf to load and merge YAML configs.
2. Create configs/base.yaml:
   seed: 42
   device: "cuda"
   paths:
     raw_data: "data/raw"
     processed_data: "data/processed"
     splits: "data/splits"
   mlflow:
     tracking_uri: "http://localhost:5000"
     experiment_prefix: "offshore-dl"
   optuna:
     storage: "sqlite:///optuna.db"
     n_trials_min: 50
     n_trials_max: 200
     convergence_patience: 20
     convergence_threshold: 0.005
```

### Step 0.4 — Docker

```
ACTIONS:
1. Create docker/Dockerfile (multi-stage build):
   Stage 1 (base): nvidia/cuda:12.4-cudnn9-runtime-ubuntu22.04
     - Python 3.11, PyTorch 2.x (CUDA), scientific stack
   Stage 2 (models): extends stage 1
     - timesfm, chronos-2, tirex, transformers, mlflow, optuna
     - Download pretrained checkpoints at build time (no runtime downloads)

2. Create docker/docker-compose.yml with 3 services:
   - train:
       build: .
       deploy.resources.reservations.devices: [driver: nvidia, count: all]
       volumes: [./data:/app/data, ./mlruns:/app/mlruns, ./configs:/app/configs]
       environment: [MLFLOW_TRACKING_URI, OPTUNA_STORAGE]
   - mlflow:
       image: ghcr.io/mlflow/mlflow:latest
       ports: ["5000:5000"]
       command: >
         mlflow server
         --backend-store-uri sqlite:///mlflow.db
         --default-artifact-root /mlruns
         --host 0.0.0.0
       volumes: [./mlruns:/mlruns]
   - optuna-db:
       image: postgres:16-alpine     # alternatively SQLite via volume mount
       volumes: [optuna-data:/var/lib/postgresql/data]
```

### Step 0.5 — MLflow Helpers (`src/utils/mlflow_utils.py`)

```
ACTIONS:
1. Implement:
   - setup_experiment(problem: str, model_name: str) -> experiment_id
     # Creates or retrieves experiment named "{prefix}/{problem}/{model_name}"
   - log_config(config: DictConfig) -> None
     # Flattens nested YAML and logs all params via mlflow.log_params
   - log_fold_metrics(fold: int, metrics: dict) -> None
   - log_model_artifact(model, model_name: str) -> None
   - create_child_run(parent_run_id: str, trial_number: int) -> context_manager
```

**Phase 0 acceptance criteria:** `docker compose up` brings up all 3 services; MLflow UI accessible at `localhost:5000`; `pytest tests/ -k "test_reproducibility"` confirms identical outputs across runs with the same seed.

---

## PHASE 1 — Data Ingestion and Preprocessing

### Step 1.1 — 3W Ingestion (`src/data/ingest_3w.py`)

```
ACTIONS:
1. Download 3W v2.0.0 from the Petrobras GitHub repository:
   - URL: https://github.com/petrobras/3W (clone or download ZIP)
   - Extract CSV/Parquet instance files
   - Save to data/raw/3w/
2. Build metadata index: list all instances with class, well_id, source type
   (real / simulated / hand-drawn)
3. Version with DVC: dvc add data/raw/3w/
4. Generate descriptive statistics: instance count per class, per well, per source
```

### Step 1.2 — NSTA Ganymede Ingestion (`src/data/ingest_ganymede.py`)

```
ACTIONS:
1. Download Ganymede field data from the NSTA portal:
   - URL: https://experience.arcgis.com/experience/50b61d215bff4072bf0649efe6e8d845
   - Filter UKCS Daily Production Data by field name = "GANYMEDE"
   - Download CSV with daily data for all wellbores
2. Save to data/raw/ganymede/
3. Version with DVC: dvc add data/raw/ganymede/
4. Generate descriptive statistics: temporal coverage, number of wellbores,
   variable availability, missingness (%) per variable, gas rate distribution
```

### Step 1.3 — Pure Transform Functions (`src/data/transforms.py`)

Each function is stateless (DataFrame in → DataFrame out) and logged as metadata in MLflow.

```python
# ===================== 3W-specific transforms =====================

def detect_frozen_values(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Detect frozen sensor values via rolling variance (window in seconds).
    Zero variance → frozen → replace with NaN."""

def causal_forward_fill(df: pd.DataFrame, limit_seconds: int = 300) -> pd.DataFrame:
    """Causal forward-fill with a hard temporal limit.
    Gaps exceeding limit_seconds remain NaN."""

def sliding_window_segmentation(
    df: pd.DataFrame, w: int, s: int, label_col: str
) -> list[dict]:
    """Segment time series into windows of size w with stride s.
    Label = majority class within each window.
    Returns list of {start, end, label, well_id}."""

def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Inverse-frequency class weights for the loss function."""

# ===================== Ganymede-specific transforms =====================

def detect_shutdowns(
    df: pd.DataFrame, zero_days_threshold: int = 2
) -> pd.DataFrame:
    """Flag periods where gas_rate == 0 for >= threshold consecutive days.
    Adds binary column 'is_shutdown'."""

def compute_ema_features(
    df: pd.DataFrame, windows: list[int] = [7, 14, 30, 90]
) -> pd.DataFrame:
    """Compute exponential moving averages at each window size (in days)."""

def compute_rate_of_change(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """First-order numerical derivative (diff / dt) for specified columns."""

def compute_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute domain-specific ratios: gas-water ratio, BHP/WHP ratio, etc."""

def log_transform(
    df: pd.DataFrame, cols: list[str], eps: float = 1e-8
) -> pd.DataFrame:
    """x' = log(x + eps) for heavily right-skewed variables."""

# ===================== Shared transforms =====================

def compute_zscore_stats(df_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute per-variable mean and std from TRAINING DATA ONLY.
    Returns {variable_name: (mean, std)}."""

def apply_zscore(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply z-score normalization using precomputed statistics (no leakage)."""
```

### Step 1.4 — 3W Preprocessing Pipeline (`src/data/preprocess_3w.py`)

```
ACTIONS:
1. Implement sequential pipeline:
   raw_3w → detect_frozen_values → causal_forward_fill
         → save intermediate Parquet

   At training time (per fold, invoked by the trainer):
     → compute_zscore_stats(train_fold)
     → apply_zscore(train_fold, val_fold, test_fold)
     → sliding_window_segmentation(w, s)
     → compute_class_weights(labels)

2. Save Parquet files partitioned by well_id to data/processed/3w/
3. Generate preprocessing report:
   - % frozen values detected and replaced
   - % NaN before/after forward-fill
   - Class distribution after segmentation
```

### Step 1.5 — Ganymede Preprocessing Pipeline (`src/data/preprocess_ganymede.py`)

```
ACTIONS:
1. Implement sequential pipeline:
   raw_ganymede → detect_shutdowns → compute_ema_features
               → compute_rate_of_change → compute_derived_ratios
               → log_transform(gas_rate)

   At training time (per fold, invoked by the trainer):
     → compute_zscore_stats(train_fold)
     → apply_zscore(train_fold, val_fold, test_fold)

2. Save Parquet files partitioned by wellbore_id to data/processed/ganymede/
3. Generate preprocessing report:
   - Shutdown periods identified (dates, durations)
   - Per-wellbore statistics (temporal range, missingness)
   - Gas rate distribution (pre- and post-log-transform)
```

### Step 1.6 — Unit Tests (`tests/test_transforms.py`)

```
ACTIONS:
1. Test each transform with synthetic data:
   - detect_frozen_values: constant series → NaN; varying series → unchanged
   - causal_forward_fill: gap < limit → filled; gap > limit → NaN preserved
   - compute_zscore_stats + apply_zscore: result has mean ≈ 0, std ≈ 1
   - sliding_window_segmentation: correct window count; labels consistent
   - detect_shutdowns: 3 zero-days → flagged; 1 zero-day → not flagged
   - log_transform: x=0 → log(eps); x>0 → log(x + eps)
2. Verify NO transform function leaks future information (causality check)
```

**Phase 1 acceptance criteria:** `data/processed/3w/` and `data/processed/ganymede/` contain valid Parquet files; all tests in `tests/test_transforms.py` pass; DVC tracks raw data artifacts.

---

## PHASE 2 — High-Throughput Data Loading

### Step 2.1 — PyTorch Datasets (`src/data/datasets.py`)

```python
class AnomalyWindowDataset(torch.utils.data.Dataset):
    """Dataset for the 3W anomaly detection task.

    - Lazy loading: precomputes window indices (start, end, label, well_id)
      in __init__; reads actual data from Parquet on __getitem__
    - cache_in_memory: bool — if True, caches all data on first epoch
      (3W fits in RAM at ~5 GB in Parquet format)
    - Returns: (window_tensor [w, n_vars], label: int, well_id: str)
    """

class ForecastDataset(torch.utils.data.Dataset):
    """Dataset for the Ganymede gas production forecasting task.

    - Sliding window: input [t-w : t], target [t+1 : t+h]
    - Lazy loading with optional in-memory caching
    - gap parameter: number of timesteps between last input and first target
    - Returns: (input_tensor [w, n_vars], target_tensor [h], metadata: dict)
    """
```

### Step 2.2 — DataLoader Factory + BalancedSampler (`src/data/dataloaders.py`)

```python
class BalancedBatchSampler(torch.utils.data.Sampler):
    """Ensures each batch contains representatives of multiple event classes.
    Implements round-robin or proportional class sampling.
    Used exclusively for the anomaly detection task."""

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    balanced: bool = False,         # Activates BalancedBatchSampler
    num_workers: int = 4,
    pin_memory: bool = True,        # Async CPU→GPU via DMA
    prefetch_factor: int = 2,       # Keep future batches queued
    persistent_workers: bool = True  # Avoid worker process respawn per epoch
) -> DataLoader:
    """Factory with defaults optimized for GPU throughput."""
```

### Step 2.3 — Tests (`tests/test_datasets.py`)

```
ACTIONS:
1. AnomalyWindowDataset with a small synthetic Parquet:
   - __len__ correct given w, s
   - Output tensor shapes match (w, n_vars)
   - In-memory cache: second iteration measurably faster
2. ForecastDataset:
   - No overlap between input window and target window
   - Gap correctly inserted when configured
3. BalancedBatchSampler:
   - Each batch contains at least 2 distinct event classes
```

**Phase 2 acceptance criteria:** DataLoaders iterate without errors on real 3W and Ganymede data; GPU utilization > 80% during iteration (verify with `nvidia-smi`).

---

## PHASE 3 — Temporal Cross-Validation

### Step 3.1 — Expanding Window CV (`src/training/cross_validation.py`)

```python
class ExpandingWindowCV:
    """Walk-forward validation for gas production forecasting.

    Parameters:
        n_folds: int = 5
        initial_train_size: int or timedelta   # T_0 >= 6 months of data
        step_size: int or timedelta             # Delta_T = 1 month
        gap: int                                 # g >= h (forecast horizon)

    Method:
        split(timestamps: pd.DatetimeIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]

    Fold k semantics:
        train: t <= T_0 + k * Delta_T
        test:  T_0 + k * Delta_T + gap < t <= T_0 + (k+1) * Delta_T + gap

    The training set grows monotonically across folds, simulating operational
    retraining with accumulated historical data.
    """
```

### Step 3.2 — Stratified GroupKFold CV (`src/training/cross_validation.py`)

```python
class StratifiedGroupKFoldCV:
    """Well-wise sliding window + class stratification for 3W anomaly detection.

    Guarantees:
        1. No well_id appears in both train and test (inter-well generalization)
        2. All event classes (1-8) are represented in each training fold
        3. Temporal ordering is preserved within each well's instances

    Uses sklearn's GroupKFold as backbone with group = well_id,
    plus a post-hoc class-presence check that resamples if any class is missing.

    Method:
        split(instances, labels, groups) -> Iterator[tuple[np.ndarray, np.ndarray]]
    """
```

### Step 3.3 — Split Persistence

```
ACTIONS:
1. On split generation, persist indices to data/splits/:
   - anomaly_fold_{k}_train.npy, anomaly_fold_{k}_test.npy
   - forecast_fold_{k}_train.npy, forecast_fold_{k}_test.npy
2. Compute SHA-256 hash of underlying data to verify integrity on reload
3. Log to MLflow as parameters: n_folds, T_0, Delta_T, gap, seed
```

### Step 3.4 — Tests (`tests/test_cross_validation.py`)

```
ACTIONS:
1. ExpandingWindowCV:
   - Training set size is strictly monotonically increasing across folds
   - No test point appears in any training set (strict causality)
   - Gap between last training timestamp and first test timestamp >= h
2. StratifiedGroupKFoldCV:
   - No well_id is shared between train and test within any fold
   - Every event class (1-8) is present in every training fold
   - Intra-well temporal ordering is preserved
```

**Phase 3 acceptance criteria:** All causality and integrity tests pass; splits are saved to disk and reproducible with the same seed.

---

## PHASE 4 — Model Implementation

**Design constraint:** All models inherit from `BaseModel` and implement `fit()`, `predict()`, and optionally `predict_quantiles()`.

### Step 4.1 — Abstract Base (`src/models/base.py`)

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Common interface for all six architectures."""

    @abstractmethod
    def fit(self, train_loader, val_loader, config: dict) -> dict:
        """Train the model. Returns dict of validation metrics."""

    @abstractmethod
    def predict(self, dataloader) -> np.ndarray:
        """Return point predictions."""

    def predict_quantiles(self, dataloader, quantiles: list[float]) -> np.ndarray:
        """Return probabilistic predictions (optional override)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support quantile prediction"
        )

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
```

### Step 4.2 — LSTM Baseline (`src/models/lstm.py`)

```
ACTIONS:
1. Implement LSTMModel(BaseModel):
   - Core: nn.LSTM with configurable hidden_size, num_layers, dropout
   - Task heads:
     * Regression head (Linear → forecast) for gas production
     * Classification head (Linear → 9-class softmax) for anomaly detection
   - fit(): training loop with early stopping on val loss
   - Must call trial.report(val_metric, epoch) at each epoch for Optuna's
     MedianPruner; check trial.should_prune()
   - Loss: MSE for forecasting; CrossEntropy with class weights for anomalies

2. Optuna search space (configs/models/lstm.yaml):
   search_space:
     hidden_size: {type: int, low: 64, high: 512}
     num_layers: {type: int, low: 1, high: 4}
     learning_rate: {type: log_float, low: 1e-5, high: 1e-2}
     dropout: {type: float, low: 0.0, high: 0.5}
     batch_size: {type: categorical, choices: [32, 64, 128]}
```

### Step 4.3 — TimesFM 2.5 Wrapper (`src/models/timesfm_wrapper.py`)

```
ACTIONS:
1. Implement TimesFMWrapper(BaseModel):
   - Load pretrained checkpoint: timesfm-2.5-200m
   - fit() = no-op (zero-shot inference only); configure context_length
   - predict() via timesfm.forecast()
   - predict_quantiles() via the native quantile head
   - FORECASTING TASK ONLY — not applicable to anomaly classification

2. Optuna search space:
   search_space:
     context_length: {type: int, low: 128, high: 16384}
     normalize_inputs: {type: categorical, choices: [true, false]}
     quantile_head: {type: categorical, choices: [true, false]}
```

### Step 4.4 — Chronos-2 Wrapper (`src/models/chronos2_wrapper.py`)

```
ACTIONS:
1. Implement Chronos2Wrapper(BaseModel):
   - Load via transformers: amazon/chronos-bolt-base (or chronos-2-*)
   - fit() = no-op (zero-shot)
   - predict() and predict_quantiles() via the native Chronos pipeline
   - Supports univariate and multivariate (group attention mechanism)
   - FORECASTING TASK ONLY

2. Optuna search space:
   search_space:
     prediction_length: {type: int, low: 12, high: 256}
     quantile_levels: {type: categorical, choices: [3, 5, 9]}
     num_samples: {type: int, low: 10, high: 100}
```

### Step 4.5 — TiRex Wrapper (`src/models/tirex_wrapper.py`)

```
ACTIONS:
1. Implement TiRexWrapper(BaseModel):
   - Load pretrained xLSTM checkpoint (tirex-35m)
   - fit() = no-op (zero-shot)
   - predict() with native missing-data handling (masking mechanism)
   - predict_quantiles() via the 9 native quantile outputs
   - FORECASTING TASK ONLY

2. Optuna search space:
   search_space:
     context_length: {type: int, low: 64, high: 4096}
     prediction_length: {type: int, low: 12, high: 512}
```

### Step 4.6 — PatchTST-FM with Fine-Tuning (`src/models/patchtst_fm.py`)

```
ACTIONS:
1. Implement PatchTSTFM(BaseModel):
   - Load ibm-granite/granite-timeseries-patchtst (patchtst-fm-r1)
   - fit() performs supervised fine-tuning:
     * Freeze encoder for the first N epochs, then unfreeze
     * Attach task-specific head:
       - Classification head (9-class) for anomaly detection
       - Regression head for gas production forecasting
   - predict() and predict_proba() for both tasks
   - APPLICABLE TO BOTH TASKS

2. Optuna search space:
   search_space:
     learning_rate: {type: log_float, low: 1e-6, high: 1e-3}
     finetune_epochs: {type: int, low: 5, high: 50}
     patch_length: {type: int, low: 8, high: 64}
     batch_size: {type: categorical, choices: [16, 32, 64]}
     freeze_epochs: {type: int, low: 0, high: 10}
```

### Step 4.7 — Time-Dependent DeepONet (`src/models/deeponet_td.py`)

```
ACTIONS:
1. Implement TDDeepONet(BaseModel):
   - Branch network: encodes sensor history (configurable: MLP or LSTM)
   - Trunk network: MLP encoding temporal output query locations
   - Output: dot product of branch and trunk representations
   - fit() trains from scratch (not pretrained)
   - Adaptable to both regression (forecasting) and classification (anomalies)
   - APPLICABLE TO BOTH TASKS

2. Optuna search space:
   search_space:
     branch_width: {type: int, low: 64, high: 256}
     trunk_width: {type: int, low: 64, high: 256}
     branch_depth: {type: int, low: 3, high: 8}
     learning_rate: {type: log_float, low: 1e-5, high: 1e-3}
     branch_type: {type: categorical, choices: ["mlp", "lstm"]}
```

**Phase 4 acceptance criteria:** Each model instantiates, trains (or runs zero-shot inference), and produces predictions on a synthetic mini-batch without errors. Output shapes are correct for both tasks.

---

## PHASE 5 — Evaluation Metrics

### Step 5.1 — Anomaly Detection Metrics (`src/evaluation/metrics_anomaly.py`)

```python
def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1-score across all 9 classes. PRIMARY METRIC."""

def auc_pr_macro(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Per-class AUC-PR, then macro average. Requires class probabilities."""

def confusion_matrix_normalized(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 9
) -> np.ndarray:
    """Row-normalized 9×9 confusion matrix."""

def early_detection_rate(
    y_true: np.ndarray, y_pred: np.ndarray, transient_mask: np.ndarray
) -> float:
    """Fraction of anomalies correctly detected during the transient phase
    (before the steady-state fault regime). transient_mask is a boolean array
    indicating which timestamps belong to the transient phase."""
```

### Step 5.2 — Forecasting Metrics (`src/evaluation/metrics_forecast.py`)

```python
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. PRIMARY METRIC."""

def mase(
    y_true: np.ndarray, y_pred: np.ndarray,
    y_train: np.ndarray, seasonal_period: int
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006).
    MASE < 1 → better than seasonal naïve; MASE > 1 → worse."""

def crps(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantile_levels: list[float]
) -> float:
    """Continuous Ranked Probability Score for probabilistic forecasts.
    Uses properscoring library or manual numerical approximation."""

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. SECONDARY METRIC."""
```

### Step 5.3 — Tests (`tests/test_metrics.py`)

```
ACTIONS:
1. f1_macro: perfect predictions → 1.0; random predictions → ~1/9
2. mase: seasonal naïve predictor → MASE = 1.0; perfect predictor → MASE = 0.0
3. crps: degenerate distribution at correct value → CRPS ≈ 0
4. early_detection_rate: 100% detection in transient → EDR = 1.0; 0% → EDR = 0.0
5. r_squared: identical pred/true → 1.0; constant prediction → R² depends on variance
```

**Phase 5 acceptance criteria:** All unit tests pass; metrics produce numerically correct values for known degenerate and edge cases.

---

## PHASE 6 — Orchestration: Optuna + MLflow + Cross-Validation

### Step 6.1 — Optuna Objective Function (`src/training/optuna_objective.py`)

```python
def create_objective(
    model_class: type[BaseModel],
    problem: str,                    # "anomaly" or "forecast"
    data_config: DictConfig,
    model_config: DictConfig,
    cv_splitter,                     # ExpandingWindowCV or StratifiedGroupKFoldCV
    parent_run_id: str               # MLflow parent run ID
) -> Callable[[optuna.Trial], float]:
    """Returns a callable objective for Optuna.

    Internal flow per trial:
    1. Sample hyperparameters from trial (defined in model_config.search_space)
    2. For each fold k in cv_splitter:
       a. Build train/val Datasets with per-fold z-score normalization
       b. Instantiate model with sampled hyperparameters
       c. Train model; report intermediate metric per epoch to trial
          (enables MedianPruner pruning)
       d. Evaluate on validation fold
       e. Log all metrics to MLflow (nested child run under parent_run_id)
    3. Return MEDIAN of primary metric across all K folds
       (median provides robustness against outlier folds)
    """
```

### Step 6.2 — Optuna Study Script (`scripts/run_optuna_study.py`)

```python
"""
Usage: python scripts/run_optuna_study.py --model lstm --problem forecast

Flow:
1. Load and merge configs (base + data + model) via OmegaConf
2. set_global_seed(config.seed)
3. Create or resume Optuna study:
   study = optuna.create_study(
       study_name=f"{model}_{problem}",
       storage=config.optuna.storage,
       sampler=TPESampler(seed=config.seed),
       pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
       direction="maximize" if problem == "anomaly" else "minimize"
   )
4. Create parent run in MLflow
5. study.optimize(
       objective,
       n_trials=config.optuna.n_trials_max,
       callbacks=[convergence_callback]
   )
6. Log best trial params and metrics to MLflow parent run
"""
```

### Step 6.3 — Convergence Callback

```python
def convergence_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Stops the study if the last 20 completed trials show < 0.5% improvement
    over the global best. Implements early stopping at the study level."""
    patience = study.user_attrs.get("patience", 20)
    threshold = study.user_attrs.get("threshold", 0.005)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed) < patience:
        return

    recent_values = [t.value for t in completed[-patience:]]
    best_recent = (min if study.direction == StudyDirection.MINIMIZE else max)(recent_values)
    best_overall = study.best_value
    relative_improvement = abs(best_recent - best_overall) / (abs(best_overall) + 1e-10)

    if relative_improvement < threshold:
        study.stop()
```

**Phase 6 acceptance criteria:** A complete Optuna study (LSTM + forecast + 5 trials minimum) runs end-to-end; MLflow shows parent run with nested child runs per trial; MedianPruner prunes at least 1 unpromising trial.

---

## PHASE 7 — Final Evaluation and Reporting

### Step 7.1 — Final Holdout Evaluation (`scripts/run_final_eval.py`)

```
FLOW:
1. For each (model, problem) combination:
   a. Load best hyperparameters from the Optuna study
   b. Retrain the model on the full train+validation data (all folds merged)
   c. Evaluate on the holdout test set (NEVER seen during optimization)
   d. Compute all metrics (primary + secondary)
   e. Generate artifacts: predictions array, diagnostic plots, confusion matrices
   f. Register the final model in MLflow Model Registry with:
      - SHA-256 hash of training data
      - Full config YAML
      - Seed used
2. Persist all results to a structured comparative table
```

### Step 7.2 — Report Generation (`scripts/generate_report.py`)

```
ACTIONS:
1. Comparative LaTeX tables (6 models × metrics):
   - Anomaly detection: F1-macro, AUC-PR, EDR per model (with std across folds)
   - Gas forecasting: MAE, MASE, CRPS, R² per model (with std across folds)
   - Bold best result per metric

2. Publication-quality figures:
   - Forecast vs. actual time series overlay (per model, per wellbore)
   - 9×9 row-normalized confusion matrices (one per model)
   - Box plots of fold-level metrics (visualize cross-validation variance)
   - Optuna convergence curves (best objective value vs. trial number)
   - Grouped bar chart: all 6 models side-by-side per metric

3. Export as:
   - MLflow artifacts (attached to the final evaluation run)
   - Standalone PDF report
   - Individual PNG/PDF figures for LaTeX inclusion in the dissertation
```

### Step 7.3 — Full Pipeline Script (`scripts/run_pipeline.py`)

```
ACTIONS:
1. Master entry point that chains all phases:
   a. Ingestion (skip if raw data exists)
   b. Preprocessing (skip if processed Parquets exist)
   c. For each (model, problem):
      - Optuna hyperparameter optimization
      - Final holdout evaluation
   d. Comparative report generation

2. CLI flags:
   --model {lstm,timesfm,chronos2,tirex,patchtst_fm,deeponet_td,all}
   --problem {anomaly,forecast,all}
   --skip-optuna          # Use existing best params
   --eval-only            # Skip training entirely, just evaluate
   --n-trials N           # Override max trials
```

**Phase 7 acceptance criteria:** Final comparative table generated with metrics for all 6 models across both tasks (where applicable); all figures render at publication quality (300 DPI, vector PDF).

---

## PHASE 8 — DVC Pipeline (End-to-End Reproducibility)

### Step 8.1 — dvc.yaml

```yaml
stages:
  ingest_3w:
    cmd: python src/data/ingest_3w.py
    deps: [src/data/ingest_3w.py]
    outs: [data/raw/3w/]

  ingest_ganymede:
    cmd: python src/data/ingest_ganymede.py
    deps: [src/data/ingest_ganymede.py]
    outs: [data/raw/ganymede/]

  preprocess_3w:
    cmd: python src/data/preprocess_3w.py
    deps:
      - src/data/preprocess_3w.py
      - src/data/transforms.py
      - data/raw/3w/
      - configs/data/3w.yaml
    outs: [data/processed/3w/]

  preprocess_ganymede:
    cmd: python src/data/preprocess_ganymede.py
    deps:
      - src/data/preprocess_ganymede.py
      - src/data/transforms.py
      - data/raw/ganymede/
      - configs/data/ganymede.yaml
    outs: [data/processed/ganymede/]

  # One stage per (model × problem) — example:
  optuna_lstm_forecast:
    cmd: >
      python scripts/run_optuna_study.py
      --model lstm --problem forecast
    deps:
      - src/
      - data/processed/ganymede/
      - configs/
    outs: [mlruns/]
    plots: [results/lstm_forecast_convergence.json]

  optuna_lstm_anomaly:
    cmd: >
      python scripts/run_optuna_study.py
      --model lstm --problem anomaly
    deps:
      - src/
      - data/processed/3w/
      - configs/
    outs: [mlruns/]

  # ... (repeat for all model × problem combinations)

  final_eval:
    cmd: python scripts/run_final_eval.py
    deps: [src/, mlruns/]
    outs: [results/]
    metrics: [results/metrics.json]

  report:
    cmd: python scripts/generate_report.py
    deps: [results/]
    outs: [results/report/]
```

---

## Execution Order

```
Phase 0  [~2 days]   Infrastructure (Docker, MLflow, seeds, configs)
  │
  ▼
Phase 1  [~3 days]   Data ingestion + preprocessing (3W + Ganymede)
  │
  ▼
Phase 2  [~1 day]    DataLoaders + BalancedBatchSampler
  │
  ├──────────────────────────┐
  ▼                          ▼
Phase 3  [~1 day]          Phase 5  [~1 day]
Cross-validation             Metrics
(ExpandingWindow +           (can run in parallel
 GroupKFold)                  with Phases 3-4)
  │                          │
  ▼                          │
Phase 4  [~5 days]           │
Model implementation         │
(~1 day per model;           │
 LSTM first as smoke test)   │
  │                          │
  ├──────────────────────────┘
  ▼
Phase 6  [~2 days]   Orchestration (Optuna + MLflow integration)
  │
  ▼
Phase 7  [~2 days]   Final evaluation + report generation
  │
  ▼
Phase 8  [~1 day]    DVC pipeline (end-to-end reproducibility)
```

**Estimated total: ~18 development days**

---

## Model × Task Applicability Matrix

| Model | Anomaly Detection | Gas Forecasting | Training Mode |
|-------|:-:|:-:|---|
| LSTM | ✅ | ✅ | From scratch |
| TimesFM 2.5 | ❌ | ✅ | Zero-shot |
| Chronos-2 | ❌ | ✅ | Zero-shot |
| TiRex | ❌ | ✅ | Zero-shot |
| PatchTST-FM | ✅ | ✅ | Fine-tuned |
| DeepONet (TD) | ✅ | ✅ | From scratch |

---

## Critical Invariants for Claude Code

These rules must never be violated. They are non-negotiable constraints derived from the dissertation methodology.

1. **Seed every entry point.** Every script must call `set_global_seed(config.seed)` before any computation. This includes Optuna objective functions (each trial must reset the seed deterministically).

2. **Never use `sklearn.model_selection.KFold` with `shuffle=True` on time series data.** Only the custom temporal CV classes from Phase 3 are permitted.

3. **Z-score normalization is computed exclusively on the training partition of each fold.** This applies to all models except zero-shot foundation models (TimesFM, Chronos-2, TiRex), which use their internal normalization. Violation constitutes data leakage.

4. **Zero-shot foundation models have no `fit()`.** TimesFM 2.5, Chronos-2, and TiRex perform inference only. Their `fit()` method is a no-op that configures context/prediction length. They are evaluated only on the forecasting task, never on anomaly classification.

5. **PatchTST-FM is the only foundation model with fine-tuning.** It is applicable to both tasks (forecasting and anomaly detection).

6. **DeepONet (TD) is trained from scratch.** It is not pretrained. It is applicable to both tasks.

7. **LSTM is the baseline.** Trained from scratch, applicable to both tasks, serves as the reference point for all comparisons.

8. **Optuna's MedianPruner requires intermediate metric reporting.** Every training loop must call `trial.report(metric, epoch)` and check `trial.should_prune()` at each epoch. Failure to implement this eliminates the computational savings from pruning.

9. **All intermediate and output data in Parquet format with Snappy compression.** Never CSV for processed data. Parquet enables columnar reads and predicate pushdown.

10. **All configuration in YAML, never hardcoded.** Use OmegaConf for hierarchical merge (base.yaml ← data/*.yaml ← models/*.yaml ← CLI overrides).
