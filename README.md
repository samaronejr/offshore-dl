# offshore-dl

**Benchmarking Deep Learning for Offshore Production Monitoring**

MSc dissertation — UFRJ/COPPE PEE  
*Production Forecasting and Anomaly Detection on Offshore Platforms Using Deep Learning*

Six architectures. Three datasets. Four paradigms. One reproducible framework.

---

## What This Is

A benchmarking framework that compares six deep learning architectures across three offshore production monitoring tasks — gas production forecasting, supervised fault classification, and unsupervised anomaly detection. Built as a reusable pipeline: plug in a new model or dataset and get the full evaluation stack (cross-validation, metrics, statistical tests, LaTeX tables) for free.

The six models span four architectural paradigms:

| Paradigm | Model | Training | Tasks |
|----------|-------|----------|-------|
| Recurrent baseline | **LSTM** | From scratch | Forecasting · Classification · Anomaly |
| Zero-shot foundation model | **TimesFM 2.5** | Pretrained (no fine-tuning) | Forecasting · Anomaly |
| Zero-shot foundation model | **Chronos-2** | Pretrained (no fine-tuning) | Forecasting · Anomaly |
| Zero-shot foundation model | **TiRex** | Pretrained (xLSTM backbone) | Forecasting · Anomaly |
| Fine-tuned foundation model | **PatchTST** | From scratch* | Forecasting · Classification · Anomaly |
| Neural operator | **Time-Dependent DeepONet** | From scratch | Forecasting · Classification · Anomaly |

<sub>*PatchTST uses the HuggingFace architecture but trains from scratch — pretrained weights are incompatible with the input dimensions of these datasets.</sub>

## Datasets

| Dataset | Source | Task | Scale |
|---------|--------|------|-------|
| **3W v2.0.0** | Petrobras | 10-class fault classification | 2,228 instances · 27 sensors · 1.8 GB |
| **Ganymede** | NSTA (UK Continental Shelf) | Gas production forecasting | 7 wells · ~49k daily rows · 25 columns |
| **CDF** | Cognite Data Fusion | Unsupervised anomaly detection | 1 compressor · 4,367 hourly rows · 12 sensors |

## Results at a Glance

### Ganymede — Gas Production Forecasting (h=7d, nested holdout test, R²_prod ↑)

| Model | R²_prod | MAE |
|-------|:-------:|:---:|
| **TimesFM** | **0.826** | 0.740 |
| **TiRex** | 0.822 | **0.658** |
| Chronos-2 | 0.734 | 0.739 |
| PatchTST | 0.556 | 1.772 |
| LSTM | 0.195 | 1.772 |
| DeepONet | −40.1 | 9.525 |

R²_prod is R² computed on productive windows only (excluding well shutdowns). Gas production measured in MMSCF (million standard cubic feet). Foundation models dominate — TimesFM and TiRex achieve R²>0.82 zero-shot. Nested evaluation: 80/20 temporal holdout, 3-fold ExpandingWindowCV inner loop. Friedman test significant on MAE at h7/h14/h30 (p<0.03).

### 3W — Fault Classification (nested holdout test, Accuracy ↑)

| Model | Method | Accuracy | F1-macro | AUC-PR |
|-------|--------|:--------:|:--------:|:------:|
| **DeepONet** | Feature extraction + fine-tune | **96.83%** | **0.964** | 0.934 |
| **LSTM** | Feature extraction + fine-tune | 96.80% | 0.963 | 0.933 |
| **PatchTST†** | Feature extraction + HPO | 96.47% | 0.962 | 0.931 |
| **TiRex** | Frozen embeddings + RF | 91.16% | 0.895 | **0.938** |

†PatchTST improved from 95.72% to 96.47% after Optuna HPO (30 trials). LSTM and DeepONet showed no improvement — manual tuning was near-optimal.

Feature extraction: 14 statistical descriptors per sensor compress each (720, 27) window into (14, 27). TiRex uses frozen xLSTM embeddings + Random Forest classifier. Nested evaluation: 80/20 stratified-group holdout, 5-fold StratifiedGroupKFold inner loop. Friedman test significant (p<0.014).


---

## Project Structure

```
offshore-dl/
├── src/offshore_dl/           # Main package
│   ├── models/                # All 6 architectures
│   │   ├── base.py            #   BaseModel ABC — common interface
│   │   ├── lstm.py            #   LSTM (classification + forecasting + anomaly)
│   │   ├── deeponet.py        #   Time-Dependent DeepONet (branch-trunk)
│   │   ├── patchtst.py        #   PatchTST via HuggingFace Transformers
│   │   ├── chronos_wrapper.py #   Chronos-2 zero-shot wrapper
│   │   ├── timesfm_wrapper.py #   TimesFM 2.5 zero-shot wrapper
│   │   ├── tirex_wrapper.py   #   TiRex zero-shot wrapper
│   │   └── tirex_classifier.py #  TiRex embedding + RF classifier
│   ├── data/                  # Data loading and preprocessing
│   │   ├── datasets.py        #   ThreeWDataset, ThreeWFeatureDataset, GanymedeDataset, CDFDataset
│   │   ├── dataloaders.py     #   DataLoader factory + BalancedBatchSampler
│   │   ├── transforms.py      #   Pure transforms (frozen values, z-score, EMA, …)
│   │   ├── feature_extractor.py #  14 statistical features per sensor (vectorized)
│   │   ├── preprocess_3w.py   #   3W preprocessing pipeline
│   │   ├── preprocess_ganymede.py
│   │   └── preprocess_cdf.py
│   ├── training/              # Training engine
│   │   ├── trainer.py         #   Trainer loop + EarlyStopping + CostTracker
│   │   ├── experiment.py      #   ExperimentRunner (CV + MLflow + metrics)
│   │   └── optuna_utils.py    #   HPO: OptunaObjective + convergence callback
│   ├── evaluation/            # Metrics and cross-validation
│   │   ├── metrics.py         #   MetricRegistry (F1, MAE, MASE, R², AUC-PR, EDR, …)
│   │   ├── cv.py              #   ExpandingWindowCV, StratifiedGroupKFoldCV, TemporalSplitCV
│   │   └── baselines.py       #   Naive baselines (seasonal naive, majority, mean reconstruction)
│   ├── analysis/
│   │   └── compare.py         #   LaTeX table generation + statistical tests
│   ├── utils/
│   │   ├── reproducibility.py #   set_global_seed() — locks 7 RNG sources
│   │   └── config.py          #   OmegaConf hierarchical YAML loader
│   └── run_experiment.py      #   CLI entrypoint
│
├── configs/                   # Hierarchical YAML configuration
│   ├── base.yaml              #   Global: seed, device, MLflow, Optuna, training
│   ├── data/
│   │   ├── 3w.yaml            #   3W dataset config (27 sensors, 10 classes)
│   │   ├── ganymede.yaml      #   Ganymede config (7 wells, target column, horizons)
│   │   └── cdf.yaml           #   CDF config (12 sensors, anomaly task)
│   └── models/
│       ├── lstm.yaml           #   Architecture + Optuna search space
│       ├── deeponet.yaml
│       └── patchtst.yaml
│
├── scripts/                   # Entrypoints and infrastructure
│   ├── run_production_ganymede.py  # Full sweep: 6 models × 4 horizons × 8 targets
│   ├── run_production_3w.py        # 3 models on full 3W (208k samples, raw windows)
│   ├── run_production_3w_features.py # 3 models + TiRex on 3W with feature extraction
│   ├── run_optuna_hpo.py            # Optuna HPO: 3W + Ganymede, 15 trials/model
│   ├── run_statistical_tests.py     # Friedman + Nemenyi + Wilcoxon significance tests
│   ├── run_production_cdf.py        # 6 models on CDF anomaly detection
│   ├── extract_tirex_embeddings.py  # TiRex memmap embedding extraction
│   ├── run_tirex_rf_folds.py        # TiRex RF per-fold evaluation
│   ├── run_tirex_rf_nested.py       # TiRex nested holdout evaluation
│   ├── archive/                     # Superseded baseline scripts
│   ├── docker_run.sh               # Docker convenience wrapper
│   ├── singularity_run.sh          # Singularity/Apptainer runner
│   ├── push_to_registry.sh         # Push image to GHCR (+ air-gapped fallback)
│   ├── nacad_job.slurm             # NACAD HPC Slurm job (Ganymede, 4h)
│   └── nacad_job_3w.slurm          # NACAD HPC Slurm job (3W, 24h)
│
├── docker/
│   ├── Dockerfile             # Multi-stage: CUDA 12.4 + Python 3.11 + FM deps
│   ├── docker-compose.yml     # train (GPU) + mlflow server
│   └── .env.template
│
├── tests/                     # ~245 tests (8 skip for FM deps)
│   ├── test_analysis.py       #   LaTeX generation, statistical tests, multi-horizon
│   ├── test_cv.py             #   CV causality, leakage guard, fold integrity
│   ├── test_metrics.py        #   Metric correctness on degenerate/edge cases
│   ├── test_transforms.py     #   All transform functions
│   ├── test_datasets.py       #   Dataset loading, shapes, windowing
│   ├── test_lstm.py           #   LSTM forward pass, all 3 tasks
│   ├── test_deeponet.py       #   DeepONet branch-trunk, classification/forecasting
│   ├── test_patchtst.py       #   PatchTST integration
│   ├── test_zero_shot.py      #   Chronos, TimesFM, TiRex wrappers
│   ├── test_training.py       #   Trainer, ExperimentRunner, MLflow logging
│   ├── test_baselines.py      #   Naive baseline correctness
│   ├── test_reproducibility.py#   Seed determinism verification
│   └── …
│
├── data/
│   ├── raw/                   # Original data (gitignored)
│   │   ├── 3w/                #   Petrobras 3W v2.0.0 parquets
│   │   ├── NSTA_datasets/     #   Ganymede daily production CSV
│   │   └── CogniteDataFusion/ #   CDF compressor sensor CSV
│   └── processed/             # Preprocessed parquets (gitignored)
│       ├── 3w/
│       ├── ganymede/
│       └── cdf/
│
├── results/                   # Model outputs (JSON)
│   ├── lstm/                  #   ganymede.json, 3w.json, cdf.json, h* files
│   ├── deeponet/
│   ├── patchtst/
│   ├── chronos/
│   ├── timesfm/
│   ├── tirex/
│   └── baselines/             #   Seasonal naive, majority, mean reconstruction
│
├── reports/                   # Publication output
│   ├── ganymede_comparison.tex              # 6-model comparison (COPPE LaTeX)
│   ├── 3w_comparison.tex                    # 3-model comparison
│   ├── cdf_comparison.tex                   # 6-model comparison
│   ├── ganymede_multihorizon_comparison.tex # 7d / 14d / 30d / 90d MAE
│   ├── ganymede_perwell_comparison.tex      # Per-well vs multi-well
│   ├── statistical_tests.json               # Friedman + Wilcoxon p-values
│   └── comparison_summary.md                # Human-readable overview
│
├── dvc.yaml               # DVC pipeline (9 stages: preprocess → train → compare)
├── pyproject.toml          # Package metadata and dependencies
└── mlruns/                 # MLflow tracking (local filesystem)
```

---

## Quick Start

### Local (CPU)

```bash
# Clone and install
git clone https://github.com/<user>/offshore-dl.git
cd offshore-dl
pip install -e ".[dev]"

# Preprocess data (assumes raw data is in data/raw/)
python -m offshore_dl.data.preprocess_3w
python -m offshore_dl.data.preprocess_ganymede
python -m offshore_dl.data.preprocess_cdf

# Run a single experiment
python -m offshore_dl.run_experiment --model lstm --dataset ganymede --max-epochs 5

# Run tests
pytest tests/ -v
```

### Docker (GPU)

```bash
# Build the image (CUDA 12.4, Python 3.11, all FM dependencies)
docker build -t offshore-dl:train -f docker/Dockerfile --target train .

# Run an experiment on GPU
scripts/docker_run.sh python -m offshore_dl.run_experiment \
    --model lstm --dataset ganymede --device cuda

# Run the full Ganymede production sweep (6 models × 4 horizons × 8 targets)
scripts/docker_run.sh python scripts/run_production_ganymede.py --device cuda

# Run tests inside the container
scripts/docker_run.sh pytest tests/ --tb=short
```

### HPC (Singularity/Apptainer + Slurm)

```bash
# Push image to GHCR (or use air-gapped tarball)
scripts/push_to_registry.sh

# On the cluster
singularity pull docker://ghcr.io/<user>/offshore-dl:train
sbatch scripts/nacad_job.slurm        # Ganymede sweep (4h)
sbatch scripts/nacad_job_3w.slurm     # 3W full training (24h)
```

### Regenerate Publication Tables

```bash
python -m offshore_dl.analysis.compare --results-dir results --output-dir reports
```

Produces 7 files in `reports/` — ready to `\input{}` directly in the COPPE/UFRJ LaTeX template.

---

## How It Works

### Architecture

Every model inherits from `BaseModel` and implements three methods:

```python
class BaseModel(nn.Module, ABC):
    def training_step(self, batch) -> Tensor:    # → scalar loss
    def predict(self, batch) -> Tensor:           # → predictions
    def configure_optimizers(self, cfg) -> Optimizer:
```

The `Trainer` owns the training loop. Models never touch it — they just compute losses and predictions. Zero-shot foundation models return `loss = 0.0` from `training_step` (no-op) and run inference in `predict`.

### Cross-Validation

Each dataset uses the CV strategy that matches its data structure:

| Dataset | CV Strategy | Rationale |
|---------|-------------|-----------|
| Ganymede | `ExpandingWindowCV` | Temporal walk-forward; training set grows monotonically |
| 3W | `TemporalSplitCV` / `StratifiedGroupKFoldCV` | Temporal split for production; group K-fold ensures no well leaks across folds |
| CDF | `TemporalSplitCV` | Single temporal split (one compressor) |

A `LeakageGuard` validates every split — no future data in training, no well overlap.

Z-score normalization is computed **from the training partition only**, per fold. No data leakage.

### Configuration

Hierarchical YAML via OmegaConf:

```
configs/base.yaml          # Global defaults (seed, device, training params)
  ← configs/data/3w.yaml   # Dataset-specific overrides
  ← configs/models/lstm.yaml  # Model-specific overrides
  ← CLI --key=value         # Runtime overrides
```

Every hyperparameter is config-driven. No magic numbers in code.

### Experiment Tracking

- **MLflow** — nested runs (parent = experiment, children = CV folds). Logs all params, metrics, and artifacts.
- **Optuna** — HPO with `MedianPruner` and a convergence callback that stops after 20 trials without improvement.
- **DVC** — `dvc.yaml` defines the full pipeline from preprocessing to final comparison tables.

### Statistical Rigor

- **Friedman test** across all 6 models on Ganymede fold-level MAE → χ² = 11.76, p = 0.038
- **Wilcoxon signed-rank** pairwise comparisons between every model pair
- Tracks with insufficient CV folds gracefully report `status: insufficient_folds` instead of crashing
- All boolean `significant` fields are native Python `bool` (not stringified numpy)

---

## Adding a New Model

1. Create `src/offshore_dl/models/my_model.py` inheriting `BaseModel`
2. Implement `training_step`, `predict`, `configure_optimizers`
3. Add a YAML config at `configs/models/my_model.yaml` with architecture params and Optuna search space
4. Register it in `src/offshore_dl/run_experiment.py` → `MODEL_REGISTRY`
5. Run: `python -m offshore_dl.run_experiment --model my_model --dataset ganymede`

The entire evaluation pipeline (CV, metrics, MLflow, comparison tables) applies automatically.

## Adding a New Dataset

1. Write a preprocessing script at `src/offshore_dl/data/preprocess_*.py`
2. Create a `Dataset` class in `datasets.py` inheriting `BaseDataset`
3. Add a YAML config at `configs/data/my_dataset.yaml`
4. Register it in `run_experiment.py` → `DATASET_REGISTRY` with the appropriate CV factory
5. Run: `python -m offshore_dl.run_experiment --model lstm --dataset my_dataset`

---

## Reproducibility

Every run is deterministic:

- `set_global_seed(42)` locks Python `random`, NumPy, PyTorch CPU, CUDA, cuDNN, and CUBLAS
- All configuration in YAML — no hardcoded values
- DVC pipeline tracks every stage dependency
- Per-fold normalization prevents data leakage
- Results are JSON with full fold-level detail, not just aggregates

To reproduce the full pipeline from a clean clone:

```bash
docker build -t offshore-dl:train -f docker/Dockerfile --target train .
# Place raw data in data/raw/
dvc repro
```

---

## Key Dependencies

| Package | Role |
|---------|------|
| PyTorch ≥ 2.3 | Core DL framework |
| OmegaConf | Hierarchical YAML config |
| MLflow ≥ 2.14 | Experiment tracking |
| Optuna ≥ 3.6 | Hyperparameter optimization |
| scikit-learn ≥ 1.4 | Metrics, preprocessing utilities |
| HuggingFace Transformers ≥ 4.40 | PatchTST architecture |
| timesfm ≥ 1.3 | Google TimesFM 2.5 |
| chronos-forecasting ≥ 1.3 | Amazon Chronos-2 |
| tirex-ts | NX-AI TiRex (xLSTM backbone) |

Foundation model dependencies are in the optional `[fm]` extra — the core package runs without them.

---

## Citation

```bibtex
@mastersthesis{santos2026offshore,
  title   = {Production Forecasting and Anomaly Detection on Offshore
             Platforms Using Deep Learning},
  author  = {Santos Júnior, Samarone Lima},
  school  = {COPPE/UFRJ — Programa de Engenharia Elétrica},
  year    = {2026},
  address = {Rio de Janeiro, Brazil}
}
```

## License

MIT
