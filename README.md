# offshore-dl

**Benchmarking Deep Learning for Offshore Production Monitoring**

MSc dissertation — UFRJ/COPPE PEE  
*Production Forecasting and Anomaly Detection on Offshore Platforms Using Deep Learning*

---

## Overview

This repository contains the code and experimental results for my MSc dissertation. It compares eight architectures for three offshore production monitoring tasks: gas production forecasting, fault classification, and anomaly detection.

The models cover different approaches:

| Paradigm | Model | Training | Tasks |
|----------|-------|----------|-------|
| Recurrent | **LSTM** | From scratch | Forecasting · Classification · Anomaly |
| Neural operator | **Time-Dependent DeepONet** | From scratch | Forecasting · Classification · Anomaly |
| Transformer | **PatchTST** | From scratch* | Forecasting · Classification · Anomaly |
| Feedforward baseline | **MLP** | From scratch | Classification |
| Gradient boosted trees | **XGBoost** | sklearn-style | Forecasting |
| Zero-shot foundation model | **TimesFM 2.5** | Pretrained (no fine-tuning) | Forecasting · Anomaly |
| Zero-shot foundation model | **Chronos-2** | Pretrained (no fine-tuning) | Forecasting · Anomaly |
| Zero-shot foundation model | **TiRex** | Pretrained (xLSTM backbone) | Forecasting · Classification · Anomaly |

<sub>*PatchTST uses the HuggingFace architecture but trains from scratch — pretrained weights are incompatible with the input dimensions of these datasets.</sub>

## Datasets

| Dataset | Source | Task | Scale |
|---------|--------|------|-------|
| **3W v2.0.0** | Petrobras | 10-class fault classification | 208,973 windows · 27 sensors · 1.8 GB |
| **Ganymede** | NSTA (UK Continental Shelf) | Gas production forecasting | 7 wells · 31,359 samples · 63 features |
| **CDF** | Cognite Data Fusion | Unsupervised anomaly detection | 1 compressor · 4,367 hourly rows · 12 sensors |

## Results at a Glance

### 3W — Fault Classification (held-out test, n=41,515)

| Model | Method | Accuracy | F1-macro | F1-weighted | AUC-PR |
|-------|--------|:--------:|:--------:|:-----------:|:------:|
| **DeepONet** | Feature extraction + fine-tune | **96.85%** | **0.964** | **0.969** | 0.934 |
| **LSTM** | Feature extraction + fine-tune | 96.69% | 0.962 | 0.967 | 0.931 |
| **PatchTST†** | Feature extraction + HPO | 96.47% | 0.962 | 0.965 | 0.931 |
| **TiRex** | Frozen embeddings + RF | 91.16% | 0.895 | 0.911 | **0.938** |
| MLP* | Flatten + feedforward | 6.49% | 0.012 | 0.008 | 0.100 |

†PatchTST improved from 95.7% to 96.5% after 30-trial Optuna HPO. LSTM and DeepONet showed no improvement — manual tuning was near-optimal. MLP's collapse to single-class prediction proves the discriminative information resides in the spatial arrangement of the (14×27) feature matrix, not the raw values.

*Historical; MLP model code removed in M005 — result retained for reference only and not reproducible from the current codebase.

### Ganymede — Gas Production Forecasting (held-out test, multi-well, MMSCF)

| Model | R²_prod h=7 | R²_prod h=90 | MAE h=7 |
|-------|:-----------:|:------------:|:-------:|
| **TimesFM** | **0.826** | 0.413 | 0.740 |
| **TiRex** | 0.822 | **0.565** | **0.658** |
| Chronos-2 | 0.734 | 0.361 | 0.739 |
| PatchTST | 0.574 | −0.718 | 1.771 |
| LSTM | 0.327 | −0.464 | 1.768 |
| XGBoost | −0.031 | −1.252 | 1.928 |
| DeepONet | −5.716 | −14.93 | 4.836 |

Foundation models outperform trained models at all horizons. TiRex shows the least degradation with increasing horizon (R²_prod 0.82→0.57). XGBoost performs better than DeepONet but worse than the neural sequence models. The production forecasting scripts now use a **per-well grouped 80/20 temporal holdout plus grouped 3-fold ExpandingWindowCV** for multi-well evaluation; regenerate benchmark tables after methodology changes to refresh the published numbers.

### CDF — Anomaly Detection (held-out test, reconstruction error ↓)

| Model | Error Mean | Error P95 |
|-------|:----------:|:---------:|
| **LSTM** | **0.005** | **0.009** |
| PatchTST | 0.061 | 0.146 |
| DeepONet | 0.219 | 0.449 |
| Chronos | 24.81 | 60.92 |
| TimesFM | 24.88 | 64.89 |
| TiRex | 24.88 | 63.19 |

Trained models perform much better here — LSTM reconstruction error is ~5000× lower than FMs. The foundation models were not designed for this type of reconstruction task.

---

## Project Structure

```
offshore-dl/
├── src/offshore_dl/           # Main package
│   ├── models/                # All 8 architectures
│   │   ├── base.py            #   BaseModel ABC — common interface
│   │   ├── lstm.py            #   Bidirectional LSTM + attention
│   │   ├── deeponet.py        #   Time-Dependent DeepONet (branch-trunk)
│   │   ├── patchtst.py        #   PatchTST via HuggingFace Transformers
│   │   ├── mlp.py             #   MLP baseline (Flatten→FC→BN→GELU→Dropout)
│   │   ├── chronos_wrapper.py #   Chronos-2 zero-shot wrapper
│   │   ├── timesfm_wrapper.py #   TimesFM 2.5 zero-shot wrapper
│   │   ├── tirex_wrapper.py   #   TiRex zero-shot wrapper
│   │   └── tirex_classifier.py #  TiRex embedding + RF classifier
│   ├── data/                  # Data loading and preprocessing
│   ├── training/              # Trainer, ExperimentRunner, Optuna HPO
│   ├── evaluation/            # MetricRegistry, CV strategies, baselines
│   ├── analysis/              # LaTeX generation, statistical tests
│   └── utils/                 # Config, reproducibility (seed 7 RNG sources)
│
├── configs/                   # Hierarchical YAML (OmegaConf)
│   ├── base.yaml              #   Global defaults
│   ├── data/                  #   Dataset configs (3w, ganymede, cdf)
│   └── models/                #   Model configs + Optuna search spaces
│
├── scripts/                   # Production sweeps, HPO, statistical tests
│   ├── run_production_ganymede.py  # 7 models × 4 horizons × 8 targets
│   ├── run_production_3w_features.py # Trained models + TiRex on 3W
│   ├── run_optuna_hpo.py      # 30-trial Bayesian HPO
│   ├── run_statistical_tests.py # Friedman/Nemenyi/Wilcoxon
│   └── ...                    # Slurm, Docker, Singularity scripts
│
├── docker/                    # Dockerfile (CUDA 12.4), docker-compose
├── tests/                     # ~250 tests
├── results/                   # JSON outputs per model per dataset
├── reports/                   # LaTeX tables, PDFs, statistical tests
├── dvc.yaml                   # DVC pipeline definition
└── pyproject.toml             # Package metadata and dependencies
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/<user>/offshore-dl.git
cd offshore-dl
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run a single experiment
python -m offshore_dl.run_experiment --model lstm --dataset ganymede --max-epochs 5

# Full production sweep (GPU recommended)
python scripts/run_production_ganymede.py --device cuda
python scripts/run_production_3w_features.py --device cuda

# Statistical tests
python scripts/run_statistical_tests.py
```

### Docker (GPU)

```bash
docker build -t offshore-dl:train -f docker/Dockerfile --target train .
scripts/docker_run.sh python scripts/run_production_ganymede.py --device cuda
```

### HPC (Singularity + Slurm)

```bash
sbatch scripts/hpc_job.slurm        # Ganymede sweep
sbatch scripts/hpc_job_3w.slurm     # 3W full training
```

---

## How It Works

### Architecture

Every trained model inherits from `BaseModel` and implements `training_step`, `predict`, and `configure_optimizers`. Zero-shot FMs return `loss = 0.0` from `training_step`. XGBoost uses sklearn-style `MultiOutputRegressor(XGBRegressor)` with a separate dispatch path (`TREE_MODELS`).

### Cross-Validation

| Dataset / sweep | Strategy | Rationale |
|-----------------|----------|-----------|
| Forecasting production (`ganymede`, `spe_berg`, `volve`, `inner_mongolia`) | grouped 80/20 temporal holdout + `GroupedExpandingWindowCV` (3-fold) | Preserve temporal order **within each well** instead of across the flattened multi-well sample index |
| 3W feature-based production | stratified-group holdout + `StratifiedGroupKFoldSKLearn` (5-fold) | No instance leakage across folds |
| 3W raw production script | `TemporalSplitCV` | Legacy raw-window baseline path |
| CDF production | temporal holdout + `SlidingWindowCV` (3-fold) | Multiple temporal folds for the single-compressor series |

Normalization is computed from the training partition only. Classification metrics use probability scores for AUC-PR and per-instance metadata for EDR when available.

### Configuration

Hierarchical YAML via OmegaConf: `base.yaml` ← `data/*.yaml` ← `models/*.yaml` ← CLI overrides. All hyperparameters are set through config files.

### Reproducibility

`set_global_seed(42)` locks Python `random`, NumPy, PyTorch CPU, CUDA, cuDNN, and CUBLAS. All results are JSON with full fold-level detail.

---

## Adding a New Model

1. Create `src/offshore_dl/models/my_model.py` — inherit `BaseModel` or use sklearn-style for tree models
2. Add `configs/models/my_model.yaml` with architecture params + Optuna search space
3. Register in `src/offshore_dl/models/__init__.py` and the relevant production script
4. Run: the full evaluation pipeline (CV, metrics, MLflow, tables) applies automatically

---

## Key Dependencies

| Package | Role |
|---------|------|
| PyTorch ≥ 2.3 | Core DL framework |
| XGBoost ≥ 2.0 | Gradient boosted trees |
| OmegaConf | Hierarchical YAML config |
| MLflow ≥ 2.14 | Experiment tracking |
| Optuna ≥ 3.6 | Hyperparameter optimization |
| scikit-learn ≥ 1.4 | Metrics, preprocessing |
| HuggingFace Transformers ≥ 4.40 | PatchTST architecture |

Foundation model deps (`timesfm`, `chronos-forecasting`, `tirex-ts`) are in the optional `[fm]` extra.

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
