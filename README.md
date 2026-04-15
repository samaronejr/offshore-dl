# offshore-dl

**Benchmarking Deep Learning for Offshore Production Monitoring**

MSc dissertation — UFRJ/COPPE PEE  
*Production Forecasting and Anomaly Detection on Offshore Platforms Using Deep Learning*

---

## Overview

This repository contains the code and experimental results for my MSc dissertation. It compares 16 architectures across three offshore production monitoring tasks: gas production forecasting, fault classification, and anomaly detection.

The models span seven paradigms, from recurrent networks and neural operators to state-space models and zero-shot foundation models:

| Paradigm | Model | Training | Tasks |
|----------|-------|----------|-------|
| Recurrent | **LSTM** | From scratch | Forecasting · Classification · Anomaly |
| Neural operator | **DeepONet** | From scratch | Forecasting · Classification · Anomaly |
| Transformer | **PatchTST** | From scratch* | Forecasting · Classification · Anomaly |
| Temporal convolution | **TCN** | From scratch | Forecasting |
| Convolution | **ConvTimeNet** | From scratch | Classification |
| Convolution | **ConvTran** | From scratch | Classification |
| Inception | **InceptionTime** | From scratch | Classification |
| State-space | **MambaSL** | From scratch | Classification |
| Anomaly detection | **FKMAD** | From scratch | Classification |
| Convolution | **HydraRocket** | sklearn-style | Classification |
| Tree ensemble | **Random Forest** | sklearn-style | Classification |
| Zero-shot FM | **Chronos-2** | Pretrained | Forecasting · Anomaly |
| Zero-shot FM | **TimesFM 2.5** | Pretrained | Forecasting · Anomaly |
| Zero-shot FM | **TiRex** | Pretrained (xLSTM) | Forecasting · Classification · Anomaly |
| Fine-tuned FM | **MOMENT** | LoRA fine-tuning | Classification (optional) |
| Fine-tuned FM | **Mantis** | Frozen encoder + RF | Classification (optional) |

<sub>*PatchTST uses the HuggingFace architecture but trains from scratch — pretrained weights are incompatible with the input dimensions of these datasets.</sub>

---

## Datasets

| Dataset | Source | Task | Scale |
|---------|--------|------|-------|
| **3W v2.0.0** | Petrobras | 10-class fault classification | 208,973 windows · 27 sensors · 1.8 GB |
| **Ganymede** | NSTA (UK Continental Shelf) | Gas production forecasting | 7 wells · 31,359 samples · 63 features |
| **CDF** | Cognite Data Fusion | Unsupervised anomaly detection | 1 compressor · 4,367 hourly rows · 12 sensors |
| **SPE Berg** | SPE Data Repository | Gas production forecasting | 53 wells · shale gas |
| **Volve** | Equinor (open release) | Oil production forecasting | 6 wells · oil field |
| **Inner Mongolia** | Public release | Gas production forecasting | 30 wells · gas |

---

## Results at a Glance

### 3W — Fault Classification (held-out test, n=41,515)

| Model | Accuracy | F1-macro | F1-weighted |
|-------|:--------:|:--------:|:-----------:|
| **FKMAD** | **96.70%** | 0.961 | **0.967** |
| **ConvTimeNet** | 96.61% | 0.962 | 0.966 |
| **Random Forest** | 96.58% | **0.964** | 0.966 |
| **MambaSL** | 96.56% | 0.961 | 0.966 |
| **PatchTST (HPO)** | 96.47% | 0.962 | 0.965 |
| PatchTST | 95.72% | 0.952 | 0.957 |
| DeepONet | 94.14% | 0.931 | 0.942 |
| LSTM | 92.21% | 0.911 | 0.921 |
| TiRex (embeddings + RF) | 91.16% | 0.895 | 0.911 |
| InceptionTime† | — | 0.012 | — |

†InceptionTime collapsed to single-class prediction due to a kernel size mismatch with the feature-matrix input format. Result retained for reference.

PatchTST improved from 95.7% to 96.5% after 30-trial Optuna HPO. Manual tuning of Random Forest, ConvTimeNet, FKMAD, and MambaSL proved near-optimal. The discriminative information in the 3W task resides in the spatial arrangement of the (14×27) feature matrix, which benefits convolution and tree-based approaches.

### Ganymede — Gas Production Forecasting (held-out test, multi-well, h=7, MMSCF)

| Model | MAE h=7 | R²\_prod h=7 |
|-------|:-------:|:------------:|
| **TiRex** | **0.658** | **0.822** |
| **TimesFM** | 0.740 | 0.826 |
| **Chronos-2** | 0.739 | 0.722 |
| LSTM | 0.873 | −0.078 |
| PatchTST | 1.005 | −0.501 |
| DeepONet | 1.224 | 0.197 |
| TCN | 3.982 | −4.293 |

Foundation models outperform trained models at all horizons. TiRex achieves the lowest MAE and TimesFM the highest R²_prod; both substantially outperform all trained architectures. The production sweep uses a **per-well grouped 80/20 temporal holdout plus grouped 3-fold ExpandingWindowCV**.

### CDF — Anomaly Detection (held-out test, reconstruction error ↓)

| Model | Error Mean |
|-------|:----------:|
| **LSTM** | **0.005** |
| PatchTST | 0.068 |
| DeepONet | 0.209 |
| Chronos-2 | 0.217 |
| TiRex | 0.235 |
| TimesFM | 0.263 |

Trained models dominate this task. LSTM reconstruction error is orders of magnitude lower than zero-shot FMs, which were not designed for this type of encoder-decoder reconstruction.

---

## Project Structure

```
offshore-dl/
├── src/offshore_dl/           # Main package
│   ├── models/                # All 16 architectures
│   │   ├── base.py            #   BaseModel ABC — common interface
│   │   ├── lstm.py            #   Bidirectional LSTM + attention
│   │   ├── deeponet.py        #   Time-Dependent DeepONet (branch-trunk)
│   │   ├── patchtst.py        #   PatchTST via HuggingFace Transformers
│   │   ├── tcn.py             #   Temporal Convolutional Network
│   │   ├── convtimenet.py     #   ConvTimeNet (multi-scale convolution)
│   │   ├── convtran.py        #   ConvTran (convolution + transformer)
│   │   ├── inception_time.py  #   InceptionTime
│   │   ├── mambasl.py         #   MambaSL (state-space model)
│   │   ├── fkmad.py           #   FKMAD anomaly detector
│   │   ├── hydra_rocket.py    #   HydraRocket (sklearn-style)
│   │   ├── chronos_wrapper.py #   Chronos-2 zero-shot wrapper
│   │   ├── timesfm_wrapper.py #   TimesFM 2.5 zero-shot wrapper
│   │   ├── tirex_wrapper.py   #   TiRex zero-shot wrapper
│   │   ├── tirex_classifier.py #  TiRex embedding + RF classifier
│   │   ├── moment_wrapper.py  #   MOMENT LoRA fine-tuning (optional)
│   │   └── mantis_wrapper.py  #   Mantis frozen encoder + RF (optional)
│   ├── data/                  # Data loading and preprocessing
│   ├── training/              # Trainer, ExperimentRunner, Optuna HPO
│   │   └── ...                #   SupCon pre-training, label smoothing
│   ├── evaluation/            # MetricRegistry, CV strategies, baselines
│   ├── analysis/              # LaTeX generation, statistical tests
│   └── utils/                 # Config, reproducibility, serialization
│       ├── config.py          #   OmegaConf helpers
│       ├── reproducibility.py #   seed locking (7 RNG sources)
│       └── serialization.py   #   JSON result I/O
│
├── configs/                   # Hierarchical YAML (OmegaConf)
│   ├── base.yaml              #   Global defaults
│   ├── data/                  #   3w, ganymede, cdf, spe_berg, volve,
│   │                          #   inner_mongolia
│   └── models/                #   Per-model configs + Optuna search spaces
│
├── docker/                    # Dockerfile (CUDA 12.4), docker-compose
├── tests/                     # 534 tests
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

# Run all tests
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

Every trained model inherits from `BaseModel` and implements `training_step`, `predict`, and `configure_optimizers`. Zero-shot FMs return `loss = 0.0` from `training_step`. HydraRocket and Random Forest use a sklearn-style dispatch path and do not use the PyTorch training loop.

### Cross-Validation

| Dataset / sweep | Strategy | Rationale |
|-----------------|----------|-----------|
| Forecasting (`ganymede`, `spe_berg`, `volve`, `inner_mongolia`) | Grouped 80/20 temporal holdout + `GroupedExpandingWindowCV` (3-fold) | Preserve temporal order within each well rather than across the flattened multi-well sample index |
| 3W feature-based | Stratified-group holdout + `StratifiedGroupKFoldSKLearn` (5-fold) | No instance leakage across event groups |
| 3W raw windows | `TemporalSplitCV` | Legacy raw-window baseline path |
| CDF | Temporal holdout + `SlidingWindowCV` (3-fold) | Multiple temporal folds for the single-compressor series |

Normalization is computed from the training partition only and applied to validation and test partitions without re-fitting.

### Training Features

- **SupCon pre-training**: supervised contrastive pre-training stage available for classification models
- **Label smoothing**: configurable `label_smoothing` parameter in training config
- **Optuna HPO**: 30-trial Bayesian search over architecture and optimizer hyperparameters
- **MLflow**: all experiments tracked with full fold-level detail in JSON outputs

### Configuration

Hierarchical YAML via OmegaConf: `base.yaml` ← `data/*.yaml` ← `models/*.yaml` ← CLI overrides. All hyperparameters are set through config files.

### Reproducibility

`set_global_seed(42)` locks Python `random`, NumPy, PyTorch CPU, CUDA, cuDNN, and CUBLAS (7 RNG sources). All results are stored as JSON with full fold-level detail via `utils/serialization.py`.

---

## Adding a New Model

1. Create `src/offshore_dl/models/my_model.py` — inherit `BaseModel` or use sklearn-style for non-differentiable models
2. Add `configs/models/my_model.yaml` with architecture params and an Optuna search space block
3. Register in `src/offshore_dl/models/__init__.py` and the relevant production script
4. Run: the full evaluation pipeline (CV, metrics, MLflow, LaTeX tables) applies automatically

---

## Key Dependencies

| Package | Role |
|---------|------|
| PyTorch ≥ 2.3 | Core DL framework |
| LightGBM | Gradient boosted trees |
| OmegaConf | Hierarchical YAML config |
| MLflow ≥ 2.14 | Experiment tracking |
| Optuna ≥ 3.6 | Hyperparameter optimization |
| scikit-learn ≥ 1.4 | Metrics, preprocessing, RF |
| HuggingFace Transformers ≥ 4.40 | PatchTST architecture |

**Optional extras:**

| Package | Role | Extra |
|---------|------|-------|
| `timesfm` | TimesFM 2.5 zero-shot FM | `[fm]` |
| `chronos-forecasting` | Chronos-2 zero-shot FM | `[fm]` |
| `tirex-ts` | TiRex zero-shot FM | `[fm]` |
| `mamba-ssm` | MambaSL state-space model | `[mamba]` |
| `aeon` | InceptionTime, HydraRocket | `[aeon]` |
| `momentfm` | MOMENT LoRA fine-tuning | `[fm]` |

Install all extras: `pip install -e ".[dev,fm,mamba,aeon]"`

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
