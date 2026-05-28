# offshore-dl

**Benchmarking Deep Learning for Offshore Production Monitoring**

MSc dissertation — UFRJ/COPPE PEE  
*Production Forecasting and Anomaly Detection on Offshore Platforms Using Deep Learning*

---

## Overview

This repository contains the code and benchmark artifacts for an MSc dissertation on offshore production monitoring. It evaluates deep learning, neural operators, state-space models, tree ensembles, and zero-shot time-series foundation models across forecasting, classification, and anomaly-detection tasks.

The post-fix documentation separates **current validated results** from historical artifacts. Historical outputs remain under `results/pre_fix/`; new fixed-code runs use `results/post_fix/`, `results/hpo/`, or campaign-specific directories.

| Paradigm | Model family | Training | Main tasks |
|----------|--------------|----------|------------|
| Recurrent | **LSTM** | From scratch | Forecasting · Classification · Anomaly |
| Neural operator | **DeepONet** | From scratch | Forecasting · Classification variants · Anomaly |
| Transformer | **PatchTST** | From scratch* | Forecasting · Classification · Anomaly |
| Temporal convolution | **TCN** | From scratch | Forecasting |
| Convolution/attention | **ConvTimeNet**, **ConvTran**, **InceptionTime** | From scratch | Classification |
| State-space | **MambaSL** | From scratch | Classification |
| Frequency/anomaly | **FKMAD** | From scratch | Classification · Anomaly |
| Kernel / tree | **HydraRocket**, **Random Forest** | sklearn-style | Classification |
| Zero-shot FM | **Chronos-2**, **TimesFM 2.5**, **TiRex** | Pretrained | Forecasting · Anomaly |
| Optional FM classifiers | **MOMENT**, **MANTIS**, **TiRex embeddings + RF** | Fine-tuned or frozen encoder | Classification |

<sub>*PatchTST uses the HuggingFace architecture but trains from scratch because available pretrained weights are incompatible with these dataset dimensions.</sub>

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

### 3W — Stage 1 validated HPO, standard 720-window classification

Held-out test size: 41,515 windows. Primary metric: **macro-F1**. The campaign was validated from `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` after 30 Optuna trials per model.

| Rank | Model | Macro-F1 | Accuracy | Trials | Best CV objective |
|---:|---|---:|---:|---:|---:|
| 1 | **Random Forest** | **0.968972** | **0.970228** | 30 | 0.966316 |
| 2 | DeepONet | 0.962579 | 0.968734 | 30 | 0.960540 |
| 3 | MambaSL | 0.962185 | 0.967313 | 30 | 0.958081 |
| 4 | LSTM | 0.960046 | 0.965314 | 30 | 0.953991 |
| 5 | FKMAD | 0.956388 | 0.964061 | 30 | 0.951980 |
| 6 | ConvTimeNet | 0.954953 | 0.959894 | 30 | 0.953499 |
| 7 | PatchTST | 0.953556 | 0.958738 | 30 | 0.957745 |

Random Forest is the strongest standard 720-window classifier by macro-F1. Deep models remain competitive, especially DeepONet and MambaSL, but do not beat the tuned Random Forest on this validated split.

### 3W — Stage 2 follow-up variants

Stage 2 explores feature and window-length variants. These runs answer different experimental questions and should **not** be merged into the Stage 1 apples-to-apples leaderboard.

| Variant | Macro-F1 | Accuracy | Interpretation |
|---|---:|---:|---|
| `window360_rf` | **0.987797** | **0.991537** | Best follow-up result; different window length, so report separately. |
| `window1440_rf` | 0.977011 | 0.989084 | Strong long-window RF variant; separate comparison. |
| `wavelet_rf` | 0.964309 | 0.966301 | Feature variant, below tuned Stage 1 RF. |
| `multiscale_rf` | 0.964184 | 0.966205 | Feature variant, below tuned Stage 1 RF. |
| `physics_rf` | 0.964070 | 0.966060 | Feature variant, below tuned Stage 1 RF. |
| `wavelet_deeponet` | 0.961252 | 0.967867 | Competitive DeepONet variant. |
| `convtran` | 0.955762 | 0.964398 | Completed baseline. |
| `multiscale_deeponet` | 0.954235 | 0.964880 | Completed variant. |
| `physics_deeponet` | 0.952286 | 0.961652 | Completed variant. |

Invalid/failed Stage 2 outputs are retained for audit only: `hydra_rocket` failed from impractical RAM allocation, and the raw deep variants (`convtimenet_raw`, `convtran_raw`, `fkmad_raw`, `mambasl_raw`) collapsed to macro-F1 0.027287 / accuracy 0.157991.

### Ganymede — Post-fix gas production forecasting

Multi-well aggregate across horizons (`h7`, `h14`, `h30`, `h90`) from `results/post_fix/<model>/ganymede_h*_multi_well.json`. Values are on the denormalized target scale. Lower is better for MAE/RMSE/MASE; higher is better for R² diagnostics.

| Model | MAE | RMSE | MASE | R² | R²_prod |
|---|---:|---:|---:|---:|---:|
| **TiRex** | **0.3617** | **1.2476** | 0.2071 | **0.3541** | -0.1490 |
| TimesFM | 0.3965 | 1.2634 | 0.2295 | 0.3362 | **-0.1283** |
| Chronos-2 | 0.5357 | 1.4412 | 0.3205 | 0.1111 | -0.3431 |
| LSTM | 0.5457 | 1.3517 | **0.0228** | 0.2455 | -0.6481 |
| TCN | 0.5677 | 1.3136 | 0.0234 | 0.2864 | -0.4099 |
| DeepONet | 0.6795 | 1.3746 | 0.0301 | 0.2221 | -0.4021 |
| PatchTST | 1.0771 | 2.1164 | 0.0474 | -0.8640 | -1.1972 |

Metric interpretation matters: zero-shot foundation models lead by absolute MAE/RMSE, while trained LSTM/TCN are strongest by grouped MASE. R²-style metrics remain unstable across wells and are diagnostics, not the headline score.

### Forecasting — full post-fix multi-dataset status

The post-fix forecasting campaign now includes Ganymede, SPE Berg, Volve, and Inner Mongolia under `results/post_fix/`. The synced aggregate summary has 2,737 valid rows across seven models, four horizons (`h7`, `h14`, `h30`, `h90`), and both multi-well and per-well modes. The full JSON result tree is large and should be archived externally; lightweight provenance is tracked in `reports/forecasting_performance_audit/forecasting_hpc_sync_summary.md` and `forecasting_post_fix_sha256_manifest.txt`.

All multi-well forecasting cells are complete:

| Dataset | Multi-well artifacts | Expected model × horizon artifacts |
|---|---:|---:|
| Ganymede | 28 | 28 |
| SPE Berg | 28 | 28 |
| Volve | 28 | 28 |
| Inner Mongolia | 28 | 28 |

Cross-dataset Borda diagnostics from `reports/forecasting_borda.json` are metric-specific; lower Borda score is better and should not be read as a raw error average.

| Metric | Best three models |
|---|---|
| MAE | TiRex 1.660 · Chronos-2 2.212 · TimesFM 3.230 |
| R²_prod | TiRex 2.588 · Chronos-2 3.077 · TimesFM 3.442 |
| MASE | PatchTST 2.619 · LSTM 3.171 · TCN 3.427 |

Remaining missing expected rows are sparse h90 per-well exclusions, not active HPC failures: Inner Mongolia (`57-14X`, `57-15X`), SPE Berg (`well_11`, `well_2`), and Volve (`NO_15_9-F-5_AH`). Report these as data-coverage exclusions whenever using per-well h90 tables.

### CDF — Post-fix anomaly detection

Post-fix CDF rerun completed on HPC job `28934` using strict raw-row gap metadata (`cv_gap=47`, `inner_gap=47`, `outer_gap=47`). Metrics are separated by semantics: trained models report reconstruction `error_*`, while foundation models report one-step forecast `forecast_error_*`; lower is better within each group.

**Trained reconstruction models**

| Model | error_mean | error_p50 | error_p95 | error_p99 | Elapsed |
|---|---:|---:|---:|---:|---:|
| `lstm` | 0.005878 | 0.005289 | 0.007909 | 0.015403 | 65.5s |
| `patchtst` | 0.081621 | 0.071603 | 0.150840 | 0.271780 | 116.7s |
| `deeponet` | 0.230735 | 0.223694 | 0.414683 | 0.425859 | 44.6s |

**Foundation forecast models**

| Model | forecast_error_mean | forecast_error_p50 | forecast_error_p95 | forecast_error_p99 | Elapsed |
|---|---:|---:|---:|---:|---:|
| `chronos` | 0.243968 | 0.226866 | 0.457304 | 0.565650 | 1243.4s |
| `tirex` | 0.268548 | 0.246441 | 0.501056 | 0.695557 | 220.7s |
| `timesfm` | 0.295999 | 0.283747 | 0.496752 | 0.589213 | 21.8s |

Do not pool the trained and foundation CDF rows into one universal anomaly-detection ranking unless a later methodology decision establishes comparable reconstruction-vs-forecast error semantics.

---

## Project Structure

```text
offshore-dl/
├── src/offshore_dl/           # Main package
│   ├── models/                # Trained, sklearn-style, and FM wrapper models
│   ├── data/                  # Data loading and preprocessing
│   ├── training/              # Trainer, ExperimentRunner, Optuna HPO
│   ├── evaluation/            # Metrics, CV strategies, baselines
│   ├── analysis/              # LaTeX generation, statistical tests
│   └── utils/                 # Config, reproducibility, serialization
├── configs/                   # Hierarchical YAML configs
├── scripts/                   # Production, HPO, Slurm, and analysis scripts
├── docker/                    # CUDA container and docker-compose files
├── tests/                     # 673 collected pytest tests
├── results/                   # Generated result artifacts by validity epoch/campaign
├── reports/                   # LaTeX/PDF/statistical reports
├── dvc.yaml                   # DVC pipeline definition
└── pyproject.toml             # Package metadata and dependencies
```

Data are not committed. Provide datasets externally under `data/raw/` or run the preprocessing entry points before training.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/<user>/offshore-dl.git
cd offshore-dl
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single experiment without MLflow
python -m offshore_dl.run_experiment --model lstm --dataset ganymede --max-epochs 5 --no-mlflow

# Production sweeps
python scripts/run_production_ganymede.py --device cuda
python scripts/run_production_3w_features.py --device cuda
python scripts/run_production_cdf.py --device cuda

# 3W HPO campaign summary/validation
python scripts/run_optuna_hpo.py --dataset 3w --models lstm patchtst --n-trials 30 --device cuda
python scripts/validate_hpo_3w_results.py --campaign-id <campaign-id> --write-summary

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
sbatch scripts/hpc_job.slurm             # Ganymede sweep
sbatch scripts/hpc_job_3w.slurm          # 3W production training
sbatch scripts/slurm_hpo_3w_array.sh     # 3W HPO array campaign
```

HPC runs should write to explicit campaign directories and be validated before their values are promoted into reports or README tables.

---

## How It Works

### Architecture

Every trained model inherits from `BaseModel` and implements `training_step`, `predict`, and `configure_optimizers`. Zero-shot FMs return `loss = 0.0` from `training_step`. HydraRocket and Random Forest use sklearn-style dispatch paths and do not use the PyTorch training loop.

### Cross-Validation

| Dataset / sweep | Strategy | Rationale |
|-----------------|----------|-----------|
| Forecasting (`ganymede`, `spe_berg`, `volve`, `inner_mongolia`) | Grouped 80/20 temporal holdout + `GroupedExpandingWindowCV` | Preserve temporal order within wells. |
| 3W feature-based | Stratified-group holdout + `StratifiedGroupKFoldSKLearn` | Avoid leakage across event groups. |
| 3W raw windows | `TemporalSplitCV` | Legacy raw-window baseline path. |
| CDF | Temporal holdout + `SlidingWindowCV` with strict raw-row gap | Avoid adjacent-window leakage for the single-compressor series. |

Normalization is fitted on the training partition only and then applied to validation/test partitions without refitting.

### Training Features

- **SupCon pre-training** for classification models.
- **Label smoothing / focal-loss variants** through model configs.
- **Optuna HPO** with resumable campaign directories and final evaluation gating.
- **MLflow** experiment tracking, plus JSON result outputs with fold-level detail.

### Configuration

Hierarchical YAML via OmegaConf: `base.yaml` ← `data/*.yaml` ← `models/*.yaml` ← CLI overrides. All hyperparameters are set through config files or CLI dotlist overrides.

### Reproducibility

`set_global_seed(42)` seeds Python `random`, NumPy, PyTorch CPU/CUDA, cuDNN, and CUBLAS workspace configuration. CUBLAS determinism depends on setting `CUBLAS_WORKSPACE_CONFIG` before CUDA initialization; use `set_global_seed(42, strict=True)` at process entry points to fail fast when that guarantee cannot be met. Results are stored as JSON with full fold-level detail via `utils/serialization.py`.

---

## Adding a New Model

1. Create `src/offshore_dl/models/my_model.py` — inherit `BaseModel` or use sklearn-style handling for non-differentiable models.
2. Add `configs/models/my_model.yaml` with architecture parameters and an Optuna search-space block.
3. Register in `src/offshore_dl/models/__init__.py` and in the relevant experiment/production dispatch path.
4. Add tests and a dry-run/smoke path before launching production sweeps.

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
| `statsmodels` | Statistical comparison reports | `[stats]` |

- Install common development extras: `pip install -e ".[dev]"`
- Install foundation-model extras: `pip install -e ".[fm]"`
- Install optional model/report extras: `pip install -e ".[mamba,aeon,stats]"`
- Install all documented extras: `pip install -e ".[dev,fm,mamba,aeon,stats]"`

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
