# scripts/

Production scripts, HPC job definitions, and utility wrappers.

## Production Sweeps

| Script | Description |
|--------|-------------|
| `run_production_ganymede.py` | Full Ganymede sweep: 7 models × 4 horizons × (multi_well + 7 per_well). Supports `--models` filter and `--dry-run`. |
| `run_production_3w_features.py` | 3W classification with statistical feature extraction. Supports `--models` filter. |
| `run_production_3w.py` | 3W classification on raw 720-step windows (no feature extraction). |
| `run_production_cdf.py` | CDF anomaly detection sweep: 6 models. |
| `run_all_production.sh` | Runs all production sweeps sequentially. |

## HPO and Analysis

| Script | Description |
|--------|-------------|
| `run_optuna_hpo.py` | 30-trial Bayesian HPO with inner CV. Supports `--models` filter. |
| `run_statistical_tests.py` | Friedman/Nemenyi/Wilcoxon significance tests across all models. |

## TiRex-Specific

| Script | Description |
|--------|-------------|
| `extract_tirex_embeddings.py` | Extract TiRex xLSTM embeddings to memmap file. |
| `run_tirex_rf_nested.py` | TiRex embedding + Random Forest nested holdout evaluation. |
| `run_tirex_rf_folds.py` | TiRex RF per-fold evaluation. |

## Infrastructure

| Script | Description |
|--------|-------------|
| `docker_run.sh` | Docker convenience wrapper (GPU passthrough, volume mounts). |
| `singularity_run.sh` | Singularity/Apptainer runner for HPC clusters. |
| `push_to_registry.sh` | Push Docker image to GHCR (+ air-gapped tarball fallback). |
| `deploy_lps.sh` | Deploy to LPS cluster. |
| `run_mlflow_tracked.sh` | Launch MLflow-tracked training run. |
| `verify_gpu.sh` | CUDA/GPU availability check. |

## Slurm Jobs (HPC)

| Script | Target |
|--------|--------|
| `hpc_job.slurm` | Ganymede sweep (4h) |
| `hpc_job_3w.slurm` | 3W full training (24h) |
| `slurm_ganymede.sh` | Ganymede production |
| `slurm_3w.sh` | 3W production |
| `slurm_cdf.sh` | CDF production |
| `slurm_hpo_lstm.sh` | LSTM HPO (30 trials) |
| `slurm_hpo_deeponet.sh` | DeepONet HPO |
| `slurm_hpo_patchtst.sh` | PatchTST HPO |
| `slurm_tirex.sh` | TiRex embedding extraction |
| `slurm_tirex_rf.sh` | TiRex RF classification |

## archive/

Superseded single-model baseline scripts from early development. Kept for reference.
