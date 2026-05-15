# scripts/

Production scripts, HPO launchers, Slurm job definitions, and analysis utilities. Run scripts from the project root; they add `src/` to `sys.path` internally.

New production benchmark writers default to `results/post_fix/` unless `--results-dir` or `OFFSHORE_DL_RESULTS_DIR` is set. HPO stays under `results/hpo/` unless `--output-dir` is set.

## Production Sweeps

| Script | Description |
|--------|-------------|
| `run_production_ganymede.py` | Full Ganymede sweep: 7 models × 4 horizons × (`multi_well` + 7 `per_well`). Supports `--models` and `--dry-run`. |
| `run_production_3w_features.py` | 3W classification with statistical, wavelet, multiscale, physics, and window-feature variants. Supports `--models`. |
| `run_production_3w.py` | 3W classification on raw 720-step windows with stratified `instance_id` group CV. |
| `run_production_cdf.py` | CDF anomaly-detection sweep. Use for the required post-fix CDF rerun before final anomaly claims. |
| `run_production_spe_berg.py` | SPE Berg forecasting sweep. |
| `run_production_volve.py` | Volve forecasting sweep. |
| `run_production_inner_mongolia.py` | Inner Mongolia forecasting sweep. |
| `run_all_production.sh` | Runs all production sweeps sequentially. |

Example:

```bash
python scripts/run_production_ganymede.py --device cuda --models lstm tcn
python scripts/run_production_3w_features.py --device cuda --models random_forest wavelet_rf
```

## 3W HPO workflow

Stage 1 is the validated apples-to-apples 720-window HPO campaign. It uses macro-F1 as the objective and requires final held-out evaluation before a result is accepted.

| Script/file | Description |
|-------------|-------------|
| `run_optuna_hpo.py` | Resumable Optuna HPO. Supports `--dataset 3w`, `--models`, `--n-trials`, `--campaign-id`, `--resume`, `--summary-only`, and final-eval gating. |
| `hpo_3w_models.txt` | Manifest of Stage 1 3W models for array launch/validation. |
| `submit_hpo_3w_array.sh` | Convenience launcher for the 3W HPO Slurm array. |
| `slurm_hpo_3w_array.sh` | Slurm array worker for 3W HPO. |
| `validate_hpo_3w_results.py` | Validates per-model HPO JSONs and writes `summary.json` only when final benchmark evidence is present. |

Typical local smoke/summary commands:

```bash
python scripts/run_optuna_hpo.py --dataset 3w --models lstm --n-trials 2 --device cpu --no-final-eval
python scripts/run_optuna_hpo.py --dataset 3w --campaign-id <campaign-id> --summary-only
python scripts/validate_hpo_3w_results.py --campaign-id <campaign-id> --write-summary
```

Typical HPC launch:

```bash
sbatch scripts/slurm_hpo_3w_array.sh
```

If a Slurm/Optuna campaign is interrupted, rerun with the same `--campaign-id --resume` and then validate with `validate_hpo_3w_results.py`; incomplete or partial outputs should not be promoted into `summary.json`.

## 3W Stage 2 variants

Stage 2 follow-ups are launched through `run_production_3w_features.py` and related Slurm wrappers. Report these separately from Stage 1 because feature sets and window lengths may differ.

Current result interpretation:

- `window360_rf` and `window1440_rf` are valid RF window-length experiments, not direct replacements for the 720-window Stage 1 leaderboard.
- `wavelet_*`, `multiscale_*`, and `physics_*` are feature-family variants.
- `hydra_rocket` exceeded practical RAM limits in the latest retry campaign.
- raw deep variants that collapsed to macro-F1 0.027287 should be treated as failed baselines.

## Forecasting and analysis utilities

| Script | Description |
|--------|-------------|
| `aggregate_forecasting_results.py` | Aggregates forecasting JSON outputs into summary tables. |
| `build_forecasting_mase_manifest.py` | Builds/validates forecasting MASE rerun manifests. |
| `populate_spe_berg_tables.py` | SPE Berg table population utility. |
| `validate_spe_berg_results.py` | Result validation for SPE Berg. |
| `run_statistical_tests.py` | Friedman/Nemenyi/Wilcoxon significance tests across model result matrices. |

Do not pool 3W Stage 2 variants with Stage 1 models in statistical tests unless the report explicitly groups them as a separate experimental family.

## TiRex-specific scripts

| Script | Description |
|--------|-------------|
| `extract_tirex_embeddings.py` | Extract TiRex xLSTM embeddings to a memmap file. |
| `run_tirex_rf_nested.py` | TiRex embeddings + Random Forest nested holdout evaluation. |
| `run_tirex_rf_folds.py` | TiRex RF per-fold evaluation. |

## Infrastructure

| Script | Description |
|--------|-------------|
| `docker_run.sh` | Docker convenience wrapper with GPU passthrough and volume mounts. |
| `singularity_run.sh` | Singularity/Apptainer runner for HPC clusters. |
| `push_to_registry.sh` | Push Docker image to GHCR plus air-gapped tarball fallback. |
| `deploy_lps.sh` | Deploy to the LPS cluster. |
| `deploy_and_rerun.sh` / `deploy_mase_rerun.sh` | Deployment helpers for rerun campaigns. |
| `run_mlflow_tracked.sh` | Launch MLflow-tracked training run. |
| `verify_gpu.sh` | CUDA/GPU availability check. |

## Slurm jobs (HPC)

| Script | Target |
|--------|--------|
| `hpc_job.slurm` | Ganymede sweep |
| `hpc_job_3w.slurm` | 3W full training |
| `slurm_ganymede.sh` | Ganymede production |
| `slurm_3w.sh` | 3W production |
| `slurm_cdf.sh` | CDF production |
| `slurm_hpo_3w_array.sh` | 3W HPO array |
| `slurm_hpo_lstm.sh` | Legacy/single-model LSTM HPO |
| `slurm_hpo_deeponet.sh` | Legacy/single-model DeepONet HPO |
| `slurm_hpo_patchtst.sh` | Legacy/single-model PatchTST HPO |
| `slurm_forecasting_mase_rerun_array.sh` | Forecasting MASE rerun array |
| `slurm_forecasting_mase_postprocess.sh` | Forecasting MASE postprocess |
| `slurm_tirex.sh` | TiRex embedding extraction |
| `slurm_tirex_rf.sh` | TiRex RF classification |

## archive/

Superseded single-model baseline scripts from early development. Kept for historical reference only.
