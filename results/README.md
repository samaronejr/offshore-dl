# results/

Generated experiment outputs are organized by validity epoch and campaign. The repository may not contain every large result artifact after a fresh clone; production and HPO commands regenerate them.

## Structure

```text
results/
├── pre_fix/        # Historical outputs produced before benchmark-validity repairs
├── post_fix/       # Fixed-code production reruns and lightweight forecasting aggregate CSVs
├── hpo/            # Optuna campaign outputs, e.g. 3W Stage 1 HPO summaries
├── stage2_3w/      # 3W follow-up variants and window-length experiments
├── .omx/           # Local OMX/runtime logs and state, not benchmark outputs
└── README.md
```

## Current validity guidance

| Location | Status | Use |
|---|---|---|
| `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` | Current validated 3W Stage 1 HPO | Headline apples-to-apples 720-window classification leaderboard. |
| `results/stage2_3w/3w-stage2-20260513T192623Z/` | Current 3W follow-up campaign | Report separately as feature/window variants; do not pool with Stage 1. |
| `results/post_fix/<model>/{ganymede,spe_berg,volve,inner_mongolia}*.json` | Current post-fix forecasting reruns | Current forecasting evidence. The full JSON tree is large; use aggregate CSVs plus report manifests in Git. |
| `results/pre_fix/` | Historical | Audit/history only unless explicitly revalidated. |

## 3W Stage 1 HPO summary

Primary ranking metric: macro-F1 on the held-out test set. All models below completed 30 Optuna trials and final evaluation.

| Rank | Model | Macro-F1 | Accuracy | Trials | Best CV objective |
|---:|---|---:|---:|---:|---:|
| 1 | Random Forest | 0.968972 | 0.970228 | 30 | 0.966316 |
| 2 | DeepONet | 0.962579 | 0.968734 | 30 | 0.960540 |
| 3 | MambaSL | 0.962185 | 0.967313 | 30 | 0.958081 |
| 4 | LSTM | 0.960046 | 0.965314 | 30 | 0.953991 |
| 5 | FKMAD | 0.956388 | 0.964061 | 30 | 0.951980 |
| 6 | ConvTimeNet | 0.954953 | 0.959894 | 30 | 0.953499 |
| 7 | PatchTST | 0.953556 | 0.958738 | 30 | 0.957745 |

## 3W Stage 2 summary

Stage 2 outputs are valid follow-up experiments only when their preprocessing/window assumptions match the claim being made.

| Variant | Macro-F1 | Accuracy | Validity note |
|---|---:|---:|---|
| `window360_rf` | 0.987797 | 0.991537 | Valid, but different window length from Stage 1. |
| `window1440_rf` | 0.977011 | 0.989084 | Valid, but different window length from Stage 1. |
| `wavelet_rf` | 0.964309 | 0.966301 | Valid feature variant. |
| `multiscale_rf` | 0.964184 | 0.966205 | Valid feature variant. |
| `physics_rf` | 0.964070 | 0.966060 | Valid feature variant. |
| `wavelet_deeponet` | 0.961252 | 0.967867 | Valid DeepONet variant. |
| `convtran` | 0.955762 | 0.964398 | Valid baseline. |
| `multiscale_deeponet` | 0.954235 | 0.964880 | Valid DeepONet variant. |
| `physics_deeponet` | 0.952286 | 0.961652 | Valid DeepONet variant. |

Failed or invalid Stage 2 outputs:

- `hydra_rocket`: failed from an impractical multi-terabyte RAM allocation during the retry campaign.
- `convtimenet_raw`, `convtran_raw`, `fkmad_raw`, `mambasl_raw`: collapsed to macro-F1 0.027287 / accuracy 0.157991 and should be treated as failed raw-window baselines.

## Ganymede post-fix summary

Current Ganymede values are fixed-code multi-well aggregates across horizons from `results/post_fix/<model>/ganymede_h*_multi_well.json`.

| Model | MAE | RMSE | MASE | R² | R²_prod |
|---|---:|---:|---:|---:|---:|
| TiRex | 0.3620 | 1.2481 | 0.2072 | 0.3537 | -0.1499 |
| TimesFM | 0.3967 | 1.2640 | 0.2297 | 0.3356 | -0.1293 |
| Chronos-2 | 0.5373 | 1.4426 | 0.3215 | 0.1094 | -0.3438 |
| LSTM | 0.5457 | 1.3519 | 0.0228 | 0.2453 | -0.6491 |
| TCN | 0.5680 | 1.3140 | 0.0234 | 0.2860 | -0.4109 |
| DeepONet | 0.6794 | 1.3748 | 0.0301 | 0.2219 | -0.4026 |
| PatchTST | 1.0774 | 2.1175 | 0.0474 | -0.8660 | -1.2016 |

Use MAE/RMSE for absolute production-scale error, MASE for scaled within-well error, and R²/R²_prod only as diagnostics.

## Full post-fix forecasting campaign

The full synced forecasting root contains current fixed-code evidence for Ganymede, SPE Berg, Volve, and Inner Mongolia. The aggregate CSVs are small enough to track, while the full per-model JSON tree is about 9.1 GiB locally and should be archived outside Git.

| Artifact | Use |
|---|---|
| `results/post_fix/forecasting_summary.csv` | Long-form 2,737-row aggregate over model, dataset, horizon, mode, and well/scenario. |
| `results/post_fix/forecasting_summary_wide_mae.csv` | Wide MAE table for report/table drafting. |
| `results/post_fix/forecasting_summary_wide_r2_prod.csv` | Wide productive-period R² diagnostic table. |
| `reports/forecasting_borda.json` | Cross-dataset Borda diagnostics by metric. |
| `reports/forecasting_performance_audit/forecasting_hpc_sync_summary.md` | Sync provenance, coverage, sparse exclusions, and key hashes. |
| `reports/forecasting_performance_audit/forecasting_post_fix_sha256_manifest.txt` | SHA256 manifest for the full synced result tree. |

Multi-well coverage is complete for all four forecasting datasets: Ganymede 28/28, SPE Berg 28/28, Volve 28/28, and Inner Mongolia 28/28 model × horizon artifacts.

Cross-dataset Borda diagnostics, lower is better:

| Metric | Leading models |
|---|---|
| MAE | TiRex 1.660 · Chronos-2 2.212 · TimesFM 3.230 |
| R²_prod | TiRex 2.588 · Chronos-2 3.077 · TimesFM 3.442 |
| MASE | PatchTST 2.619 · LSTM 3.171 · TCN 3.427 |

Remaining expected-but-missing rows are sparse h90 per-well exclusions: Inner Mongolia (`57-14X`, `57-15X`), SPE Berg (`well_11`, `well_2`), and Volve (`NO_15_9-F-5_AH`). Do not relaunch these as ordinary failed jobs unless the dataset/split policy changes.

## CDF post-fix summary

Current CDF outputs are available on the HPC result root and summarized in `reports/cdf_post_fix_summary_2026-05.json`. Per-model JSONs were generated under `results/post_fix/<model>/cdf.json` on `LPS_loginServer:/home/samarone.lima/offshore-dl` and validated for strict raw-row gap metadata and non-empty fold results.

### Trained reconstruction metrics

| Model | error_mean | error_p50 | error_p95 | error_p99 |
|---|---:|---:|---:|---:|
| `lstm` | 0.005878 | 0.005289 | 0.007909 | 0.015403 |
| `patchtst` | 0.081621 | 0.071603 | 0.150840 | 0.271780 |
| `deeponet` | 0.230735 | 0.223694 | 0.414683 | 0.425859 |

### Foundation forecast metrics

| Model | forecast_error_mean | forecast_error_p50 | forecast_error_p95 | forecast_error_p99 |
|---|---:|---:|---:|---:|
| `chronos` | 0.243968 | 0.226866 | 0.457304 | 0.565650 |
| `tirex` | 0.268548 | 0.246441 | 0.501056 | 0.695557 |
| `timesfm` | 0.295999 | 0.283747 | 0.496752 | 0.589213 |

CDF trained and foundation rows are reported separately because they encode different anomaly semantics.

## Historical/pre-fix caveats

- `pre_fix/` results are preserved for audit/history.
- Old forecasting `mase` / grouped MASE values are non-authoritative unless rerun with repaired chronological MASE plumbing.
- Old CDF production CV results under `pre_fix/` are non-authoritative because those runs used zero CV gap before the strict raw-row gap repair; use the post-fix CDF summary above for current anomaly-detection evidence.
- Classification metrics are not directly invalidated by the MASE or CDF repairs. Older classification outputs remain under `pre_fix/` until rerun and should be treated as audit lineage.
- New production CLIs default writers to `results/post_fix/`; override with `--results-dir` or `OFFSHORE_DL_RESULTS_DIR` only for deliberate runs.
- HPO artifacts remain under `results/hpo/` unless `--output-dir` is provided, and are not final benchmark outputs unless final evaluation is present.
