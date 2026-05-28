# Final Dissertation Tables — 2026-05

Source revision: `44402db9fc6eef8bc731c57445bbb9c8f07b5319`
Purpose: source-backed markdown tables for dissertation drafting. The claim ledger remains the authoritative claim-to-source map: `reports/dissertation_claim_ledger_2026-05.md`.

## T1 — 3W Stage 1 validated HPO, standard 720-window classification

Primary metric: held-out macro-F1. Source: `README.md:48-62`, `results/README.md:26-38`, and HPC `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json`.

| Rank | Model | Macro-F1 | Accuracy | Trials | Best CV objective |
|---:|---|---:|---:|---:|---:|
| 1 | Random Forest | 0.968972 | 0.970228 | 30 | 0.966316 |
| 2 | DeepONet | 0.962579 | 0.968734 | 30 | 0.960540 |
| 3 | MambaSL | 0.962185 | 0.967313 | 30 | 0.958081 |
| 4 | LSTM | 0.960046 | 0.965314 | 30 | 0.953991 |
| 5 | FKMAD | 0.956388 | 0.964061 | 30 | 0.951980 |
| 6 | ConvTimeNet | 0.954953 | 0.959894 | 30 | 0.953499 |
| 7 | PatchTST | 0.953556 | 0.958738 | 30 | 0.957745 |

## T2 — 3W Stage 2 follow-up variants

Stage 2 is not pooled with Stage 1 because it changes feature/window assumptions. Source: `README.md:64-80`, `results/README.md:40-59`.

| Variant | Macro-F1 | Accuracy | Validity note |
|---|---:|---:|---|
| `window360_rf` | 0.987797 | 0.991537 | Valid follow-up; different window length from Stage 1. |
| `window1440_rf` | 0.977011 | 0.989084 | Valid follow-up; different window length from Stage 1. |
| `wavelet_rf` | 0.964309 | 0.966301 | Valid feature variant. |
| `multiscale_rf` | 0.964184 | 0.966205 | Valid feature variant. |
| `physics_rf` | 0.964070 | 0.966060 | Valid feature variant. |
| `wavelet_deeponet` | 0.961252 | 0.967867 | Valid DeepONet variant. |
| `convtran` | 0.955762 | 0.964398 | Valid completed baseline. |
| `multiscale_deeponet` | 0.954235 | 0.964880 | Valid DeepONet variant. |
| `physics_deeponet` | 0.952286 | 0.961652 | Valid DeepONet variant. |

## T3 — Ganymede post-fix multi-well forecasting

Values are arithmetic means across h7/h14/h30/h90 multi-well rows from `results/post_fix/forecasting_summary.csv`. Lower is better for MAE/RMSE/MASE; higher is better for R²/R²_prod diagnostics.

| Model | MAE | RMSE | MASE | R² | R²_prod |
|---|---:|---:|---:|---:|---:|
| TiRex | 0.3620 | 1.2481 | 0.2072 | 0.3537 | -0.1499 |
| TimesFM | 0.3967 | 1.2640 | 0.2297 | 0.3356 | -0.1293 |
| Chronos-2 | 0.5373 | 1.4426 | 0.3215 | 0.1094 | -0.3438 |
| LSTM | 0.5457 | 1.3519 | 0.0228 | 0.2453 | -0.6491 |
| TCN | 0.5680 | 1.3140 | 0.0234 | 0.2860 | -0.4109 |
| DeepONet | 0.6794 | 1.3748 | 0.0301 | 0.2219 | -0.4026 |
| PatchTST | 1.0774 | 2.1175 | 0.0474 | -0.8660 | -1.2016 |

## T4 — Cross-dataset forecasting Borda diagnostics

Lower Borda score is better. These are cross-dataset ranking diagnostics from `reports/forecasting_borda.json`, not raw-scale metric averages.

| Metric | Rank 1 | Rank 2 | Rank 3 | Full order |
|---|---|---|---|---|
| MAE | TiRex 1.660 | Chronos-2 2.212 | TimesFM 3.230 | TiRex 1.660, Chronos-2 2.212, TimesFM 3.230, PatchTST 4.721, LSTM 4.972, TCN 5.276, DeepONet 5.928 |
| R²_prod | TiRex 2.588 | Chronos-2 3.077 | TimesFM 3.442 | TiRex 2.588, Chronos-2 3.077, TimesFM 3.442, PatchTST 4.153, LSTM 4.527, TCN 4.688, DeepONet 5.524 |
| MASE | PatchTST 2.619 | LSTM 3.171 | TCN 3.427 | PatchTST 2.619, LSTM 3.171, TCN 3.427, TiRex 4.179, DeepONet 4.240, Chronos-2 4.701, TimesFM 5.662 |

## T5 — Forecasting sparse h90 per-well exclusions

These are expected-but-missing sparse rows from `reports/forecasting_performance_audit/forecasting_coverage_audit.csv` and `forecasting_hpc_sync_summary.md`; they are data-coverage exclusions, not active HPC failures.

| Dataset | Horizon | Mode | Missing scenarios | Missing model rows | Reason |
|---|---:|---|---|---:|---|
| inner_mongolia | 90 | per_well | 57-14X, 57-15X | 14 | Too sparse for valid h90 per-well temporal split. |
| spe_berg | 90 | per_well | well_11, well_2 | 14 | Too sparse for valid h90 per-well temporal split. |
| volve | 90 | per_well | NO_15_9-F-5_AH | 7 | Too sparse for valid h90 per-well temporal split. |

## T6 — CDF trained reconstruction metrics

Lower is better. Source: `reports/cdf_post_fix_summary_2026-05.json`. Compare these rows only within trained reconstruction semantics.

| Model | error_mean | error_p50 | error_p95 | error_p99 | Elapsed |
|---|---:|---:|---:|---:|---:|
| LSTM | 0.005878 | 0.005289 | 0.007909 | 0.015403 | 65.5s |
| PatchTST | 0.081621 | 0.071603 | 0.150840 | 0.271780 | 116.7s |
| DeepONet | 0.230735 | 0.223694 | 0.414683 | 0.425859 | 44.6s |

## T7 — CDF foundation forecast metrics

Lower is better. Source: `reports/cdf_post_fix_summary_2026-05.json`. Compare these rows only within one-step foundation forecast semantics.

| Model | forecast_error_mean | forecast_error_p50 | forecast_error_p95 | forecast_error_p99 | Elapsed |
|---|---:|---:|---:|---:|---:|
| Chronos-2 | 0.243968 | 0.226866 | 0.457304 | 0.565650 | 1243.4s |
| TiRex | 0.268548 | 0.246441 | 0.501056 | 0.695557 | 220.7s |
| TimesFM | 0.295999 | 0.283747 | 0.496752 | 0.589213 | 21.8s |

## Table usage notes

- T1 and T2 answer different classification questions; never pool them.
- T4 rankings are metric-specific; do not infer one universal forecasting winner.
- T6 and T7 use different anomaly semantics; never pool them into one CDF ranking.
- `results/pre_fix/` remains audit history only.
