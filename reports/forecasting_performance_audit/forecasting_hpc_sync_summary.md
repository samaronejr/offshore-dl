# Forecasting HPC Sync Summary — 2026-05-27

This is the local provenance note for the final synced forecasting artifacts from `LPS_loginServer:/home/samarone.lima/offshore-dl` after the split TCN Inner Mongolia h90 recovery job.

## Source jobs

| Job | Purpose | Verified state |
| --- | --- | --- |
| `30271` | Split retry for `tcn inner_mongolia h90` multi-well plus sparse per-well slices | all array tasks reported `COMPLETED` in `sacct` |
| `30272` | Forecasting postprocess after `30271` | `COMPLETED`; wrote aggregate summary, Borda, and audit artifacts |

## Local artifact root

- Result root: `results/post_fix/`
- Result files: 2,784
- Result tree size: 9,699,091,075 bytes (9.03 GiB)
- Full SHA256 manifest: `reports/forecasting_performance_audit/forecasting_post_fix_sha256_manifest.txt` (2,784 files)
- Aggregate row count: 2,737 rows in `results/post_fix/forecasting_summary.csv`

## Multi-well coverage

| Dataset | Multi-well artifacts | Expected model × horizon artifacts |
| --- | ---: | ---: |
| Ganymede | 28 | 28 |
| SPE Berg | 28 | 28 |
| Volve | 28 | 28 |
| Inner Mongolia | 28 | 28 |

## Remaining expected-but-missing sparse h90 per-well rows

These rows are data-coverage exclusions, not active HPC failures. The `tcn inner_mongolia h90 per_well 57-14X` retry logged only one usable sample before the temporal split and zero train/test samples after splitting.

| Dataset | Horizon | Mode | Missing scenarios | Missing model rows |
| --- | ---: | --- | --- | ---: |
| inner_mongolia | 90 | per_well | 57-14X, 57-15X | 14 |
| spe_berg | 90 | per_well | well_11, well_2 | 14 |
| volve | 90 | per_well | NO_15_9-F-5_AH | 7 |

## Cross-dataset Borda diagnostics

Lower Borda score is better. Use these as cross-dataset ranking diagnostics, not as raw-scale metric averages.

| Metric | Ranking |
| --- | --- |
| MAE | tirex 1.660, chronos 2.212, timesfm 3.230, patchtst 4.721, lstm 4.972, tcn 5.276, deeponet 5.928 |
| R2_prod | tirex 2.588, chronos 3.077, timesfm 3.442, patchtst 4.153, lstm 4.527, tcn 4.688, deeponet 5.524 |
| MASE | patchtst 2.619, lstm 3.171, tcn 3.427, tirex 4.179, deeponet 4.240, chronos 4.701, timesfm 5.662 |

## Key aggregate artifact hashes

| Artifact | SHA256 |
| --- | --- |
| `results/post_fix/forecasting_summary.csv` | `4205912a1dd450723e2900eb9bd3a22b19e9b33a607f27714098d66b7f001779` |
| `results/post_fix/forecasting_summary_wide_mae.csv` | `749f32a8bf7e116a8a7076e69d0257b0b1413a115476065d35321811dabb8b5c` |
| `results/post_fix/forecasting_summary_wide_r2_prod.csv` | `3f04eeb69370137c6d949e0519ca7fc9b0f20510eccefefe1df4cd2dd1b0b01c` |
| `reports/forecasting_borda.json` | `07d22f9cdff72849ffdd4ec775b4b7686833ba14a56a8782ac291e5aa41bfdc5` |
| `reports/forecasting_performance_audit/forecasting_coverage_audit.csv` | `1af935a4a0c3fe21d13ef1f3dfa8b1fad780d9348d3afb1a091430284372bac4` |
| `reports/forecasting_performance_audit/forecasting_result_rows_with_effective_mase.csv` | `14578a7e243414176c4bf34ef7de42628e4cca99091cc8831df6b2d6f3c4ce7d` |

## Git policy

The full `results/post_fix/` JSON tree is intentionally kept out of the commit because it is large (~9.1 GiB locally). Commit lightweight reports, manifests, and provenance summaries instead; archive or transfer the result tree externally when needed.
