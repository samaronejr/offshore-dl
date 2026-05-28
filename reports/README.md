# reports/

Result tables, statistical tests, and analysis reports. Current dissertation-facing artifacts are CDF-gated. Historical/pre-fix outputs are retained only for audit lineage.

## Current report status

| File | Status | Notes |
|------|--------|-------|
| `dissertation_results_2026-05.tex` / `dissertation_results_2026-05.pdf` | Current dissertation snapshot | CDF-gated aggregate snapshot generated from manifest and post-fix summaries. |
| `dissertation_result_manifest_2026-05.md` | Current claim manifest | Maps headline rows to source revision, result path, status, and caveat. |
| `cdf_post_fix_summary_2026-05.json` | Current CDF summary | Compact post-fix CDF summary from HPC job `28934`. |
| `statistical_tests_nested.json` | Current for Ganymede and CDF post-fix roots | CDF uses separated trained reconstruction and foundation forecast groups; 3W Stage 2 is not pooled. |
| `forecasting_borda.json` | Current full post-fix forecasting diagnostic | Regenerated from `results/post_fix` on HPC across Ganymede, SPE Berg, Volve, and Inner Mongolia; lower Borda score is better. |
| `forecasting_performance_audit/forecasting_hpc_sync_summary.md` | Current forecasting provenance summary | Documents jobs `30271`/`30272`, local sync size, coverage, sparse exclusions, and key artifact hashes. |
| `forecasting_performance_audit/forecasting_post_fix_sha256_manifest.txt` | Current result-tree checksum manifest | SHA256 manifest for the synced `results/post_fix/` tree; the large JSON tree itself is not intended for Git. |
| `all_results.tex` / `all_results.pdf` | Historical aggregate | Retained, but not the current CDF-gated dissertation snapshot. |
| `audit_fix_report.md` | Current audit-history documentation | Describes code fixes, not new benchmark rankings. |
| `post_merge_audit_log.md` | Historical rerun log | Useful for provenance, not a current leaderboard. |
| `baseline_results_report.md` | Historical | Pre-HPO/pre-fix baseline snapshot. |

## Current result sources for regenerated reports

- 3W Stage 1 HPO headline: `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` (available on HPC).
- 3W Stage 2 follow-ups: `results/stage2_3w/3w-stage2-20260513T192623Z/` (available on HPC; separate family).
- Forecasting post-fix campaign: `results/post_fix/<model>/{ganymede,spe_berg,volve,inner_mongolia}*.json` plus aggregate CSVs under `results/post_fix/` (available on HPC and synced locally for the 2026-05-27 audit).
- CDF post-fix anomaly detection: `results/post_fix/<model>/cdf.json` from HPC job `28934`.

## Statistical tests

`statistical_tests_nested.json` was regenerated from `OFFSHORE_DL_RESULTS_DIR=results/post_fix`. It includes:

1. Ganymede post-fix forecasting fold tests.
2. CDF trained reconstruction tests over `error_mean` / `error_p50`.
3. CDF foundation forecast tests over `forecast_error_mean` / `forecast_error_p50`.

It intentionally does not create one pooled trained-vs-foundation CDF ranking. Stage 2 3W variants also remain separate from Stage 1.

## Regenerating

```bash
# CDF rerun on HPC
sbatch scripts/slurm_rerun_cdf.sh

# Statistical tests from explicit post-fix root
OFFSHORE_DL_RESULTS_DIR=results/post_fix python scripts/run_statistical_tests.py

# Forecasting aggregate/Borda from explicit post-fix root
OFFSHORE_DL_RESULTS_DIR=results/post_fix python scripts/aggregate_forecasting_results.py

# Compile dissertation snapshot
tectonic reports/dissertation_results_2026-05.tex
```

## Caveats for writing dissertation text

- Treat `pre_fix/` outputs as audit history, not current benchmark evidence.
- State that 3W uses macro-F1 as the primary classification metric.
- Keep Stage 1 and Stage 2 3W results in separate tables.
- For forecasting, name the metric before naming a winner: TiRex leads cross-dataset MAE and R²_prod Borda diagnostics, while PatchTST/LSTM/TCN are strongest by MASE Borda after the full post-fix sync.
- Treat sparse h90 per-well exclusions as data-coverage caveats, not failed compute: Inner Mongolia `57-14X`/`57-15X`, SPE Berg `well_11`/`well_2`, and Volve `NO_15_9-F-5_AH`.
- For CDF, compare trained reconstruction rows separately from foundation forecast rows.
- Use R² and R²_prod as diagnostics because they remain unstable across wells.
