# Dissertation Claim Ledger — 2026-05

Source revision: `44402db9fc6eef8bc731c57445bbb9c8f07b5319`
Purpose: map every dissertation-facing benchmark claim to a metric, source artifact, caveat, and validity status. Claims in this file are evidence controls for the final MSc write-up; prose should cite a claim ID rather than restating untracked assumptions.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| Current | Valid current evidence for dissertation tables/conclusions. |
| Current with caveat | Valid only with the named caveat or reporting separation. |
| Diagnostic | Useful for interpretation but not a primary headline metric. |
| Audit only | Historical, failed, invalid, or non-comparable output; do not use as current benchmark evidence. |

## Claim ledger

| Claim ID | Section | Claim | Metric / basis | Source artifact | Commit | Caveat | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C-3W-S1-001 | 3W Stage 1 classification | Random Forest is the strongest standard 720-window classifier. | Held-out macro-F1 = 0.968972; accuracy = 0.970228; 30 HPO trials. | `README.md:48-62`; `results/README.md:26-38`; canonical HPO path `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` on HPC. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Applies only to the Stage 1 standard 720-window benchmark. | Current |
| C-3W-S1-002 | 3W Stage 1 classification | DeepONet, MambaSL, and LSTM are competitive but do not beat RF in Stage 1. | Macro-F1: DeepONet 0.962579, MambaSL 0.962185, LSTM 0.960046 vs RF 0.968972. | `README.md:52-62`; `results/README.md:30-38`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Do not reinterpret as architecture inferiority outside this split/window. | Current |
| C-3W-S2-001 | 3W Stage 2 follow-ups | `window360_rf` is the best Stage 2 follow-up result. | Macro-F1 = 0.987797; accuracy = 0.991537. | `README.md:64-80`; `results/README.md:40-59`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Different window length; report separately from Stage 1. | Current with caveat |
| C-3W-S2-002 | 3W Stage 2 follow-ups | Stage 2 feature/window variants must not be merged into the Stage 1 apples-to-apples leaderboard. | Protocol separation rule. | `README.md:64-80`; `results/README.md:40-59`; `reports/README.md:58-66`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Stage 2 answers follow-up questions, not the standard benchmark ranking. | Current with caveat |
| C-3W-S2-003 | 3W Stage 2 follow-ups | Failed/invalid Stage 2 raw deep outputs are audit-only. | Failed/collapsed outputs: HydraRocket RAM failure; raw deep variants macro-F1 0.027287 / accuracy 0.157991. | `README.md:80`; `results/README.md:56-59`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Do not use as current performance evidence. | Audit only |
| C-FC-GAN-001 | Ganymede forecasting | TiRex has the best Ganymede multi-well MAE/RMSE and best R² in the current summary. | MAE 0.3620; RMSE 1.2481; R² 0.3537. | `README.md:82-96`; `results/README.md:61-75`; `results/post_fix/forecasting_summary.csv`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Ganymede-only multi-well aggregate; R²/R²_prod are diagnostics. | Current |
| C-FC-GAN-002 | Ganymede forecasting | LSTM and TCN are strongest by grouped/scaled MASE in the current Ganymede summary. | MASE: LSTM 0.0228; TCN 0.0234; DeepONet 0.0301. | `README.md:86-96`; `results/README.md:65-75`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Metric-specific; does not overturn MAE/RMSE conclusion. | Current with caveat |
| C-FC-FULL-001 | Full forecasting campaign | Current post-fix forecasting evidence covers Ganymede, SPE Berg, Volve, and Inner Mongolia. | 2,737 aggregate rows; seven models; four horizons; multi-well and per-well modes. | `README.md:98-100`; `reports/forecasting_performance_audit/forecasting_hpc_sync_summary.md:12-18`; `results/post_fix/forecasting_summary.csv`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Full per-model JSON tree is archived outside ordinary Git. | Current |
| C-FC-FULL-002 | Full forecasting campaign | Multi-well forecasting coverage is complete for all four forecasting datasets. | 28/28 model × horizon multi-well artifacts for each dataset. | `README.md:102-109`; `reports/forecasting_performance_audit/forecasting_hpc_sync_summary.md:20-27`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Per-well h90 sparse exclusions still apply. | Current |
| C-FC-BORDA-001 | Full forecasting campaign | TiRex leads cross-dataset MAE Borda diagnostics. | MAE Borda: TiRex 1.660, Chronos-2 2.212, TimesFM 3.230. | `README.md:111-117`; `reports/forecasting_borda.json`; `reports/forecasting_performance_audit/forecasting_hpc_sync_summary.md:39-47`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Borda score is a ranking diagnostic, not a raw-scale error average. | Diagnostic |
| C-FC-BORDA-002 | Full forecasting campaign | TiRex leads cross-dataset R²_prod Borda diagnostics. | R²_prod Borda: TiRex 2.588, Chronos-2 3.077, TimesFM 3.442. | `README.md:111-117`; `reports/forecasting_borda.json`; `reports/forecasting_performance_audit/forecasting_hpc_sync_summary.md:39-47`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | R²/R²_prod are diagnostics and unstable across wells. | Diagnostic |
| C-FC-BORDA-003 | Full forecasting campaign | PatchTST, LSTM, and TCN lead cross-dataset MASE Borda diagnostics. | MASE Borda: PatchTST 2.619, LSTM 3.171, TCN 3.427. | `README.md:111-117`; `reports/forecasting_borda.json`; `reports/forecasting_performance_audit/forecasting_hpc_sync_summary.md:39-47`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Metric-specific; do not combine with MAE/R²_prod into one universal winner. | Diagnostic |
| C-FC-SPARSE-001 | Forecasting sparse exclusions | Remaining expected missing h90 per-well rows are data-coverage exclusions, not active HPC failures. | Missing scenarios: Inner Mongolia 57-14X/57-15X; SPE Berg well_11/well_2; Volve NO_15_9-F-5_AH. | `README.md:119`; `reports/forecasting_performance_audit/forecasting_hpc_sync_summary.md:29-37`; `reports/forecasting_performance_audit/forecasting_coverage_audit.csv`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Do not relaunch as ordinary failed jobs unless dataset/split policy changes. | Current with caveat |
| C-FC-ARTIFACT-001 | Forecasting artifacts | The full forecasting result tree should stay outside ordinary Git. | 2,784 files; 9,699,091,075 bytes; SHA256 manifest tracked. | `reports/forecasting_performance_audit/forecasting_hpc_sync_summary.md:12-18`; `reports/forecasting_performance_audit/forecasting_post_fix_sha256_manifest.txt`; `results/README.md:77-100`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Use external archive/checksum for bulk artifacts. | Current |
| C-CDF-001 | CDF anomaly detection | CDF trained reconstruction rows must be reported separately from foundation forecast rows. | Different anomaly semantics: reconstruction `error_*` vs one-step `forecast_error_*`. | `README.md:121-144`; `results/README.md:102-122`; `reports/README.md:58-66`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Do not pool into a universal anomaly ranking. | Current with caveat |
| C-CDF-002 | CDF anomaly detection | LSTM leads the trained reconstruction CDF table. | error_mean 0.005878; error_p50 0.005289; error_p95 0.007909; error_p99 0.015403. | `README.md:125-131`; `results/README.md:106-112`; `reports/cdf_post_fix_summary_2026-05.json`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Compare only against trained reconstruction models. | Current |
| C-CDF-003 | CDF anomaly detection | Among foundation forecast CDF rows, Chronos has the lowest forecast_error_mean/p50/p95/p99 in the current table. | forecast_error_mean 0.243968; p50 0.226866; p95 0.457304; p99 0.565650. | `README.md:133-144`; `results/README.md:114-122`; `reports/cdf_post_fix_summary_2026-05.json`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Compare only against foundation forecast rows. | Current |
| C-HIST-001 | Historical validity | `results/pre_fix/` outputs are audit history, not current benchmark evidence. | Validity epoch separation. | `results/README.md:124-131`; `reports/README.md:58-66`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Do not use pre-fix rows for current dissertation claims unless explicitly revalidated. | Audit only |
| C-HIST-002 | Historical validity | Old forecasting MASE/grouped MASE values are non-authoritative unless rerun with repaired chronological MASE plumbing. | MASE validity repair caveat. | `results/README.md:124-131`; `reports/forecasting_performance_audit/forecasting_comparability_ledger.md`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Use post-fix forecasting artifacts and effective MASE audits. | Audit only |
| C-HIST-003 | Historical validity | Old CDF zero-gap production CV outputs are non-authoritative. | Strict raw-row gap repair required cv_gap/inner_gap/outer_gap = 47. | `README.md:121-123`; `results/README.md:124-131`; `reports/cdf_post_fix_summary_2026-05.json`. | `44402db9fc6eef8bc731c57445bbb9c8f07b5319` | Use current post-fix CDF summary only. | Audit only |

## Dissertation writing rules derived from the ledger

1. Never write “best model” without adding dataset family, metric, and benchmark family.
2. Use Stage 1 3W as the standard classification benchmark; use Stage 2 only as follow-up evidence.
3. For forecasting, separate absolute-error claims from normalized/scaled-error claims.
4. For CDF, use two tables: trained reconstruction and foundation forecast.
5. Treat sparse h90 per-well rows as documented data-coverage exclusions, not compute failures.
6. Treat `pre_fix/` and failed/collapsed Stage 2 rows as audit lineage only.

## Validation checklist

- [x] Every headline result claim has a source artifact.
- [x] Every winner claim names a metric.
- [x] Stage 1 and Stage 2 3W are separated.
- [x] CDF reconstruction and forecast semantics are separated.
- [x] Forecasting sparse exclusions are explicitly named.
- [x] Heavy forecasting JSON tree policy is documented.
