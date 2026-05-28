# Dissertation Results Narrative Guide — 2026-05

Use this guide to turn the final benchmark tables into dissertation prose. The source-of-truth mapping is `reports/dissertation_claim_ledger_2026-05.md`; the table artifact is `reports/dissertation_final_tables_2026-05.md`.

## Core thesis

The benchmark supports a metric-specific, protocol-separated conclusion: classical tree models are strongest for the validated 3W classification benchmark, zero-shot foundation models dominate absolute forecasting-error diagnostics, trained/deep temporal models remain competitive under scaled forecasting diagnostics, and CDF anomaly detection must be reported under separate reconstruction-vs-forecast semantics.

## Classification narrative

For the standard 720-window 3W classification benchmark, Random Forest is the strongest current model by held-out macro-F1 (0.968972) and accuracy (0.970228). DeepONet, MambaSL, and LSTM remain competitive, but they do not exceed Random Forest under this validated Stage 1 split.

Stage 2 should be described as follow-up exploration, not as a replacement leaderboard. `window360_rf` achieves the strongest follow-up result (macro-F1 0.987797), but the shorter window changes the task setup and therefore must be reported separately from Stage 1. Failed/collapsed raw deep variants are audit lineage only.

## Forecasting narrative

The full post-fix forecasting campaign covers Ganymede, SPE Berg, Volve, and Inner Mongolia with complete multi-well model × horizon coverage. The safest forecasting conclusion is metric-specific:

- By cross-dataset MAE Borda diagnostics, TiRex ranks first, followed by Chronos-2 and TimesFM.
- By cross-dataset R²_prod Borda diagnostics, TiRex also ranks first, followed by Chronos-2 and TimesFM.
- By cross-dataset MASE Borda diagnostics, PatchTST, LSTM, and TCN rank highest.

Do not collapse these into one universal forecasting winner. MAE/RMSE answer absolute production-scale error; MASE answers normalized/scaled error; R²/R²_prod are diagnostics and remain unstable across wells.

For Ganymede specifically, TiRex leads the current multi-well aggregate on MAE/RMSE/R², while LSTM/TCN lead MASE-style scaled diagnostics. Present this as an example of metric-dependent interpretation.

Sparse h90 per-well exclusions must be stated in any per-well forecasting table: Inner Mongolia (`57-14X`, `57-15X`), SPE Berg (`well_11`, `well_2`), and Volve (`NO_15_9-F-5_AH`). These are data-coverage exclusions, not failed HPC jobs.

## CDF anomaly narrative

CDF anomaly-detection results must remain split by semantics. Trained models report reconstruction errors (`error_*`), while foundation models report one-step forecast errors (`forecast_error_*`). LSTM leads the trained reconstruction table, and Chronos-2 leads the foundation forecast table by forecast-error metrics. Do not pool the two CDF groups into one universal anomaly ranking.

## Threats to validity to include

1. Current headline benchmark values are primarily single-seed / fixed-campaign evidence.
2. Sparse h90 per-well scenarios cannot support valid temporal-split metrics.
3. Optional foundation-model dependencies and hardware/runtime availability can affect reproducibility.
4. R² and R²_prod are diagnostic rather than headline metrics because well-level production dynamics can make them unstable.
5. Historical `pre_fix/` artifacts are retained for audit lineage but should not be used as current evidence.

## Recommended conclusion wording

> Under the validated post-fix benchmark protocol, Random Forest remains the strongest standard-window 3W classifier by macro-F1. For production forecasting, no single model dominates under all metrics: zero-shot foundation models, especially TiRex, are strongest by absolute-error and productive-period ranking diagnostics, while trained/deep temporal models are strongest under scaled MASE diagnostics. For CDF anomaly detection, reconstruction and forecast-error semantics require separate comparisons, preventing a single pooled anomaly leaderboard.

## Do-not-write list

- Do not write “TiRex is the best forecasting model” without naming MAE/R²_prod Borda and the diagnostic nature of the claim.
- Do not write “PatchTST is the best forecasting model” without naming MASE Borda and the scaled-error context.
- Do not merge Stage 1 and Stage 2 3W tables.
- Do not pool CDF trained reconstruction and foundation forecast rows.
- Do not treat sparse h90 exclusions as failed jobs.
