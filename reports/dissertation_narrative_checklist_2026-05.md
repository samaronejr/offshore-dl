# Dissertation narrative checklist — May 2026

> **Supersession notice (2026-05-28):** This checklist is retained for audit lineage. Current prose should use `reports/dissertation_results_narrative_2026-05.md` plus claim IDs from `reports/dissertation_claim_ledger_2026-05.md`.

Generated: 2026-05-15T03:20:16Z
Source revision: `62780b2dd5aa42a01b0cdb1ce07a7cf6416cea5b`

## Safe headline templates

- **3W Stage 1:** On the standard 720-window 3W benchmark, tuned Random Forest leads by held-out macro-F1 (`0.968972`).
- **3W Stage 2:** Window-length Random Forest follow-ups improve macro-F1 (`window360_rf=0.987797`, `window1440_rf=0.977011`), but they are separate ablations because the window length/sample counts differ from Stage 1.
- **Ganymede:** Zero-shot FMs lead by MAE/RMSE (TiRex best MAE/RMSE), while trained LSTM/TCN lead by grouped MASE; do not state a universal forecasting winner without naming the metric.
- **CDF:** Post-fix job `28934` passed. LSTM leads trained reconstruction by `error_mean`; Chronos leads FM forecast by `forecast_error_mean`. Do not pool trained and FM rows.

## Wording guardrails

- Do not cite `pre_fix/` outputs as current evidence.
- Do not pool Stage 2 3W variants into Stage 1 tables.
- Do not silently omit invalid/failed Stage 2 rows (`hydra_rocket`, raw collapsed variants).
- Do not say CDF is final while `summary_production_cdf.json` or per-model CDF schema validation is missing.
- Use R² and R²_prod as diagnostics for forecasting, not as sole winner criteria.

## Report freeze checklist

- [x] CDF post-fix summary exists and expected model statuses are explicit.
- [x] CDF schema validation passed for every `ok` model.
- [x] 3W Stage 2 window audit created.
- [x] Manifest created with accepted/failed/invalid/pending rows.
- [x] CDF trained-vs-FM metric guard documented.
- [x] Statistical tests regenerated from explicit result root.
- [x] Aggregate report source/PDF regenerated or clearly marked historical.
