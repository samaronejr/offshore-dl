# reports/

Result tables, statistical tests, and analysis reports. Some reports predate the latest post-fix HPO/rerun campaigns; check the validity notes below before citing them.

## Current report status

| File | Status | Notes |
|------|--------|-------|
| `all_results.tex` / `all_results.pdf` | Needs regeneration before final dissertation use | May not include the validated 3W Stage 1 HPO, 3W Stage 2 follow-ups, or post-fix Ganymede aggregate framing. |
| `statistical_tests_nested.json` | Needs regeneration before final statistical claims | Do not pool 3W Stage 2 variants with Stage 1 models unless grouped as a separate experiment family. |
| `forecasting_borda.json` | Historical/diagnostic until regenerated | Verify against `results/post_fix/` before citing. |
| `audit_fix_report.md` | Current as audit-history documentation | Describes code fixes, not new benchmark rankings. |
| `post_merge_audit_log.md` | Historical rerun log | Useful for provenance, not a current leaderboard. |
| `baseline_results_report.md` | Historical | Pre-HPO/pre-fix baseline snapshot. |
| `experimental_findings_3w_improvements.md` | Needs review against latest Stage 1/Stage 2 results | Update if reused in dissertation text. |
| `ganymede_improvement_results.md` | Needs review against post-fix Ganymede summaries | Update if reused in dissertation text. |

## Current result sources for regenerated reports

- 3W Stage 1 HPO headline: `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json`
- 3W Stage 2 follow-ups: `results/stage2_3w/3w-stage2-20260513T192623Z/`
- Ganymede post-fix forecasting: `results/post_fix/<model>/ganymede*.json`
- CDF anomaly detection: rerun with `scripts/run_production_cdf.py` before making final post-fix CDF claims.

## LaTeX tables

| File | Description |
|------|-------------|
| `all_results.tex` | Complete benchmark tables. Regenerate after result validation so it reflects current 3W/Ganymede/CDF status. |
| `all_results.pdf` | Compiled PDF of `all_results.tex`. |
| `ieee_paper.tex` / `ieee_paper.pdf` | Paper draft artifacts. Verify numbers before submission. |
| `xgboost_results.tex` / `xgboost_results.pdf` | XGBoost-specific report artifacts. |

## Statistical tests

| File | Description |
|------|-------------|
| `statistical_tests_nested.json` | Friedman/Nemenyi/Wilcoxon output. Regenerate from the validated result matrix before final claims. |

Recommended grouping for new tests:

1. 3W Stage 1 standard 720-window models.
2. 3W Stage 2 feature/window variants as a separate family.
3. Ganymede post-fix forecasting models, with MAE/RMSE and MASE reported separately.
4. CDF only after a post-fix rerun exists.

## Regenerating

```bash
# Validate HPO campaign before promoting 3W Stage 1 values
python scripts/validate_hpo_3w_results.py --campaign-id <campaign-id> --write-summary

# Statistical tests (reads result JSONs and writes statistical_tests_nested.json)
python scripts/run_statistical_tests.py

# Compile LaTeX → PDF
tectonic reports/all_results.tex
```

## Caveats for writing dissertation text

- Treat `pre_fix/` outputs as audit history, not current benchmark evidence.
- State that 3W uses macro-F1 as the primary classification metric.
- Keep Stage 1 and Stage 2 3W results in separate tables.
- For Ganymede, foundation models lead by MAE/RMSE, while LSTM/TCN lead by MASE; do not imply one universal winner without naming the metric.
- Use R² and R²_prod as diagnostics because they remain unstable across wells.
