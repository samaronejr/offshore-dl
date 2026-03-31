# reports/

Result tables, statistical tests, and analysis reports.

## LaTeX Tables

| File | Description |
|------|-------------|
| `all_results.tex` | **Complete benchmark** — 14 tables covering all 8 models × 3 datasets: 3W classification, Ganymede multi-horizon forecasting (test + CV + per-well), CDF anomaly detection, and cross-dataset summary |
| `all_results.pdf` | Compiled PDF of the above |
| `xgboost_results.tex` | XGBoost-specific detailed tables (subset of `all_results.tex`) |
| `xgboost_results.pdf` | Compiled PDF of the above |

## Statistical Tests

| File | Description |
|------|-------------|
| `statistical_tests_nested.json` | Friedman test + Nemenyi post-hoc + pairwise Wilcoxon signed-rank tests. Covers 3W (5 models, 5-fold), Ganymede (7 models, 3-fold, 4 horizons), CDF (6 models, 3-fold). |

## Historical Reports

| File | Description |
|------|-------------|
| `baseline_results_report.md` | Snapshot of pre-HPO baseline results (historical — predates MLP/XGBoost) |
| `audit_fix_report.md` | Post-merge code audit: 30 findings fixed (4 critical, 14 high, 12 medium) |
| `post_merge_audit_log.md` | Log of production re-runs after audit fixes |

## Regenerating

```bash
# Statistical tests (reads from results/, writes statistical_tests_nested.json)
python scripts/run_statistical_tests.py

# Compile LaTeX → PDF
tectonic reports/all_results.tex
```
