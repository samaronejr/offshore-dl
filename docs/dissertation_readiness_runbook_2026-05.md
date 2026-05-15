# Dissertation readiness runbook — May 2026

Generated: 2026-05-15T03:20:16Z
Local branch: `main`
Local source revision: `62780b2dd5aa42a01b0cdb1ce07a7cf6416cea5b`
Remote execution host/path: `LPS_loginServer:/home/samarone.lima/offshore-dl`
Ralph plan: `.omx/plans/ralplan-dissertation-readiness-next-steps-20260515.md`

## Scope and constraints

This runbook executes the CDF-gated dissertation-readiness pass. It does **not** add new models, run extra seeds, edit raw/processed data, or cherry-pick only favorable metrics. Accepted, failed, invalid, pending, and blocked artifacts remain visible.

## Source and artifact inventory

| Artifact family | Local status | Remote status | Canonical/current path | Notes |
|---|---|---|---|---|
| 3W Stage 1 HPO | `blocked-missing-local` | `available-remote` | `/home/samarone.lima/offshore-dl/results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` | Seven accepted models, 30 Optuna trials/model, standard 720-window classification. |
| 3W Stage 2 follow-ups | `blocked-missing-local` | `available-remote` | `/home/samarone.lima/offshore-dl/results/stage2_3w/3w-stage2-20260513T192623Z/` | Separate feature/window family; not pooled into Stage 1. |
| Ganymede post-fix | `blocked-missing-local` | `available-remote` | `/home/samarone.lima/offshore-dl/results/post_fix/<model>/ganymede_h*_multi_well.json` | Four multi-well horizons per current model. |
| CDF post-fix | `blocked-missing-local` | `available-remote` | `/home/samarone.lima/offshore-dl/results/post_fix/<model>/cdf.json` | HPC job `28934` completed; compact summary copied to `reports/cdf_post_fix_summary_2026-05.json`. |

Local result roots intentionally remain incomplete in this working tree; remote result paths are the current evidence roots until explicitly synced or regenerated locally.

## Execution fixes applied during Ralph

| Fix | Files | Evidence |
|---|---|---|
| Disable MLflow for CDF Slurm reruns | `scripts/run_production_cdf.py`, `scripts/slurm_rerun_cdf.sh`, `scripts/slurm_cdf.sh` | `--no-mlflow` appears in `python scripts/run_production_cdf.py --help`; targeted CDF production tests pass. |
| Bind live source into CDF Singularity jobs | `scripts/slurm_rerun_cdf.sh`, `scripts/slurm_cdf.sh` | Wrappers bind `$PROJECT/src`, `$PROJECT/scripts`, `$PROJECT/configs` into `/app`, avoiding stale in-image script behavior. |
| Enforce CDF metric-semantics guard | `scripts/run_production_cdf.py`, `scripts/run_statistical_tests.py` | FM CDF metrics use `forecast_error_*`; statistical tests separate trained reconstruction from foundation forecast groups. |

## CDF gate status

- Clean rerun job `28934` completed on `LPS_loginServer`.
- All expected CDF models completed with status `ok`: `lstm`, `deeponet`, `patchtst`, `chronos`, `timesfm`, `tirex`.
- Schema validation passed: finite `test_metrics`, three `cv_fold_results` per model, and `cv_gap_policy=strict_raw_row` with `inner_gap=outer_gap=47`.
- CDF metric semantics are separated: trained models use `error_*`; foundation models use `forecast_error_*`.

## Blocking policy

Final statistical/report freeze proceeded after these gates passed:

1. CDF post-fix summary exists and records each expected model as `ok` or `error`.
2. Successful CDF JSONs contain finite `test_metrics`, non-empty `cv_fold_results`, and strict raw-row split metadata.
3. 3W Stage 2 audit exists and preserves the separate-family caveat.
4. Dissertation result manifest exists and represents accepted, failed, invalid, blocked, and pending rows.
5. CDF trained-vs-FM metric semantics are explicit before any statistical claim.

## Working-tree status policy

This runbook is an execution artifact, not the source of truth for the final
diff. Use `git status --short` in the verification evidence for the current
file list. No raw or processed data files were modified during this pass.

## Deferred backlog

- Repeat seeds for committee robustness requests.
- Optional rerun/validation of `window360_rf` if robustness beyond current split is required.
- Broader thesis prose rewrite after final result freeze.
