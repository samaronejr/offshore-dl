# 3W Stage 2 window/feature audit — May 2026

Generated: 2026-05-15T03:20:16Z
Source revision: `62780b2dd5aa42a01b0cdb1ce07a7cf6416cea5b`
Evidence root: `LPS_loginServer:/home/samarone.lima/offshore-dl/results/stage2_3w/3w-stage2-20260513T192623Z/`
Script anchors: `scripts/run_production_3w_features.py:180-192`, `scripts/run_production_3w_features.py:265`, `scripts/run_production_3w_features.py:1289-1317`.

## Verdict

`window360_rf` and `window1440_rf` are valid Stage 2 follow-up results, but they are **not** replacements for the Stage 1 standard 720-window 3W leaderboard because they use different window lengths and separate data configs. They may be reported as window-length ablations only.

## Implementation evidence

- `window360_rf` uses `configs/data/3w_window_360.yaml` with `window_size=360`.
- `window1440_rf` uses `configs/data/3w_window_1440.yaml` with `window_size=1440`.
- Both variants are isolated under `WINDOW_MODELS = ["window360_rf", "window1440_rf"]`.
- Holdout uses `mode="stratified_group"` with `instance_id` groups, so windows from the same event group are not split across train/test.
- Outputs are written as `<STAGE2_ROOT>/<variant>/3w.json`.

## Result inventory

| Variant | Family | Macro-F1 | Accuracy | Train windows | Test windows | CV folds | CV macro-F1 mean ± std | Reporting caveat |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `window360_rf` | window-length RF | 0.987797 | 0.991537 | 168899 | 42301 | 5 | 0.962753 ± 0.007159 | Valid separate-window follow-up; do not compare as same 720-window Stage 1 protocol. |
| `window1440_rf` | window-length RF | 0.977011 | 0.989084 | 163569 | 40950 | 5 | 0.967167 ± 0.013009 | Valid separate-window follow-up; do not compare as same 720-window Stage 1 protocol. |
| `wavelet_rf` | feature/model follow-up | 0.964309 | 0.966301 | 167458 | 41515 | 5 | 0.966873 ± 0.014308 | Valid Stage 2 follow-up; separate from Stage 1 HPO. |
| `multiscale_rf` | feature/model follow-up | 0.964184 | 0.966205 | 167458 | 41515 | 5 | 0.962612 ± 0.015364 | Valid Stage 2 follow-up; separate from Stage 1 HPO. |
| `physics_rf` | feature/model follow-up | 0.964070 | 0.966060 | 167458 | 41515 | 5 | 0.964747 ± 0.015457 | Valid Stage 2 follow-up; separate from Stage 1 HPO. |
| `wavelet_deeponet` | feature/model follow-up | 0.961252 | 0.967867 | 167458 | 41515 | 5 | 0.943519 ± 0.019610 | Valid Stage 2 follow-up; separate from Stage 1 HPO. |
| `convtran` | feature/model follow-up | 0.955762 | 0.964398 | 167458 | 41515 | 5 | 0.913343 ± 0.028352 | Valid Stage 2 follow-up; separate from Stage 1 HPO. |
| `multiscale_deeponet` | feature/model follow-up | 0.954235 | 0.964880 | 167458 | 41515 | 5 | 0.933229 ± 0.016434 | Valid Stage 2 follow-up; separate from Stage 1 HPO. |
| `physics_deeponet` | feature/model follow-up | 0.952286 | 0.961652 | 167458 | 41515 | 5 | 0.928817 ± 0.019338 | Valid Stage 2 follow-up; separate from Stage 1 HPO. |
| `convtimenet_raw` | raw-window failed baseline | 0.027287 | 0.157991 | 167458 | 41515 | 5 | 0.563444 ± 0.438073 | Collapsed raw-window baseline; preserve as failed/invalid evidence, not headline result. |
| `convtran_raw` | raw-window failed baseline | 0.027287 | 0.157991 | 167458 | 41515 | 5 | 0.899481 ± 0.044640 | Collapsed raw-window baseline; preserve as failed/invalid evidence, not headline result. |
| `fkmad_raw` | raw-window failed baseline | 0.027287 | 0.157991 | 167458 | 41515 | 5 | 0.201781 ± 0.355572 | Collapsed raw-window baseline; preserve as failed/invalid evidence, not headline result. |
| `mambasl_raw` | raw-window failed baseline | 0.027287 | 0.157991 | 167458 | 41515 | 5 | 0.898605 ± 0.026231 | Collapsed raw-window baseline; preserve as failed/invalid evidence, not headline result. |

## Leakage and reporting checks

- Split assumption: stratified-group holdout over `instance_id` groups; no evidence of event-group mixing in the Stage 2 window path.
- Window-family caveat: `window360_rf` and `window1440_rf` use different sample counts (`42301` and `40950` test windows) from standard Stage 1 (`41515` held-out windows), so their macro-F1 values must be labeled as window-length follow-ups.
- Invalid/failed statuses remain visible: `hydra_rocket` failed from impractical RAM allocation; raw deep variants collapsed to macro-F1 `0.027287` / accuracy `0.157991`.
