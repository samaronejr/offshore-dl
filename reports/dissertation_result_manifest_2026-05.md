# Dissertation result manifest — May 2026

Generated: 2026-05-15T03:20:16Z
Source revision: `62780b2dd5aa42a01b0cdb1ce07a7cf6416cea5b`
Local branch: `main`
Primary evidence status: 3W/Ganymede available on HPC; CDF post-fix job `28934` completed and passed the schema/metric-semantics gate.

Status vocabulary: `accepted`, `accepted-separate-family`, `failed`, `invalid`, `available-remote`.

## 3W Stage 1 HPO — standard 720-window classification

| Dataset | Family | Model | Primary metric | Value | Accuracy | Trials | Result path | Status | Caveat |
|---|---|---|---|---:|---:|---:|---|---|---|
| 3W | Stage 1 HPO standard 720-window | `random_forest` | macro-F1 | 0.968972 | 0.970228 | 30 | `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` / `random_forest.json` | accepted / available-remote | Apples-to-apples Stage 1 leaderboard row. |
| 3W | Stage 1 HPO standard 720-window | `deeponet` | macro-F1 | 0.962579 | 0.968734 | 30 | `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` / `deeponet.json` | accepted / available-remote | Apples-to-apples Stage 1 leaderboard row. |
| 3W | Stage 1 HPO standard 720-window | `mambasl` | macro-F1 | 0.962185 | 0.967313 | 30 | `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` / `mambasl.json` | accepted / available-remote | Apples-to-apples Stage 1 leaderboard row. |
| 3W | Stage 1 HPO standard 720-window | `lstm` | macro-F1 | 0.960046 | 0.965314 | 30 | `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` / `lstm.json` | accepted / available-remote | Apples-to-apples Stage 1 leaderboard row. |
| 3W | Stage 1 HPO standard 720-window | `fkmad` | macro-F1 | 0.956388 | 0.964061 | 30 | `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` / `fkmad.json` | accepted / available-remote | Apples-to-apples Stage 1 leaderboard row. |
| 3W | Stage 1 HPO standard 720-window | `convtimenet` | macro-F1 | 0.954953 | 0.959894 | 30 | `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` / `convtimenet.json` | accepted / available-remote | Apples-to-apples Stage 1 leaderboard row. |
| 3W | Stage 1 HPO standard 720-window | `patchtst` | macro-F1 | 0.953556 | 0.958738 | 30 | `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` / `patchtst.json` | accepted / available-remote | Apples-to-apples Stage 1 leaderboard row. |

## 3W Stage 2 — separate feature/window follow-ups

| Dataset | Family | Variant | Primary metric | Value | Accuracy | Result path | Status | Caveat |
|---|---|---|---|---:|---:|---|---|---|
| 3W | Stage 2 follow-up | `window360_rf` | macro-F1 | 0.987797 | 0.991537 | `results/stage2_3w/3w-stage2-20260513T192623Z/window360_rf/3w.json` | accepted-separate-family / available-remote | Different window length; do not pool with Stage 1. |
| 3W | Stage 2 follow-up | `window1440_rf` | macro-F1 | 0.977011 | 0.989084 | `results/stage2_3w/3w-stage2-20260513T192623Z/window1440_rf/3w.json` | accepted-separate-family / available-remote | Different window length; do not pool with Stage 1. |
| 3W | Stage 2 follow-up | `wavelet_rf` | macro-F1 | 0.964309 | 0.966301 | `results/stage2_3w/3w-stage2-20260513T192623Z/wavelet_rf/3w.json` | accepted-separate-family / available-remote | Stage 2 follow-up; report separately from Stage 1. |
| 3W | Stage 2 follow-up | `multiscale_rf` | macro-F1 | 0.964184 | 0.966205 | `results/stage2_3w/3w-stage2-20260513T192623Z/multiscale_rf/3w.json` | accepted-separate-family / available-remote | Stage 2 follow-up; report separately from Stage 1. |
| 3W | Stage 2 follow-up | `physics_rf` | macro-F1 | 0.964070 | 0.966060 | `results/stage2_3w/3w-stage2-20260513T192623Z/physics_rf/3w.json` | accepted-separate-family / available-remote | Stage 2 follow-up; report separately from Stage 1. |
| 3W | Stage 2 follow-up | `wavelet_deeponet` | macro-F1 | 0.961252 | 0.967867 | `results/stage2_3w/3w-stage2-20260513T192623Z/wavelet_deeponet/3w.json` | accepted-separate-family / available-remote | Stage 2 follow-up; report separately from Stage 1. |
| 3W | Stage 2 follow-up | `convtran` | macro-F1 | 0.955762 | 0.964398 | `results/stage2_3w/3w-stage2-20260513T192623Z/convtran/3w.json` | accepted-separate-family / available-remote | Stage 2 follow-up; report separately from Stage 1. |
| 3W | Stage 2 follow-up | `multiscale_deeponet` | macro-F1 | 0.954235 | 0.964880 | `results/stage2_3w/3w-stage2-20260513T192623Z/multiscale_deeponet/3w.json` | accepted-separate-family / available-remote | Stage 2 follow-up; report separately from Stage 1. |
| 3W | Stage 2 follow-up | `physics_deeponet` | macro-F1 | 0.952286 | 0.961652 | `results/stage2_3w/3w-stage2-20260513T192623Z/physics_deeponet/3w.json` | accepted-separate-family / available-remote | Stage 2 follow-up; report separately from Stage 1. |
| 3W | Stage 2 follow-up | `convtimenet_raw` | macro-F1 | 0.027287 | 0.157991 | `results/stage2_3w/3w-stage2-20260513T192623Z/convtimenet_raw/3w.json` | invalid | Collapsed raw-window baseline; retained as failed/invalid evidence. |
| 3W | Stage 2 follow-up | `convtran_raw` | macro-F1 | 0.027287 | 0.157991 | `results/stage2_3w/3w-stage2-20260513T192623Z/convtran_raw/3w.json` | invalid | Collapsed raw-window baseline; retained as failed/invalid evidence. |
| 3W | Stage 2 follow-up | `fkmad_raw` | macro-F1 | 0.027287 | 0.157991 | `results/stage2_3w/3w-stage2-20260513T192623Z/fkmad_raw/3w.json` | invalid | Collapsed raw-window baseline; retained as failed/invalid evidence. |
| 3W | Stage 2 follow-up | `mambasl_raw` | macro-F1 | 0.027287 | 0.157991 | `results/stage2_3w/3w-stage2-20260513T192623Z/mambasl_raw/3w.json` | invalid | Collapsed raw-window baseline; retained as failed/invalid evidence. |
| 3W | Stage 2 follow-up | `hydra_rocket` | macro-F1 | — | — | `{stage2_path}/hydra_rocket/` | failed | Impractical multi-terabyte RAM allocation; no accepted metric. |

## Ganymede post-fix forecasting — multi-well horizons aggregate

| Dataset | Family | Model | MAE | RMSE | MASE | R² | R²_prod | Result path | Status | Caveat |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| Ganymede | foundation forecasting | `tirex` | 0.361722 | 1.247611 | 0.207086 | 0.354137 | -0.149018 | `results/post_fix/tirex/ganymede_h*_multi_well.json` | accepted / available-remote | Winner depends on metric: FMs lead MAE/RMSE; trained LSTM/TCN lead grouped MASE. |
| Ganymede | foundation forecasting | `timesfm` | 0.396500 | 1.263426 | 0.229494 | 0.336157 | -0.128277 | `results/post_fix/timesfm/ganymede_h*_multi_well.json` | accepted / available-remote | Winner depends on metric: FMs lead MAE/RMSE; trained LSTM/TCN lead grouped MASE. |
| Ganymede | foundation forecasting | `chronos` | 0.535708 | 1.441213 | 0.320461 | 0.111144 | -0.343061 | `results/post_fix/chronos/ganymede_h*_multi_well.json` | accepted / available-remote | Winner depends on metric: FMs lead MAE/RMSE; trained LSTM/TCN lead grouped MASE. |
| Ganymede | trained forecasting | `lstm` | 0.545664 | 1.351740 | 0.022818 | 0.245523 | -0.648115 | `results/post_fix/lstm/ganymede_h*_multi_well.json` | accepted / available-remote | Winner depends on metric: FMs lead MAE/RMSE; trained LSTM/TCN lead grouped MASE. |
| Ganymede | trained forecasting | `tcn` | 0.567727 | 1.313566 | 0.023382 | 0.286406 | -0.409899 | `results/post_fix/tcn/ganymede_h*_multi_well.json` | accepted / available-remote | Winner depends on metric: FMs lead MAE/RMSE; trained LSTM/TCN lead grouped MASE. |
| Ganymede | trained forecasting | `deeponet` | 0.679516 | 1.374582 | 0.030147 | 0.222079 | -0.402093 | `results/post_fix/deeponet/ganymede_h*_multi_well.json` | accepted / available-remote | Winner depends on metric: FMs lead MAE/RMSE; trained LSTM/TCN lead grouped MASE. |
| Ganymede | trained forecasting | `patchtst` | 1.077130 | 2.116416 | 0.047367 | -0.864030 | -1.197210 | `results/post_fix/patchtst/ganymede_h*_multi_well.json` | accepted / available-remote | Winner depends on metric: FMs lead MAE/RMSE; trained LSTM/TCN lead grouped MASE. |

## CDF post-fix anomaly detection

| Dataset | Family | Model | Primary metric | Value | Secondary metric | Result path | Status | Caveat |
|---|---|---|---|---:|---:|---|---|---|
| CDF | trained reconstruction | `lstm` | error_mean | 0.005878 | error_p50=0.005289 | `results/post_fix/lstm/cdf.json` | accepted / available-remote | Strict raw-row CDF gap; compare only within trained reconstruction group. |
| CDF | trained reconstruction | `patchtst` | error_mean | 0.081621 | error_p50=0.071603 | `results/post_fix/patchtst/cdf.json` | accepted / available-remote | Strict raw-row CDF gap; compare only within trained reconstruction group. |
| CDF | trained reconstruction | `deeponet` | error_mean | 0.230735 | error_p50=0.223694 | `results/post_fix/deeponet/cdf.json` | accepted / available-remote | Strict raw-row CDF gap; compare only within trained reconstruction group. |
| CDF | foundation forecast | `chronos` | forecast_error_mean | 0.243968 | forecast_error_p50=0.226866 | `results/post_fix/chronos/cdf.json` | accepted / available-remote | One-step forecast-error semantics; compare only within FM forecast group. |
| CDF | foundation forecast | `tirex` | forecast_error_mean | 0.268548 | forecast_error_p50=0.246441 | `results/post_fix/tirex/cdf.json` | accepted / available-remote | One-step forecast-error semantics; compare only within FM forecast group. |
| CDF | foundation forecast | `timesfm` | forecast_error_mean | 0.295999 | forecast_error_p50=0.283747 | `results/post_fix/timesfm/cdf.json` | accepted / available-remote | One-step forecast-error semantics; compare only within FM forecast group. |

## CDF metric-semantics guard

- Trained CDF models (`lstm`, `deeponet`, `patchtst`) report reconstruction metrics as `error_*`.
- Foundation CDF models (`chronos`, `timesfm`, `tirex`) report one-step forecast metrics as `forecast_error_*`.
- `reports/statistical_tests_nested.json` marks CDF `status=metric_semantics_separated` and `pooled_status=not_run_metric_semantics_differ`.
- A single pooled CDF ranking is blocked unless a later methodology decision establishes comparable error semantics.
