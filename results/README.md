# results/

Generated experiment outputs are organized by validity epoch and campaign. The repository may not contain every large result artifact after a fresh clone; production and HPO commands regenerate them.

## Structure

```text
results/
├── pre_fix/        # Historical outputs produced before benchmark-validity repairs
├── post_fix/       # Fixed-code production reruns, including current Ganymede outputs
├── hpo/            # Optuna campaign outputs, e.g. 3W Stage 1 HPO summaries
├── stage2_3w/      # 3W follow-up variants and window-length experiments
├── .omx/           # Local OMX/runtime logs and state, not benchmark outputs
└── README.md
```

## Current validity guidance

| Location | Status | Use |
|---|---|---|
| `results/hpo/3w/3w-hpo-latest-20260510T180941Z/summary.json` | Current validated 3W Stage 1 HPO | Headline apples-to-apples 720-window classification leaderboard. |
| `results/stage2_3w/3w-stage2-20260513T192623Z/` | Current 3W follow-up campaign | Report separately as feature/window variants; do not pool with Stage 1. |
| `results/post_fix/<model>/ganymede*.json` | Current post-fix Ganymede reruns | Current forecasting evidence for Ganymede. |
| `results/pre_fix/` | Historical | Audit/history only unless explicitly revalidated. |

## 3W Stage 1 HPO summary

Primary ranking metric: macro-F1 on the held-out test set. All models below completed 30 Optuna trials and final evaluation.

| Rank | Model | Macro-F1 | Accuracy | Trials | Best CV objective |
|---:|---|---:|---:|---:|---:|
| 1 | Random Forest | 0.968972 | 0.970228 | 30 | 0.966316 |
| 2 | DeepONet | 0.962579 | 0.968734 | 30 | 0.960540 |
| 3 | MambaSL | 0.962185 | 0.967313 | 30 | 0.958081 |
| 4 | LSTM | 0.960046 | 0.965314 | 30 | 0.953991 |
| 5 | FKMAD | 0.956388 | 0.964061 | 30 | 0.951980 |
| 6 | ConvTimeNet | 0.954953 | 0.959894 | 30 | 0.953499 |
| 7 | PatchTST | 0.953556 | 0.958738 | 30 | 0.957745 |

## 3W Stage 2 summary

Stage 2 outputs are valid follow-up experiments only when their preprocessing/window assumptions match the claim being made.

| Variant | Macro-F1 | Accuracy | Validity note |
|---|---:|---:|---|
| `window360_rf` | 0.987797 | 0.991537 | Valid, but different window length from Stage 1. |
| `window1440_rf` | 0.977011 | 0.989084 | Valid, but different window length from Stage 1. |
| `wavelet_rf` | 0.964309 | 0.966301 | Valid feature variant. |
| `multiscale_rf` | 0.964184 | 0.966205 | Valid feature variant. |
| `physics_rf` | 0.964070 | 0.966060 | Valid feature variant. |
| `wavelet_deeponet` | 0.961252 | 0.967867 | Valid DeepONet variant. |
| `convtran` | 0.955762 | 0.964398 | Valid baseline. |
| `multiscale_deeponet` | 0.954235 | 0.964880 | Valid DeepONet variant. |
| `physics_deeponet` | 0.952286 | 0.961652 | Valid DeepONet variant. |

Failed or invalid Stage 2 outputs:

- `hydra_rocket`: failed from an impractical multi-terabyte RAM allocation during the retry campaign.
- `convtimenet_raw`, `convtran_raw`, `fkmad_raw`, `mambasl_raw`: collapsed to macro-F1 0.027287 / accuracy 0.157991 and should be treated as failed raw-window baselines.

## Ganymede post-fix summary

Current Ganymede values are fixed-code multi-well aggregates across horizons from `results/post_fix/<model>/ganymede_h*_multi_well.json`.

| Model | MAE | RMSE | MASE | R² | R²_prod |
|---|---:|---:|---:|---:|---:|
| TiRex | 0.3617 | 1.2476 | 0.2071 | 0.3541 | -0.1490 |
| TimesFM | 0.3965 | 1.2634 | 0.2295 | 0.3362 | -0.1283 |
| Chronos-2 | 0.5357 | 1.4412 | 0.3205 | 0.1111 | -0.3431 |
| LSTM | 0.5457 | 1.3517 | 0.0228 | 0.2455 | -0.6481 |
| TCN | 0.5677 | 1.3136 | 0.0234 | 0.2864 | -0.4099 |
| DeepONet | 0.6795 | 1.3746 | 0.0301 | 0.2221 | -0.4021 |
| PatchTST | 1.0771 | 2.1164 | 0.0474 | -0.8640 | -1.1972 |

Use MAE/RMSE for absolute production-scale error, MASE for scaled within-well error, and R²/R²_prod only as diagnostics.

## Historical/pre-fix caveats

- `pre_fix/` results are preserved for audit/history.
- Old forecasting `mase` / grouped MASE values are non-authoritative unless rerun with repaired chronological MASE plumbing.
- Old CDF production CV results are non-authoritative because those runs used zero CV gap before the strict raw-row gap repair.
- Classification metrics are not directly invalidated by MASE/CDF fixes, but historical classification outputs remain under `pre_fix/` until rerun.
- New production CLIs default writers to `results/post_fix/`; override with `--results-dir` or `OFFSHORE_DL_RESULTS_DIR` only for deliberate runs.
- HPO artifacts remain under `results/hpo/` unless `--output-dir` is provided, and are not final benchmark outputs unless final evaluation is present.
