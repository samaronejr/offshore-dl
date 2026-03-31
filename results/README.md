# results/

All experiment outputs are stored as JSON files, one subdirectory per model.

## Structure

```
results/
├── baselines/          # Naive baselines (majority, seasonal naive, mean reconstruction)
├── lstm/               # LSTM results: 3w.json, cdf.json, ganymede_h*_multi_well.json, ganymede_h*_per_well_*.json
├── deeponet/           # DeepONet results (same structure as lstm/)
├── patchtst/           # PatchTST results
├── mlp/                # MLP results (3W classification only)
├── xgboost/            # XGBoost results (Ganymede forecasting only)
├── chronos/            # Chronos-2 zero-shot results (Ganymede + CDF)
├── timesfm/            # TimesFM 2.5 zero-shot results (Ganymede + CDF)
├── tirex/              # TiRex zero-shot results (Ganymede + CDF)
├── hpo/                # Optuna HPO outputs (3W: lstm, deeponet, patchtst)
├── tirex_3w_nested.json           # TiRex 3W classification (standalone file)
├── summary_production_3w_features.json  # 3W sweep summary
└── summary_production_cdf.json          # CDF sweep summary
```

## JSON Schema

Each result file contains:

```json
{
  "test_metrics": { "mae": ..., "rmse": ..., "r2_prod": ... },
  "cv_aggregate": { "mae_mean": ..., "mae_std": ... },
  "cv_fold_results": [ { "fold_idx": 0, "metrics": { ... } }, ... ],
  "n_train": ...,
  "n_test": ...,
  "n_cv_folds": ...
}
```

Classification files use `accuracy`, `f1_macro`, `f1_weighted`, `auc_pr` instead of forecasting metrics. CDF files use `error_mean`, `error_std`, `error_p50`, `error_p95`, `error_p99`.

## Ganymede Naming Convention

- `ganymede_h{7,14,30,90}_multi_well.json` — all 7 wells combined
- `ganymede_h{7,14,30,90}_per_well_{well_id}.json` — single well
- `ganymede.json` — legacy copy of h7 multi_well (backward compat)
- `summary_production_ganymede.json` — full sweep status + timings
