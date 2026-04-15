# results/

All experiment outputs are stored as JSON files, one subdirectory per model.

## Structure

```
results/
├── baselines/              # Naive baselines (majority, seasonal naive, mean reconstruction)
├── lstm/                   # LSTM results
├── deeponet/               # DeepONet results
├── patchtst/               # PatchTST results
├── tcn/                    # TCN results
├── convtimenet/            # ConvTimeNet results
├── convtimenet_raw/        # ConvTimeNet raw window results
├── convtran/               # ConvTran results
├── convtran_raw/           # ConvTran raw window results
├── inception_time/         # InceptionTime results
├── inception_time_raw/     # InceptionTime raw window results
├── mambasl/                # MambaSL results
├── mambasl_raw/            # MambaSL raw window results
├── fkmad/                  # FKMAD results
├── fkmad_raw/              # FKMAD raw window results
├── random_forest/          # Random Forest results
├── multiscale_deeponet/    # DeepONet with multi-scale 3W features
├── multiscale_rf/          # Random Forest with multi-scale 3W features
├── chronos/                # Chronos-2 zero-shot results
├── timesfm/                # TimesFM 2.5 zero-shot results
├── tirex/                  # TiRex zero-shot results
├── hpo/                    # Optuna HPO outputs
├── tirex_3w_nested.json                 # TiRex 3W classification (standalone)
├── summary_production_3w_features.json  # 3W feature sweep summary
└── summary_production_cdf.json          # CDF sweep summary
```

## JSON Schema

Each result file contains:

```json
{
  "test_metrics": { "mae": ..., "rmse": ..., "r2_prod": ... },
  "test_predictions": [ ... ],
  "test_probabilities": [ ... ],
  "cv_aggregate": { "mae_mean": ..., "mae_std": ... },
  "cv_fold_results": [ { "fold_idx": 0, "metrics": { ... } }, ... ],
  "n_train": ...,
  "n_test": ...,
  "n_cv_folds": ...
}
```

Classification files use `accuracy`, `f1_macro`, `f1_weighted`, `auc_pr` instead of forecasting metrics. CDF files use `error_mean`, `error_std`, `error_p50`, `error_p95`, `error_p99`. `test_predictions` and `test_probabilities` store model outputs for post-hoc analysis.

## Naming Conventions

**Ganymede:**
- `ganymede_h{7,14,30,90}_multi_well.json` — all 7 wells combined
- `ganymede_h{7,14,30,90}_per_well_{well_id}.json` — single well
- `ganymede.json` — legacy copy of h7 multi_well (backward compat)
- `summary_production_ganymede.json` — full sweep status + timings

**SPE Berg:**
- `spe_berg.json` — full dataset classification results
- `spe_berg_fold_{n}.json` — per-fold results

**Volve:**
- `volve_h{7,14,30,90}.json` — forecasting results per horizon
- `volve_per_well_{well_id}.json` — single well results

**Inner Mongolia:**
- `inner_mongolia.json` — fault detection results
- `inner_mongolia_fold_{n}.json` — per-fold results
