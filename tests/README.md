# tests/

673 collected pytest tests cover models, data pipelines, training, metrics, cross-validation, HPO plumbing, production-script logic, and end-to-end integration.

Collection evidence for this README update:

```text
pytest --collect-only -q tests
# 673 tests collected
```

## Test files

| File | Scope |
|------|-------|
| `test_lstm.py` | LSTM forward pass and task modes |
| `test_deeponet.py` | DeepONet branch-trunk architecture |
| `test_deeponet_recon_clf.py` | DeepONet reconstruction/classification variant |
| `test_deeponet_trunk_clf.py` | DeepONet trunk-classifier variant |
| `test_patchtst.py` | PatchTST integration and shape behavior |
| `test_tcn.py` | TCN dilated convolutions and output shapes |
| `test_convtimenet.py` | ConvTimeNet multi-scale convolution behavior |
| `test_convtran.py` | ConvTran convolution/attention classifier |
| `test_inception_time.py` | InceptionTime inception blocks |
| `test_mambasl.py` | MambaSL state-space model behavior |
| `test_fkmad.py` | FKMAD model and anomaly/classification behavior |
| `test_hydra_rocket.py` | Hydra + ROCKET kernels and sklearn-style behavior |
| `test_random_forest.py` | Random Forest feature pipeline and metrics |
| `test_zero_shot.py` | Chronos, TimesFM, TiRex, MOMENT, MANTIS wrapper paths with conditional skips |
| `test_training.py` | Trainer, ExperimentRunner, early stopping, schedulers, MLflow behavior |
| `test_cv.py` | Temporal/grouped/stratified CV strategies and leakage guards |
| `test_metrics.py` | MetricRegistry correctness |
| `test_loss_and_metric_edge_cases.py` | Edge-case losses and metric handling |
| `test_datasets.py` | Dataset loading and shape contracts |
| `test_spe_berg_dataset.py` | SPE Berg dataset behavior |
| `test_volve_dataset.py` | Volve dataset behavior |
| `test_inner_mongolia_dataset.py` | Inner Mongolia dataset behavior |
| `test_transforms.py` | z-score, EMA, frozen-value transforms |
| `test_augmentations.py` | Jitter, scaling, time-warp, and related augmentations |
| `test_physics.py` | Physics-informed feature extraction and constraints |
| `test_wavelet.py` | Wavelet feature extraction |
| `test_baselines.py` | Naive forecasting/classification/anomaly baselines |
| `test_config.py` | YAML config loading, merge order, overrides |
| `test_production.py` | Production script dispatch, output naming, dry-run/CLI behavior |
| `test_hpo_forecasting.py` | Forecasting HPO behavior |
| `test_optuna_lr_routing.py` | Optuna learning-rate/search-space routing regressions |
| `test_forecasting_audit.py` | Forecasting validity/audit regressions |
| `test_sweep_utils.py` | Shared sweep helper behavior |
| `test_results_utils.py` | Result utility behavior |
| `test_analysis.py` | LaTeX/table/statistical comparison helpers |
| `test_packaging_imports.py` | Package import/export behavior |
| `test_integration.py` | Tiny end-to-end pipeline smoke tests |
| `test_reproducibility.py` | Determinism and seed behavior |
| `conftest.py` | Shared fixtures for synthetic data, temp dirs, configs, and device selection |

## Running

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_lstm.py -v

# Skip FM-dependent tests (no GPU / no FM packages)
pytest tests/ -v -k "not zero_shot"

# Skip slow integration tests
pytest tests/ -v -k "not integration"

# Collect without running tests; useful for docs-only checks
pytest --collect-only -q tests
```

## Documentation-only verification

For README-only updates, a full test run is not required unless commands or code paths changed. Use:

```bash
git diff --check
pytest --collect-only -q tests
python -m offshore_dl.run_experiment --help >/tmp/offshore_dl_help.txt
python scripts/run_optuna_hpo.py --help >/tmp/hpo_help.txt
python scripts/validate_hpo_3w_results.py --help >/tmp/hpo_validator_help.txt
```

## Benchmark-regression focus

The tests most relevant to the May 2026 benchmark fixes are:

- `test_optuna_lr_routing.py` — protects HPO learning-rate/search-space routing.
- `test_hpo_forecasting.py` — protects forecasting HPO behavior.
- `test_forecasting_audit.py` — protects forecasting/MASE validity assumptions.
- `test_production.py` — protects production script dispatch and output schema.
- `test_cv.py` — protects splitters and leakage guards.
- `test_metrics.py` and `test_loss_and_metric_edge_cases.py` — protect metrics used in README tables.
