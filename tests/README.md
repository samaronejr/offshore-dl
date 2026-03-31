# tests/

Around 250 tests covering models, data pipelines, training, metrics, and end-to-end integration.

## Test Files

| File | Scope | Tests |
|------|-------|-------|
| `test_lstm.py` | LSTM forward pass, all 3 tasks | Classification, forecasting, anomaly shapes |
| `test_deeponet.py` | DeepONet branch-trunk architecture | Classification + forecasting modes |
| `test_patchtst.py` | PatchTST integration | HuggingFace wrapper, patch shapes |
| `test_mlp.py` | MLP baseline | Flatten shape, fit/predict, classification |
| `test_xgboost.py` | XGBoost pipeline | Flatten, fit/predict, metrics, JSON schema (13 tests) |
| `test_zero_shot.py` | Chronos, TimesFM, TiRex wrappers | Inference shapes, FM-specific edge cases |
| `test_training.py` | Trainer, ExperimentRunner | Training loop, early stopping, MLflow logging |
| `test_cv.py` | Cross-validation strategies | Causality, leakage guard, fold integrity |
| `test_metrics.py` | MetricRegistry | Correctness on degenerate/edge cases |
| `test_datasets.py` | Dataset loading | Shapes, windowing, feature extraction |
| `test_transforms.py` | Transform functions | z-score, EMA, frozen-value handling |
| `test_baselines.py` | Naive baselines | Seasonal naive, majority, mean reconstruction |
| `test_config.py` | YAML config loading | Merge order, overrides |
| `test_production.py` | Production script logic | Dry-run, model dispatch, output schema |
| `test_analysis.py` | LaTeX generation | Table formatting, statistical tests |
| `test_integration.py` | End-to-end | Full pipeline: load → train → evaluate |
| `test_reproducibility.py` | Seed determinism | Same seed → same output |
| `conftest.py` | Shared fixtures | Synthetic data, temp directories, device selection |

## Running

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_xgboost.py -v

# Skip FM-dependent tests (no GPU / no FM packages)
pytest tests/ -v -k "not zero_shot"
```
