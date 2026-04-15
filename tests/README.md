# tests/

534 tests covering models, data pipelines, training, metrics, cross-validation, and end-to-end integration.

## Test Files

| File | Scope | Tests |
|------|-------|-------|
| `test_lstm.py` | LSTM forward pass, all 3 tasks | Classification, forecasting, anomaly shapes |
| `test_deeponet.py` | DeepONet branch-trunk architecture | Classification + forecasting modes |
| `test_patchtst.py` | PatchTST integration | HuggingFace wrapper, patch shapes |
| `test_tcn.py` | TCN dilated convolutions | Receptive field, causal masking, output shapes |
| `test_convtimenet.py` | ConvTimeNet multi-scale conv | Depthwise separable blocks, forward pass |
| `test_convtran.py` | ConvTran conv + attention | Local attention, positional encoding shapes |
| `test_inception_time.py` | InceptionTime inception blocks | Multi-scale kernel concatenation, shapes |
| `test_mambasl.py` | MambaSL state-space model | Selective scan, sequence modeling |
| `test_fkmad.py` | FKMAD frequency-kernel detector | Anomaly scoring, frequency decomposition |
| `test_hydra_rocket.py` | Hydra + ROCKET kernels | Kernel generation, fit/predict, no training |
| `test_random_forest.py` | Random Forest pipeline | Feature extraction, fit/predict, metrics |
| `test_zero_shot.py` | Chronos, TimesFM, TiRex, MOMENT, MANTIS | Inference shapes, FM-specific edge cases |
| `test_training.py` | Trainer, ExperimentRunner | Training loop, early stopping, LR schedulers, NaN handling, MLflow logging |
| `test_cv.py` | Cross-validation strategies | Causality, leakage guard, fold integrity, grouped strategies |
| `test_metrics.py` | MetricRegistry | Correctness on degenerate/edge cases |
| `test_datasets.py` | Dataset loading | Shapes, windowing, feature extraction |
| `test_spe_berg_dataset.py` | SPEBergDataset | Loading, splits, class distribution |
| `test_volve_dataset.py` | VolveDataset | Loading, well splits, time alignment |
| `test_inner_mongolia_dataset.py` | InnerMongoliaDataset | Loading, fault label integrity |
| `test_transforms.py` | Transform functions | z-score, EMA, frozen-value handling |
| `test_augmentations.py` | Data augmentation | Jitter, scaling, time-warp transforms |
| `test_physics.py` | Physics-informed constraints | Constraint satisfaction, penalty terms |
| `test_wavelet.py` | Wavelet feature extraction | Decomposition levels, coefficient shapes |
| `test_baselines.py` | Naive baselines | Seasonal naive, majority, mean reconstruction |
| `test_config.py` | YAML config loading | Merge order, overrides, `load_merged_config` |
| `test_production.py` | Production script logic | Dry-run, model dispatch, output schema |
| `test_analysis.py` | LaTeX generation | Table formatting, statistical tests |
| `test_integration.py` | End-to-end | Full pipeline: load → train → evaluate |
| `test_reproducibility.py` | Seed determinism | Same seed → same output across full e2e training runs |
| `conftest.py` | Shared fixtures | Synthetic data, temp directories, device selection |

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
```
