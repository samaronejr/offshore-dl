# Benchmark Validity Fix Summary

## Changed files
- `README.md`
- `scripts/run_optuna_hpo.py`
- `scripts/run_production_ganymede.py`
- `scripts/run_production_spe_berg.py`
- `scripts/run_production_volve.py`
- `scripts/run_production_inner_mongolia.py`
- `scripts/validate_spe_berg_results.py`
- `src/offshore_dl/evaluation/__init__.py`
- `src/offshore_dl/evaluation/cv.py`
- `src/offshore_dl/models/base.py`
- `src/offshore_dl/run_experiment.py`
- `src/offshore_dl/training/experiment.py`
- `tests/test_cv.py`
- `tests/test_metrics.py`
- `tests/test_training.py`

## Fixes implemented
1. Added grouped forecasting split utilities:
   - `GroupedTemporalHoldoutSplitter`
   - `GroupedExpandingWindowCV`
2. Switched forecasting experiment builders, production scripts, and HPO entrypoints to use grouped per-well temporal splitting for multi-well datasets.
3. Removed shutdown-window prefiltering from the primary forecasting benchmark paths.
4. Fixed classification metric plumbing so evaluation now passes:
   - probability scores for AUC-PR
   - `instance_id` metadata for true EDR
5. Fixed nested retraining so retrain-train and retrain-validation subsets are disjoint.
6. Updated README and validation script text to describe the corrected benchmark protocol.
7. Added regression tests for grouped splits, probability-aware metrics, instance-aware EDR, and retrain train/val separation.

## Simplifications made
- Centralized forecasting split behavior into reusable grouped CV utilities instead of repeating ad hoc flat-index temporal logic.
- Reused a single forward pass during classification evaluation to derive both hard predictions and probability scores.
- Reused the existing nested evaluation path instead of introducing a second benchmark pipeline.

## Verification
### Passed
- `pytest -q tests/test_cv.py tests/test_metrics.py tests/test_training.py tests/test_production.py`
- `python -m compileall src tests scripts`
- workspace diagnostics check: no project diagnostics reported

### Notes
- `ruff check` still reports many pre-existing unrelated lint issues across the repository; those were not addressed in this change.

## Remaining risks
- Benchmark result JSONs, reports, and manuscript tables have not been regenerated after the methodology fix, so existing published numbers may still reflect the old protocol.
- The forecasting fix assumes well-local temporal ordering inside each dataset's `_samples` index; if any dataset later changes sample ordering, grouped split utilities should be revalidated.
- The current retrain policy now uses a disjoint validation subset for checkpoint selection; if a strict final refit on 100% of the training pool is desired, an extra post-selection refit step is still TODO.

## TODO
1. Rerun affected production forecasting pipelines (`ganymede`, `spe_berg`, `volve`, `inner_mongolia`).
2. Rerun affected HPO jobs where historical best params/results depended on the old split logic.
3. Regenerate result JSON summaries, statistical tests, and manuscript/report tables.
4. Review any downstream narrative claims that cite old AUC-PR, EDR, or forecasting benchmark numbers.
