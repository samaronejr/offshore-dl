# results/

Experiment outputs are organized by validity epoch after the May 2026 benchmark-validity repair.

## Structure

```text
results/
├── pre_fix/      # Historical results produced before the MASE/CV/PatchTST/MLflow repair
├── post_fix/     # New reruns produced after the repair
├── .omx/         # Local OMX/runtime logs and state, not benchmark outputs
└── README.md
```

## Validity guidance

- `pre_fix/` results are preserved for audit/history.
- Treat old forecasting `mase` / grouped MASE values as non-authoritative unless rerun with repaired chronological MASE plumbing.
- Treat old CDF production CV results as non-authoritative because those runs used zero CV gap before the strict raw-row gap repair.
- Classification metrics are not directly invalidated by the MASE/CDF fixes, but remain under `pre_fix/` until rerun.
- Write all new post-repair experiment outputs to `results/post_fix/`.
- Production CLIs now default writers to `results/post_fix/`; override with `--results-dir` or `OFFSHORE_DL_RESULTS_DIR` only for deliberate runs. HPO campaign artifacts remain under `results/hpo/` unless `--output-dir` is provided, and are not final benchmark outputs unless final evaluation is present.

## Previous layout

The historical layout was one subdirectory per model plus summary files. That full tree now lives under `results/pre_fix/` unchanged.
