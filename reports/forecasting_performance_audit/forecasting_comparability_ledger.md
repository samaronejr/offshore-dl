# Forecasting Comparability Ledger

| proposal                                                            | tier                                      | rationale                                                                                                          |
| ------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Fix MASE denominator metadata and zero-denominator handling         | table-preserving                          | Changes metric validity without changing data splits, windows, horizons, or training protocol.                     |
| Recompute MASE from stored MAE plus raw-train denominator           | recompute-only comparable                 | Comparable if predictions/targets/MAE remain from the same original protocol and only the denominator is repaired. |
| Regenerate MASE-first weighted tables                               | table-preserving                          | Changes reporting aggregation, not experiment protocol; original row-level metrics remain visible.                 |
| Rerun trained models after MASE fix only                            | recompute-only comparable                 | Comparable if seeds, splits, windows, data filters, and model configs are unchanged.                               |
| Enable shutdown filtering                                           | protocol-changing/not directly comparable | Changes sample population; must be dual-reported beside unfiltered benchmark.                                      |
| Change target transforms, input windows, horizons, modes, or splits | protocol-changing/not directly comparable | Changes task definition or evaluation protocol; cannot be mixed into original ranking.                             |
| Add ensembles or new baselines                                      | protocol-changing/not directly comparable | Can be reported as an additional model family only after denominator provenance is fixed and labeled.              |

## Rule

Any shutdown-filtered, target-protocol-altered, split-altered, mode-altered, feature-population-altered, or horizon-altered result is dual-reported and never mixed into the original-protocol ranking.
