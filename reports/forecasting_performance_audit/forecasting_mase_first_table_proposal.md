# MASE-First Table Proposal

## Policy

- Primary metric: MASE.
- Primary scenario key: dataset × horizon × mode.
- Per-well rows are averaged inside each dataset × horizon × mode scenario before cross-scenario aggregation.
- Stored MASE and recomputed raw-train MASE are both retained; `effective_mase` uses recomputed raw-train denominators where available.

## Cross-scenario summary using stored MASE

| model    | scenario_count | mean_mase | median_mase | mean_rank |
| -------- | -------------- | --------- | ----------- | --------- |
| tirex    | 8              | 1.08      | 1.009       | 1         |
| chronos  | 16             | 1.498     | 1.234       | 1.688     |
| timesfm  | 8              | 1.232     | 1.177       | 2.875     |
| lstm     | 16             | 1557      | 2.08        | 3.438     |
| tcn      | 9              | 3.603     | 2.761       | 4.444     |
| deeponet | 16             | 1078      | 2.899       | 4.688     |
| patchtst | 12             | 6.902e+04 | 3.259       | 4.917     |

## Cross-scenario summary using audited/effective MASE

| model    | scenario_count | mean_mase | median_mase | mean_rank |
| -------- | -------------- | --------- | ----------- | --------- |
| tirex    | 8              | 0.5315    | 0.4418      | 1.75      |
| lstm     | 16             | 1.487     | 1.201       | 1.875     |
| chronos  | 16             | 1.347     | 0.9511      | 2.188     |
| timesfm  | 8              | 0.5829    | 0.486       | 3.625     |
| deeponet | 16             | 2.388     | 1.651       | 4.5       |
| tcn      | 9              | 2.555     | 2.238       | 5         |
| patchtst | 12             | 2.478     | 2.079       | 5.167     |

## Largest audited MASE changes

    | model    | dataset  | horizon | mode     | well_or_group | fold_or_split | stored_mase | recomputed_mase | delta      | denominator_source    | classification              |
| -------- | -------- | ------- | -------- | ------------- | ------------- | ----------- | --------------- | ---------- | --------------------- | --------------------------- |
| patchtst | ganymede | 7       | per_well | 49_22-Z06     | test          | 2.906e+06   | 7.333           | -2.906e+06 | double_denorm_suspect | suspect_stored_recomputable |
| patchtst | ganymede | 14      | per_well | 49_22-Z06     | test          | 2.891e+06   | 4.437           | -2.891e+06 | double_denorm_suspect | suspect_stored_recomputable |
| patchtst | ganymede | 7       | per_well | 49_22-Z08     | cv_fold_1     | 8.781e+04   | 13.36           | -8.78e+04  | double_denorm_suspect | suspect_stored_recomputable |
| patchtst | ganymede | 14      | per_well | 49_22-Z08     | cv_fold_1     | 8.737e+04   | 5.599           | -8.736e+04 | double_denorm_suspect | suspect_stored_recomputable |
| lstm     | ganymede | 90      | per_well | 49_22-Z06     | test          | 5.806e+04   | 0.08889         | -5.806e+04 | double_denorm_suspect | suspect_stored_recomputable |
| lstm     | ganymede | 30      | per_well | 49_22-Z06     | test          | 4.263e+04   | 0.06296         | -4.263e+04 | double_denorm_suspect | suspect_stored_recomputable |
| lstm     | ganymede | 14      | per_well | 49_22-Z06     | test          | 3.963e+04   | 0.06083         | -3.963e+04 | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 14      | per_well | 49_22-Z06     | test          | 3.816e+04   | 0.05856         | -3.816e+04 | double_denorm_suspect | suspect_stored_recomputable |
| lstm     | ganymede | 7       | per_well | 49_22-Z06     | test          | 3.383e+04   | 0.08537         | -3.383e+04 | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 90      | per_well | 49_22-Z06     | test          | 3.297e+04   | 0.05047         | -3.297e+04 | double_denorm_suspect | suspect_stored_recomputable |
| patchtst | ganymede | 7       | per_well | 49_22-Z05Z    | cv_fold_0     | 2.821e+04   | 1.799           | -2.821e+04 | double_denorm_suspect | suspect_stored_recomputable |
| patchtst | ganymede | 14      | per_well | 49_22-Z05Z    | cv_fold_0     | 2.807e+04   | 1.122           | -2.807e+04 | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 30      | per_well | 49_22-Z06     | test          | 2.58e+04    | 0.03811         | -2.58e+04  | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 7       | per_well | 49_22-Z06     | test          | 2.342e+04   | 0.0591          | -2.342e+04 | double_denorm_suspect | suspect_stored_recomputable |
| patchtst | ganymede | 7       | per_well | 49_22-Z06     | cv_fold_0     | 9080        | 2.288           | -9078      | double_denorm_suspect | suspect_stored_recomputable |
| patchtst | ganymede | 14      | per_well | 49_22-Z06     | cv_fold_0     | 8497        | 1.383           | -8495      | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 90      | per_well | 49_22-Z05Z    | cv_fold_0     | 6282        | 0.2492          | -6281      | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 14      | per_well | 49_22-Z06     | cv_fold_0     | 5308        | 0.8639          | -5307      | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 14      | per_well | 49_22-Z05Z    | cv_fold_0     | 5070        | 0.2027          | -5070      | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 30      | per_well | 49_22-Z05Z    | cv_fold_0     | 4556        | 0.1757          | -4556      | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 7       | per_well | 49_22-Z06     | cv_fold_0     | 3592        | 0.905           | -3591      | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 7       | per_well | 49_22-Z05Z    | cv_fold_0     | 3469        | 0.2212          | -3469      | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 90      | per_well | 49_22-Z06     | cv_fold_0     | 3144        | 0.51            | -3143      | double_denorm_suspect | suspect_stored_recomputable |
| deeponet | ganymede | 30      | per_well | 49_22-Z06     | cv_fold_0     | 2394        | 0.3713          | -2393      | double_denorm_suspect | suspect_stored_recomputable |
| lstm     | ganymede | 30      | per_well | 49_22-Z05Z    | cv_fold_0     | 2090        | 0.08061         | -2090      | double_denorm_suspect | suspect_stored_recomputable |

## Draft table outputs

- `forecasting_mase_first_raw_wide_original.csv`
- `forecasting_mase_first_raw_wide_effective.csv`
- `forecasting_mase_weighted_summary_original.csv`
- `forecasting_mase_weighted_summary_effective.csv`
