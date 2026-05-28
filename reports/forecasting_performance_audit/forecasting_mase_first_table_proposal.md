# MASE-First Table Proposal

## Policy

- Primary metric: MASE.
- Primary scenario key: dataset × horizon × mode.
- Per-well rows are averaged inside each dataset × horizon × mode scenario before cross-scenario aggregation.
- Stored MASE and recomputed raw-train MASE are both retained; `effective_mase` uses recomputed raw-train denominators where available.

## Cross-scenario summary using stored MASE

| model    | scenario_count | mean_mase | median_mase | mean_rank |
| -------- | -------------- | --------- | ----------- | --------- |
| lstm     | 31             | 15.1      | 0.4187      | 2.516     |
| patchtst | 31             | 15.14     | 0.3259      | 2.839     |
| tcn      | 31             | 15.26     | 0.3916      | 3.29      |
| tirex    | 31             | 1.862     | 1.345       | 4.032     |
| deeponet | 31             | 15.39     | 0.7432      | 4.677     |
| chronos  | 31             | 1.856     | 1.359       | 4.71      |
| timesfm  | 31             | 3.176     | 1.878       | 5.935     |

## Cross-scenario summary using audited/effective MASE

| model    | scenario_count | mean_mase | median_mase | mean_rank |
| -------- | -------------- | --------- | ----------- | --------- |
| tirex    | 32             | 2.877     | 1.376       | 1.188     |
| chronos  | 32             | 2.875     | 1.394       | 2.031     |
| timesfm  | 32             | 3.92      | 1.862       | 3.188     |
| lstm     | 32             | 5.644     | 2.16        | 4.75      |
| patchtst | 32             | 5.271     | 2.083       | 5.094     |
| tcn      | 32             | 5.747     | 2.164       | 5.219     |
| deeponet | 32             | 7.954     | 3.838       | 6.531     |

## Largest audited MASE changes

    | model    | dataset  | horizon | mode     | well_or_group  | fold_or_split | stored_mase | recomputed_mase | delta  | denominator_source | classification |
| -------- | -------- | ------- | -------- | -------------- | ------------- | ----------- | --------------- | ------ | ------------------ | -------------- |
| tcn      | volve    | 14      | per_well | NO_15_9-F-5_AH | test          | 2482        | 287.2           | -2195  | raw_train          | valid_stored   |
| deeponet | volve    | 14      | per_well | NO_15_9-F-5_AH | test          | 2481        | 287             | -2194  | raw_train          | valid_stored   |
| patchtst | volve    | 14      | per_well | NO_15_9-F-5_AH | test          | 2472        | 286             | -2186  | raw_train          | valid_stored   |
| lstm     | volve    | 14      | per_well | NO_15_9-F-5_AH | test          | 2471        | 286             | -2185  | raw_train          | valid_stored   |
| tcn      | volve    | 7       | per_well | NO_15_9-F-5_AH | cv_fold_2     | 691.6       | 100.1           | -591.5 | raw_train          | valid_stored   |
| deeponet | volve    | 7       | per_well | NO_15_9-F-5_AH | cv_fold_2     | 691.5       | 100.1           | -591.4 | raw_train          | valid_stored   |
| lstm     | volve    | 7       | per_well | NO_15_9-F-5_AH | cv_fold_2     | 691.5       | 100.1           | -591.4 | raw_train          | valid_stored   |
| patchtst | volve    | 7       | per_well | NO_15_9-F-5_AH | cv_fold_2     | 691.2       | 100             | -591.2 | raw_train          | valid_stored   |
| lstm     | spe_berg | 90      | per_well | well_20        | test          | 189.9       | 15.56           | -174.4 | raw_train          | valid_stored   |
| tcn      | spe_berg | 90      | per_well | well_20        | test          | 189.4       | 15.51           | -173.9 | raw_train          | valid_stored   |
| deeponet | spe_berg | 90      | per_well | well_20        | test          | 188.5       | 15.44           | -173   | raw_train          | valid_stored   |
| patchtst | spe_berg | 30      | per_well | well_25        | test          | 159.2       | 45.74           | -113.5 | raw_train          | valid_stored   |
| patchtst | spe_berg | 30      | per_well | well_19        | cv_fold_0     | 110.7       | 9.069           | -101.6 | raw_train          | valid_stored   |
| patchtst | spe_berg | 30      | per_well | well_19        | test          | 134.6       | 43.96           | -90.64 | raw_train          | valid_stored   |
| lstm     | spe_berg | 90      | per_well | well_19        | test          | 104.8       | 15.34           | -89.47 | raw_train          | valid_stored   |
| tcn      | spe_berg | 90      | per_well | well_19        | test          | 103.9       | 15.21           | -88.72 | raw_train          | valid_stored   |
| deeponet | spe_berg | 90      | per_well | well_19        | test          | 102.3       | 14.98           | -87.35 | raw_train          | valid_stored   |
| patchtst | spe_berg | 14      | per_well | well_25        | test          | 134         | 48.37           | -85.62 | raw_train          | valid_stored   |
| patchtst | spe_berg | 90      | per_well | well_19        | test          | 97.47       | 14.27           | -83.2  | raw_train          | valid_stored   |
| timesfm  | volve    | 7       | per_well | NO_15_9-F-5_AH | test          | 118.9       | 46.95           | -71.91 | raw_train          | valid_stored   |
| patchtst | spe_berg | 30      | per_well | well_19        | cv_fold_2     | 87.15       | 19.86           | -67.29 | raw_train          | valid_stored   |
| patchtst | spe_berg | 90      | per_well | well_26        | test          | 84.29       | 19.47           | -64.82 | raw_train          | valid_stored   |
| patchtst | spe_berg | 30      | per_well | well_19        | cv_fold_1     | 59.45       | 8.83            | -50.62 | raw_train          | valid_stored   |
| patchtst | spe_berg | 7       | per_well | well_25        | test          | 153.9       | 104.3           | -49.57 | raw_train          | valid_stored   |
| deeponet | volve    | 7       | per_well | NO_15_9-F-5_AH | cv_fold_1     | 55.46       | 8.163           | -47.3  | raw_train          | valid_stored   |

## Draft table outputs

- `forecasting_mase_first_raw_wide_original.csv`
- `forecasting_mase_first_raw_wide_effective.csv`
- `forecasting_mase_weighted_summary_original.csv`
- `forecasting_mase_weighted_summary_effective.csv`
