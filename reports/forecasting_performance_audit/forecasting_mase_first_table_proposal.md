# MASE-First Table Proposal

## Policy

- Primary metric: MASE.
- Primary scenario key: dataset × horizon × mode.
- Per-well rows are averaged inside each dataset × horizon × mode scenario before cross-scenario aggregation.
- Stored MASE and recomputed raw-train MASE are both retained; `effective_mase` uses recomputed raw-train denominators where available.

## Cross-scenario summary using stored MASE

| model    | scenario_count | mean_mase | median_mase | mean_rank |
| -------- | -------------- | --------- | ----------- | --------- |
| lstm     | 8              | 0.108     | 0.1041      | 1.875     |
| patchtst | 8              | 0.1198    | 0.1071      | 3         |
| tcn      | 8              | 0.1498    | 0.1357      | 3.75      |
| tirex    | 8              | 0.2075    | 0.2104      | 3.75      |
| deeponet | 8              | 0.1723    | 0.1622      | 4.75      |
| chronos  | 8              | 0.3369    | 0.2191      | 5.25      |
| timesfm  | 8              | 0.2302    | 0.228       | 5.625     |

## Cross-scenario summary using audited/effective MASE

| model    | scenario_count | mean_mase | median_mase | mean_rank |
| -------- | -------------- | --------- | ----------- | --------- |
| tirex    | 8              | 0.1996    | 0.1936      | 1         |
| timesfm  | 8              | 0.2216    | 0.2167      | 2.75      |
| chronos  | 8              | 0.3052    | 0.2117      | 2.75      |
| lstm     | 8              | 0.5221    | 0.4601      | 3.875     |
| tcn      | 8              | 0.7363    | 0.6144      | 4.875     |
| deeponet | 8              | 0.8783    | 0.7489      | 5.75      |
| patchtst | 8              | 1.075     | 1.033       | 7         |

## Largest audited MASE changes

    | model    | dataset  | horizon | mode     | well_or_group | fold_or_split | stored_mase | recomputed_mase | delta | denominator_source | classification |
| -------- | -------- | ------- | -------- | ------------- | ------------- | ----------- | --------------- | ----- | ------------------ | -------------- |
| patchtst | ganymede | 7       | per_well | 49_22-Z06     | cv_fold_2     | 0.1487      | 5.86            | 5.711 | raw_train          | valid_stored   |
| tcn      | ganymede | 7       | per_well | 49_22-Z04     | test          | 1.036       | 6.569           | 5.533 | raw_train          | valid_stored   |
| deeponet | ganymede | 7       | per_well | 49_22-Z07     | test          | 0.9953      | 5.861           | 4.865 | raw_train          | valid_stored   |
| deeponet | ganymede | 7       | per_well | 49_22-Z04     | test          | 0.8654      | 5.486           | 4.621 | raw_train          | valid_stored   |
| tcn      | ganymede | 7       | per_well | 49_22-Z04     | cv_fold_0     | 0.8632      | 5.386           | 4.523 | raw_train          | valid_stored   |
| deeponet | ganymede | 7       | per_well | 49_22-Z07     | cv_fold_2     | 0.8516      | 5.099           | 4.247 | raw_train          | valid_stored   |
| patchtst | ganymede | 7       | per_well | 49_22-Z06     | cv_fold_1     | 0.1029      | 4.056           | 3.953 | raw_train          | valid_stored   |
| patchtst | ganymede | 14      | per_well | 49_22-Z06     | cv_fold_2     | 0.1482      | 3.538           | 3.39  | raw_train          | valid_stored   |
| patchtst | ganymede | 30      | per_well | 49_22-Z06     | cv_fold_2     | 0.1469      | 3.348           | 3.202 | raw_train          | valid_stored   |
| patchtst | ganymede | 90      | per_well | 49_22-Z06     | cv_fold_2     | 0.1446      | 3.258           | 3.114 | raw_train          | valid_stored   |
| deeponet | ganymede | 14      | per_well | 49_22-Z07     | test          | 1.229       | 3.732           | 2.503 | raw_train          | valid_stored   |
| deeponet | ganymede | 90      | per_well | 49_22-Z04     | cv_fold_0     | 1.158       | 3.641           | 2.483 | raw_train          | valid_stored   |
| deeponet | ganymede | 90      | per_well | 49_22-Z04     | cv_fold_2     | 1.151       | 3.552           | 2.401 | raw_train          | valid_stored   |
| patchtst | ganymede | 14      | per_well | 49_22-Z06     | cv_fold_1     | 0.1025      | 2.448           | 2.346 | raw_train          | valid_stored   |
| patchtst | ganymede | 7       | per_well | 49_22-Z06     | cv_fold_0     | 0.05783     | 2.281           | 2.223 | raw_train          | valid_stored   |
| patchtst | ganymede | 30      | per_well | 49_22-Z06     | cv_fold_1     | 0.1014      | 2.313           | 2.212 | raw_train          | valid_stored   |
| tcn      | ganymede | 90      | per_well | 49_22-Z04     | test          | 1.081       | 3.292           | 2.211 | raw_train          | valid_stored   |
| tcn      | ganymede | 14      | per_well | 49_22-Z04     | test          | 0.935       | 3.12            | 2.185 | raw_train          | valid_stored   |
| tcn      | ganymede | 30      | per_well | 49_22-Z04     | test          | 1.102       | 3.267           | 2.165 | raw_train          | valid_stored   |
| patchtst | ganymede | 90      | per_well | 49_22-Z06     | cv_fold_1     | 0.09963     | 2.246           | 2.147 | raw_train          | valid_stored   |
| deeponet | ganymede | 7       | per_well | 49_22-Z04     | cv_fold_0     | 0.402       | 2.508           | 2.106 | raw_train          | valid_stored   |
| deeponet | ganymede | 90      | per_well | 49_22-Z04     | test          | 0.987       | 3.007           | 2.02  | raw_train          | valid_stored   |
| tcn      | ganymede | 14      | per_well | 49_22-Z04     | cv_fold_0     | 0.8076      | 2.74            | 1.932 | raw_train          | valid_stored   |
| deeponet | ganymede | 30      | per_well | 49_22-Z07     | test          | 1.138       | 2.977           | 1.839 | raw_train          | valid_stored   |
| deeponet | ganymede | 14      | per_well | 49_22-Z07     | cv_fold_2     | 0.8472      | 2.627           | 1.78  | raw_train          | valid_stored   |

## Draft table outputs

- `forecasting_mase_first_raw_wide_original.csv`
- `forecasting_mase_first_raw_wide_effective.csv`
- `forecasting_mase_weighted_summary_original.csv`
- `forecasting_mase_weighted_summary_effective.csv`
