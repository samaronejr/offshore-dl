# Forecasting Coverage Audit

## Global status distribution

| status  | count |
| ------- | ----- |
| ok      | 2737  |
| missing | 35    |

## Status by dataset

| dataset        | status  | count |
| -------------- | ------- | ----- |
| ganymede       | ok      | 224   |
| inner_mongolia | missing | 14    |
| inner_mongolia | ok      | 826   |
| spe_berg       | missing | 14    |
| spe_berg       | ok      | 1498  |
| volve          | missing | 7     |
| volve          | ok      | 189   |

## Unavailable reasons (sample)

    _No rows._

## Notes

- Expected grid is built from the four forecasting data configs and the discovered/default forecasting model set.
- `missing` means a model × dataset × horizon × mode × well/scenario artifact is expected by config but no aggregate-ingestable JSON currently exists.
- `unavailable` means an artifact exists but records dependency/runtime unavailability rather than metrics.
