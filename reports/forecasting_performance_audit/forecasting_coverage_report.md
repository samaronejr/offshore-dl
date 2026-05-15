# Forecasting Coverage Audit

## Global status distribution

| status  | count |
| ------- | ----- |
| missing | 2548  |
| ok      | 224   |

## Status by dataset

| dataset        | status  | count |
| -------------- | ------- | ----- |
| ganymede       | ok      | 224   |
| inner_mongolia | missing | 840   |
| spe_berg       | missing | 1512  |
| volve          | missing | 196   |

## Unavailable reasons (sample)

    _No rows._

## Notes

- Expected grid is built from the four forecasting data configs and the discovered/default forecasting model set.
- `missing` means a model × dataset × horizon × mode × well/scenario artifact is expected by config but no aggregate-ingestable JSON currently exists.
- `unavailable` means an artifact exists but records dependency/runtime unavailability rather than metrics.
