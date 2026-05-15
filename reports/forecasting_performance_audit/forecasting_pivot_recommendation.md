# Forecasting Pivot Recommendation

## Recommendation

Proceed next with **metric repair + MASE-first table regeneration before model/objective/HPO work**.

## Evidence

- Coverage status counts: `{'missing': 2548, 'ok': 224}`.
- Denominator source counts in the Gate 2 audit: `{'raw_train': 224, 'ad_hoc_validation_fallback': 2}`.
- Audit classification counts: `{'valid_stored': 224, 'path_audited_no_current_artifact': 2}`.
- Zero-MASE/nonzero-MAE audited records: `0`.
- Audited records with absolute recomputed-vs-stored MASE delta > 0.25: `106`.

## Rank sensitivity snapshot

| model    | stored_mean_rank | effective_mean_rank | rank_delta |
| -------- | ---------------- | ------------------- | ---------- |
| patchtst | 3                | 7                   | 4          |
| timesfm  | 5.625            | 2.75                | -2.875     |
| tirex    | 3.75             | 1                   | -2.75      |
| chronos  | 5.25             | 2.75                | -2.5       |
| lstm     | 1.875            | 3.875               | 2          |
| tcn      | 3.75             | 4.875               | 1.125      |
| deeponet | 4.75             | 5.75                | 1          |

## Ranked next steps

1. **Repair metric implementation and metadata**: make zero denominator with nonzero MAE invalid/inf instead of silently `0.0`; store denominator source, raw naive MAE, and whether `y_train` was used.
2. **Repair trained-model denominator collection**: stop denormalizing raw dataset targets a second time; preserve chronological/group provenance for MASE scale data.
3. **Regenerate MASE-first tables from repaired metrics**: use the weighted dataset × horizon × mode policy generated in this audit.
4. **Rerun only where artifacts are insufficient**: trained JSONs usually lack predictions/targets, so controlled reruns may be needed after metric repair if stored MAE + raw denominator is not accepted as sufficient.
5. **Only then start model/objective/HPO improvements**: tune against validated MASE and keep any shutdown-filter/window/protocol sensitivity dual-reported.

## Pivot lane

Default lane: `metric repair + table regeneration without broad production reruns`.

Escalate to controlled reruns only for rows where repaired MASE cannot be recomputed from trustworthy existing artifacts.
