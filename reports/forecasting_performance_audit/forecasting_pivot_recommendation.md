# Forecasting Pivot Recommendation

## Recommendation

Proceed next with **metric repair + MASE-first table regeneration before model/objective/HPO work**.

## Evidence

- Coverage status counts: `{'missing': 2161, 'unavailable': 352, 'ok': 259}`.
- Denominator source counts in the Gate 2 audit: `{'unavailable': 352, 'double_denorm_suspect': 240, 'validation_target_fallback': 64, 'ad_hoc_validation_fallback': 2}`.
- Audit classification counts: `{'unavailable': 352, 'suspect_stored_recomputable': 201, 'invalid_stored_recomputable': 103, 'path_audited_no_current_artifact': 2}`.
- Zero-MASE/nonzero-MAE audited records: `103`.
- Audited records with absolute recomputed-vs-stored MASE delta > 0.25: `237`.

## Rank sensitivity snapshot

| model    | stored_mean_rank | effective_mean_rank | rank_delta |
| -------- | ---------------- | ------------------- | ---------- |
| lstm     | 3.438            | 1.875               | -1.562     |
| timesfm  | 2.875            | 3.625               | 0.75       |
| tirex    | 1                | 1.75                | 0.75       |
| tcn      | 4.444            | 5                   | 0.5556     |
| chronos  | 1.688            | 2.188               | 0.5        |
| patchtst | 4.917            | 5.167               | 0.25       |
| deeponet | 4.688            | 4.5                 | -0.1875    |

## Ranked next steps

1. **Repair metric implementation and metadata**: make zero denominator with nonzero MAE invalid/inf instead of silently `0.0`; store denominator source, raw naive MAE, and whether `y_train` was used.
2. **Repair trained-model denominator collection**: stop denormalizing raw dataset targets a second time; preserve chronological/group provenance for MASE scale data.
3. **Regenerate MASE-first tables from repaired metrics**: use the weighted dataset × horizon × mode policy generated in this audit.
4. **Rerun only where artifacts are insufficient**: trained JSONs usually lack predictions/targets, so controlled reruns may be needed after metric repair if stored MAE + raw denominator is not accepted as sufficient.
5. **Only then start model/objective/HPO improvements**: tune against validated MASE and keep any shutdown-filter/window/protocol sensitivity dual-reported.

## Pivot lane

Default lane: `metric repair + table regeneration without broad production reruns`.

Escalate to controlled reruns only for rows where repaired MASE cannot be recomputed from trustworthy existing artifacts.
