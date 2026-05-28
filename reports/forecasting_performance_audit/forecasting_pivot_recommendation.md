# Forecasting Pivot Recommendation

## Recommendation

Proceed next with **metric repair + MASE-first table regeneration before model/objective/HPO work**.

## Evidence

- Coverage status counts: `{'ok': 2737, 'missing': 35}`.
- Denominator source counts in the Gate 2 audit: `{'raw_train': 220, 'ad_hoc_validation_fallback': 2}`.
- Audit classification counts: `{'valid_stored': 220, 'path_audited_no_current_artifact': 2}`.
- Zero-MASE/nonzero-MAE audited records: `0`.
- Audited records with absolute recomputed-vs-stored MASE delta > 0.25: `135`.

## Rank sensitivity snapshot

| model    | stored_mean_rank | effective_mean_rank | rank_delta |
| -------- | ---------------- | ------------------- | ---------- |
| tirex    | 4.032            | 1.188               | -2.845     |
| timesfm  | 5.935            | 3.188               | -2.748     |
| chronos  | 4.71             | 2.031               | -2.678     |
| patchtst | 2.839            | 5.094               | 2.255      |
| lstm     | 2.516            | 4.75                | 2.234      |
| tcn      | 3.29             | 5.219               | 1.928      |
| deeponet | 4.677            | 6.531               | 1.854      |

## Ranked next steps

1. **Repair metric implementation and metadata**: make zero denominator with nonzero MAE invalid/inf instead of silently `0.0`; store denominator source, raw naive MAE, and whether `y_train` was used.
2. **Repair trained-model denominator collection**: stop denormalizing raw dataset targets a second time; preserve chronological/group provenance for MASE scale data.
3. **Regenerate MASE-first tables from repaired metrics**: use the weighted dataset × horizon × mode policy generated in this audit.
4. **Rerun only where artifacts are insufficient**: trained JSONs usually lack predictions/targets, so controlled reruns may be needed after metric repair if stored MAE + raw denominator is not accepted as sufficient.
5. **Only then start model/objective/HPO improvements**: tune against validated MASE and keep any shutdown-filter/window/protocol sensitivity dual-reported.

## Pivot lane

Default lane: `metric repair + table regeneration without broad production reruns`.

Escalate to controlled reruns only for rows where repaired MASE cannot be recomputed from trustworthy existing artifacts.
