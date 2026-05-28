# Dissertation Prose Audit — 2026-05-28

Purpose: dispose stale or ambiguous dissertation-facing prose against the current claim ledger.

Source of truth:
- `reports/dissertation_claim_ledger_2026-05.md`
- `reports/dissertation_final_tables_2026-05.md`
- `reports/dissertation_results_narrative_2026-05.md`

## Dispositions

| Artifact | Disposition | Claim-control rationale |
| --- | --- | --- |
| `reports/ieee_paper.tex` | Marked historical with a visible May 2026 validity notice. | The draft contains older forecasting and classification prose; the notice blocks citation as current evidence and points to the claim ledger/final tables. |
| `reports/dissertation_result_manifest_2026-05.md` | Marked historical/superseded. | May-15 manifest contains earlier source revision and older Ganymede aggregates; current claims use `C-*` IDs in the claim ledger. |
| `reports/dissertation_narrative_checklist_2026-05.md` | Marked historical/superseded. | Current narrative control moved to `reports/dissertation_results_narrative_2026-05.md`. |
| `docs/dissertation_readiness_runbook_2026-05.md` | Marked execution-history/superseded. | The runbook documents the May-15 CDF gate; current dissertation claims/tables are controlled by the May-28 evidence pack. |
| `reports/README.md` | Updated current-status table. | It now distinguishes current claim-control artifacts from historical manifests/checklists and lists the external archive manifest. |
| `reports/dissertation_results_2026-05.tex` / `.pdf` | Refreshed and compiled in the table-integration phase. | The snapshot now transcribes `reports/dissertation_final_tables_2026-05.md` and is current for the May-28 evidence freeze. |

## Current prose rules verified

- Stage 1 3W and Stage 2 3W are separate (`C-3W-S1-*`, `C-3W-S2-*`).
- Forecasting claims name metric family before model winner (`C-FC-GAN-*`, `C-FC-BORDA-*`).
- Sparse h90 rows are data-coverage exclusions, not compute failures (`C-FC-SPARSE-001`).
- CDF trained reconstruction and foundation forecast rows remain separate (`C-CDF-*`).
- Historical/pre-fix artifacts are audit lineage only (`C-HIST-*`).

## Table-integration status

Completed in the May-28 finalization pass: `reports/dissertation_results_2026-05.tex` was regenerated from `reports/dissertation_final_tables_2026-05.md`, compiled to `reports/dissertation_results_2026-05.pdf`, and included in the refreshed compact evidence bundle. Treat this snapshot as current for the May-28 evidence freeze, subject to the claim ledger and metric/family guardrails above.
