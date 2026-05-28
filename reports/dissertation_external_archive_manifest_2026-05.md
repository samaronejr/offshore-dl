# Dissertation External Archive Manifest — 2026-05

Created: 2026-05-28
Working branch: `fix/dissertation-finalization-validation`
Base source commit before finalization edits: `ca6a042cfd82171f85f55581f571feb57f8bbaba`
Durable local storage directory: `/home/samarone/offshore_dl_archives/dissertation_2026-05`

This manifest records local/off-repository archive storage only. It contains no access tokens, private keys, passwords, or remote credentials. The final self-referential Git commit hash is intentionally not embedded here; use the final annotated tag `dissertation-evidence-2026-05` after freeze to identify the committed repository revision.

## Archived files

| File | Source path | Stored path | Size (bytes) | SHA256 | Purpose |
|---|---|---|---:|---|---|
| `offshore_dl_dissertation_evidence_20260527.tar.gz` | `dist/offshore_dl_dissertation_evidence_20260527.tar.gz` | `/home/samarone/offshore_dl_archives/dissertation_2026-05/offshore_dl_dissertation_evidence_20260527.tar.gz` | 969298 | `5f634baf4d82c2fc674f0366ab61a73fd55adfcd62d18ea0b73d1f7426ee46ac` | Compact dissertation-facing evidence bundle: README guidance, claim ledger, final tables, narrative guide, prose-audit ledger, CDF compact summary, Borda diagnostics, forecasting audit files, aggregate CSVs, and regenerated dissertation/IEEE LaTeX-PDF snapshots. |
| `forecasting_post_fix_20260527.tar.gz` | `/home/samarone/offshore_dl_archives/forecasting_post_fix_20260527.tar.gz` | `/home/samarone/offshore_dl_archives/dissertation_2026-05/forecasting_post_fix_20260527.tar.gz` | 1275007526 | `9a1eb8c6c3c1f8b14f55ecfd6b1ab951acaffbad761fcd1a6372a2df1ab03e90` | Full external forecasting post-fix archive, including large per-model/per-dataset result JSON artifacts not intended for Git. |

## Verification commands

```bash
sha256sum -c dist/offshore_dl_dissertation_evidence_20260527.tar.gz.sha256
sha256sum -c ~/offshore_dl_archives/forecasting_post_fix_20260527.tar.gz.sha256
sha256sum /home/samarone/offshore_dl_archives/dissertation_2026-05/offshore_dl_dissertation_evidence_20260527.tar.gz
sha256sum /home/samarone/offshore_dl_archives/dissertation_2026-05/forecasting_post_fix_20260527.tar.gz
```

Expected copied-file checksums:

```text
5f634baf4d82c2fc674f0366ab61a73fd55adfcd62d18ea0b73d1f7426ee46ac  /home/samarone/offshore_dl_archives/dissertation_2026-05/offshore_dl_dissertation_evidence_20260527.tar.gz
9a1eb8c6c3c1f8b14f55ecfd6b1ab951acaffbad761fcd1a6372a2df1ab03e90  /home/samarone/offshore_dl_archives/dissertation_2026-05/forecasting_post_fix_20260527.tar.gz
```

## Git policy

- The compact bundle and full archive remain outside Git/ignored storage.
- The full `results/post_fix/<model>/*.json` tree is not committed.
- Raw and processed datasets under `data/` are not included.
