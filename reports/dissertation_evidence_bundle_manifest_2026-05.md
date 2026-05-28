# Dissertation Evidence Bundle Manifest — 2026-05

Bundle path: `dist/offshore_dl_dissertation_evidence_20260527.tar.gz`
Checksum path: `dist/offshore_dl_dissertation_evidence_20260527.tar.gz.sha256`

## SHA256

`5f634baf4d82c2fc674f0366ab61a73fd55adfcd62d18ea0b73d1f7426ee46ac`

## Contents policy

The bundle includes lightweight dissertation-facing evidence: README guidance, claim ledger, final tables, narrative guide, prose-audit ledger, forecasting aggregate CSVs, regenerated dissertation and IEEE paper LaTeX/PDF snapshots, CDF compact summary, Borda diagnostics, and forecasting audit/manifests. It intentionally excludes raw data, the large per-model forecasting JSON tree, this manifest file, and the external archive manifest to avoid checksum self-reference; use `~/offshore_dl_archives/forecasting_post_fix_20260527.tar.gz` and its `.sha256` for the external full forecasting JSON archive.

## Verification

Created with:

`tar -czf dist/offshore_dl_dissertation_evidence_20260527.tar.gz ...`
`sha256sum dist/offshore_dl_dissertation_evidence_20260527.tar.gz > dist/offshore_dl_dissertation_evidence_20260527.tar.gz.sha256`
