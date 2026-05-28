# Dissertation Evidence Bundle Manifest — 2026-05

Bundle path: `dist/offshore_dl_dissertation_evidence_20260527.tar.gz`
Checksum path: `dist/offshore_dl_dissertation_evidence_20260527.tar.gz.sha256`

## SHA256

`3175f8059d987f960a16f1395f7872480c70f57f81e0586b6791ed3c1d466be6`

## Contents policy

The bundle includes lightweight dissertation-facing evidence: README guidance, claim ledger, final tables, narrative guide, forecasting aggregate CSVs, CDF compact summary, Borda diagnostics, and forecasting audit/manifests. It intentionally excludes raw data, the large per-model forecasting JSON tree, and this manifest file itself to avoid checksum self-reference; use `~/offshore_dl_archives/forecasting_post_fix_20260527.tar.gz` and its `.sha256` for the external full forecasting JSON archive.

## Verification

Created with:

`tar -czf dist/offshore_dl_dissertation_evidence_20260527.tar.gz ...`
`sha256sum dist/offshore_dl_dissertation_evidence_20260527.tar.gz > dist/offshore_dl_dissertation_evidence_20260527.tar.gz.sha256`
