# Check Models Output Index

Generated on: 2026-07-04 00:01:55 BST

## Run Snapshot

- Mode: triage
- Selection basis: ungrounded (ungrounded)
- Models tested: 61
- Successful: 57
- Failed: 0

## For Model Users

- Start with [model_selection.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_selection.md) for the practical shortlist and memory/speed buckets.
- Use [model_capabilities.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_capabilities.md) for current status plus historical reliability.
- Inspect [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) before treating a triage-clean model as visually correct.

## For Maintainers

- Start with [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) for owner grouping, traceback excerpts, and failure clusters.
- Use [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md) for paste-ready upstream issue drafts.
- Use [latest_by_cluster.json](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/latest_by_cluster.json) to find the current-run repro bundles without scanning the archive.

## Machine-Readable Data

- [run.json](https://github.com/jrp2014/check_models/blob/main/src/output/run.json) contains the stable run-level contract.
- [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) contains per-model diagnostics, timings, and review payloads.
- [results.tsv](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.tsv) is optimized for spreadsheet inspection.
