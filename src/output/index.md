# Check Models Output Index

Generated on: 2026-07-13 05:57:30 BST

## Run Snapshot

- Evaluation lane: assisted
- Metadata exposed to prompt: yes
- Selection basis: metadata-assisted visual verification (grounded)
- Models tested: 61
- Successful: 60
- Failed: 1

## Primary Artifacts

- [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) is the maintainer-first failure and integration route.
- [results.html](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.html) is the complete standalone interactive run report.
- [model_selection.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_selection.md) is the reliability-gated current-run chooser.
- [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) preserves complete per-model output evidence.
- [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) is the primary per-model machine-readable stream.

## Supporting Artifacts

- [results.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.md) is the compatibility Markdown summary.
- [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md) is the supporting automated-review digest.
- [model_capabilities.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_capabilities.md) aggregates lane-matched current and historical capability signals.
- [results.tsv](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.tsv) supports spreadsheet inspection.
- [results.history.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.history.jsonl) is the append-only supporting history stream.
- [run.json](https://github.com/jrp2014/check_models/blob/main/src/output/run.json) contains the stable run-level contract.
- [issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md) indexes supporting issue drafts.
- [latest_by_cluster.json](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/latest_by_cluster.json) indexes supporting reproduction bundles.

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
