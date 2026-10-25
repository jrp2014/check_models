# Check Models Output Index

Run: started 2026-10-25 01:53:46 BST; check_models 0.17.91 @ baa72d7e8

Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed

This run records model responses to one shared image and prompt (evaluation
lane: blind). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run at a glance

- Run duration: 4m 10s
- Evaluation lane: blind
- Prompt hints: no description or keyword hints
- Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed
- Input image: JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- Models attempted: 43 (completed 43, crashed 0, indeterminate 0)
- Mechanical checks: no concerns detected 37, concerns detected 1, major concerns 5 (generation 2, answer format 3), not assessed 0
- Observations, most important first: Response repeats the same text (2), Unrecognised model control tokens remain visible (1), Required labelled fields not detected (3), Response appears cut off at the token limit (2), Repeated keyword entries (3)

## Start here

- [Run summary](https://github.com/jrp2014/check_models/blob/main/src/output/issues/run_summary.md) — per-model quality ranking, crash triage, and paste-ready issue body

## Artifacts

- [results.html (self-contained page; download to view, GitHub shows its source)](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.html)
- [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)
- [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl)
- [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)
- [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log)
