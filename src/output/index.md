# Check Models Output Index

Run: started 2026-11-11 18:43:43 GMT; check_models 0.17.119 @ 95efedab3

Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed

This run records model responses to one shared image and prompt (evaluation
lane: blind). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run at a glance

- Run duration: 3m 44s
- Evaluation lane: blind
- Prompt hints: no description or keyword hints
- Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed
- Input image: JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- Models attempted: 41 (completed 41, crashed 0, indeterminate 0)
- Mechanical checks: no concerns detected 34, concerns detected 1, major concerns 6 (generation 1, answer format 5), not assessed 0
- Observations, most important first: Response repeats the same text (1), Unrecognised model control tokens remain visible (2), Required labelled fields not detected (5), Response appears cut off at the token limit (2), Conversation-role control tokens remain visible (1); 1 more kind in diagnostics

## Start here

- [Run summary](https://github.com/jrp2014/check_models/blob/main/src/output/issues/run_summary.md) — per-model quality ranking, crash triage, and paste-ready issue body

## Artifacts

- [results.html (self-contained page; download to view, GitHub shows its source)](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.html)
- [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)
- [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl)
- [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)
- [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log)
