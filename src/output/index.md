# Check Models Output Index

Run: started 2026-10-16 23:15:50 BST; check_models 0.17.71 @ 5bfc8f11d

Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed

This run records model responses to one shared image and prompt (evaluation
lane: blind). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run at a glance

- Run duration: 5m 34s
- Evaluation lane: blind
- Prompt hints: no description or keyword hints
- Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed
- Input image: JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- Models attempted: 51 (completed 51, crashed 0, indeterminate 0)
- Mechanical checks: no concerns detected 41, concerns detected 0, major concerns 10 (generation 4, answer format 6), not assessed 0
- Observations, most important first: Response repeats the same text (4), Generation was stopped early after sustained repeated output (1), Final answer emitted twice (1), Unrecognised model control tokens remain visible (2), Required labelled fields not detected (6); 2 more kinds in diagnostics

## Start here

- [Run summary](https://github.com/jrp2014/check_models/blob/main/src/output/issues/run_summary.md) — per-model quality ranking, crash triage, and paste-ready issue body

## Artifacts

- [results.html (self-contained page; download to view, GitHub shows its source)](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.html)
- [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)
- [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl)
- [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)
- [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log)
