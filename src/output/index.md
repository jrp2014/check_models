# Check Models Output Index

Run: started 2026-10-16 13:52:16 BST; check_models 0.17.55 @ c960bfb1e

Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed

This run records model responses to one shared image and prompt (evaluation
lane: assisted). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run at a glance

- Run duration: 10m 44s
- Evaluation lane: assisted
- Prompt hints: the image's description and keyword hints were included in the prompt, so field content may be copied from them rather than seen
- Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed
- Input image: JPEG, 6,213 x 9,320 pixels (57.9 MP), 52.1 MB
- Models attempted: 47 (completed 47, crashed 0, indeterminate 0)
- Mechanical checks: no concerns detected 26, concerns detected 10, major concerns 11 (generation 3, answer format 8), not assessed 0
- Observations, most important first: Response repeats the same text (3), Generation was stopped early after sustained repeated output (2), Unrecognised model control tokens remain visible (2), Required labelled fields not detected (10), Response appears cut off at the token limit (2); 2 more kinds in diagnostics

## Start here

- [Run summary](https://github.com/jrp2014/check_models/blob/main/src/output/issues/run_summary.md) — per-model quality ranking, crash triage, and paste-ready issue body

## Artifacts

- [results.html (self-contained page; download to view, GitHub shows its source)](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.html)
- [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)
- [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl)
- [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)
- [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log)
