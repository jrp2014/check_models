# Check Models Output Index

Run: started 2026-10-10 23:35:53 BST; check_models 0.17.42 @ 505bca0c6

Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed

This run records model responses to one shared image and prompt (evaluation
lane: assisted). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run at a glance

- Run duration: 13m 37s
- Evaluation lane: assisted
- Prompt hints: the image's description and keyword hints were included in the prompt, so field content may be copied from them rather than seen
- Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed
- Input image: JPEG, 5,800 x 8,389 pixels (48.7 MP), 40.8 MB
- Models attempted: 53 (completed 53, crashed 0, indeterminate 0)
- Mechanical checks: no concerns detected 30, concerns detected 9, major concerns 14 (generation 6, answer format 8), not assessed 0
- Observations, most important first: Response repeats the same text (5), Generation was stopped early after sustained repeated output (3), Unrecognised model control tokens remain visible (2), Required labelled fields not detected (9), Response appears cut off at the token limit (5); 5 more kinds in diagnostics

## Start here

- [Run summary](https://github.com/jrp2014/check_models/blob/main/src/output/issues/run_summary.md) — per-model quality ranking, crash triage, and paste-ready issue body

## Artifacts

- [results.html (self-contained page; download to view, GitHub shows its source)](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.html)
- [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)
- [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl)
- [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)
- [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log)
