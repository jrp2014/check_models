# Check Models Output Index

Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed

This run records model responses to one shared image and prompt (evaluation
lane: assisted). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run at a glance

- Run duration: 23m 02s
- Evaluation lane: assisted
- Prompt hints: the image's description and keyword hints were included in the prompt, so field content may be copied from them rather than seen
- Assessment: General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed
- Input image: JPEG, 9,984 x 6,656 pixels (66.5 MP), 44.7 MB
- Models attempted: 41 (completed 38, crashed 3, indeterminate 0)
- Mechanical checks: no concerns detected 28, concerns detected 4, major concerns 6, not assessed 3
- Top observations: Response repeats the same text (1), Generation was stopped early after sustained repeated output (2), Unrecognised model control tokens remain visible (3), Required labelled fields not detected (4), Response appears cut off at the token limit (2)

## Start here

- [Run summary](https://github.com/jrp2014/check_models/blob/main/src/output/issues/run_summary.md) — per-model quality ranking, crash triage, and paste-ready issue body

## Artifacts

- [results.html (self-contained page; download to view, GitHub shows its source)](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.html)
- [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)
- [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl)
- [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)
- [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log)

## Issue drafts

- [mlx-community/InternVL3_5-30B-A3B-4bit](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_InternVL3_5-30B-A3B-4bit.md)
- [mlx-community/Llama-3.2-11B-Vision-Instruct-4bit](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Llama-3.2-11B-Vision-Instruct-4bit.md)
- [mlx-community/Mage-VL-OptiQ-4bit](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Mage-VL-OptiQ-4bit.md)
