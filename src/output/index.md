# Check Models Output Index

These models serve many purposes; this run probes exactly one narrow task:
producing catalogue metadata for a single photograph from the assisted-lane
prompt and whatever context it supplies — here, camera-recorded capture
context plus draft descriptive hints previously produced by a more capable
model. Results say nothing about a model's fitness for other uses.

## Run at a glance

- Models attempted: 42 (completed 41, crashed 1, indeterminate 0)
- Usability: usable 15, usable with caveats 18, unusable 8, not evaluated 1
- Top observations: Response repeats the same text (2), Unrecognised model control tokens remain visible (1), Required fields are missing or empty (5), Response repeats the task instructions instead of only returning the requested fields (2), Extra text appears before the Title field (4)

## Start here

- [Run summary](https://github.com/jrp2014/check_models/blob/main/src/output/issues/run_summary.md) — per-model quality ranking, crash triage, and paste-ready issue body

## Artifacts

- [results.html](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.html)
- [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)
- [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl)
- [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)
- [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log)

## Issue drafts

- [tencent/Youtu-VL-4B-Instruct](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_tencent_Youtu-VL-4B-Instruct.md)
