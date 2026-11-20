---
name: weekly-sweep-update
description: >
  Drafts factual, maintainer-ready weekly diagnostic sweep updates from
  retained check_models artifacts (results.jsonl, run_summary.md,
  diagnostics.md, model_gallery.md, accuracy_notes.yaml, and
  native_repro_notes.yaml). Use when preparing a periodic sweep summary or
  upstream status update without overstating comparability or unverified fixes.
---

# Weekly Sweep Update

Prepare a concise, evidence-grounded diagnostic update from a retained
`check_models` sweep (for example `src/output/issues/draft_weekly_sweep_YYYYMMDD.md`).
Follow the [canonical project guidance](../../../.github/copilot-instructions.md)
for authorization boundaries: drafting local Markdown does not authorize
publishing upstream, running inference, or modifying caches.

## 1. Read the primary retained evidence

Inspect these artifacts in order before writing prose:

1. `src/output/issues/run_summary.md` — counts, phase timings, baseline
   comparability verdict, upstream commit ranges, and mechanical checks table.
2. `src/output/reports/diagnostics.md` — highlighted triage cases, complete
   observed outputs, processor/tokenizer provenance, and system environment.
3. `src/output/results.jsonl` — exact machine facts (metadata header, per-model
   `rendered_prompt`, `generate_kwargs`, `sampling_sources`, `tail_tokens`,
   `system_telemetry`).
4. `src/output/reports/model_gallery.md` — final answers for manual captioning
   notes (strictly labelled as manual inspection, never runtime bug claims).
5. `src/output/native_repro_notes.yaml` and `src/output/accuracy_notes.yaml` —
   confirmed native checks and human accuracy notes that apply to this run.

## 2. Check comparability and image-independent shifts

- State the evaluation lane (`blind`, `assisted`, or `triage`), input image
  dimensions/byte size/SHA-256, and whether the original photograph is
  published.
- If the image, prompt, lane, hardware, or execution mode changed from the
  baseline sweep, state explicitly that outputs and timings are **not a
  controlled regression benchmark** and withhold speed/fix claims.
- Still report **image-independent contract shifts** visible in `results.jsonl`
  and upstream commit ranges:
  - models switching between pre-closed and pre-opened thinking templates,
  - changes in how many models trigger the automatic thinking budget,
  - processor, tokenizer, image-processor, or EOS ID changes,
  - upstream `mlx` / `mlx-vlm` / `transformers` commit ranges, clearly naming
    which changes were actually exercised by the tested roster vs. untested
    surfaces (server, training, new families).

## 3. Audit every anomaly beyond automatic routing

- Start from crashes and `observation_needs_reproduction` models, then scan all
  `major concerns` and `concerns detected` rows so genuine structural anomalies
  (duplicate final answers, token-cap exhaustion inside unclosed thinking
  blocks, incomplete answers under checkpoint sampling defaults) are never
  missed merely because of a routing category.
- For each highlighted case, cite exact facts from `results.jsonl`:
  generated token count, stop reason, whether the rendered prompt pre-opened a
  thinking block, whether the auto thinking budget engaged, effective sampling
  settings, tail token IDs (if recorded), and whether a native reproduction on
  this build/input has been run.
- Never claim an upstream bug is fixed or reproduced unless a matched control
  or native reproduction established it.

## 4. Separate captioning usefulness from runtime bugs

- Put manual observations on caption/keyword quality in a clearly separated
  section labelled as manual review, not runtime bug claims.
- Note objective hint-reuse context in `assisted` runs (such as non-visual
  source tags retained verbatim from keyword hints, or high hint-text overlap)
  so readers do not mistake hint copying for independent visual recognition.

## 5. Timing context and proposed follow-ups

- Report whole-sweep duration and phase breakdown (`model_load`, `prompt_prep`,
  `generation`, `cleanup`, plus prefill/first-token share), power source,
  start-of-run available RAM / swap, and any outside GPU activity.
- Note that single-pass sweeps have no warm-up or repeats; avoid throughput
  regression claims.
- Put any unrun native follow-up commands in a `<details>` block marked clearly
  as **not run**, naming whether they require the unpublished original image or
  a digest-verified stand-in preview.
