# Output Report Integrity Design

## Goal

Make retained reports accurately classify prompt-seeded thinking wrappers, avoid
presenting stale run logs as current evidence, and keep issue and gallery summaries
compact without discarding exact evidence.

## Scope

The change covers four accepted findings:

1. A complete empty thinking block in the rendered prompt is neutral. Only an
   opening delimiter in generated text, or an opening delimiter seeded without a
   matching seeded/generated close, may be incomplete.
2. Report-only regeneration must detect when `check_models.log` or
   `environment.log` clearly predates the JSONL run represented by the reports.
   Stale files remain untouched but their issue-ready links are replaced by an
   explicit unavailable/stale fact.
3. Aggregate crash reports show the exception type, the number of repeated
   unexpected model parameters, and representative parameter families. Complete
   exception evidence remains in diagnostics and the dedicated crash draft.
4. Markdown chooser tables omit output previews. Full generated output remains in
   per-model gallery sections and the HTML report retains its richer chooser.

## Design

### Reasoning classification

`_detect_reasoning_output()` will classify delimiters by their source. A delimiter
pair fully present in `seeded_text` does not create a generated trace. A seeded
opening delimiter without a seeded close may be completed by the generated close.
An opening delimiter present in generated text still requires a generated close.
This preserves valid Qwen-style prompt-prefilled thinking while leaving genuine
Kimi-style token-cap failures unusable.

### Artifact integrity

When run JSON supplies both `generated_at` and `total_runtime_seconds`, their
difference defines the retained run window. For the two human log files, the first
parseable timestamp is compared with that window using a small clock tolerance. A
timestamp outside the window is marked stale; missing run timing or unparseable
legacy files remain available rather than producing a speculative warning.
Issue-ready artifact tables will omit only positively stale links and explain the
omission.

### Compact crashes

Aggregate issue summaries will compact long `Received N parameters not in model`
messages. The compact form retains `N`, a bounded sample, and grouped top-level
families such as `audio_tower` and `language_model`. Other exception messages are
unchanged. Diagnostics and dedicated issue drafts retain exact full messages.

### Markdown gallery

The current-run and avoid tables will contain model, assessment, observations and
resource facts only. Output previews are redundant with the linked per-model
sections and are removed from Markdown. The HTML selector remains unchanged.

## Validation

- Focused quality-analysis tests distinguish a complete seeded empty block from a
  seeded open block completed or left incomplete by generation.
- Report tests use temporary paths to prove stale links are withheld, current links
  remain, aggregate crash evidence is bounded, and full evidence is retained.
- Markdown tests assert chooser headers and rows no longer contain previews while
  per-model output remains.
- Report-only regeneration runs against the current JSONL without appending history
  or modifying the stale log/environment files.
- The normal format, lint and full quality gates must pass.

## Non-goals

- No semantic image-quality scorer is introduced.
- No run UUID or schema-version migration is added.
- No model inference is rerun.
- No stale or missing evidence is reconstructed.
