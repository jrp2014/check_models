# Paste-Ready Run Issue Summary Design

## Problem

`reports/diagnostics.md` is the complete maintainer evidence for a run, so a
large model matrix makes it too long to paste into one GitHub issue. Individual
`issues/issue_*.md` files cover hard crashes, but they do not provide the desired
cross-model view. Maintainers need one compact issue body that explains the run,
surfaces gross failures, lists every other highlighted model, and points to the
full retained evidence.

## Goals

- Generate `src/output/issues/run_summary.md` as a complete, paste-ready GitHub
  issue body for the current run.
- Keep the report compact and skimmable while covering the whole run.
- Give actionable failures, especially crashes, more detail than ordinary
  mechanical observations.
- Link every surfaced model to its complete evidence in
  `reports/diagnostics.md`.
- Generate the same artifact from retained `results.jsonl` and `run.json`
  without rerunning any model.
- Preserve the cached assessment as authoritative; regeneration must not
  reclassify model results.

## Non-Goals

- Do not replace the full diagnostics, gallery, JSONL, run JSON, logs, or
  per-crash issue drafts.
- Do not infer likely ownership, root cause, model quality, or recommended fixes.
- Do not embed complete tracebacks, model output, prompts, Python reproduction
  scripts, or full environment inventories.
- Do not list clean models individually; report their count and link to the
  gallery instead.

## Artifact Contract

The artifact lives at `output/issues/run_summary.md`. It is generated whenever
the retained run has an actionable failure, an observation requiring
reproduction, or an indeterminate attempt. A completely clean run removes any
stale summary instead of producing an issue body with nothing to report.

The entire file is safe to paste into a GitHub issue body. Its first-level
heading is also the suggested issue title. For a representative 62-model run,
the target is approximately 100–150 lines. Growth is bounded primarily by the
number of highlighted models, not by traceback, output, prompt, or environment
size.

## Content and Ordering

1. **Title and scope** — a neutral heading such as “mlx-vlm compatibility
   findings across 62 cached vision-language models”, followed by the run time,
   evaluation lane, attempted/completed/crashed/indeterminate counts, and a
   reminder that observations are mechanical facts from one image.
2. **Actionable failures** — one expanded subsection per actionable failure,
   ordered as in the canonical result set. Each subsection contains:
   - model identifier and resolved revision;
   - failure phase, stage, exception type, and root exception chain;
   - a one-line parameterised reproduction invocation using the retained image,
     revision, and prompt-file convention;
   - links to the model anchor in full diagnostics and its individual crash
     draft when one exists.
3. **Other surfaced results** — one compact table, excluding the expanded
   actionable failures, with columns `Model`, `Execution / usability`,
   `Observations`, and `Full evidence`. It includes every observation-needing-
   reproduction or indeterminate model and preserves canonical result order.
4. **Clean completions** — one sentence with the clean count and a link to the
   full gallery.
5. **Run context** — compact generation settings and only the core environment
   facts needed for triage: macOS, chip, Python, `mlx-vlm`, `mlx`, and
   `transformers`.
6. **Full artifacts** — links to diagnostics, gallery, JSONL, run JSON,
   environment log, and execution log.

Observation labels use the existing human-readable mapping. Missing values are
omitted rather than rendered as `unavailable`. Tables contain no inferred next
action; their evidence links make the factual rows actionable.

## Data Flow and Integration

The summary renderer consumes the schema `2.0` metadata and result records in
`results.jsonl`, plus optional run-level fields from `run.json`. It reads the
serialized `assessment` object verbatim and never calls `_assess_result()`.
This provides one implementation path for both normal end-of-run generation and
report-only regeneration from retained output.

Normal finalization continues to write canonical JSONL first. After diagnostics,
per-crash drafts, and run JSON are available, it renders `run_summary.md`, then
writes the output index with a prominent `Run issue summary` link followed by
the individual crash drafts. Summary-generation failure is isolated like other
optional report failures and cannot corrupt JSONL or diagnostics.

A callable report-only entry point accepts an output directory, reads its
`results.jsonl` and optional `run.json`, discovers current per-crash drafts, and
writes or removes `issues/run_summary.md`. This entry point is used once to
generate the requested artifact from the existing checked-in output without
model inference.

Links use the repository's existing artifact-target helper: temporary test
outputs receive relative links, while default production output receives
publication-ready GitHub links. Model evidence links append the existing stable
diagnostics anchors.

## Stale and Invalid Input Handling

- Before rendering, remove a stale `run_summary.md` when the current records have
  no surfaced results.
- Reject a missing JSONL file, missing metadata header, unsupported schema
  version, malformed JSON, or malformed assessment with a clear `ValueError` or
  `OSError`; normal report orchestration records the isolated artifact failure.
- Treat `run.json` as optional enrichment. If it is absent or malformed, retain
  the summary using JSONL metadata and omit unavailable run-level fields.
- Never modify `results.jsonl`, `run.json`, diagnostics, logs, gallery, or
  per-crash issue drafts during report-only regeneration.

## Documentation

Update the README artifact list and output tree to describe
`issues/run_summary.md` as the compact aggregate issue body. Update the
`[Unreleased]` changelog and retain the existing description of diagnostics as
the complete evidence source.

## Testing and Acceptance

Add focused tests to existing report-generation and Markdown-formatting test
files that prove:

- a mixed run produces the ordered sections and compact surfaced-results table;
- crashes are expanded outside the table with root facts, revision, minimal
  reproduction, and evidence links;
- clean models are counted but not listed;
- no complete traceback, generated output, full prompt, or Python script leaks
  into the summary;
- cached JSONL assessments are preserved without reclassification;
- a clean run removes a stale summary;
- malformed or unsupported retained input fails clearly without mutating source
  artifacts;
- the output index links the aggregate summary before individual crash drafts;
- generated Markdown passes the repository Markdown lint configuration.

Finally, render `src/output/issues/run_summary.md` from the existing retained
`src/output/results.jsonl` and `src/output/run.json`, inspect its size and links,
and run the full prescribed quality gate.
