# Issue-ready Diagnostics and Formatted Gallery Design

**Status:** Approved on 2026-07-26

## Context

The retained reports serve two different audiences:

- `model_gallery.md` helps people compare locally run vision-language models
  for image captioning and descriptive metadata.
- `diagnostics.md` is intended to be pasted into an mlx-vlm GitHub issue and
  must let maintainers identify and reproduce integration failures quickly.

The current gallery is useful, but its chooser flattens output previews and its
complete evidence shows model-authored Markdown only as literal code. The
current diagnostics report is 3,552 lines and 203,618 bytes for a 62-model run.
It repeats a full CLI reproduction and Python script for every highlighted
model: those two repeated sections account for about 134 KB. The complete model
outputs themselves account for only about 39 KB.

The run contains one actionable crash, 29 completed models with recorded
observations, and 32 clean completions. The report should preserve complete
evidence for the first two groups while representing the clean group as compact
context.

## Goals

1. Make `diagnostics.md` self-contained, paste-ready, and practical to skim as
   an mlx-vlm issue report.
2. Retain complete evidence for every crash, indeterminate attempt, and
   completed run with recorded observations.
3. Retain summary runtime, provenance, and performance context for clean
   completions without repeating their output.
4. Show model output readably in the gallery while retaining an exact raw copy.
5. Reuse the cached assessment and existing report-block infrastructure without
   adding classification or scoring heuristics.
6. Simplify touched reporting paths, remove superseded code, and keep every new
   interface fully and narrowly typed. Record source growth honestly when later
   approved evidence contracts expand the original scope.

## Non-goals

- Do not alter model execution, prompts, generation settings, classifications,
  observation detection, or JSONL evidence.
- Do not select detailed evidence using model names, image-specific keywords,
  scores, or a new severity heuristic.
- Do not truncate complete highlighted output or crash evidence to meet a fixed
  report-size target.
- Do not hand-edit or reconstruct the currently retained run artefacts.
- Do not add a Python Markdown-rendering dependency solely for report display.

## Audience and evidence rule

The existing cached `ResultAssessment` remains authoritative. Detail selection
is a direct presentation rule:

- `actionable_failure`: complete evidence, expanded;
- `observation_needs_reproduction`: complete evidence, collapsed;
- `indeterminate` execution: complete evidence, collapsed;
- completed with maintainer status `none`: summary context only.

This preserves the existing separation between model-user usability and
maintainer triage. It does not reclassify prompt noncompliance, token caps,
thinking traces, repetition, or special tokens.

## Diagnostics report contract

`diagnostics.md` uses the following order.

### 1. Run summary

Show attempted, evaluated, completed, crashed, and indeterminate counts. Also
show compact counts for maintainer status, usability, and recorded observation
codes so the issue can be understood before opening evidence.

### 2. Triage table

List every actionable, observed, or indeterminate model with execution,
usability, observations, and an internal evidence link. This is the primary
skim surface.

### 3. Actionable failures

Render each crash expanded. Evidence order is:

1. root exception chain;
2. complete traceback;
3. concise execution, model, processor, tokenizer, stop, and token-wrapper
   provenance;
4. complete partial output when present;
5. captured stdout/stderr when present.

Do not emit placeholder output or stream sections whose evidence is unavailable.

### 4. Completed runs with observations

Render one collapsed `<details>` entry per model. Its summary names the model,
usability, and observation labels. Inside, retain concise diagnostic facts and
the complete model output. Do not repeat the prompt or reproduction script.

### 5. Indeterminate attempts

Render one collapsed entry per attempt with the captured connectivity or other
indeterminate evidence. Keep the distinction between an indeterminate attempt
and a model crash.

### 6. Clean completion context

Render one collapsed three-column table:

- **Model**;
- **Runtime identity**: short resolved revision, processor class, and stop
  reason;
- **Performance**: prompt/generated tokens, valid generation throughput, and
  peak memory.

This table supplies breadth and comparison context without reproducing clean
model output. Full clean output remains in the gallery and JSONL.

### 7. Shared reproduction and provenance

Emit the following once per report:

- the exact run prompt;
- common generation, template, and load settings;
- a model/revision table for highlighted entries;
- one parameterised native mlx-vlm Python reproduction script that accepts a
  model ID and revision and runs one model per invocation;
- component versions and relevant Apple Silicon/system facts.

The existing per-crash issue drafts remain complete and may retain their direct
single-model reproduction. Only the aggregated diagnostics report removes
per-entry reproduction duplication.

## Gallery output contract

The gallery remains the model-selection and output-comparison report.

### Chooser previews

Preserve source newlines within the existing bounded preview using `<br>` in
table cells. Continue escaping pipes and HTML so output cannot corrupt the
table. Truncation remains presentation-only; the complete output is unaffected.

### Complete model entries

For completed output, show two views derived from the same captured string:

1. **Readable output**: inert escaped `<pre>` presentation. Preserve source line
   breaks and spacing without allowing model-authored headings, lists, HTML, or
   mentions to affect the surrounding report.
2. **Exact raw output**: the unmodified captured text once, inside a collapsed,
   dynamically sized fenced code block.

The readable view is explicitly presentation-only. The raw block remains the
canonical evidence and preserves nested fences, tabs, and trailing whitespace.
Empty output and unavailable evidence remain distinct.

Crash entries continue to prioritise the traceback, partial output, and captured
upstream streams. The standalone HTML report mirrors the new information
hierarchy using escaped preformatted model text; it does not introduce a
separate Markdown parser.

## Implementation structure

Use the existing report block types as the common representation for Markdown
and HTML. Tighten the recursive report-block annotations from `object` to a
narrow `ReportBlock` union while touching this code.

Build the diagnostic partitions and their display blocks once from
`ReportRenderContext`. Both Markdown and HTML consume that structure. Delete the
parallel format-specific diagnostic entry and partition assembly where the
shared builders make it redundant.

Use one small output-presentation helper for gallery and diagnostic evidence.
Keep format-specific escaping at the rendering boundary:

- Markdown readable output: escaped inert `<pre>` content;
- Markdown raw output: existing safe dynamic fence;
- HTML output: existing escaped `<pre><code>` rendering.

Extend the existing native mlx-vlm reproduction helpers only enough to produce
the shared parameterised script. Do not create a second argument-construction
path.

The refactor must remain inside the intentional `check_models.py` monolith. It
must remove superseded paths, avoid duplicate abstractions, and retain full,
narrow parameter and return annotations. Line count is a review signal rather
than an acceptance proxy once approved follow-ups add new evidence contracts.

## Failure and safety behaviour

- Report generation continues to use the cached assessment and provenance
  invariants; renderers never call classification again.
- Arbitrary model HTML cannot alter report structure.
- Model-authored mentions cannot notify GitHub users from the readable view.
- Exact raw evidence remains inert inside a code fence.
- Missing optional metrics render as an explicit short marker in tables.
- Missing output or captured streams do not produce large `unavailable` code
  blocks.
- A report-rendering failure remains isolated from canonical JSONL evidence.

## Tests

Add or update focused tests in the existing report and Markdown test modules.
They must prove that:

- chooser previews retain newlines as `<br>` and remain table-safe;
- readable output preserves headings, paragraphs, lists, emphasis, and line
  breaks;
- HTML and GitHub mentions are neutralised in readable output;
- exact raw output appears once and is byte-for-byte unchanged, including nested
  fences and trailing whitespace;
- actionable crashes are expanded and traceback-first;
- observation and indeterminate evidence is collapsed with informative
  summaries;
- clean completions appear only in the compact runtime/performance table;
- every highlighted result retains complete output or failure evidence;
- the prompt and parameterised reproduction script appear exactly once;
- Markdown and HTML consume the same cached assessments and partitions;
- generated representative reports pass Markdown lint.

All generated test artefacts go to temporary or ignored paths. Tests must not
rewrite tracked `src/output/` files.

## Documentation and retained artefacts

Update the user and implementation documentation to state the three output
roles:

- gallery: model selection and complete human-readable output comparison;
- diagnostics: self-contained, issue-ready maintainer triage;
- JSONL and logs: exhaustive machine-readable and raw operational evidence.

Record the change under `[Unreleased]` in `CHANGELOG.md`.

The current tracked reports remain faithful to their completed run. The next
genuine model run will generate reports under this contract; implementation
verification uses temporary fixture reports rather than hand-rewriting retained
evidence.

## Acceptance criteria

The change is complete when:

1. all agreed gallery and diagnostics contracts above have focused tests;
2. representative generated Markdown passes repo-local Markdown lint;
3. full `make quality` passes;
4. Skylos audit and danger scans remain clean without new suppressions;
5. a recorded `wc -l` and diff check account for source growth, no superseded
   duplicate path remains, and the report-block and evidence types are narrow;
6. tracked current-run outputs have not been modified by validation; and
7. documentation and `[Unreleased]` changelog entries describe the new roles.

## Approved follow-up: classification and provenance hardening

The post-run review extends this design without adding semantic scoring:

- strict catalog prompts apply even to short non-empty output;
- copied instruction spans, multi-item title fields, and duplicate-dominated
  keyword lists are mechanically unusable;
- authoritative context values are excluded from instruction-echo spans,
  conventional bold Markdown labels remain valid, and unexpected catalog preamble
  text is mechanically unusable;
- configured role tokens are distinguished from unknown-token leakage, while a
  role boundary inside output is retained as factual evidence;
- observation details retain exact section names, fragments, and tokens;
- diagnostics include each highlighted completed output once and omit empty fact
  rows; the gallery keeps the readable plus exact views;
- Markdown galleries publish a bounded reference preview and lead with end-to-end
  time;
- run provenance records producer dirtiness and per-model completion timestamps;
- human counts call completed-plus-crashed results “conclusive outcomes,” while the
  stable schema `2.0` field name remains unchanged for compatibility.
