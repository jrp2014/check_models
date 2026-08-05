# Output Report Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct reasoning assessments and make retained Markdown artifacts accurate, bounded and skimmable without rerunning models.

**Architecture:** Extend the existing assessment and typed report-block paths in
`src/check_models.py`; do not add a parallel report builder. Keep exact evidence in
JSONL, diagnostics and dedicated crash drafts while making aggregate views
publication-safe and compact. Derive optional artifact freshness only from retained
run timing evidence.

**Tech Stack:** Python 3.13, pytest, existing report blocks, `datetime`, Rich/tabulate

## Global Constraints

- Activate the `mlx-vlm` conda environment before Python or Make commands.
- Keep `src/check_models.py` as the intentional monolith.
- Prefer a net-subtractive implementation: remove redundant preview work and
  consolidate through existing typed report helpers rather than adding parallel
  builders.
- Tests write only to `tmp_path` or gitignored `test_*` output locations.
- Do not rerun model inference or append `results.history.jsonl`.
- Preserve full raw evidence outside aggregate summaries.

---

### Task 1: Correct Prompt-seeded Reasoning Classification

**Files:**

- Modify: `src/tests/test_quality_analysis.py`
- Modify: `src/check_models.py`

**Interfaces:**

- Consumes: `_detect_reasoning_output(text, delimiter_pairs, seeded_text)`
- Produces: unchanged `ReasoningOutputSignals`; corrected seeded-pair semantics

- [ ] **Step 1: Add failing tests**

Add cases proving that `<think></think>` in `seeded_text` with an ordinary generated
answer is neutral, while a seeded open-only block remains incomplete unless the
generated text supplies its closing delimiter.

- [ ] **Step 2: Verify red**

Run:
`pytest src/tests/test_quality_analysis.py -k 'seeded_thinking' -q`

Expected: the complete seeded-wrapper test fails with
`thinking_trace_incomplete`.

- [ ] **Step 3: Implement the minimal fix**

For each configured delimiter pair, distinguish generated start position from a
seeded start. Ignore a pair fully closed in seeded text; allow a seeded open-only
pair to find its close in generated text; require generated starts to close in
generated text.

- [ ] **Step 4: Verify green**

Run the focused quality-analysis tests and confirm the genuine generated unclosed
trace remains unusable.

### Task 2: Withhold Positively Stale Log Evidence

**Files:**

- Modify: `src/tests/test_report_generation.py`
- Modify: `src/check_models.py`

**Interfaces:**

- Extend: `RunIssueSummarySource` with optional retained run-window facts
- Extend: `_load_run_issue_enrichment()` to validate `generated_at` and
  `total_runtime_seconds`
- Produce: a stale-artifact set consumed by the issue artifact table

- [ ] **Step 1: Add failing report tests**

Build temporary run JSON, JSONL, log and environment fixtures. Assert that files
whose first timestamp is outside the derived run window are omitted from the full
artifact links and reported as stale, while matching files remain linked. Assert
that missing timing evidence preserves legacy behaviour.

- [ ] **Step 2: Verify red**

Run the new test nodes and confirm stale files are still linked.

- [ ] **Step 3: Implement bounded freshness detection**

Parse only the established local timestamp format. Derive run start from
`generated_at - total_runtime_seconds`, allow a small clock tolerance, and classify
only timestamps clearly outside the start/end window. Do not change or delete the
artifact files.

- [ ] **Step 4: Verify green**

Run the focused summary tests and malformed-run-JSON compatibility tests.

### Task 3: Compact Aggregate Crash Messages

**Files:**

- Modify: `src/tests/test_report_generation.py`
- Modify: `src/check_models.py`

**Interfaces:**

- Extend: `_run_issue_summary_exception_lines(failure)`
- Preserve: `_diagnostics_exception_chain()` and dedicated crash draft evidence

- [ ] **Step 1: Add a failing long-parameter test**

Use a synthetic `Received 12 parameters not in model` exception containing audio
tower and repeated language-model layer paths. Assert the aggregate summary keeps
the count and representative families, excludes most repeated paths, and links to
full evidence.

- [ ] **Step 2: Verify red**

Run the new node and confirm the aggregate currently includes every path.

- [ ] **Step 3: Implement message compaction**

Recognise only the established parameter-mismatch shape. Render the exception type,
declared/observed count, grouped leading components, and a small representative
sample. Leave all other messages unchanged.

- [ ] **Step 4: Verify green and exact-evidence retention**

Run the aggregate and diagnostics tests; verify full exception text remains in the
dedicated diagnostics/draft path.

### Task 4: Narrow Markdown Chooser Tables

**Files:**

- Modify: `src/tests/test_report_generation.py`
- Modify: `src/check_models.py`

**Interfaces:**

- Modify: `_render_gallery_chooser(rows)` Markdown columns only
- Preserve: HTML chooser and complete per-model evidence

- [ ] **Step 1: Change chooser expectations first**

Assert `Output preview` and preview text are absent from both Markdown chooser
tables, while model links, usability, observations, resource facts and complete
model evidence remain.

- [ ] **Step 2: Verify red**

Run the focused Markdown report tests and confirm the preview assertions fail.

- [ ] **Step 3: Remove redundant preview cells**

Delete only the preview columns/cells from current-run and avoid Markdown tables.
Do not change `GalleryRow`, HTML rendering or complete evidence sections.

- [ ] **Step 4: Verify green**

Run Markdown/report tests and markdownlint against temporary generated reports.

### Task 5: Documentation, Full Verification and Replay

**Files:**

- Modify: `CHANGELOG.md`
- Modify intentionally through report-only replay:
  `src/output/issues/run_summary.md`, `src/output/reports/diagnostics.md`,
  `src/output/reports/model_gallery.md`, `src/output/reports/results.html`,
  `src/output/results.jsonl`, `src/output/run.json`, `src/output/index.md`

- [ ] **Step 1: Update the Unreleased changelog**

Describe corrected seeded reasoning, stale-evidence withholding, aggregate crash
compaction and narrower Markdown choosers.

- [ ] **Step 2: Run prescribed static/full gates**

Run format, safe lint fixes, lint and `make quality`. Confirm validation does not
rewrite tracked retained outputs; restore only test-created side effects proven to
come from validation.

- [ ] **Step 3: Replay existing evidence**

Use the supported report-only regeneration path with the current schema-2.0 JSONL
and run JSON. Do not rerun inference and do not append history.

- [ ] **Step 4: Audit replayed artifacts**

Confirm 13 false thinking observations disappear, the genuine Kimi trace remains,
stale log/environment links are withheld, the crash summary is bounded, Markdown
chooser width is reduced, canonical GitHub links remain, and JSONL/history evidence
is not lost.

- [ ] **Step 5: Archive completed design and plan**

Move both documents under `docs/notes/archive/superpowers/` after verification.

- [ ] **Step 6: Commit and push**

Stage code, tests, changelog, archived planning documents, and intentionally
regenerated current-run artifacts only. Commit with a conventional message and
push `codex/fix-output-report-integrity`.
