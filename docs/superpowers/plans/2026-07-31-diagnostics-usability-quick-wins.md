# Diagnostics and Gallery Quick Wins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make selector reports usable-first, put actionable crash facts above optional issue traceback detail, link generated crash drafts from the output index, and advertise the existing triage caption lane.

**Architecture:** Keep all runtime and assessment behaviour unchanged. Add one shared gallery usability sort key, add a presentation flag to the existing diagnostic model-block builder, pass the already-generated issue-report mapping to the index renderer, and update human documentation.

**Tech Stack:** Python 3.13, pytest, Rich-backed report primitives, Markdown/HTML renderers, markdownlint, Ruff, mypy, ty, pyrefly.

## Global Constraints

- Activate the `mlx-vlm` Conda environment before every Python, pytest, or make command.
- Keep implementation inside the intentional `src/check_models.py` monolith.
- Reuse cached `ResultAssessment`; renderers must not reclassify results.
- Preserve complete crash evidence, JSONL schema `2.0`, and the `run.json` artifact contract.
- Add tests only to existing `src/tests/test_*.py` files.
- Route generated test artifacts to `tmp_path`; do not rewrite tracked `src/output/` files.
- Update `CHANGELOG.md` under `[Unreleased]` for the shipped report behaviour.

---

### Task 1: Use one usable-first gallery ordering policy

**Files:**

- Modify: `src/tests/test_report_generation.py`
- Modify: `src/check_models.py:6954-7002`
- Modify: `src/check_models.py:8563-8577`
- Modify: `src/check_models.py:9045-9065`

**Interfaces:**

- Consumes: `GalleryRow.usability: ModelUsability` and cached `ResultAssessment` values.
- Produces: `_gallery_usability_sort_key(usability: ModelUsability) -> int`, shared by Markdown and HTML renderers.

- [ ] **Step 1: Write failing ordering assertions**

Extend `test_gallery_uses_skim_first_chooser_order_and_cached_assessments` to extract the chooser and assert literal model order:

```python
assert chooser.index("org/usable") < chooser.index("org/unusable")
assert chooser.index("org/unusable") < chooser.index("org/not-evaluated")
assert content.index("### org/usable") < content.index("### org/unusable")
assert content.index("### org/unusable") < content.index("### org/not-evaluated")
```

Add an HTML report assertion using the same three cached assessments and assert the first occurrences within `id="chooser-table"` are usable, unusable, then not evaluated.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_report_generation.py -k "skim_first_chooser_order or html_chooser_uses_usable_first" -vv --maxfail=1
```

Expected: failure because current renderer order is unusable, not evaluated, usable.

- [ ] **Step 3: Implement the shared sort key**

Add:

```python
def _gallery_usability_sort_key(usability: ModelUsability) -> int:
    return {
        "usable": 0,
        "usable_with_caveats": 1,
        "unusable": 2,
        "not_evaluated": 3,
    }[usability]
```

Replace all three local `usability_order` mappings with this function.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit the independently testable ordering change**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "feat: show usable models first in galleries"
```

### Task 2: Keep crash facts above collapsed issue traceback evidence

**Files:**

- Modify: `src/tests/test_report_generation.py`
- Modify: `src/check_models.py:8097-8145`
- Modify: `src/check_models.py:15230-15280`

**Interfaces:**

- Consumes: `_diagnostics_model_blocks(...)` existing crash evidence.
- Produces: `_diagnostics_model_blocks(..., collapse_traceback: bool = False) -> tuple[ReportBlock, ...]`.

- [ ] **Step 1: Write a failing direct-issue presentation test**

Extend `test_crash_diagnostics_and_issue_draft_keep_complete_primary_evidence_first` with literal assertions that:

```python
assert issue_content.index("#### Root exception and chain") < issue_content.index("#### Execution and provenance")
assert issue_content.index("#### Execution and provenance") < issue_content.index("Complete traceback")
assert "<summary>Complete traceback</summary>" in issue_content
assert traceback_text in issue_content
assert "<summary>Complete traceback</summary>" not in diagnostics_content
```

This catches accidental loss of exact evidence and accidental collapse of the full workbook.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_report_generation.py::test_crash_diagnostics_and_issue_draft_keep_complete_primary_evidence_first -vv
```

Expected: failure because execution facts currently follow the expanded traceback.

- [ ] **Step 3: Implement the presentation option**

Build root exception, execution/provenance, then traceback blocks in that order. When `collapse_traceback=True`, wrap the exact `ReportCodeBlock` in `ReportDetails("Complete traceback", ...)`; otherwise retain the expanded `ReportSection`.

Pass `collapse_traceback=True` only from `_generate_github_issue_reports`. Leave aggregate diagnostics and HTML diagnostics on the default.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2. Expected: pass with the complete traceback still present byte-for-byte.

- [ ] **Step 5: Commit the independently testable issue presentation change**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "feat: keep crash facts above issue traceback detail"
```

### Task 3: Link current-run crash drafts from the output index

**Files:**

- Modify: `src/tests/test_report_generation.py`
- Modify: `src/check_models.py:14605-14622`
- Modify: `src/check_models.py:15450-15530`

**Interfaces:**

- Consumes: `DiagnosticsArtifacts.issue_reports: Mapping[str, Path]`.
- Produces: `generate_output_index_report(..., issue_reports: Mapping[str, Path] | None = None) -> None`.

- [ ] **Step 1: Write failing conditional-link tests**

Keep `test_output_index_links_only_current_run_artifacts` unchanged to cover the no-draft case. Add a test that passes two deliberately reverse-ordered issue paths and asserts:

```python
assert "## Issue drafts" in content
assert "[org/a](issues/issue_org_a.md)" in content
assert "[org/z](issues/issue_org_z.md)" in content
assert content.index("[org/a]") < content.index("[org/z]")
```

Extend the existing link-style artifact test to assert issue links use GitHub targets in `github` mode and relative targets in `relative` mode.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_report_generation.py -k "output_index or selected_output_link_style" -vv --maxfail=1
```

Expected: the new test errors or fails because the index renderer does not accept or render issue reports.

- [ ] **Step 3: Implement conditional issue links and orchestration**

Add the optional mapping parameter, append a heading and sorted model-labelled links only when the mapping is non-empty, and invoke the output-index renderer after diagnostics with `diagnostics_artifacts.issue_reports`. Preserve the existing seven-link output when omitted or empty.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit the independently testable navigation change**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "feat: link crash drafts from output index"
```

### Task 4: Document the triage caption lane and shipped behaviour

**Files:**

- Modify: `src/README.md:990-1019`
- Modify: `CHANGELOG.md` under `[Unreleased]`

**Interfaces:**

- Consumes: existing `--eval-mode triage` CLI behaviour.
- Produces: human-facing guidance only; no runtime interface changes.

- [ ] **Step 1: Update README guidance**

Change the triage intended-use wording to identify plain-caption comparison and add this example before the blind example:

```bash
# Compare models as plain image captioners
python -m check_models --image photo.jpg --eval-mode triage
```

- [ ] **Step 2: Update the changelog**

Add an `[Unreleased]` entry covering usable-first gallery ordering, above-fold issue facts with collapsible traceback detail, output-index crash-draft links, and triage-lane documentation.

- [ ] **Step 3: Run Markdown validation**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && npx --prefix src markdownlint-cli2 src/README.md CHANGELOG.md docs/superpowers/specs/2026-07-31-diagnostics-usability-quick-wins-design.md docs/superpowers/plans/2026-07-31-diagnostics-usability-quick-wins.md
```

Expected: zero Markdown errors.

- [ ] **Step 4: Commit documentation and the implementation plan**

```bash
git add src/README.md CHANGELOG.md docs/superpowers/plans/2026-07-31-diagnostics-usability-quick-wins.md
git commit -m "docs: clarify caption selection workflow"
```

### Task 5: Run repository verification

**Files:**

- Verify only; production edits are allowed only if a failing gate identifies a defect and a new focused regression test is added first.

**Interfaces:**

- Consumes: all preceding tasks.
- Produces: fresh verification evidence for handoff.

- [ ] **Step 1: Validate the active environment**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && cd src && python -m tools.validate_env
```

- [ ] **Step 2: Run focused report tests**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_report_generation.py src/tests/test_metrics_modes.py -q
```

- [ ] **Step 3: Run prescribed formatting and lint sequence**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make format && make -C src lint-fix && make lint
```

- [ ] **Step 4: Run commit hygiene**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && bash src/tools/run_commit_hygiene.sh
```

- [ ] **Step 5: Run the full quality gate**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make quality
```

- [ ] **Step 6: Inspect final repository state**

```bash
git status --short
git diff --check
git log --oneline -5
```

Expected: clean worktree, no whitespace errors, and the design plus four implementation commits on `codex/diagnostics-usability-quick-wins`.
