# Completed-Model Summary Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ambiguous pipe-delimited completed-model bullets with actionability-ordered Rich tables in the terminal and persistent log.

**Architecture:** Keep `log_summary()` and its cached-assessment input unchanged. Refactor only `_log_completed_models_list()` to partition completed results by usability and render each non-empty group through the existing `_log_rich_table()` boundary; reuse `_gallery_observation_labels()` for severity-ordered human labels.

**Tech Stack:** Python 3.13, Rich, pytest, Ruff

## Global Constraints

- Keep `src/check_models.py` as a single-file monolith.
- Preserve cached assessments; do not call `_assess_result()` from the completed-model renderer.
- Render `unusable`, then `usable_with_caveats`, then `usable`.
- Omit empty groups and preserve alphabetic model ordering within each group.
- Do not modify files under `src/output/`.

---

### Task 1: Render grouped completed-model tables

**Files:**

- Modify: `src/tests/test_metrics_modes.py`
- Modify: `src/check_models.py` (`_log_completed_models_list`)
- Modify: `CHANGELOG.md` (`[Unreleased]`)

**Interfaces:**

- Consumes: `_log_rich_table(...)`, `_gallery_observation_labels(...)`, and the existing `Mapping[str, ResultAssessment]` supplied by `log_summary()`.
- Produces: the unchanged `_log_completed_models_list(completed, assessments) -> None` interface with grouped Rich-table output.

- [ ] **Step 1: Write the failing behavioural test**

Add a test that calls the real `_log_completed_models_list()` with deliberately
unsorted completed results covering all three usability values. Supply reversed
observation codes `("missing_requested_sections", "repeated_output")` and assert
the rendered log contains:

```python
assert messages.index("Unusable (1)") < messages.index("Usable with caveats (2)")
assert messages.index("Usable with caveats (2)") < messages.index("Usable (1)")
assert messages.index("Response repeats the same text") < messages.index(
    "Required fields are missing or empty"
)
assert messages.index("org/a-caveat") < messages.index("org/z-caveat")
assert "| usability=" not in messages
```

Slice the text from `Usable (1)` onward and assert that clean-group rendering
does not contain `Maintainer` or `Observations`. Patch `get_terminal_width` to a
stable width only if needed to make the real Rich rendering deterministic.

- [ ] **Step 2: Run the test and verify the old bullets fail it**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_metrics_modes.py::test_completed_model_summary_uses_actionability_ordered_tables -q
```

Expected: FAIL because the output has no usability-group headings and still
contains `| usability=` bullets.

- [ ] **Step 3: Implement the minimal grouped renderer**

Inside `_log_completed_models_list()`:

```python
groups = (
    ("unusable", "Unusable"),
    ("usable_with_caveats", "Usable with caveats"),
    ("usable", "Usable"),
)
```

For each group, alphabetically sort matching results and skip an empty group.
Log `<Label> (<count>):`. For `usable`, call `_log_rich_table()` with only the
`Model` column. For the two actionable groups, render `Model`, `Maintainer`, and
`Observations`; replace underscores in the maintainer value with spaces and use
`_gallery_observation_labels(assessment.observations)` for the last column.

- [ ] **Step 4: Run focused tests and verify green**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_metrics_modes.py -q
```

Expected: all metrics-mode tests pass.

- [ ] **Step 5: Document and verify the complete change**

Add an `[Unreleased]` changelog bullet describing the grouped terminal/log
tables. Then run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make format
make -C src lint-fix
make lint
bash src/tools/run_commit_hygiene.sh
make quality
```

Expected: all formatting, lint, type, static-analysis, pytest, shellcheck, and
Markdown checks pass without changes under `src/output/`.

- [ ] **Step 6: Commit and push only implementation files**

```bash
git add CHANGELOG.md src/check_models.py src/tests/test_metrics_modes.py
git commit -m "feat: tabulate completed model summary"
git push origin main
```

---

### Task 2: Separate replayed per-model result blocks

**Files:**

- Modify: `src/tests/test_metrics_modes.py`
- Modify: `src/check_models.py` (`finalize_execution`)
- Modify: `CHANGELOG.md` (`[Unreleased]`)

- [x] Add a behavioural finalization test requiring one horizontal rule after
  model 1's metrics and before model 2's summary.
- [x] Insert the existing `print_cli_separator()` only between consecutive
  per-model blocks.
- [x] Run focused and full quality gates.
