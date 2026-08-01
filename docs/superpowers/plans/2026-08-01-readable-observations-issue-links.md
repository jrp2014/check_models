# Readable Observations and Paste-Safe Issue Links Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept conventional Markdown catalogue headings and make future issue-ready reports readable, severity-ordered, execution-grouped, and safe to paste into GitHub.

**Architecture:** Keep stable assessment codes and usability decisions unchanged, but add one shared human observation presentation layer in the existing `src/check_models.py` monolith. Give issue-ready cross-file links an explicit GitHub policy rather than mutating the global link mode, and render non-actionable surfaced results as one table per execution status.

**Tech Stack:** Python 3.13, compiled regular expressions, typed dataclasses and `TypedDict` records, pytest, Ruff, mypy, ty, Pyrefly, Skylos, markdownlint.

## Global Constraints

- Always activate the `mlx-vlm` Conda environment before Python or Make commands.
- Keep `src/check_models.py` as the intentional single-file monolith.
- Add tests only to existing `src/tests/test_*.py` files.
- Do not rewrite any retained `src/output/` artifact; the maintainer will rerun the model matrix.
- Stable JSONL observation codes, usability rules, and schema version remain unchanged.
- `--link-style relative` remains effective for local-navigation artifacts.
- Issue-ready cross-file links must target `https://github.com/jrp2014/check_models/blob/main/src/output/`.
- Same-document diagnostic anchors remain local.
- Update `CHANGELOG.md` under `[Unreleased]` and `src/README.md` for the user-visible behavior.

---

### Task 1: Accept standard Markdown catalogue headings

**Files:**

- Modify: `src/tests/test_quality_analysis.py` near the catalogue-contract tests
- Modify: `src/check_models.py` at `CATALOG_SECTION_PATTERN`

**Interfaces:**

- Consumes: `analyze_generation_text(...)`, `_extract_catalog_sections(text: str) -> dict[str, str]`, and `_assess_result(result: PerformanceResult) -> ResultAssessment`
- Produces: `CATALOG_SECTION_PATTERN` support for one-to-six-hash headings while preserving the existing field contract

- [ ] **Step 1: Add the failing Pixtral-format regression test**

Add this test beside `test_markdown_bold_catalog_labels_satisfy_requested_sections`:

```python
@pytest.mark.parametrize("heading", ["#", "###", "######"])
def test_markdown_heading_catalog_labels_satisfy_requested_sections(heading: str) -> None:
    result = _result(
        f"{heading} Title:\nTwo Cats Lounging on Red Couch\n\n"
        f"{heading} Description:\nTwo cats relax together on a red couch.\n\n"
        f"{heading} Keywords:\n"
        "cats, lounging, red couch, remote controls, relaxed, indoor, comfort, "
        "feline, domestic, resting",
        prompt=CATALOG_PROMPT,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.missing_sections == []
    assert check_models._assess_result(result).usability == "usable"
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_quality_analysis.py::test_markdown_heading_catalog_labels_satisfy_requested_sections -q
```

Expected: the `###` and `######` cases fail because all three fields are reported missing.

- [ ] **Step 3: Make the smallest parser change**

Replace the single-character Markdown prefix in `CATALOG_SECTION_PATTERN` with a standard heading-or-marker prefix while retaining the required colon:

```python
CATALOG_SECTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?im)^[ \t]*(?:(?:#{1,6})[ \t]+|[>*-][ \t]*)?"
    r"\*{0,2}[ \t]*(title|description|keywords)"
    r"(?::[ \t]*\*{0,2}|\*{0,2}[ \t]*:)[ \t]*(.*)$",
)
```

Do not accept seven hashes, omit the colon requirement, or weaken the existing non-empty and single-line-title checks.

- [ ] **Step 4: Run focused catalogue tests and verify GREEN**

Run:

```bash
pytest src/tests/test_quality_analysis.py -k "catalog or markdown_heading or markdown_bold" -q
```

Expected: all selected tests pass, including the exact multi-hash format.

- [ ] **Step 5: Commit the parser fix**

```bash
git add src/check_models.py src/tests/test_quality_analysis.py
git commit -m "fix: accept markdown catalogue headings"
```

---

### Task 2: Translate and severity-order human observations

**Files:**

- Modify: `src/tests/test_report_generation.py` near gallery and run-summary assertions
- Modify: `src/check_models.py` near `_gallery_observation_labels`

**Interfaces:**

- Consumes: `ObservationCode`, `JsonlObservationDetailsRecord`, and existing stable observation tuples
- Produces: `_human_observation_labels(observations: Sequence[ObservationCode], *, details: JsonlObservationDetailsRecord | None = None) -> str`
- Produces: `_gallery_observation_labels(observations: Sequence[ObservationCode]) -> str` as a compatibility wrapper for existing renderers

- [ ] **Step 1: Add failing tests for readable labels and severity order**

Add focused assertions that deliberately supply observations in reverse/non-severity order:

```python
def test_human_observation_labels_are_readable_and_severity_ordered() -> None:
    labels = check_models._human_observation_labels(
        (
            "no_keyword_overlap",
            "missing_requested_sections",
            "unexpected_special_token",
            "repeated_output",
        ),
        details={"missing_sections": ["title", "keywords"]},
    )

    assert labels == (
        "Response repeats the same text; "
        "Unrecognised model control tokens remain visible; "
        "Missing or empty fields: Title, Keywords; "
        "Keywords do not overlap the supplied keyword hints"
    )
```

Add a second test covering every `ObservationCode` and assert that no rendered label contains an underscore or the unexplained phrase `catalogue instructions`.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest src/tests/test_report_generation.py -k "human_observation_labels" -q
```

Expected: failure because `_human_observation_labels` does not exist and current labels are raw underscored codes.

- [ ] **Step 3: Add the shared presentation mapping and priority**

Define typed constants beside `_gallery_observation_labels`:

```python
_OBSERVATION_DISPLAY_PRIORITY: Final[tuple[ObservationCode, ...]] = (
    "empty_output",
    "repeated_output",
    "unexpected_special_token",
    "missing_requested_sections",
    "prompt_instruction_echo",
    "unexpected_catalog_preamble",
    "token_cap_truncation",
    "thinking_trace_incomplete",
    "role_boundary_token_present",
    "thinking_trace_present",
    "configured_wrapper_present",
    "minimal_output",
    "draft_returned_unchanged",
    "no_keyword_overlap",
)

_OBSERVATION_DISPLAY_LABELS: Final[dict[ObservationCode, str]] = {
    "empty_output": "No response text was returned",
    "minimal_output": "Response is unusually short",
    "repeated_output": "Response repeats the same text",
    "missing_requested_sections": "Required fields are missing or empty",
    "token_cap_truncation": "Response appears cut off at the token limit",
    "prompt_instruction_echo": (
        "Response repeats the task instructions instead of only returning the requested fields"
    ),
    "unexpected_catalog_preamble": "Extra text appears before the Title field",
    "unexpected_special_token": "Unrecognised model control tokens remain visible",
    "configured_wrapper_present": "Expected model wrapper tokens remain visible",
    "thinking_trace_present": "Internal reasoning text remains visible",
    "thinking_trace_incomplete": "Internal reasoning block appears incomplete",
    "role_boundary_token_present": "Conversation-role control tokens remain visible",
    "no_keyword_overlap": "Keywords do not overlap the supplied keyword hints",
    "draft_returned_unchanged": (
        "Title, Description and Keywords copy all supplied hints unchanged"
    ),
}
```

Implement `_human_observation_labels` by deduplicating supplied codes, iterating `_OBSERVATION_DISPLAY_PRIORITY`, and joining labels with `;`. When `details` includes `missing_sections`, replace the generic missing-field label with:

```python
f"Missing or empty fields: {', '.join(field.title() for field in missing_sections)}"
```

Keep `_gallery_observation_labels` as a thin call to `_human_observation_labels(observations)` so every human renderer receives the new language without changing machine records.

- [ ] **Step 4: Run report and assessment tests and verify GREEN**

Run:

```bash
pytest src/tests/test_report_generation.py src/tests/test_quality_analysis.py -k "observation or gallery or assessment" -q
```

Expected: readable-label tests pass; existing tests that assert raw labels are updated only where the visible wording intentionally changed.

- [ ] **Step 5: Commit the presentation layer**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "feat: make report observations reader-friendly"
```

---

### Task 3: Group aggregate results and force paste-safe issue links

**Files:**

- Modify: `src/tests/test_report_generation.py` in run-summary and link-style tests
- Modify: `src/check_models.py` near `_markdown_artifact_target`, `_run_issue_summary_artifact_link`, and `_run_issue_summary_other_section`

**Interfaces:**

- Consumes: `_github_output_artifact_url(...)`, `JsonlResultRecord`, `_human_observation_labels(...)`, and `ReportSection`
- Produces: `_issue_markdown_artifact_target(*, report_filename: Path, artifact_filename: Path, anchor: str | None = None) -> str`
- Produces: `_run_issue_summary_surfaced_sections(results: Sequence[JsonlResultRecord], *, output_paths: ReportOutputPaths, summary_path: Path) -> tuple[ReportSection, ...]`

- [ ] **Step 1: Add failing tests for GitHub links under relative mode**

Extend the run-summary fixture test so it renders with:

```python
with patch.object(check_models._LinkStyleState, "value", "relative"):
    summary = check_models.generate_run_issue_summary_report(output_paths)
```

Assert every cross-file target in the resulting issue summary begins with:

```text
https://github.com/jrp2014/check_models/blob/main/src/output/
```

Also retain an output-index test under relative mode and assert its links remain relative.

- [ ] **Step 2: Add a failing execution-group layout test**

Use literal schema-2.0 fixture rows for:

- one actionable crash, which must appear only in expanded crash evidence;
- one completed observed result;
- one non-actionable crashed retained result;
- one indeterminate result.

Assert these headings and columns:

```python
assert "## Crashes requiring action" in content
assert "## Completed attempts requiring review" in content
assert "## Crashed attempts requiring review" in content
assert "## Indeterminate attempts requiring review" in content
assert "| Model | Usability | Observed result | Evidence |" in content
assert "Execution / usability" not in content
```

Assert the actionable crash model occurs in its expanded heading but not in the crashed-results table.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
pytest src/tests/test_report_generation.py -k "run_issue_summary and (relative or execution or expands)" -q
```

Expected: relative-mode issue links are local and the report still emits one combined `Other surfaced results` table.

- [ ] **Step 4: Add an explicit issue-link helper**

Implement `_issue_markdown_artifact_target` without reading `_LinkStyleState.value`:

```python
def _issue_markdown_artifact_target(
    *,
    report_filename: Path,
    artifact_filename: Path,
    anchor: str | None = None,
) -> str:
    """Return a repository URL for a cross-file link pasted into a GitHub issue."""
    github_url = _github_output_artifact_url(artifact_filename, anchor=anchor)
    if github_url is not None:
        return github_url
    return _markdown_artifact_target(
        report_filename=report_filename,
        artifact_filename=artifact_filename,
        anchor=anchor,
    )
```

Change `_run_issue_summary_artifact_link` to call this helper. Do not change `_output_index_link`, gallery navigation, or internal `ReportLink` anchors.

- [ ] **Step 5: Replace the combined table with execution-group sections**

Replace `_run_issue_summary_other_section` with `_run_issue_summary_surfaced_sections`. Group rows in the explicit order `completed`, `crashed`, `indeterminate`, omit empty groups, and use these headings:

```python
heading_by_execution: dict[ExecutionStatus, str] = {
    "completed": "Completed attempts requiring review",
    "crashed": "Crashed attempts requiring review",
    "indeterminate": "Indeterminate attempts requiring review",
}
```

Each row must contain:

```python
(
    result["model"],
    assessment["usability"].replace("_", " "),
    _human_observation_labels(
        assessment["observations"],
        details=assessment.get("details"),
    ),
    evidence_link,
)
```

Use `ReportTable(("Model", "Usability", "Observed result", "Evidence"), ...)`. Rename `Actionable failures` to `Crashes requiring action`, and extend the report blocks with the returned tuple of execution sections.

- [ ] **Step 6: Run focused report tests and verify GREEN**

Run:

```bash
pytest src/tests/test_report_generation.py -k "run_issue_summary or output_link_style or output_index" -q
pytest src/tests/test_markdown_formatting.py -q
```

Expected: grouped-table, paste-safe-link, local-index-link, and Markdown formatting tests all pass.

- [ ] **Step 7: Commit the aggregate report change**

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_markdown_formatting.py
git commit -m "feat: group issue results and preserve repository links"
```

---

### Task 4: Document behavior and run the complete gate

**Files:**

- Modify: `src/README.md` in the output-artifact and `--link-style` documentation
- Modify: `CHANGELOG.md` under `[Unreleased]`
- Verify only: `src/output/` remains byte-for-byte untouched and absent from `git status`

**Interfaces:**

- Consumes: completed behavior from Tasks 1-3
- Produces: maintainer-facing documentation and a fully verified branch

- [ ] **Step 1: Update the README**

Document that:

- conventional Markdown heading labels such as `### Title:` satisfy the structured catalogue contract;
- human observations are explanatory and severity-ordered;
- the aggregate issue report separates completed, crashed, and indeterminate attempts;
- `--link-style relative` affects local-navigation artifacts, while issue-ready cross-file links remain canonical GitHub repository URLs.

- [ ] **Step 2: Update the changelog**

Add one `[Unreleased]` fixed entry for multi-hash Markdown section recognition and one changed entry for readable observations, severity ordering, grouped execution tables, and paste-safe issue links.

- [ ] **Step 3: Run prescribed formatting and lint preparation**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make format
make -C src lint-fix
make lint
```

Expected: safe Ruff fixes only; all lint checks pass. Do not run unsafe Ruff fixes unless previewing them is demonstrably faster than a manual correction, and never accept them without critical diff review.

- [ ] **Step 4: Run focused suites**

Run:

```bash
pytest src/tests/test_quality_analysis.py src/tests/test_report_generation.py src/tests/test_markdown_formatting.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Prove retained outputs were not touched**

Run:

```bash
git status --short
git diff -- src/output/
```

Expected: no `src/output/` path appears in status or diff.

- [ ] **Step 6: Run commit hygiene and the full quality gate**

Run:

```bash
bash src/tools/run_commit_hygiene.sh
make quality
```

Expected: all type checkers, Ruff, Vulture, Skylos, pytest, ShellCheck, and markdownlint pass; Skylos reports no quality or dead-code findings.

- [ ] **Step 7: Inspect the final diff and commit documentation**

Run:

```bash
git diff --check
git status --short
git diff --stat
git add src/README.md CHANGELOG.md
git commit -m "docs: explain issue-ready report presentation"
```

Expected: the branch is clean after the commit, `src/output/` is unchanged, and the history contains focused commits for parsing, observation presentation, grouped issue rendering, and documentation.
