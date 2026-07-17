# Report Evidence Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the gallery's skim-first tables without shortening successful model output, and add complete expandable generated-output evidence to diagnostics issue sections.

**Architecture:** Keep `src/check_models.py` as the intentional single-file implementation. The gallery will read complete successful text directly while retaining a bounded failure diagnostic; diagnostics will compose existing `ReportSection`, `ReportDetails`, and `ReportCodeBlock` primitives from existing issue-cluster results.

**Tech Stack:** Python 3.13, existing Markdown report primitives, pytest, Ruff, mypy, ty, pyrefly, markdownlint, and the project quality scripts.

## Global Constraints

- The package baseline is `0.8.6`, already committed and pushed before implementation.
- Keep `src/check_models.py` as a single-file monolith.
- Reuse existing report primitives and generation-result helpers.
- Do not add lint or type-checker suppressions.
- Successful output displayed in the consolidated gallery summary must not lose content.
- Diagnostics generated output must be complete, per-model, collapsed by default, and expandable by clicking.
- Hard crashes remain conclusive without generated-output evidence.
- Do not change JSONL, TSV, history, repro-bundle, or other machine schemas.
- Tests must write generated artifacts under `tmp_path`, never tracked `src/output/` paths.
- Preserve unrelated working-tree changes, including concurrently regenerated output artifacts.

---

### Task 1: Consolidate the gallery summary and preserve complete output

**Files:**

- Modify: `src/tests/test_report_generation.py` in `TestMarkdownGalleryReport`
- Modify: `src/check_models.py` around the gallery output helpers, summary builders, and `generate_markdown_gallery_report`

**Interfaces:**

- Consumes: `PerformanceResult`, `_generation_text_value()`, `_collapse_preview_whitespace()`, `_build_result_output_preview()`, `_gallery_summary_status()`, and `_gallery_summary_signal()`.
- Produces: `_gallery_success_output(result: PerformanceResult) -> str` and one `_build_gallery_output_cost_summary_section(report_context: ReportRenderContext) -> list[str]` headed `Model Output and Cost Summary`.

- [ ] **Step 1: Write the failing gallery tests**

Add this behavior test and update existing summary tests to use the consolidated heading:

```python
def test_gallery_consolidates_summary_with_complete_model_output(
    self,
    tmp_path: Path,
) -> None:
    complete_text = "BEGIN " + ("distinct middle evidence " * 30) + "END-SENTINEL"
    result = PerformanceResult(
        model_name="org/complete-output",
        success=True,
        generation=_MockGeneration(
            text=complete_text,
            prompt_tokens=18,
            generation_tokens=200,
            generation_tps=42.0,
            peak_memory=2.5,
        ),
        total_time=1.25,
        generation_time=0.75,
        model_load_time=0.50,
    )
    out = tmp_path / "model_gallery.md"
    generate_markdown_gallery_report(
        results=[result],
        filename=out,
        prompt="Describe this image briefly.",
    )

    content = out.read_text(encoding="utf-8")
    summary = _extract_markdown_subsection(
        content,
        "## Model Output and Cost Summary",
        end_headings=("## Quick Navigation", "## Model Gallery"),
    )
    assert "## Model Quality Summary" not in content
    assert "## All Model Output and Cost Summary" not in content
    assert "BEGIN" in summary
    assert "END-SENTINEL" in summary
    assert "[tail]" not in summary
    assert "Gen tok" in summary
    assert "Peak GB" in summary
    assert "Quality signal" in summary
```

In the existing issue-style and output/cost tests, extract
`## Model Output and Cost Summary` and retain assertions for status,
diagnostic, runtime, memory, and quality cells.

- [ ] **Step 2: Run the gallery tests and verify RED**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py::TestMarkdownGalleryReport -q'
```

Expected: FAIL because the consolidated heading does not exist and the current
summary preview omits `END-SENTINEL`.

- [ ] **Step 3: Implement the minimal consolidated gallery renderer**

Replace the successful preview helper with:

```python
def _gallery_success_output(result: PerformanceResult) -> str:
    """Return complete successful model text flattened for a Markdown table cell."""
    output = _generation_text_value(result.generation)
    return _collapse_preview_whitespace(output) if output else ""
```

Rename the remaining preview constant to:

```python
GALLERY_FAILURE_DIAGNOSTIC_PREVIEW_CHARS: Final[int] = 220
```

Update `_gallery_output_cost_preview()` so successful results call
`_gallery_success_output(result)` without a character limit. Failed results
continue through `_build_result_output_preview()` with
`GALLERY_FAILURE_DIAGNOSTIC_PREVIEW_CHARS`.

Delete `_gallery_summary_preview()`,
`GALLERY_QUALITY_SUMMARY_PREVIEW_CHARS`, and
`_build_gallery_quality_summary_section()`. Change the retained section to:

```python
return [
    "## Model Output and Cost Summary",
    "",
    (
        "Every model in this run, with complete successful output or a concise "
        "failure diagnostic beside runtime, memory, and quality signals."
    ),
    "",
    *_guard_markdownlint_block(table_lines, rules=MARKDOWNLINT_GALLERY_SUMMARY_RULES),
    "",
]
```

Remove the call to `_build_gallery_quality_summary_section(report_context)`
from `generate_markdown_gallery_report()` and retain the output/cost call.

- [ ] **Step 4: Run the gallery tests and verify GREEN**

Run the command from Step 2.

Expected: all `TestMarkdownGalleryReport` tests PASS and the ending sentinel is
present in the consolidated table.

- [ ] **Step 5: Run dead-code and Markdown regression checks**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_markdown_formatting.py src/tests/test_report_generation.py::TestMarkdownGalleryReport -q'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && cd src && vulture check_models.py --min-confidence 80'
```

Expected: PASS with no obsolete gallery helper reported as dead code.

- [ ] **Step 6: Commit the gallery change**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "refactor: preserve complete gallery output"
```

---

### Task 2: Add complete expandable output evidence to diagnostics

**Files:**

- Modify: `src/tests/test_report_generation.py` in `TestDiagnosticsReport`
- Modify: `src/check_models.py` near `_diagnostics_cluster_issue_sections`

**Interfaces:**

- Consumes: `IssueCluster`, `_issue_cluster_results()`, `_generation_text_value()`, `ReportSection`, `ReportDetails`, `ReportCodeBlock`, `render_report_markdown()`, and `_guard_markdownlint_block()`.
- Produces: `_diagnostics_generated_output_section(cluster: IssueCluster) -> list[str]`, inserted after expected/actual behaviour for each scoped cluster.

- [ ] **Step 1: Write the failing diagnostics evidence tests**

Add this test:

```python
def test_scoped_harness_cluster_includes_complete_expandable_output(
    self,
    tmp_path: Path,
) -> None:
    complete_text = (
        "BEGIN-EVIDENCE "
        + ("model reasoning context " * 20)
        + "<|end|> END-EVIDENCE"
    )
    output = tmp_path / "diagnostics.md"
    result = _make_harness_success(
        "org/stop-token",
        text=complete_text,
        harness_type="stop_token",
        harness_detail="token_leak:<|end|>",
    )
    generate_diagnostics_report(
        results=[result],
        filename=output,
        versions={"mlx-vlm": "0.6.5"},
        system_info={},
        prompt="Describe the image.",
    )

    content = output.read_text(encoding="utf-8")
    assert "### Generated Output Evidence" in content
    assert "<details>" in content
    assert "<summary>Complete generated output: org/stop-token</summary>" in content
    assert "```text" in content
    assert "BEGIN-EVIDENCE" in content
    assert "END-EVIDENCE" in content
    assert "[tail]" not in content
```

Add a failure-only test using `_make_failure_with_details()` that asserts
`### Generated Output Evidence` is absent when generation text is unavailable.

- [ ] **Step 2: Run the diagnostics tests and verify RED**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py::TestDiagnosticsReport -q'
```

Expected: FAIL because diagnostics issue sections do not emit generated-output
details.

- [ ] **Step 3: Implement the shared-primitives diagnostics section**

Add beside the existing Markdownlint rule constants:

```python
MARKDOWNLINT_DETAILS_RULES: Final[str] = "MD033"
```

Add before `_diagnostics_cluster_issue_sections()`:

```python
def _diagnostics_generated_output_section(cluster: IssueCluster) -> list[str]:
    """Render complete captured model output as per-model expandable evidence."""
    details = tuple(
        ReportDetails(
            summary=f"Complete generated output: {result.model_name}",
            blocks=(ReportCodeBlock(output, language="text"),),
        )
        for result in _issue_cluster_results(cluster)
        if (output := _generation_text_value(result.generation).strip())
    )
    if not details:
        return []
    return _guard_markdownlint_block(
        render_report_markdown(
            (
                ReportSection(
                    title="Generated Output Evidence",
                    level=3,
                    blocks=details,
                ),
            )
        ),
        rules=MARKDOWNLINT_DETAILS_RULES,
    )
```

In `_diagnostics_cluster_issue_sections()`, append after expected/actual and
before `_native_repro_block()`:

```python
parts.extend(_diagnostics_generated_output_section(cluster))
parts.append("")
```

Preserve the crash matrix and captured stdout/stderr policy.

- [ ] **Step 4: Run the diagnostics tests and verify GREEN**

Run the command from Step 2.

Expected: all `TestDiagnosticsReport` tests PASS, including the test that keeps
captured process stdout/stderr out of compact diagnostics.

- [ ] **Step 5: Verify Markdown rendering and lint safety**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_markdown_formatting.py src/tests/test_report_generation.py::TestDiagnosticsReport::test_multiple_diagnostic_clusters_use_lint_safe_heading_hierarchy -q'
```

Expected: PASS with local MD033 guards around expandable evidence.

- [ ] **Step 6: Commit the diagnostics change**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "feat: retain complete diagnostics output evidence"
```

---

### Task 3: Document, verify, and publish the report refinement

**Files:**

- Modify: `src/README.md` in **Understanding the Output** and **Gallery Markdown Report**
- Modify: `CHANGELOG.md` under `[Unreleased]`
- Review only: `src/output/reports/model_gallery.md`
- Review only: `src/output/reports/diagnostics.md`

**Interfaces:**

- Consumes: the implemented gallery and diagnostics behavior from Tasks 1 and 2.
- Produces: user-facing documentation and a fully verified branch; no new runtime interface or schema.

- [ ] **Step 1: Update documentation**

State that the gallery has one consolidated output/cost/quality summary with
complete successful output, followed by structured per-model sections. State
that diagnostics retains complete generated output in per-model expandable
evidence blocks while tracebacks and process logs remain bounded.

Add beneath the `0.8.6` baseline changelog note:

```markdown
- Consolidate duplicate gallery summaries without shortening successful model
  output, and retain complete affected-model output in expandable diagnostics
  evidence blocks.
```

- [ ] **Step 2: Run formatting and Ruff in the required order**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make format'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make -C src lint-fix'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make lint'
```

Expected: formatted and Ruff-clean without new suppressions.

- [ ] **Step 3: Run focused report suites**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_markdown_formatting.py -q'
```

Expected: PASS. Tests leave tracked `src/output/` artifacts unchanged.

- [ ] **Step 4: Run the full quality gate**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make quality'
```

Expected: every configured quality stage passes. If unrelated regenerated
`src/output/` changes make a snapshot assertion fail, verify the committed
branch in a clean worktree rather than editing or discarding those artifacts.

- [ ] **Step 5: Review generated-artifact behavior without fabricating output**

If a supported replay/finalization path exists, regenerate through it and
verify:

```text
model_gallery.md: one summary; complete successful output; detailed sections retained
diagnostics.md: complete output inside closed per-model details blocks
```

If no replay path exists, do not manually edit tracked reports and do not rerun
all models solely to manufacture fixtures. Record that the next normal
benchmark will refresh them.

- [ ] **Step 6: Commit documentation**

```bash
git add CHANGELOG.md src/README.md
git commit -m "docs: explain complete report evidence"
```

- [ ] **Step 7: Push the completed branch**

```bash
git push origin codex/report-evidence-refinement
```

Expected: remote CI receives only intentional source, test, and documentation
changes; unrelated local output regeneration remains unstaged.
