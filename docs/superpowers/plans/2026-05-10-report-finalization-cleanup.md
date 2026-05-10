# Report Finalization Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate report artifact generation and logging behind one small internal artifact list.

**Architecture:** Keep `src/check_models.py` as the project monolith. Add a private `ReportArtifact` dataclass near the report/finalization helpers, construct artifacts from `ReportGenerationInputs`, and use the artifact list for report jobs, directory creation, and success-path logging.

**Tech Stack:** Python 3.13, dataclasses, pathlib, pytest, existing `make` quality targets.

---

## Task 1: Add Artifact Metadata Test

**Files:**

- Modify: `src/tests/test_metrics_modes.py`
- Modify: `src/check_models.py`

- [ ] **Step 1: Write the failing test**

Add this test after `test_finalize_execution_logs_configured_log_and_env_paths` in `src/tests/test_metrics_modes.py`:

```python
def test_report_generation_uses_single_artifact_plan(
    tmp_path: Path,
) -> None:
    """Report generation should expose one ordered artifact plan for jobs and logs."""
    args = argparse.Namespace(
        output_html=tmp_path / "report.html",
        output_markdown=tmp_path / "report.md",
        output_gallery_markdown=tmp_path / "gallery.md",
        output_review=tmp_path / "review.md",
        output_tsv=tmp_path / "report.tsv",
        output_jsonl=tmp_path / "report.jsonl",
    )
    inputs = check_models.ReportGenerationInputs(
        args=args,
        results=[],
        library_versions={"mlx": "0.0.0"},
        prompt="prompt",
        metadata=None,
        overall_time=1.0,
        image_path=None,
        system_info={},
        report_context=check_models._build_report_render_context(results=[], prompt="prompt"),
        jsonl_output_path=args.output_jsonl.resolve(),
        log_output_path=(tmp_path / "check_models.log").resolve(),
        env_output_path=(tmp_path / "environment.log").resolve(),
        review_output_path=args.output_review.resolve(),
        runtime_fingerprint={},
    )

    artifacts = check_models._build_report_artifacts(inputs)

    assert [artifact.key for artifact in artifacts] == [
        "html",
        "markdown",
        "markdown_gallery",
        "review",
        "tsv",
        "jsonl",
    ]
    assert [artifact.label.strip() for artifact in artifacts] == [
        "HTML Report:",
        "Markdown Report:",
        "Gallery Report:",
        "Review Report:",
        "TSV Report:",
        "JSONL Report:",
    ]
    assert all(artifact.path.is_absolute() for artifact in artifacts)
    assert all(artifact.job is not None for artifact in artifacts)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n mlx-vlm pytest src/tests/test_metrics_modes.py::test_report_generation_uses_single_artifact_plan -q
```

Expected: FAIL because `_build_report_artifacts` does not exist.

- [ ] **Step 3: Implement artifact metadata**

In `src/check_models.py`, add this dataclass near `ReportGenerationInputs`:

```python
@dataclass(frozen=True)
class ReportArtifact:
    """A generated artifact path plus its optional generation job."""

    key: str
    label: str
    path: Path
    job: Callable[[], None] | None = None
```

Add `_build_report_artifacts(inputs: ReportGenerationInputs) -> tuple[ReportArtifact, ...]` above `_generate_reports_and_log_outputs`. It should resolve paths once, build the six report jobs in order, and return `ReportArtifact` entries for HTML, Markdown, gallery, review, TSV, and JSONL.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run -n mlx-vlm pytest src/tests/test_metrics_modes.py::test_report_generation_uses_single_artifact_plan -q
```

Expected: PASS.

## Task 2: Refactor Report Generation

**Files:**

- Modify: `src/check_models.py`
- Modify: `src/tests/test_metrics_modes.py`

- [ ] **Step 1: Write the failing integration-style test**

Extend `test_finalize_execution_logs_configured_log_and_env_paths` so the captured messages are asserted in this order:

```python
artifact_positions = [
    next(i for i, msg in enumerate(messages) if "HTML Report:" in msg),
    next(i for i, msg in enumerate(messages) if "Markdown Report:" in msg),
    next(i for i, msg in enumerate(messages) if "Gallery Report:" in msg),
    next(i for i, msg in enumerate(messages) if "Review Report:" in msg),
    next(i for i, msg in enumerate(messages) if "TSV Report:" in msg),
    next(i for i, msg in enumerate(messages) if "JSONL Report:" in msg),
]
assert artifact_positions == sorted(artifact_positions)
```

- [ ] **Step 2: Run test to verify it fails if labels/order are not unified**

Run:

```bash
conda run -n mlx-vlm pytest src/tests/test_metrics_modes.py::test_finalize_execution_logs_configured_log_and_env_paths -q
```

Expected before implementation: FAIL if old label spacing or order does not match the unified artifact list.

- [ ] **Step 3: Use artifact metadata in `_generate_reports_and_log_outputs`**

Replace the local `report_jobs` tuple and later `report_paths` tuple with:

```python
artifacts = _build_report_artifacts(inputs)
```

Loop over `artifacts` for jobs:

```python
for artifact in artifacts:
    if artifact.job is None:
        continue
    try:
        artifact.job()
    except (OSError, ValueError) as err:
        logger.exception("Failed to generate %s report.", artifact.key)
        _write_report_failure_jsonl(
            filename=inputs.jsonl_output_path,
            failed_report=artifact.key,
            error=err,
        )
```

Loop over `artifacts` for success logging:

```python
for artifact in artifacts:
    log_file_path(artifact.path, label=artifact.label)
```

Keep log/env logging after report artifact logging.

- [ ] **Step 4: Run focused tests**

Run:

```bash
conda run -n mlx-vlm pytest src/tests/test_metrics_modes.py::test_report_generation_uses_single_artifact_plan src/tests/test_metrics_modes.py::test_finalize_execution_logs_configured_log_and_env_paths -q
```

Expected: PASS.

## Task 3: Centralize Finalization Paths

**Files:**

- Modify: `src/check_models.py`
- Modify: `src/tests/test_metrics_modes.py`

- [ ] **Step 1: Tighten the pruning test**

Keep `test_finalize_execution_prunes_canonical_repro_bundle_dir`, and add an assertion that `_clean_stale_toplevel_reports` receives resolved output/report directories:

```python
patch("check_models._clean_stale_toplevel_reports", return_value=0) as clean_stale_reports,
```

After `finalize_execution(...)`, assert:

```python
clean_stale_reports.assert_called_once_with(tmp_path, reports_dir)
```

- [ ] **Step 2: Run test to verify current behavior**

Run:

```bash
conda run -n mlx-vlm pytest src/tests/test_metrics_modes.py::test_finalize_execution_prunes_canonical_repro_bundle_dir -q
```

Expected: PASS or FAIL depending on current path object resolution. If it passes, keep it as regression coverage for the refactor.

- [ ] **Step 3: Refactor resolved path setup**

In `finalize_execution`, compute these once:

```python
html_output_path = args.output_html.resolve()
markdown_output_path = args.output_markdown.resolve()
gallery_output_path = args.output_gallery_markdown.resolve()
review_output_path = args.output_review.resolve()
tsv_output_path = args.output_tsv.resolve()
jsonl_output_path = args.output_jsonl.resolve()
diagnostics_path = args.output_diagnostics.resolve()
log_output_path = args.output_log.resolve()
env_output_path = args.output_env.resolve()
```

Create directories from one tuple of resolved output paths, pass the resolved review/log/env/jsonl paths into `ReportGenerationInputs`, and use `diagnostics_path.parent.parent` for repro pruning as before.

- [ ] **Step 4: Run focused finalization tests**

Run:

```bash
conda run -n mlx-vlm pytest src/tests/test_metrics_modes.py::test_finalize_execution_logs_configured_log_and_env_paths src/tests/test_metrics_modes.py::test_finalize_execution_prunes_canonical_repro_bundle_dir -q
```

Expected: PASS.

## Task 4: Docs and Verification

**Files:**

- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update changelog**

Add an `[Unreleased]` bullet:

```markdown
- Streamlined report finalization by driving report generation and artifact
  logging from one internal artifact list, reducing duplicated output-path
  handling without changing report formats.
```

- [ ] **Step 2: Run focused report tests**

Run:

```bash
conda run -n mlx-vlm pytest src/tests/test_metrics_modes.py src/tests/test_report_generation.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the full project gate**

Run:

```bash
make quality
```

Expected: PASS. If it fails on the known environment `datasets`/`fsspec` mismatch before tests begin, report that separately and run the narrower test suite as evidence.
