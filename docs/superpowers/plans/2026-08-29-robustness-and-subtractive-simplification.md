# Robustness and Subtractive Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make report finalization fail-soft, make every output assessment canonical, and materially shorten the monolith and tests by removing duplicate artifact, schema, and CLI contracts.

**Architecture:** Preserve `src/check_models.py` as the production monolith. First consolidate text observations and artifact outcomes at existing pure/side-effect boundaries; then release one schema-3 JSONL machine contract and one output-root CLI contract, deleting `run.json`, per-artifact path flags, and obsolete aliases rather than maintaining adapters.

**Tech Stack:** Python 3.13+, argparse, dataclasses, JSON Lines, Rich, pytest, Ruff, mypy, ty, pyrefly, Skylos.

**Spec:** `docs/superpowers/specs/2026-08-29-robustness-and-subtractive-simplification-design.md`

## Global Constraints

- Activate the `mlx-vlm` conda environment before every Python or Make command.
- Keep `src/check_models.py` as the intentional production monolith.
- Preserve complete generated output, failure tracebacks, captured upstream output, provenance, prompt diagnostics, timing, memory, telemetry, and reproduction evidence.
- Preserve `results.history.jsonl` on schema `1.0`; it remains local-only.
- Add or modify tests only in existing `src/tests/test_*.py` files.
- Send generated validation artifacts to `tmp_path` or another untracked path; never rewrite tracked `src/output/` assets.
- Do not add a schema-validation dependency; consolidation must produce net deletion.
- Update `CHANGELOG.md` under `[Unreleased]` and update public documentation for every schema or CLI change.
- Run the robustness phase through the full quality gate before starting the breaking contract phase.
- Release the breaking contract phase as version `0.16.0`.

## File Map

- Modify: `src/check_models.py` — canonical observations, artifact isolation and outcomes, schema 3, output-root CLI, compatibility deletion.
- Modify: `src/tools/analyze_output_quality.py` — consume canonical observations and status.
- Modify: `src/tests/test_analyze_output_quality.py` — canonical standalone verdicts.
- Modify: `src/tests/test_process_image_mock.py` — remove the cached quality-string contract while retaining structured failure-output evidence.
- Modify: `src/tests/test_quality_analysis.py` — pure observation projection and usability tests.
- Modify: `src/tests/test_report_generation.py` — artifact containment, outcome-driven links, constraint aggregation, schema-3 loading and comparison.
- Modify: `src/tests/test_jsonl_output.py` — schema-3 serialization and removal of run JSON.
- Modify: `src/tests/test_parameter_validation.py` — output-root and CLI compatibility removal.
- Modify: `src/tests/test_cli_help_output.py` — new public CLI surface.
- Modify: `src/tests/test_cli_integration.py` — output-root integration.
- Modify: `src/tests/test_e2e_smoke.py` — one output argument instead of seven.
- Modify: `src/tests/test_metrics_modes.py` — verbose always means detailed metrics.
- Modify: `src/tests/test_dependency_sync.py` — schema/version invariants and removal of retired semantic archaeology.
- Modify: `src/README.md` — schema, artifact layout, CLI and metrics documentation.
- Modify: `docs/IMPLEMENTATION_GUIDE.md` — canonical retained-artifact architecture.
- Modify: `.github/copilot-instructions.md` — output contract and current monolith map/size.
- Modify: `src/pyproject.toml` — version `0.16.0`.
- Modify: `CHANGELOG.md` — robustness and breaking-contract entries.
- Move after completion: this plan and its design spec to `docs/notes/archive/superpowers/`.

---

## Phase 1: Backwards-compatible robustness

### Task 1: Canonicalise text observations and remove the cached verdict string

**Files:**

- Modify: `src/check_models.py:1850-1895, 4761-4789, 8680-8809, 13652-13930, 16845-17098`
- Modify: `src/tools/analyze_output_quality.py`
- Test: `src/tests/test_analyze_output_quality.py`
- Test: `src/tests/test_process_image_mock.py`
- Test: `src/tests/test_quality_analysis.py`
- Test: `src/tests/test_metrics_modes.py`

**Interfaces:**

- Produces: `_quality_observations(*, text: str, generated_tokens: int | None, analysis: GenerationQualityAnalysis | None, stop_reason: str | None = None) -> tuple[ObservationCode, ...]`.
- Produces: `_completed_assessment(observations: Sequence[ObservationCode]) -> ResultAssessment`.
- Preserves: `_assessment_observations(result: PerformanceResult) -> tuple[ObservationCode, ...]` as a thin result adapter.
- Removes: `PerformanceResult.quality_issues`, `GenerationQualityAnalysis.requested_max_tokens`, `_analyze_text_quality`, and `_build_quality_issues_string`.

- [ ] **Step 1: Add canonical standalone-classification failures**

Add this parameterised JSON-mode test to `src/tests/test_analyze_output_quality.py`:

```python
@pytest.mark.parametrize(
    ("text", "observation", "status", "exit_code"),
    [
        ("", "empty_output", "unusable", 1),
        ("<think>unfinished reasoning", "thinking_trace_incomplete", "unusable", 1),
        ("x", "minimal_output", "observation", 0),
    ],
)
def test_json_mode_uses_canonical_assessment(
    text: str,
    observation: str,
    status: str,
    exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["analyze_output_quality.py", "--text", text, "--json"])

    assert main() == exit_code
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == status
    assert observation in payload["assessment"]["observations"]
    assert payload["assessment"]["usability"] == (
        "unusable" if exit_code else "usable_with_caveats"
    )
```

- [ ] **Step 2: Run the new test and confirm the current drift**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_analyze_output_quality.py::test_json_mode_uses_canonical_assessment -q
```

Expected: FAIL because empty text is `clean`, incomplete thinking is only an
observation with exit status 0, and minimal text has no observation.

- [ ] **Step 3: Extract the pure observation and completed-assessment helpers**

Refactor the current body of `_assessment_observations` into this shape in
`src/check_models.py`:

```python
def _quality_observations(
    *,
    text: str,
    generated_tokens: int | None,
    analysis: GenerationQualityAnalysis | None,
    stop_reason: str | None = None,
) -> tuple[ObservationCode, ...]:
    observations: list[ObservationCode] = []
    if not text.strip():
        observations.append("empty_output")
    elif _detect_minimal_output(text, generated_tokens)[0]:
        observations.append("minimal_output")
    if analysis is not None and analysis.is_repetitive:
        observations.append("repeated_output")
    elif analysis is None and _detect_repetitive_output(text)[0]:
        observations.append("repeated_output")
    if stop_reason == "repetition_abort":
        observations.append("repetition_abort")
    if analysis is None:
        return tuple(observations)

    non_thinking_wrappers = set(analysis.configured_generation_wrappers).difference(
        analysis.thinking_trace_markers
    )
    candidates: tuple[tuple[bool, ObservationCode], ...] = (
        (bool(analysis.missing_sections), "missing_requested_sections"),
        (
            analysis.likely_capped and bool(analysis.token_cap_reasons),
            "token_cap_truncation",
        ),
        (analysis.instruction_echo, "prompt_instruction_echo"),
        (bool(analysis.unexpected_catalog_preamble), "unexpected_catalog_preamble"),
        (bool(analysis.unexpected_special_tokens), "unexpected_special_token"),
        (bool(non_thinking_wrappers), "configured_wrapper_present"),
        (analysis.thinking_only_output, "missing_final_answer"),
        (analysis.thinking_trace_incomplete, "thinking_trace_incomplete"),
        (bool(analysis.role_boundary_tokens), "role_boundary_token_present"),
        (_has_catalog_constraint_violation(analysis), "catalog_constraint_violation"),
        (analysis.keyword_overlap == "no_overlap", "no_keyword_overlap"),
        (bool(analysis.unchanged_draft_fields), "draft_returned_unchanged"),
    )
    observations.extend(code for condition, code in candidates if condition)
    return tuple(observations)


def _completed_assessment(observations: Sequence[ObservationCode]) -> ResultAssessment:
    ordered = tuple(observations)
    usability: ModelUsability = (
        "unusable"
        if set(ordered) & _UNUSABLE_OBSERVATIONS
        else "usable_with_caveats"
        if ordered
        else "usable"
    )
    maintainer_status: MaintainerStatus = (
        "observation_needs_reproduction"
        if set(ordered) & _INTEGRATION_SIGNAL_OBSERVATIONS
        else "none"
    )
    return ResultAssessment("completed", usability, maintainer_status, ordered)
```

Make `_assessment_observations` extract text, token count, and stop reason and
delegate to `_quality_observations`. Make `_assess_result` delegate completed
results to `_completed_assessment` while retaining its crash/indeterminate
branch.

- [ ] **Step 4: Remove the cached string and unused analysis field**

Delete the following production surface and all call arguments that exist only
for it:

```text
PerformanceResult.quality_issues
GenerationQualityAnalysis.requested_max_tokens
_analyze_text_quality
_build_quality_issues_string
_FIELD_LABELS["quality_issues"]
```

Call `analyze_generation_text` directly from `_populate_result_quality_analysis`
and failure-stdout analysis. Retain `quality_analysis` on failure results as
neutral captured-output evidence. In `process_models`, log canonical codes:

```python
assessment = _assess_result(result)
if assessment.observations:
    logger.info(
        "Mechanical observations for %s: %s",
        result.model_name,
        ", ".join(assessment.observations),
    )
```

- [ ] **Step 5: Make the standalone tool serialize the canonical assessment**

After `analyze_generation_text`, call `_quality_observations` and
`_completed_assessment`. Replace private status logic with:

```python
observations = _quality_observations(
    text=output_text,
    generated_tokens=estimated_tokens,
    analysis=analysis,
)
assessment = _completed_assessment(observations)
exit_code = 1 if assessment.usability == "unusable" else 0
status = (
    "unusable"
    if assessment.usability == "unusable"
    else "observation"
    if assessment.observations
    else "clean"
)
```

Add `assessment.execution`, `assessment.usability`,
`assessment.maintainer_status`, and `list(assessment.observations)` to JSON.
Render human labels from `_human_observation_labels` rather than a second issue
string.

- [ ] **Step 6: Update tests that constructed or asserted `quality_issues`**

In `test_process_image_mock.py` retain assertions that repetitive stdout
produces `result.quality_analysis.is_repetitive`, but delete assertions about
`result.quality_issues`. In `test_metrics_modes.py` and
`test_report_generation.py`, remove fixture keyword arguments for the deleted
field and assert canonical observation labels instead.

- [ ] **Step 7: Run focused classification and processing tests**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_analyze_output_quality.py src/tests/test_quality_analysis.py src/tests/test_process_image_mock.py src/tests/test_metrics_modes.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the canonical observation refactor**

```bash
git add src/check_models.py src/tools/analyze_output_quality.py src/tests/test_analyze_output_quality.py src/tests/test_quality_analysis.py src/tests/test_process_image_mock.py src/tests/test_metrics_modes.py src/tests/test_report_generation.py
git commit -m "refactor: canonicalise output observations"
```

### Task 2: Make reports and comparisons true isolation boundaries

**Files:**

- Modify: `src/check_models.py:21315-21530`
- Test: `src/tests/test_report_generation.py`

**Interfaces:**

- Preserves: `_generate_reports_and_log_outputs(inputs) -> tuple[ReportArtifactOutcome, ...]`.
- Changes: every ordinary `Exception` from a renderer or comparison becomes a logged failed outcome or a skipped comparison.
- Preserves: propagation of `KeyboardInterrupt` and `SystemExit`.

- [ ] **Step 1: Add an unexpected-exception artifact regression**

Extend the existing `test_report_orchestration_passes_generated_issue_drafts_to_index`
fixture with a separate test:

```python
def test_report_artifact_key_error_is_contained(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inputs = _report_generation_inputs(tmp_path, result=_make_success("org/model"))
    with patch.object(
        check_models,
        "generate_html_report",
        side_effect=KeyError("unexpected renderer field"),
    ):
        outcomes = check_models._generate_reports_and_log_outputs(inputs)

    html_outcome = _report_outcome(outcomes, "html")
    assert html_outcome.succeeded is False
    assert "unexpected renderer field" in (html_outcome.error_message or "")
    assert _report_outcome(outcomes, "jsonl").succeeded is True
    assert inputs.output_paths.jsonl.is_file()
    assert "Failed to generate html report" in caplog.text
```

Add this local helper beside `_report_outcome` and use it in the new boundary
tests:

```python
def _report_generation_inputs(
    tmp_path: Path,
    *,
    result: check_models.PerformanceResult,
) -> check_models.ReportGenerationInputs:
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    context = _build_report_render_context(results=[result], prompt="Describe the image.")
    return check_models.ReportGenerationInputs(
        results=[result],
        library_versions=_stub_versions(),
        prompt="Describe the image.",
        metadata=None,
        overall_time=1.0,
        image_path=None,
        system_info={},
        report_context=context,
        output_paths=output_paths,
        run_args=Namespace(
            compare_with="none",
            isolate=False,
            max_tokens=32,
            temperature=0.0,
            trust_remote_code=False,
            revision=None,
        ),
        runtime_fingerprint={},
    )
```

- [ ] **Step 2: Add diagnostics, summary, and comparison containment cases**

Parameterise the existing orchestration test so `KeyError` from
`_write_diagnostics_artifacts` and `generate_run_issue_summary_report` produces
failed outcomes without escaping. Add a comparison test:

```python
def test_unexpected_comparison_error_degrades_to_no_comparison(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    report_inputs = _report_generation_inputs(
        tmp_path,
        result=_make_success("org/model"),
    )
    with patch.object(check_models, "compare_run_results", side_effect=KeyError("bad diff")):
        comparison = check_models._compute_run_comparison(report_inputs)

    assert comparison is None
    assert "Comparison skipped" in caplog.text
```

- [ ] **Step 3: Run the containment tests and verify the current escapes**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py -k "key_error_is_contained or comparison_error_degrades" -q
```

Expected: FAIL with uncaught `KeyError`.

- [ ] **Step 4: Broaden only the explicit isolation catches**

Change the three report catches and the comparison boundary to:

```python
except Exception as error:  # noqa: BLE001 - report isolation must contain renderer defects
    logger.exception("Failed to generate %s report.", artifact.key)
```

For `_compute_run_comparison`, put baseline resolution, current-source
validation, `compare_run_results`, and comparison rendering behind the same
ordinary-exception boundary and log `Comparison skipped: unexpected comparison
failure (...)`. Do not catch `BaseException`.

- [ ] **Step 5: Run the complete report-orchestration test cluster**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py -k "report_orchestration or artifact or comparison" -q
```

Expected: PASS.

- [ ] **Step 6: Commit report containment**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "fix: contain unexpected report failures"
```

### Task 3: Aggregate constraint failures by their own bounds

**Files:**

- Modify: `src/check_models.py:19947-20021`
- Test: `src/tests/test_report_generation.py`

**Interfaces:**

- Preserves: `_run_issue_summary_constraint_breakdown(results) -> ReportSection | None`.
- Changes: title and keyword counts are grouped by `(minimum, maximum)` before below/above totals and medians are computed.

- [ ] **Step 1: Add a mixed-range regression**

Add a test that builds two completed result records with title facts `6` against
`[2, 4]` and `5` against `[6, 8]`:

```python
def test_constraint_breakdown_keeps_each_declared_range() -> None:
    first = _issue_summary_result(
        "org/above",
        observations=["catalog_constraint_violation"],
        details={"title_word_count": 6, "title_word_range": [2, 4]},
    )
    second = _issue_summary_result(
        "org/below",
        observations=["catalog_constraint_violation"],
        details={"title_word_count": 5, "title_word_range": [6, 8]},
    )

    section = check_models._run_issue_summary_constraint_breakdown(
        [cast("check_models.JsonlResultRecord", first), cast("check_models.JsonlResultRecord", second)]
    )
    assert section is not None
    rendered = "\n".join(check_models.render_report_markdown((section,)))
    assert "outside 2-4 words (0 below, 1 above" in rendered
    assert "outside 6-8 words (1 below, 0 above" in rendered
```

Extend `_issue_summary_result` with `details: dict[str, object] | None = None`
and place it in the returned assessment only when non-`None`:

```python
assessment: dict[str, object] = {
    "execution": "completed",
    "usability": usability,
    "maintainer_status": maintainer_status,
    "observations": observations,
}
if details is not None:
    assessment["details"] = details
```

- [ ] **Step 2: Run the regression and confirm the last-range bug**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py::test_constraint_breakdown_keeps_each_declared_range -q
```

Expected: FAIL because both counts are currently summarised against `[6, 8]`.

- [ ] **Step 3: Group observations by bounds**

Replace the two observed lists and last-range variables with:

```python
title_by_range: dict[tuple[int, int], list[int]] = {}
keyword_by_range: dict[tuple[int, int], list[int]] = {}


def _record_outside_range(
    groups: dict[tuple[int, int], list[int]],
    count: JsonLike,
    bounds_value: JsonLike,
) -> None:
    bounds = _int_pair(bounds_value)
    if (
        isinstance(count, int)
        and not isinstance(count, bool)
        and bounds is not None
        and not bounds[0] <= count <= bounds[1]
    ):
        groups.setdefault(bounds, []).append(count)
```

Render sorted `groups.items()` through the existing `_range_line`. Preserve the
existing one-range wording so normal outputs do not churn unnecessarily.

- [ ] **Step 4: Run constraint and run-summary tests**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py -k "constraint or run_issue_summary" -q
```

Expected: PASS.

- [ ] **Step 5: Commit the aggregation fix**

```bash
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "fix: preserve constraint ranges in summaries"
```

### Task 4: Make current outcomes the artifact navigation contract

**Files:**

- Modify: `src/check_models.py:1400-1435, 20340-20680, 21160-21860`
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_metrics_modes.py`

**Interfaces:**

- Produces: one `ReportArtifact` dataclass containing `key`, `public_key`, `label`, `path`, `dashboard_label`, `dashboard_purpose`, and `job`.
- Removes: `ReportArtifactSpec` and `_build_report_artifact_specs`.
- Changes: `generate_output_index_report(filename: Path, *, artifacts: Sequence[ReportArtifact], run_issue_summary: Path | None = None, issue_reports: Mapping[str, Path] | None = None, assessments: Sequence[tuple[str, ResultAssessment]] | None = None) -> None` accepts current available artifacts rather than constructing an unconditional path list.
- Changes: `_print_reports_dashboard(artifacts: Sequence[ReportArtifact], outcomes: Sequence[ReportArtifactOutcome], history_path: Path | None = None, *, run_issue_summary: Path | None = None) -> None` accepts current artifacts and outcomes.

- [ ] **Step 1: Add stale-artifact navigation regressions**

Write a stale HTML file, force current HTML generation to raise, and assert the
current index and dashboard omit it:

```python
def test_failed_current_artifact_is_omitted_from_navigation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _report_generation_inputs(tmp_path, result=_make_success("org/model"))
    inputs.output_paths.html.parent.mkdir(parents=True, exist_ok=True)
    inputs.output_paths.html.write_text("stale html", encoding="utf-8")

    with patch.object(check_models, "generate_html_report", side_effect=KeyError("html failed")):
        outcomes = check_models._generate_reports_and_log_outputs(inputs)

    index = inputs.output_paths.index.read_text(encoding="utf-8")
    assert "results.html" not in index
    check_models._print_reports_dashboard(
        check_models._build_report_artifacts(inputs),
        outcomes,
        history_path=None,
    )
    assert "results.html" not in capsys.readouterr().err
```

- [ ] **Step 2: Run the test and confirm existence is currently mistaken for success**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py::test_failed_current_artifact_is_omitted_from_navigation -q
```

Expected: FAIL because the index links every configured artifact and the
dashboard uses `Path.exists()`.

- [ ] **Step 3: Merge artifact description and task state**

Expand `ReportArtifact` and delete `ReportArtifactSpec`:

```python
@dataclass(frozen=True)
class ReportArtifact:
    key: str
    public_key: str
    label: str
    path: Path
    dashboard_label: str
    dashboard_purpose: str
    job: Callable[[], None] | None = None
```

Build these objects once in `_build_report_artifacts`. Derive the public
artifact manifest, log labels, dashboard rows, and output-index rows from the
same tuple.

- [ ] **Step 4: Pass successful outcomes into the index and dashboard**

Create the successful-key set once:

```python
successful_keys = frozenset(outcome.key for outcome in outcomes if outcome.succeeded)
available = tuple(artifact for artifact in artifacts if artifact.key in successful_keys)
```

Pass `available` to `generate_output_index_report` and
`_print_reports_dashboard`. Keep log and environment rows conditional on their
current run production, not mere prior existence. While `--open-report` still
exists, open HTML only when `"html" in successful_keys`.

- [ ] **Step 5: Remove duplicated artifact maps and update tests**

Delete `_build_report_artifact_specs` and any jobs dictionary keyed separately
from the artifact descriptions. Update tests to inspect `_build_report_artifacts`
and outcome-driven navigation.

- [ ] **Step 6: Run report and finalisation tests**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_metrics_modes.py -q
```

Expected: PASS.

- [ ] **Step 7: Run the complete robustness-phase gate**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make format
make -C src lint-fix
make lint
bash src/tools/run_commit_hygiene.sh
make quality
```

Expected: every formatter, lint, type, static-analysis, test, shell and
Markdown check passes.

- [ ] **Step 8: Commit the artifact contract**

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_metrics_modes.py
git commit -m "refactor: drive artifact navigation from outcomes"
```

---

## Phase 2: Version 0.16 subtractive contract

### Task 5: Replace results plus run JSON with one schema-3 retained run

**Files:**

- Modify: `src/check_models.py:980-1305, 17770-19520, 20340-21580`
- Modify: `src/tests/test_jsonl_output.py`
- Modify: `src/tests/test_report_generation.py`
- Modify: `src/tests/test_dependency_sync.py`

**Interfaces:**

- Produces: `RetainedRun(metadata: JsonlMetadataRecord, results: tuple[JsonlResultRecord, ...])`.
- Produces: `_build_retained_run(inputs: ReportGenerationInputs, *, comparison: RunComparison | None = None) -> RetainedRun`.
- Produces: `_write_retained_run(run: RetainedRun, filename: Path) -> None`.
- Produces: `_load_retained_run_text(text: str, label: str) -> RetainedRun` and `_load_retained_run(path: Path) -> RetainedRun`.
- Changes: `JSONL_FORMAT_VERSION` and `JsonlMetadataRecord.format_version` to literal `"3.0"`.
- Removes: `RunJsonReportRecord`, `RunImageRecord` only if absorbed directly, `RunPromptBurdenRecord`, `save_run_json_report`, `_load_run_issue_enrichment`, `_narrow_run_issue_enrichment`, and `ReportOutputPaths.run_json`.

- [ ] **Step 1: Add a schema-3 metadata contract test**

Replace the separate JSONL-header and run-JSON tests with one assertion:

```python
def test_schema_3_metadata_contains_complete_run_context(tmp_path: Path) -> None:
    retained = _retained_run_fixture(tmp_path)
    path = tmp_path / "results.jsonl"
    check_models._write_retained_run(retained, path)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    header = rows[0]
    assert header["format_version"] == "3.0"
    assert header["prompt_sha256"] == hashlib.sha256(
        header["prompt"].encode("utf-8")
    ).hexdigest()
    assert header["counts"]["models_attempted"] == len(rows) - 1
    assert header["producer"]["name"] == "check_models"
    assert header["image"]["sha256"]
    assert header["generation_settings"]
    assert "comparison" in header
    assert "run.json" not in json.dumps(header)
```

The fixture must use `tmp_path` and a synthetic `PerformanceResult`; it must not
read or write tracked output.

- [ ] **Step 2: Add one-file run-summary and baseline tests**

Update `_write_issue_summary_fixture` to put image, generation settings,
producer, trust, runtime, counts, artifacts, and comparison in the JSONL
metadata row. Add assertions that deleting any sibling `run.json` has no effect
on run-summary regeneration or baseline comparability.

- [ ] **Step 3: Run the new schema tests and confirm the missing fields**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_jsonl_output.py -k "schema_3 or complete_run_context" -q
pytest src/tests/test_report_generation.py -k "one_file or baseline" -q
```

Expected: FAIL because JSONL is schema 2 and enrichment still comes from
`run.json`.

- [ ] **Step 4: Define the schema-3 metadata and retained-run value**

Extend `JsonlMetadataRecord` with these exact keys and types:

```python
class JsonlMetadataRecord(TypedDict, total=False):
    _type: Required[Literal["metadata"]]
    format_version: Required[Literal["3.0"]]
    prompt: Required[str]
    prompt_sha256: Required[str]
    system: Required[dict[str, str]]
    timestamp: Required[str]
    total_runtime_seconds: Required[float]
    counts: Required[RunOutcomeCounts]
    artifacts: Required[dict[str, str]]
    producer: Required[CheckModelsProvenanceRecord]
    image: Required[RunImageRecord | None]
    generation_settings: Required[dict[str, JsonLike]]
    trust_remote_code: Required[bool]
    comparison: Required[dict[str, JsonLike] | None]
    cache_discovery: NotRequired[list[CacheDiscoveryEntryRecord]]
    library_versions: LibraryVersionDict
    runtime_fingerprint: dict[str, RuntimeProbeResult]
    eval_mode: EvaluationLane
    metadata_exposed_to_prompt: bool
    component_provenance: dict[str, ComponentProvenanceRecord]
    execution_mode: Literal["in_process", "isolated"]
```

Add:

```python
@dataclass(frozen=True)
class RetainedRun:
    metadata: JsonlMetadataRecord
    results: tuple[JsonlResultRecord, ...]
```

Keep prompt-burden and model-provenance facts on each result row; delete the
duplicated metadata maps.

- [ ] **Step 5: Build, validate and serialize one retained value**

Make `_build_retained_run` reuse `_build_jsonl_result_record` and one metadata
builder. Make `_write_retained_run` emit exactly one metadata line followed by
ordered result lines:

```python
def _write_retained_run(run: RetainedRun, filename: Path) -> None:
    lines = [json.dumps(run.metadata)]
    lines.extend(json.dumps(result) for result in run.results)
    _write_text_file(filename, "\n".join(lines) + "\n")
```

Make one loader own line-aware JSON decoding, schema validation, assessment
vocabulary validation, failure validation, model-provenance consistency, and
metadata enrichment narrowing. Reuse that loader from run-summary regeneration
and comparison baseline resolution.

- [ ] **Step 6: Compute comparison before the single write**

Build current result records in memory, resolve the retained baseline from git,
and call `compare_run_results` with the current `RetainedRun.results`. Add the
serialized comparison to metadata and write `results.jsonl` once. A schema-2
baseline must log an incompatible-baseline reason and return no comparison; do
not add a schema-2 adapter.

- [ ] **Step 7: Delete run JSON and every cross-file join**

Delete:

```text
RUN_JSON_SCHEMA_VERSION
DEFAULT_RUN_JSON_OUTPUT
RunJsonReportRecord
RunPromptBurdenRecord
save_run_json_report
_run_prompt_burden_records
_load_run_issue_enrichment
_narrow_run_issue_enrichment
--output-run-json
ReportOutputPaths.run_json
include_run_json parameters and branches
```

Remove `run.json` from the artifact registry, output index, dashboard, run
summary, public exports, docs, and tests. Keep the per-result `model_burden`
payload.

- [ ] **Step 8: Run all machine-artifact and report tests**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_jsonl_output.py src/tests/test_report_generation.py src/tests/test_dependency_sync.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit schema 3**

```bash
git add src/check_models.py src/tests/test_jsonl_output.py src/tests/test_report_generation.py src/tests/test_dependency_sync.py
git commit -m "refactor!: make JSONL the sole machine artifact"
```

### Task 6: Replace seven output flags with one output root

**Files:**

- Modify: `src/check_models.py:1365-1380, 20560-20600, 21130-21148, 22110-22165`
- Modify: `src/tests/test_parameter_validation.py`
- Modify: `src/tests/test_cli_help_output.py`
- Modify: `src/tests/test_cli_integration.py`
- Modify: `src/tests/test_e2e_smoke.py`
- Modify: `src/tests/test_error_message_consistency.py`
- Modify: `src/tests/test_invalid_arguments.py`
- Modify: `src/tests/test_edge_case_images.py`
- Modify: `src/tests/test_version_env_reporting.py`
- Modify: `src/tests/test_metrics_modes.py`

**Interfaces:**

- Produces: `ReportOutputPaths.from_root(root: Path) -> ReportOutputPaths`.
- Produces: CLI option `--output-dir PATH`, defaulting to `src/output`.
- Removes: `OutputPathArgumentSpec`, `OUTPUT_PATH_ARGUMENT_SPECS`, and all seven `--output-*` options.

- [ ] **Step 1: Add output-root parser and layout tests**

Add to `test_parameter_validation.py`:

```python
def test_output_dir_derives_the_canonical_layout(tmp_path: Path) -> None:
    parser = check_models._build_cli_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    paths = check_models._resolve_report_output_paths(args)

    assert paths.index == tmp_path / "index.md"
    assert paths.jsonl == tmp_path / "results.jsonl"
    assert paths.html == tmp_path / "reports" / "results.html"
    assert paths.gallery_markdown == tmp_path / "reports" / "model_gallery.md"
    assert paths.diagnostics == tmp_path / "reports" / "diagnostics.md"
    assert paths.log == tmp_path / "check_models.log"
    assert paths.environment == tmp_path / "environment.log"
```

Add a help test requiring `--output-dir` and rejecting every former output
flag.

- [ ] **Step 2: Run parser tests and confirm the new option is absent**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_parameter_validation.py src/tests/test_cli_help_output.py -k "output" -q
```

Expected: FAIL because only individual artifact flags exist.

- [ ] **Step 3: Add `ReportOutputPaths.from_root`**

Implement:

```python
@classmethod
def from_root(cls, root: Path) -> Self:
    root = root.expanduser().resolve()
    reports = root / "reports"
    return cls(
        index=root / DEFAULT_OUTPUT_INDEX.name,
        html=reports / DEFAULT_HTML_OUTPUT.name,
        gallery_markdown=reports / DEFAULT_GALLERY_MD_OUTPUT.name,
        jsonl=root / DEFAULT_JSONL_OUTPUT.name,
        diagnostics=reports / DEFAULT_DIAGNOSTICS_OUTPUT.name,
        log=root / DEFAULT_LOG_OUTPUT.name,
        environment=root / DEFAULT_ENV_OUTPUT.name,
    )
```

Make `_resolve_report_output_paths` call this classmethod. Make
`regenerate_run_issue_summary(output_dir)` use the same classmethod.

- [ ] **Step 4: Replace the argparse output table**

Delete `OutputPathArgumentSpec`, `OUTPUT_PATH_ARGUMENT_SPECS`, and
`_add_output_path_arguments`. Add to the output group:

```python
output_group.add_argument(
    "--output-dir",
    type=Path,
    default=_SCRIPT_DIR / "output",
    help="Root directory for the canonical report, machine-data, issue, and log layout.",
)
```

Keep `--compare-with` and `--link-style` in their existing semantic groups.

- [ ] **Step 5: Shorten test CLI construction**

Replace every seven-flag temporary-output argument list with:

```python
["--output-dir", str(output_dir)]
```

Replace repeated `Namespace(output_html=..., output_jsonl=..., ...)`
construction with `Namespace(output_dir=output_dir)` or direct
`ReportOutputPaths.from_root(output_dir)` according to what the test exercises.

- [ ] **Step 6: Run CLI and e2e tests**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_parameter_validation.py src/tests/test_cli_help_output.py src/tests/test_cli_integration.py src/tests/test_e2e_smoke.py src/tests/test_error_message_consistency.py src/tests/test_invalid_arguments.py src/tests/test_edge_case_images.py src/tests/test_version_env_reporting.py -q
```

Expected: PASS; cached-model e2e cases may retain their existing documented
skip when the fixture model is unavailable.

- [ ] **Step 7: Commit the output-root contract**

```bash
git add src/check_models.py src/tests/test_parameter_validation.py src/tests/test_cli_help_output.py src/tests/test_cli_integration.py src/tests/test_e2e_smoke.py src/tests/test_error_message_consistency.py src/tests/test_invalid_arguments.py src/tests/test_edge_case_images.py src/tests/test_version_env_reporting.py src/tests/test_metrics_modes.py
git commit -m "refactor!: replace output flags with one output root"
```

### Task 7: Remove legacy evaluation and presentation flags

**Files:**

- Modify: `src/check_models.py:860-870, 1690-1700, 11530-11548, 15075-15480, 15770-15835, 21760-21858, 22350-22655`
- Modify: `src/tests/test_parameter_validation.py`
- Modify: `src/tests/test_cli_help_output.py`
- Modify: `src/tests/test_metrics_modes.py`

**Interfaces:**

- Keeps evaluation inputs: `auto`, `triage`, `blind`, `assisted`.
- Removes evaluation inputs: `stress`, `quality`.
- Removes: `QUALITY_MAX_TOKENS`, `--open-report`, `webbrowser`, and `--detailed-metrics`.
- Changes: `verbose=True` always renders the existing detailed metric tree and detailed legend.

- [ ] **Step 1: Add the reduced CLI contract tests**

Add:

```python
@pytest.mark.parametrize("retired_mode", ["stress", "quality"])
def test_retired_eval_modes_are_rejected(retired_mode: str) -> None:
    parser = check_models._build_cli_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--eval-mode", retired_mode])
    assert exc_info.value.code == 2
```

Update help assertions so `--open-report` and `--detailed-metrics` are absent.
Add a metrics test calling `print_model_result(result, verbose=True)` and
asserting phase timings and stop reason appear without a second flag.

- [ ] **Step 2: Run the contract tests and confirm compatibility remains**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_parameter_validation.py src/tests/test_cli_help_output.py src/tests/test_metrics_modes.py -k "retired or verbose or detailed" -q
```

Expected: FAIL because aliases and both presentation flags still exist.

- [ ] **Step 3: Delete legacy evaluation branches**

Narrow the requested type and parser choices:

```python
type RequestedEvaluationMode = EvaluationLane | Literal["auto"]
```

Delete legacy warnings and the quality-specific token default. `_resolve_eval_mode`
must map only `auto` through metadata and return concrete lanes unchanged.

- [ ] **Step 4: Make verbose metrics unambiguous**

Remove the `detailed_metrics` argument from `print_model_result` and
`log_metrics_legend`. In verbose paths, call the existing detailed renderer
unconditionally:

```python
if result.generation and verbose:
    _log_verbose_success_details_mode(
        result,
        detailed=True,
        analysis=result.quality_analysis,
        prompt=prompt,
        context_marker=context_marker,
    )
```

Then inline or rename `_log_verbose_success_details_mode` if `detailed` becomes
constant and delete the now-dead compact-verbose branch.

- [ ] **Step 5: Remove browser-opening support**

Delete the `webbrowser` import, parser option, and finalisation branch. Retain
the Rich dashboard's clickable file links as the sole open/navigation feature.

- [ ] **Step 6: Run CLI and metrics tests**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_parameter_validation.py src/tests/test_cli_help_output.py src/tests/test_metrics_modes.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit compatibility deletion**

```bash
git add src/check_models.py src/tests/test_parameter_validation.py src/tests/test_cli_help_output.py src/tests/test_metrics_modes.py
git commit -m "refactor!: remove retired CLI compatibility"
```

### Task 8: Prune historical negative tests without weakening live contracts

**Files:**

- Modify: `src/tests/test_dependency_sync.py:940-1085`
- Modify: `src/tests/test_report_generation.py:2040-2120, 3500-3550, 3680-3980`

**Interfaces:**

- Removes tests for retired implementation names and duplicated absence checks.
- Preserves one current assessment-vocabulary test across JSONL, Markdown, HTML, diagnostics, and logs.
- Preserves all security, escaping, path, malformed-input, isolation, discovery, native-parity, thinking, reproduction, and evidence-round-trip tests.

- [ ] **Step 1: Record the pre-prune test count**

```bash
rg -n '^def test_|^    def test_' src/tests | wc -l
```

Record the count in the commit message body, not in production documentation.

- [ ] **Step 2: Delete the retired semantic identifier blacklist**

Delete `test_production_source_has_no_retired_semantic_scoring_api`. Current
typed schemas, positive render tests, Ruff, Vulture, and Skylos replace its
list of historical symbol names.

- [ ] **Step 3: Consolidate source-string and README archaeology**

Delete `test_production_logs_use_facts_first_observation_labels` and
`test_public_readme_uses_only_exported_report_api`. Add current public API and
status assertions to the existing positive CLI/help/report tests instead of
searching for removed prose.

- [ ] **Step 4: Collapse overlapping HTML absence tests**

Delete the standalone tests whose distinct purpose is only to assert absence
of retired semantic projections:

```text
test_html_ignores_legacy_semantic_winners
test_html_contains_gallery_and_diagnostics_without_semantic_scores
test_triage_html_report_suppresses_cataloging_scores
test_retained_artifacts_have_no_owner_confidence_path
```

Keep `test_standalone_html_does_not_build_legacy_semantic_context`; it protects
the live no-reclassification seam used when callers supply a report context.
Add one compact assertion to the cross-artifact assessment test:

```python
retired_terms = ("quality score", "semantic winner", "owner_confidence", "suspected_owner")
for artifact_text in (jsonl_text, gallery, html_report, diagnostics):
    lowered = artifact_text.casefold()
    assert all(term not in lowered for term in retired_terms)
```

- [ ] **Step 5: Retain the nanobind policy regression**

Do not delete `test_no_ci_only_mlx_core_stub_generation`; it guards a current
dependency and CI policy established after a real stub-generation failure.

- [ ] **Step 6: Run the affected suites and compare test counts**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_dependency_sync.py src/tests/test_report_generation.py -q
rg -n '^def test_|^    def test_' src/tests | wc -l
```

Expected: PASS, with fewer tests and no reduction in the listed live-contract
categories.

- [ ] **Step 7: Commit test pruning**

```bash
git add src/tests/test_dependency_sync.py src/tests/test_report_generation.py
git commit -m "test: remove retired reporting archaeology"
```

### Task 9: Document, version, verify and archive

**Files:**

- Modify: `src/README.md`
- Modify: `docs/IMPLEMENTATION_GUIDE.md`
- Modify: `.github/copilot-instructions.md`
- Modify: `src/pyproject.toml`
- Modify: `CHANGELOG.md`
- Move: `docs/superpowers/plans/2026-08-29-robustness-and-subtractive-simplification.md`
- Move: `docs/superpowers/specs/2026-08-29-robustness-and-subtractive-simplification-design.md`

**Interfaces:**

- Publishes: version `0.16.0`.
- Documents: schema-3 `results.jsonl`, canonical output layout, removed flags and aliases, verbose metrics, fail-soft report outcomes, and one-time baseline incompatibility.
- Archives: the completed plan and design under `docs/notes/archive/superpowers/`.

- [ ] **Step 1: Update public CLI and artifact documentation**

In `src/README.md`:

- replace seven output-path rows and examples with `--output-dir`;
- remove `run.json` from every artifact list and diagram;
- describe schema-3 metadata plus per-result rows;
- remove `stress`, `quality`, `--open-report`, and `--detailed-metrics`;
- state that verbose mode includes detailed phase timings;
- state that the first comparison against a schema-2 retained run is skipped
  and the next schema-3 run establishes the new baseline; and
- update the report-only regeneration example to require only the output root
  containing `results.jsonl`.

- [ ] **Step 2: Update maintainer architecture instructions**

In `docs/IMPLEMENTATION_GUIDE.md` and `.github/copilot-instructions.md`, describe
`results.jsonl` as the sole current-run machine contract, `ReportArtifactOutcome`
as the source for current navigation, and `--output-dir` as the only output
location control. Update the monolith line count after formatting.

- [ ] **Step 3: Record the release in the changelog and package metadata**

Set:

```toml
version = "0.16.0"
```

Under `[Unreleased]`, record:

- canonical standalone/main observation semantics;
- ordinary report/comparison exception containment;
- stale-artifact navigation prevention;
- range-correct constraint aggregation;
- schema-3 JSONL replacing run JSON;
- `--output-dir` replacing individual output paths;
- removed aliases and presentation flags; and
- subtractive test cleanup.

- [ ] **Step 4: Run dependency and documentation synchronisation**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make deps-sync
```

Expected: README dependency blocks remain synchronized; no dependency changes
are introduced.

- [ ] **Step 5: Render a representative report set without a model run**

Use the existing report-generation fixtures through pytest:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py -k "regenerate_run_issue_summary or output_index or artifact" -q
pytest src/tests/test_jsonl_output.py -q
```

Expected: PASS, with every generated test artifact under `tmp_path`.

- [ ] **Step 6: Run prescribed formatting and lint preparation**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make format
make -C src lint-fix
make lint
```

Expected: PASS. If Ruff advertises an unsafe fix that is faster than manual
correction, inspect its diff before applying it and reject any semantic change
that is not plainly correct.

- [ ] **Step 7: Run commit hygiene and the complete quality gate**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
bash src/tools/run_commit_hygiene.sh
make quality
```

Expected: all formatting, lint, mypy, ty, pyrefly, Vulture, Skylos,
suppression-audit, pytest, shellcheck, and markdownlint checks pass.

- [ ] **Step 8: Inspect final size, status and diff hygiene**

```bash
wc -l src/check_models.py src/tests/test_*.py
git diff --check
git status --short
git diff --stat
```

Expected: `git diff --check` emits nothing; tracked `src/output/` artifacts are
unchanged; production plus test code shows material net deletion across the
complete phase even if schema-3 tests add some lines.

- [ ] **Step 9: Archive the completed planning documents**

```bash
mkdir -p docs/notes/archive/superpowers/plans docs/notes/archive/superpowers/specs
git mv docs/superpowers/plans/2026-08-29-robustness-and-subtractive-simplification.md docs/notes/archive/superpowers/plans/
git mv docs/superpowers/specs/2026-08-29-robustness-and-subtractive-simplification-design.md docs/notes/archive/superpowers/specs/
```

- [ ] **Step 10: Commit the release documentation and archive**

```bash
git add CHANGELOG.md src/README.md docs/IMPLEMENTATION_GUIDE.md .github/copilot-instructions.md src/pyproject.toml docs/notes/archive/superpowers/plans/2026-08-29-robustness-and-subtractive-simplification.md docs/notes/archive/superpowers/specs/2026-08-29-robustness-and-subtractive-simplification-design.md
git commit -m "release: prepare version 0.16.0"
```

## Execution Checkpoints

- After Task 4: review the backwards-compatible robustness phase and its full
  quality-gate result before beginning schema or CLI deletion.
- After Task 5: review the exact schema-3 JSONL fixture and confirm the one-time
  schema-2 baseline loss is acceptable before deleting output flags.
- After Task 7: inspect `check_models --help` and one temporary output tree
  before pruning tests.
- After Task 9: do not rerun the real model matrix solely for this refactor;
  allow the next scheduled/user-run sweep to replace tracked schema-2 outputs.
