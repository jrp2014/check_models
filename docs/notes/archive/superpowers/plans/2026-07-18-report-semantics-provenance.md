# Report Semantics and Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct maintainer issue readiness, make recommendation semantics consistent, repair HTML/TSV output, and add local component/model provenance without changing multi-image behaviour.

**Architecture:** Extend the existing `JsonlReviewRecord`, maintainer-triage payload, `ModelRecommendationView`, diagnostics snapshot, and report context rather than adding a parallel classifier. Keep `src/check_models.py` as the project monolith; report builders consume cached canonical facts, while shared local-only provenance collectors feed JSONL, run metadata, diagnostics, and repro bundles.

**Tech Stack:** Python 3.13, dataclasses and TypedDict, pytest, `csv`, importlib metadata, local Git/Hugging Face cache inspection, Markdown/HTML/TSV report generation.

---

## File Map

- Modify `src/check_models.py`: canonical statuses, issue readiness, report renderers, TSV writer, and provenance collection.
- Modify `src/tests/test_quality_analysis.py`: presentation-warning recommendation behaviour.
- Modify `src/tests/test_report_generation.py`: cross-report semantics and observation/issue-draft classification.
- Modify `src/tests/test_html_formatting.py`: numeric sorting and accessibility contracts.
- Modify `src/tests/test_tsv_output.py`: literal compact adaptive TSV contract.
- Modify `src/tests/test_jsonl_output.py`: component and model provenance contracts.
- Modify `src/README.md`: document statuses, observation readiness, TSV, and provenance.
- Modify `CHANGELOG.md`: record the maintainer-relevant behaviour change.

### Task 1: Make Review Status Canonical and Presentation-Aware

**Files:**

- Modify: `src/check_models.py` near `JsonlReviewRecord`, `_classify_user_bucket`, `_recommendation_eligibility`, `_model_recommendation_status`, and `_generate_model_gallery_section`
- Test: `src/tests/test_quality_analysis.py`
- Test: `src/tests/test_report_generation.py`

- [ ] **Step 1: Write failing tests for canonical recommendation semantics**

Add focused tests proving that execution success is independent of recommendation status and that presentation warnings are never recommended:

```python
def test_complete_visible_thinking_trace_is_a_user_caveat() -> None:
    analysis = check_models.analyze_generation_text(
        "<think>inspect the scene</think> A harbour at sunset.",
        generated_tokens=16,
        prompt="Describe the image.",
        requested_max_tokens=128,
        model_name="org/thinking-model",
    )

    assert analysis.has_thinking_trace
    assert not analysis.thinking_trace_incomplete
    assert analysis.user_bucket == "caveat"


def test_token_cap_is_never_presentation_ready() -> None:
    analysis = check_models.analyze_generation_text(
        "A useful but capped description of the scene.",
        generated_tokens=32,
        prompt="Describe the image.",
        requested_max_tokens=32,
    )

    assert analysis.likely_capped
    assert analysis.user_bucket == "caveat"
```

In `test_report_generation.py`, construct successful `PerformanceResult` rows for `recommended`, `caveat`, and `avoid`, then assert gallery heading icons and model-selection status cells use those buckets rather than `success`.

- [ ] **Step 2: Run the focused tests and confirm the old behaviour fails**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_quality_analysis.py -k "thinking_trace or token_cap" -q'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py -k "recommendation_status or gallery_icon" -q'
```

Expected: at least the thinking-trace/token-cap bucket assertions and gallery icon assertion fail because the current code can emit `recommended` or choose `✅` from `success=True`.

- [ ] **Step 3: Add precise status types and one shared execution-outcome helper**

Add the following aliases and helper near the report/review types:

```python
type ExecutionOutcome = Literal["completed", "failed", "indeterminate"]
type RecommendationStatus = Literal["recommended", "caveat", "avoid", "not-evaluated"]


def _execution_outcome(result: PerformanceResult) -> ExecutionOutcome:
    """Return whether this run completed, failed conclusively, or was indeterminate."""
    if _is_indeterminate_connectivity_failure(result):
        return "indeterminate"
    return "completed" if result.success else "failed"
```

Annotate `JsonlReviewRecord.user_bucket`, `ModelRecommendationView` status access, and status-returning helpers with `RecommendationStatus` where the current TypedDict/dataclass layout permits it. Do not add casts or checker suppressions to hide imprecise values.

- [ ] **Step 4: Make `_classify_user_bucket` enforce presentation readiness**

Extend `_classify_user_bucket` with a `has_presentation_warning` parameter and use a single ordered mapping:

```python
def _classify_user_bucket(
    *,
    verdict: str,
    hint_relationship: str,
    has_contract_issue: bool,
    utility_grade: str = "",
    has_presentation_warning: bool = False,
) -> RecommendationStatus:
    if verdict in {"harness", "cutoff_degraded", "runtime_failure"}:
        return "avoid"
    if verdict == "unknown_runtime_anomaly":
        return "caveat"
    if verdict in {"token_cap", "context_budget"} or has_presentation_warning:
        return "caveat"
    if (
        verdict in {"model_shortcoming", "semantic_mismatch"}
        or hint_relationship == "degrades_trusted_hints"
    ):
        return "avoid"
    if (
        verdict == "clean"
        and hint_relationship
        in {"improves_trusted_hints", "preserves_trusted_hints", "not_evaluated"}
        and not has_contract_issue
    ):
        return "recommended"
    return "caveat"
```

At the `analyze_generation_text` call site, compute the argument from visible thinking/reasoning/control/formatting evidence already calculated for the same analysis. A complete trace remains informational for owner attribution but is a presentation caveat. Change the token-cap mapping from potentially `recommended` to always `caveat`.

- [ ] **Step 5: Derive eligibility and every report label from the review bucket**

Change `_recommendation_eligibility` so a completed result is eligible only when `review["user_bucket"] == "recommended"`. Preserve compatibility status as a separate fact. Add one icon helper and reuse it in the gallery:

```python
def _recommendation_icon(status: RecommendationStatus) -> str:
    return {
        "recommended": "✅",
        "caveat": "⚠️",
        "avoid": "❌",
        "not-evaluated": "❔",
    }[status]
```

Make `_model_recommendation_status`, gallery headings, shortlist filters, review groupings, history current-status derivation, and HTML/TSV status cells read the cached review bucket. Keep `clean-triage-pass` as a display label returned by `_review_display_bucket_label`; do not store it as a fifth recommendation value.

- [ ] **Step 6: Run focused tests and commit the semantic change**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_quality_analysis.py src/tests/test_report_generation.py -k "thinking_trace or token_cap or recommendation_status or gallery_icon or model_selection" -q'
```

Expected: selected tests pass.

Commit:

```bash
git add src/check_models.py src/tests/test_quality_analysis.py src/tests/test_report_generation.py
git commit -m "fix: unify report recommendation semantics"
```

### Task 2: Separate Confirmed Maintainer Issues from Reproduction Observations

**Files:**

- Modify: `src/check_models.py` near `PromptDiagnostics`, `JsonlMaintainerTriageRecord`, `DiagnosticsSnapshot`, `_build_jsonl_maintainer_triage_record`, `_build_diagnostics_snapshot`, `_build_issue_clusters`, and diagnostics renderers
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_jsonl_output.py`

- [ ] **Step 1: Write failing thinking/template and context-boundary regression tests**

Add result factories with prompt diagnostics matching the captured run. The thinking case must use `generate_kwargs` with no enabled thinking flag, a rendered prompt ending in `<think>`, a closed trace, `skip_special_tokens=False`, and a usable final answer. Assert:

```python
triage = check_models._maintainer_triage_for_result(result)
assert triage is not None
assert triage["issue_readiness"] == "needs-reproduction"
assert triage["issue_subtype"] == "thinking_configuration"
assert triage["confidence"] == "medium"
assert "stop_token" not in triage["issue_subtype"]
```

Build a context case with 4,103 prompt tokens, 4,097 non-text tokens, visual-input burden, and output `Cat.`. Assert it remains a `context_budget`/`caveat` observation, has a controlled reduced-image next action, and produces no `IssueCluster` or issue draft.

Generate diagnostics under `tmp_path` and assert both observations retain complete output and decisive evidence. Do not inspect or rewrite `src/output/`.

- [ ] **Step 2: Run the classification tests and confirm issue clustering currently fails**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py -k "thinking_configuration or context_boundary or issue_readiness" -q'
```

Expected: failures because triage has no readiness field and `_build_issue_clusters` currently converts successful harness/context signals directly into issue drafts.

- [ ] **Step 3: Add issue-readiness types and cached observation partitioning**

Add:

```python
type IssueReadiness = Literal["confirmed", "needs-reproduction", "not-evaluated"]


class MaintainerObservation(TypedDict):
    result: PerformanceResult
    subtype: str
    owner: str
    summary: str
    next_action: str
```

Use a frozen dataclass instead of a TypedDict if forward references or checker precision make it clearer. Extend `JsonlMaintainerTriageRecord` with required `issue_readiness`. Extend `DiagnosticsSnapshot` with an immutable `observations` collection, defaulting empty so existing test construction stays concise.

Implement narrow helpers:

```python
def _has_template_opened_thinking(result: PerformanceResult) -> bool:
    diagnostics = result.prompt_diagnostics
    if diagnostics is None:
        return False
    thinking_enabled = diagnostics.generate_kwargs.get("enable_thinking") is True
    preview = diagnostics.rendered_prompt_preview or ""
    return not thinking_enabled and any(preview.rstrip().endswith(start) for start, _ in THINKING_TRACE_DELIMITER_PAIRS)


def _issue_readiness_for_result(
    result: PerformanceResult,
    review: JsonlReviewRecord,
    analysis: GenerationQualityAnalysis | None,
) -> IssueReadiness:
    if _is_indeterminate_connectivity_failure(result):
        return "not-evaluated"
    if not result.success:
        return "confirmed"
    if _has_template_opened_thinking(result):
        return "needs-reproduction"
    if review["verdict"] == "context_budget":
        return "needs-reproduction"
    return "confirmed" if analysis is not None and analysis.has_harness_issue else "needs-reproduction"
```

Keep text-sanity/model-quality observations out of upstream library issue readiness unless existing direct evidence already makes them confirmed.

- [ ] **Step 4: Correct subtype, owner, confidence, and next actions**

When `_has_template_opened_thinking` is true, override generic token-leak routing with:

```python
issue_subtype = "thinking_configuration"
suspected_owner = "model-config / mlx-vlm"
confidence: MaintainerConfidence = "medium"
next_action = (
    "Re-run with the rendered template and effective thinking kwargs captured; compare an "
    "explicit thinking-disabled template before filing a stop-token issue."
)
```

When a successful result has `context_budget`, use subtype `context_boundary`, medium or low confidence according to available prompt-burden scalars, owner `model-config / mlx-vlm / mlx`, and a next action requiring a reduced-image/lower-visual-token control. Do not special-case model identifiers.

- [ ] **Step 5: Partition observations before building issue clusters**

In `_build_diagnostics_snapshot`, cache maintainer triage for each successful diagnostic result, move `needs-reproduction` rows into `snapshot.observations`, and leave only confirmed rows in the inputs that `_build_issue_clusters` turns into issue-ready drafts. Render an `Observations Requiring Controlled Reproduction` section in `diagnostics.md` using existing `ReportSection`, `ReportDetails`, `ReportCodeBlock`, prompt-diagnostic, and complete-output helpers.

Attach `issue_cluster_id/path/acceptance_signal` only when `issue_readiness == "confirmed"`. Preserve `issue_readiness`, subtype, owner, confidence, next action, and evidence in JSONL for observations.

- [ ] **Step 6: Run focused report/JSONL tests and commit**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py -k "thinking_configuration or context_boundary or issue_readiness or observation" -q'
```

Expected: selected tests pass; no issue files are written for the two observation fixtures.

Commit:

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py
git commit -m "fix: gate maintainer issues on reproducible evidence"
```

### Task 3: Repair HTML Sorting, Filters, and Accessibility

**Files:**

- Modify: `src/check_models.py` near `NUMERIC_FIELD_PATTERNS`, `is_numeric_field`, `_html_result_row_attrs`, `_build_html_results_table`, and `_build_full_html_document`
- Test: `src/tests/test_html_formatting.py`
- Test: `src/tests/test_report_generation.py`

- [ ] **Step 1: Write failing HTML defect tests**

Add tests that render a row with peak memory `1.00 GB` and error package `mlx-vlm`. Parse or inspect the HTML and assert:

```python
assert 'data-sort-value="1"' in html_text
assert 'data-sort-value="1.00.883108eeee"' not in html_text
assert not check_models.is_numeric_field("error_package")
assert "<caption>Per-model execution, quality, and performance results</caption>" in html_text
assert 'scope="col"' in html_text
assert 'aria-sort="none"' in html_text
assert 'role="status"' in html_text
assert 'aria-live="polite"' in html_text
assert 'data-recommendation="caveat"' in html_text
```

Also assert the filter options are `recommended`, `caveat`, `avoid`, and `not-evaluated`, not `eligible=true/false`.

- [ ] **Step 2: Run HTML tests and confirm current defects**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_html_formatting.py src/tests/test_report_generation.py -k "html and (sort or accessibility or recommendation_filter)" -q'
```

Expected: the malformed sort key, numeric error package, and missing accessibility assertions fail.

- [ ] **Step 3: Fix typed numeric classification and sort keys**

Remove `error_package` from `NUMERIC_FIELD_PATTERNS` and replace the broad substring fallback in `is_numeric_field` with explicit known metric suffixes/fields so textual names containing `time`, `token`, or `memory` cannot become numeric accidentally.

Add an anchored display parser:

```python
_HTML_LEADING_NUMBER_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _html_sort_value(value: str, *, numeric: bool) -> str:
    if not numeric:
        return value.casefold()
    match = _HTML_LEADING_NUMBER_RE.match(value.replace(",", ""))
    return match.group(1) if match is not None else ""
```

Use this helper in `_build_html_results_table` instead of stripping every non-numeric character from the full display string.

- [ ] **Step 4: Add accessible table/filter semantics and canonical recommendation metadata**

Start the table with the caption. Put `scope="col" aria-sort="none"` on each `th`. Add `data-recommendation` from the canonical review bucket in `_html_result_row_attrs`; keep `data-status` for execution outcome. Replace the eligibility filter with a recommendation filter, mark the filter-count element `role="status" aria-live="polite"`, and update `sortResults` to reset all headers to `none` and set the active header to `ascending` or `descending`.

- [ ] **Step 5: Run HTML tests and commit**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_html_formatting.py src/tests/test_report_generation.py -k "html" -q'
```

Expected: selected tests pass.

Commit:

```bash
git add src/check_models.py src/tests/test_html_formatting.py src/tests/test_report_generation.py
git commit -m "fix: repair HTML report sorting and semantics"
```

### Task 4: Replace Aligned TSV with a Compact Literal TSV

**Files:**

- Modify: `src/check_models.py` imports, TSV constants, and `generate_tsv_report`
- Test: `src/tests/test_tsv_output.py`
- Test: `src/tests/test_report_generation.py`

- [ ] **Step 1: Write failing literal/adaptive TSV tests**

Use `csv.reader(..., delimiter="\t")` after skipping the metadata comment. Assert headers equal their stripped values, no cell contains alignment padding, and every record has the same length. Add a completed-only run where failure and optional image/score columns are all empty and assert they are absent. Add a second run with a populated optional value and assert that column appears.

Assert the core contains:

```python
expected_core = {
    "Model Name",
    "execution_status",
    "recommendation_status",
    "compatibility_status",
    "Prompt (tokens)",
    "Generation (tokens)",
    "Gen TPS",
    "Peak (GB)",
    "Total (s)",
    "prompt_burden_kind",
    "Generated Text",
}
assert expected_core <= set(headers)
```

Retain existing tests for newline/tab normalization, exact full generated output, and the absence of a duplicate preview.

- [ ] **Step 2: Run TSV tests and confirm tabulate padding/blank-column failures**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_tsv_output.py src/tests/test_report_generation.py -k "tsv" -q'
```

Expected: new stripped/adaptive/status assertions fail against the aligned `tabulate` output.

- [ ] **Step 3: Build rows as named fields, drop empty optional fields, and write with `csv.writer`**

Import `csv`. Keep a tuple of stable core field names and a tuple of optional field names. Build each TSV row as `dict[str, str]`, adding canonical execution/recommendation status from the same helpers used by HTML and gallery. Determine columns with:

```python
active_optional_fields = tuple(
    field_name
    for field_name in TSV_OPTIONAL_FIELDS
    if any(row.get(field_name, "") for row in row_records)
)
output_fields = (*TSV_CORE_FIELDS, *active_optional_fields)
```

Write the metadata comment, headers, and rows through:

```python
buffer = io.StringIO(newline="")
buffer.write(f"# generated_at: {local_now_str()}; format=compact-adaptive; exhaustive=results.jsonl\n")
writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
writer.writerow([header_by_field[field_name].strip() for field_name in output_fields])
writer.writerows(
    [row.get(field_name, "") for field_name in output_fields]
    for row in row_records
)
_write_text_file(filename, buffer.getvalue())
```

Continue replacing embedded tabs with spaces and physical newlines with the literal sequence `\n`. Apply `MAX_TSV_CELL_CHARS` only to compact diagnostic fields; never truncate `Generated Text`.

- [ ] **Step 4: Remove obsolete TSV alignment code and run tests**

Delete the `tabulate(..., tablefmt="tsv")` path and any now-single-use helpers/constants. Keep `tabulate` itself if console/Markdown paths still use it.

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_tsv_output.py src/tests/test_report_generation.py -k "tsv" -q'
```

Expected: all TSV tests pass.

- [ ] **Step 5: Commit the TSV change**

```bash
git add src/check_models.py src/tests/test_tsv_output.py src/tests/test_report_generation.py
git commit -m "fix: emit compact literal TSV reports"
```

### Task 5: Add Shared Local Component and Model Provenance

**Files:**

- Modify: `src/check_models.py` near metadata TypedDicts, distribution metadata helpers, `_resolve_model_snapshot_path`, JSONL/run/repro/diagnostics generation, and finalization inputs
- Test: `src/tests/test_jsonl_output.py`
- Test: `src/tests/test_report_generation.py`

- [ ] **Step 1: Write failing component/model provenance tests**

Mock `distribution`, `direct_url.json`, source directories, and local Git command output. Assert the collector returns publication-safe records:

```python
assert provenance["mlx-vlm"] == {
    "version": "0.6.4",
    "install_type": "editable",
    "source_location": "~/src/mlx-vlm",
    "source_revision": "abc123",
    "direct_url": "file://~/src/mlx-vlm",
    "vcs_revision": None,
}
```

Mock a local snapshot path ending in `snapshots/<sha>` and assert requested and resolved revisions remain distinct:

```python
assert model_provenance == {
    "model": "org/model",
    "requested_revision": "main",
    "resolved_revision": snapshot_sha,
    "snapshot_path": "~/.cache/huggingface/hub/models--org--model/snapshots/" + snapshot_sha,
}
```

Assert no URL opener, Hub download, or remote Git operation is invoked. Add integration assertions that JSONL metadata, `run.json`, diagnostics environment, and repro metadata contain the same component payload.

- [ ] **Step 2: Run provenance tests and confirm the fields are absent**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_jsonl_output.py src/tests/test_report_generation.py -k "provenance or resolved_revision or install_type" -q'
```

Expected: new component/model provenance assertions fail.

- [ ] **Step 3: Add typed provenance payloads and a publication-safe path helper**

Add TypedDicts:

```python
class ComponentProvenanceRecord(TypedDict):
    version: str | None
    install_type: Literal["editable", "wheel", "source-tree", "unknown"]
    source_location: str | None
    source_revision: str | None
    direct_url: str | None
    vcs_revision: str | None


class ModelProvenanceRecord(TypedDict):
    model: str
    requested_revision: str | None
    resolved_revision: str | None
    snapshot_path: str | None
```

Use `_home_relative_report_text` for all source/cache paths. Extend `JsonlMetadataRecord`, `JsonlResultRecord`, `HistoryRunRecord` only where the payload is actually emitted; do not add duplicate per-row component data.

- [ ] **Step 4: Extend existing distribution metadata helpers**

Parse the allowlisted `direct_url.json` once per distribution. Classify editable installs from `dir_info.editable`; wheels from installed metadata without editability; check_models source tree from `_REPO_ROOT`; otherwise unknown. Resolve a local `file:` URL to a path without contacting the network. Obtain a source revision only with the existing bounded local command helper:

```python
revision = _run_macos_toolchain_command(
    ("git", "-C", str(source_path), "rev-parse", "HEAD")
)
```

Do not run this command unless the source path exists locally. Preserve `vcs_info.commit_id` from direct-URL metadata separately. Collect records for `check_models`, `mlx`, `mlx-vlm`, `mlx-lm`, `transformers`, `tokenizers`, and `Pillow`, reusing `get_library_versions()`.

- [ ] **Step 5: Derive local model snapshot provenance**

Add:

```python
def _resolved_snapshot_revision(snapshot_path: Path | None) -> str | None:
    if snapshot_path is None:
        return None
    return snapshot_path.name if snapshot_path.parent.name == "snapshots" else None
```

Build model provenance from `_resolve_model_snapshot_path` and the requested revision already passed into model loading/run arguments. Cache it in report context or result data so all artifact generation uses the same selected snapshot and does not rescan differently.

- [ ] **Step 6: Thread the shared payload through artifacts**

Add component provenance to the JSONL metadata header and `run.json` (increment `schema_version` additively). Add per-result model provenance to JSONL. Reuse the same component/model records in diagnostic environment tables and repro `metadata.json`. Keep human tables compact: component, version/install type, and short revision; machine artifacts retain full normalized fields.

Ensure existing absolute system strings are passed through `_home_relative_report_text` before publication. Do not alter internal filesystem operations or raw exception objects.

- [ ] **Step 7: Run provenance tests and commit**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_jsonl_output.py src/tests/test_report_generation.py -k "provenance or revision or metadata_record or run_json or repro" -q'
```

Expected: selected tests pass without network access.

Commit:

```bash
git add src/check_models.py src/tests/test_jsonl_output.py src/tests/test_report_generation.py
git commit -m "feat: record component and model provenance"
```

### Task 6: Document the Contract and Run the Full Quality Gate

**Files:**

- Modify: `src/README.md`
- Modify: `CHANGELOG.md`
- Verify: all modified source/test files

- [ ] **Step 1: Update user and maintainer documentation**

In `src/README.md`, add concise sections/tables defining:

- execution outcome versus recommendation status;
- the four recommendation values and strict presentation-ready meaning of `recommended`;
- confirmed issue versus observation requiring reproduction;
- compact adaptive TSV and JSONL as the exhaustive fixed-schema artifact; and
- local-only component/model provenance, including `unknown` behaviour and home-relative paths.

State explicitly that the reports assess one current image/run and do not imply multi-image generalisation.

- [ ] **Step 2: Add an `[Unreleased]` changelog entry**

Record:

```markdown
- Correct maintainer triage so thinking-template and single-run visual
  context-boundary signals remain reproduction observations until isolated.
- Unify execution and recommendation semantics across reports, reserving
  `recommended` for presentation-ready output.
- Repair HTML numeric sorting/accessibility and emit compact literal TSV.
- Record publication-safe component install/source and local model snapshot
  provenance in maintainer and machine artifacts.
```

- [ ] **Step 3: Run focused test files**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_quality_analysis.py src/tests/test_report_generation.py src/tests/test_html_formatting.py src/tests/test_tsv_output.py src/tests/test_jsonl_output.py -q'
```

Expected: all selected tests pass.

- [ ] **Step 4: Verify tracked output was not changed by tests**

Run:

```bash
git status --short src/output
```

Expected: no output. If tests changed tracked output, diagnose and redirect the test output; do not discard user output with a destructive Git command.

- [ ] **Step 5: Run formatting, lint repair/check, and the complete quality gate in order**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make format'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make -C src lint-fix'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make lint'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make quality'
```

Expected: every command exits 0. Fix source and annotations instead of adding lint/type suppressions. Repeat affected focused tests after any correction.

- [ ] **Step 6: Inspect the final diff and commit documentation/cleanup**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Confirm `src/check_models.py` remains a monolith, no new standalone tests/scripts or multi-image logic were added, and tracked `src/output/` is unchanged.

Commit:

```bash
git add src/README.md CHANGELOG.md src/check_models.py src/tests/test_quality_analysis.py src/tests/test_report_generation.py src/tests/test_html_formatting.py src/tests/test_tsv_output.py src/tests/test_jsonl_output.py
git commit -m "docs: explain report status and provenance contracts"
```

- [ ] **Step 7: Perform verification-before-completion review**

Compare the final implementation with every acceptance criterion in
`docs/superpowers/specs/2026-07-18-report-semantics-provenance-design.md`. Report
the exact focused-test and full-quality results, any intentionally unregenerated
tracked reports, and the branch/commit state without claiming success from stale
or partial test output.
