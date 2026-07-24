# Subtractive Reporting Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the overlapping score and report pipelines with one conservative, immutable current-run assessment and three consistent human renderings while preserving complete model evidence.

**Architecture:** Keep `src/check_models.py` as the intentional production monolith. Convert each resolved `PerformanceResult` once into a `ResultAssessment`; make JSONL, diagnostics, gallery, and HTML consume that cached value; preserve raw history only as append-only secondary data; then delete the retired reports, semantic scores, compatibility projections, and dead tests rather than retaining aliases.

**Tech Stack:** Python 3.13, `dataclasses`, `TypedDict`, `Literal`, mlx-vlm/MLX, pytest, Ruff, mypy, ty, pyright/pyrefly, Skylos, markdownlint-cli2.

## Global Constraints

- Work from the approved design in
  `docs/superpowers/specs/2026-07-24-subtractive-reporting-simplification-design.md`.
- Read `.github/copilot-instructions.md` before implementation and keep
  `src/check_models.py` as one production file.
- Activate the `mlx-vlm` Conda environment before every Python, pytest, or
  Make invocation.
- Preserve all pre-existing working-tree changes. Review the starting diff and
  stage only implementation-owned hunks.
- Never run validation against tracked `src/output/` paths. Every test-generated
  artifact must use `tmp_path` or a gitignored `test_*` location.
- Do not add compatibility aliases for retired statuses, fields, reports, or
  command-line options.
- Do not add bare `noqa`, blanket `type: ignore`, lint-file exclusions, or broad
  exception handling.
- Preserve complete generated output. A preview may aid navigation but never
  replace the full output in JSONL, gallery, diagnostics, or HTML.
- Treat a successful but suspicious response as an observation, not as an
  upstream failure. Only a hard crash or a directly recorded API-contract
  violation can be `actionable_failure`.
- Do not make model-specific or image-specific keyword rules. Keyword overlap
  remains a one-way smell: no overlap may add a caveat, while partial overlap
  cannot improve or reduce status.
- Do not read history while computing or rendering current-run results.
- Keep each task green before committing it. Do not refresh checked-in sample
  output until the final documentation task.

---

## Task 1: Establish the Minimal Current-run Assessment Contract

**Files:**

- Modify: `src/check_models.py:1288-13340`
- Modify: `src/check_models.py:13616-13660`
- Modify: `src/tests/test_report_generation.py:2036-2210`
- Modify: `src/tests/test_quality_analysis.py:430-490`

### Interface delivered by this task

```python
type ExecutionStatus = Literal["completed", "crashed", "indeterminate"]
type ModelUsability = Literal[
    "usable",
    "usable_with_caveats",
    "unusable",
    "not_evaluated",
]
type MaintainerStatus = Literal[
    "actionable_failure",
    "observation_needs_reproduction",
    "none",
]
type ObservationCode = Literal[
    "empty_output",
    "minimal_output",
    "repeated_output",
    "missing_requested_sections",
    "token_cap_truncation",
    "prompt_instruction_echo",
    "unexpected_special_token",
    "thinking_trace_present",
    "thinking_trace_incomplete",
    "no_keyword_overlap",
]


@dataclass(frozen=True)
class ResultAssessment:
    execution: ExecutionStatus
    usability: ModelUsability
    maintainer_status: MaintainerStatus
    observations: tuple[ObservationCode, ...]
```

- [ ] Add parameterized tests proving the exact execution, usability, and
  maintainer matrices. Construct `PerformanceResult` values directly and
  assert the complete immutable `ResultAssessment`, including observation
  ordering.

```python
@pytest.mark.parametrize(
    ("success", "connectivity", "expected"),
    [
        (True, False, "completed"),
        (False, False, "crashed"),
        (False, True, "indeterminate"),
    ],
)
def test_result_assessment_uses_three_execution_states(
    success: bool,
    connectivity: bool,
    expected: check_models.ExecutionStatus,
) -> None:
    error = "server disconnected without sending a response" if connectivity else "boom"
    result = PerformanceResult(
        model_name="example/model",
        success=success,
        generation=_MockGeneration(text="A complete response.") if success else None,
        error_message=None if success else error,
    )

    assert check_models._assess_result(result).execution == expected
```

- [ ] Add focused observation tests for empty output, one- or two-word minimal
  output, contiguous repetition, missing prompt-requested sections, recorded
  token-cap truncation, instruction echo, configured thinking wrappers,
  incomplete thinking traces, unexpected special tokens, and zero keyword
  overlap. Explicitly prove that an empty expected thinking wrapper is not an
  error observation and that partial keyword overlap adds no observation.

- [ ] Run the new tests and confirm they fail because `ResultAssessment` and
  `_assess_result` do not yet exist.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_quality_analysis.py -q -x
```

Expected: the first new contract test fails during collection or execution on
the missing minimal assessment API.

- [ ] Replace `ExecutionOutcome`, `RecommendationStatus`,
  `MaintainerReadiness`, `CompatibilityStatus`, `OutputAnomaly`,
  `ModelUserAssessment`, `MaintainerAssessment`, `CanonicalAssessment`, and
  their presentation dataclasses with the four narrow aliases and
  `ResultAssessment` above.

- [ ] Implement `_execution_status`, `_assessment_observations`, and
  `_assess_result`. Reuse existing factual detectors, but project only the
  approved codes. Use the following decision matrix exactly:

```python
_UNUSABLE_OBSERVATIONS: Final[frozenset[ObservationCode]] = frozenset(
    {
        "empty_output",
        "repeated_output",
        "missing_requested_sections",
        "token_cap_truncation",
    }
)


def _assess_result(result: PerformanceResult) -> ResultAssessment:
    execution = _execution_status(result)
    observations = _assessment_observations(result) if execution == "completed" else ()
    if execution != "completed":
        usability: ModelUsability = "not_evaluated"
    elif set(observations) & _UNUSABLE_OBSERVATIONS:
        usability = "unusable"
    elif observations:
        usability = "usable_with_caveats"
    else:
        usability = "usable"

    if execution == "crashed":
        maintainer_status: MaintainerStatus = "actionable_failure"
    elif observations:
        maintainer_status = "observation_needs_reproduction"
    else:
        maintainer_status = "none"

    return ResultAssessment(execution, usability, maintainer_status, observations)
```

  Keep `no_keyword_overlap`, `thinking_trace_present`,
  `unexpected_special_token`, `prompt_instruction_echo`, and `minimal_output`
  non-fatal by themselves. Require recorded stop/token-cap evidence before
  emitting `token_cap_truncation`. Preserve the existing direct connectivity
  signature logic for `indeterminate`.

- [ ] Change `ReportRenderContext.assessments` to
  `tuple[tuple[str, ResultAssessment], ...]`. Delete the cached user and
  maintainer presentation projections. Build one assessment per model in
  `_build_report_render_context` after prompt-aware quality facts are final.

- [ ] Remove tests for the superseded score/recommendation/readiness matrices
  and retain only the new structural matrix and detector tests.

- [ ] Run the focused assessment tests.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_quality_analysis.py -q -x
```

Expected: all retained assessment and quality-analysis tests pass.

- [ ] Commit the minimal assessment contract.

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_quality_analysis.py
git commit -m "refactor: reduce model result assessment"
```

---

## Task 2: Make JSONL and Run JSON the Narrow Machine Contract

**Files:**

- Modify: `src/check_models.py:1288-1620`
- Modify: `src/check_models.py:27347-27620`
- Modify: `src/tests/test_jsonl_output.py:100-1365`
- Modify: `src/tests/test_report_generation.py:616-860`

### Interface delivered by this task

```python
class JsonlAssessmentRecord(TypedDict):
    execution: ExecutionStatus
    usability: ModelUsability
    maintainer_status: MaintainerStatus
    observations: list[ObservationCode]


class JsonlFailureRecord(TypedDict, total=False):
    phase: str | None
    stage: str | None
    code: str | None
    message: str | None
    exception_type: str | None
    exception_module: str | None
    package: str | None
    traceback: str | None
    exception_chain: list[dict[str, str]]


class JsonlResultRecord(TypedDict):
    _type: Literal["result"]
    model: str
    timestamp: str
    assessment: JsonlAssessmentRecord
    generated_text: str
    captured_output_on_fail: str
    failure: JsonlFailureRecord | None
    metrics: JsonlMetricsRecord
    timing: JsonlTimingRecord
    model_provenance: ModelProvenanceRecord
    prompt_diagnostics: dict[str, JsonLike] | None


class RunOutcomeCounts(TypedDict):
    models_attempted: int
    models_evaluated: int
    models_completed: int
    models_crashed: int
    models_indeterminate: int
```

- [ ] Replace tests that assert `review`, `maintainer_triage`, confidence,
  grades, semantic scores, and duplicate top-level status fields with tests for
  the exact structures above. Assert JSONL `format_version == "2.0"` and run
  JSON `schema_version == "2.0"`.

- [ ] Add a cross-record test with one completed, one crashed, and one
  connectivity-indeterminate result. Assert the five counts are mutually
  consistent and that `models_evaluated == models_completed + models_crashed`.

- [ ] Add round-trip evidence tests using output containing tabs, Unicode,
  Markdown fences, HTML-looking thinking tags, and newlines. Assert
  `generated_text` is byte-for-byte equal after `json.loads` and that an empty
  successful output is present as `""`, not omitted.

- [ ] Run the focused machine-artifact tests and confirm they fail on the old
  schema.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_jsonl_output.py src/tests/test_report_generation.py -q -x
```

Expected: first failure is an old schema/status assertion.

- [ ] Replace the old JSONL record types with the exact narrow records above.
  Make `_assessment_to_json` the only status serializer:

```python
def _assessment_to_json(assessment: ResultAssessment) -> JsonlAssessmentRecord:
    return {
        "execution": assessment.execution,
        "usability": assessment.usability,
        "maintainer_status": assessment.maintainer_status,
        "observations": list(assessment.observations),
    }
```

- [ ] Rewrite the per-result branch of `save_jsonl_report` to obtain the cached
  assessment from `ReportRenderContext`. If direct callers omit a context,
  build a local context once for the whole input sequence; do not classify
  inside the row loop. Always serialize complete `generated_text` and
  `captured_output_on_fail` strings.

- [ ] Replace `_run_outcome_counts` with counts over cached execution status.
  Remove `models_total`, `models_successful`, `models_failed`, and any count
  based on report filtering.

- [ ] Rewrite `save_run_json_report` to schema `2.0`. Retain prompt/image
  identity, run settings, component and model provenance, counts, runtime, and
  artifact paths. Remove `semantic_rankings_grounded`, selection-score fields,
  capability paths, and score-derived prompt-burden judgements. Preserve raw
  prompt/image burden measurements when recorded.

- [ ] Remove obsolete JSONL TypedDicts and builders:
  `JsonlMetadataAgreementRecord`, `JsonlReviewRecord`,
  `JsonlMaintainerTriageRecord`, score fields in `JsonlResultRecord`, and their
  serialization helpers. Keep exception-chain, prompt-diagnostic, timing,
  resource, and provenance helpers.

- [ ] Run the focused machine-artifact tests.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_jsonl_output.py src/tests/test_report_generation.py -q -x
```

Expected: all retained JSONL/run JSON tests pass with schema `2.0`.

- [ ] Commit the machine contract.

```bash
git add src/check_models.py src/tests/test_jsonl_output.py src/tests/test_report_generation.py
git commit -m "refactor: narrow machine report schema"
```

---

## Task 3: Rebuild the Model Gallery Around Complete Evidence

**Files:**

- Modify: `src/check_models.py:3022-3135`
- Modify: `src/check_models.py:12767-13210`
- Modify: `src/check_models.py:17869-18248`
- Modify: `src/tests/test_report_generation.py:3778-4450`
- Modify: `src/tests/test_markdown_formatting.py:164-560`

### Interface delivered by this task

```python
MIN_THROUGHPUT_SAMPLE_TOKENS: Final[int] = 16


@dataclass(frozen=True)
class GalleryRow:
    model: str
    usability: ModelUsability
    observations: tuple[ObservationCode, ...]
    generation_tps: float | None
    peak_memory_gb: float | None
    generation_tokens: int | None
    output_preview: str
```

- [ ] Replace gallery tests for score-derived review summaries with contract
  tests for the skim-first order: chooser, unusable/not-evaluated rows,
  resource groupings, then complete per-model evidence.

- [ ] Add a complete-output test whose text exceeds every prior preview limit
  and contains triple backticks. Assert the report uses a longer valid fence
  and contains the whole string once inside an expandable `<details>` block.

- [ ] Add chooser tests proving that fewer than 16 generated tokens display
  `insufficient sample`, are excluded from fastest/average throughput, and do
  not erase raw timing metrics from the per-model evidence.

- [ ] Add deterministic-order tests: unusable and not-evaluated models first;
  usable models ordered by explicit lowest-memory and fastest-valid-generation
  policies; ties broken by full model name.

- [ ] Run the gallery tests and confirm they fail on score-derived headings and
  shortened evidence.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_markdown_formatting.py -q -x
```

- [ ] Replace gallery review/score helpers with `_gallery_row`,
  `_valid_generation_tps`, `_render_gallery_chooser`, and
  `_render_gallery_model`. `_valid_generation_tps` must be mechanical:

```python
def _valid_generation_tps(result: PerformanceResult) -> float | None:
    tokens = _generation_int_metric(result.generation, "generation_tokens")
    rate = _generation_float_metric(result.generation, "generation_tps")
    if tokens is None or tokens < MIN_THROUGHPUT_SAMPLE_TOKENS:
        return None
    return rate if rate is not None and rate >= 0 else None
```

- [ ] Make `generate_markdown_gallery_report` consume only the cached
  assessment, raw metrics, prompt facts, and complete generated or crash
  evidence. It must not call any detector, score, review, history, ownership,
  or recommendation function.

- [ ] Render every completed output with `_append_markdown_code_block`; render
  `empty output` explicitly before an empty fenced block. For crashed rows,
  show traceback first and partial/captured output second.

- [ ] Keep observations as short labels in the chooser. Put factual timing,
  token, memory, stop reason, prompt burden, revision, processor/tokenizer, and
  generation settings in the expandable per-model evidence rather than
  widening the chooser table.

- [ ] Run focused gallery and Markdown tests.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_markdown_formatting.py -q -x
```

Expected: gallery tests pass with no report-local assessment calls.

- [ ] Commit the model-user report.

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_markdown_formatting.py
git commit -m "refactor: make model gallery evidence first"
```

---

## Task 4: Rebuild Diagnostics and Restrict Issue Drafts

**Files:**

- Modify: `src/check_models.py:13933-17128`
- Modify: `src/check_models.py:28300-29480`
- Modify: `src/tests/test_report_generation.py:4553-6870`
- Modify: `src/tests/test_markdown_formatting.py:61-145`

### Interface delivered by this task

```python
def generate_diagnostics_report(
    results: Sequence[PerformanceResult],
    filename: Path,
    *,
    prompt: str,
    library_versions: LibraryVersionDict,
    system_info: Mapping[str, str],
    report_context: ReportRenderContext,
    image_path: Path | None = None,
) -> None:
    """Write current-run maintainer evidence without reclassifying results."""
```

- [ ] Replace cluster/readiness/confidence tests with four report sections:
  outcome counts, actionable crashes, successful observations requiring
  reproduction, and indeterminate attempts. Assert environment and provenance
  appear after model evidence.

- [ ] Add a crash evidence-order test with a long traceback, partial output,
  and captured stderr. Assert root exception and complete traceback appear
  before either secondary stream and no truncation marker is introduced.

- [ ] Add a successful-anomaly test with complete bizarre/repetitive output.
  Assert it is labelled `observation_needs_reproduction`, retains all output,
  names no suspected owner, and creates no issue draft.

- [ ] Add issue-draft tests proving that each `crashed` result creates one
  draft, each draft repeats the same complete traceback and factual provenance
  as diagnostics, and completed or indeterminate results create none.

- [ ] Run the diagnostic tests and confirm the old issue-cluster pipeline fails
  the new contract.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_markdown_formatting.py -q -x
```

- [ ] Replace `DiagnosticsSnapshot`, issue-cluster policy, ownership confidence,
  regression context, reproduction ratios, issue matrices, and report-local
  classifications with direct partitions over cached `ResultAssessment`.

```python
def _partition_diagnostics(
    context: ReportRenderContext,
) -> tuple[
    tuple[PerformanceResult, ...],
    tuple[PerformanceResult, ...],
    tuple[PerformanceResult, ...],
]:
    assessments = dict(context.assessments)
    actionable = tuple(
        result
        for result in context.result_set.results
        if assessments[result.model_name].maintainer_status == "actionable_failure"
    )
    observations = tuple(
        result
        for result in context.result_set.results
        if assessments[result.model_name].maintainer_status
        == "observation_needs_reproduction"
    )
    indeterminate = tuple(
        result
        for result in context.result_set.results
        if assessments[result.model_name].execution == "indeterminate"
    )
    return actionable, observations, indeterminate
```

- [ ] Make diagnostics render exact factual fields: phase, stage, exception
  module/type/message/chain, package when recorded, model revision, processor
  and tokenizer classes, stop reason, prompt and generation tokens, configured
  EOS/thinking tokens, complete traceback, complete generated/partial output,
  captured stdout/stderr, and a reproduction command. Render unavailable
  optional facts as `unavailable`.

- [ ] Reduce issue draft generation to an iteration over
  `actionable_failure`. A directly evidenced successful protocol violation may
  enter this status only if `PerformanceResult` already contains an explicit
  recorded contract breach; do not infer one from generated prose. Since no
  such explicit fact exists in the current result schema, the initial
  implementation intentionally treats hard crashes as the only issue-draft
  source.

- [ ] Delete routine issue-queue indexes, reproduction-bundle indexes,
  regression/readiness sections, owner-confidence prose, and acceptance-signal
  synthesis. Retain a direct reproduction command and factual issue draft for
  each actionable crash.

- [ ] Run focused diagnostic tests.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_markdown_formatting.py -q -x
```

Expected: diagnostics and conditional issue-draft tests pass.

- [ ] Commit the maintainer report.

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_markdown_formatting.py
git commit -m "refactor: simplify maintainer diagnostics"
```

---

## Task 5: Make HTML a Faithful Rendering of Gallery and Diagnostics

**Files:**

- Modify: `src/check_models.py:10533-10720`
- Modify: `src/check_models.py:17327-17868`
- Modify: `src/tests/test_report_generation.py:2373-2768`
- Modify: `src/tests/test_html_formatting.py`
- Modify: `src/tests/test_html_full_name.py`
- Modify: `src/tests/test_total_runtime_reporting.py`

- [ ] Add a cross-artifact test that renders JSONL, diagnostics, gallery, and
  HTML from one context containing `usable`, `usable_with_caveats`, `unusable`,
  crashed, and indeterminate results. Parse the JSONL and assert each exact
  status string appears for the same model in all relevant human artifacts.

- [ ] Add HTML evidence tests asserting the complete generated text survives
  `html.escape`, appears inside `<details><summary>`, and round-trips after
  `html.unescape`. Test literal `<thinking>`, ampersands, quotes, and Unicode.

- [ ] Add a test proving HTML contains both the model-user chooser/gallery and
  maintainer diagnostics, but no grade, numerical quality score, inferred
  owner confidence, or semantic winner language.

- [ ] Run HTML tests and confirm they fail against the independent legacy HTML
  table and recommendation pipeline.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_html_formatting.py src/tests/test_html_full_name.py src/tests/test_total_runtime_reporting.py -q -x
```

- [ ] Replace `_build_html_results_table`, chart/recommendation summaries, and
  report-local status helpers with HTML renderers over the same `GalleryRow`,
  `ResultAssessment`, diagnostic partitions, and factual evidence helpers used
  by Markdown.

- [ ] Keep filtering and navigation as presentation only. Filter values must be
  the exact `execution`, `usability`, and `maintainer_status` strings from the
  cached assessment. Do not compute an HTML-only status.

- [ ] Keep `generate_html_report` standalone, deterministic, and free of
  network access. Preserve total runtime and run provenance, but remove charts
  and tables whose inputs were semantic scores or historical projections.

- [ ] Run focused HTML and cross-artifact tests.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_html_formatting.py src/tests/test_html_full_name.py src/tests/test_total_runtime_reporting.py -q -x
```

Expected: all retained HTML and cross-artifact tests pass.

- [ ] Commit the HTML mirror.

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_html_formatting.py src/tests/test_html_full_name.py src/tests/test_total_runtime_reporting.py
git commit -m "refactor: align html with canonical reports"
```

---

## Task 6: Retire Duplicate Artifacts and Remove History from Current Reporting

**Files:**

- Modify: `src/check_models.py:1740-1805`
- Modify: `src/check_models.py:2197-2235`
- Modify: `src/check_models.py:17869-20145`
- Modify: `src/check_models.py:26682-27345`
- Modify: `src/check_models.py:29270-30255`
- Modify: `src/tests/test_metrics_modes.py:642-815`
- Modify: `src/tests/test_metrics_modes.py:958-1125`
- Modify: `src/tests/test_jsonl_output.py:1368-1700`
- Modify: `src/tests/test_report_generation.py:435-2035`
- Delete: `src/tests/test_tsv_output.py`

### Retained artifact contract

```python
@dataclass(frozen=True)
class ReportOutputPaths:
    index: Path
    html: Path
    gallery_markdown: Path
    jsonl: Path
    run_json: Path
    diagnostics: Path
    log: Path
    environment: Path
```

- [ ] Replace artifact-plan tests with an exact retained-key assertion:

```python
assert tuple(spec.key for spec in check_models._build_report_artifact_specs(paths)) == (
    "output_index",
    "html",
    "markdown_gallery",
    "diagnostics",
    "jsonl",
    "run_json",
)
```

  Assert the run manifest and `index.md` contain only those reports plus the
  log/environment paths, and do not mention results Markdown, review, model
  selection, model capability Markdown/JSON, TSV, issue queues, or repro
  indexes.

- [ ] Add a no-history-read test that patches `_load_history_run_records` to
  raise if called, then runs report finalization against temporary output paths.
  Assert report generation succeeds and current statuses are unchanged.

- [ ] Run the artifact/finalization tests and confirm they fail because the old
  artifacts are still scheduled.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_metrics_modes.py src/tests/test_jsonl_output.py src/tests/test_report_generation.py -q -x
```

- [ ] Shrink `ReportOutputPaths`, `ReportGenerationInputs`, default path
  constants, published-artifact name sets, `_resolve_report_output_paths`,
  `_build_report_artifact_specs`, `_public_output_artifact_map`,
  `_build_report_artifacts`, the Rich dashboard, directory creation, and stale
  cleanup to the retained set.

- [ ] Delete the command-line options and environment/default plumbing that
  exist solely for the retired outputs. Do not accept and ignore legacy options.

- [ ] Delete these generators and their dedicated helpers:
  `generate_markdown_report`, `generate_review_report`,
  `generate_model_selection_report`, `generate_model_capability_scorecard`, and
  `generate_tsv_report`. Remove their exports.

- [ ] Delete historical comparison, transition, regression, and capability
  aggregation from finalization and logging. Keep `append_history_record` only
  as raw secondary data, with factual execution/timing/resource fields and no
  current recommendation or semantic score. Do not load history during report
  generation.

- [ ] Delete `src/tests/test_tsv_output.py`. Remove only the tests in other
  files that exclusively cover retired artifacts or historical projections;
  retain runtime logging, raw history append, metrics, provenance, and cleanup
  tests.

- [ ] Simplify `generate_output_index_report` to a small link list with no
  independent analysis.

- [ ] Run artifact/finalization tests.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_metrics_modes.py src/tests/test_jsonl_output.py src/tests/test_report_generation.py -q -x
```

Expected: retained artifact plan and no-history-read tests pass.

- [ ] Commit the artifact retirement.

```bash
git add -A src/check_models.py src/tests/test_metrics_modes.py src/tests/test_jsonl_output.py src/tests/test_report_generation.py src/tests/test_tsv_output.py
git commit -m "refactor: retire duplicate report artifacts"
```

---

## Task 7: Delete Semantic Scoring and Compatibility Cruft

**Files:**

- Modify: `src/check_models.py:340-620`
- Modify: `src/check_models.py:4990-8300`
- Modify: `src/check_models.py:10340-12190`
- Modify: `src/check_models.py:13200-13930`
- Modify: `src/check_models.py:18250-20010`
- Modify: `src/check_models.py:24600-25320`
- Modify: `src/tests/test_cataloging_utility.py`
- Modify: `src/tests/test_quality_analysis.py`
- Modify: `src/tests/test_pure_logic_functions.py`
- Modify: `src/tests/test_result_sorting.py`

- [ ] Add or retain narrow tests for only the mechanical detectors still used
  by `ResultAssessment`: exact prompt instruction echo, requested-section
  parsing, contiguous repetition, thinking markers, configured special-token
  handling, recorded truncation, and weak keyword no-overlap.

- [ ] Add a source-level regression test in the existing dependency/quality
  audit that rejects the retired public symbols and grade constant:

```python
for retired_name in (
    "GRADE_EMOJIS",
    "ModelRecommendationView",
    "ModelCapabilityRow",
    "MachineArtifactFacts",
    "_model_selection_score",
    "_caption_usefulness_score",
    "_recommendation_quality_score",
    "_score_metadata_title",
    "_score_metadata_description",
    "_score_metadata_keywords",
):
    assert retired_name not in source
```

- [ ] Run the focused logic tests before deletion and record the retained
  detector baseline.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_cataloging_utility.py src/tests/test_quality_analysis.py src/tests/test_pure_logic_functions.py src/tests/test_result_sorting.py -q -x
```

- [ ] Follow callers and delete A-F/composite scoring, metadata-agreement score
  production, information-gain/task-compliance/visual-grounding scorecards,
  semantic ranking, winner selection, score deltas, capability/history score
  aggregation, and score-only configuration fields.

- [ ] Delete score-only dataclasses and contexts, including
  `ReportTriageContext`, `ModelRecommendationView`, `MachineArtifactFacts`,
  `ModelCapabilityRow`, `PreparedTableData`, score grades, winner rows, and
  compatibility projections. Retain a small factual stats structure only if
  it is still used by the log or retained reports.

- [ ] Remove shortening helpers from report paths. Retain any log-only safety
  truncation only when it protects terminal usability and cannot affect stored
  evidence.

- [ ] Reduce `GenerationQualityAnalysis` to fields required by retained
  mechanical observations or runtime logging. Delete semantic verdict, owner,
  confidence, score, borrowing, fabrication, genericity, verbosity, and
  report-only fields plus their builders.

- [ ] Reduce `src/tests/test_cataloging_utility.py` to the weak keyword-overlap
  and retained parser/detector tests, or move those tests into an existing
  logical test module and delete the file if nothing score-related remains.

- [ ] Run focused logic and full non-slow tests.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_cataloging_utility.py src/tests/test_quality_analysis.py src/tests/test_pure_logic_functions.py src/tests/test_result_sorting.py -q -x
pytest -q -m "not slow and not e2e" -x
```

Expected: all retained tests pass; no primary output depends on semantic scores.

- [ ] Confirm the refactor is a substantial net deletion and inspect every
  remaining score/classification match.

```bash
git diff --stat origin/main...HEAD -- src/check_models.py src/tests
rg -n "GRADE_EMOJIS|utility_score|capability_score|owner_confidence|semantic_rank|winner|ModelRecommendationView|MachineArtifactFacts" src/check_models.py src/tests
```

Expected: the diff reports materially fewer lines, and every remaining match is
either raw historical input documentation scheduled for removal or a false
positive that does not affect assessment.

- [ ] Commit the dead-code deletion.

```bash
git add src/check_models.py src/tests/test_cataloging_utility.py src/tests/test_quality_analysis.py src/tests/test_pure_logic_functions.py src/tests/test_result_sorting.py
git commit -m "refactor: delete semantic scoring pipeline"
```

---

## Task 8: Tighten Upstream Types and Remove Avoidable Suppressions

**Files:**

- Modify: `src/check_models.py:20-2100`
- Modify: `src/check_models.py:8800-9400`
- Modify: `src/check_models.py:20200-22650`
- Modify: `src/tools/generate_stubs.py`
- Modify: `src/tests/test_dependency_sync.py`
- Modify: `src/tests/test_process_image_mock.py`
- Modify after stub refresh: `typings/mlx_vlm/generate/types.pyi`
- Modify after stub refresh: `typings/mlx_vlm/generate/__init__.pyi`
- Modify after stub refresh: `typings/.stub_manifest.json`

- [ ] Add type/stub contract tests that require the generated mlx-vlm stub to
  expose `mlx_vlm.generate.types.GenerateKwargs`, including `eos_tokens`,
  `skip_special_tokens`, `enable_thinking`, `thinking_start_token`,
  `thinking_end_token`, and `thinking_budget`.

- [ ] Under `TYPE_CHECKING`, import the upstream type without introducing a
  runtime import dependency:

```python
if TYPE_CHECKING:
    from mlx_vlm.generate.types import GenerateKwargs
```

  Remove the duplicated local `GenerateExtraKwargs`. Annotate
  `_build_generate_extra_kwargs` as returning `GenerateKwargs`, and use that
  type at the `generate` call boundary. Retain the narrower local
  `ChatTemplateKwargs` because it describes the separate chat-template API.

- [ ] Keep `src/tools/generate_stubs.py` generating/importing the upstream
  `GenerateKwargs` location for environments whose installed package lacks
  inline typing. Remove obsolete fallback content only after the stub tests
  prove the actual current upstream layout.

- [ ] Run stub generation and type checks.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make stubs
make typecheck
make ty
```

Expected: all type checkers pass with the upstream generation contract and no
unresolved `mlx.core` import.

- [ ] Audit every remaining suppression in `src/check_models.py`, affected
  tests, and generated report strings:

```bash
rg -n "noqa|type: ignore|markdownlint-disable|skylos: ignore|nosec" src/check_models.py src/tests src/tools
```

- [ ] Remove complexity suppressions whose functions were deleted, generated
  Markdown suppression comments made obsolete by shorter reports, and stale
  Skylos markers. Replace broad `BLE001` catches with the narrow documented
  upstream exception sets already used at the surrounding runtime boundary.
  Retain `S603`/`S310` only where arguments or URLs are structurally constrained
  and add a specific justification to the existing suppression audit test.

- [ ] Run formatting, safe lint fixes, Ruff, type checks, and the suppression
  audit.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make format
make -C src lint-fix
make lint
make stubs
make typecheck
make ty
pytest src/tests/test_dependency_sync.py src/tests/test_process_image_mock.py -q -x
```

Expected: all checks pass without new blanket suppressions.

- [ ] Commit the type and suppression cleanup.

```bash
git add src/check_models.py src/tools/generate_stubs.py src/tests/test_dependency_sync.py src/tests/test_process_image_mock.py typings/mlx_vlm/generate/types.pyi typings/mlx_vlm/generate/__init__.pyi typings/.stub_manifest.json
git commit -m "refactor: tighten report and generation types"
```

---

## Task 9: Align Documentation, Changelog, and Checked-in Artifact Policy

**Files:**

- Modify: `src/README.md`
- Modify: `CHANGELOG.md`
- Modify: `.github/copilot-instructions.md` only if its artifact list is stale
- Modify: `.markdownlint-cli2.jsonc` only if a single documented
  `details`/`summary` allowance is required
- Modify: `src/tests/test_dependency_sync.py`
- Modify: `src/tests/test_report_generation.py`
- Modify intentionally at the end: retained files under `src/output/`
- Delete intentionally at the end: checked-in retired default artifacts under
  `src/output/`

- [ ] Update documentation tests first. Assert the README names only
  `diagnostics.md`, `model_gallery.md`, `results.html`, `results.jsonl`,
  `run.json`, the tiny index, log, environment, and append-only history. Assert
  it documents the three exact status vocabularies and explains that complete
  output is evidence.

- [ ] Update `src/README.md` to remove retired flags/artifacts, semantic grades,
  capability scores, ownership confidence, and history-derived recommendation
  claims. Document `insufficient sample`, indeterminate connectivity, weak
  no-keyword-overlap, configured thinking tokens, and conditional issue drafts.

- [ ] Add a concise `[Unreleased]` changelog entry describing the breaking
  schema `2.0`, retired reports, facts-first assessment, complete evidence, and
  type/lint cleanup. Preserve all existing uncommitted changelog edits.

- [ ] Prefer ordinary Markdown accepted by the existing configuration. If
  expandable `<details>` remains the sole necessary raw-HTML construct, add one
  project-level `MD033` allowed-elements entry for `details` and `summary` with
  an explanatory comment; do not emit per-report markdownlint disable comments.

- [ ] Generate refreshed artifacts only through a deliberate real or fixture
  run after tests are green. Retain and review only the approved output set.
  Delete checked-in retired reports and machine scorecards. Do not alter the
  user's newest raw run before taking a separate diff snapshot.

- [ ] Run documentation, output-policy, and Markdown lint tests.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_dependency_sync.py src/tests/test_report_generation.py -q -x
npx markdownlint-cli2 "**/*.md" "!src/node_modules/**" "!**/node_modules/**"
```

Expected: documentation/output policy tests and Markdown lint pass.

- [ ] Commit documentation and intentional sample-output changes separately so
  generated evidence remains reviewable.

```bash
git add src/README.md CHANGELOG.md .github/copilot-instructions.md .markdownlint-cli2.jsonc src/tests/test_dependency_sync.py src/tests/test_report_generation.py
git commit -m "docs: describe simplified report outputs"
git add -A src/output
git commit -m "chore: refresh simplified report examples"
```

  Omit unchanged optional files from `git add`; do not create empty commits.

---

## Task 10: Full Verification and Final Cruft Audit

**Files:**

- Verify: repository-wide
- Modify only if a verification failure identifies a defect in the preceding
  tasks

- [ ] Snapshot the working tree and confirm unrelated starting changes remain
  intact and unstaged unless they were intentionally incorporated.

```bash
git status --short --branch
git diff --stat
```

- [ ] Run the prescribed quality sequence in the Conda environment.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make format
make -C src lint-fix
make lint
make quality
```

Expected: the complete project quality gate passes.

- [ ] Re-run artifact, assessment, and evidence contract tests directly so the
  final report is not based only on an aggregate Make target.

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_markdown_formatting.py src/tests/test_html_formatting.py -q
```

Expected: all focused contract tests pass.

- [ ] Prove tests and formatters did not rewrite user-generated output except
  for the intentional final artifact refresh.

```bash
git status --short src/output
git diff -- src/output
```

- [ ] Run final removal and placeholder scans.

```bash
rg -n "generate_(markdown_report|review_report|model_selection_report|model_capability_scorecard|tsv_report)|GRADE_EMOJIS|ModelRecommendationView|MachineArtifactFacts|owner_confidence|semantic_rankings_grounded" src/check_models.py src/tests src/README.md
rg -n "\b(TODO|TBD|FIXME|XXX)\b" src/check_models.py src/tests src/README.md CHANGELOG.md
rg -n "noqa|type: ignore|markdownlint-disable|skylos: ignore|nosec" src/check_models.py src/tests src/tools
```

Expected: no retired pipeline symbol or placeholder remains; every surviving
suppression is narrow, necessary, documented, and accepted by the audit.

- [ ] Confirm `src/check_models.py` is materially shorter and that the retained
  assessment/rendering flow is easy to trace.

```bash
git diff --numstat origin/main...HEAD -- src/check_models.py
rg -n "def (_assess_result|save_jsonl_report|generate_diagnostics_report|generate_markdown_gallery_report|generate_html_report|save_run_json_report)" src/check_models.py
```

- [ ] Commit only genuine verification fixes, then present the final commit
  list, line-count reduction, test totals, quality-gate result, and any retained
  suppression justifications to the user. Do not merge or push unless the user
  explicitly requests it.
