# Canonical Assessment and Report Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain two purpose-specific classifications—one for model users and one for maintainers—while deriving both from one evidence record, publishing only confirmed upstream issues, simplifying keyword interpretation and confidence language, and making each classification agree across its relevant artifacts.

**Architecture:** Keep `src/check_models.py` as the intentional production monolith. Collect one immutable observed-evidence record from each `PerformanceResult`, then classify two immutable views: `ModelUserAssessment` for captioning/keywording usefulness and `MaintainerAssessment` for ownership, reproduction, and issue readiness. Cache both in `ReportRenderContext` and project each through its own narrow presentation record. Migrate one artifact family at a time and delete duplicated evidence derivation and report-local reclassification, while preserving genuinely audience-specific policy.

**Tech Stack:** Python 3.13, frozen dataclasses, narrow `Literal` aliases and `TypedDict` serialization views, pytest, Ruff, mypy, ty, Pyright, Markdownlint, existing report renderers and history formats.

---

## File Map

- Modify `src/check_models.py`: evidence, canonical assessment, report context, artifact projections, history aggregation, and deletion of old classifiers.
- Modify `src/tests/test_quality_analysis.py`: normalization, capped reasoning, numeric repetition, and recommendation invariants.
- Modify `src/tests/test_cataloging_utility.py`: keyword-overlap states and weak-signal policy.
- Modify `src/tests/test_error_classification.py`: failure-origin boundaries and readiness matrix.
- Modify `src/tests/test_process_image_mock.py`: upstream invocation facts and raw-output preservation.
- Modify `src/tests/test_report_generation.py`: diagnostics, issue gate, cross-report consistency, chooser semantics, capability history, and provenance wording.
- Modify `src/tests/test_jsonl_output.py`: assessment fields and schema versioning.
- Modify `src/tests/test_tsv_output.py`: rectangular canonical columns.
- Modify `src/tests/test_html_formatting.py`: canonical status/filter values and full expandable evidence.
- Modify `src/README.md`: report vocabulary, proxy scope, issue-readiness gate, and chooser interpretation.
- Modify `CHANGELOG.md`: document the maintainer-facing semantic and refactoring changes under `[Unreleased]`.

## Implementation Constraints

- Activate `mlx-vlm` before every Python command.
- Use synthetic, unrelated fixtures; do not encode current-image, subject, or model names in production rules.
- Add tests only to existing `src/tests/test_*.py` files.
- Write generated validation artifacts only to `tmp_path`, `/private/tmp`, or gitignored `test_*` paths.
- Preserve complete raw output and exception evidence.
- Add no lint/type-check suppression unless the repository's documented audit proves it necessary.
- Keep all production implementation in `src/check_models.py`; do not split the monolith.
- Finish with fewer than 30,998 lines in `src/check_models.py`, the baseline at plan creation.

### Task 0: Restore a Green Baseline Before the Refactor

**Files:**

- Modify: `src/tests/test_report_generation.py`
- Modify: `src/check_models.py` near `_UNKNOWN_OWNER`, `export_failure_repro_bundles`, and `generate_tsv_report`
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_tsv_output.py`

The 2026-07-19 GitHub run failed before this plan was written. Local
reproduction with Skylos 4.29.0 identifies three gate findings: the literal
`"unknown"` reaches the configured duplicate threshold of 40,
`export_failure_repro_bundles` has cyclomatic complexity 25, and
`generate_tsv_report` has cyclomatic complexity 29, against a threshold of 24.
The published-artifact test also treats an indeterminate connectivity result as
a conclusive runtime failure.

- [ ] **Step 1: Add the failing indeterminate-artifact assertion**

Refine `test_published_failure_artifacts_match_canonical_runtime_triage` so it
partitions unsuccessful records by their existing canonical review verdict:

```python
conclusive_failures = [
    record
    for record in records
    if record.get("_type") == "result"
    and not record["success"]
    and record["review"]["verdict"] != "indeterminate"
]
indeterminate = [
    record
    for record in records
    if record.get("_type") == "result"
    and record["review"]["verdict"] == "indeterminate"
]

assert all(record["compatibility_status"] == "indeterminate" for record in indeterminate)
assert all(
    "issue_cluster_path" not in record["maintainer_triage"]
    for record in indeterminate
)
```

Run the existing runtime-failure/issue assertions only over
`conclusive_failures`. Do not require that the checked-in run contain a
conclusive failure; a clean or connectivity-interrupted run is valid.

- [ ] **Step 2: Run the exact published-artifact test**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py::test_published_failure_artifacts_match_canonical_runtime_triage -q'
```

Expected: the test passes with the current indeterminate output set.

- [ ] **Step 3: Remove the duplicate-string finding without suppression**

Reuse `_UNKNOWN_OWNER` at existing owner/package fallbacks that currently spell
the same literal directly, including HTML error-package data, issue owner
fallbacks, failure counters, and failure package lists. Do not replace schema
`Literal["unknown"]` members or non-owner meanings merely to silence the
detector.

- [ ] **Step 4: Extract pure repro-bundle construction helpers**

Move per-result payload construction from `export_failure_repro_bundles` into a
typed `_build_failure_repro_bundle_payload` helper. Move model/cluster index
updates into `_record_repro_bundle_index`. Keep file iteration and error
handling in the public exporter. This lowers complexity while preserving the
schema and makes payload construction directly testable.

- [ ] **Step 5: Extract TSV record materialization helpers**

Move TSV escaping to module scope and extract `_build_tsv_row_records` plus a
small `_write_tsv_records` helper. `generate_tsv_report` should resolve the
context/table, select fields, call the helpers, and handle `OSError`; it should
not classify statuses itself. Preserve literal complete generated output and
header-first rectangular output.

- [ ] **Step 6: Run focused regression tests**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_tsv_output.py -k "published_failure_artifacts or repro_bundle or tsv" -q'
```

- [ ] **Step 7: Run both exact Skylos gates**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && cd src && TERM=dumb NO_COLOR=1 CLICOLOR=0 FORCE_COLOR=0 PY_COLORS=0 skylos . --quality --secrets --sca --gate --no-upload --format concise'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && cd src && TERM=dumb NO_COLOR=1 CLICOLOR=0 FORCE_COLOR=0 PY_COLORS=0 skylos . -a'
```

Expected: no duplicate-string or complexity findings and no suppressions added.

- [ ] **Step 8: Run the fast quality gate and commit the baseline fix**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && bash src/tools/run_quality_checks.sh --fast'
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_tsv_output.py
git commit -m "fix: restore static quality baseline"
```

### Task 1: Introduce the Canonical Types and Pure Readiness Gate

**Files:**

- Modify: `src/check_models.py` near `ExecutionOutcome`, `IssueReadiness`, `JsonlReviewRecord`, and `PerformanceResult`
- Test: `src/tests/test_error_classification.py`

- [ ] **Step 1: Write the table-driven readiness test first**

Add narrow tests for the approved publication matrix:

```python
@pytest.mark.parametrize(
    ("origin", "reproduction", "has_output_anomaly", "expected"),
    [
        ("harness_preflight", "not_run", False, "harness_observation"),
        ("external_service", "not_run", False, "not_applicable"),
        ("unknown", "not_run", False, "needs_reproduction"),
        ("upstream_load", "not_run", False, "issue_ready"),
        ("upstream_generation", "not_run", False, "issue_ready"),
        ("unknown", "confirmed", True, "issue_ready"),
        ("unknown", "not_reproduced", True, "not_applicable"),
        ("unknown", "indeterminate", True, "needs_reproduction"),
        ("unknown", "not_run", True, "needs_reproduction"),
    ],
)
def test_maintainer_readiness_matrix(
    origin: check_models.FailureOrigin,
    reproduction: check_models.ControlledReproductionStatus,
    has_output_anomaly: bool,
    expected: check_models.MaintainerReadiness,
) -> None:
    assert (
        check_models._maintainer_readiness(
            failure_origin=origin,
            reproduction_status=reproduction,
            has_output_anomaly=has_output_anomaly,
        )
        == expected
    )
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_error_classification.py -k maintainer_readiness_matrix -q'
```

Expected: collection or attribute failure because the new aliases and pure helper do not exist.

- [ ] **Step 3: Add full, narrow internal types**

Replace the old readiness alias and add immutable records near the existing result/report types:

```python
type FailureOrigin = Literal[
    "harness_preflight",
    "upstream_load",
    "upstream_generation",
    "external_service",
    "unknown",
]
type MaintainerReadiness = Literal[
    "issue_ready",
    "needs_reproduction",
    "harness_observation",
    "not_applicable",
]
type ControlledReproductionStatus = Literal[
    "not_run",
    "confirmed",
    "not_reproduced",
    "indeterminate",
]
type KeywordOverlapState = Literal["not_assessable", "no_overlap", "some_overlap"]
type HistoricalReliability = Literal[
    "stable",
    "variable",
    "insufficient_evidence",
    "consistently_unsuitable",
]
type OutputAnomaly = Literal[
    "special_token_wrapper",
    "special_token_leak",
    "thinking_trace",
    "reasoning_budget_exhausted",
    "text_repetition",
    "numeric_repetition",
    "mixed_script_corruption",
    "encoding_corruption",
    "missing_required_sections",
    "token_cap_truncation",
    "irrelevant_output_smell",
]


@dataclass(frozen=True)
class ObservedEvidence:
    """Immutable runtime, output, and proxy facts without policy decisions."""

    upstream_boundary: Literal["not_started", "load_started", "generation_started"]
    failure_phase: str | None
    exception_origin: str | None
    exception_chain: tuple[FailureException, ...]
    raw_output: str
    stop_reason: str | None
    requested_tokens: int | None
    generated_tokens: int | None
    anomalies: tuple[OutputAnomaly, ...]
    reproduction_status: ControlledReproductionStatus
    keyword_overlap: KeywordOverlapState


@dataclass(frozen=True)
class ModelUserAssessment:
    """Current-run usefulness decision for local model users."""

    execution_outcome: ExecutionOutcome
    compatibility_status: CompatibilityStatus
    current_recommendation: RecommendationStatus
    output_anomalies: tuple[OutputAnomaly, ...]
    keyword_overlap: KeywordOverlapState
    evidence_codes: tuple[str, ...]


@dataclass(frozen=True)
class MaintainerAssessment:
    """Ownership and publication decision for library/model maintainers."""

    failure_origin: FailureOrigin
    maintainer_readiness: MaintainerReadiness
    suspected_owner: str | None
    owner_confidence: MaintainerConfidence | None
    reproduction_status: ControlledReproductionStatus
    evidence_codes: tuple[str, ...]
    next_action: str


@dataclass(frozen=True)
class CanonicalAssessment:
    """The two purpose-specific decisions derived from one evidence record."""

    model_user: ModelUserAssessment
    maintainer: MaintainerAssessment
```

Add only fields required by later tests while preserving full/narrow types. Do not use untyped dictionaries as internal substitutes.

- [ ] **Step 4: Implement the small pure gate**

```python
def _maintainer_readiness(
    *,
    failure_origin: FailureOrigin,
    reproduction_status: ControlledReproductionStatus,
    has_output_anomaly: bool,
) -> MaintainerReadiness:
    if failure_origin == "harness_preflight":
        return "harness_observation"
    if failure_origin == "external_service" or reproduction_status == "not_reproduced":
        return "not_applicable"
    if failure_origin in {"upstream_load", "upstream_generation"}:
        return "issue_ready"
    if reproduction_status == "confirmed":
        return "issue_ready"
    if reproduction_status == "indeterminate" or failure_origin == "unknown":
        return "needs_reproduction"
    return "needs_reproduction" if has_output_anomaly else "not_applicable"
```

- [ ] **Step 5: Run tests and commit**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_error_classification.py -q'
git add src/check_models.py src/tests/test_error_classification.py
git commit -m "refactor: define canonical assessment policy"
```

Expected: error-classification tests pass.

### Task 2: Record Execution Boundaries and Classify Failure Origin

**Files:**

- Modify: `src/check_models.py` near `PerformanceResult`, `process_image_with_model`, `_run_model_generation`, `_build_failed_result`, and exception helpers
- Test: `src/tests/test_process_image_mock.py`
- Test: `src/tests/test_error_classification.py`

- [ ] **Step 1: Add failing boundary tests**

Cover four distinct paths:

```python
@pytest.mark.parametrize(
    ("boundary", "phase", "message", "expected"),
    [
        ("not_started", "model_preflight", "invalid local image input", "harness_preflight"),
        ("load_started", "model_load", "loader raised", "upstream_load"),
        ("generation_started", "decode", "generator raised", "upstream_generation"),
        (
            "load_started",
            "model_load",
            "server disconnected without sending a response",
            "external_service",
        ),
    ],
)
def test_failure_origin_follows_execution_boundary(
    boundary: Literal["not_started", "load_started", "generation_started"],
    phase: str,
    message: str,
    expected: check_models.FailureOrigin,
) -> None:
    result = check_models.PerformanceResult(
        model_name="example/model",
        generation=None,
        success=False,
        failure_phase=phase,
        error_message=message,
        upstream_boundary=boundary,
    )

    assert check_models._failure_origin(result) == expected
```

Also assert that a failure with no trustworthy boundary/origin evidence returns `unknown`.

- [ ] **Step 2: Run focused tests and confirm failures**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_process_image_mock.py src/tests/test_error_classification.py -k "upstream_entry or upstream_load_failure or upstream_generation_failure or external_service or unknown_failure_origin" -q'
```

- [ ] **Step 3: Add an explicit boundary field**

Add this field to `PerformanceResult`:

```python
upstream_boundary: Literal["not_started", "load_started", "generation_started"] = "not_started"
```

Advance it immediately before calling upstream load and generation, carry it through `_build_failed_result`, and retain it on successful results. Do not classify from model identifiers. Prefer the exception-chain `origin` and module fields over error-message text; use the existing connectivity detector only for the external-service override.

- [ ] **Step 4: Build observed failure facts once**

Implement `_failure_origin(result)` and `_collect_observed_evidence(result)` as pure functions. Map rerun evidence to the narrow reproduction status, treating connectivity during the rerun as `indeterminate`. Preserve `result.error_traceback`, `exception_chain`, and generated text unchanged.

- [ ] **Step 5: Run focused and full affected tests, then commit**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_process_image_mock.py src/tests/test_error_classification.py src/tests/test_jsonl_output.py -q'
git add src/check_models.py src/tests/test_process_image_mock.py src/tests/test_error_classification.py src/tests/test_jsonl_output.py
git commit -m "fix: retain upstream execution boundaries"
```

### Task 3: Normalize Only the Scoring Copy and Tighten Output Anomalies

**Files:**

- Modify: `src/check_models.py` near `_strip_empty_thinking_wrappers`, `_text_sanity_numeric_loop_issue`, token diagnostics, section parsing, and `analyze_generation_text`
- Test: `src/tests/test_quality_analysis.py`

- [ ] **Step 1: Add synthetic regression tests**

Add tests proving all of the following:

- valid `Title`/`Description`/`Keywords` inside recognized generic special-token wrappers are parsed;
- the raw generated text still includes those wrappers;
- wrapper presence remains an `OutputAnomaly`/evidence code;
- coherent reasoning that reaches the cap becomes `reasoning_budget_exhausted`, not numeric repetition, mixed-script corruption, or token soup;
- `42 42 42 42 42 42` remains contiguous numeric repetition;
- dates, coordinates, dimensions, and repeated factual values embedded in coherent prose are not numeric repetition;
- genuine mixed-script/tokenizer corruption is still detected.

Use fixtures about unrelated scenes such as a workshop, a shoreline, and a market. Do not use current output text.

- [ ] **Step 2: Run focused tests and confirm the wrapper/capped-prose cases fail**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_quality_analysis.py -k "wrapped_valid_output or coherent_capped_reasoning or numeric_repetition or mixed_script" -q'
```

- [ ] **Step 3: Introduce one normalized analysis copy**

Add a helper that returns normalized text plus evidence, leaving the caller's raw text untouched:

```python
@dataclass(frozen=True)
class NormalizedOutput:
    text: str
    removed_wrappers: tuple[str, ...] = ()


def _normalize_output_for_analysis(
    text: str,
    *,
    known_special_tokens: Sequence[str] = (),
) -> NormalizedOutput:
    """Return a scoring copy while retaining every removed wrapper as evidence."""
```

Implement the body by recognizing tokenizer-reported special tokens and
balanced generic wrapper structure, removing only recognized wrapper spans,
normalizing whitespace once, and returning the removed literal wrappers in
encounter order. Pass `NormalizedOutput.text` to semantic/section scoring, but
pass the original text to leakage, display, JSONL, and issue evidence.

- [ ] **Step 4: Require contiguous numeric degeneration**

Replace the global `Counter` majority heuristic with a contiguous-run detector over text after structured numeric metadata removal. Require the configured minimum run length of the same numeric token. Keep `_remove_structured_numeric_metadata` and delete the now-unused majority threshold constant/import path if possible.

- [ ] **Step 5: Separate capped reasoning from corruption**

If an expected thinking trace is coherent but exhausts the requested token budget before a final answer, emit `reasoning_budget_exhausted`/`thinking_incomplete`. Do not also emit numeric-loop, mixed-script, or degeneration unless those independent mechanical detectors genuinely match.

- [ ] **Step 6: Run tests and commit**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_quality_analysis.py -q'
git add src/check_models.py src/tests/test_quality_analysis.py
git commit -m "fix: distinguish wrappers and capped reasoning"
```

### Task 4: Reduce Keyword Evaluation to a Weak Overlap Smell

**Files:**

- Modify: `src/check_models.py` near metadata agreement, trusted-term detection, `compute_cataloging_utility`, recommendation classification, and report humanizers
- Test: `src/tests/test_cataloging_utility.py`
- Test: `src/tests/test_quality_analysis.py`

- [ ] **Step 1: Add failing overlap-state and recommendation-invariant tests**

```python
@pytest.mark.parametrize(
    ("reference", "generated", "expected"),
    [
        ((), ("harbour",), "not_assessable"),
        (("red boats",), (), "not_assessable"),
        (("wooden benches",), ("city lights",), "no_overlap"),
        (("garden paths", "flowers"), ("flower", "trees"), "some_overlap"),
    ],
)
def test_keyword_overlap_state(reference, generated, expected) -> None:
    assert check_models._keyword_overlap_state(reference, generated) == expected
```

Add invariants that:

- omitting any individual reference keyword does not demote a structurally valid result;
- `no_overlap` alone never yields `avoid`;
- `no_overlap` plus independent irrelevant prose and a broken contract may yield `avoid`;
- primary report prose does not enumerate missed terms or claim keyword recall.

- [ ] **Step 2: Run focused tests and confirm the old score/missing-term behaviour fails**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_cataloging_utility.py src/tests/test_quality_analysis.py -k "keyword_overlap_state or individual_keyword or no_overlap" -q'
```

- [ ] **Step 3: Implement elementary overlap only**

Reuse existing case-folding, punctuation/whitespace normalization, and generic singular/plural handling. Return only `not_assessable`, `no_overlap`, or `some_overlap`. Do not add embeddings, semantic dictionaries, aliases, categories, or subject-specific terms.

- [ ] **Step 4: Remove missing-keyword decision paths**

Stop using `missing_context_terms`, `missed_terms`, or recall-like percentages to choose current recommendation, maintainer subtype, confidence, or primary diagnostics. Preserve legacy raw measurements in versioned machine output only where needed for compatibility, clearly mark them experimental, and stop rendering missing-term lists in primary human reports.

- [ ] **Step 5: Run tests and commit**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_cataloging_utility.py src/tests/test_quality_analysis.py src/tests/test_report_generation.py -k "keyword or recommendation" -q'
git add src/check_models.py src/tests/test_cataloging_utility.py src/tests/test_quality_analysis.py src/tests/test_report_generation.py
git commit -m "fix: treat keyword overlap as a weak signal"
```

### Task 5: Compute and Cache Both Purpose-Specific Assessments per Result

**Files:**

- Modify: `src/check_models.py` near `_build_report_render_context`, `ReportRenderContext`, review/triage helpers, `MachineArtifactFacts`, and history serialization
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_jsonl_output.py`

- [ ] **Step 1: Add assessment and cache invariants**

Test that one context contains exactly one `(model_name, CanonicalAssessment)` bundle per resolved result and that each bundle contains distinct model-user and maintainer assessments. Patch `_build_canonical_assessment` and assert its call count equals the number of models during context construction and remains unchanged while rendering Markdown, HTML, TSV, and JSONL.

Add independence invariants:

- `avoid` plus `not_applicable` is valid for a poor but non-defective model output;
- `not_evaluated` plus `issue_ready` is valid for an upstream crash that never produced a caption;
- changing maintainer reproduction evidence cannot silently promote model-user recommendation;
- changing a weak model-quality signal cannot silently make an upstream issue issue-ready.

Add a table-driven recommendation test covering:

- failed and indeterminate execution;
- missing required sections;
- severe corruption or contiguous repetition;
- capped/truncated output;
- one `no_overlap` smell;
- `no_overlap` plus at least one independent irrelevance/contract signal;
- clean contract output.

- [ ] **Step 2: Run focused tests and confirm they fail**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py -k "canonical_assessment or assessment_cache or recommendation_matrix" -q'
```

- [ ] **Step 3: Implement assessment and presentation builders**

Add `_build_canonical_assessment(evidence, result)` as a coordinator for two small policy functions. `_classify_model_user_assessment` uses execution, required sections, severe corruption/repetition/cutoff, and multiple independent irrelevance signals. `_classify_maintainer_assessment` uses origin, reproduction status, owner evidence, and the confirmed-only gate. Neither calls or overrides the other.

Add two small immutable presentation records: `ModelUserPresentation` for recommendation label/icon/reason and `MaintainerPresentation` for owner/readiness/evidence/next action. Share low-level escaping and evidence formatting, not audience decisions.

Store both on `ReportRenderContext`:

```python
assessments: tuple[tuple[str, CanonicalAssessment], ...] = ()
user_presentations: tuple[tuple[str, ModelUserPresentation], ...] = ()
maintainer_presentations: tuple[tuple[str, MaintainerPresentation], ...] = ()
```

Provide typed lookup helpers and build the tuples once in `_build_report_render_context`.

- [ ] **Step 4: Make old serialized views projections only**

During migration, populate `JsonlReviewRecord` from the cached model-user assessment and `JsonlMaintainerTriageRecord` from the cached maintainer assessment. Populate shared `MachineArtifactFacts` from the appropriate fields of both. These serialized views may preserve stable field names but must not classify independently. Remove the cached review/triage fields and `*_payload_ready` flags from `PerformanceResult` when no callers need them.

- [ ] **Step 5: Run tests and commit**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py -q'
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py
git commit -m "refactor: cache canonical model assessments"
```

### Task 6: Enforce Confirmed-Only Diagnostics and Issue Generation

**Files:**

- Modify: `src/check_models.py` near `_build_diagnostics_snapshot`, `_build_issue_clusters`, diagnostics sections, issue draft renderers, and issue JSONL queue
- Test: `src/tests/test_report_generation.py`

- [ ] **Step 1: Add publication-gate tests**

Add one parameterized test that generates reports into `tmp_path` for every readiness state and asserts:

- only `issue_ready` creates an issue cluster, issue JSONL queue entry, and issue Markdown file;
- `needs_reproduction` appears only under controlled-reproduction observations;
- `harness_observation` appears under harness observations;
- `not_applicable` connectivity appears under indeterminate attempts;
- an upstream load/generation crash is issue-ready without a second run;
- serious unconfirmed anomalies may produce an empty issue directory/queue.

Assert complete output is present in an expandable code block for output observations and complete relevant traceback evidence is present for crashes.

- [ ] **Step 2: Run the focused diagnostics tests and confirm failures**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py -k "confirmed_only or readiness_section or complete_evidence or upstream_crash_issue_ready" -q'
```

- [ ] **Step 3: Filter before clustering**

Change `_build_issue_clusters` to receive only assessment/result pairs whose readiness is `issue_ready`. Remove readiness checks from individual harness/text-sanity/stack branches and issue renderers. An empty filtered input returns an empty tuple and removes stale current-run issue artifacts through the existing output lifecycle.

- [ ] **Step 4: Reorder diagnostics by readiness**

Render these sections in order: confirmed findings, needs reproduction, harness observations, indeterminate attempts, model-quality observations, environment/provenance. Use the canonical presentation record for labels and next action. Keep the model's complete raw output or relevant traceback in `<details>` blocks.

- [ ] **Step 5: Run diagnostics tests and commit**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py -k "diagnostic or issue" -q'
git add src/check_models.py src/tests/test_report_generation.py
git commit -m "fix: publish only confirmed upstream issues"
```

### Task 7: Unify Current-Run Human and Machine Artifacts

**Files:**

- Modify: `src/check_models.py` near result/review/model-selection/gallery/HTML/TSV/JSONL/run builders
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_jsonl_output.py`
- Test: `src/tests/test_tsv_output.py`
- Test: `src/tests/test_html_formatting.py`

- [ ] **Step 1: Add a cross-artifact semantic contract**

Build a mixed synthetic context containing `recommended`, `caveat`, `avoid`, and `not_evaluated`. Render all artifacts into `tmp_path`, parse them with existing helpers, and assert every artifact publishes the same current recommendation, compatibility, failure origin, maintainer readiness, and keyword-overlap state for each model.

Also assert:

- TSV is header-first and every row has the same width;
- HTML filter/status data attributes use canonical values;
- HTML and gallery preserve complete expandable output;
- JSONL metadata format version is bumped deliberately;
- `run.json` exposes the same counts and common settings.

- [ ] **Step 2: Run the contract test and confirm divergence**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py src/tests/test_html_formatting.py -k "cross_artifact or canonical_columns or canonical_filters" -q'
```

- [ ] **Step 3: Migrate report projections**

Make model-user surfaces (`model_selection.md`, gallery, user portions of HTML/TSV/JSONL, and current recommendation summaries) read the cached model-user assessment. Make maintainer surfaces (`diagnostics.md`, issue queue/drafts, maintainer portions of HTML/TSV/JSONL) read the cached maintainer assessment. `review.md` becomes an audit showing both decisions without calculating either. Keep public legacy fields when practical, but add explicit fields and bump JSONL from `2.0` to `2.1`; bump `run.json` only if its schema shape changes incompatibly.

Use underscore-valued machine fields (`not_evaluated`, `issue_ready`) and presentation-only hyphenation/labels. Do not permit report-specific synonyms such as `needs_triage`, `passed`, or `ineligible` to become stored recommendation states.

- [ ] **Step 4: Simplify the Quick Chooser**

Put Quick Chooser first. Rank only recommended rows for normal memory/speed choices. If a resource tier has none, label the best caveated row explicitly as a fallback rather than silently treating it as recommended. Show named sort policies and move the complete current-run matrix to an expandable appendix. Winner labels in `results.md` must say `fastest recommended`, `lowest-memory recommended`, or `fastest completed` as appropriate.

- [ ] **Step 5: Scope all score language**

Replace unqualified `grounded`, `visual quality`, and `best quality` claims with:

```text
Automated metadata-assisted proxy; one image; no human visual ground truth.
```

Keep 0–100 composites only as clearly experimental secondary diagnostics. Exclude them from current recommendation and use speed/memory only to order already usable rows.

- [ ] **Step 6: Run artifact suites and commit**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py src/tests/test_html_formatting.py -q'
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py src/tests/test_html_formatting.py
git commit -m "fix: align current-run report semantics"
```

### Task 8: Separate Historical Reliability from Current Recommendation

**Files:**

- Modify: `src/check_models.py` near `HistoryModelResultRecord`, `ModelCapabilityRunSignal`, `ModelCapabilityRow`, capability aggregation/rendering, and capability JSON
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_jsonl_output.py`

- [ ] **Step 1: Add history/current separation tests**

Create lane-compatible synthetic history for:

- repeated recommended runs (`stable`);
- mixed recommended/failure/caveat runs (`variable`);
- one run (`insufficient_evidence`);
- repeated avoid/crash runs (`consistently_unsuitable`).

Assert a current `caveat` remains `caveat` even when history is `stable`, and a current `recommended` remains `recommended` even when history is `variable`. Assert Markdown and capability JSON expose separate `current_recommendation` and `historical_reliability` fields plus run count, recommended rate, and variability reason.

- [ ] **Step 2: Run focused tests and confirm old `Use` semantics fail**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py -k "historical_reliability or capability_current_recommendation" -q'
```

- [ ] **Step 3: Replace ambiguous capability fields**

Change `ModelCapabilityRow.recommendation` to narrow `historical_reliability`, add `current_recommendation: RecommendationStatus`, and optionally retain a separate `strongest_task_signal` string. Replace `_model_capability_recommendation` with a small reliability aggregation over lane-compatible run counts and recommended/avoid outcomes. Rename the Markdown `Use` column to `Historical reliability`; keep `Current recommendation` separate.

Version the capability JSON payload if it has a declared schema version. Do not let aggregate scores override either status.

- [ ] **Step 4: Run capability/history tests and commit**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py -k "capability or history" -q'
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py
git commit -m "fix: separate historical and current status"
```

### Task 9: Complete Run Conditions and Provenance

**Files:**

- Modify: `src/check_models.py` near run manifest, diagnostics environment, prompt diagnostics, model provenance, and image profile rendering
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_jsonl_output.py`

- [ ] **Step 1: Add provenance/run-contract tests**

Assert human diagnostics and `run.json` include:

- raw image width, height, megapixels, size/hash when available;
- processed width/height and patch count, or the literal `unavailable` in human reports;
- total/text/non-text prompt burden or explicit unavailability;
- `trust_remote_code`;
- prompt SHA-256;
- common generation settings;
- check_models and library component revisions;
- requested and resolved model revisions.

Do not expose an absolute private input path when the existing publication-safe image record is available.

- [ ] **Step 2: Run focused tests and confirm missing fields**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py -k "run_contract or processed_dimensions_unavailable or trust_remote_code or source_revision" -q'
```

- [ ] **Step 3: Project existing provenance into human reports**

Reuse `_collect_component_provenance`, `ModelProvenanceRecord`, `RunImageRecord`, `PromptDiagnostics`, and generation-settings collectors. Add no second provenance subsystem. Render unavailable processed evidence explicitly and keep raw-image burden distinct from prompt text length.

- [ ] **Step 4: Run tests and commit**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py -q'
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py
git commit -m "feat: complete report run provenance"
```

### Task 10: Delete Superseded Classifiers and Prove the Monolith Shrinks

**Files:**

- Modify: `src/check_models.py`
- Modify: `src/README.md`
- Modify: `CHANGELOG.md`
- Test: existing affected test files only

- [ ] **Step 1: Inventory remaining duplicate decisions**

Run:

```bash
rg -n "_review_analysis_for_result|_build_jsonl_review_record|_recommendation_status_for_result|_issue_readiness_for_result|_build_jsonl_maintainer_triage_record|_model_capability_recommendation|needs_triage|needs-reproduction|confirmed" src/check_models.py
```

Expected before cleanup: legacy helpers or old state names remain.

- [ ] **Step 2: Delete migrated private paths**

Delete competing classification logic, including old review/triage caches, readiness branches, report-specific recommendation derivation, duplicated owner/next-action branches, and old capability `Use` decisions. Retain serialization builders only as shallow projections of `CanonicalAssessment` and `AssessmentPresentation`. Consolidate repeated label/icon tables and table-row assembly where that removes code without obscuring types.

Do not retain private compatibility wrappers merely to reduce the apparent patch size.

- [ ] **Step 3: Add a source invariant against fixture-specific classifiers**

Extend an existing dependency/quality-policy test to inspect production classifier source and reject known synthetic fixture/model terms used by tests. Keep the assertion general enough to prevent subject/model exceptions without forbidding legitimate documentation or test fixtures.

- [ ] **Step 4: Update documentation**

Document in `src/README.md`:

- current recommendation, historical reliability, compatibility, and maintainer readiness as separate dimensions;
- confirmed-only issue generation;
- weak keyword-overlap semantics;
- the one-image, metadata-assisted proxy limitation;
- Quick Chooser fallback and sort policies;
- complete evidence/provenance locations.

Add the refactor and behavior changes to `CHANGELOG.md` under `[Unreleased]`. Reiterate that lint suppressions should be avoided and must pass the existing purpose audit.

- [ ] **Step 5: Prove source contraction**

```bash
wc -l src/check_models.py
```

Expected: fewer than 30,998 lines. If not, revisit duplicated review, triage, recommendation, humanizer, and row-construction helpers before proceeding.

- [ ] **Step 6: Run formatting and static checks in the required order**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make format'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make -C src lint-fix'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make lint'
```

Expected: formatting and lint pass with no new suppression.

- [ ] **Step 7: Run focused regression suites**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_error_classification.py src/tests/test_process_image_mock.py src/tests/test_quality_analysis.py src/tests/test_cataloging_utility.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py src/tests/test_html_formatting.py -q'
```

Expected: all selected tests pass and tracked `src/output/` remains unchanged.

- [ ] **Step 8: Generate a complete temporary artifact set**

Use existing report test fixtures or the existing CLI mock path to generate every report into a temporary directory. Validate Markdown, JSON, JSONL, TSV rectangularity, and HTML structure there. Do not run a command that rewrites `src/output/`.

- [ ] **Step 9: Run the full quality gate**

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make quality'
```

Expected: pytest, Ruff, mypy, ty, Pyright, Markdownlint, dependency policy, and suppression audits all pass.

- [ ] **Step 10: Verify repository hygiene and commit**

```bash
git status --short
git diff --check
git diff -- src/output
```

Expected: no generated `src/output/` changes, no ephemeral plans/artifacts, and no whitespace errors.

```bash
git add src/check_models.py src/tests/test_error_classification.py src/tests/test_process_image_mock.py src/tests/test_quality_analysis.py src/tests/test_cataloging_utility.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py src/tests/test_html_formatting.py src/README.md CHANGELOG.md
git commit -m "refactor: unify canonical report assessment"
```

## Final Acceptance Audit

- [ ] Only `issue_ready` assessments create issue drafts or issue-queue entries.
- [ ] Harness preflight failures are `harness_observation`, not upstream defects.
- [ ] Upstream-origin crashes are immediately issue-ready.
- [ ] External-service failures are indeterminate and excluded from evaluated-failure counts.
- [ ] Raw output and traceback evidence are retained in complete expandable blocks.
- [ ] Wrapper normalization affects only analysis copies and cannot hide leakage evidence.
- [ ] Coherent capped reasoning is not classified as numeric/token corruption.
- [ ] Missing individual keywords cannot demote a model.
- [ ] Zero keyword overlap is only a smell and cannot alone yield `avoid`.
- [ ] Current recommendation agrees across all model-user Markdown, HTML, TSV, JSONL, and run/history projections.
- [ ] Maintainer readiness agrees across diagnostics, issue artifacts, HTML, TSV, and JSONL.
- [ ] Model-user and maintainer classifications remain independent and can legitimately disagree.
- [ ] Historical reliability is visibly and structurally separate from current recommendation.
- [ ] Human reports state the one-image, no-human-ground-truth proxy limitation.
- [ ] Diagnostics distinguish raw image burden from prompt text/non-text burden and unavailable processed evidence.
- [ ] `trust_remote_code`, revisions, prompt hash, image identity, and generation settings are visible.
- [ ] No new lint/type suppressions exist.
- [ ] `src/check_models.py` is shorter than 30,998 lines.
- [ ] Full `make quality` passes.
