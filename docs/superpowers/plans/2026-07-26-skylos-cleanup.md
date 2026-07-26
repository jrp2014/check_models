# Skylos Finding Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clear the ten full-audit Skylos findings and the separate documentation danger false positive without changing report behavior or weakening checks, then archive obsolete one-off tooling and completed planning records.

**Architecture:** Keep production PEP 695 aliases intact, make test-only types scanner-local and equally narrow, and reduce each complex function by extracting cohesive typed data-building boundaries inside the existing `check_models.py` monolith. Preserve every emitted string, record field, ordering rule, and metric through existing behavioral tests.

**Tech Stack:** Python 3.13, pytest, Ruff 0.16+, mypy, ty, pyrefly, Skylos 4.30, Markdownlint.

## Global Constraints

- Do not split `src/check_models.py`.
- Do not add Ruff, typing, or Skylos suppressions.
- Do not raise thresholds or change Skylos gate/advisory policy.
- Keep types full and narrow; do not replace literal domains with plain `str`.
- Do not modify tracked `src/output/` artifacts.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Run Python and Make commands only after activating Conda environment `mlx-vlm`.
- Keep active tools used by Make, CI, hooks, setup, updates, or maintained analysis.

---

### Task 1: Remove scanner-only phantom-reference and prose findings

**Files:**

- Modify: `src/tests/test_cataloging_utility.py`
- Modify: `src/tests/test_exif_extraction.py`
- Modify: `src/tests/test_metrics_modes.py`
- Modify: `src/tests/test_quality_analysis.py`
- Modify: `src/tests/test_report_generation.py`
- Modify: `docs/superpowers/specs/2026-07-24-subtractive-reporting-simplification-design.md`

**Interfaces:**

- Consumes: existing production aliases `KeywordOverlapState`, `ObservationCode`, `ExecutionStatus`, `LibraryVersionDict`, and `UpstreamBoundary` without changing them.
- Produces: scanner-local `Literal` annotations with the same exact value domains and direct use of the already-imported `PIL.Image` module in the EXIF test.

- [ ] **Step 1: Confirm the red scanner baseline**

Run:

```bash
cd src && skylos . -a --llm
```

Expected: ten findings, including six `SKY-L012` phantom references.

Run:

```bash
make skylos-danger-llm
```

Expected: one `SKY-D260` finding for the phrase `output token count`.

- [ ] **Step 2: Replace cross-module test annotations with equally narrow local types**

Use `Literal` in test modules for the exact domains:

```python
type ExpectedKeywordOverlapState = Literal["not_assessable", "no_overlap", "some_overlap"]
type ExpectedExecutionStatus = Literal["completed", "crashed", "indeterminate"]
type ExpectedUpstreamBoundary = Literal["not_started", "load_started", "generation_started"]
```

Use the complete `ObservationCode` literal domain for the observation tuple. Replace
the `LibraryVersionDict` test import with its exact structural type
`dict[str, str | None]`. In the URL EXIF test, patch `Image.open` on the direct
Pillow import rather than through `check_models.Image`.

- [ ] **Step 3: Rephrase the danger false positive**

Replace `output token count` with `number of generated tokens`, preserving the
meaning of the archived design requirement.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest src/tests/test_cataloging_utility.py src/tests/test_exif_extraction.py src/tests/test_metrics_modes.py src/tests/test_quality_analysis.py src/tests/test_report_generation.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Verify the seven findings are gone**

Run both Skylos commands from Step 1. Expected: the six `SKY-L012` findings and
the one `SKY-D260` finding are absent; only four `SKY-Q301` findings remain.

- [ ] **Step 6: Commit**

```bash
git add src/tests/test_cataloging_utility.py src/tests/test_exif_extraction.py src/tests/test_metrics_modes.py src/tests/test_quality_analysis.py src/tests/test_report_generation.py docs/superpowers/specs/2026-07-24-subtractive-reporting-simplification-design.md
git commit -m "test: remove Skylos phantom references"
```

### Task 2: Simplify gallery prompt fact construction

**Files:**

- Modify: `src/check_models.py` near `_gallery_prompt_facts`
- Test: `src/tests/test_report_generation.py`

**Interfaces:**

- Consumes: `PerformanceResult`, `PromptDiagnostics`, and `ModelProvenanceRecord`.
- Produces: `_gallery_fact(value: object | None, *, missing: str = "not captured") -> str` and an output-equivalent `_gallery_prompt_facts(...) -> tuple[tuple[str, str], ...]` below complexity 24.

- [ ] **Step 1: Confirm the specific red finding**

Run `cd src && skylos . -a --llm`. Expected: `SKY-Q301` for
`_gallery_prompt_facts` at complexity 25.

- [ ] **Step 2: Run the report characterization tests before editing**

Run:

```bash
pytest src/tests/test_report_generation.py -q
```

Expected: pass, establishing the current report output contract.

- [ ] **Step 3: Implement the minimal refactor**

Add `_gallery_fact` to render `None` and empty strings as the caller-selected
missing marker while preserving zero, booleans, strings, and JSON-compatible
values. Use it for prompt scalar fields and provenance lookups. Keep the processed
image formatting and generation-settings JSON ordering unchanged.

- [ ] **Step 4: Verify behavior and complexity**

Run the report test file and `cd src && skylos . -a --llm`. Expected: report tests
pass and `_gallery_prompt_facts` is absent from findings.

- [ ] **Step 5: Commit**

```bash
git add src/check_models.py
git commit -m "refactor: simplify gallery prompt facts"
```

### Task 3: Simplify comparison and performance logging

**Files:**

- Modify: `src/check_models.py` near `_log_model_comparison_table_and_charts` and `_log_performance_highlights`
- Test: `src/tests/test_metrics_modes.py`

**Interfaces:**

- Consumes: `PerformanceResult`, `ResultAssessment`, `ImageInputProfile`, and existing metric-formatting helpers.
- Produces: a frozen `ModelComparisonData` value with typed rows, TPS entries, total-time entries, and crashed results; narrow row/data collectors; and an image-memory logging helper. Both flagged logger functions remain their existing public signatures and fall below complexity 24.

- [ ] **Step 1: Confirm the two red findings**

Run `cd src && skylos . -a --llm`. Expected: `SKY-Q301` for both logging
functions, each at complexity 29.

- [ ] **Step 2: Run logging characterization tests**

Run:

```bash
pytest src/tests/test_metrics_modes.py -k "log_summary" -q
```

Expected: pass, covering row layout, charts, ranking eligibility, memory context,
and unusable-result exclusion.

- [ ] **Step 3: Extract comparison data construction**

Add:

```python
@dataclass(frozen=True)
class ModelComparisonData:
    rows: tuple[tuple[str, ...], ...]
    tps_entries: tuple[tuple[str, float], ...]
    total_time_entries: tuple[tuple[str, float], ...]
    crashed: tuple[PerformanceResult, ...]
```

Build each table row in one typed helper and collect ranking inputs in another.
Keep the exact sort key, `C/X/I` and `U/C/X/-` codes, numeric precision, table
columns, chart titles, and crash-stage counting.

- [ ] **Step 4: Extract metric sampling and image-memory logging**

Collect TPS, peak-memory, and load-time samples in a typed helper. Move only the
image-specific delta calculation/logging into a helper. Preserve all existing
eligibility rules, averages, units, labels, and zero/finite checks.

- [ ] **Step 5: Verify behavior and complexity**

Run the focused logging tests and `cd src && skylos . -a --llm`. Expected: the
tests pass and neither logging function appears in the report.

- [ ] **Step 6: Commit**

```bash
git add src/check_models.py
git commit -m "refactor: simplify performance summary logging"
```

### Task 4: Split JSONL result construction by schema boundary

**Files:**

- Modify: `src/check_models.py` near `_build_jsonl_result_record`
- Test: `src/tests/test_jsonl_output.py`

**Interfaces:**

- Consumes: `PerformanceResult`, `ExecutionStatus`, and the existing JSONL typed dictionaries.
- Produces: `_build_jsonl_metrics_record(result: PerformanceResult, recommended_working_set_bytes: int | None) -> JsonlMetricsRecord` and `_build_jsonl_failure_record(result: PerformanceResult, execution: ExecutionStatus) -> JsonlFailureRecord | None`; `_build_jsonl_result_record` remains output-compatible and below complexity 24.

- [ ] **Step 1: Confirm the red finding**

Run `cd src && skylos . -a --llm`. Expected: `SKY-Q301` for
`_build_jsonl_result_record` at complexity 26.

- [ ] **Step 2: Run JSONL characterization tests**

Run:

```bash
pytest src/tests/test_jsonl_output.py -q
```

Expected: pass, covering metrics, working-set percentages, failure fields,
exception chains, prompt diagnostics, timing, provenance, and exact output text.

- [ ] **Step 3: Implement schema-boundary builders**

Move metric population unchanged into `_build_jsonl_metrics_record`. Move failure
population and chronological exception-chain serialization unchanged into
`_build_jsonl_failure_record`. Compose both results in
`_build_jsonl_result_record`; do not add, remove, rename, or default JSON fields.

- [ ] **Step 4: Verify behavior and zero audit findings**

Run the JSONL tests and `cd src && skylos . -a --llm`. Expected: tests pass and
the audit reports zero findings.

- [ ] **Step 5: Commit**

```bash
git add src/check_models.py
git commit -m "refactor: simplify JSONL result construction"
```

### Task 5: Document and verify the completed cleanup

**Files:**

- Modify: `CHANGELOG.md`
- Verify: all files changed by Tasks 1–4

**Interfaces:**

- Consumes: the completed zero-finding implementation.
- Produces: an `[Unreleased]` maintainer note and fresh evidence from every quality tool, with Skylos still advisory by policy.

- [ ] **Step 1: Update the changelog**

Add an `[Unreleased]` entry stating that PEP 695 aliases remain intact, test-only
phantom references were removed, and four report/log/JSONL functions were
simplified below the current Skylos complexity threshold without suppressions.

- [ ] **Step 2: Run prescribed formatting and linting**

```bash
make format
make -C src lint-fix
make lint
```

Expected: no formatting or Ruff errors.

- [ ] **Step 3: Run the full quality gate**

```bash
make quality
```

Expected: all static checks and all 733-or-more tests pass; no suppression-audit
failure and no tracked output changes.

- [ ] **Step 4: Run both explicit Skylos scans**

```bash
cd src && skylos . -a --llm
make skylos-danger-llm
```

Expected: both report zero findings. The commands remain advisory; no gate policy
or threshold is changed.

- [ ] **Step 5: Inspect the final diff**

Run `git diff --check`, `git status --short`, and a suppression/threshold diff
search. Expected: only intended source, tests, documentation, plan/spec, and
changelog changes; no `src/output/`, ignore-list, or threshold changes.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: record Skylos cleanup"
```

### Task 6: Archive obsolete tooling and completed planning records

**Files:**

- Delete: `src/tools/qwen3_vl_sequential_repro.py`
- Modify: `src/tests/test_validate_env.py`
- Modify: `src/tests/test_dependency_sync.py`
- Modify: `src/README.md`
- Modify: `docs/CONTRIBUTING.md`
- Modify: `docs/IMPLEMENTATION_GUIDE.md`
- Create: `docs/notes/archive/QWEN3_VL_SEQUENTIAL_REPRO.md`
- Move: completed pre-26-July files from `docs/superpowers/plans/` and `docs/superpowers/specs/` to `docs/notes/archive/superpowers/`
- Delete: the duplicate suppression-audit files whose names end in `2.md`
- Move: `docs/notes/SKYLOS_QUALITY_BACKLOG_2026_05.md` to `docs/notes/archive/`
- Modify: `docs/notes/README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes: repository references, Make/CI/hook entry points, and Git history.
- Produces: an active `src/tools/` containing only maintained utilities, active
  planning directories containing only the current Skylos work, and a navigable
  historical archive. No executable compatibility wrapper replaces the retired
  Qwen3 probe.

- [ ] **Step 1: Reconfirm the archival inventory**

Use `rg` to verify every tool except `qwen3_vl_sequential_repro.py` is referenced
by Make, CI, hooks, setup/update code, maintained tests, or a current documented
analysis workflow. Confirm that the two `2.md` files are byte-for-byte duplicates
with `shasum` and `diff -q`.

- [ ] **Step 2: Archive the Qwen3 reproducer**

Remove the executable and its dedicated tests and active instructions. Add
`docs/notes/archive/QWEN3_VL_SEQUENTIAL_REPRO.md` recording its purpose, native
Metal-abort risk, former command shape, retirement date, and the Git command that
retrieves its final source:

```bash
git show 9225c527:src/tools/qwen3_vl_sequential_repro.py
```

Do not retain a dormant Python copy under the archive.

- [ ] **Step 3: Archive completed planning material**

Create `docs/notes/archive/superpowers/plans/` and
`docs/notes/archive/superpowers/specs/`. Move every plan/specification dated
17–24 July 2026 into the matching archive directory. Delete, rather than move,
the two exact duplicate suppression-audit `2.md` files. Leave only
`2026-07-26-skylos-cleanup.md` and its design in the active directories.

- [ ] **Step 4: Archive the stale Skylos backlog and refresh the index**

Move `SKYLOS_QUALITY_BACKLOG_2026_05.md` into `docs/notes/archive/`. Update
`docs/notes/README.md` so `GPS_DATA_FORMAT_EXPLANATION.md` remains the only active
note and the archive description covers historical tool, plan, specification,
review, and backlog records.

- [ ] **Step 5: Update the changelog and validate references**

Record the archival under `[Unreleased]`. Run repository-wide `rg` searches for
the retired executable and old active plan/spec paths. Expected: references occur
only in the archive note, changelog/history prose, or current archival plan/spec;
no Make, CI, hook, setup, or update command points at an archived file.

- [ ] **Step 6: Run affected tests and full verification**

```bash
pytest src/tests/test_validate_env.py src/tests/test_dependency_sync.py -q
make format
make -C src lint-fix
make lint
make quality
cd src && skylos . -a --llm
make skylos-danger-llm
```

Expected: all tests and quality checks pass, both explicit Skylos scans report
zero findings, and `git status --short` contains no `src/output/` changes.

- [ ] **Step 7: Commit**

```bash
git add CHANGELOG.md src/tools/qwen3_vl_sequential_repro.py src/tests/test_validate_env.py src/tests/test_dependency_sync.py src/README.md docs/CONTRIBUTING.md docs/IMPLEMENTATION_GUIDE.md docs/notes docs/superpowers/plans docs/superpowers/specs
git commit -m "chore: archive obsolete development artifacts"
```
