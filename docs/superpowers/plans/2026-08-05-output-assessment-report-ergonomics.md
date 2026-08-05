# Output Assessment and Report Ergonomics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct incomplete-output assessment and consolidate compact,
reproducible retained reports without penalising valid closed thinking sections.

**Architecture:** Extend canonical quality evidence just enough to distinguish a
complete thinking section from missing final output, then project that evidence
once through `ResultAssessment`. Reuse shared Rich, ordering, and reproduction
builders across terminal, Markdown, and HTML consumers; remove legacy parallel
formatters and fake local-image commands.

**Tech Stack:** Python 3.13, pytest, Rich, tabulate, Markdown report blocks,
conda `mlx-vlm`.

## Global Constraints

- Keep `src/check_models.py` as the intentional single-file monolith.
- A closed thinking block followed by substantive final text is neutral; retain
  its markers only as machine evidence.
- Keep assessment mechanical, image-independent, and model-name-independent.
- A sanitised gallery preview is not exact reproduction media unless its digest
  matches the run manifest.
- Run Python only inside the conda `mlx-vlm` environment; never use `uv`.
- Do not rewrite the tracked fresh `src/output/` run during tests or validation.
- Prefer compact shared helpers and existing external libraries over parallel
  formatting or classification code.

---

### Task 1: Canonical incomplete-output assessment

**Files:**

- Modify: `src/check_models.py`
- Test: `src/tests/test_quality_analysis.py`
- Test: `src/tests/test_jsonl_output.py`

**Interfaces:**

- Produces: `ReasoningOutputSignals.thinking_only_output: bool`, optional seeded
  thinking starts in `_detect_reasoning_output`, and observation code
  `missing_final_answer`.
- Preserves: legacy `thinking_trace_present` schema acceptance for retained JSONL,
  while new complete traces remain neutral.

- [ ] **Step 1: Write failing assessment tests**

Add focused cases equivalent to:

```python
def test_closed_thinking_with_final_answer_is_neutral_machine_evidence() -> None:
    result = _result("<think>Inspect.</think> Two cats sleep on a pink couch.")
    assessment = check_models._assess_result(result)
    assert assessment == check_models.ResultAssessment("completed", "usable", "none", ())
    assert check_models._observation_details(result)["thinking_trace_markers"] == [
        "<think>",
        "</think>",
    ]


def test_closed_thinking_without_final_answer_is_unusable() -> None:
    result = _result("<think>Inspect the scene.</think>")
    assert check_models._assess_result(result).observations == ("missing_final_answer",)


def test_prompt_seeded_thinking_open_is_closed_by_generated_marker() -> None:
    result = check_models.PerformanceResult(
        model_name="example/seeded-thinking",
        success=True,
        generation=_Generation(
            "Inspect the scene.</think> Two cats sleep on a pink couch.",
            generation_tokens=18,
        ),
        prompt_diagnostics=check_models.PromptDiagnostics(
            rendered_prompt_preview="<image>Describe this image.<think>",
        ),
    )
    context = check_models._build_report_render_context(
        results=[result], prompt="Describe this image.", system_info={}
    )
    assert dict(context.assessments)[result.model_name].observations == ()


@pytest.mark.parametrize("tail", ["The", "Based on the image:\n\n*   **"])
def test_degraded_token_cap_is_unusable(tail: str) -> None:
    result = _result(tail, generated_tokens=500, requested_max_tokens=500)
    assert "token_cap_truncation" in check_models._assess_result(result).observations
    assert check_models._assess_result(result).usability == "unusable"
```

Update JSONL detail expectations so neutral traces still retain markers and new
observations serialize and validate.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_quality_analysis.py src/tests/test_jsonl_output.py -k "thinking or token_cap or final_answer" -q
```

Expected: failures showing complete thinking is still caveated, seeded opening
markers are not recognised, thinking-only output lacks an observation, and
dangling Markdown is not cutoff evidence.

- [ ] **Step 3: Implement the minimal canonical evidence changes**

Extend `_detect_reasoning_output` to accept seeded opening markers derived from
`PromptDiagnostics.rendered_prompt_preview`, find a matching close, and test for
alphanumeric final text after the closing marker. Add `thinking_only_output` to
the quality records. In `_detect_likely_cutoff`, add named evidence for incomplete
thinking and dangling Markdown/list syntax.

Project observations compactly:

```python
(analysis.likely_capped and bool(analysis.token_cap_reasons), "token_cap_truncation"),
(analysis.thinking_only_output, "missing_final_answer"),
(analysis.thinking_trace_incomplete, "thinking_trace_incomplete"),
```

Do not emit `thinking_trace_present` for new complete output. Exclude thinking
markers from `configured_wrapper_present`, but retain every marker in
`_observation_details`. Add truncation, missing final answer, and incomplete
thinking to unusable evidence. Make `_build_quality_issues_string` consume the
same conditions instead of retaining its old token-cap predicate.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit the assessment slice**

```bash
git add src/check_models.py src/tests/test_quality_analysis.py src/tests/test_jsonl_output.py
git commit -m "fix: classify incomplete model output accurately"
```

### Task 2: Shared actionability ordering and compact terminal tables

**Files:**

- Modify: `src/check_models.py`
- Test: `src/tests/test_metrics_modes.py`
- Test: `src/tests/test_report_generation.py`

**Interfaces:**

- Produces: one `_assessment_actionability_key(...)` used by diagnostics,
  run-summary JSON records, and terminal completed groups.
- Reuses: `_log_rich_table` for the model comparison and completed-model tables.

- [ ] **Step 1: Write failing presentation tests**

Add assertions that:

- the model comparison is emitted through a captured Rich table without the
  manual fixed-width header;
- observed completed tables contain `Model` and `Observations`, not `Maintainer`;
- unusable/repetition evidence precedes caveats and minimal output in both
  diagnostics triage and run summary;
- diagnostics uses `Crashes requiring action`;
- gallery/HTML headings say `Usable Models (Including Caveats)`.

- [ ] **Step 2: Run presentation tests and verify RED**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_metrics_modes.py src/tests/test_report_generation.py -k "comparison or completed_model or triage or gallery_ranking or actionable" -q
```

Expected: failures on the manual comparison header, redundant column, diagnostics
ordering/heading, and ambiguous gallery headings.

- [ ] **Step 3: Consolidate rendering and ordering**

Extract a small sort-key helper taking model name, usability, and observations.
Replace the nested run-summary sort function and use the helper when partitioning
diagnostics and completed model groups. Replace the comparison header/row logging
loop with:

```python
_log_rich_table(
    headers=("#", "Model", "E/U", "Val", "Load", "Prep", "First", "Remain", "Clean", "Total", "TPS", "GB"),
    rows=comparison.rows,
)
```

Remove the completed-table `Maintainer` value and width allocation. Rename the
shared Markdown/HTML section text rather than post-processing rendered output.

- [ ] **Step 4: Run presentation tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit the presentation slice**

```bash
git add src/check_models.py src/tests/test_metrics_modes.py src/tests/test_report_generation.py
git commit -m "refactor: consolidate actionable report presentation"
```

### Task 3: Self-contained shared reproduction and artifact manifest

**Files:**

- Modify: `src/check_models.py`
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_jsonl_output.py`

**Interfaces:**

- Reuses: `_run_image_record`, `_reproduction_input_blocks`,
  `_issue_public_image_source_url`, and native `ReproCommandSpec`.
- Removes: diagnostics-only `reproduce.py`/`prompt.txt` assembly.

- [ ] **Step 1: Write failing reproduction tests**

For local-only media, generate diagnostics under `tmp_path` and assert exact
prompt, format, dimensions, size, and SHA are present while `reproduce.py`,
`prompt.txt`, `--image cats.jpg`, and a claimed complete command are absent.

For public media with a matching digest, assert diagnostics contains
`curl --fail --location`, `shasum -a 256 --check`, `python -m mlx_vlm.generate`,
`MODEL_ID`, and `RESOLVED_REVISION`.

Add run-JSON tests asserting `issues/run_summary.md` appears in `artifacts` only
when cached assessments contain an actionable, observed, or indeterminate result.

- [ ] **Step 2: Run reproduction tests and verify RED**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py -k "reproduction or artifact_manifest or run_summary" -q
```

Expected: diagnostics still contains the fake parameterised local command and
run JSON omits the conditional summary artifact.

- [ ] **Step 3: Reuse the publication-safe reproduction blocks**

Build one `RunImageRecord` inside `_diagnostics_shared_context_blocks` using the
original image path/profile plus optional public source URL. Replace the prompt,
generic command, and generated Python script sections with
`_reproduction_input_blocks(model_name="MODEL_ID", ...,
resolved_revision="RESOLVED_REVISION")`. Delete the now-unused parameterised
script builder and local image-ref helper if no consumer remains.

In `save_run_json_report`, derive the artifact map once and conditionally add
`issues/run_summary.md` when the canonical assessments contain a surfaced result.

- [ ] **Step 4: Run reproduction tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit the reproduction slice**

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_jsonl_output.py
git commit -m "fix: make diagnostic reproduction inputs self-contained"
```

### Task 4: Documentation follow-up, archival, and verification

**Files:**

- Create: `docs/notes/PROJECT_SKILL_GAPS.md` only if a demonstrated recurring gap
  remains after using the adapted guidance.
- Move: completed files from `docs/superpowers/specs/` and
  `docs/superpowers/plans/` into `docs/notes/archive/superpowers/`.
- Move: superseded implementation reviews from `docs/notes/` into
  `docs/notes/archive/`.
- Modify: `CHANGELOG.md`.

**Interfaces:**

- Preserves: `docs/notes/GPS_DATA_FORMAT_EXPLANATION.md` as evergreen reference,
  `docs/notes/README.md`, and the current live spec/plan until final completion.

- [ ] **Step 1: Record only demonstrated skill gaps**

Review how the adapted native-repro and upstream-issue skills performed during
Tasks 1-3. If a recurring output-audit workflow remains uncovered, record its
trigger, required inputs, hand-off boundaries, and why repository instructions
are insufficient. Otherwise, add no new note or skill.

- [ ] **Step 2: Archive completed implementation material**

Use Git-aware moves for the four completed 2026-08-01/02 spec-plan pairs, plus
`DIAGNOSTICS_USEFULNESS_RECOMMENDATIONS.md` and
`CHECK_MODELS_MONOLITH_COMPRESSION_REVIEW.md`. Do not archive evergreen reference
documentation. After final implementation completion, move this plan/spec into
the same archive hierarchy.

- [ ] **Step 3: Update the changelog and run prescribed formatting/lint order**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make format
make -C src lint-fix
make lint
```

Review every automated change and ensure tracked `src/output/` files are still
only the user’s original run changes.

- [ ] **Step 4: Run focused and full verification**

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
pytest src/tests/test_quality_analysis.py src/tests/test_jsonl_output.py src/tests/test_metrics_modes.py src/tests/test_report_generation.py -q
make quality
```

Expected: zero test, typing, lint, Markdown, shell, security, or dependency-policy
failures. Record the pre-existing `datasets`/`fsspec` environment warning
separately if `tools.validate_env` still reports it.

- [ ] **Step 5: Review, archive the completed live plan/spec, and commit**

Inspect scoped diffs and verify no generated run artifact was staged. Move this
completed plan/spec into `docs/notes/archive/superpowers/`, then run scoped
Markdown lint and `git diff --check` excluding raw generated evidence.

```bash
git add CHANGELOG.md .github .agents docs src/check_models.py src/tests
git commit -m "fix: improve output assessment and report ergonomics"
git push -u origin codex/improve-output-assessment-reports
```
