# Run Regression Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the harness defects exposed by the 2 August run and submit the independently reproduced Idefics3 stopping-token fix to `mlx-vlm`.

**Architecture:** Keep all harness behaviour in the existing `src/check_models.py` monolith and extend its typed quality/report records rather than adding a parallel report pipeline. Isolate the upstream fix in a separate `mlx-vlm` worktree and PR; do not move model-capability observations upstream without a native runtime reproduction.

**Tech Stack:** Python 3.13, mlx-vlm, Transformers tokenizers, Hugging Face cache metadata, pytest, Ruff, mypy, ty, Pyrefly, markdownlint, GitHub CLI.

## Global Constraints

- Activate the `mlx-vlm` Conda environment before every Python or Make command.
- Keep `src/check_models.py` as the intentional single-file monolith.
- Add tests only to existing `src/tests/test_*.py` files.
- Do not rewrite the committed `src/output/` snapshot.
- Preserve exact generated model text in machine and evidence artifacts.
- Apply only safe Ruff fixes routinely; inspect every unsafe fix before accepting it.
- Never use `uv` in either repository.

---

### Task 1: Accept valid custom processors and recover cached loads

**Files:**

- Modify: `src/tests/test_process_image_mock.py`
- Modify: `src/check_models.py` near `_load_model`, `_resolve_model_snapshot_path`, and `_run_model_preflight_validators`

**Interfaces:**

- Consumes: `ProcessImageParams`, `_has_external_connectivity_signal(...)`, and `_resolve_model_snapshot_path(...)`
- Produces: a load retry using a matching local snapshot after connectivity-only failure
- Produces: processor preflight based on callable/tokenizer interfaces, not an `image_processor` attribute

- [ ] **Step 1: Write failing processor and cache-retry tests**

Add tests that prove a callable Step-like processor with `tokenizer` and
`detokenizer` passes preflight without `image_processor`, and that `_load_model`
retries a connection-reset failure with a resolved local snapshot but does not
retry a model/config error or a mismatched requested revision.

```python
class _FakeStepProcessor(_FakeProcessor):
    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {}

def test_preflight_accepts_custom_image_processor_without_attribute() -> None:
    check_models._run_model_preflight_validators(
        model_identifier="org/step",
        processor=_FakeStepProcessor(),
        config={"model_type": "step3p7"},
    )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest src/tests/test_process_image_mock.py -k "preflight_accepts or load_model_retries" -q
```

Expected: the Step-like processor is rejected and the local retry is absent.

- [ ] **Step 3: Implement the minimal runtime fixes**

Remove the `image_processor is None` rejection. In `_load_model`, keep the first
Hub-ID call unchanged; on a connectivity exception only, resolve a local
snapshot, verify any requested immutable revision, and call `load()` once with
the snapshot path. Preserve `--force-download` by not falling back when it is
set.

- [ ] **Step 4: Verify GREEN**

Run the focused tests above plus the existing process-image mock file.

---

### Task 2: Record catalogue constraints and escaped turn boundaries

**Files:**

- Modify: `src/tests/test_quality_analysis.py`
- Modify: `src/tests/test_jsonl_output.py`
- Modify: `src/check_models.py` near `ObservationCode`, `GenerationQualityAnalysis`, `_collect_prompt_quality_signals`, `_configured_role_boundaries`, and `_observation_details`

**Interfaces:**

- Produces: `catalog_constraint_violation` in `ObservationCode`
- Produces: `title_word_count`, `keyword_count`, and `duplicate_keywords` detail fields
- Extends: `role_boundary_token_present` to configured `turn`, `message`, and `utterance` tokens

- [ ] **Step 1: Write failing literal contract tests**

Cover a four-word title, nineteen keywords, duplicate case-insensitive keywords,
a compliant result, and an Idefics-style `<end_of_utterance>` suffix. Assert
that only the defective results receive the new caveat and exact details.

```python
assert check_models._assess_result(result).usability == "usable_with_caveats"
assert "catalog_constraint_violation" in assessment.observations
assert result.quality_analysis.duplicate_keywords == ["halesworth"]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest src/tests/test_quality_analysis.py src/tests/test_jsonl_output.py -k "constraint or utterance" -q
```

- [ ] **Step 3: Implement mechanical validation**

Parse the already-recognised sections once, count title words with the existing
word-token convention, split keywords with `_split_catalog_keywords`, and
deduplicate by case-folded collapsed whitespace. Run these checks only when the
prompt requests the catalogue contract and all three sections are present.

Extend `_configured_role_boundaries` to match configured token names containing
`user`, `assistant`, `system`, `turn`, `message`, or `utterance` while preserving
the existing leading-assistant exemption.

- [ ] **Step 4: Verify GREEN**

Run the two complete focused test files.

---

### Task 3: Make the issue summary actionable and provenance-complete

**Files:**

- Modify: `src/tests/test_report_generation.py`
- Modify: `src/tests/test_jsonl_output.py`
- Modify: `src/check_models.py` near `RunIssueSummarySource`, `_load_run_issue_enrichment`, `_human_observation_labels`, `_run_issue_summary_surfaced_sections`, `_run_issue_summary_context_section`, and `_collect_check_models_provenance`

**Interfaces:**

- Produces: a readable catalogue-constraint label built from structured details
- Produces: `_run_issue_summary_failure_label(...) -> str`
- Extends: `RunIssueSummarySource` with optional producer provenance

- [ ] **Step 1: Write failing report/provenance tests**

Use hand-authored schema-2.0 fixtures to assert:

- a network-reset indeterminate row says `Network connection reset during model loading`;
- repeated/empty/truncated rows sort ahead of missing-field/count caveats;
- run context includes remote-code trust and producer version/revision/dirty;
- generated-output-only Git status is clean, while a source edit remains dirty;
- the summary states that `main` links are mutable and names the producer revision.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py -k "failure_reason or severity or provenance or mutable" -q
```

- [ ] **Step 3: Implement the report changes**

Load and narrow `producer` from run JSON. Use structured failure stage, phase,
code, exception type, and message to build a bounded plain-language fallback
when observations are empty. Sort rows by the index of their first display
priority, then usability and case-folded model name.

Call Git status with pathspec exclusions for tracked `src/output/` files when
computing producer dirtiness. Keep all other tracked changes significant.

- [ ] **Step 4: Verify GREEN**

Run both complete focused files and markdownlint against a temporary rendered
summary.

---

### Task 4: Reduce detailed crash-report environment noise

**Files:**

- Modify: `src/tests/test_report_generation.py`
- Modify: `src/check_models.py` in the direct issue-report environment builder

**Interfaces:**

- Consumes: existing component provenance and complete environment artifact path
- Produces: a bounded relevant-component table plus a canonical link to full environment evidence

- [ ] **Step 1: Add a failing direct-issue rendering test**

Assert that `mlx-vlm`, `mlx`, `transformers`, `tokenizers`, Python, macOS, chip,
and check_models revision remain visible; unrelated compiler/SDK fingerprints do
not; and the full environment artifact is linked.

- [ ] **Step 2: Verify RED**

Run the single direct-issue test.

- [ ] **Step 3: Narrow the rendered component subset and add the evidence link**

Reuse existing typed report blocks and canonical issue-link helpers. Do not
remove facts from `environment.log`, `run.json`, or diagnostics.

- [ ] **Step 4: Verify GREEN**

Run the complete report-generation test file.

---

### Task 5: Implement and test the upstream Idefics3 EOS fix

**Files:**

- Modify in an isolated `mlx-vlm` worktree: `mlx_vlm/models/idefics3/processing_idefics3.py`
- Modify in that worktree: `mlx_vlm/tests/test_processors.py`

**Interfaces:**

- Produces: an Idefics3 processor stop-token ID that the shared loader merges into `StoppingCriteria.eos_token_ids`

- [ ] **Step 1: Create the isolated upstream worktree and branch**

Create a `codex/idefics3-end-of-utterance-eos` branch from current upstream
`main` without modifying the maintainer's existing checkout.

- [ ] **Step 2: Write a failing processor-level test**

Use a fake tokenizer whose existing EOS IDs are `[128001, 128008, 128009]` and
whose `<end_of_utterance>` ID is `128258`. Assert the effective stopping IDs are:

```python
[128001, 128008, 128009, 128258]
```

and that duplicates are not introduced.

- [ ] **Step 3: Verify RED**

Run the narrow upstream processor test with pytest in the same Conda environment.

- [ ] **Step 4: Implement the smallest processor/loader contract**

Expose the processor-specific stop ID and merge it in the shared processor loader
after tokenizer/model EOS discovery. Preserve order and every configured EOS ID.

- [ ] **Step 5: Verify GREEN and native behaviour**

Run the focused processor tests, upstream formatting/pre-commit on edited files,
then repeat the cached native Idefics3 command without explicit `--eos-tokens`.
Expected output contains the three catalogue fields and no
`<end_of_utterance>`.

- [ ] **Step 6: Commit, push, and open the upstream PR**

Use a factual PR body containing the exact revision, native before/control
outputs, environment, and focused test command. Do not file a separate issue
unless maintainers require one.

---

### Task 6: Documentation, full verification, and check_models PR

**Files:**

- Modify: `CHANGELOG.md`
- Modify: `src/README.md` only where report semantics need user documentation

- [ ] **Step 1: Update `[Unreleased]` and user-facing report documentation**

Describe corrected processor compatibility, cache fallback, catalogue caveats,
escaped turn-token detection, and issue-summary provenance.

- [ ] **Step 2: Run prescribed formatting and lint sequence**

```bash
make format
make -C src lint-fix
make lint
bash src/tools/run_commit_hygiene.sh
```

- [ ] **Step 3: Run the full quality gate**

```bash
make quality
```

- [ ] **Step 4: Review the final diff and output hygiene**

Confirm `src/output/` is byte-unchanged, no unrelated files changed, every new
observation has typed JSON detail evidence, and both native ownership claims are
documented.

- [ ] **Step 5: Commit, push, and open the check_models PR**

The PR body should link the upstream Idefics3 PR and explain why the remaining
bad outputs are classified as model limitations rather than mlx-vlm defects.
