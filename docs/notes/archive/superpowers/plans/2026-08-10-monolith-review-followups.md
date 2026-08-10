# Monolith Review Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the verified diagnostic drift and duplicated configuration paths identified by the monolith review without undertaking speculative structural changes.

**Architecture:** Keep `src/check_models.py` as the intentional monolith, but introduce small shared data builders at existing responsibility boundaries. Primary runs and differential reruns will construct `ProcessImageParams` through one helper; console modes will consume one warning representation; native repro flags will be rendered from declarative specifications; Markdown/HTML chooser prose and real/dry-run model selection will share their semantic sources.

**Tech Stack:** Python 3.13, pytest, argparse, dataclasses, Rich, Ruff, mypy, ty, pyrefly.

## Global Constraints

- Activate the `mlx-vlm` conda environment before Python or Make commands.
- Preserve the intentional single-file `src/check_models.py` architecture.
- Add tests only to existing `src/tests/test_*.py` files.
- Keep validation output in `tmp_path`; never rewrite tracked `src/output/` artifacts.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Use one red-green cycle per behavioural task, then run formatting, lint and `make quality`.

---

### Task 1: Preserve differential-rerun configuration

**Files:**

- Modify: `src/check_models.py`
- Test: `src/tests/test_jsonl_output.py`

**Interfaces:**

- Produces: `_process_image_params_from_args(args, *, model_identifier, image_path, prompt, max_tokens=None, temperature=None, timeout=None, verbose=None) -> ProcessImageParams`.
- Consumes: normalized argparse fields already used by `process_models`.

- [x] Add a test that patches `process_image_with_model`, invokes `_run_differential_reruns`, and asserts that revision, adapter, resize, EOS, special-token, processor, thinking and KV settings survive while prompt, token limit, temperature, timeout and verbosity use triage overrides.
- [x] Run that test and confirm it fails because those fields are currently omitted.
- [x] Add the shared params builder and use it from both `process_models` and `_run_differential_reruns`.
- [x] Run the focused JSONL tests and confirm they pass.

### Task 2: Single-source console quality warnings

**Files:**

- Modify: `src/check_models.py`
- Test: `src/tests/test_metrics_modes.py`

**Interfaces:**

- Produces: `_quality_warning_messages(analysis, generated_tokens) -> tuple[str, ...]` and `_log_quality_warnings(...) -> None`.
- Consumes: `GenerationQualityAnalysis` and the generated-token count.

- [x] Add tests showing that both preview and verbose success modes report incomplete thinking, instruction echo, catalog preamble, unexpected tokens, missing sections, repetition and evidenced token-cap truncation.
- [x] Run the focused tests and confirm verbose mode fails the parity assertions.
- [x] Move warning decisions into the shared helper while retaining neutral full-budget information in preview output.
- [x] Run the focused metrics tests and confirm they pass.

### Task 3: Consolidate native repro flags

**Files:**

- Modify: `src/check_models.py`
- Test: `src/tests/test_report_generation.py`

**Interfaces:**

- Produces: declarative optional-pair, optional-sequence, non-default-pair and boolean-flag specifications consumed by `_build_native_mlx_vlm_cli_tokens`.
- Preserves: `build_native_mlx_vlm_repro_command_spec(...) -> ReproCommandSpec`.

- [x] Add a direct unit test asserting all supported retained native flags, portable adapter paths, resolved-revision precedence, and omission of default-valued flags.
- [x] Add a load-crash test asserting a one-line minimal command with prompt `x` and `--max-tokens 8`.
- [x] Run both tests and confirm they fail against the manual/current load-repro implementation.
- [x] Replace repeated flag branches with declarative specifications and add a minimal-load mode used only before image decoding.
- [x] Run the focused report-generation tests and confirm they pass.

### Task 4: Align chooser prose and model selection

**Files:**

- Modify: `src/check_models.py`
- Test: `src/tests/test_markdown_formatting.py`
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_model_discovery.py`

**Interfaces:**

- Produces: shared chooser explanation sentences escaped by each renderer.
- Produces: `_selected_model_identifiers(args, *, allow_empty=False) -> list[str]` used by execution and dry-run presentation.

- [x] Add report tests requiring both Markdown and HTML to explain `Prefill/first` and cross-attention prompt-token limitations.
- [x] Add a dry-run test requiring the same ineffective-exclusion warning and filtered identifiers as real selection.
- [x] Run focused tests and confirm the HTML prose and dry-run warning assertions fail.
- [x] Extract shared chooser prose and shared model-selection logic, retaining format-specific tables and dry-run presentation.
- [x] Run the focused report and discovery tests and confirm they pass.

### Task 5: Narrow cleanup

**Files:**

- Modify: `src/check_models.py`
- Modify: `src/tests/test_model_discovery.py`
- Modify: `src/tests/test_html_formatting.py`

**Interfaces:**

- Removes: unused `TimingStrategy`, timer injection, `HEADER_SPLIT_LENGTH`, `NUMERIC_FIELD_PATTERNS`, `is_numeric_value`, and `is_numeric_field`.
- Produces: one logger binding, one image-verification helper, one HTML entity-ampersand regex, and one triage prompt constant.

- [x] Remove the numeric-helper change-detector tests and unused public exports, then run their former test files.
- [x] Replace defensive `getattr` calls on typed prompt/image dataclasses with direct optional access.
- [x] Define the logger once, remove the unused timer protocol/seam and unused header constant.
- [x] Extract the duplicated image verification tail and single-source the ampersand regex and triage prompt.
- [x] Run pure-logic, image-workflow, EXIF, formatting and process-image tests.

### Task 6: Documentation and verification

**Files:**

- Modify: `CHANGELOG.md`
- Move after completion: `docs/superpowers/plans/2026-08-10-monolith-review-followups.md` to `docs/notes/archive/superpowers/plans/`

- [x] Record the rerun fidelity, warning parity, repro rendering, chooser/dry-run consistency and compact cleanup under `[Unreleased]`.
- [x] Run `make format`, `make -C src lint-fix`, `make lint`, `bash src/tools/run_commit_hygiene.sh`, and `make quality`.
- [x] Confirm `git diff --check`, inspect the final diff for unrelated output churn, and archive this completed plan.
