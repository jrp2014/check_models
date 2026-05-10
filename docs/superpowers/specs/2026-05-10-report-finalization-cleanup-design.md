# Report Finalization Cleanup Design

## Goal

Streamline the report-generation and finalization path in `src/check_models.py` by consolidating report artifact metadata, resolving output paths once, and making finalization easier to read and test without changing the core benchmark behavior.

## Scope

This cleanup covers the code that runs after model processing has produced `PerformanceResult` values. It includes report generation, report path logging, history/diagnostics setup, stale-report cleanup, and tests/docs for those behaviors.

This cleanup does not split `src/check_models.py`, rewrite the report renderers, change model execution, change CLI flags, or remove production output artifacts from version control.

## Architecture

Keep the existing single-file monolith, but introduce a small internal data structure for generated artifacts. Each artifact entry should describe the report key, resolved path, user-facing label, and callable used to generate it when applicable. Finalization will build this artifact list once, ensure parent directories exist, run jobs in order, and log generated paths from the same source of truth.

The data-driven structure should remain private to the report/finalization section. It should use existing dataclasses and type aliases where possible and avoid a larger abstraction than the cleanup needs.

## Components

- `ReportGenerationInputs`: keep as the main finalization payload, but prefer resolved paths on it where it already has dedicated fields.
- New private artifact metadata type: represent each generated report/log artifact with a concise typed container.
- `_generate_reports_and_log_outputs`: replace parallel tuples for jobs and labels with one artifact list.
- `finalize_execution`: reduce repeated `.resolve()` calls, centralize output directory creation, and keep history/diagnostics sequencing intact.
- Existing report generators: keep their public signatures and rendering behavior stable unless a small wording/order cleanup is needed.

## Data Flow

1. `finalize_execution` computes system info, runtime fingerprint, report context, and resolved output paths.
2. `finalize_execution` creates output directories for all report files that will be written.
3. `_generate_reports_and_log_outputs` receives resolved paths and builds the artifact list.
4. Each report job runs in a deterministic order: HTML, Markdown, gallery Markdown, review Markdown, TSV, JSONL.
5. Report failures still append a JSONL failure record for individual report failures where that behavior already exists.
6. Success logging lists generated report paths, log path, and environment path using artifact metadata instead of a second hard-coded tuple.
7. History append, diagnostics generation, repro-bundle pruning, and stale top-level report cleanup keep their existing order.

## Error Handling

Report generation should preserve the existing fail-soft behavior: individual HTML/Markdown/gallery/review failures are logged and recorded to JSONL without preventing later artifacts from attempting to generate. TSV and JSONL generation should continue to be handled by the surrounding report-generation error guard.

Path creation failures should still surface through the existing report/finalization exception logging rather than being silently ignored.

## Output/Workflow Changes

Small user-facing cleanup is acceptable:

- Report artifact logging may be reordered or relabeled for consistency.
- Log labels should use one consistent alignment style.
- No report content section should be removed.
- No CLI flag or default path should change in this pass.

## Tests

Update existing tests only:

- `src/tests/test_metrics_modes.py`: cover configured log/env path logging and pruning behavior after path resolution changes.
- `src/tests/test_report_generation.py`: cover report generation with supplied report context and artifact-link behavior if labels/order change.
- Run focused tests for report/finalization changes before the full quality gate.

Expected verification commands:

```bash
make test
make quality
```

## Documentation

Update `CHANGELOG.md` under `[Unreleased]` with the refactor. Update `src/README.md` or `docs/IMPLEMENTATION_GUIDE.md` only if user-facing report artifact labels or workflow wording changes.

## Non-Goals

- No broad CLI parser restructuring.
- No model execution refactor.
- No generated artifact git policy change.
- No standalone test scripts.
- No new runtime dependency.
