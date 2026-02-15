# Changelog

Notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- New preflight package-risk diagnostics in `check_models.py` to surface common
  upstream compatibility issues early (MLX/MLX-VLM/MLX-LM/Transformers version
  mismatches and known problematic package states).
- End-of-run model comparison summary in `check_models.log` now includes a
  compact per-model table plus ASCII charts (TPS, overall efficiency, failure
  stage frequency) to make same-run model triage faster.
- History comparison output now includes run-over-run tabular deltas and
  transition-oriented ASCII charts to highlight regressions/recoveries/new or
  missing models across successive runs.
- Additional diagnostics stack-signal and long-context-breakdown analysis
  sections to better triage quality and failure patterns.
- Phase-aware failure attribution in model execution (`import`, `model_load`,
  `tokenizer_load`, `processor_load`, `model_preflight`, `prefill`, `decode`)
  so reports can identify the first failing runtime stage.
- Canonical failure metadata for each failed model:
  machine-readable `error_code` plus stable `error_signature` for clustering
  related failures across models/runs.
- Per-failure reproducibility bundle export (`output/repro_bundles/*.json`)
  with args, environment fingerprint, prompt/image hashes, traceback, captured
  output, and exact rerun command.
- Diagnostics report now emits copy/paste-ready issue templates per failure
  cluster, including canonical signature, minimal repro command, environment
  fingerprint, and likely upstream issue tracker.
- Model preflight validators for tokenizer/processor/config/snapshot layout to
  detect packaging and compatibility defects before generation begins.

### Changed

- Hardened runtime dependency handling for core MLX stack:
  `mlx`, `mlx-vlm`, and `mlx-lm` are now treated as required for model execution.
- Added explicit early package/runtime preflight in
  `src/tools/run_quality_checks.sh` to fail fast when required runtime deps are
  missing or broken.
- Updated CLI execution flow so argument/path validation and `--dry-run`
  behavior are evaluated before runtime dependency hard-fail, while still
  enforcing a hard stop before any model inference starts.
- Improved type-checking signal quality without globally hiding import/stub
  issues:
  - `pyright` missing import/stub diagnostics now emit warnings.
  - `mypy` uses targeted third-party overrides instead of global
    `ignore_missing_imports`.
  - `pyrefly` no longer uses wildcard missing-import ignore.
  - `ty` unresolved-import is now warning-level (visible, non-blocking).
- Strengthened E2E runtime gating to require usable `mlx` + `mlx-vlm` +
  `mlx-lm` for inference smoke tests.
- Improved CI MLX stub-generation step robustness to tolerate missing/generated
  stub-path variance as a non-fatal warning (typing-accuracy degradation only).
- Improved prompt-context compaction and keyword-hint summarization for long
  metadata inputs to reduce prompt bloat while preserving useful grounding.
- Improved diagnostics report reproducibility command construction and captured
  output sanitization for cleaner issue filing.
- Diagnostics reports now include preflight compatibility warnings in
  `diagnostics.md` (with likely package ownership and suggested issue trackers),
  and surface those warnings in the priority summary.
- Refactored portions of `check_models.py` for readability/maintainability
  without intended behavioral change.
- Tightened report/summary typing with explicit TypedDict/type-alias structures
  for model issue summaries and aggregate performance statistics.
- Failure clustering in diagnostics now keys off canonical signatures instead of
  raw message text heuristics, improving cross-model bucketing stability.
- JSONL result rows now include `failure_phase`, `error_code`, and
  `error_signature` fields (metadata format version bumped to `1.2`).

### Fixed

- Fixed CI regression where runtime dependency hard-fail masked CLI argument and
  folder validation tests by firing too early in startup.
- Fixed diagnostics markdown formatting/lint edge cases in generated reports.
- Fixed `reportTypedDictNotRequiredAccess` lint failures by replacing direct
  optional-key TypedDict indexing with safe `.get(..., default)` patterns in
  summary formatters and JSONL tests.
- Fixed remaining `ModelIssueSummary` optional-key TypedDict access warnings in
  analysis helpers by replacing `setdefault(...)` read-path usage with explicit
  `get(...)` plus guarded initialization, preserving runtime behavior while
  satisfying strict Pylance/Pyright checks.
- Removed weak/avoidable lint suppressions by:
  - replacing shell word-splitting patterns with array-safe handling in quality
    and hook scripts;
  - replacing unnecessary test `type: ignore` usage with explicit type asserts.

## [0.1.1] - 2026-02-13

### Changed

- Final run summary now reports configured `--output-log` and `--output-env` paths
  instead of always showing default log file locations.
- Synced documentation with current CLI behavior and outputs, including:
  `--output-diagnostics`, `--revision`, `--adapter-path`, and
  `--prefill-step-size`, plus `results.history.jsonl` / `diagnostics.md`.
- Updated quality and bot guidance docs (`AGENTS.md`, Copilot instructions,
  contributing docs, and pre-commit hook naming) to match the current CI gate.
- Improved `src/tools/run_quality_checks.sh` conda initialization by resolving
  the base path via `conda info --base` before fallback probe paths.

## [0.1.0] - 2026-02-08

### Added

- `--revision` CLI flag for pinning model versions (branch, tag, or commit hash)
- `--adapter-path` CLI flag for applying LoRA adapter weights
- `--prefill-step-size` CLI flag wired through to `generate()`
- TSV metadata comment line (`# generated_at: <timestamp>`) at top of output
- `error_type` and `error_package` columns in TSV reports for programmatic triage
- Warning when the default image folder (`~/Pictures/Processed`) does not exist
- JSONL v1.1 format with shared metadata header (prompt, system info, timestamp)
- 37 unit tests for pure-logic functions (`test_pure_logic_functions.py`)
- 11 report-generation edge-case tests (`test_report_generation.py`)
- 4 mock-based `process_image_with_model` tests (`test_process_image_mock.py`)
- Append-only history JSONL (`results.history.jsonl`) with per-run regression/recovery
  comparison summary (always prints "Regressions" and "Recoveries" sections)
- IPTC/IIM metadata extraction (keywords, caption) via Pillow `IptcImagePlugin`
- XMP metadata extraction (dc:subject, dc:title, dc:description) via `Image.getxmp()`
- Windows EXIF XPKeywords extraction (UTF-16LE semicolon-delimited)
- Keyword merging across IPTC, XMP, and XP sources (deduplicated, order-preserved)
- Structured stock-photo cataloguing prompt with Title/Description/Keywords sections
  and keyword taxonomy guidance (subjects, concepts, mood, style, colors, use-case)
- Existing metadata seeded into prompt (description, title, keywords, GPS, date)
- `Pillow[xmp]` extra for XMP metadata support (pulls in `defusedxml` transitively)
- Diagnostics report (`output/diagnostics.md`) auto-generated when failures or
  harness issues are detected — structured for filing upstream GitHub issues
  against mlx-vlm / mlx / transformers with error-pattern clustering, full
  error messages, traceback excerpts, environment table, and priority summary
- `--output-diagnostics` CLI flag for specifying diagnostics report path
- YAML config schema validation — warns on unknown threshold keys
- `CHANGELOG.md` (this file)
- `quality-strict` and `install-markdownlint` targets in root Makefile
- `src/package-lock.json` for deterministic npm CI caching
...existing code...

### Changed

- `DEFAULT_TEMPERATURE` set to `0.0` (greedy/deterministic, matching mlx-vlm upstream)
- `--folder` and `--image` are now a mutually exclusive group in argparse
- Renamed `_apply_exclusions` → `apply_exclusions` (public API for testability)
- Bare URLs in generated Markdown reports are now auto-wrapped as `<URL>`
- Fixed `KeyboardInterrupt` / `SystemExit` / fatal exception handlers
- Unified CI action versions (`checkout@v4`, `setup-python@v5`, `setup-node@v4`)
  with concurrency groups and artifact upload on failure
- Added npm caching to CI quality workflow
- Fixed `setup_conda_env.sh` path references in documentation
  (was `./setup_conda_env.sh`, now `bash src/tools/setup_conda_env.sh`)
- Cleaned up VS Code settings (removed deprecated Python linting settings,
  aligned `typeCheckingMode`, fixed `launch.json` and `tasks.json`)
- Removed duplicated HF cache setup from `test_model_discovery.py` (now in `conftest.py`)
- Documented TSV and JSONL output format details in `src/README.md`
- Updated `QUALITY_IMPROVEMENT_PLAN_2026_02.md` — all items resolved

### Removed

- Duplicated sections in `IMPLEMENTATION_GUIDE.md` (Error Handling, Markdown Linting)
- 30 stale files archived from `docs/notes/` to `docs/notes/archive/`


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
