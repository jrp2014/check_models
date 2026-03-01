# Changelog

Notable changes to this project will be documented in this file.

## [Unreleased]

### Added

### Changed

- Refactored `diagnostics.md` output structure for improved issue triage utility:
  - merged 'Potential Stack Issues' into a sub-section of 'Harness/Integration
    Issues' to reduce redundancy;
  - replaced the generic attachment guidance for JSON repro bundles with explicit
    direct GitHub repository links to the `output/repro_bundles/` directory.
- Tightened default cataloging prompt generation in `src/check_models.py`:
  - now requires strict three-section output with explicit anti-CoT and
    anti-verbatim-copy rules;
  - keeps `Context:` metadata hints concise and tagged as high-confidence
    hints rather than instruction text.
- Strengthened output-quality heuristics for prompt-contract enforcement:
  - added explicit flags for missing sections, title length, description
    sentence count, keyword count, and keyword-duplication violations;
  - added reasoning/prompt-echo leakage detection and context-regurgitation
    (`context-echo`) detection;
  - improved context-ignorance matching with alias support (for example
    `UK` vs `United Kingdom`) and filtering of non-semantic prompt-label terms.
- Improved quality-issue serialization robustness: JSONL list conversion now
  preserves single issue items that contain commas inside parenthesized detail
  payloads (for example repetitive phrase previews).
-- Polished generated report outputs for faster maintainer triage without
  changing core runtime behavior:
  - added a compact `Action Summary` near the top of `diagnostics.md` with
    explicit owner labels and next actions;
  - expanded diagnostics reproducibility guidance to include both exact rerun
    commands and portable dependency/import probes that do not require local
    image assets (now centralized in a single portable triage section instead
    of repeated per failure cluster);
  - added a concise `Action Snapshot` near the top of `results.md` to separate
    framework/runtime failures from low-utility model watchlist signals.
- Updated local MLX build integration in `src/tools/update.sh` to support
  explicit `MLX_METAL_JIT` pass-through via `-DMLX_METAL_JIT=<ON|OFF>` when
  requested, while leaving MLX's default behavior untouched when unset; also
  refreshed docs that previously referenced the older
  `MLX_BUILD_METAL_KERNELS` mapping.
- Hardened local-build detection in `src/tools/update.sh` for `mlx-lm` and
  `mlx-vlm`:
  - verifies editable install origin paths against local repo paths after local
    rebuilds (instead of relying on version strings);
  - preserves editable installs when deciding whether to skip PyPI MLX
    ecosystem upgrades, even when package versions look like release versions.
- Expanded local stub generation defaults to include `transformers` alongside
  `mlx_lm`, `mlx_vlm`, and `tokenizers` (`tools/generate_stubs.py`,
  `tools/update.sh`, and README command docs).
- Tightened type-checker stub integration:
  - mypy now prefers following generated stubs for `mlx_lm`, `mlx_vlm`,
    `transformers`, and `tokenizers` (`follow_imports = "silent"` override);
  - quality checks now emit explicit warnings when expected stub packages are
    missing or contain invalid syntax.
- Reduced upstream stubgen noise during `transformers` stub generation:
  known non-actionable `auto_docstring` diagnostics are now suppressed and
  replaced with a concise suppression count, while actionable/non-zero-exit
  stubgen output is still surfaced.
- Improved failed-model summary lines in `check_models.log` to include decoded
  maintainer hints (`owner≈... | component=... | likely=...`) plus a compact
  normalized symptom excerpt, making canonical error codes actionable without
  consulting internal token mappings.
- Raised the project Transformers floor to `>=5.2.0` and aligned packaging,
  runtime checks, and docs/tests to that policy.
- Aligned preflight package-floor diagnostics in `src/check_models.py` with
  current upstream dependency declarations from `mlx-vlm` and `mlx-lm`
  repositories (instead of stricter ad-hoc floors), reducing false-positive
  compatibility warnings.
- Updated backend-guard behavior for Transformers integration: when
  `MLX_VLM_ALLOW_TF` is not set, `check_models.py` now applies both legacy
  `TRANSFORMERS_NO_*` and compatibility `USE_*` env guards, and diagnostics now
  explicitly report when newer Transformers versions ignore both families.
- Updated `src/tools/update.sh` so local MLX builds apply `MLX_METAL_JIT`
  through MLX's current CMake build flag
  (`CMAKE_ARGS=-DMLX_BUILD_METAL_KERNELS=<ON|OFF>`), ensuring the selected
  kernel mode is honored during `pip install -e .`.
- Audited inline comments in `src/check_models.py` and removed stale
  refactor-history notes that no longer describe current behavior, while
  keeping explanatory comments for runtime/error-handling decisions.
- Updated maintainer map in `.github/copilot-instructions.md` to reflect the
  current monolith/test sizes and refreshed function line anchors.
- Normalized `src/README.md` command examples to use
  `python -m check_models`, matching the documented package entrypoint.
- Further compressed `src/check_models.py` (about 200 lines) by deduplicating
  EXIF date/time extraction paths, centralizing special-token leak pattern
  tables, and trimming verbose internal docstrings while preserving behavior.
- Diagnostics report generation now emits clearer issue-facing sections:
  `To reproduce` uses a single repro bullet, model output is shown inline when
  available, and technical traceback/captured logs are grouped in one
  collapsible `Detailed trace logs` section.
- Diagnostics report layout now surfaces `Priority Summary` near the top for
  faster triage, while moving the full `Environment` table near the bottom
  (just before reproducibility details).
- Terminal alignment now manages Unicode display width via `wcwidth` with
  safe fallback behavior, improving centered headers and metric-label padding
  when wide glyphs/emoji appear in output.
- Tightened model-generation typing in `src/check_models.py` by reducing
  ambiguous `Any` usage in `_load_model` / `_run_model_generation` where
  upstream function signatures allow safe narrowing.
- Terminal `Model Comparison (current run)` table now explicitly right-aligns
  numeric columns (TPS/timing/memory) and left-aligns text columns for easier
  visual scanning.
- Improved dependency-management tooling:
  - `src/tools/check_outdated.py` now uses JSON output with timeout/network-aware
    handling, and groups results into pyproject-managed vs unmanaged packages.
  - `src/tools/validate_env.py` now parses dependency specs robustly (including
    extras syntax) and validates installed versions against declared constraints.
- Stub generation now targets a broader default set (`mlx-lm`, `mlx-vlm`,
  `tokenizers`) and the quality gate runs a stub preflight before type checks.
- CI MLX core stubs are now written to repo-local `typings/` instead of
  mutating `site-packages`, improving reproducibility across runs.
- Refactored diagnostics/reporting internals for readability and lower
  duplication without intended behavioral changes:
  - removed single-use diagnostics wrappers and unused model cleanup state;
  - deduplicated diagnostics list + traceback normalization paths;
  - centralized diagnostics prose mappings;
  - simplified repro command assembly.
- Simplified diagnostics failure-cluster filing guidance so it only includes
  the repro command bullet; full traceback/captured-output diagnostics remain
  available in the existing collapsible sections.
- Aligned runtime-dependency preflight behavior between CLI and diagnostics:
  when core runtime packages are unavailable, the CLI now logs a structured
  environment-failure message and writes a minimal `diagnostics.md` focused on
  the missing dependencies and environment fingerprint instead of per-model
  failures.
- Centralized diagnostics configuration (history depth, snippet lengths,
  traceback tail lines, and cluster thresholds) into a single
  `DiagnosticsConfig` struct near the diagnostics helpers to make future tuning
  easier.
- Added concise one-line micro-summaries at the top of the Harness/Integration
  Issues and Long-Context/Stack Issues sections in `diagnostics.md` to improve
  scanability in large reports.
- Updated the `Models Not Flagged` diagnostics subsection for successful models
  with non-fatal quality issues:
  - renamed the warnings bucket to `Ran, but with quality warnings`;
  - added a per-model one-line warning summary derived from already captured
    quality-analysis signals.
- Optimized hot paths in `src/check_models.py`:
  - Hugging Face cache scans are now reused via a per-run cache helper.
  - Quality analysis is reused when `PerformanceResult.quality_analysis`
    already exists, avoiding repeated `analyze_generation_text()` calls.
  - Diagnostics generation now computes failure clusters once and reuses them
    across sections.

### Fixed

- Fixed Markdown report generation so `results.md` avoids markdownlint
  violations from prompt/error rendering:
  - prompt output now uses a plain fenced code block (not blockquote-wrapped)
    to prevent `MD028/no-blanks-blockquote` when prompt text contains blank
    lines;
  - gallery error prose now escapes emphasis markers in non-URL segments so
    identifiers like `LanguageModel.__call__()` do not trigger
    `MD050/strong-style`.

- Replaced a brittle type-only dependency on internal Transformers module
  `transformers.tokenization_python.PythonBackend` with a stable `Any` cast at
  the mlx-vlm call boundary, avoiding reliance on non-public import paths.
- Fixed `src/Makefile` `ci` target to use available commands (ruff + mypy +
  dependency sync check) instead of referencing removed tooling.
- Added `--check` mode to `src/tools/update_readme_deps.py` so CI/developers can
  verify README dependency block sync without rewriting files.
- Fixed noisy accidental pasted output in `src/tools/update.sh` banner section.
- `tools.validate_env` now treats the known `pip check` Torch
  "not supported on this platform" message as a warning (non-fatal), while still
  failing on real dependency inconsistencies.
- Fixed `_load_model` return typing to align with runtime processor type,
  eliminating `ty` `invalid-return-type` failures in CI/local quality checks.
- Improved diagnostics issue-report readability/safety by using clearer prose and
  escaping token-leak snippets for Markdown/HTML-safe rendering.
- Stabilized terminal summary-table alignment by sanitizing non-ASCII note
  glyphs (for example warning emoji) in the model comparison table output.

## [0.2.0] - 2026-02-15

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
- Reduced `check_models.py` redundancy in model-issue summarization by removing
  dead `context_ignored` summary output paths and consolidating quality/delta
  bucketing flow while preserving report/log semantics.
- Added maintainer-oriented monolith guidance to
  `docs/IMPLEMENTATION_GUIDE.md` describing refactor order, correctness-vs-
  performance boundaries, and practical navigation/checklist steps for
  `check_models.py`.
- Further reduced duplicated summary-rendering logic by introducing shared
  top-performer/resource metric collectors reused by both HTML and Markdown
  issue summaries, keeping output semantics unchanged while trimming repeated
  key checks/formatting branches.
- Reused a shared quality-issue section collector for HTML and Markdown
  summaries, reducing duplicate failed/repetitive/hallucination/formatting
  extraction logic while preserving section content and emphasis.

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

## [0.1.1] - 2026-02-15

### Changed

- Compressed `src/check_models.py` logic (~45 lines removed) by deduplicating EXIF time extraction, quality-issue formatting, and token-leakage pattern data.
- Refactored `src/tools/check_outdated.py` to remove redundant string parsing logic and improve table detection.
- Improved test robustness by replacing flaky `time.sleep()` calls with deterministic `os.utime()` timestamp updates in `conftest.py` and image workflow tests.
- Reverted CI test skips for missing dependencies; the suite now enforces a hard failure if `mlx-vlm` is missing, ensuring environment integrity.
- Final run summary now reports configured `--output-log` and `--output-env` paths
  instead of always showing default log file locations.
- Synced documentation with current CLI behavior and outputs, including:
  `--output-diagnostics`, `--revision`, `--adapter-path`, and
  `--prefill-step-size`, plus `results.history.jsonl` / `diagnostics.md`.
- Updated quality and bot guidance docs (`AGENTS.md`, Copilot instructions,
  contributing docs, and pre-commit hook naming) to match the current CI gate.
- Improved `src/tools/run_quality_checks.sh` conda initialization by resolving
  the base path via `conda info --base` before fallback probe paths.

### Fixed

- Fixed `check_outdated.py` logic to correctly identify outdated package tables.

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
