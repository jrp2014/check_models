# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
- YAML config schema validation — warns on unknown threshold keys
- `CHANGELOG.md` (this file)
- `quality-strict` and `install-markdownlint` targets in root Makefile
- `src/package-lock.json` for deterministic npm CI caching

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
