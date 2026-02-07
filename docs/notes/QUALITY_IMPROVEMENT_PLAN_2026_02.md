# Quality Improvement Plan — check_models

*Updated: 2026-02-06 — Removed all completed items. See git history for the original full plan.*

## Completed Items (summary)

All **High** and most **Medium** priority items have been addressed:

- ~~Fix stale make target references in docs~~ — CONTRIBUTING.md cleaned up
- ~~Unify CI action versions, add concurrency groups~~ — both workflows unified (checkout@v4, setup-python@v5, setup-node@v4), concurrency groups added, artifact upload on failure, pip cache in dependency-sync
- ~~Fix KeyboardInterrupt/SystemExit/fatal exception handlers~~ — warning+exit(130), bare raise, exc_info=True
- ~~Add YAML config schema validation~~ — warns on unknown threshold keys and top-level sections
- ~~Add unit tests for pure-logic functions~~ — 37 tests in test_pure_logic_functions.py
- ~~Align temperature default~~ — DEFAULT_TEMPERATURE=0.0 (greedy, matches upstream)
- ~~Clean up docs/notes/~~ — 30 files archived to docs/notes/archive/
- ~~Add prefill_step_size CLI flag~~ — wired through to generate()
- ~~Deduplicate JSONL~~ — v1.1 format with metadata header line
- ~~Remove duplicated IMPLEMENTATION_GUIDE sections~~ — merged Error Handling and Markdown Linting
- ~~Rename _apply_exclusions to apply_exclusions~~ — public API for testability
- ~~Add bare URL wrapping in markdown reports~~ — _wrap_bare_urls() + 4 tests

---

## Remaining Items

All items have been completed. This plan is now fully resolved. ✅

---

## Recently Completed Items

### (i) Documentation

- [x] Fix `./setup_conda_env.sh` path in copilot-instructions.md and src/README.md
  — changed to `bash src/tools/setup_conda_env.sh` (3 refs in copilot-instructions, 2 in src/README)
- [x] Add `quality-strict` and `install-markdownlint` targets to root Makefile
  — delegate to `$(MAKE) -C $(SRC) <target>`

### (ii) CI

- [x] Add npm caching to CI quality workflow
  — `cache: 'npm'` + `cache-dependency-path: src/package-lock.json` in `setup-node@v4`
  — generated `src/package-lock.json` for the lockfile reference

### (iii) Local Testing & Maintenance

- [x] Expose commonly-used `src/Makefile` targets in root Makefile
  — added `quality-strict` and `install-markdownlint` targets

### (v) Output Usefulness

- [x] Add `# generated_at: ...` metadata comment line at top of TSV output
  — `generate_tsv_report()` now writes a `# generated_at: <timestamp>` line before data
- [x] Add `error_type` and `error_package` columns to TSV for failed models
  — new columns appended to TSV header; populated from `ModelResult.error_type`/`error_package`
  — existing TSV tests updated to skip the metadata comment line

### (vi) Consistency with mlx-vlm

- [x] Add `--revision` CLI flag for model version pinning
  — `ProcessImageParams.revision`, wired through `_load_model()` → `mlx_vlm.utils.load(revision=...)`
- [x] Add `--adapter-path` CLI flag for LoRA adapter support
  — `ProcessImageParams.adapter_path`, wired through `_load_model()` → `mlx_vlm.utils.load(adapter_path=...)`

### (vii) Test Suite

- [x] Add report generation tests for empty/all-failed/mixed inputs
  — 11 tests in `test_report_generation.py` (HTML, Markdown, TSV edge cases)
- [x] Remove duplicated HF cache setup in `test_model_discovery.py`
  — inline setup removed; `conftest.py` handles HF cache env
- [x] Add mock-based test for `process_image_with_model()`
  — 4 tests in `test_process_image_mock.py` (success, TimeoutError, ValueError, OSError)

### (viii) Script Functionality

- [x] `--folder` and `--image` mutually exclusive group
  — already existed in argparse (confirmed at line ~8473); no change needed
- [x] Warn when default folder doesn't exist
  — `main_cli()` now logs `logger.warning(...)` when `DEFAULT_FOLDER` is missing

### (ix) Other

- [x] Add `CHANGELOG.md`
  — Keep a Changelog 1.1.0 format with full [Unreleased] section
