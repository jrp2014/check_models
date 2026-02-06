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

### Priority Summary

| Priority | Item | Area |
| -------- | ---- | ---- |
| **Medium** | Fix `./setup_conda_env.sh` path in root README | (i) |
| **Medium** | Add npm caching to CI quality workflow | (ii) |
| **Low** | Add `--revision` CLI flag for model version pinning | (vi) |
| **Low** | Add `--adapter-path` CLI flag for LoRA support | (vi) |
| **Low** | Add `generate_html_report()` / `generate_markdown_report()` tests | (vii) |
| **Low** | Add `mutually_exclusive_group` for `--folder`/`--image` | (viii) |
| **Low** | Warn when default folder doesn't exist | (viii) |
| **Low** | Add TSV metadata header comment | (v) |
| **Low** | Add `error_type`/`error_package` columns to TSV | (v) |
| **Low** | Add a `CHANGELOG.md` | (ix) |

---

### (i) Documentation

- [ ] Fix `./setup_conda_env.sh` path in root README (file is at `src/tools/setup_conda_env.sh`)
- [ ] Add `quality-strict` target to root Makefile or remove remaining docs references

### (ii) CI

- [ ] Add npm caching (`cache: 'npm'` in `setup-node@v4`)

### (iii) Local Testing & Maintenance

- [ ] Consider exposing commonly-used `src/Makefile` targets in root Makefile (or documenting the `make -C src` pattern)

### (v) Output Usefulness

- [ ] Add `# generated_at: ...` comment line at top of TSV output
- [ ] Add `error_type` and `error_package` columns to TSV for failed models

### (vi) Consistency with mlx-vlm

- [ ] Add `--revision` CLI flag for model version pinning
- [ ] Add `--adapter-path` CLI flag for LoRA support

### (vii) Test Suite

- [ ] Add `generate_html_report()` / `generate_markdown_report()` tests with empty/all-failed inputs
- [ ] Remove duplicated HF cache setup in `test_model_discovery.py`
- [ ] Add mock-based test for `process_image_with_model()`

### (viii) Script Functionality

- [ ] Add `mutually_exclusive_group` for `--folder` and `--image` (or document precedence)
- [ ] Warn when using the default folder and it doesn't exist

### (ix) Other

- [ ] Add a `CHANGELOG.md`
