# Quality Improvement Plan — check_models

## Priority Summary

| Priority | Item | Area |
| -------- | ---- | ---- |
| **High** | Fix stale make target references in docs | (i) |
| **High** | Unify CI action versions, add concurrency groups | (ii) |
| **High** | Fix `KeyboardInterrupt`/`SystemExit` exception handlers | (iv) |
| **Medium** | Add YAML config schema validation (warn on unknown keys) | (iv) |
| **Medium** | Add unit tests for pure-logic functions (validate_model_identifier, _apply_exclusions, compute_* family) | (vii) |
| **Medium** | Align temperature default with upstream (0.0 vs 0.1) or document the delta | (vi) |
| **Medium** | Clean up `docs/notes/` — archive completed notes | (ix) |
| **Low** | Add `prefill_step_size` CLI flag | (vi) |
| **Low** | Deduplicate JSONL prompt/system_info across rows | (v) |
| **Low** | Remove duplicated IMPLEMENTATION_GUIDE sections | (i) |
| **Low** | Consolidate to 1-2 type checkers | (iii) |

---

## (i) Documentation & Agent Guidance

### Strengths

- `src/README.md` (1100+ lines) is exceptionally thorough.
- `.github/copilot-instructions.md` and `docs/IMPLEMENTATION_GUIDE.md` provide excellent agent guidance.

### Issues

1. **Stale make targets in docs** — `docs/CONTRIBUTING.md` references `make audit`, `make sync-deps`, `make check-outdated`, `make upgrade-deps`. None exist in the root Makefile; they may exist in the src Makefile as `make -C src <target>` etc., but docs describe them as root-level. Also `make quality-strict` is referenced in IMPLEMENTATION_GUIDE and `src/README.md` but doesn't exist in either Makefile.
2. **Duplicated sections in IMPLEMENTATION_GUIDE** — "Markdown Linting" appears twice (~L490 and ~L540); two consecutive `### Error Handling` headings near L300.
3. **Root README references `./setup_conda_env.sh`** but the file is at `src/tools/setup_conda_env.sh` — no symlink at root.
4. **33 files in `docs/notes/`** — all historical AI-generated review artifacts. Only `docs/notes/README.md` indexes them. Consider archiving or trimming to the 2-3 still-useful ones (`GPS_DATA_FORMAT_EXPLANATION.md`, `LOCAL_MLX_DEV_WORKFLOW.md`).
5. **Mild redundancy** between `copilot-instructions.md` and `IMPLEMENTATION_GUIDE.md` — the AI agent rules appear in both. Intentional for different audiences but increases maintenance burden.

### Actions

- [ ] Fix or remove stale make target references in `CONTRIBUTING.md` and `IMPLEMENTATION_GUIDE.md`
- [ ] Add `quality-strict` target to root Makefile or remove docs references
- [ ] Remove duplicated "Markdown Linting" and "Error Handling" sections in IMPLEMENTATION_GUIDE
- [ ] Fix `./setup_conda_env.sh` path in root README (or add a symlink)
- [ ] Archive `docs/notes/` historical files into a subfolder or prune

---

## (ii) GitHub CI Robustness

### Strengths

- Runs on `macos-15` (matching target platform), 10-minute timeout, pip caching in quality workflow.

### Issues

1. **Action version inconsistency** — `quality.yml` uses `checkout@v4` / `setup-python@v5` / `setup-node@v3`, while `dependency-sync.yml` uses `checkout@v6` / `setup-python@v6`. Should be consistent.
2. **`actions/setup-node@v3` is outdated** — current is v4.
3. **No `concurrency` groups** — pushing to a PR triggers both `push` and `pull_request`, running duplicate jobs that waste CI minutes.
4. **No artifact upload** — failed test logs / quality check outputs aren't preserved as CI artifacts for debugging.
5. **No npm cache** — `npm install --prefix src` runs every CI job without caching.
6. **Stub generation silently swallows errors** — `2>/dev/null || true` in the nanobind stubgen step hides failures.
7. **No pip cache in dependency-sync workflow** — unlike the quality workflow.

### Actions

- [ ] Unify action versions across both workflows (latest stable: checkout@v4, setup-python@v5, setup-node@v4)
- [ ] Add `concurrency` groups to both workflows (`concurrency: { group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: true }`)
- [ ] Add artifact upload step for quality check output on failure
- [ ] Add npm caching (or use `actions/setup-node@v4` with `cache: 'npm'`)
- [ ] Add pip cache to dependency-sync workflow
- [ ] Remove silent error suppression from stub generation step

---

## (iii) Local Testing & Maintenance

### Strengths

- Excellent environment auto-detection in `src/Makefile`, clean root Makefile with help output, well-structured `pyproject.toml`.

### Issues

1. **Root Makefile exposes `stubs` in help** but 13 useful targets from `src/Makefile` (`install-all`, `bootstrap-dev`, `install-markdownlint`, `test-cov`, `check-outdated`, `audit`, etc.) are only accessible via `make -C src <target>`.
2. **Four type checkers configured** (mypy, pyright, pyrefly, ty) — maintenance overhead is high. Most projects use one or two.
3. **`make quality` delegates to a bash script** (`src/tools/run_quality_checks.sh`) rather than composing make targets, making the CI quality step opaque.

### Actions

- [ ] Consider exposing commonly-used src targets in root Makefile (or documenting `make -C src` pattern)
- [ ] Evaluate consolidating to 2 type checkers (mypy + one modern alternative)
- [ ] Consider making `run_quality_checks.sh` call make targets instead of raw commands (for transparency)

---

## (iv) Python Code Quality

### Strengths

- Full type annotations, `from __future__ import annotations`, `Final` constants, `Protocol` for structural typing, `frozen=True` dataclasses, `raise ... from e` consistently, keyword-only parameters, no `# type: ignore` or `# noqa` suppressions anywhere.

### Issues

1. **8,680 lines in one file** — acknowledged as deliberate in IMPLEMENTATION_GUIDE. The ~15 quality detection functions (~L1336–2250) and 7 `compute_*` advanced metrics functions (~L2258–2600) are natural extraction candidates if the file grows further.
2. **`logger.exception()` on `KeyboardInterrupt`** at L8340 — logs a full traceback for a user Ctrl+C. Should be `logger.info()` or `logger.warning()`.
3. **`logger.exception()` on `SystemExit`** at L8343 — catches `SystemExit(0)` (normal exit) and logs with full ERROR-level traceback, then re-raises. Noisy for normal shutdown.
4. **Broad `(OSError, ValueError, RuntimeError)` catch** at L8345 uses `logger.critical()` without `exc_info=True`, hiding the traceback from the user.
5. **No YAML schema validation** in `load_quality_config()` — a typo in `quality_config.yaml` (e.g., `repetition_ration` instead of `repetition_ratio`) is silently ignored because `QualityThresholds.from_config()` filters to valid dataclass fields only. A warning for unrecognised keys would catch misconfigurations.
6. **Global mutable `QUALITY` singleton** at L287 is updated in-place via `setattr` in `load_quality_config()`. Works but is fragile.

### Actions

- [ ] Fix `KeyboardInterrupt` handler: `logger.info("Execution interrupted by user.")` + `sys.exit(130)`
- [ ] Fix `SystemExit` handler: remove it entirely (let it propagate naturally) or only catch non-zero exits
- [ ] Fix fatal error handler: add `exc_info=True` to `logger.critical()` call
- [ ] Add warning in `QualityThresholds.from_config()` for unrecognised YAML keys
- [ ] (Low) Track single-file size; extract modules if it exceeds ~10k lines

---

## (v) Output Usefulness

### Strengths

- Four output formats (MD, HTML, JSONL, TSV). Markdown report is exceptionally rich. HTML has filter controls and collapsible sections. JSONL is fully self-contained per record.

### Issues

1. **TSV lacks a metadata/header comment line** — no format version or generation timestamp (MD and HTML include these).
2. **Failed model rows in TSV are mostly empty** — all metric columns blank, error info only in the Output column. JSONL has structured `error_type`/`error_package` fields that TSV lacks.
3. **JSONL repeats identical `context.prompt` and system_info** in every line for a batch run — significant bloat for 40+ model runs.

### Actions

- [ ] Add a `# generated_at: ...` comment line at top of TSV output
- [ ] Add `error_type` and `error_package` columns to TSV for failed models
- [ ] (Low) Factor out repeated JSONL fields into a separate `_metadata.json` file or header line

---

## (vi) Consistency with mlx / mlx-lm / mlx-vlm

### Strengths

- All parameters passed to `load()`, `apply_chat_template()`, and `generate()` are valid and not deprecated. Type stubs match upstream accurately.

### Issues

1. **Temperature default mismatch** — check_models uses `DEFAULT_TEMPERATURE = 0.1`; upstream mlx-vlm defaults to `0.0` (greedy/deterministic). For a benchmarking tool, `0.0` gives fully reproducible results.
2. **`trust_remote_code` flows through `**kwargs`** — `load()` doesn't have it as a named parameter; relies on kwargs propagation. Works but fragile.
3. **Unused upstream parameters that could improve quality:**
   - `prefill_step_size` — reduces peak memory during prefill (default 2048)
   - `revision` — pin a specific model version for reproducibility
   - `adapter_path` — LoRA adapter support

### Actions

- [ ] Change `DEFAULT_TEMPERATURE` to `0.0` to match upstream and ensure reproducibility (or document why 0.1 is preferred)
- [ ] Add `--prefill-step-size` CLI flag (maps to `generate()` kwarg)
- [ ] (Low) Add `--revision` CLI flag for model version pinning
- [ ] (Low) Add `--adapter-path` CLI flag for LoRA support

---

## (vii) Test Suite Quality & Coverage

### Strengths

- 33 test files, well-structured `conftest.py` with session-scoped environment detection, image fixtures at 3 fidelity levels, auto-cleanup, custom markers.

### Issues

1. **Untested pure-logic functions** (testable without mlx-vlm):
   - `load_quality_config()` — no test for YAML loading, default fallback, or invalid keys
   - `validate_model_identifier()` — pure string validation
   - `_apply_exclusions()` — pure list filtering
   - `prepare_prompt()` — prompt assembly logic
   - 7 `compute_*` advanced metrics functions
2. **HTML report testing is shallow** — `test_html_formatting.py` only tests the HTML escaper, not `generate_html_report()` or `_build_full_html_document()`.
3. **Duplicated HF cache setup** in `test_model_discovery.py` (L12-28) — already handled by conftest.
4. **No mock-based model inference tests** — `process_image_with_model()` is only tested via e2e smoke tests requiring actual models.

### Actions

- [ ] Add unit tests for `validate_model_identifier()`, `_apply_exclusions()`, `prepare_prompt()`
- [ ] Add unit tests for `load_quality_config()` with valid, invalid, and missing YAML
- [ ] Add parametrised tests for the `compute_*` metrics family
- [ ] Add `generate_html_report()` / `generate_markdown_report()` tests with empty/all-failed inputs
- [ ] Remove duplicated HF cache setup in `test_model_discovery.py`
- [ ] (Low) Add mock-based test for `process_image_with_model()` (mock `mlx_vlm.generate.generate`)

---

## (viii) Script Functionality

### Strengths

- Comprehensive CLI with 20+ flags, both short and long forms. `BooleanOptionalAction` for `--trust-remote-code`. Thorough validation in `validate_cli_arguments()`.

### Issues

1. **No mutual exclusion between `--folder` and `--image`** — if both provided, behavior depends on internal logic rather than parser-level enforcement.
2. **Silent default to `~/Pictures/Processed`** when neither `--folder` nor `--image` is specified — could surprise users if directory doesn't exist.

### Actions

- [ ] Add `mutually_exclusive_group` for `--folder` and `--image` (or document the precedence)
- [ ] Warn (not error) when using the default folder and it doesn't exist

---

## (ix) Other Areas

1. **`docs/notes/` housekeeping** — 33 historical files. Only 2-3 have enduring reference value. The rest document completed tasks. Could be archived into `docs/notes/archive/` or deleted.
2. **`update.sh` issues** — already addressed in previous conversation (conda update --all, missing PyPI fallback, eager build-tool upgrades, no post-flight verification).
3. **No `CHANGELOG.md`** — for a tool with this level of maturity, a changelog would help users track breaking changes across versions.

### Actions

- [ ] Archive or prune `docs/notes/` historical files
- [ ] (Low) Add a `CHANGELOG.md`
