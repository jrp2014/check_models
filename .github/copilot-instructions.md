## Copilot / AI Agent Instructions — check_models

Benchmarking tool for MLX Vision Language Models on Apple Silicon. macOS-only, Python 3.13+, conda `mlx-vlm` environment required.

---

### 1. Environment — always do this first

```bash
conda activate mlx-vlm          # REQUIRED before any python/make command
cd src && python -m tools.validate_env && cd ..   # quick sanity check
```

If the environment doesn't exist: `bash src/tools/setup_conda_env.sh`.
For one-off commands without activating: `conda run -n mlx-vlm python ...`.
**Never run bare `python` without the conda environment active.**

### 2. Key files (read before editing)

| File | Purpose | Size |
| ------ | --------- | ------ |
| `src/check_models.py` | **Single-file CLI monolith** (~27,900 lines). All logic lives here. | ★ primary edit target |
| `src/check_models_data/quality_config.yaml` | Runtime thresholds loaded by `load_quality_config()` | Edit thresholds here, not in Python |
| `src/pyproject.toml` | Packaging, dependencies, tool config (ruff, mypy, pytest) | Update when adding imports |
| `src/tests/conftest.py` | Shared fixtures: `test_image`, `minimal_test_image`, `realistic_test_image`, `folder_with_images`, etc. | Use existing fixtures |
| `src/tests/test_*.py` | ~15,100 lines across ~36 test files | Add tests to existing files |
| `docs/IMPLEMENTATION_GUIDE.md` | Detailed coding standards and architecture decisions | Reference for conventions |
| `src/README.md` | Full CLI docs, all flags, usage examples (~1,200 lines) | Reference for CLI behavior |

### 3. Navigating `src/check_models.py` (section map)

The file is organized in this order — search for these exact landmark headers (formatted as comment blocks) to jump directly to the target area instead of relying on line numbers:

| Section | Key contents | Landmark Header / Search Tag |
| --------- | ------------- | ---------------------------- |
| Imports, config & optional dependency guards | `MISSING_DEPENDENCIES`, `QualityThresholds`, `load_quality_config()` | `SECTION: IMPORTS, CONFIG & OPTIONAL DEPENDENCY GUARDS` |
| Type aliases, protocols & JSONL records | `SupportsGenerationResult`, `SupportsExifIfd`, `JsonlResultRecord` | `SECTION: TYPE ALIASES, PROTOCOLS & JSONL RECORDS` |
| App constants & core result types | `PerformanceResult`, `ResultSet`, `ProcessImageParams`, report block primitives | `SECTION: APP CONSTANTS & CORE RESULT TYPES` |
| Timing, logging & Rich console plumbing | `PerfCounterTimer`, `TimeoutManager`, `LogStyles`, `StyleAwareRichHandler` | `SECTION: TIMING, LOGGING & RICH CONSOLE PLUMBING` |
| Formatting, escaping & detector helpers | `fmt_num`, report escapers, `_detect_repetitive_output`, harness detectors | `SECTION: FORMATTING, ESCAPING & DETECTOR HELPERS` |
| Metrics, scoring & field formatting | `compute_vocabulary_diversity`, `compute_cataloging_utility`, `analyze_generation_text`, `format_field_value` | `SECTION: METRICS, SCORING & FIELD FORMATTING` |
| Console, system & image metadata helpers | CLI Rich helpers, library/system info, EXIF/XMP extraction | `SECTION: CONSOLE, SYSTEM & IMAGE METADATA HELPERS` |
| Diagnostics/report context builders | `DiagnosticsConfig`, `IssueCluster`, `ReportRenderContext`, repro command specs | `SECTION: DIAGNOSTICS/REPORT CONTEXT BUILDERS` |
| Report generators & runtime fingerprints | `generate_diagnostics_report`, `generate_html_report`, `generate_markdown_report`, `collect_runtime_fingerprint()` | `SECTION: REPORT GENERATORS & RUNTIME FINGERPRINTS` |
| Model processing | CLI argument validation, cache scan, `_load_model`, `process_image_with_model` | `SECTION: MODEL PROCESSING` |
| CLI run helpers & logging | `setup_environment`, `find_and_validate_image`, `process_models`, result logging | `SECTION: CLI RUN HELPERS & LOGGING` |
| Result enrichment/history/finalization | quality enrichment, JSONL/history, issue drafts, repro bundles, `finalize_execution` | `SECTION: RESULT ENRICHMENT/HISTORY/FINALIZATION` |
| Main orchestration & argparse | `main()`, `main_cli()`, `_build_cli_parser()` | `SECTION: MAIN ORCHESTRATION & ARGPARSE` |

### 4. Architecture & patterns

- **Single CLI runner**: discovers models (HF cache scan), runs each with per-model isolation (timeouts, try/except), generates multi-format reports (`HTML`, `Markdown`, `TSV`, `JSONL`).
- **Configuration hierarchy**: `src/check_models_data/quality_config.yaml` → `QualityThresholds` / `FormattingThresholds` dataclasses. Never sprinkle magic numbers.
- **Dependencies**: optional packages are guarded with `try/except ImportError` → populate `MISSING_DEPENDENCIES`; core runtime deps (`mlx`, `mlx-vlm`, `mlx-lm`) now hard-fail before inference.
- **Display normalization**: ALL metric formatting goes through `format_field_value(field_name, value)`. Do not format metrics inline.
- **Type aliases**: `MetricValue = int | float | str | bool | None` is the value type for metrics.
- **Protocols over ABCs**: typing for optional deps uses `Protocol` classes (e.g., `SupportsGenerationResult`).
- **Reports write to** `src/output/reports/` (HTML, Markdown, TSV, diagnostics) and `src/output/` (JSONL, history, logs). Additional conditional outputs: `src/output/issues/` (generated GitHub issue templates), `src/output/repro_bundles/` (JSON reproduction bundles).
- **Security**: defaults to `--trust-remote-code` and warns when enabled. The CLI no longer mutates `transformers` backend-selection environment variables at startup.

### 5. Make targets (all run from repo root)

| Target | What it does |
| -------- | ------------- |
| `make quality` | **Primary gate**: checks Ruff formatting + lint, mypy, ty, pyrefly, vulture, Skylos quality/secrets/SCA plus `-a` audit, full pytest, shellcheck, markdownlint |
| `make skylos-danger` | Advisory Skylos `--danger` scan for workflow and security findings; not merge-blocking yet, but the current repo-root scan is clean and could be promoted later |
| `make skylos-danger-llm` | Advisory Skylos `--danger` scan with LLM-optimized output for agent triage |
| `make skylos-verify` | Run `skylos verify` with repo project context for narrow post-edit agent checks |
| `make vulture` | Run Vulture dead-code scan for `src/check_models.py` and `src/tools/`. *Note: Vulture commonly flags `TypedDict` keys and `Protocol` signatures as "unused" because they are evaluated statically and not tracked natively in runtime logic flows. Treat these as false positives.* |
| `make test` | Pytest-only shortcut for faster local test loops. Do not run it again after a successful `make quality`; `make quality` already runs the full pytest suite. |
| `make dev` | Install editable with `[dev,extras,torch]` |
| `make install` | Install editable (runtime only) |
| `make format` | Apply `ruff format src/` before running the full quality gate |
| `make -C src lint-fix` | Apply safe Ruff lint fixes (`ruff check --fix`) before running the full quality gate |
| `make lint` | Run Ruff lint early so lint errors are cleared before the full quality gate |
| `make ci` | Full strict CI pipeline |
| `make stubs` | Auto-generate `typings/` stubs for `mlx-lm`, `mlx-vlm`, `transformers`, `tokenizers` |
| `make deps-sync` | Sync README dependency blocks with pyproject.toml |
| `make clean` | Remove caches and generated outputs |

### 6. Testing guidance

- **Test markers**: `@pytest.mark.slow`, `@pytest.mark.e2e`, `@pytest.mark.subprocess`. Tests in `test_e2e_smoke.py` are auto-marked `slow` + `e2e`.
- **Run a single test file**: `pytest src/tests/test_parameter_validation.py -q`
- **Run with filter**: `pytest src/tests/test_html_formatting.py -k "specific_case" -vv --maxfail=1`
- **Fixtures** (from `conftest.py`): `test_image` (100×100), `minimal_test_image` (10×10), `realistic_test_image` (640×480 with shapes), `folder_with_images`, `empty_folder`, `mlx_vlm_available`, `fixture_model_cached`.
- **Many tests assert exact strings** — if you change report formats or CLI output, update `src/output/` fixtures and check formatting tests.
- **Add tests to existing files** (e.g., `test_parameter_validation.py` for new CLI flags, `test_html_formatting.py` for report changes). Do not create standalone test scripts.
- **Validation artifact hygiene**: Validation tests must not rewrite tracked `src/output/` assets. Route generated outputs to `tmp_path`, another temp directory, or gitignored `src/output/test_*` paths so `make quality` never requires restoring benchmark snapshots after it runs.

### 7. CI and hooks

- **Skylos advisory job**: GitHub Actions `skylos-advisory` on `ubuntu-latest` runs `bash src/tools/run_skylos_danger_advisory.sh` so workflow-security findings are surfaced separately from the blocking quality gate. The current repo-root advisory scan is clean, so this path is now a viable candidate for promotion if the team wants stricter enforcement.
- **Static CI job**: GitHub Actions `static-quality` on `macos-15`, Python 3.13, Node.js 22. It installs `src/.[dev]`, runs `npm install --ignore-scripts --prefix src`, generates MLX stubs via nanobind into `typings/`, then runs `bash src/tools/run_quality_checks.sh`, including Skylos quality/secrets/SCA and `-a` audit checks.
- **Runtime CI job**: separate `runtime-smoke` job runs `bash src/tools/run_runtime_smoke.sh` so Metal/runtime failures do not mask static quality results.
- **Dependency sync CI job**: `.github/workflows/dependency-sync.yml` runs on `ubuntu-latest` with path filters and verifies `python -m tools.update_readme_deps --check`.
- **Pre-commit hooks**: either `pre-commit install` or `cd src && python -m tools.install_precommit_hook`. Both install the same two stages:
  - commit stage: `bash src/tools/run_commit_hygiene.sh`
  - push stage: `bash src/tools/check_quality_simple.sh`
- **PRs must pass**: workflow YAML validation, dependency sync check, ruff format + lint, mypy, ty, pyrefly, vulture, Skylos quality/secrets/SCA and `-a` audit, pytest, shellcheck, markdownlint, plus the isolated runtime smoke probe. Skylos `--danger` runs separately in advisory mode with GitHub annotations and summaries, but the clean advisory queue means it can be promoted later without carrying known debt.

### 8. Coding conventions (quick reference)

- `from __future__ import annotations` at top of every file
- Full type annotations: all parameters + return types. Use `| None`, `list[str]` (not `Optional`, `List`)
- Prefer explicit symbol imports when practical (e.g., `from check_models import foo`), especially in tests; avoid broad module imports when only a few symbols are used.
- `Final` for constants: `TIMEOUT: Final[float] = 5.0`
- `pathlib.Path` for all paths; convert to `str` only at library call boundaries
- `raise SystemExit(code)` instead of `sys.exit()` (better for type narrowing)
- Catch specific exceptions, not bare `except Exception`. Use `raise ... from e` for context
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`

### 9. Change workflow (single checklist)

1. `conda activate mlx-vlm`
2. `git checkout -b feature/your-change`
3. Edit `src/check_models.py` (and/or other files in `src/`)
4. Add/update tests in `src/tests/` for the change
5. If you added imports or updated package thresholds → update `src/pyproject.toml` or `src/check_models_data/dependency_policy.py`, then run `make deps-sync` to rebuild README dependencies
6. If you added/changed CLI flags → update the CLI reference table in `src/README.md` (§ Command Line Reference)
7. `make format` — apply Ruff formatting before the full quality gate
8. `make -C src lint-fix` — apply safe Ruff fixes when lint reports fixable issues
9. `make lint` — clear Ruff lint errors before running the full gate
10. `bash src/tools/run_commit_hygiene.sh` — verify local commit hygiene
11. `make quality` — run the full quality gate check, including the full pytest suite
12. If report formats changed → update `src/output/` fixtures intentionally; validation tests must not rewrite tracked `src/output/` assets just to prove a change
13. Update `CHANGELOG.md` under `[Unreleased]` for any maintainer-relevant change (features, fixes, refactors, tooling/docs workflow updates)
14. `git commit -m "feat: description"` and push

### 10. Agentic skills (`.agents/skills/`)

Skills provide structured, step-by-step workflows for recurring tasks. Read the
relevant `SKILL.md` **before** starting work of that kind.

| Skill | When to use | File |
| ----- | ----------- | ---- |
| `add-or-fix-type-checking` | Typing errors from mypy, ty, pyrefly, or `make quality` | `.agents/skills/add-or-fix-type-checking/SKILL.md` |

### 11. Common edit recipes

**Review workflow security or agent-generated changes:**

1. Run `make skylos-danger` for the advisory JSON/annotation-style scan
2. Run `make skylos-danger-llm` when you want the same findings with code context tuned for an AI/code-review agent
3. Run `make skylos-verify ARGS='--file path/to/file --range L1:L2'` for narrow post-edit AI-defect verification

**Add a CLI flag:**

1. Add `argparse` argument in `_build_cli_parser()` under `SECTION: MAIN ORCHESTRATION & ARGPARSE` in `src/check_models.py`
2. Wire it through `main()` → `process_image_with_model()` or relevant function
3. Add test in `src/tests/test_parameter_validation.py`
4. Update the CLI reference table in `src/README.md` (§ Command Line Reference)
5. Run `pytest src/tests/test_parameter_validation.py src/tests/test_cli_help_output.py -q`

**Change a quality threshold:**

1. Edit `src/check_models_data/quality_config.yaml` (preferred) or `QualityThresholds` dataclass in `SECTION: IMPORTS, CONFIG & OPTIONAL DEPENDENCY GUARDS`
2. Run `pytest src/tests/test_quality_analysis.py -q`

**Modify report output:**

1. Edit `generate_html_report` or `generate_markdown_report` under `SECTION: REPORT GENERATORS & RUNTIME FINGERPRINTS`
2. Update `src/output/` fixture files if test assertions reference them
3. Run `pytest src/tests/test_html_formatting.py src/tests/test_markdown_formatting.py -q`

**Add a new quality detector:**

1. Add `_detect_your_pattern(text: str) -> tuple[bool, str | None]` following existing patterns under `SECTION: METRICS, SCORING & FIELD FORMATTING`
2. Wire it into the quality analysis pipeline
3. Add thresholds to `src/check_models_data/quality_config.yaml` and `QualityThresholds`
4. Add test in `src/tests/test_quality_analysis.py`

### 12. What NOT to do

- **Don't split `check_models.py`** into multiple files — the monolith structure is intentional
- **Don't hardcode magic numbers** — use `quality_config.yaml` or dataclass fields
- **Don't suppress lints** (`# noqa`, `# type: ignore`) without a documented reason
- **Don't run `python` without conda** — always `conda activate mlx-vlm` first
- **Don't create ad-hoc test scripts** — add tests to existing `src/tests/test_*.py` files
- **Don't duplicate formatting logic** — extend `format_field_value` for new metrics
- **Don't over-extract helpers** — a single well-commented function is preferred over many tiny one-use helpers (see `docs/IMPLEMENTATION_GUIDE.md` § Philosophy)

### 13. Dependency Synchronization and Policy

This repository implements a strict dependency alignment and verification policy to ensure type safety and runtime compatibility across the MLX stack:

- **Dependency Policy Definitive Specs**: All package version floors and compatibility rules are declared in `src/check_models_data/dependency_policy.py`.
- **pyproject.toml Alignment**: When adding or updating third-party libraries, declare the dependency range in `src/pyproject.toml`.
- **Auto-Syncing README**: The CLI README documentation contains an auto-generated dependencies table block. After editing `pyproject.toml` or dependency policies, you must execute `make deps-sync` (which runs `python -m tools.update_readme_deps`) to rebuild the README alignment blocks.
- **CI Dependency Sync Check**: The CI pipeline runs `python -m tools.update_readme_deps --check` to verify that README markdown blocks match `pyproject.toml` exactly. Failures will block pull request approvals.
