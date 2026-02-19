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
| `src/check_models.py` | **Single-file CLI monolith** (~10,700 lines). All logic lives here. | ★ primary edit target |
| `src/quality_config.yaml` | Runtime thresholds loaded by `load_quality_config()` | Edit thresholds here, not in Python |
| `src/pyproject.toml` | Packaging, dependencies, tool config (ruff, mypy, pytest) | Update when adding imports |
| `src/tests/conftest.py` | Shared fixtures: `test_image`, `minimal_test_image`, `realistic_test_image`, `folder_with_images`, etc. | Use existing fixtures |
| `src/tests/test_*.py` | ~5,000 lines across ~35 test files | Add tests to existing files |
| `docs/IMPLEMENTATION_GUIDE.md` | Detailed coding standards and architecture decisions | Reference for conventions |
| `src/README.md` | Full CLI docs, all flags, usage examples (~1,100 lines) | Reference for CLI behavior |

### 3. Navigating `src/check_models.py` (section map)

The file is organized in this order — use these landmarks to jump to the right area:

| Section | Key contents | Approx. lines |
| --------- | ------------- | --------------- |
| Imports & env setup | `MISSING_DEPENDENCIES` dict, optional dep guards | 1–580 |
| Protocol types | `SupportsGenerationResult`, `SupportsExifIfd` | 590–620 |
| Data classes | `PerformanceResult`, `ResultSet`, `ProcessImageParams` | 677–900 |
| Timing & timeout | `PerfCounterTimer`, `TimeoutManager` | 913–1000 |
| Colors & logging | `Colors`, `LogStyles`, `ColoredFormatter` | 1013–1310 |
| Quality detection | `_detect_repetitive_output`, `_detect_hallucination_patterns`, etc. | 1589–2520 |
| Advanced metrics | `compute_vocabulary_diversity`, `compute_efficiency_metrics`, etc. | 2520–3310 |
| Field formatting | `format_field_value` (centralized display normalization) | 3474–3650 |
| Diagnostics/report generators | `generate_diagnostics_report` (~6101), `generate_html_report` (~6544), `generate_markdown_report` (~6739) | 6100–7600 |
| Image processing | `process_image_with_model` (~7695) | 7695–9600 |
| Run history/finalization | `compare_history_records`, `append_history_record`, `finalize_execution` | 9831–10270 |
| CLI parsing | `main()` (~10277), `main_cli()` (~10385) | 10277–10679 |

### 4. Architecture & patterns

- **Single CLI runner**: discovers models (HF cache scan), runs each with per-model isolation (timeouts, try/except), generates multi-format reports (`HTML`, `Markdown`, `TSV`, `JSONL`).
- **Configuration hierarchy**: `src/quality_config.yaml` → `QualityThresholds` / `FormattingThresholds` dataclasses. Never sprinkle magic numbers.
- **Optional deps**: guarded with `try/except ImportError` → populate `MISSING_DEPENDENCIES` dict → graceful runtime fallbacks. Follow the same pattern when adding features.
- **Display normalization**: ALL metric formatting goes through `format_field_value(field_name, value)`. Do not format metrics inline.
- **Type aliases**: `MetricValue = int | float | str | None` is the value type for metrics.
- **Protocols over ABCs**: typing for optional deps uses `Protocol` classes (e.g., `SupportsGenerationResult`).
- **Reports write to** `src/output/` (`results.html`, `results.md`, `results.jsonl`, `results.tsv`, `diagnostics.md`, `results.history.jsonl`, `check_models.log`, `environment.log`).
- **Security**: defaults to `--trust-remote-code`. Env vars `TRANSFORMERS_NO_TF`, `TRANSFORMERS_NO_FLAX`, `TRANSFORMERS_NO_JAX` are set to avoid heavy backends unless `MLX_VLM_ALLOW_TF=1`.

### 5. Make targets (all run from repo root)

| Target | What it does |
| -------- | ------------- |
| `make quality` | **Primary gate**: ruff format + lint + mypy + ty + pyrefly + pytest + shellcheck + markdownlint |
| `make test` | `pytest src/tests/ -v` |
| `make dev` | Install editable with `[dev,extras,torch]` |
| `make install` | Install editable (runtime only) |
| `make format` | `ruff format src/` |
| `make ci` | Full strict CI pipeline |
| `make stubs` | Generate `typings/` stubs for the MLX ecosystem (`mlx-lm`, `mlx-vlm`, `tokenizers`) |
| `make deps-sync` | Sync README dependency blocks with pyproject.toml |
| `make clean` | Remove caches and generated outputs |

### 6. Testing guidance

- **Test markers**: `@pytest.mark.slow`, `@pytest.mark.e2e`, `@pytest.mark.subprocess`. Tests in `test_e2e_smoke.py` are auto-marked `slow` + `e2e`.
- **Run a single test file**: `pytest src/tests/test_parameter_validation.py -q`
- **Run with filter**: `pytest src/tests/test_html_formatting.py -k "specific_case" -vv --maxfail=1`
- **Fixtures** (from `conftest.py`): `test_image` (100×100), `minimal_test_image` (10×10), `realistic_test_image` (640×480 with shapes), `folder_with_images`, `empty_folder`, `mlx_vlm_available`, `fixture_model_cached`.
- **Many tests assert exact strings** — if you change report formats or CLI output, update `src/output/` fixtures and check formatting tests.
- **Add tests to existing files** (e.g., `test_parameter_validation.py` for new CLI flags, `test_html_formatting.py` for report changes). Do not create standalone test scripts.

### 7. CI environment

- **Platform**: GitHub Actions, `macos-15`, Python 3.13, Node.js 22
- **Pipeline** (`.github/workflows/quality.yml`): checkout → install deps (`pip install -e src/.[dev]`) → generate MLX stubs via nanobind → `bash src/tools/run_quality_checks.sh`
- **PRs must pass**: ruff format + lint, mypy, ty, pyrefly, pytest, shellcheck, dependency sync, markdownlint
- **Pre-commit hooks**: install with `cd src && python -m tools.install_precommit_hook`

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
5. If you added imports → update `src/pyproject.toml` then `make deps-sync`
6. `make quality` — fix issues until clean
7. `make test` — ensure no regressions
8. If report formats changed → update `src/output/` fixtures
9. Update `CHANGELOG.md` under `[Unreleased]` for any maintainer-relevant change (features, fixes, refactors, tooling/docs workflow updates)
10. `git commit -m "feat: description"` and push

### 10. Common edit recipes

**Add a CLI flag:**

1. Add `argparse` argument near line ~10400 in `src/check_models.py`
2. Wire it through `main()` → `process_image_with_model()` or relevant function
3. Add test in `src/tests/test_parameter_validation.py`
4. Run `pytest src/tests/test_parameter_validation.py src/tests/test_cli_help_output.py -q`

**Change a quality threshold:**

1. Edit `src/quality_config.yaml` (preferred) or `QualityThresholds` dataclass (~line 182)
2. Run `pytest src/tests/test_quality_analysis.py -q`

**Modify report output:**

1. Edit `generate_html_report` (~line 6544) or `generate_markdown_report` (~line 6739)
2. Update `src/output/` fixture files if test assertions reference them
3. Run `pytest src/tests/test_html_formatting.py src/tests/test_markdown_formatting.py -q`

**Add a new quality detector:**

1. Add `_detect_your_pattern(text: str) -> tuple[bool, str | None]` following existing patterns (~line 1356–2520)
2. Wire it into the quality analysis pipeline
3. Add thresholds to `src/quality_config.yaml` and `QualityThresholds`
4. Add test in `src/tests/test_quality_analysis.py`

### 11. What NOT to do

- **Don't split `check_models.py`** into multiple files — the monolith structure is intentional
- **Don't hardcode magic numbers** — use `quality_config.yaml` or dataclass fields
- **Don't suppress lints** (`# noqa`, `# type: ignore`) without a documented reason
- **Don't run `python` without conda** — always `conda activate mlx-vlm` first
- **Don't create ad-hoc test scripts** — add tests to existing `src/tests/test_*.py` files
- **Don't duplicate formatting logic** — extend `format_field_value` for new metrics
- **Don't over-extract helpers** — a single well-commented function is preferred over many tiny one-use helpers (see `docs/IMPLEMENTATION_GUIDE.md` § Philosophy)
