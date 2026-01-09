## Copilot / AI Agent Instructions — check_models

Purpose: help AI coding agents be productive immediately in this repository by highlighting the project structure, common workflows, and concrete examples to edit/test.

- **Quick entry points:** `src/check_models.py` is the single-file CLI implementation and public entry for the tool. See `src/README.md` for full CLI flags and examples.
- **Run locally (recommended):**

```bash
# create / activate environment (repo has helper script)
./setup_conda_env.sh
conda activate mlx-vlm

# run help in module form (uses src package)
python -m check_models --help

# run unit + integration tests
make test

# run linters / typechecks
make quality
```

- **Outputs:** runtime reports and artifacts are written to the `src/output/` directory (`results.html`, `results.md`, `results.jsonl`, `results.tsv`). Prefer editing/reporting code in `src/check_models.py` and verify changes by re-running the CLI and relevant tests.

- **Key files to inspect when changing behavior:**
  - `src/check_models.py` — main CLI logic, constants, quality thresholds (see `FormattingThresholds`, `QualityThresholds`) and output generators (`generate_html_report`, `generate_markdown_report`).
  - `src/quality_config.yaml` — overrides runtime `QualityThresholds` loaded by `load_quality_config()`.
  - `src/pyproject.toml` and `src/check_models.egg-info/entry_points.txt` — packaging/entry points information.
  - `src/tests/` — unit and integration tests. Examples: `src/tests/test_cli_help_output.py` (CLI semantics), `src/tests/test_model_discovery.py` (cache discovery patterns), `src/tests/test_html_formatting.py` (report formatting).

- **Architecture & patterns (concrete):**
  - Single CLI runner performs model discovery (Hugging Face cache), per-model isolation (timeouts, try/except), generation via `mlx_vlm.generate.generate`, and produces multi-format reports.
  - Optional dependencies are handled with graceful fallbacks (e.g., Pillow, mlx, mlx-vlm). When adding features, follow the existing pattern of providing informative fallbacks and populating `MISSING_DEPENDENCIES`.
  - Configuration is centralized in dataclasses at the top of `src/check_models.py` (`FormattingThresholds`, `QualityThresholds`). Prefer editing those or `src/quality_config.yaml` rather than sprinkling magic numbers.

- **Developer conventions & test guidance:**
  - Python 3.13+ is the target runtime (see `src/README.md`). CI/dev environments expect `conda activate mlx-vlm` from the repo helper.
  - Use the Make targets: `make test` (pytest), `make quality` (formatting, lint, types), `make install` / `make dev` for dependency installs.
  - Many tests are integration-level (image workflow, model discovery). When running a focused change, run specific pytest markers or file-level tests (e.g., `pytest src/tests/test_cli_help_output.py -q`).

- **Editing guidance / safe change checklist:**
  1. Update `src/check_models.py` implementation.
  2. Add or update tests in `src/tests/` demonstrating CLI usage or unit behavior (examples show how to call the module with `python -m check_models`).
  3. Run `make test` and `make quality` locally; adjust code until tests and linters pass.
  4. If changing report shapes, update `src/output/` fixtures referenced by tests (search tests for references to `results.md/jsonl/tsv`).

- **Common edit examples (copy-paste friendly):**
  - To add a new CLI flag with a default and unit test: modify `argparse` usage in `src/check_models.py`, add a small unit test in `src/tests/test_parameter_validation.py`, and run `pytest src/tests/test_parameter_validation.py -q`.
  - To change thresholds: prefer `src/quality_config.yaml` or dataclass fields (`FormattingThresholds`, `QualityThresholds`) at top of `src/check_models.py`.

- **Environment / security notes:**
  - The tool defaults to `--trust-remote-code` (see `src/README.md`). When creating code paths that do model loading, maintain the same flag checks and document security implications in the README.
  - The code sets `TRANSFORMERS_NO_TF`, `TRANSFORMERS_NO_FLAX`, `TRANSFORMERS_NO_JAX` by default to avoid loading heavy backends on macOS unless `MLX_VLM_ALLOW_TF=1` is set.

- **When in doubt:**
  - Use `src/README.md` and `README.md` for concrete CLI examples and expected outputs; replicate the same invocation form in tests and examples.
  - Keep outputs deterministic where tests expect exact strings (many tests assert CLI output/formatting). Check `src/tests/test_cli_help_output.py` and formatting tests for expected shapes.

- **Contributor & Implementation notes (high-value snippets):**
  - **Setup & env:** Use `./setup_conda_env.sh` then `conda activate mlx-vlm`. Validate with `python -m tools.validate_env`.
  - **Make targets:** `make dev` (dev deps + editable install), `make install` (runtime), `make test`, `make quality` (format + lint + types), `make ci` (full CI pipeline, strict), `make install-torch` (optional PyTorch).
  - **Pre-commit & CI:** Install hooks via `python -m tools.install_precommit_hook`. PRs must pass ruff, mypy, tests, dependency sync, and markdown linting in CI.
  - **Commit style:** Use conventional commits (`feat:, fix:, docs:, test:, chore:`) for PR clarity.
  - **Adding deps:** When adding imports, update `src/pyproject.toml` and run `python -m tools.update_readme_deps` / `make deps-sync` so README blocks stay in sync.
  - **Tests & fixtures:** Many tests assert exact CLI/output shapes; if you change report formats update `src/output/` fixtures referenced by tests.
  - **Implementation conventions:**
    - Full type annotations required for new code; prefer `from __future__ import annotations` and `| None`/`list[str]` syntax.
    - Use `pathlib.Path` for paths and convert at function boundaries.
    - Prefer centralized formatters (`format_field_value`) and dataclasses (`FormattingThresholds`, `QualityThresholds`) for display/thresholds—edit `src/quality_config.yaml` for runtime overrides.
    - Handle optional deps with `try/except ImportError` + explicit fallbacks and populate `MISSING_DEPENDENCIES`.
  - **AI agent rules (from IMPLEMENTATION_GUIDE):**
    - Read `src/README.md` and `src/quality_config.yaml` before changes.
    - Always run `make quality` and `make test` locally; do not add new standalone test scripts.
    - Do not hardcode magic thresholds—use config/dataclasses.
    - Avoid suppressing lints without documented reason; prefer fixing the cause.
    - **Environment discipline:** Always use `conda activate mlx-vlm` or `conda run -n mlx-vlm python ...` for single commands; never run `python` without ensuring the environment is active.

## Step-by-step edit checklist

Follow this minimal, repeatable workflow when making code or documentation changes.

1. Create a feature branch:

```bash
git checkout -b feature/your-change
```

1. Prepare environment (recommended):

```bash
./setup_conda_env.sh
conda activate mlx-vlm
```

1. Run quick validations before editing:

```bash
python -m tools.validate_env
python -m check_models --help
```

1. Implement changes in `src/` (follow type and style rules in `docs/IMPLEMENTATION_GUIDE.md`).

2. Update dependencies if you added imports:

```bash
# Edit src/pyproject.toml
python -m tools.update_readme_deps
make deps-sync
```

1. Run and fix formatting/lint/type checks locally:

```bash
ruff format
make quality   # runs format + lint + mypy + markdownlint (if available)
```

1. Run targeted tests for your change:

```bash
# Single file
pytest src/tests/test_parameter_validation.py -q
# Or run the full suite (slow)
make test
```

1. Run pre-commit hooks locally and auto-fix issues:

```bash
python -m tools.install_precommit_hook
pre-commit run --all-files
```

1. Update documentation & fixtures if output formats changed (update `src/output/` fixtures referenced by tests).

2. Commit and push using conventional commit messages:

```bash
git add .
git commit -m "feat: brief description"
git push origin feature/your-change
```

## CI & pre-commit validation (concrete commands)

- Install and run pre-commit hooks:

```bash
python -m tools.install_precommit_hook
pre-commit run --all-files
```

- Run the full quality gate (what CI runs locally):

```bash
make quality
make test
```

- If you need strict markdown linting locally (CI may require it):

```bash
# install Node.js/npm first, then
make install-markdownlint
make quality-strict
```

## Quick verification snippets

- Run the CLI dry-run to see what would execute:

```bash
python -m check_models --dry-run --folder ~/Pictures/Processed
```

- Run a single model quickly (use `--trust-remote-code` only for trusted models):

```bash
python -m check_models --image test.jpg --models mlx-community/nanoLLaVA --timeout 120
```

- Re-run failing tests with verbose output and capture logs:

```bash
pytest src/tests/test_html_formatting.py -k "specific_case" -vv --maxfail=1
```

## When changing output formats or report shapes

- Update `src/output/` fixtures referenced by tests (`results.md`, `results.jsonl`, `results.tsv`, `results.html`).
- Run `pytest` to ensure no regression in formatting tests (many tests assert exact strings).

## Implementation Agent reminders (condensed)

- Always prefer configuration (`src/quality_config.yaml`, `FormattingThresholds`, `QualityThresholds`) over hardcoded values.
- Use `pathlib.Path`, full type annotations, and the centralized `format_field_value` for display formatting.
- Handle optional deps with `try/except ImportError` and populate `MISSING_DEPENDENCIES`.

