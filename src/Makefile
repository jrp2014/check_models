# Package-local Makefile for mlx-vlm-check (lives alongside pyproject.toml)

PYTHON ?= python
# CONDA_ENV can be overridden: make test CONDA_ENV=my-custom-env
CONDA_ENV ?= mlx-vlm
VIRTUAL_ENV := $(shell echo $$VIRTUAL_ENV)
CONDA_ACTIVE := $(shell echo $$CONDA_DEFAULT_ENV)

# Detect and use active Python environment (supports venv, virtualenv, uv, conda, etc.)
ifdef VIRTUAL_ENV
    # Virtual environment is active (venv/virtualenv/uv/poetry/pipenv/etc.)
    RUN_PY := $(PYTHON)
    RUN_TOOL_PREFIX :=
    ENV_TYPE := venv
    ENV_NAME := $(notdir $(VIRTUAL_ENV))
else ifdef CONDA_DEFAULT_ENV
    # In a conda environment - decide if we need conda run
    ifeq ($(CONDA_ACTIVE),$(CONDA_ENV))
        # Correct conda environment is active
        RUN_PY := $(PYTHON)
        RUN_TOOL_PREFIX :=
        ENV_TYPE := conda
        ENV_NAME := $(CONDA_ENV)
    else
        # Different conda environment is active, use conda run
        RUN_PY := conda run -n $(CONDA_ENV) $(PYTHON)
        RUN_TOOL_PREFIX := conda run -n $(CONDA_ENV)
        ENV_TYPE := conda
        ENV_NAME := $(CONDA_ENV)
        $(warning ⚠️  Conda env '$(CONDA_ACTIVE)' is active, but targeting '$(CONDA_ENV)')
        $(warning ⚠️  Commands will use: conda run -n $(CONDA_ENV))
        $(warning ⚠️  To avoid this, either activate the target env or set CONDA_ENV=$(CONDA_ACTIVE))
    endif
else
    # No environment active - check if conda is available for fallback
    ifneq ($(shell command -v conda 2>/dev/null),)
        # Conda available but no env active, use conda run
        RUN_PY := conda run -n $(CONDA_ENV) $(PYTHON)
        RUN_TOOL_PREFIX := conda run -n $(CONDA_ENV)
        ENV_TYPE := conda
        ENV_NAME := $(CONDA_ENV)
        $(info ℹ️  No environment active; commands will use: conda run -n $(CONDA_ENV))
    else
        # No environment manager detected, use system Python
        RUN_PY := $(PYTHON)
        RUN_TOOL_PREFIX :=
        ENV_TYPE := system
        ENV_NAME := system
        $(warning ⚠️  No virtual environment detected; using system Python)
        $(warning ℹ️  Consider activating venv/conda first: source .venv/bin/activate or conda activate $(CONDA_ENV))
    endif
endif

# Upgrade pip via whichever manager owns it. When conda owns pip, a pip
# self-upgrade interleaves two pip versions in site-packages and breaks pip
# at import time; when pip owns itself (pypi entry only), conda update would
# fail or reintroduce a second conda-owned copy. `conda list --no-pip` hides
# pypi-installed packages, so a hit there means conda owns pip.
ifeq ($(ENV_TYPE),conda)
    UPGRADE_PIP_CMD := if conda list -n $(CONDA_ENV) --no-pip '^pip$$' 2>/dev/null | grep -qE '^pip[[:space:]]'; then conda update -n $(CONDA_ENV) -y pip; else $(RUN_PY) -m pip install --upgrade pip; fi
else
    UPGRADE_PIP_CMD := $(RUN_PY) -m pip install --upgrade pip
endif

MYPY_CONFIG := pyproject.toml
FMT_PATHS := check_models.py tests tools check_models_data

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "\033[1;35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
	@echo "\033[1;35m  MLX VLM Check - Vision-Language Model Benchmarking Tool\033[0m"
	@echo "\033[1;35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n"
	@echo "\033[1;33m🚀 PRIMARY TARGET:\033[0m\n"
	@echo "  \033[1;36mmake check_models ARGS='--model <id> --image <path>'\033[0m"
	@echo "    Run the VLM checker (the main purpose of this project)\n"
	@echo "\033[1mDevelopment targets:\033[0m\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | grep -v "check_models:" | sed 's/:.*##/: /' | sort
	@echo "\n\033[2mEnvironment: active='$(CONDA_ACTIVE)' target='$(CONDA_ENV)'\033[0m"

# Installation
.PHONY: install
install: ## Editable install (runtime only)
	$(RUN_PY) -m pip install -e .

.PHONY: install-dev
install-dev: ## Editable install with dev extras
	$(RUN_PY) -m pip install -e .[dev]

.PHONY: install-torch
install-torch: ## Install PyTorch dependencies (needed for some models)
	$(RUN_PY) -m pip install -e .[torch]

.PHONY: install-all
install-all: ## Install all optional dependencies (dev + extras + torch)
	$(RUN_PY) -m pip install -e .[dev,extras,torch]

.PHONY: bootstrap-dev
bootstrap-dev: ## pip bootstrap runtime + dev + extras + torch + hooks + validation
	$(UPGRADE_PIP_CMD)
	$(RUN_PY) -m pip install -e .[dev,extras,torch]
	$(RUN_PY) -m pip install huggingface_hub
	@echo "[bootstrap] Installing markdownlint-cli2 (for markdown linting)..."
	@if command -v npm >/dev/null 2>&1; then \
		echo "[bootstrap] npm found, installing markdownlint-cli2..."; \
		npm install --silent --no-audit --no-fund 2>/dev/null || echo "⚠️  npm install failed (run 'npm install' inside src)"; \
	else \
		echo "[bootstrap] npm not found - markdownlint-cli2 will use npx on-demand"; \
		echo "  To install npm: https://nodejs.org/ or 'brew install node'"; \
	fi
	@echo "[bootstrap] Installing commit/push hooks..."
	@$(RUN_TOOL_PREFIX) pre-commit install || echo "⚠️  Could not install git hooks (pre-commit ships with the dev extras)"
	@if command -v pre-commit >/dev/null 2>&1; then \
		echo "[bootstrap] Installing pre-commit framework hooks..."; \
		$(RUN_TOOL_PREFIX) pre-commit install || echo "⚠️  pre-commit install failed"; \
	else \
		echo "[bootstrap] pre-commit framework not found (optional)"; \
		echo "  Install with: pip install pre-commit && pre-commit install"; \
	fi
	@echo "[bootstrap] Running environment validation..."
	@$(RUN_PY) -m tools.validate_env || echo "⚠️  Validation found issues (run 'python -m tools.validate_env' for details)"
	@echo "✓ Bootstrap complete"

# Markdown linting setup
.PHONY: install-markdownlint
install-markdownlint: ## Install markdownlint-cli2 via npm (requires Node.js/npm)
	@if command -v npm >/dev/null 2>&1; then \
		echo "Installing markdownlint-cli2 via npm..."; \
		npm install --no-audit --no-fund; \
		echo "✓ markdownlint-cli2 installed to node_modules/.bin/"; \
		echo "  Add node_modules/.bin to PATH or quality checks will use npx fallback"; \
	else \
		echo "❌ npm not found - install Node.js first"; \
		echo "  macOS: brew install node"; \
		echo "  Or download from: https://nodejs.org/"; \
		exit 1; \
	fi

# Environment validation
.PHONY: validate-env
validate-env: ## Validate the environment via tools.validate_env (single source of truth)
	@echo "🔍 Validating environment: $(ENV_NAME) ($(ENV_TYPE))"
	@$(RUN_PY) -m tools.validate_env

# ------------------------
# Run script
# ------------------------
.PHONY: check_models
check_models: ## Run the VLM checker (pass args via ARGS='--model <id> --image <path> ...')
	$(RUN_PY) -m check_models $(ARGS)

.PHONY: analyze-quality
analyze-quality: ## Run output quality diagnostic script (pass args via ARGS='--text "..."' or ARGS='--file ...')
	$(RUN_PY) -m tools.analyze_output_quality $(ARGS)

# Quality
.PHONY: format
format: ## Ruff format
	$(RUN_TOOL_PREFIX) ruff format $(FMT_PATHS)

.PHONY: lint
lint: ## Ruff lint
	$(RUN_TOOL_PREFIX) ruff check $(FMT_PATHS)

.PHONY: lint-fix
lint-fix: ## Ruff lint --fix
	$(RUN_TOOL_PREFIX) ruff check --fix $(FMT_PATHS)

.PHONY: typecheck
typecheck: ## mypy type checking
	$(RUN_TOOL_PREFIX) mypy --config-file $(MYPY_CONFIG) check_models.py tests

.PHONY: ty
ty: ## Ty type checking with resolved repo interpreter
	bash tools/run_ty_check.sh $(ARGS)

.PHONY: vulture
vulture: ## Vulture dead-code scan (configured in pyproject.toml)
	$(RUN_PY) -m vulture

# Tests
.PHONY: test
test: ## Run pytest
	@echo "[test] running tests"; \
	$(RUN_TOOL_PREFIX) pytest -q

.PHONY: test-cov
test-cov: ## Run pytest with coverage
	$(RUN_TOOL_PREFIX) pytest --cov=check_models --cov-report=term-missing --cov-report=xml -q

# Dependency management
.PHONY: deps-sync
deps-sync: ## Sync README dependency blocks
	$(RUN_PY) tools/update_readme_deps.py

.PHONY: update
update: ## Full updater via tools/update.sh (conda/brew, local MLX builds, smoke)
	bash tools/update.sh

.PHONY: update-quick
update-quick: ## Quick in-env refresh: upgrade pip and reinstall project (dev+extras+torch)
	@echo "[update-quick] Updating pip in environment: $(ENV_NAME) ($(ENV_TYPE))"
	@$(UPGRADE_PIP_CMD)
	@echo "[update-quick] Reinstalling project with all dependencies (dev+extras+torch)..."
	@$(RUN_PY) -m pip install -e .[dev,extras,torch] --upgrade
	@echo "✓ Environment updated and project reinstalled"
	@echo "  To change dependency bounds, edit pyproject.toml and run: make deps-sync"

.PHONY: probe-python-next
probe-python-next: ## Check whether the next Python (default 3.14) is viable in a throwaway conda env; never touches mlx-vlm
	bash tools/probe_python_next.sh

# Aggregates
.PHONY: check
check: format lint typecheck test ## Core quality pipeline
	@echo "✓ All core checks completed"

.PHONY: quality
quality: ## Run full quality checks (ruff+typing+vulture+Skylos+pytest+shellcheck+markdownlint)
	bash tools/run_quality_checks.sh

.PHONY: skylos-danger
skylos-danger: ## Run advisory Skylos danger scan for workflow/security findings
	bash tools/run_skylos_danger_advisory.sh

.PHONY: skylos-danger-llm
skylos-danger-llm: ## Write advisory Skylos danger findings in LLM-friendly format
	bash tools/run_skylos_danger_advisory.sh --llm

.PHONY: skylos-verify
skylos-verify: ## Run the Skylos verifier on a file/range (pass ARGS='--file ... --range ...')
	bash tools/run_skylos_verify.sh $(ARGS)

.PHONY: quality-strict
quality-strict: ## Run quality checks and require markdownlint availability
	@echo "[quality-strict] verifying markdownlint availability..."
	@if [ -x "node_modules/.bin/markdownlint-cli2" ] || command -v markdownlint-cli2 >/dev/null 2>&1 || command -v npx >/dev/null 2>&1; then \
		echo "[quality-strict] markdownlint found"; \
	else \
		echo "[quality-strict] ERROR: markdownlint-cli2 not available"; \
		echo "  Install with: make install-markdownlint or ensure npx is available"; \
		exit 1; \
	fi
	bash tools/run_quality_checks.sh

# CI pipeline
.PHONY: ci
ci: ## Run full CI pipeline (alias for quality checks)
	@echo "[ci] Running full quality pipeline for local-to-remote parity..."
	@bash tools/run_quality_checks.sh

# Maintenance
.PHONY: check-outdated
check-outdated: ## Check for outdated dependencies
	$(RUN_PY) tools/check_outdated.py

.PHONY: audit
audit: ## Run security audit on dependencies (pip-audit ships in the dev extras)
	$(RUN_PY) -m pip_audit

# Cleanup
.PHONY: clean-pyc
clean-pyc: ## Remove Python cache files and bytecode
	find . -name '__pycache__' -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete -o -name '*.pyo' -delete -o -name '*~' -delete

.PHONY: clean-build
clean-build: ## Remove build artifacts (build/, dist/, *.egg-info/)
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-test
clean-test: ## Remove test and coverage artifacts
	rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/

.PHONY: clean-mlx
clean-mlx: ## Clean build artifacts from local MLX development repos
	@bash tools/clean_builds.sh

.PHONY: clean-mlx-dry-run
clean-mlx-dry-run: ## Show what would be cleaned (dry run)
	@bash tools/clean_builds.sh --dry-run

.PHONY: clean
clean: clean-pyc clean-build clean-test ## Clean all generated files (pyc, build, test artifacts)
	@echo "✓ Cleanup complete"

.PHONY: clean-all
clean-all: clean ## Clean everything
	@echo "✓ Deep cleanup complete (use 'make clean-mlx' to clean local MLX repos)"
