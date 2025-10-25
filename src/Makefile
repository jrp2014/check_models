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

MYPY_CONFIG := pyproject.toml
FMT_PATHS := check_models.py tests tools

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
bootstrap-dev: ## pip bootstrap runtime + dev + extras + hooks + validation
	$(RUN_PY) -m pip install --upgrade pip
	$(RUN_PY) -m pip install -e .[dev,extras]
	$(RUN_PY) -m pip install "huggingface_hub[cli]"
	@echo "[bootstrap] Installing markdownlint-cli2 (for markdown linting)..."
	@if command -v npm >/dev/null 2>&1; then \
		echo "[bootstrap] npm found, installing markdownlint-cli2..."; \
		npm install --silent 2>/dev/null || echo "⚠️  npm install failed (markdownlint will use npx fallback)"; \
	else \
		echo "[bootstrap] npm not found - markdownlint-cli2 will use npx on-demand"; \
		echo "  To install npm: https://nodejs.org/ or 'brew install node'"; \
	fi
	@echo "[bootstrap] Installing pre-commit hooks..."
	@$(RUN_PY) -m tools.install_precommit_hook || echo "⚠️  Could not install custom git hook"
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
		npm install; \
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
validate-env: ## Validate that required packages are installed in the environment
	@echo "🔍 Validating environment: $(ENV_NAME) ($(ENV_TYPE))"
	@$(RUN_PY) -c "import sys; print(f'Python: {sys.version.split()[0]}')"
	@$(RUN_PY) -c "import mlx; print('MLX: ✓')" || (echo "❌ mlx not found - run 'make install'" && exit 1)
	@$(RUN_PY) -c "import mlx_vlm; v=getattr(mlx_vlm,'__version__','installed'); print(f'MLX-VLM: {v}')" || (echo "❌ mlx-vlm not found - run 'make install'" && exit 1)
	@$(RUN_PY) -c "import PIL; print(f'Pillow: {PIL.__version__}')" || (echo "❌ Pillow not found - run 'make install'" && exit 1)
	@echo "✓ Core runtime dependencies present"
	@$(RUN_PY) -m tools.validate_env 2>/dev/null || echo "ℹ️  Run 'python -m tools.validate_env' for detailed validation"

# Stubs
.PHONY: stubs
stubs: ## Generate/update local type stubs into ../typings/
	$(RUN_PY) tools/generate_stubs.py

.PHONY: stubs-clear
stubs-clear: ## Remove all generated stubs
	rm -rf ../typings

# ------------------------
# Run script
# ------------------------
.PHONY: check_models
check_models: ## Run the VLM checker (pass args via ARGS='--model <id> --image <path> ...')
	$(RUN_PY) -m check_models $(ARGS)

.PHONY: check_models-demo
check_models-demo: ## Demo run with minimal flags (override via ARGS to customize)
	$(RUN_PY) -m check_models $(ARGS)

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

# Tests
.PHONY: test
test: ## Run pytest
	@echo "[test] running tests"; \
	$(RUN_TOOL_PREFIX) pytest -q

.PHONY: test-cov
test-cov: ## Run pytest with coverage
	$(RUN_TOOL_PREFIX) pytest --cov=vlm --cov-report=term-missing --cov-report=xml -q

# Dependency management
.PHONY: deps-sync
deps-sync: ## Sync README dependency blocks
	$(RUN_PY) tools/update_readme_deps.py

.PHONY: update
update: ## Update conda environment and reinstall project with all dependencies
	@echo "[update] Updating pip in environment: $(ENV_NAME) ($(ENV_TYPE))"
	@$(RUN_PY) -m pip install --upgrade pip
	@echo "[update] Reinstalling project with all dependencies (dev+extras+torch)..."
	@$(RUN_PY) -m pip install -e .[dev,extras,torch] --upgrade
	@echo "✓ Environment updated and project reinstalled"
	@echo "  To also upgrade lock files, run: make upgrade-deps"

.PHONY: update-env
update-env: update ## Alias for 'update' target

# Aggregates
.PHONY: check
check: format lint typecheck test ## Core quality pipeline
	@echo "✓ All core checks completed"

.PHONY: quality
quality: ## Run consolidated quality script (ruff format+lint+stubgen+mypy on core file)
	$(RUN_PY) tools/check_quality.py
	@echo "[quality] NOTE: unit tests not executed by this target; use 'make check' or 'make ci' for full validation"

# CI pipeline
.PHONY: ci
ci: ## Run full CI pipeline (format check, quality script, dependency sync, tests)
	@echo "[ci] ruff format --check (fail if unformatted)" && $(RUN_TOOL_PREFIX) ruff format --check $(FMT_PATHS) || (echo "[ci] Formatting issues detected; run 'make format'" && exit 1)
	@echo "[ci] quality (ruff check + mypy)" && $(RUN_PY) tools/check_quality.py --no-format --no-fix || exit 1
	@echo "[ci] dependency sync check" && $(RUN_PY) tools/update_readme_deps.py --check || exit 1
	@echo "[ci] running tests"; \
	TEST_OUTPUT=$$($(RUN_TOOL_PREFIX) pytest 2>&1); RC=$$?; echo "$$TEST_OUTPUT" > .ci_tests.txt; echo "$$TEST_OUTPUT"; \
	if [ $$RC -ne 0 ]; then echo "[ci] tests failed (rc=$$RC)"; exit $$RC; fi; \
	PASSED=$$(echo "$$TEST_OUTPUT" | grep -Eo '[0-9]+ passed' | tail -1 | awk '{print $$1}'); \
	if [ -z "$$PASSED" ] || [ "$$PASSED" -lt 1 ]; then echo "[ci] ERROR: No tests passed"; exit 1; fi; \
	SKIPPED=$$(echo "$$TEST_OUTPUT" | grep -Eo '[0-9]+ skipped' | tail -1 | awk '{print $$1}' || echo 0); \
	if [ "$$SKIPPED" != "0" ] && [ -n "$$SKIPPED" ]; then \
	  echo "[ci] ERROR: $$SKIPPED tests skipped (skips not allowed in CI)"; \
	  exit 1; \
	fi; \
	echo "[ci] SUCCESS ($$PASSED tests passed; no skips)"

# Maintenance
.PHONY: check-outdated
check-outdated: ## Check for outdated dependencies
	$(RUN_PY) tools/check_outdated.py

.PHONY: audit
audit: ## Run security audit on dependencies
	@$(RUN_PY) -m pip install pip-audit --quiet 2>/dev/null || echo "[audit] Installing pip-audit..."
	@$(RUN_PY) -m pip install pip-audit --quiet || (echo "[audit] pip-audit installation failed"; exit 1)
	$(RUN_PY) -m pip_audit

# Cleanup
.PHONY: clean-pyc
clean-pyc: ## Remove Python cache files and bytecode
	find . -name '__pycache__' -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete -o -name '*.pyo' -delete -o -name '*~' -delete

.PHONY: clean
clean: clean-pyc ## Clean up Python cache files
	@echo "Cleanup complete"
