# Makefile for mlx-vlm-check workflows
#
# Usage examples:
#   make help
#   make install-dev            # install package + dev extras into current (or conda run)
#   make format                 # ruff format
#   make lint                   # ruff lint (no fix)
#   make lint-fix               # ruff lint with --fix
#   make typecheck              # mypy type checking
#   make test                   # run pytest test suite
#   make quality                # run integrated quality script (format + lint + mypy)
#   make run ARGS="--verbose --models microsoft/Florence-2-large --image demo.jpg"
#   make smoke                  # (placeholder) run an ultra-fast smoke invocation
#
# Variables you can override:
#   CONDA_ENV=mlx-vlm  PYTHON=python  ARGS="--verbose"
#   For example: make test CONDA_ENV=mlx-vlm
#
# If the active conda env (CONDA_DEFAULT_ENV) is not the target, commands are
# executed via `conda run -n $(CONDA_ENV)` to avoid manual activation.

# ---- Configuration ----
CONDA_ENV ?= mlx-vlm
PYTHON ?= python
REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CONDA_ACTIVE := $(shell echo $$CONDA_DEFAULT_ENV)

# Decide whether to prefix commands with conda run
ifeq ($(CONDA_ACTIVE),$(CONDA_ENV))
	RUN_PY := $(PYTHON)
	RUN_TOOL_PREFIX :=
else
	RUN_PY := conda run -n $(CONDA_ENV) $(PYTHON)
	RUN_TOOL_PREFIX := conda run -n $(CONDA_ENV)
endif

MYPY_CONFIG := vlm/pyproject.toml
FMT_PATHS := vlm/check_models.py vlm/tests
PACKAGE := mlx-vlm-check

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Available targets (default CONDA_ENV=$(CONDA_ENV)):\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/: /' | sort
	@echo "\nEnvironment detection: active='$(CONDA_ACTIVE)' target='$(CONDA_ENV)'"
	@if [ "$(CONDA_ACTIVE)" != "$(CONDA_ENV)" ]; then \
		echo "(Will invoke commands through: conda run -n $(CONDA_ENV))"; \
	else \
		echo "(Using already-active environment)"; \
	fi

# ---- Installation ----
.PHONY: install-dev
install-dev: ## Install package in editable mode with dev extras
	$(RUN_PY) -m pip install -e .[dev]

.PHONY: install
install: ## Install runtime package only (no dev extras)
	$(RUN_PY) -m pip install -e .

# ---- Formatting & Lint ----
.PHONY: format
format: ## Apply code formatting (ruff format)
	$(RUN_TOOL_PREFIX) ruff format $(FMT_PATHS)

.PHONY: lint
lint: ## Run ruff lint (no fixes)
	$(RUN_TOOL_PREFIX) ruff check $(FMT_PATHS)

.PHONY: lint-fix
lint-fix: ## Run ruff lint with auto-fixes
	$(RUN_TOOL_PREFIX) ruff check --fix $(FMT_PATHS)

# ---- Type Checking ----
.PHONY: typecheck
typecheck: ## Run mypy type checking
	$(RUN_TOOL_PREFIX) mypy --config-file $(MYPY_CONFIG) vlm/check_models.py vlm/tests

# ---- Tests ----
.PHONY: test
test: ## Run pytest suite
	$(RUN_TOOL_PREFIX) pytest -q

.PHONY: test-cov
test-cov: ## Run pytest with coverage (terminal + XML)
	$(RUN_TOOL_PREFIX) pytest --cov=vlm --cov-report=term-missing --cov-report=xml -q

# ---- Aggregate Quality ----
.PHONY: quality
quality: ## Run integrated quality script (format + lint + mypy)
	$(RUN_PY) vlm/tools/check_quality.py

.PHONY: quality-strict
quality-strict: ## Run quality script requiring tools & skipping stub generation
	$(RUN_PY) vlm/tools/check_quality.py --require --no-stubs

# ---- Execution / Smoke ----
ARGS ?=
.PHONY: run
run: ## Run the check_models CLI with $(ARGS)
	$(RUN_PY) vlm/check_models.py $(ARGS)

.PHONY: smoke
smoke: ## Very small smoke test (override ARGS for custom) - placeholder
	$(RUN_PY) vlm/check_models.py --help >/dev/null
	@echo "Smoke help invocation OK"

# ---- Combined convenience target ----
.PHONY: check
check: format lint typecheck test ## Run format, lint, typecheck, tests (fast aggregate)
	@echo "All core checks completed"

# ---- Cleanup helpers ----
.PHONY: clean-pyc
clean-pyc: ## Remove Python cache artifacts
	find . -name '__pycache__' -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete -o -name '*.pyo' -delete -o -name '*~' -delete

.PHONY: clean
clean: clean-pyc ## General cleanup (currently only pyc)
	@echo "Cleanup complete"
