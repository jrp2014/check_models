# Package-local Makefile for mlx-vlm-check (lives alongside pyproject.toml)

PYTHON ?= python
CONDA_ENV ?= mlx-vlm
CONDA_ACTIVE := $(shell echo $$CONDA_DEFAULT_ENV)

ifeq ($(CONDA_ACTIVE),$(CONDA_ENV))
	RUN_PY := $(PYTHON)
	RUN_TOOL_PREFIX :=
else
	RUN_PY := conda run -n $(CONDA_ENV) $(PYTHON)
	RUN_TOOL_PREFIX := conda run -n $(CONDA_ENV)
endif

MYPY_CONFIG := pyproject.toml
FMT_PATHS := check_models.py tests tools

# Warn if wrong conda environment is active
ifneq ($(CONDA_ACTIVE),$(CONDA_ENV))
    ifneq ($(CONDA_ACTIVE),)
        $(warning ⚠️  Expected conda env '$(CONDA_ENV)' but '$(CONDA_ACTIVE)' is active)
        $(warning ⚠️  Activate with: conda activate $(CONDA_ENV))
        $(warning ⚠️  Commands will use: conda run -n $(CONDA_ENV))
    endif
endif

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
	@echo "[bootstrap] Installing pre-commit hooks..."
	@$(RUN_PY) -m vlm.tools.install_precommit_hook || echo "⚠️  Could not install custom git hook"
	@if command -v pre-commit >/dev/null 2>&1; then \
		echo "[bootstrap] Installing pre-commit framework hooks..."; \
		$(RUN_TOOL_PREFIX) pre-commit install || echo "⚠️  pre-commit install failed"; \
	else \
		echo "[bootstrap] pre-commit framework not found (optional)"; \
		echo "  Install with: pip install pre-commit && pre-commit install"; \
	fi
	@echo "[bootstrap] Running environment validation..."
	@$(RUN_PY) -m vlm.tools.validate_env || echo "⚠️  Validation found issues (run 'python -m vlm.tools.validate_env' for details)"
	@echo "✓ Bootstrap complete"

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
	$(RUN_PY) -m vlm.check_models $(ARGS)

.PHONY: check_models-demo
check_models-demo: ## Demo run with minimal flags (override via ARGS to customize)
	$(RUN_PY) -m vlm.check_models $(ARGS)

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
	@echo "[test] collecting tests"; \
	$(RUN_TOOL_PREFIX) pytest --collect-only -q > .test_collect.txt || exit 1; \
	COUNT=$$(awk '/::/ {c++} END {print c+0}' .test_collect.txt); \
	if [ "$$COUNT" -lt 1 ]; then \
	  if [ "$$CI" != "true" ] && [ "$$ALLOW_EMPTY_TESTS" = "1" ]; then \
	    echo "[test] WARNING: 0 tests collected (local override ALLOW_EMPTY_TESTS=1)"; \
	  else \
	    echo "[test] ERROR: 0 tests collected"; \
	    exit 1; \
	  fi; \
	else \
	  echo "[test] collected $$COUNT tests"; \
	fi; \
	$(RUN_TOOL_PREFIX) pytest -q

.PHONY: test-cov
test-cov: ## Run pytest with coverage
	$(RUN_TOOL_PREFIX) pytest --cov=vlm --cov-report=term-missing --cov-report=xml -q

# Dependency management
.PHONY: deps-sync
deps-sync: ## Sync README dependency blocks
	$(RUN_PY) tools/update_readme_deps.py

.PHONY: lock-deps
lock-deps: ## Generate requirements.txt from requirements.in using pip-tools
	@echo "[lock-deps] Installing pip-tools if needed..."
	@$(RUN_PY) -m pip install pip-tools --quiet || echo "pip-tools already installed"
	@echo "[lock-deps] Compiling requirements.txt from requirements.in..."
	$(RUN_TOOL_PREFIX) pip-compile requirements.in --output-file=requirements.txt --resolver=backtracking
	@echo "[lock-deps] Compiling requirements-dev.txt from requirements-dev.in..."
	$(RUN_TOOL_PREFIX) pip-compile requirements-dev.in --output-file=requirements-dev.txt --resolver=backtracking
	@echo "✓ Lock files generated"

.PHONY: sync-deps
sync-deps: ## Sync installed packages with requirements.txt (using pip-sync)
	@$(RUN_PY) -m pip install pip-tools --quiet || echo "pip-tools already installed"
	$(RUN_TOOL_PREFIX) pip-sync requirements-dev.txt
	@echo "✓ Dependencies synced"

.PHONY: upgrade-deps
upgrade-deps: ## Upgrade all dependencies to latest compatible versions
	@$(RUN_PY) -m pip install pip-tools --quiet || echo "pip-tools already installed"
	$(RUN_TOOL_PREFIX) pip-compile requirements.in --output-file=requirements.txt --upgrade --resolver=backtracking
	$(RUN_TOOL_PREFIX) pip-compile requirements-dev.in --output-file=requirements-dev.txt --upgrade --resolver=backtracking
	@echo "✓ Dependencies upgraded. Review changes and commit lock files."

# Aggregates
.PHONY: check
check: format lint typecheck test ## Core quality pipeline
	@echo "All core checks completed"

.PHONY: quality
quality: ## Run consolidated quality script (ruff format+lint+stubgen+mypy on core file)
	$(RUN_PY) tools/check_quality.py
	@echo "[quality] NOTE: unit tests not executed by this target; use 'make check' or 'make ci' for full validation"

# CI pipeline
.PHONY: ci
ci: ## Run full CI pipeline (format check, quality script, dependency sync, test count sanity, tests)
	@echo "[ci] ruff format --check (fail if unformatted)" && $(RUN_TOOL_PREFIX) ruff format --check $(FMT_PATHS) || (echo "[ci] Formatting issues detected; run 'make format'" && exit 1)
	@echo "[ci] quality (ruff check + mypy)" && $(RUN_PY) tools/check_quality.py --no-format --no-fix || exit 1
	@echo "[ci] dependency sync check" && $(RUN_PY) tools/update_readme_deps.py --check || exit 1
	@echo "[ci] collecting tests (root config)"; \
	$(RUN_TOOL_PREFIX) pytest --collect-only -q > .ci_collect.txt || exit 1; \
	COUNT=$$(awk '/::/ {c++} END {print c+0}' .ci_collect.txt); \
	if [ "$$COUNT" -lt 1 ]; then echo "[ci] ERROR: No tests collected"; cat .ci_collect.txt; exit 1; else echo "[ci] collected $$COUNT tests"; fi; \
	echo $$COUNT > .ci_count
	@echo "[ci] running tests"; \
	COUNT=$$(cat .ci_count); \
	TEST_OUTPUT=$$($(RUN_TOOL_PREFIX) pytest 2>&1); RC=$$?; echo "$$TEST_OUTPUT" > .ci_tests.txt; echo "$$TEST_OUTPUT"; \
	if [ $$RC -ne 0 ]; then echo "[ci] tests failed (rc=$$RC)"; exit $$RC; fi; \
	PASSED=$$(grep -Eo '[0-9]+ passed' .ci_tests.txt | tail -1 | awk '{print $$1}'); \
	if [ -z "$$PASSED" ]; then echo "[ci] ERROR: Could not parse passed test count"; exit 1; fi; \
	SKIPPED=$$(grep -Eo '[0-9]+ skipped' .ci_tests.txt | tail -1 | awk '{print $$1}'); [ -z "$$SKIPPED" ] && SKIPPED=0; \
	if [ "$$PASSED" -lt 1 ]; then echo "[ci] ERROR: Zero tests passed"; exit 1; fi; \
	if [ "$$SKIPPED" -gt 0 ]; then \
	  echo "[ci] ERROR: $$SKIPPED tests skipped (skips not allowed in CI)"; \
	  exit 1; \
	fi; \
	if [ "$$PASSED" != "$$COUNT" ]; then echo "[ci] ERROR: Collected $$COUNT tests but only $$PASSED passed"; exit 1; fi; \
	echo "[ci] SUCCESS (all $$PASSED tests executed; no skips)"

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
clean-pyc:
	find . -name '__pycache__' -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete -o -name '*.pyo' -delete -o -name '*~' -delete

.PHONY: clean
clean: clean-pyc
	@echo "Cleanup complete"
