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

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available targets (package-local):\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/: /' | sort
	@echo "\nEnvironment: active='$(CONDA_ACTIVE)' target='$(CONDA_ENV)'"

# Installation
.PHONY: install
install: ## Editable install (runtime only)
	$(RUN_PY) -m pip install -e .

.PHONY: install-dev
install-dev: ## Editable install with dev extras
	$(RUN_PY) -m pip install -e .[dev]

.PHONY: bootstrap-dev
bootstrap-dev: ## pip bootstrap runtime + dev + extras
	$(RUN_PY) -m pip install --upgrade pip
	$(RUN_PY) -m pip install -e .[dev,extras]
	@echo "Bootstrap complete (local pyproject)"

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
	  if [ "$$ALLOW_EMPTY_TESTS" = "1" ]; then echo "[test] WARNING: 0 tests collected (override ALLOW_EMPTY_TESTS=1)"; else echo "[test] ERROR: 0 tests collected (set ALLOW_EMPTY_TESTS=1 to override)"; exit 1; fi; \
	else echo "[test] collected $$COUNT tests"; fi; \
	$(RUN_TOOL_PREFIX) pytest -q

.PHONY: test-cov
test-cov: ## Run pytest with coverage
	$(RUN_TOOL_PREFIX) pytest --cov=vlm --cov-report=term-missing --cov-report=xml -q

# Dependency sync
.PHONY: deps-sync
deps-sync: ## Sync README dependency blocks
	$(RUN_PY) tools/update_readme_deps.py

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
	if [ "$$SKIPPED" -gt 0 ] && [ "$$PASSED" -lt "$$COUNT" ]; then \
	  if [ "$$ALLOW_SKIPS" != "1" ]; then \
	    echo "[ci] ERROR: Skipped tests detected ($$SKIPPED) and passed $$PASSED != collected $$COUNT (set ALLOW_SKIPS=1 to override)"; exit 1; \
	  else \
	    echo "[ci] WARNING: $$SKIPPED tests skipped (override enabled)"; \
	  fi; \
	fi; \
	if [ "$$PASSED" != "$$COUNT" ]; then echo "[ci] ERROR: Collected $$COUNT tests but only $$PASSED passed"; exit 1; fi; \
	echo "[ci] SUCCESS (all $$PASSED tests executed; no skips)"

# Cleanup
.PHONY: clean-pyc
clean-pyc:
	find . -name '__pycache__' -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete -o -name '*.pyo' -delete -o -name '*~' -delete

.PHONY: clean
clean: clean-pyc
	@echo "Cleanup complete"
