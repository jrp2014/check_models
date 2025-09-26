## Root Makefile: friendly aliases that proxy to the package Makefile in ./vlm

.DEFAULT_GOAL := help

VLM := vlm
FWD := $(MAKE) -C $(VLM)

.PHONY: help
help: ## Show this help with aligned target descriptions
	@echo "Available targets:\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) \
		| sed 's/:.*##/: /' \
		| awk 'BEGIN {FS=": "; C=28} {printf "  \033[36m%-*s\033[0m %s\n", C, $$1, $$2}'
	@echo "\n(These commands proxy to ./vlm/Makefile so you don\'t need \"-C vlm\".)"

# ------------------------
# Install / bootstrap
# ------------------------
.PHONY: install
install: ## Editable install (runtime only)
	@$(FWD) install

.PHONY: install-dev
install-dev: ## Editable install with dev extras
	@$(FWD) install-dev

.PHONY: bootstrap-dev
bootstrap-dev: ## Pip bootstrap runtime + dev + extras
	@$(FWD) bootstrap-dev

# ------------------------
# Quality & tests
# ------------------------
.PHONY: format
format: ## Ruff format
	@$(FWD) format

.PHONY: lint
lint: ## Ruff lint
	@$(FWD) lint

.PHONY: lint-fix
lint-fix: ## Ruff lint with --fix
	@$(FWD) lint-fix

.PHONY: typecheck
typecheck: ## Mypy type checking
	@$(FWD) typecheck

.PHONY: test
test: ## Run pytest
	@$(FWD) test

.PHONY: test-cov
test-cov: ## Run pytest with coverage
	@$(FWD) test-cov

.PHONY: deps-sync
deps-sync: ## Sync README dependency blocks from pyproject
	@$(FWD) deps-sync

.PHONY: check
check: ## Format + lint + typecheck + tests
	@$(FWD) check

.PHONY: quality
.PHONY: lint-sh
lint-sh: ## Lint shell scripts with shellcheck
	@if command -v shellcheck >/dev/null 2>&1; then \
	  shellcheck vlm/tools/*.sh; \
	else \
	  echo "shellcheck not found. Install with: brew install shellcheck"; exit 1; \
	fi

.PHONY: fmt-sh
fmt-sh: ## Format shell scripts with shfmt
	@if command -v shfmt >/dev/null 2>&1; then \
	  shfmt -w -i 2 -bn -ci vlm/tools/*.sh; \
	else \
	  echo "shfmt not found. Install with: brew install shfmt"; exit 1; \
	fi

.PHONY: spell
spell: ## Spell-check with codespell
	@if command -v codespell >/dev/null 2>&1; then \
	  codespell -q 3 --skip "*.lock,*.min.js,*.map,*.svg,*.png,*.jpg,*.ico,.git,dist,build,.venv"; \
	else \
	  echo "codespell not found. Install with: pip install codespell"; exit 1; \
	fi
quality: ## Consolidated quality script (ruff format+lint+mypy on core file)
	@$(FWD) quality
.PHONY: lint-md
lint-md: ## Lint Markdown files with markdownlint-cli2 (via npx if available)
	@if command -v npx >/dev/null 2>&1; then \
	  npx --yes markdownlint-cli2 "**/*.md" "#node_modules"; \
	elif command -v markdownlint-cli2 >/dev/null 2>&1; then \
	  markdownlint-cli2 "**/*.md" "#node_modules"; \
	else \
	  echo "markdownlint not found. Install with: npm i -g markdownlint-cli2"; exit 1; \
	fi

.PHONY: ci
ci: ## Full CI pipeline (format check, quality, deps sync, tests)
	@$(FWD) ci

# ------------------------
# Typings
# ------------------------
.PHONY: check_models
check_models: ## Run the VLM checker (pass args via ARGS='--model <id> --image <path> ...')
	@$(FWD) check_models

.PHONY: stubs
stubs: ## Generate/update local type stubs into ./typings/
	@$(FWD) stubs

.PHONY: stubs-clear
stubs-clear: ## Remove all generated stubs
	@$(FWD) stubs-clear

# ------------------------
# Cleanup
# ------------------------
.PHONY: clean-pyc
clean-pyc: ## Remove Python caches
	@$(FWD) clean-pyc

.PHONY: clean
clean: ## Remove caches and temporary files
	@$(FWD) clean

# Fallback: forward any unknown target directly to ./vlm
%:
	@$(FWD) $@
