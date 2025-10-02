## Root Makefile: friendly aliases that proxy to the package Makefile in ./vlm

.DEFAULT_GOAL := help

VLM := vlm
FWD := $(MAKE) -C $(VLM)

.PHONY: help
help: ## Show this help with aligned target descriptions
	@echo "\033[1;35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
	@echo "\033[1;35m  MLX VLM Check - Vision-Language Model Benchmarking Tool\033[0m"
	@echo "\033[1;35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n"
	@echo "\033[1;33m🚀 PRIMARY TARGET:\033[0m\n"
	@echo "  \033[1;36mmake check_models ARGS='--model <id> --image <path>'\033[0m"
	@echo "    Run the VLM checker (the main purpose of this project)\n"
	@echo "\033[1mDevelopment targets:\033[0m\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | grep -v "check_models:" \
		| sed 's/:.*##/: /' \
		| awk 'BEGIN {FS=": "; C=28} {printf "  \033[36m%-*s\033[0m %s\n", C, $$1, $$2}'
	@echo "\n\033[2m(These commands proxy to ./vlm/Makefile so you don't need \"-C vlm\".)\033[0m"

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
quality: ## Consolidated quality script (ruff format+lint+mypy on core file)
	@$(FWD) quality

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
