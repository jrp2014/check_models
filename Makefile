.DEFAULT_GOAL := help

SRC := src

.PHONY: help
help: ## Show this help message
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  MLX VLM Check - Vision-Language Model Benchmarking  "
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "🚀 Getting Started:"
	@echo "  make install          Install the package"
	@echo "  make run              Show usage help"
	@echo ""
	@echo "📊 Common Tasks:"
	@echo "  make demo             Run on example (if you have images)"
	@echo "  make clean            Remove generated files"
	@echo ""
	@echo "🛠️  Development:"
	@echo "  make dev              Setup dev environment"
	@echo "  make update           Update conda environment and project dependencies"
	@echo "  make test             Run tests"
	@echo "  make check            Run format, lint, typecheck, and tests"
	@echo "  make quality          Run linting and type checks"
	@echo "  make ci               Run full CI pipeline (strict)"
	@echo "  make format           Format code with ruff"
	@echo "  make stubs            Generate type stubs for mlx-vlm"
	@echo ""
	@echo "📚 Documentation: See docs/CONTRIBUTING.md for details"

.PHONY: install
install: ## Install the package in editable mode
	pip install -e $(SRC)/

.PHONY: dev
dev: ## Setup dev environment with all dependencies
	pip install -e "$(SRC)/[dev,extras,torch]"

.PHONY: run
run: ## Show usage help
	python -m check_models --help

.PHONY: demo
demo: ## Run demo with verbose output
	python -m check_models --verbose

.PHONY: test
test: ## Run tests with pytest
	pytest $(SRC)/tests/ -v

.PHONY: check
check: ## Run core quality pipeline (format, lint, typecheck, test)
	@$(MAKE) -C $(SRC) check

.PHONY: quality
quality: ## Run linting and type checks
	@$(MAKE) -C $(SRC) quality

.PHONY: ci
ci: ## Run full CI pipeline (strict mode)
	@$(MAKE) -C $(SRC) ci

.PHONY: format
format: ## Format code with ruff
	ruff format $(SRC)/

.PHONY: lint
lint: ## Lint code with ruff
	ruff check $(SRC)/

.PHONY: typecheck
typecheck: ## Run mypy type checking
	mypy $(SRC)/check_models.py

.PHONY: clean
clean: ## Remove generated files and caches
	rm -rf output/*.html output/*.md
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

.PHONY: check_models
check_models: ## Run VLM checker (pass args: make check_models ARGS='--model X --image Y')
	$(MAKE) -C $(SRC) check_models ARGS='$(ARGS)'

.PHONY: update
update: ## Update conda environment and reinstall project dependencies
	$(MAKE) -C $(SRC) update

.PHONY: update-env
update-env: update ## Alias for 'update' target


.PHONY: deps-sync
deps-sync: ## Sync README dependency blocks with pyproject.toml
	$(MAKE) -C $(SRC) deps-sync

.PHONY: stubs
stubs: ## Generate type stubs for mlx-vlm
	$(MAKE) -C $(SRC) stubs

.PHONY: stubs-clear
stubs-clear: ## Remove generated type stubs
	$(MAKE) -C $(SRC) stubs-clear

