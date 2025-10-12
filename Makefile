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
	@echo ""
	@echo "📚 Documentation: See docs/CONTRIBUTING.md for details"

.PHONY: install
install:
	pip install -e $(SRC)/

.PHONY: dev
dev:
	pip install -e "$(SRC)/[dev,extras,torch]"

.PHONY: run
run:
	python -m check_models --help

.PHONY: demo
demo:
	python -m check_models --verbose

.PHONY: test
test:
	pytest $(SRC)/tests/ -v

.PHONY: check
check:
	@$(MAKE) -C $(SRC) check

.PHONY: quality
quality:
	@$(MAKE) -C $(SRC) quality

.PHONY: ci
ci:
	@$(MAKE) -C $(SRC) ci

.PHONY: format
format:
	ruff format $(SRC)/

.PHONY: lint
lint:
	ruff check $(SRC)/

.PHONY: typecheck
typecheck:
	mypy $(SRC)/check_models.py

.PHONY: clean
clean:
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
