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
	@echo "  make quality          Run full quality checks (ruff format+lint+mypy+suppression-audit+ty+pyrefly+pytest+shellcheck+markdownlint)"
	@echo "  make ty               Run Ty type checking with the resolved mlx-vlm interpreter"
	@echo "  make ci               Run full CI pipeline (strict)"
	@echo "  make format           Format code with ruff"
	@echo "  make stubs            Generate type stubs for mlx-vlm (typings/)"
	@echo ""
	@echo "📚 Documentation: See docs/CONTRIBUTING.md for details"

.PHONY: install
install: ## Install the package in editable mode
	@$(MAKE) -C $(SRC) install

.PHONY: dev
dev: ## Setup dev environment with all dependencies
	@$(MAKE) -C $(SRC) install-all

.PHONY: run
run: ## Show usage help
	@$(MAKE) -C $(SRC) check_models ARGS='--help'

.PHONY: demo
demo: ## Run demo with verbose output
	@$(MAKE) -C $(SRC) check_models ARGS='--verbose'

.PHONY: test
test: ## Run tests with pytest
	@$(MAKE) -C $(SRC) test

.PHONY: check
check: ## Run core quality pipeline (format, lint, typecheck, test)
	@$(MAKE) -C $(SRC) check

.PHONY: quality
quality: ## Run full quality checks (ruff format+lint+mypy+suppression-audit+ty+pyrefly+pytest+shellcheck+markdownlint)
	@$(MAKE) -C $(SRC) quality

.PHONY: ci
ci: ## Run full CI pipeline (strict mode)
	@$(MAKE) -C $(SRC) ci

.PHONY: format
format: ## Format code with ruff
	@$(MAKE) -C $(SRC) format

.PHONY: lint
lint: ## Lint code with ruff
	@$(MAKE) -C $(SRC) lint

.PHONY: typecheck
typecheck: ## Run mypy type checking
	@$(MAKE) -C $(SRC) typecheck

.PHONY: ty
ty: ## Run Ty type checking with the resolved repo interpreter
	@$(MAKE) -C $(SRC) ty ARGS='$(ARGS)'

.PHONY: clean
clean: ## Remove generated files and caches
	@$(MAKE) -C $(SRC) clean
	rm -f $(SRC)/output/results.html $(SRC)/output/results.md $(SRC)/output/model_gallery.md
	rm -f $(SRC)/output/results.tsv $(SRC)/output/results.jsonl $(SRC)/output/results.history.jsonl
	rm -f $(SRC)/output/diagnostics.md $(SRC)/output/check_models.log $(SRC)/output/environment.log
	find $(SRC)/output/repro_bundles -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-all
clean-all: clean ## Deep clean including build artifacts and stubs
	@$(MAKE) -C $(SRC) clean-all

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

.PHONY: quality-strict
quality-strict: ## Run quality checks with strict markdown linting (requires node/npm)
	$(MAKE) -C $(SRC) quality-strict

.PHONY: install-markdownlint
install-markdownlint: ## Install markdownlint-cli2 via npm (requires Node.js)
	$(MAKE) -C $(SRC) install-markdownlint

