# Root shim Makefile: forwards targets to package-local Makefile in ./vlm

.PHONY: help
help:
	@$(MAKE) -C vlm help

.PHONY: quality
quality: ## Run consolidated quality checks (alias to vlm/quality)
	@$(MAKE) -C vlm quality

.PHONY: ci
ci: ## Full CI pipeline (alias to vlm/ci)
	@$(MAKE) -C vlm ci

# Forward any target to ./vlm/Makefile
%:
	@$(MAKE) -C vlm $@
