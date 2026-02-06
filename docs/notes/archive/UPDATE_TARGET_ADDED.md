# Update Target Added to Makefile

## Summary

Added new `update` target to both root and `src/` Makefiles to simplify updating conda environments and project dependencies.

## Changes Made

### 1. Root Makefile (`/Makefile`)

Added two new targets that delegate to `src/Makefile`:

```makefile
.PHONY: update
update: ## Update conda environment and reinstall project dependencies
    $(MAKE) -C $(SRC) update

.PHONY: update-env
update-env: update ## Alias for 'update' target
```

Also updated the help message to include the new target:

```text
🛠️  Development:
  make dev              Setup dev environment
  make update           Update conda environment and project dependencies
  ...
```

### 2. Source Makefile (`src/Makefile`)

Added the implementation in the dependency management section:

```makefile
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
```

## Usage

From either the root directory or `src/` directory:

```bash
# Update pip and reinstall project with all dependencies
make update

# Or use the alias
make update-env
```

## What It Does

1. **Updates pip** in the current Python environment (conda or venv)
2. **Reinstalls the project** in editable mode with all optional dependency groups:
   - `dev` - Development tools (ruff, mypy, pytest, etc.)
   - `extras` - Optional runtime dependencies
   - `torch` - PyTorch for certain model backends
3. Uses the `--upgrade` flag to get latest compatible versions
4. Respects the Makefile's environment detection logic (works with conda, venv, system Python)

## Relationship to Other Targets

- **`make dev`** - Initial setup (installs dev dependencies)
- **`make update`** - Update existing installation (new target)
- **`make upgrade-deps`** - Regenerate lock files with newer versions
- **`make sync-deps`** - Sync environment to match lock files exactly

## Notes

- The target automatically detects whether you're in conda or a venv
- It will show the environment name and type when running
- Does NOT update lock files automatically (use `make upgrade-deps` for that)
- Works with the existing `CONDA_ENV` override if you need to target a different environment

## Example Output

```text
[update] Updating pip in environment: mlx-vlm (conda)
Requirement already satisfied: pip in ...
[update] Reinstalling project with all dependencies (dev+extras+torch)...
Obtaining file:///Users/jrp/Documents/AI/mlx/scripts/src
...
✓ Environment updated and project reinstalled
  To also upgrade lock files, run: make upgrade-deps
```
