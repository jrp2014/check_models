# Documentation Alignment Summary

## Overview

This document summarizes the documentation updates made to align with the Makefile changes, particularly the addition of help comments (`##`) to all targets and the new `update` target.

## Changes Made

### 1. Root README.md (`/README.md`)

**Updated**: Available Commands section

**Changes**:

- Added `make update` - Update conda environment and project dependencies
- Added `make check` - Run core quality pipeline (format, lint, typecheck, test)
- Added `make ci` - Run full CI pipeline (strict mode)
- Added `make lint` - Lint code with ruff
- Added `make typecheck` - Run mypy type checking

**Benefit**: The command table now comprehensively documents all major Makefile targets that users and contributors need.

### 2. CONTRIBUTING.md (`docs/CONTRIBUTING.md`)

**Updated**: Dependency Management section

**Changes**:
- Added `make update` to the dependency management command list
- Added new subsection: "Keeping Your Environment Up-to-Date"
- Documented when to use `make update` vs `make upgrade-deps`
- Clarified that `make update` is recommended after pulling changes

**Key Points**:
```bash
# After pulling changes from the repository:
make update

# This command:
# - Updates pip in your conda/venv environment
# - Reinstalls the project with all dependencies (dev, extras, torch)
# - Ensures you have the latest versions compatible with lock files
```

**Note**: Clarified that `make update` does NOT update lock files; use `make upgrade-deps` for that.

### 3. Makefile Help Comments

**All targets now have `##` help annotations**:

#### Root Makefile (`/Makefile`)

- `install` - Install the package in editable mode
- `dev` - Setup dev environment with all dependencies
- `run` - Show usage help
- `demo` - Run demo with verbose output
- `test` - Run tests with pytest
- `check` - Run core quality pipeline (format, lint, typecheck, test)
- `quality` - Run linting and type checks
- `ci` - Run full CI pipeline (strict mode)
- `format` - Format code with ruff
- `lint` - Lint code with ruff
- `typecheck` - Run mypy type checking
- `clean` - Remove generated files and caches
- `check_models` - Run VLM checker
- `update` - Update conda environment and reinstall project dependencies
- `update-env` - Alias for 'update' target

#### Source Makefile (`src/Makefile`)

- `clean-pyc` - Remove Python cache files and bytecode
- `clean` - Clean up Python cache files

## Consistency Check

✅ **Root README.md**: Documents all major user-facing commands  
✅ **CONTRIBUTING.md**: Documents all development commands including dependency management  
✅ **Root Makefile**: All targets have `##` help comments  
✅ **Source Makefile**: All targets have `##` help comments  
✅ **Help output**: Custom formatted help in root Makefile remains functional  
✅ **Auto-discovery**: All `##` annotations enable grep-based help systems

## Usage Patterns

### For Users

```bash
# View all available commands
make help

# Quick start
make install
make check_models ARGS='--model X --image Y'

# Update after pulling changes
make update
```

### For Contributors

```bash
# Setup development environment
make dev

# Update after pulling changes
make update

# Run quality checks
make quality

# Run full CI pipeline
make ci

# Run tests
make test
```

### For Dependency Management

```bash
# Update environment (after pulling changes)
make update

# Upgrade dependencies to latest
make upgrade-deps

# Regenerate lock files
make lock-deps

# Sync environment with lock files
make sync-deps

# Check for outdated packages
make check-outdated
```

## Benefits of These Changes

1. **Discoverability**: All commands are now properly documented in both Makefiles and README files
2. **Consistency**: Same information available via `make help`, README.md, and CONTRIBUTING.md
3. **Future-proof**: Help annotations enable automated help generation if needed later
4. **User Experience**: Clear documentation helps new users and contributors get started quickly
5. **Maintenance**: Centralized command documentation reduces confusion and outdated docs

## Related Files

- `/Makefile` - Root-level build automation
- `/README.md` - Project overview and quick start
- `/docs/CONTRIBUTING.md` - Contributor guide
- `/docs/notes/UPDATE_TARGET_ADDED.md` - Technical details of the `update` target
- `/src/Makefile` - Source-level build automation

## Next Steps

All documentation is now aligned. Future changes should:

1. Update `##` comments in Makefiles when adding/modifying targets
2. Update README.md Available Commands section for user-facing commands
3. Update CONTRIBUTING.md for development-related commands
4. Run `make help` to verify changes appear correctly
