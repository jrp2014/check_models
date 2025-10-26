# Contributing to MLX VLM Check

Thank you for your interest in contributing to MLX VLM Check! This document guides you through the practical workflow of contributing: getting set up, making changes, and submitting them.

**Target audience**: New contributors, anyone submitting a pull request.

> **Note**: This project was restructured in October 2025. The old `vlm/` directory is now `src/`. All `make` commands now use the root `Makefile` instead of `make -C src` or `make -C vlm`.

**Related documents**:

- [README.md](../README.md) - Project overview and quick start
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Detailed coding standards and technical conventions
- [src/README.md](../src/README.md) - Detailed package usage and CLI documentation
- [notes/](notes/) - Design notes and project evolution

## Table of Contents

- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Quality Checks](#quality-checks)
- [Testing](#testing)
- [Dependency Management](#dependency-management)
- [Submitting Changes](#submitting-changes)
- [Getting Help](#getting-help)

## Development Setup

### Prerequisites

- macOS (preferably with Apple Silicon)
- Python 3.13+
- conda or miniconda
- Git

### Initial Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/jrp2014/scripts.git
   cd scripts
   ```

2. **Create and activate conda environment**:

   ```bash
   conda create -n mlx-vlm python=3.13
   conda activate mlx-vlm
   ```

3. **Install development dependencies**:

   ```bash
   make bootstrap-dev
   ```

   This will:
   - Install all Python dependencies (runtime + dev + extras)
   - Install the package in editable mode
   - Install huggingface_hub CLI tools (`huggingface-cli`)
   - Install markdownlint-cli2 (if Node.js/npm is available)
   - Set up git hooks automatically via pre-commit

   **Optional dependencies**:

   - **PyTorch** (needed for some models): `make install-torch`
   - **Everything** (extras + torch + dev): `make install-all`
   - **Markdown linting only**: `make install-markdownlint` (requires Node.js/npm)

4. **Verify installation**:

   ```bash
   cd src
   python -m tools.validate_env
   ```

### Manual Environment Validation

If bootstrap didn't work or you want to check your environment:

```bash
# Check environment health (run from src/ directory)
cd src
python -m tools.validate_env

# Auto-fix common issues
python -m tools.validate_env --fix
```

## Making Changes

### Workflow

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:

   - Follow the [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) conventions
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks locally**:

   ```bash
   make quality
   ```

4. **Run tests**:

   ```bash
   make test
   ```

### Code Style

- **Python**: Follow PEP 8, enforced by ruff
- **Line length**: 100 characters (configurable exceptions for long strings)
- **Type annotations**: Required for all new code
- **Docstrings**: Required for non-obvious functions

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed conventions.

## Quality Checks

### Automated Checks

The project uses several automated quality checks:

1. **Ruff** (formatting and linting):

   ```bash
   make format      # Format code
   make lint        # Check linting (alias for quality)
   ```

2. **Mypy** (type checking):

   ```bash
   make typecheck   # Type check with mypy
   ```

3. **Tests**:

   ```bash
   make test        # Run all tests
   ```

4. **Combined quality check**:

   ```bash
   make quality     # Runs format + lint + typecheck + markdownlint (if available)
   ```

5. **Markdown linting** (optional):

   Automatically included in `make quality` if `markdownlint-cli2` is installed.

   ```bash
   # Install markdown linting (requires Node.js/npm)
   make install-markdownlint

   # Or run via npx (on-demand download)
   npx markdownlint-cli2 '**/*.md'
   ```

   If neither npm nor npx is available, markdown linting is gracefully skipped with a warning.

### Git Hooks

The project uses git hooks to enforce quality:

- **Pre-commit**: Automatically runs ruff format/lint, mypy, and dependency sync checks
- **Pre-push**: Runs full quality checks before pushing

To bypass hooks (not recommended):

```bash
git commit --no-verify
git push --no-verify
```

### Continuous Integration

All pull requests must pass:

- Ruff format check (no unformatted code)
- Ruff lint check (all rules)
- Mypy type checking (strict mode)
- Dependency sync verification
- All tests (no skips allowed in CI)
- Markdown linting

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest src/tests/test_specific.py

# Run with verbose output
pytest src/tests/ -v
```

### Writing Tests

- Place tests in `src/tests/`
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Include both positive and negative test cases
- Aim for high coverage of critical paths

## Submitting Changes

### Pull Request Process

1. **Ensure all checks pass**:

   ```bash
   make ci
   ```

2. **Update documentation**:

   - Update README.md if adding user-facing features
   - Update IMPLEMENTATION_GUIDE.md if changing technical conventions
   - Add docstrings to new functions

3. **Commit your changes**:

   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `refactor:` for code refactoring
   - `test:` for adding tests
   - `chore:` for maintenance tasks

4. **Push and create PR**:

   ```bash
   git push origin feature/your-feature-name
   ```

   Then create a pull request on GitHub.

### PR Requirements

Your PR must:

- Pass all CI checks
- Include tests for new functionality
- Update relevant documentation
- Have a clear description of changes
- Reference any related issues

## Dependency Management

### Dependency Structure

The project organizes dependencies into groups:

- **Runtime**: Core dependencies needed to run `check_models.py` (mlx, mlx-vlm, Pillow, etc.)
- **Extras**: Optional enhancements (psutil, tokenizers, mlx-lm, transformers)
- **Torch**: PyTorch stack for models that require it (torch, torchvision, torchaudio)
- **Dev**: Development tools (ruff, mypy, pytest)

Install specific groups as needed:

```bash
# Runtime only (default)
pip install -e .

# With extras
pip install -e ".[extras]"

# With PyTorch (needed for some models)
pip install -e ".[torch]"
make install-torch  # from root

# Everything
pip install -e ".[extras,torch,dev]"
make install-all  # from root
```

### Managing Dependencies

Dependencies are defined in `src/pyproject.toml` and mirrored in simple `requirements.txt` files.

```bash
# Update environment and reinstall project (recommended after pulling changes):
make update

# Check for outdated packages:
make check-outdated

# Security audit:
make audit

# Sync README dependency blocks:
make deps-sync
```

#### Advanced: Using update.sh for MLX Development

If you're actively developing MLX libraries locally, use `src/tools/update.sh` for more granular control:

```bash
cd src
bash tools/update.sh
```

**When to use `update.sh`**:

- You have local development builds of MLX, MLX-LM, or MLX-VLM
- You need to update specific dependency groups independently
- You want to skip certain groups (e.g., skip torch on Apple Silicon)
- You need environment diagnostics and validation

**When to use `make update`**:

- Standard workflow (using released packages from PyPI)
- Quick updates after pulling repository changes
- You don't need special handling for local MLX builds

**Features of update.sh**:

- Automatically detects and uses local MLX development builds
- Validates Python version (>= 3.13 required)
- Checks for virtual environment activation
- Offers per-group installation (runtime, extras, torch, dev)
- Verifies dependency sync across pyproject.toml, requirements*.txt, and update.sh itself

**Environment Variables**:

- `INSTALL_TORCH=1`: Install PyTorch (if needed for specific models)
- `MLX_METAL_JIT=OFF`: Disable Metal shader JIT compilation (default: `ON` for local MLX builds for better performance)

**Example usage**:

```bash
# Install with JIT disabled for debugging
MLX_METAL_JIT=OFF bash tools/update.sh

# Install with PyTorch support
INSTALL_TORCH=1 bash tools/update.sh
```

### Updating Dependencies

1. Edit `src/pyproject.toml` to update the dependency specification
2. Edit `src/requirements.txt` or `src/requirements-dev.txt` to match
3. Run `make deps-sync` to update README blocks
4. Test thoroughly
5. Commit both files

### Keeping Your Environment Up-to-Date

After pulling changes from the repository, update your environment with:

```bash
make update
```

This command will:

- Update pip in your conda/venv environment
- Reinstall the project with all dependencies (dev, extras, torch)
- Ensure you have the latest versions compatible with lock files

### Cleaning Build Artifacts

Build artifacts can accumulate during development, especially when working with local MLX builds:

```bash
# Clean project artifacts (recommended for regular cleanup)
make clean              # Remove __pycache__, build/, dist/, test caches

# Deep clean including type stubs
make clean-all          # Also removes typings/

# Clean local MLX development repositories
make clean-mlx          # Remove build artifacts from mlx, mlx-lm, mlx-vlm, mlx-data

# Preview what would be cleaned (dry run)
make clean-mlx-dry-run

# Or use the script directly
bash src/tools/clean_builds.sh
bash src/tools/clean_builds.sh --dry-run  # See what would be cleaned
```

**What gets cleaned:**

- Python build artifacts: `build/`, `dist/`, `*.egg-info/`, `.eggs/`
- Bytecode caches: `__pycache__/`, `*.pyc`, `*.pyo`
- Test caches: `.pytest_cache/`, `.mypy_cache/`
- Type stubs: `typings/` (only with `clean-all`)

**Note on Metal kernel caches:** Compiled Metal shaders are cached by macOS in system temp directories and persist across builds. These are automatically managed by the Metal framework and cleared on reboot. To manually clear (use with caution): `sudo rm -rf /tmp/com.apple.metal/*`

**When to clean:**

- Before rebuilding MLX with different compiler flags (e.g., changing `MLX_METAL_JIT`)
- When experiencing odd build or runtime errors
- To free up disk space
- After switching between different MLX versions or branches

**Note**: This does NOT update the lock files themselves. To upgrade dependencies to newer versions, use `make upgrade-deps` instead.

## Release Process

(To be documented when automated releases are set up)

## Getting Help

- Check [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for coding conventions
- Review existing issues on GitHub
- Ask questions in issue comments
- Run `make help` to see available commands

## Code of Conduct

- Be respectful and constructive
- Focus on what is best for the project
- Show empathy towards other contributors

---

Thank you for contributing! 🎉
