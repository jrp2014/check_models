# Contributing to MLX VLM Check

Thank you for your interest in contributing to MLX VLM Check! This document guides you through the practical workflow of contributing: getting set up, making changes, and submitting them.

**Target audience**: New contributors, anyone submitting a pull request.

> [!NOTE]
> This project was restructured in October 2025. The old `vlm/` directory is now `src/`. All `make` commands now use the root `Makefile` instead of `make -C src` or `make -C vlm`.

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
   git clone https://github.com/jrp2014/check_models.git
   cd check_models
   ```

2. **Create and activate conda environment**:

   ```bash
   conda create -n mlx-vlm python=3.13
   conda activate mlx-vlm
   ```

3. **Install development dependencies**:

   ```bash
   make dev
   ```

   This will:
   - Install all Python dependencies (runtime + dev + extras + torch)
   - Install the package in editable mode

   **Optional dependencies**:

   - **PyTorch** (needed for some models): `make install-torch`
   - **Everything** (extras + torch + dev): `make install-all`
   - **Markdown linting only**: `make install-markdownlint` (requires Node.js/npm)

4. **Verify installation**:

   ```bash
   # Verify environment
   python -m tools.validate_env
   
   # Install git pre-commit hook (important!)
   python -m tools.install_precommit_hook
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

- **Pre-commit**:
  - Automatically formats Python files with ruff
  - Auto-fixes Markdown files with markdownlint (via npx)
  - Syncs README dependencies when pyproject.toml changes
  - Re-stages all fixed files automatically
- **Pre-push**: Runs full quality checks (format check, lint, type check, tests) before pushing

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

### Quick Verification

To verify that your environment and `mlx-vlm` are working correctly before running the full test suite, you can use the CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/nanoLLaVA --image /path/to/image.jpg
```

Alternatively, refer to the official smoke test script:

- [test_smoke.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/tests/test_smoke.py)

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

Dependencies are defined in `src/pyproject.toml` as the single source of truth.

```bash
# Update environment and reinstall project (recommended after pulling changes):
make update

# Check for outdated packages:
make -C src check-outdated

# Security audit:
make -C src audit

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

- `SKIP_TORCH=1`: Skip PyTorch installation (torch is included by default)
- `MLX_METAL_JIT=ON`: Enable Metal shader runtime compilation for smaller binaries (default: `OFF` for pre-built kernels)

**MLX_METAL_JIT Trade-offs**:

- `OFF` (default): Pre-built GPU kernels, larger binary (~100MB+ metallib), instant execution
- `ON`: Runtime compilation, smaller binary, cold-start delay (few hundred ms to few seconds on first use per kernel, then cached permanently)

**Example usage**:

```bash
# Standard build with pre-built kernels (default)
bash tools/update.sh

# Smaller binary with runtime compilation (cold start penalty)
MLX_METAL_JIT=ON bash tools/update.sh

# Install with PyTorch support
SKIP_TORCH=1 bash tools/update.sh
```

### Updating Dependencies

1. Edit `src/pyproject.toml` to update the dependency specification:
   - Runtime dependencies: `[project.dependencies]`
   - Dev dependencies: `[project.optional-dependencies.dev]`
   - Optional extras: `[project.optional-dependencies.extras]`
2. Run `make deps-sync` to update README blocks
3. Test thoroughly with `make test`
4. Commit the changes

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

According to the [official MLX documentation](https://ml-explore.github.io/mlx/build/html/install.html#binary-size-minimization), the Metal kernel cache persists across reboots and is system-managed for performance.

**When to clean:**

- Before rebuilding MLX with different compiler flags (e.g., changing `MLX_METAL_JIT`)
- When experiencing odd build or runtime errors
- To free up disk space
- After switching between different MLX versions or branches

**Note**: This does NOT update the lock files themselves. To upgrade dependencies to newer versions, edit `src/pyproject.toml` and run `make update`.

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
