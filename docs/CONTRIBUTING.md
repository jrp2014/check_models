# Contributing to MLX VLM Check

Thank you for your interest in contributing to MLX VLM Check! This document guides you through the practical workflow of contributing: getting set up, making changes, and submitting them.

**Target audience**: New contributors, anyone submitting a pull request.

> **Note**: This project was restructured in October 2025. The old `vlm/` directory is now `src/`. Most commands now use the root `Makefile` instead of `make -C src`.

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
   make dev
   ```

   This will:
   - Install all dependencies (runtime + dev + extras)
   - Install the package in editable mode
   - Set up git hooks automatically via pre-commit (if configured)

   **Optional dependencies**:

   - **PyTorch** (needed for some models): `make install-torch`
   - **Everything** (extras + torch + dev): `make install-all`

4. **Verify installation**:

   ```bash
   python -m vlm.tools.validate_env
   ```

### Manual Environment Validation

If bootstrap didn't work or you want to check your environment:

```bash
# Check environment health
python -m vlm.tools.validate_env

# Auto-fix common issues
python -m vlm.tools.validate_env --fix
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
   make quality     # Runs format + lint + typecheck
   ```

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
make -C vlm test

# Run with coverage
make -C vlm test-cov

# Run specific test file
cd vlm && pytest tests/test_specific.py

# Run with verbose output
cd vlm && pytest -v
```

### Writing Tests

- Place tests in `vlm/tests/`
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Include both positive and negative test cases
- Aim for high coverage of critical paths

## Submitting Changes

### Pull Request Process

1. **Ensure all checks pass**:

   ```bash
   make -C vlm ci
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

### Using pip-tools

The project uses pip-tools for reproducible builds:

```bash
# Add a new dependency
# 1. Edit vlm/requirements.in or vlm/requirements-dev.in
# 2. Regenerate lock files:
make -C vlm lock-deps

# Upgrade all dependencies to latest:
make -C vlm upgrade-deps

# Sync your environment with lock files:
make -C vlm sync-deps

# Check for outdated packages:
make -C vlm check-outdated

# Security audit:
make -C vlm audit
```

### Updating Dependencies

1. Edit `vlm/requirements.in` or `vlm/requirements-dev.in`
2. Run `make -C vlm lock-deps`
3. Test thoroughly
4. Commit both `.in` and `.txt` files

## Release Process

(To be documented when automated releases are set up)

## Getting Help

- Check [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for coding conventions
- Review existing issues on GitHub
- Ask questions in issue comments
- Run `make -C vlm help` to see available commands

## Code of Conduct

- Be respectful and constructive
- Focus on what is best for the project
- Show empathy towards other contributors

---

Thank you for contributing! 🎉
