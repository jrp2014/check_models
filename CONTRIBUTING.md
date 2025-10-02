# Contributing to MLX VLM Check

Thank you for your interest in contributing to MLX VLM Check! This document guides you through the practical workflow of contributing: getting set up, making changes, and submitting them.

**Target audience**: New contributors, anyone submitting a pull request.

**Related documents**:

- [README.md](README.md) - What this project does and how to use it
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Detailed coding standards and technical conventions
- [vlm/README.md](vlm/README.md) - Package-specific usage and CLI documentation

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
- Python 3.12+
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
   conda create -n mlx-vlm python=3.12
   conda activate mlx-vlm
   ```

3. **Bootstrap development environment**:
   ```bash
   make -C vlm bootstrap-dev
   ```

   This will:
   - Install all dependencies (runtime + dev + extras)
   - Install git pre-commit hooks
   - Install pre-commit framework hooks (if available)
   - Validate the environment

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
   make -C vlm quality
   ```

4. **Run tests**:
   ```bash
   make -C vlm test
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
   make -C vlm format    # Format code
   make -C vlm lint      # Check linting
   make -C vlm lint-fix  # Auto-fix issues
   ```

2. **Mypy** (type checking):
   ```bash
   make -C vlm typecheck
   ```

3. **Tests**:
   ```bash
   make -C vlm test
   make -C vlm test-cov  # With coverage
   ```

4. **Combined quality check**:
   ```bash
   make -C vlm quality   # Runs format + lint + typecheck
   make -C vlm check     # Runs quality + tests
   make -C vlm ci        # Full CI pipeline
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
