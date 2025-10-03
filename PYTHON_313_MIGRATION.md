# Python 3.13 Migration Guide

## Overview

The project has been upgraded to require Python 3.13 (previously 3.12). This document guides you through updating your development environment.

## Why Upgrade?

- **Performance**: Python 3.13 includes significant performance improvements
- **New Features**: Access to the latest Python language features
- **Security**: Latest security patches and updates
- **Compatibility**: All project dependencies are compatible with 3.13

## Migration Steps

### Option 1: Recreate Conda Environment (Recommended)

This is the cleanest approach:

```bash
# 1. Deactivate current environment
conda deactivate

# 2. Remove old environment
conda remove -n mlx-vlm --all

# 3. Create new environment with Python 3.13
conda create -n mlx-vlm python=3.13 -y

# 4. Activate new environment
conda activate mlx-vlm

# 5. Bootstrap development environment
cd /path/to/scripts
make -C vlm bootstrap-dev

# 6. Verify installation
python --version  # Should show Python 3.13.x
python -m vlm.tools.validate_env
```

### Option 2: Update Existing Environment

If you want to update in place:

```bash
# 1. Activate your environment
conda activate mlx-vlm

# 2. Update Python version
conda install python=3.13 -y

# 3. Reinstall dependencies (in case of binary compatibility issues)
pip install --force-reinstall -e "vlm/[dev,extras]"

# 4. Verify installation
python --version
python -m vlm.tools.validate_env
```

### Option 3: Automated Setup Script

Use the provided setup script:

```bash
cd /path/to/scripts/vlm/tools
./setup_conda_env.sh

# Follow the prompts - the script will:
# - Detect existing mlx-vlm environment
# - Offer to recreate it with Python 3.13
# - Install all dependencies
# - Verify the installation
```

## Verification

After migration, verify everything works:

```bash
# Check Python version
python --version  # Should show 3.13.x

# Validate environment
python -m vlm.tools.validate_env

# Run quality checks
make -C vlm quality

# Run tests
make -C vlm test

# Try running the tool
make check_models ARGS="--help"
```

## What Changed

### Configuration Files

- **pyproject.toml**: `requires-python = ">=3.13"`
- **CI/CD workflows**: GitHub Actions use Python 3.13
- **Type checkers**: mypy and pylance configured for 3.13
- **Documentation**: All examples updated

### Code Compatibility

- ✅ All existing code is compatible with Python 3.13
- ✅ Type hints using `|` syntax (PEP 604) work perfectly
- ✅ All dependencies tested and compatible
- ✅ No code changes required for the upgrade

## Troubleshooting

### Issue: "conda: command not found"

**Solution**: Install miniconda or anaconda first:

```bash
# Install miniconda (recommended)
brew install --cask miniconda

# Initialize conda for your shell
conda init zsh  # or bash, fish, etc.

# Restart your shell
exec $SHELL
```

### Issue: "Python 3.13 not available in conda"

**Solution**: Update conda:

```bash
conda update -n base -c defaults conda
```

If still not available, wait a few days for conda-forge to add Python 3.13 packages, or use `pip` in a virtual environment instead.

### Issue: "Binary incompatibility" errors

**Solution**: Reinstall packages with binary components:

```bash
pip uninstall mlx mlx-vlm Pillow -y
pip install --no-cache-dir mlx mlx-vlm Pillow
```

### Issue: "Import errors after upgrade"

**Solution**: Clear Python caches and reinstall:

```bash
# Clear Python caches
find vlm -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find vlm -type f -name "*.pyc" -delete

# Reinstall in development mode
pip install -e "vlm/[dev,extras]"
```

### Issue: "torch/torchvision/torchaudio import errors"

**Solution**: Some models require PyTorch. Install the torch extras:

```bash
# From root directory
make install-torch

# Or directly
cd vlm
pip install -e ".[torch]"

# Or install everything
make install-all  # from root
pip install -e ".[extras,torch,dev]"  # from vlm/
```

### Issue: "Pre-commit hooks failing"

**Solution**: Reinstall pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

## CI/CD Updates

If you maintain a fork or have custom CI:

- Update GitHub Actions workflows to use `python-version: '3.13'`
- Update any Docker images or containers to Python 3.13
- Update local testing scripts to use Python 3.13

## Rollback (If Needed)

If you encounter issues and need to roll back:

```bash
# Checkout previous commit
git checkout HEAD~1

# Recreate environment with Python 3.12
conda remove -n mlx-vlm --all
conda create -n mlx-vlm python=3.12 -y
conda activate mlx-vlm
make -C vlm bootstrap-dev
```

Then report the issue on GitHub.

## Questions or Issues?

- Check [CONTRIBUTING.md](CONTRIBUTING.md) for general setup help
- Review [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for technical details
- Open an issue on GitHub if you encounter migration problems

---

**Migration Status**: ✅ Complete - All files updated, tests passing

**Tested On**: 
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.13.0

**Last Updated**: October 3, 2025
