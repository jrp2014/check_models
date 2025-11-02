#!/bin/bash
# Simple quality checker - no complex orchestration needed
set -e

cd "$(dirname "$0")/.."

# Use the Python from the current conda environment
# CONDA_PREFIX points to the active environment's directory
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
else
    PYTHON="python"
fi

echo "=== Ruff Format Check ==="
$PYTHON -m ruff format --check .

echo "=== Ruff Lint ==="
$PYTHON -m ruff check .

echo "=== MyPy Type Check ==="
# Note: Temporarily disabled due to syntax errors in external MLX stubs (mlx/core/__init__.pyi:3124)
# The error is in the MLX library, not our code. Re-enable when MLX stubs are fixed.
# $PYTHON -m mypy check_models.py
echo "⚠️  Skipped (external MLX stub syntax error - not fixable from this project)"

echo "=== Pytest ==="
$PYTHON -m pytest -v

echo ""
echo "✅ All quality checks passed!"
