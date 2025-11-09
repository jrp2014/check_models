#!/bin/bash
# Unified quality checker for local and CI environments
set -euo pipefail

# --- Environment Setup ---
# Navigate to the script's directory to ensure consistent paths
cd "$(dirname "$0")"

# Activate conda environment if available and not in CI
if [[ "${CI:-false}" == "false" ]] && command -v conda &> /dev/null; then
    # Initialize conda for bash
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    fi
    conda activate mlx-vlm 2>/dev/null || true
fi

# Determine Python executable
PYTHON="python"
if [[ "${CI:-false}" == "false" ]] && [ -n "$CONDA_PREFIX" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
fi

# --- Quality Checks ---
# All paths are relative to the `src` directory
cd ..

echo "=== Ruff Format ==="
# In CI mode, only check; in local mode, fix automatically
if [[ "${CI:-false}" == "true" ]]; then
    $PYTHON -m ruff format --check .
else
    $PYTHON -m ruff format .
fi

echo "=== Ruff Lint ==="
# In CI mode, only check; in local mode, fix automatically
if [[ "${CI:-false}" == "true" ]]; then
    $PYTHON -m ruff check .
else
    $PYTHON -m ruff check --fix .
fi

echo "=== MyPy Type Check ==="
echo "⚠️  Skipped (external MLX stub syntax error - not fixable from this project)"
# $PYTHON -m mypy check_models.py

echo "=== Pytest ==="
$PYTHON -m pytest -v

# Markdown linting runs from the repo root
cd ..

echo "=== Markdown Lint ==="
if command -v markdownlint-cli2 &> /dev/null; then
    markdownlint-cli2 --config .markdownlint.jsonc "**/*.md"
elif command -v npx &> /dev/null; then
    npx markdownlint-cli2 --config .markdownlint.jsonc "**/*.md"
else
    echo "⚠️  Skipped (markdownlint-cli2 or npx not found)"
fi

echo ""
echo "✅ All quality checks passed!"
