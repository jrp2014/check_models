#!/usr/bin/env bash
# Simplified quality checks for pre-push hook (fast checks only)
set -euo pipefail

# Navigate to the script's directory
cd "$(dirname "$0")"

# Determine Python executable
if [[ "${CI:-false}" == "true" ]]; then
    PYTHON="python3"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
else
    PYTHON="python3"
fi

# Go to src root
cd ..

echo "=== Ruff Format (Check) ==="
$PYTHON -m ruff format --check .

echo "=== Ruff Lint ==="
$PYTHON -m ruff check .

echo "=== MyPy Type Check ==="
$PYTHON -m mypy check_models.py

echo ""
echo "✅ Fast quality checks passed!"
