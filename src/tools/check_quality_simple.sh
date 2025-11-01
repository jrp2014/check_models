#!/bin/bash
# Simple quality checker - no complex orchestration needed
set -e

cd "$(dirname "$0")/.."

echo "=== Ruff Format Check ==="
python -m ruff format --check .

echo "=== Ruff Lint ==="
python -m ruff check .

echo "=== MyPy Type Check ==="
python -m mypy check_models.py

echo "=== Pytest ==="
python -m pytest -v

echo ""
echo "✅ All quality checks passed!"
