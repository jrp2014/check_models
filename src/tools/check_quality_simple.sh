#!/bin/bash
# Simple quality checker - no complex orchestration needed
set -e

cd "$(dirname "$0")/.."

echo "=== Ruff Format Check ==="
ruff format --check .

echo "=== Ruff Lint ==="
ruff check .

echo "=== MyPy Type Check ==="
mypy check_models.py

echo "=== Pytest ==="
pytest -v

echo ""
echo "✅ All quality checks passed!"
