#!/usr/bin/env bash
# Fast static checks plus non-slow tests for pre-push hooks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_quality.sh"

cd "$(quality_src_root)"
quality_setup_python

quality_require_command ty "Install dev dependencies with: pip install -e .[dev]"
quality_require_command pyrefly "Install dev dependencies with: pip install -e .[dev]"

echo "=== Workflow YAML Validation ==="
quality_validate_yaml_files \
    "$(quality_repo_root)/.github/workflows/quality.yml" \
    "$(quality_repo_root)/.github/workflows/dependency-sync.yml" \
    "$(quality_repo_root)/.pre-commit-config.yaml"

echo "=== Dependency Sync (Check) ==="
"$QUALITY_PYTHON" -m tools.update_readme_deps --check

echo "=== Ruff Format (Check) ==="
"$QUALITY_PYTHON" -m ruff format --check .

echo "=== Ruff Lint ==="
"$QUALITY_PYTHON" -m ruff check .

echo "=== MyPy Type Check ==="
"$QUALITY_PYTHON" -m mypy check_models.py

echo "=== Suppression Audit ==="
"$QUALITY_PYTHON" -m tools.check_suppressions

echo "=== Ty Type Check ==="
ty check check_models.py

echo "=== Pyrefly Type Check ==="
pyrefly check check_models.py

echo "=== Pytest (fast set) ==="
"$QUALITY_PYTHON" -m pytest -q -m "not slow and not e2e"

echo ""
echo "✅ Fast quality checks passed!"
