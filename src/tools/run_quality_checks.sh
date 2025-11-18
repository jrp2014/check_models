#!/usr/bin/env bash
# Unified quality checker for local and CI environments
set -euo pipefail

# --- Environment Setup ---
# Navigate to the script's directory to ensure consistent paths
cd "$(dirname "$0")"

# Activate conda environment if available and not in CI
if [[ "${CI:-false}" == "false" ]] && command -v conda &> /dev/null; then
    # Initialize conda for bash
    # shellcheck disable=SC1091
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    fi
    conda activate mlx-vlm 2>/dev/null || true
fi

# Determine Python executable
# In CI, use the python from PATH (set up by GitHub Actions)
# Locally, prefer conda environment python if available
if [[ "${CI:-false}" == "true" ]]; then
    PYTHON="python3"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
else
    PYTHON="python3"
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
# Ensure mypy uses the correct Python environment by setting MYPYPATH
# In CI, explicitly use the python from PATH to avoid system Python
if [[ "${CI:-false}" == "true" ]]; then
    # Use 'python' instead of 'python3' to match setup-python action
    python -m mypy check_models.py
else
    $PYTHON -m mypy check_models.py
fi

echo "=== Pytest ==="
$PYTHON -m pytest -v

echo "=== ShellCheck ==="
if command -v shellcheck &> /dev/null; then
    # Find all shell scripts and check them
    # shellcheck disable=SC2046
    shellcheck $(find tools -name "*.sh" -type f)
else
    echo "⚠️  Skipped (shellcheck not found)"
    echo "   Install with: brew install shellcheck"
fi

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
