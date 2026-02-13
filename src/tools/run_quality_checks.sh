#!/usr/bin/env bash
# Unified quality checker for local and CI environments
set -euo pipefail

CONDA_ENV="mlx-vlm"

# --- Environment Setup ---
# Navigate to the script's directory to ensure consistent paths
cd "$(dirname "$0")"

# Activate conda environment if available and not in CI
if [[ "${CI:-false}" == "false" ]] && command -v conda &> /dev/null; then
    # Initialize conda for bash
    # shellcheck disable=SC1091
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090,SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    elif [ -f "/opt/homebrew/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/opt/homebrew/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    if ! conda activate $CONDA_ENV 2>/dev/null; then
        echo "⚠️  Warning: Could not activate conda environment '$CONDA_ENV'"
        echo "   Run: conda activate $CONDA_ENV"
    fi
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

echo "=== Dependency Sync ==="
$PYTHON -m tools.update_readme_deps


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
# Debug: Print MLX version in CI to help diagnose stub issues
if [[ "${CI:-false}" == "true" ]]; then
    echo "Debug: Checking installed MLX version..."
    $PYTHON -m pip show mlx || echo "MLX not found via pip"
    
    # WORKAROUND: MLX 0.29.4 has a syntax error in __init__.pyi that crashes mypy.
    # We delete the broken stub file in CI so mypy ignores it (we have ignore_missing_imports=True).
    # TODO: Remove this workaround once MLX >= 0.29.6 is available in CI.
    #MLX_LOC=$($PYTHON -m pip show mlx | grep Location | cut -d: -f2 | xargs)
    #if [[ -n "$MLX_LOC" && -f "$MLX_LOC/mlx/core/__init__.pyi" ]]; then
    #    echo "⚠️  Workaround: Removing broken mlx/core/__init__.pyi to fix mypy..."
    #    rm "$MLX_LOC/mlx/core/__init__.pyi"
    #fi
fi

# Ensure mypy uses the correct Python environment by setting MYPYPATH
# In CI, explicitly use the python from PATH to avoid system Python
if [[ "${CI:-false}" == "true" ]]; then
    # Use 'python' instead of 'python3' to match setup-python action
    python -m mypy check_models.py
else
    $PYTHON -m mypy check_models.py
fi

echo "=== Ty Type Check ==="
if ! command -v ty &> /dev/null; then
    echo "⚠️  'ty' not found. Installing..."
    $PYTHON -m pip install ty
fi
ty check check_models.py

echo "=== Pyrefly Type Check ==="
if ! command -v pyrefly &> /dev/null; then
    echo "⚠️  'pyrefly' not found. Installing..."
    $PYTHON -m pip install pyrefly
fi
pyrefly check check_models.py

echo "=== Pytest ==="
$PYTHON -m pytest -v

echo "=== ShellCheck ==="
if ! command -v shellcheck &> /dev/null; then
    if command -v brew &> /dev/null; then
        echo "⚠️  'shellcheck' not found. Installing via Homebrew..."
        brew install shellcheck
    else
        echo "⚠️  Skipped (shellcheck not found and brew not available)"
        echo "   Install with: brew install shellcheck"
    fi
fi

if command -v shellcheck &> /dev/null; then
    # Find all shell scripts and check them
    # shellcheck disable=SC2046
    shellcheck $(find tools -name "*.sh" -type f)
fi

# Markdown linting runs from the repo root
cd ..

echo "=== Markdown Lint ==="
if [ -x "src/node_modules/.bin/markdownlint-cli2" ]; then
    src/node_modules/.bin/markdownlint-cli2 --config .markdownlint.jsonc "**/*.md" "!src/output/**" "!src/node_modules/**" "!**/node_modules/**"
elif command -v markdownlint-cli2 &> /dev/null; then
    markdownlint-cli2 --config .markdownlint.jsonc "**/*.md" "!src/output/**" "!src/node_modules/**" "!**/node_modules/**"
elif command -v npx &> /dev/null; then
    npx --no-install markdownlint-cli2 --config .markdownlint.jsonc "**/*.md" "!src/output/**" "!src/node_modules/**" "!**/node_modules/**"
else
    echo "⚠️  Skipped (markdownlint-cli2 or npx not found)"
fi

echo ""
echo "✅ All quality checks passed!"
