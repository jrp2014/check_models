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

echo "=== Runtime Dependency Preflight ==="
$PYTHON - <<'PY'
from __future__ import annotations

import importlib.metadata
import sys

required = ("mlx", "mlx-vlm", "mlx-lm")
missing_packages: list[str] = []
versions: dict[str, str] = {}

for package in required:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        missing_packages.append(package)

if missing_packages:
    print(
        "❌ Required runtime dependencies unavailable: "
        + ", ".join(sorted(missing_packages))
        + ". Install/repair these packages before running quality checks.",
    )
    sys.exit(1)

import check_models

missing_runtime = {
    name: message
    for name, message in check_models.MISSING_DEPENDENCIES.items()
    if name in set(required)
}
if missing_runtime:
    print("❌ Required runtime dependencies unavailable:")
    for name in sorted(missing_runtime):
        print(f"   [{name}] {missing_runtime[name]}")
    sys.exit(1)

print(
    "✓ Runtime dependencies ready: "
    + ", ".join(f"{name}=={versions[name]}" for name in required),
)
PY

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
    # Find all shell scripts and check them.
    shell_scripts=()
    while IFS= read -r -d '' script_path; do
        shell_scripts+=("$script_path")
    done < <(find tools -name "*.sh" -type f -print0)

    if [ "${#shell_scripts[@]}" -gt 0 ]; then
        shellcheck "${shell_scripts[@]}"
    fi
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
