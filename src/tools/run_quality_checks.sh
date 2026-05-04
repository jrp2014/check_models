#!/usr/bin/env bash
# Deterministic quality checker for local runs, hooks, and CI.
set -euo pipefail

QUALITY_MODE="full"
if [ "$#" -gt 1 ]; then
    echo "Usage: bash tools/run_quality_checks.sh [--fast|--full]" >&2
    exit 2
fi
if [ "$#" -eq 1 ]; then
    case "$1" in
        --fast|fast)
            QUALITY_MODE="fast"
            ;;
        --full|full)
            QUALITY_MODE="full"
            ;;
        *)
            echo "Usage: bash tools/run_quality_checks.sh [--fast|--full]" >&2
            exit 2
            ;;
    esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_quality.sh"

cd "$(quality_src_root)"
quality_setup_python

quality_require_python_tool ty "Install dev dependencies with: pip install -e .[dev]"
quality_require_python_tool pyrefly "Install dev dependencies with: pip install -e .[dev]"
quality_require_python_tool vulture "Install dev dependencies with: pip install -e .[dev]"
if [ "$QUALITY_MODE" = "full" ]; then
    quality_require_command shellcheck "Install with: brew install shellcheck"
fi

echo "=== Workflow YAML Validation ==="
quality_validate_yaml_files \
    "$(quality_repo_root)/.github/workflows/quality.yml" \
    "$(quality_repo_root)/.github/workflows/dependency-sync.yml" \
    "$(quality_repo_root)/.pre-commit-config.yaml"

echo "=== Dependency Sync (Check) ==="
"$QUALITY_PYTHON" -m tools.update_readme_deps --check

if [ "$QUALITY_MODE" = "full" ]; then
    mkdir -p ../typings

    echo "=== Type Stub Refresh (Best Effort) ==="
    stub_refresh_failed=0
    if ! "$QUALITY_PYTHON" -m tools.generate_stubs --skip-if-fresh \
        mlx_lm mlx_vlm transformers tokenizers; then
        echo "⚠️  Stub refresh warning: could not regenerate local third-party stubs; continuing"
        stub_refresh_failed=1
    fi

    echo "=== Type Stub Contract Check ==="
    if ! "$QUALITY_PYTHON" -m tools.generate_stubs --check --refresh-manifest-on-check \
        mlx_lm mlx_vlm transformers tokenizers; then
        if [ "$stub_refresh_failed" -eq 1 ]; then
            echo "❌ Stub refresh and integrity checks failed"
        fi
        exit 1
    fi
fi

if [ "$QUALITY_MODE" = "fast" ]; then
    echo "=== Ruff Format (Check) ==="
else
    echo "=== Ruff Format ==="
fi
"$QUALITY_PYTHON" -m ruff format --check .

echo "=== Ruff Lint ==="
"$QUALITY_PYTHON" -m ruff check .

echo "=== MyPy Type Check ==="
"$QUALITY_PYTHON" -m mypy check_models.py

echo "=== Suppression Audit ==="
"$QUALITY_PYTHON" -m tools.check_suppressions

echo "=== Ty Type Check ==="
quality_run_ty_check check_models.py

echo "=== Pyrefly Type Check ==="
quality_run_pyrefly_check

echo "=== Vulture Dead Code Check ==="
quality_run_python_tool vulture

if [ "$QUALITY_MODE" = "fast" ]; then
    echo "=== Pytest (fast set) ==="
    "$QUALITY_PYTHON" -m pytest -q -m "not slow and not e2e"

    # Markdown linting runs from the repo root
    cd "$(quality_repo_root)"

    echo "=== Markdown Lint ==="
    quality_run_markdownlint \
        --config .markdownlint.jsonc \
        "**/*.md" \
        "!src/node_modules/**" \
        "!**/node_modules/**"

    echo ""
    echo "✅ Fast quality checks passed!"
    exit 0
fi

echo "=== Pytest ==="
"$QUALITY_PYTHON" -m pytest -v

echo "=== ShellCheck ==="
shell_scripts=()
while IFS= read -r -d '' script_path; do
    shell_scripts+=("$script_path")
done < <(find tools -name "*.sh" -type f -print0)

if [ "${#shell_scripts[@]}" -gt 0 ]; then
    shellcheck -x "${shell_scripts[@]}"
fi

# Markdown linting runs from the repo root
cd "$(quality_repo_root)"

echo "=== Markdown Lint ==="
quality_run_markdownlint \
    --config .markdownlint.jsonc \
    "**/*.md" \
    "!src/node_modules/**" \
    "!**/node_modules/**"

echo ""
echo "✅ All quality checks passed!"
