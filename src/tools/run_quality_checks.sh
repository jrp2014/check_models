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

set --

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common_quality.sh
source "$SCRIPT_DIR/common_quality.sh"

# Git exports GIT_DIR into hook processes (always for linked worktrees).
# With GIT_DIR set and GIT_WORK_TREE unset, git treats the *current
# directory* as the work tree, so once this script has cd'ed into src/ every
# `git diff` run by a tool reports the entire repository as deleted: Skylos's
# SKY-L021 regression pass then flags every validation call as "removed" and
# pyrefly's file enumeration doubles. Drop the variable so git rediscovers the
# repository from the working directory, exactly as it does outside a hook.
unset GIT_DIR

cd "$(quality_src_root)"
quality_setup_python

quality_require_python_tool ty "Install dev dependencies with: pip install -e .[dev]"
quality_require_python_tool pyrefly "Install dev dependencies with: pip install -e .[dev]"
quality_require_python_tool vulture "Install dev dependencies with: pip install -e .[dev]"
quality_require_python_tool skylos "Install dev dependencies with: pip install -e .[dev]"
if [ "$QUALITY_MODE" = "full" ]; then
    quality_require_command shellcheck "Install with: brew install shellcheck"
fi

# ---------------------------------------------------------------------------
# Lanes. Skylos (three scans, ~20 s) and pytest (~15 s) dominate the gate and
# are independent of everything else, so they run in the background while the
# quick static checks stream in the foreground. Each lane's output is captured
# and printed whole, in a fixed order, once it finishes, so the gate log reads
# exactly as it did when everything was sequential. A failing foreground step
# exits under `set -e` and the EXIT trap kills the lanes; a failing lane is
# reported after both lanes have printed, so no evidence is lost.
# ---------------------------------------------------------------------------
LANE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/check_models-quality-lanes.XXXXXX")"
SKYLOS_LOG="$LANE_DIR/skylos.log"
PYTEST_LOG="$LANE_DIR/pytest.log"
SKYLOS_PID=""
PYTEST_PID=""
cleanup_lanes() {
    local pid
    for pid in $SKYLOS_PID $PYTEST_PID; do
        kill "$pid" 2>/dev/null || true
    done
    rm -rf "$LANE_DIR"
}
trap cleanup_lanes EXIT

# Lanes are forked directly from this shell (never inside a `$(...)` capture):
# `wait` only knows its own children and returns 127 for anyone else's.
lane_wait() {
    local status=0
    wait "$1" || status=$?
    return "$status"
}

# A failing skylos gate can offer a "Continue anyway?" prompt and a deployment
# wizard that pushes commits. </dev/null alone does NOT prevent that: 4.33.x
# decides from stdout.isatty(), not stdin, so a terminal run gets the prompt and
# then aborts on the EOF. Only calls reaching run_gate_interaction can prompt —
# `skylos cicd gate` and a bare `--gate` scan with no `--format`. Both calls
# below are safe by construction: `--format concise --gate` routes to skylos's
# quiet gate path, and `-a` without `--gate` never gates at all. </dev/null
# stays as defence in depth (and the lane's stdout is a file, never a TTY). If
# you add a skylos call that can prompt, pipe its stdout and read the status
# from PIPESTATUS (see run_skylos_danger_advisory.sh) rather than reaching for
# --strict, which discards the configured [tool.skylos.gate] thresholds in
# favour of fail-on-any-finding.
lane_skylos() {
    echo "=== Skylos Quality Gate ==="
    TERM=dumb NO_COLOR=1 CLICOLOR=0 FORCE_COLOR=0 PY_COLORS=0 \
        quality_run_skylos . --quality --secrets --sca --gate --no-upload --format concise </dev/null

    echo "=== Skylos Audit Gate ==="
    TERM=dumb NO_COLOR=1 CLICOLOR=0 FORCE_COLOR=0 PY_COLORS=0 \
        quality_run_skylos . -a </dev/null

    if [ "$QUALITY_MODE" = "full" ]; then
        echo "=== Skylos Danger Gate ==="
        bash "$SCRIPT_DIR/run_skylos_danger_advisory.sh" --full --gate
    fi
}

lane_pytest() {
    if [ "$QUALITY_MODE" = "fast" ]; then
        echo "=== Pytest (fast set) ==="
        "$QUALITY_PYTHON" -m pytest -q -n auto --maxprocesses=8 -m "not slow and not e2e"
    else
        echo "=== Pytest ==="
        "$QUALITY_PYTHON" -m pytest -v -n auto --maxprocesses=8
    fi
}

run_markdownlint_step() {
    # Markdown linting runs from the repo root.
    (
        cd "$(quality_repo_root)"
        echo "=== Markdown Lint ==="
        quality_run_markdownlint \
            --config .markdownlint.jsonc \
            "**/*.md" \
            "!src/node_modules/**" \
            "!**/node_modules/**" \
            "!**/.worktrees/**" \
            "!**/.claude/**"
    )
}

(
    set -euo pipefail
    lane_skylos
) > "$SKYLOS_LOG" 2>&1 &
SKYLOS_PID="$!"
(
    set -euo pipefail
    lane_pytest
) > "$PYTEST_LOG" 2>&1 &
PYTEST_PID="$!"

echo "=== Workflow YAML Validation ==="
# Glob rather than enumerate so a new workflow file cannot silently skip
# validation.
workflow_yaml_files=()
while IFS= read -r -d '' workflow_file; do
    workflow_yaml_files+=("$workflow_file")
done < <(find "$(quality_repo_root)/.github/workflows" \
    \( -name "*.yml" -o -name "*.yaml" \) -type f -print0 | sort -z)
if [ "${#workflow_yaml_files[@]}" -eq 0 ]; then
    echo "❌ No workflow files found under .github/workflows" >&2
    exit 1
fi
quality_validate_yaml_files \
    "${workflow_yaml_files[@]}" \
    "$(quality_repo_root)/.pre-commit-config.yaml"

echo "=== Dependency Sync (Check) ==="
"$QUALITY_PYTHON" -m tools.update_readme_deps --check

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
quality_run_pyrefly_check "$@"

echo "=== Vulture Dead Code Check ==="
quality_run_python_tool vulture

if [ "$QUALITY_MODE" = "full" ]; then
    echo "=== ShellCheck ==="
    shell_scripts=()
    while IFS= read -r -d '' script_path; do
        shell_scripts+=("$script_path")
    done < <(find tools -name "*.sh" -type f -print0)

    if [ "${#shell_scripts[@]}" -gt 0 ]; then
        shellcheck -x "${shell_scripts[@]}"
    fi
fi

run_markdownlint_step

skylos_status=0
lane_wait "$SKYLOS_PID" || skylos_status=$?
cat "$SKYLOS_LOG"
if [ "$skylos_status" -ne 0 ]; then
    echo "❌ Skylos lane failed (exit $skylos_status)" >&2
fi
pytest_status=0
lane_wait "$PYTEST_PID" || pytest_status=$?
cat "$PYTEST_LOG"
if [ "$pytest_status" -ne 0 ]; then
    echo "❌ Pytest lane failed (exit $pytest_status)" >&2
fi
if [ "$skylos_status" -ne 0 ] || [ "$pytest_status" -ne 0 ]; then
    exit 1
fi

echo ""
if [ "$QUALITY_MODE" = "fast" ]; then
    echo "✅ Fast quality checks passed!"
else
    echo "✅ All quality checks passed!"
fi
