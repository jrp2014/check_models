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

# The mode is parsed. Clear the positional parameters so a helper called with
# "$@" (as ShellCheck SC2119 asks of helpers that read arguments) gets none.
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
            "!**/.claude/**" \
            "!.agents/skills/hf-cli/references/commands.md"
    )
}

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
quality_check_file_syntax -- \
    "${workflow_yaml_files[@]}" \
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
quality_run_ty_check check_models.py

echo "=== Pyrefly Type Check ==="
quality_run_pyrefly_check "$@"

echo "=== Vulture Dead Code Check ==="
quality_run_python_tool vulture

# A failing skylos gate can offer a "Continue anyway?" prompt and a deployment
# wizard that pushes commits. </dev/null alone does NOT prevent that: 4.33.x
# decides from stdout.isatty(), not stdin, so a terminal run gets the prompt and
# then aborts on the EOF. Only calls reaching run_gate_interaction can prompt —
# `skylos cicd gate` and a bare `--gate` scan with no `--format`. The call
# below is safe by construction: `--format concise --gate` routes to skylos's
# quiet gate path. </dev/null stays as defence in depth. If you add a skylos
# call that can prompt, pipe its
# stdout and read the status from PIPESTATUS (see run_skylos_danger_advisory.sh)
# rather than reaching for --strict, which discards the configured
# [tool.skylos.gate] thresholds in favour of fail-on-any-finding.
#
# Skylos runs before pytest, never concurrently with it: its dead-code grep
# verification aborts (SKY-ANALYSIS-INCOMPLETE) when files appear or vanish
# under it, and keeping the tree provably quiet during an overlap cost more
# machinery than the ~15 s it saved. With nothing scanning the tree during
# pytest, its ordinary gitignored caches (.pytest_cache, __pycache__) stay in
# place, so `pytest --lf` sees the same last-failed set as the gate.
# One gated scan carries every check the former ungated `-a` audit pass
# added (ai-defects); the audit pass only printed a grade card over the same
# files, at the cost of a second full analysis (~9 s). Danger stays in its
# own wrapper, which post-filters the JSON report.
#
# The danger scan runs alongside the quality scan. Each Skylos run keeps about
# one core busy (it analyses check_models.py in one worker), so on 4 cores the
# pair took ~125 s side by side against ~220 s one after the other (Skylos
# 5.7.0, same findings and reports either way). Neither writes into the tree the other scans while it runs: the
# quality scan writes only Skylos's own src/.skylos/cache, the danger scan
# writes its report to the repository root's .skylos/ when it finishes. The
# danger output is held in a file and printed whole after the quality output,
# and either failure fails the gate. The suppression audit stays before them:
# it writes temporary copies of suppressed files beside the originals.
# The dependency report (below) must not be in the tree either scan reads. A
# whole-tree --sca report was ~15 MB, over the 8 MB at which the secret scan
# stops (SKY-ANALYSIS-INCOMPLETE); the scan below writes ~16 KB, but a stray
# report is still not source. It is written only after both scans finish and
# removed after its own step; this also clears one an interrupted run left
# behind.
sca_report_path=".skylos/skylos-dependency-scan.json"
rm -f "$sca_report_path"
if [ "$QUALITY_MODE" = "full" ]; then
    danger_heading="Skylos Danger Gate"
    danger_args=(--full --gate)
else
    # Fast mode keeps the danger scan the former ungated `-a` audit pass
    # provided, advisory only (findings are printed, never blocking here).
    danger_heading="Skylos Danger Advisory (fast, non-blocking)"
    danger_args=(--full)
fi
danger_log="$(mktemp "${TMPDIR:-/tmp}/skylos-danger.XXXXXX")"
bash "$SCRIPT_DIR/run_skylos_danger_advisory.sh" "${danger_args[@]}" </dev/null >"$danger_log" 2>&1 &
danger_pid=$!
# An interrupted or failed run must not leave the danger scan running.
trap 'kill "$danger_pid" 2>/dev/null || true; rm -f "$danger_log"' EXIT
echo "=== Skylos Quality Gate ==="
quality_gate_status=0
quality_run_skylos_quality_gate || quality_gate_status=$?
danger_status=0
wait "$danger_pid" || danger_status=$?
trap - EXIT
echo "=== $danger_heading ==="
cat "$danger_log"
rm -f "$danger_log"
if [ "$quality_gate_status" -ne 0 ]; then
    exit "$quality_gate_status"
fi
if [ "$danger_status" -ne 0 ]; then
    exit "$danger_status"
fi

# The dependency scan runs on its own because Skylos's ignore list does not
# reach dependency advisories: tools.check_dependency_advisories fails on any
# advisory outside its short accepted list, and on an unfinished scan.
echo "=== Skylos Dependency Scan ==="
# Skylos only writes reports inside the project ("could not safely write
# output file" for a temp path), so the report goes in .skylos/ and is
# removed as soon as it has been checked.
# The target is pyproject.toml, not `.`: Skylos looks for dependency
# manifests by walking the project root (src/, the target's directory)
# whatever the target, so the report is the same, but a `.` target also
# repeats the source and dead-code analysis the quality gate has just done
# (~1 s instead of ~35 s on 4 cores, Skylos 5.6.0). A test checks the walk
# still finds a manifest outside the target.
mkdir -p .skylos
set +e
TERM=dumb NO_COLOR=1 CLICOLOR=0 FORCE_COLOR=0 PY_COLORS=0 \
    quality_run_skylos pyproject.toml --sca --json -o "$sca_report_path" --no-upload \
    </dev/null >/dev/null
sca_exit_code=$?
set -e
if [ ! -f "$sca_report_path" ]; then
    echo "❌ Skylos dependency scan did not produce $sca_report_path (exit $sca_exit_code)." >&2
    exit "$((sca_exit_code == 0 ? 1 : sca_exit_code))"
fi
sca_check_status=0
"$QUALITY_PYTHON" -m tools.check_dependency_advisories "$sca_report_path" || sca_check_status=$?
rm -f "$sca_report_path"
if [ "$sca_check_status" -ne 0 ]; then
    exit "$sca_check_status"
fi

if [ "$QUALITY_MODE" = "fast" ]; then
    echo "=== Pytest (fast set) ==="
    "$QUALITY_PYTHON" -m pytest -q -n auto --maxprocesses=10 -m "not slow and not e2e"

    run_markdownlint_step

    echo ""
    quality_print_skylos_advice
    echo "✅ Fast quality checks passed!"
    exit 0
fi

echo "=== Pytest ==="
"$QUALITY_PYTHON" -m pytest -v -n auto --maxprocesses=10

echo "=== ShellCheck ==="
shell_scripts=()
while IFS= read -r -d '' script_path; do
    shell_scripts+=("$script_path")
done < <(find tools -name "*.sh" -type f -print0)

if [ "${#shell_scripts[@]}" -gt 0 ]; then
    shellcheck -x "${shell_scripts[@]}"
fi

run_markdownlint_step

echo ""
quality_print_skylos_advice
echo "✅ All quality checks passed!"
