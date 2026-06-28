#!/usr/bin/env bash
# Advisory Skylos danger scan for local triage and CI workflow hardening.
set -euo pipefail

MODE="auto"
DIFF_BASE=""
WRITE_LLM_REPORT=0

usage() {
    cat >&2 <<'EOF'
Usage: bash tools/run_skylos_danger_advisory.sh [--full] [--diff-base REF] [--llm]

Runs Skylos --danger separately from the blocking quality gate.
Default behavior:
  - local runs: full-repo advisory scan
  - GitHub Actions PR runs: diff-aware advisory scan using origin/$GITHUB_BASE_REF

Options:
  --full           Run a full-repo advisory scan.
  --diff-base REF  Limit findings to changed lines since REF.
  --llm            Also write an LLM-optimized report for agent triage.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --full)
            MODE="full"
            shift
            ;;
        --diff-base)
            if [ "$#" -lt 2 ]; then
                usage
                exit 2
            fi
            MODE="diff"
            DIFF_BASE="$2"
            shift 2
            ;;
        --llm)
            WRITE_LLM_REPORT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common_quality.sh
source "$SCRIPT_DIR/common_quality.sh"

cd "$(quality_repo_root)"
quality_setup_python
quality_require_python_tool skylos "Install dev dependencies with: pip install -e .[dev]"

if [ "$MODE" = "auto" ] && [ -n "${GITHUB_BASE_REF:-}" ]; then
    candidate_ref="origin/$GITHUB_BASE_REF"
    if git rev-parse --verify "$candidate_ref" >/dev/null 2>&1; then
        MODE="diff"
        DIFF_BASE="$candidate_ref"
    else
        MODE="full"
    fi
fi

if [ "$MODE" = "auto" ]; then
    MODE="full"
fi

if [ "$MODE" = "diff" ] && [ -z "$DIFF_BASE" ]; then
    echo "❌ --diff-base requires a git ref." >&2
    exit 2
fi

mkdir -p .skylos

report_path=".skylos/skylos-danger-advisory.json"
llm_report_path=".skylos/skylos-danger-advisory.llm.txt"
scan_args=(. --danger --json -o "$report_path" --no-upload)
llm_args=(. --danger --llm -o "$llm_report_path" --no-upload)

if [ "$MODE" = "diff" ]; then
    scan_args+=(--diff "$DIFF_BASE")
    llm_args+=(--diff "$DIFF_BASE")
fi

echo "=== Skylos Danger Advisory ==="
if [ "$MODE" = "diff" ]; then
    echo "Mode: changed lines since $DIFF_BASE"
else
    echo "Mode: full repository"
fi

set +e
quality_run_python_tool skylos "${scan_args[@]}"
scan_exit_code=$?
set -e

if [ ! -f "$report_path" ]; then
    echo "❌ Skylos danger advisory did not produce $report_path." >&2
    exit "$scan_exit_code"
fi

quality_run_python_tool skylos cicd annotate --input "$report_path" --severity medium

gate_args=(cicd gate --input "$report_path" --advisory)
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    gate_args+=(--summary)
fi
if [ "$MODE" = "diff" ]; then
    gate_args+=(--diff-base "$DIFF_BASE")
fi
quality_run_python_tool skylos "${gate_args[@]}"

danger_count="$($QUALITY_PYTHON - "$report_path" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
danger_findings = report.get("danger", [])

if not isinstance(danger_findings, list):
    raise SystemExit("0")

print(sum(1 for finding in danger_findings if isinstance(finding, dict)))
PY
)"

if [ "$scan_exit_code" -ne 0 ] && [ "$danger_count" -eq 0 ]; then
    echo "❌ Skylos danger advisory exited with $scan_exit_code without recording findings." >&2
    exit "$scan_exit_code"
fi

if [ "$WRITE_LLM_REPORT" -eq 1 ]; then
    set +e
    quality_run_python_tool skylos "${llm_args[@]}"
    llm_exit_code=$?
    set -e

    if [ ! -f "$llm_report_path" ]; then
        echo "❌ Skylos LLM report was requested but not produced." >&2
        exit "$llm_exit_code"
    fi

    echo "LLM triage report: $llm_report_path"
fi

if [ "$danger_count" -eq 0 ]; then
    echo "✅ Skylos danger advisory found no issues. JSON report: $report_path"
else
    echo "⚠️  Skylos danger advisory reported $danger_count findings. JSON report: $report_path"
fi
