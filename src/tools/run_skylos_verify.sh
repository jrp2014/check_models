#!/usr/bin/env bash
# Deterministic Skylos verifier wrapper for post-edit agent checks.
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bash tools/run_skylos_verify.sh --file PATH [--range L1:L2] [additional skylos verify args]

Examples:
  bash tools/run_skylos_verify.sh --file src/check_models.py --range 100:130
  bash tools/run_skylos_verify.sh --file src/check_models.py --range 100:130 --no-fail
  printf '{"file":"src/check_models.py","code":"print(1)\n","range":"1:1"}' | \
    bash tools/run_skylos_verify.sh --stdin --no-fail

This wrapper always runs `skylos verify` from the repository root with
`--project-context` so file-level checks keep access to surrounding repo facts.
EOF
}

if [ "$#" -eq 0 ]; then
    usage
    exit 2
fi

case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common_quality.sh
source "$SCRIPT_DIR/common_quality.sh"

cd "$(quality_repo_root)"
quality_setup_python
quality_require_python_tool skylos "Install dev dependencies with: pip install -e .[dev]"

echo "=== Skylos Verify ==="
quality_run_python_tool skylos verify . --project-context "$@"
