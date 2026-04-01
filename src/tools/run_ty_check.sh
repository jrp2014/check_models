#!/usr/bin/env bash
# Deterministic Ty runner that resolves the repo Python environment explicitly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_quality.sh"

cd "$(quality_src_root)"
quality_setup_python
quality_require_python_tool ty "Install dev dependencies with: pip install -e .[dev]"

if [ "$#" -eq 0 ]; then
    set -- check_models.py
fi

echo "=== Ty Type Check ==="
quality_run_ty_check "$@"