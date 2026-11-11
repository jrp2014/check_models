#!/usr/bin/env bash
# Lightweight runtime/Metal smoke probe for CI and manual validation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common_quality.sh
source "$SCRIPT_DIR/common_quality.sh"

cd "$(quality_src_root)"
quality_setup_python

echo "=== Runtime Dependency Smoke ==="
"$QUALITY_PYTHON" "$SCRIPT_DIR/runtime_smoke.py"
