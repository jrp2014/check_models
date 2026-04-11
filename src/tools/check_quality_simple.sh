#!/usr/bin/env bash
# Fast static checks plus non-slow tests for pre-push hooks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/run_quality_checks.sh" --fast
