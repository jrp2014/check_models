#!/usr/bin/env bash
# Simplified quality checker - runs directly without complex orchestration
set -euo pipefail

# Navigate to src directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Run the simple quality checks
exec bash tools/check_quality_simple.sh

