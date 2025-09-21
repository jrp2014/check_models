#!/usr/bin/env bash
# Wrapper to run quality checks in a non-interactive way, bypassing interactive zsh init issues.
# Sets ZSH_DISABLE_COMPFIX to avoid compinit prompts and executes the Python quality script directly.
set -euo pipefail
export ZSH_DISABLE_COMPFIX=true
# Ensure we run from repo root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
exec python vlm/tools/check_quality.py "$@"
