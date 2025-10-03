#!/usr/bin/env bash
# Wrapper to run quality checks via Makefile so ruff/mypy run in the conda env.
set -euo pipefail
export ZSH_DISABLE_COMPFIX=true
# Ensure we run from repo root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Use package-local Makefile which proxies tools through 'conda run -n mlx-vlm'
exec make -C vlm quality

