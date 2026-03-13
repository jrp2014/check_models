#!/usr/bin/env bash
# Shared helpers for local hooks, static quality checks, and runtime smoke.

CONDA_ENV="${CONDA_ENV:-mlx-vlm}"

quality_tools_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

quality_src_root() {
    cd "$(quality_tools_dir)/.." && pwd
}

quality_repo_root() {
    cd "$(quality_tools_dir)/../.." && pwd
}

quality_activate_conda() {
    local conda_base=""

    if [ -n "${CONDA_PREFIX:-}" ] || ! command -v conda >/dev/null 2>&1; then
        return 0
    fi

    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090,SC1091
        source "$conda_base/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    elif [ -f "/opt/homebrew/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/opt/homebrew/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi

    if command -v conda >/dev/null 2>&1; then
        conda activate "$CONDA_ENV" 2>/dev/null || true
    fi
}

quality_setup_python() {
    quality_activate_conda

    if [ -n "${CONDA_PREFIX:-}" ]; then
        QUALITY_PYTHON="$CONDA_PREFIX/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        QUALITY_PYTHON="python3"
    else
        QUALITY_PYTHON="python"
    fi

    export QUALITY_PYTHON
}

quality_require_command() {
    local command_name="$1"
    local install_hint="$2"

    if command -v "$command_name" >/dev/null 2>&1; then
        return 0
    fi

    echo "❌ Required command '$command_name' not found." >&2
    if [ -n "$install_hint" ]; then
        echo "   $install_hint" >&2
    fi
    return 1
}

quality_validate_yaml_files() {
    "$QUALITY_PYTHON" - "$@" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

errors: list[str] = []

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    try:
        text = path.read_text(encoding="utf-8")
        list(yaml.safe_load_all(text))
    except OSError as err:
        errors.append(f"{path}: could not read file ({err})")
    except yaml.YAMLError as err:
        errors.append(f"{path}: invalid YAML ({err})")

if errors:
    print("❌ YAML validation failed:")
    for error in errors:
        print(f"   - {error}")
    raise SystemExit(1)
PY
}

quality_run_markdownlint() {
    local src_root
    src_root="$(quality_src_root)"

    if [ -x "$src_root/node_modules/.bin/markdownlint-cli2" ]; then
        "$src_root/node_modules/.bin/markdownlint-cli2" "$@"
        return
    fi

    if command -v markdownlint-cli2 >/dev/null 2>&1; then
        markdownlint-cli2 "$@"
        return
    fi

    if command -v npx >/dev/null 2>&1; then
        npx --no-install markdownlint-cli2 "$@"
        return
    fi

    echo "❌ markdownlint-cli2 not found." >&2
    echo "   Install with: npm install --prefix src" >&2
    return 1
}
