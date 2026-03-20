#!/usr/bin/env bash
# Shared helpers for local hooks, static quality checks, and runtime smoke.

CONDA_ENV="${CONDA_ENV:-mlx-vlm}"

quality_source_conda_sh() {
    local conda_sh_path="$1"

    if [ ! -f "$conda_sh_path" ]; then
        return 1
    fi

    # shellcheck disable=SC1090,SC1091
    source "$conda_sh_path"
}

quality_find_conda_sh() {
    local conda_base=""
    local candidate=""

    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
            printf '%s\n' "$conda_base/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    for candidate in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh" \
        "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
        "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" \
        "/opt/homebrew/Caskroom/mambaforge/base/etc/profile.d/conda.sh" \
        "/opt/homebrew/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh"
    do
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

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
    local conda_sh_path=""

    if [ -n "${CONDA_PREFIX:-}" ]; then
        return 0
    fi

    if conda_sh_path="$(quality_find_conda_sh)"; then
        quality_source_conda_sh "$conda_sh_path"
    fi

    if command -v conda >/dev/null 2>&1; then
        conda activate "$CONDA_ENV" 2>/dev/null || true
    fi
}

quality_find_conda_env_python() {
    local env_prefix=""
    local env_python=""
    local candidate_base=""

    if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "$CONDA_PREFIX")" = "$CONDA_ENV" ]; then
        env_python="$CONDA_PREFIX/bin/python"
        if [ -x "$env_python" ]; then
            printf '%s\n' "$env_python"
            return 0
        fi
    fi

    if command -v conda >/dev/null 2>&1; then
        env_prefix="$(conda env list 2>/dev/null | awk -v env_name="$CONDA_ENV" '$1 == env_name { print $NF; exit }')"
        if [ -n "$env_prefix" ] && [ -x "$env_prefix/bin/python" ]; then
            printf '%s\n' "$env_prefix/bin/python"
            return 0
        fi
    fi

    for candidate_base in \
        "$HOME/miniconda3" \
        "$HOME/miniforge3" \
        "$HOME/mambaforge" \
        "/opt/homebrew/Caskroom/miniconda/base" \
        "/opt/homebrew/Caskroom/miniforge/base" \
        "/opt/homebrew/Caskroom/mambaforge/base" \
        "/opt/homebrew/anaconda3" \
        "$HOME/anaconda3"
    do
        env_python="$candidate_base/envs/$CONDA_ENV/bin/python"
        if [ -x "$env_python" ]; then
            printf '%s\n' "$env_python"
            return 0
        fi
    done

    return 1
}

quality_setup_python() {
    local env_python=""

    quality_activate_conda

    if env_python="$(quality_find_conda_env_python)"; then
        QUALITY_PYTHON="$env_python"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
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

try:
    import yaml
except ModuleNotFoundError as err:
    print("❌ PyYAML is required for YAML validation in quality hooks.")
    print("   Activate the repo conda env first: conda activate mlx-vlm")
    raise SystemExit(1) from err

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
