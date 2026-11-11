#!/usr/bin/env bash
# Shared helpers for local hooks, static quality checks, and runtime smoke.

CONDA_ENV="${CONDA_ENV:-mlx-vlm}"

quality_find_conda_bin() {
    local candidate=""

    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi

    for candidate in \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/mambaforge/bin/conda" \
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda" \
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
        "/opt/homebrew/Caskroom/mambaforge/base/bin/conda" \
        "/opt/homebrew/anaconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda"
    do
        if [ -x "$candidate" ]; then
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

quality_find_conda_env_python() {
    local conda_bin=""
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

    if conda_bin="$(quality_find_conda_bin)"; then
        env_prefix="$("$conda_bin" env list 2>/dev/null | awk -v env_name="$CONDA_ENV" '$1 == env_name { print $NF; exit }')"
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

quality_python_fallback_allowed() {
    case "${QUALITY_ALLOW_PYTHON_FALLBACK:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
    esac

    case "${CI:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
    esac

    return 1
}

quality_path_tool_fallback_allowed() {
    case "${QUALITY_ALLOW_PATH_TOOLS:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
    esac

    return 1
}

quality_setup_python() {
    local env_python=""


    if env_python="$(quality_find_conda_env_python)"; then
        QUALITY_PYTHON="$env_python"
        QUALITY_PYTHON_SOURCE="conda-env:$CONDA_ENV"
    elif quality_python_fallback_allowed; then
        if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
            QUALITY_PYTHON="$CONDA_PREFIX/bin/python"
            QUALITY_PYTHON_SOURCE="active-conda:$CONDA_PREFIX"
        elif command -v python3 >/dev/null 2>&1; then
            QUALITY_PYTHON="$(command -v python3)"
            QUALITY_PYTHON_SOURCE="fallback:python3"
        elif command -v python >/dev/null 2>&1; then
            QUALITY_PYTHON="$(command -v python)"
            QUALITY_PYTHON_SOURCE="fallback:python"
        else
            echo "❌ Unable to resolve required Python interpreter." >&2
            return 1
        fi
    else
        echo "❌ Unable to resolve required conda environment '$CONDA_ENV'." >&2
        echo "   Activate it first: conda activate $CONDA_ENV" >&2
        echo "   CI may set CI=true; local overrides may set QUALITY_ALLOW_PYTHON_FALLBACK=1." >&2
        return 1
    fi

    export QUALITY_PYTHON
    export QUALITY_PYTHON_SOURCE
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

quality_find_python_tool() {
    local tool_name="$1"
    local python_bin_dir=""

    python_bin_dir="$(dirname "$QUALITY_PYTHON")"
    if [ -x "$python_bin_dir/$tool_name" ]; then
        printf '%s\n' "$python_bin_dir/$tool_name"
        return 0
    fi

    if quality_path_tool_fallback_allowed && command -v "$tool_name" >/dev/null 2>&1; then
        command -v "$tool_name"
        return 0
    fi

    return 1
}

quality_require_python_tool() {
    local tool_name="$1"
    local install_hint="$2"

    if quality_find_python_tool "$tool_name" >/dev/null 2>&1; then
        return 0
    fi

    echo "❌ Required command '$tool_name' not found." >&2
    if [ -n "$install_hint" ]; then
        echo "   $install_hint" >&2
    fi
    return 1
}

quality_run_python_tool() {
    local tool_name="$1"
    local tool_path=""

    tool_path="$(quality_find_python_tool "$tool_name")" || return 1
    shift
    "$tool_path" "$@"
}

# Every Skylos call in the gate goes through here.
# SKYLOS_NO_CLIPBOARD=1 (Skylos 5.0+) keeps it off the system clipboard: an
# interactive run would otherwise copy its README badge there, overwriting
# whatever the user had.
# SKYLOS_GREP_BUDGET caps the dead-code grep verification (Skylos's default
# is 120 s) and Skylos exits 2 as "analysis incomplete" when the cap is hit.
# The value is a cap, not a duration, so a generous one costs nothing when
# verification finishes early and keeps a slow CI runner from tripping it.
quality_run_skylos() {
    SKYLOS_NO_CLIPBOARD=1 SKYLOS_GREP_BUDGET="${SKYLOS_GREP_BUDGET:-300}" \
        quality_run_python_tool skylos "$@"
}

# The gated Skylos quality scan. Advice-only rules (e.g. SKY-P401) are
# listed but not counted towards max_quality, so a passing gate can still
# print findings: keep those lines in QUALITY_SKYLOS_ADVICE for
# quality_print_skylos_advice to repeat at the end of the run. The output
# goes to a file (shown afterwards) rather than a pipe, so the scan's own
# exit status survives set -e and pipefail in the caller. A failing scan
# returns its status unchanged.
quality_run_skylos_quality_gate() {
    local gate_log status=0
    QUALITY_SKYLOS_ADVICE=""
    gate_log="$(mktemp "${TMPDIR:-/tmp}/skylos-quality-gate.XXXXXX")" || return 1
    TERM=dumb NO_COLOR=1 CLICOLOR=0 FORCE_COLOR=0 PY_COLORS=0 \
        quality_run_skylos . --quality --secrets --ai-defects --gate --no-upload --format concise \
        </dev/null >"$gate_log" || status=$?
    cat "$gate_log"
    if [ "$status" -eq 0 ]; then
        # Concise finding lines are "<path>:<line>  <SKY-rule>  <message>";
        # the path may contain spaces.
        QUALITY_SKYLOS_ADVICE="$(grep -E '^.+:[0-9]+  SKY-' "$gate_log" || true)"
    fi
    rm -f "$gate_log"
    return "$status"
}

quality_print_skylos_advice() {
    if [ -n "${QUALITY_SKYLOS_ADVICE:-}" ]; then
        echo "⚠️  Skylos advice (passes the gate, not blocking; fix it or add a reasoned ignore):"
        printf '%s\n' "$QUALITY_SKYLOS_ADVICE" | sed 's/^/    /'
        echo ""
    fi
}

quality_resolve_python_path() {
    if [ -n "${QUALITY_PYTHON:-}" ] && [ -x "$QUALITY_PYTHON" ]; then
        printf '%s\n' "$QUALITY_PYTHON"
        return 0
    fi

    if [ -n "${QUALITY_PYTHON:-}" ] && command -v "$QUALITY_PYTHON" >/dev/null 2>&1; then
        command -v "$QUALITY_PYTHON"
        return 0
    fi

    return 1
}

quality_print_ty_diagnostics() {
    local python_path="$1"
    local ty_path="$2"
    shift 2

    echo "[ty] target conda env: ${CONDA_ENV}"
    echo "[ty] active conda env: ${CONDA_DEFAULT_ENV:-<none>}"
    echo "[ty] resolved python (${QUALITY_PYTHON_SOURCE:-unknown}): ${python_path}"
    echo "[ty] resolved ty: ${ty_path}"
    echo "[ty] check targets: ${*:-check_models.py}"

    if [ "${QUALITY_PYTHON_SOURCE:-}" != "conda-env:${CONDA_ENV}" ]; then
        echo "[ty] warning: target env '${CONDA_ENV}' was not resolved directly; import resolution may differ from the expected repo conda environment." >&2
    fi
}

quality_run_ty_check() {
    local python_path=""
    local ty_path=""

    python_path="$(quality_resolve_python_path)" || return 1
    ty_path="$(quality_find_python_tool ty)" || return 1
    quality_print_ty_diagnostics "$python_path" "$ty_path" "$@"
    "$ty_path" check --python "$python_path" "$@"
}

quality_print_pyrefly_diagnostics() {
    local python_path="$1"
    local pyrefly_path="$2"
    local config_path="$3"
    local target_display=""
    shift 3

    if [ "$#" -eq 0 ]; then
        target_display="<project discovery>"
    elif [ "$#" -gt 8 ]; then
        target_display="$# explicit files (git ls-files enumeration)"
    else
        target_display="$*"
    fi

    echo "[pyrefly] active conda env: ${CONDA_DEFAULT_ENV:-<none>}"
    echo "[pyrefly] resolved python (${QUALITY_PYTHON_SOURCE:-unknown}): ${python_path}"
    echo "[pyrefly] resolved pyrefly: ${pyrefly_path}"
    echo "[pyrefly] config: ${config_path}"
    echo "[pyrefly] check targets: ${target_display}"
}

# Explicit targets switch Pyrefly to single-file checking mode, which
# skips filesystem discovery entirely — so the gate's file set cannot be
# eaten by ignore files or discovery heuristics (a linked worktree under
# .claude/worktrees is hidden and ignored by its parent repository). git
# ls-files is the same authority the other gates align with and keeps
# gitignore semantics without hand-mirrored exclude globs.
quality_pyrefly_default_targets() {
    git -C "$(quality_src_root)" ls-files --cached --others --exclude-standard -- '*.py' '*.pyi' \
        | LC_ALL=C sort -u \
        | grep -v '^tools/\.archived/' || true
}

quality_run_pyrefly_check() {
    local python_path=""
    local pyrefly_path=""
    local config_path=""
    local output_path=""
    local exit_code=0
    local -a targets=("$@")

    if [ "${#targets[@]}" -eq 0 ]; then
        local src_root=""
        src_root="$(quality_src_root)"
        while IFS= read -r target; do
            targets+=("$src_root/$target")
        done < <(quality_pyrefly_default_targets)
        if [ "${#targets[@]}" -eq 0 ]; then
            echo "❌ Pyrefly target enumeration found no Python files (git ls-files returned nothing)." >&2
            return 1
        fi
    fi

    python_path="$(quality_resolve_python_path)" || return 1
    pyrefly_path="$(quality_find_python_tool pyrefly)" || return 1
    # [tool.pyrefly] in pyproject.toml is the whole configuration; only the
    # interpreter (whose site-packages Pyrefly reads) is decided per run.
    config_path="$(quality_src_root)/pyproject.toml"
    # The output goes to a file in $TMPDIR and is then printed, rather than
    # through `| tee`: a pipeline reports tee's status unless the caller set
    # pipefail, which would turn Pyrefly's errors into a pass.
    output_path="$(mktemp "${TMPDIR:-/tmp}/pyrefly-output.XXXXXX")"

    quality_print_pyrefly_diagnostics "$python_path" "$pyrefly_path" "$config_path" "${targets[@]}"
    if "$pyrefly_path" check \
        -c "$config_path" \
        --python-interpreter-path "$python_path" \
        --min-severity warn \
        --output-format full-text \
        "${targets[@]}" > "$output_path" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi
    cat "$output_path"

    if [ "$exit_code" -eq 0 ] && grep -Eq '^[[:space:]]*WARN([[:space:]]|$)' "$output_path"; then
        echo "❌ Pyrefly emitted warnings; treat warnings as quality failures." >&2
        exit_code=1
    fi

    rm -f "$output_path"
    return "$exit_code"
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
