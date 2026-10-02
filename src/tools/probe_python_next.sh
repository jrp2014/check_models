#!/usr/bin/env bash
# Probe whether a newer Python is viable for this project's MLX stack, using a
# throwaway conda env so the working `mlx-vlm` env is never touched.
#
# The working env stays on the validated Python (see CLAUDE.md / pyproject
# requires-python). This script answers "has <next Python> become viable yet?"
# by installing the stack into an isolated env and running four checks:
#
#   1. build:   (PROBE_SOURCE_BUILD=1) the local mlx checkout compiles for that
#               Python — the one signal PyPI wheels cannot give, and the thing
#               that would actually break `tools/update.sh` after a switch. It
#               runs first and its result is pinned for check 2, so a missing
#               mlx wheel on PyPI cannot hide it. The build runs in a throwaway
#               clone of the checkout's HEAD: an in-place build would overwrite
#               python/mlx/lib/{libmlx.dylib,mlx.metallib}, which the working
#               env's editable install loads.
#   2. wheels:  the project and every other runtime dependency resolve and
#               import (mlx from check 1 when it ran, otherwise from PyPI);
#   3. tests:   the fast pytest lane passes against that stack;
#   4. torch:   (PROBE_TORCH=1) the torch extra installs and torch/torchvision
#               import. update.sh installs that extra by default because
#               transformers 5 builds torch-backed processors for about half
#               the roster (and its mlx install, `-e .[dev]`, needs torch via
#               mlx's dev extra). Reported on its own line, outside the core
#               verdict.
#
# Exit status is the core verdict (checks 1-3): 0 viable, 1 not viable. Every
# check runs unless an earlier one it depends on failed.
#
# Usage:
#   bash tools/probe_python_next.sh                 # 3.15, wheels + tests
#   PROBE_PYTHON=3.16 bash tools/probe_python_next.sh
#   PROBE_SOURCE_BUILD=1 bash tools/probe_python_next.sh   # build mlx from source first
#   PROBE_TORCH=1 bash tools/probe_python_next.sh          # also check the torch extra
#   PROBE_RECREATE=1 bash tools/probe_python_next.sh       # fresh env first
#   PROBE_MLX_REPO=/path/to/mlx                            # mlx checkout (default: sibling)
#
# conda + pip only (never uv). Read-only with respect to the working env.

set -euo pipefail

PROBE_PYTHON="${PROBE_PYTHON:-3.15}"
PROBE_ENV="${PROBE_ENV:-mlx-vlm-${PROBE_PYTHON//./}}"
PROBE_SOURCE_BUILD="${PROBE_SOURCE_BUILD:-0}"
PROBE_TORCH="${PROBE_TORCH:-0}"
PROBE_RECREATE="${PROBE_RECREATE:-0}"

# Both values reach `conda create` and a conda-meta file path; allowlist them
# strictly so a stray value cannot become a shell or path traversal vector.
if [[ ! "$PROBE_PYTHON" =~ ^3\.[0-9]{1,2}$ ]]; then
    echo "❌ PROBE_PYTHON must look like 3.NN (got '$PROBE_PYTHON')" >&2
    exit 1
fi
if [[ ! "$PROBE_ENV" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
    echo "❌ PROBE_ENV must be a plain conda env name (got '$PROBE_ENV')" >&2
    exit 1
fi
if [[ "$PROBE_ENV" == "mlx-vlm" ]]; then
    echo "❌ Refusing to use the working env 'mlx-vlm' as the probe env." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MLX_REPO="${PROBE_MLX_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)/mlx}"

if ! command -v conda >/dev/null 2>&1; then
    echo "❌ conda not found on PATH" >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "$PROBE_ENV" ]]; then
    echo "❌ Refusing to run while $PROBE_ENV is the active env; run from another env." >&2
    exit 1
fi

# Holds the mlx clone, its build tree and the mlx pin; removed on exit.
probe_tmp_root="${TMPDIR:-/tmp}"
PROBE_TMP="$(mktemp -d "${probe_tmp_root%/}/probe-python-next.XXXXXXXX")"
cleanup_probe_tmp() {
    rm -rf "$PROBE_TMP"
}
trap cleanup_probe_tmp EXIT

section() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

if [[ "$PROBE_RECREATE" == "1" ]] && conda env list | grep -qE "^${PROBE_ENV}\s"; then
    echo "[probe] Removing existing $PROBE_ENV (PROBE_RECREATE=1)..."
    conda env remove -n "$PROBE_ENV" -y >/dev/null
fi

if ! conda env list | grep -qE "^${PROBE_ENV}\s"; then
    echo "[probe] Creating $PROBE_ENV with python=${PROBE_PYTHON}.* ..."
    conda create -n "$PROBE_ENV" -y "python=${PROBE_PYTHON}.*" pip >/dev/null
    # Resolve the prefix from conda itself, canonicalise it, and require the
    # conda-meta directory to already exist before writing the pin file.
    conda_prefix="$(conda run -n "$PROBE_ENV" python -c 'import sys; print(sys.prefix)')"
    conda_prefix="$(cd "$conda_prefix" && pwd -P)"
    pin_dir="$conda_prefix/conda-meta"
    if [[ ! -d "$pin_dir" ]]; then
        echo "❌ Expected conda-meta directory missing at $pin_dir" >&2
        exit 1
    fi
    # Same protection as the working env: pin the minor so `conda update`
    # can never jump the interpreter and orphan every cp-tagged package.
    printf 'python %s.*\n' "$PROBE_PYTHON" > "$pin_dir/pinned"
fi

PY="$(conda run -n "$PROBE_ENV" python -c 'import sys; print(sys.executable)')"
echo "[probe] Interpreter: $("$PY" -V) at $PY"
echo ""

# passed | failed | skipped; tests and wheels stay "skipped" when a check they
# depend on did not pass.
build_status="skipped"
wheels_status="skipped"
tests_status="skipped"
torch_status="not checked"
# A --constraint pinning the check-1 mlx build, so no later pip run swaps it
# for (or fails looking for) a PyPI wheel.
MLX_PIN_ARGS=()

# Build the checkout's HEAD in a clone under $PROBE_TMP and pin the result.
build_mlx_from_clone() {
    local build_dir="$PROBE_TMP/mlx"
    local mlx_version
    if ! git clone -q --no-local --depth 1 "$MLX_REPO" "$build_dir"; then
        echo "❌ Could not clone $MLX_REPO for the build"
        return 1
    fi
    # pip's isolated build, as tools/update.sh runs it: mlx's [build-system]
    # requires (setuptools, cmake, typing_extensions) resolve for this Python.
    # Non-editable and without update.sh's `.[dev]` extra, which pulls torch;
    # check 4 reports torch on its own.
    (cd "$build_dir" && "$PY" -m pip install -q . 2>&1 | tail -15) || return 1
    "$PY" -c "import mlx.core as mx; print('   built mlx', mx.__version__, '- metal available:', mx.metal.is_available())" || return 1
    mlx_version="$("$PY" -c 'import importlib.metadata as m; print(m.version("mlx"))')" || return 1
    printf 'mlx==%s\n' "$mlx_version" > "$PROBE_TMP/mlx-pin.txt"
    MLX_PIN_ARGS=(--constraint "$PROBE_TMP/mlx-pin.txt")
    echo "   mlx==$mlx_version pinned for the checks below"
}

section "Check 1/4: local mlx source build (the update.sh signal)"
if [[ "$PROBE_SOURCE_BUILD" != "1" ]]; then
    echo "   skipped (set PROBE_SOURCE_BUILD=1 to compile $MLX_REPO for Python $PROBE_PYTHON)"
    echo "   Note: this is the check that decides whether tools/update.sh would keep"
    echo "   working after a switch; PyPI wheels passing does not imply it. Without"
    echo "   it, check 2 needs an mlx wheel for Python $PROBE_PYTHON on PyPI."
elif [[ ! -e "$MLX_REPO/.git" ]]; then
    echo "   skipped: no local mlx checkout at $MLX_REPO"
else
    echo "   Building mlx $(git -C "$MLX_REPO" rev-parse --short HEAD 2>/dev/null || echo '(unknown HEAD)') from a throwaway clone of $MLX_REPO"
    echo "   (the checkout itself is not written to; uncommitted changes are not built)"
    if build_mlx_from_clone; then
        echo "✓ mlx source build succeeded for Python $PROBE_PYTHON"
        build_status="passed"
    else
        echo "❌ mlx source build FAILED for Python $PROBE_PYTHON — do not switch the working env yet"
        build_status="failed"
    fi
fi
echo ""

section "Check 2/4: project and runtime dependencies resolve and import"
if [[ "$build_status" == "failed" ]]; then
    echo "   skipped: the mlx source build failed (check 1)"
else
    if [[ "$build_status" == "passed" ]]; then
        echo "   mlx: the check-1 source build; everything else from PyPI"
    else
        echo "   mlx and everything else from PyPI"
    fi
    wheels_status="failed"
    if "$PY" -m pip install -q ${MLX_PIN_ARGS[@]+"${MLX_PIN_ARGS[@]}"} -e "${PROJECT_ROOT}[extras]"; then
        echo "✓ pip install -e .[extras] succeeded"
        if "$PY" - <<'PY'
import importlib, sys
mods = ("mlx.core", "mlx_vlm", "transformers", "tokenizers", "PIL", "numpy", "huggingface_hub")
failed = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:  # noqa: BLE001 - probe reports every failure kind
        failed.append(f"{m}: {type(exc).__name__}: {exc}")
if failed:
    print("\n".join(failed))
    sys.exit(1)
import mlx.core as mx, mlx_vlm, transformers
print(f"   mlx {mx.__version__}, mlx-vlm {mlx_vlm.__version__}, transformers {transformers.__version__}")
print(f"   metal available: {mx.metal.is_available()}")
PY
        then
            echo "✓ All runtime imports OK"
            wheels_status="passed"
        else
            echo "❌ Import failures above"
        fi
    elif [[ "$build_status" == "passed" ]]; then
        echo "❌ pip install failed — a runtime dependency other than mlx is not yet available for Python $PROBE_PYTHON"
    else
        echo "❌ pip install failed — the PyPI stack is not yet available for Python $PROBE_PYTHON"
        echo "   If only mlx lacks a wheel, PROBE_SOURCE_BUILD=1 builds it from the local checkout first."
    fi
fi
echo ""

section "Check 3/4: fast pytest lane"
if [[ "$wheels_status" != "passed" ]]; then
    echo "   skipped: the stack did not install and import (check 2)"
else
    tests_status="failed"
    if "$PY" -m pip install -q ${MLX_PIN_ARGS[@]+"${MLX_PIN_ARGS[@]}"} pytest pytest-xdist \
        && (cd "$PROJECT_ROOT" && "$PY" -m pytest -q -m "not slow and not e2e" -x -p no:cacheprovider 2>&1 | tail -3); then
        echo "✓ Fast test lane passed"
        tests_status="passed"
    else
        echo "❌ Fast test lane failed on Python $PROBE_PYTHON"
    fi
fi
echo ""

section "Check 4/4: torch extra (outside the core verdict)"
if [[ "$PROBE_TORCH" != "1" ]]; then
    echo "   skipped (set PROBE_TORCH=1 to install the torch extra and import torch/torchvision)"
else
    # The extra's own requirements from pyproject.toml, installed without the
    # project, so the result does not depend on mlx resolving in check 2.
    torch_reqs=()
    torch_req_lines="$("$PY" -c 'import sys, tomllib; print("\n".join(tomllib.load(open(sys.argv[1], "rb"))["project"]["optional-dependencies"]["torch"]))' "$PROJECT_ROOT/pyproject.toml" || true)"
    while IFS= read -r req; do
        if [[ -n "$req" ]]; then
            torch_reqs+=("$req")
        fi
    done <<<"$torch_req_lines"
    torch_status="unavailable"
    if [[ ${#torch_reqs[@]} -eq 0 ]]; then
        echo "❌ Could not read the torch extra from $PROJECT_ROOT/pyproject.toml"
    else
        echo "   installing: ${torch_reqs[*]}"
    fi
    if [[ ${#torch_reqs[@]} -gt 0 ]] \
        && "$PY" -m pip install -q ${MLX_PIN_ARGS[@]+"${MLX_PIN_ARGS[@]}"} "${torch_reqs[@]}" \
        && "$PY" -c "import torch, torchvision; print('   torch', torch.__version__, '- torchvision', torchvision.__version__)"; then
        echo "✓ torch extra installs and torch/torchvision import"
        torch_status="available"
    else
        echo "⚠️  torch extra unavailable: about half the roster would fail"
    fi
fi
echo ""

core_viable=0
if [[ "$build_status" != "failed" && "$wheels_status" == "passed" && "$tests_status" == "passed" ]]; then
    core_viable=1
fi

section "Verdict for Python $PROBE_PYTHON"
echo "   mlx source build: $build_status | install + imports: $wheels_status | fast tests: $tests_status"
if [[ "$core_viable" == "1" && "$build_status" == "passed" ]]; then
    echo "✓ Core: viable, including the mlx source build tools/update.sh relies on"
elif [[ "$core_viable" == "1" ]]; then
    echo "✓ Core: viable from PyPI; the mlx source build tools/update.sh relies on was not checked"
else
    echo "❌ Core: not viable yet"
fi
case "$torch_status" in
    available) echo "✓ Torch extra: available" ;;
    unavailable) echo "⚠️  Torch extra: unavailable: about half the roster would fail" ;;
    *) echo "   Torch extra: not checked (PROBE_TORCH=1)" ;;
esac
echo ""
echo "[probe] Done. Working env untouched: $(conda run -n mlx-vlm python -V 2>/dev/null || echo 'mlx-vlm env not queried')"
if [[ "$core_viable" != "1" ]]; then
    exit 1
fi
