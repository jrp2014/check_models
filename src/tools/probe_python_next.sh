#!/usr/bin/env bash
# Probe whether a newer Python is viable for this project's MLX stack, using a
# throwaway conda env so the working `mlx-vlm` env is never touched.
#
# The working env stays on the validated Python (see CLAUDE.md / pyproject
# requires-python). This script answers "has <next Python> become viable yet?"
# by installing the PyPI stack into an isolated env and running three checks:
#
#   1. wheels:  every runtime dependency resolves and imports;
#   2. tests:   the fast pytest lane passes against the PyPI stack;
#   3. build:   (optional) the local mlx source tree compiles for that Python —
#               the one signal PyPI wheels cannot give, and the thing that
#               would actually break `tools/update.sh` after a switch.
#
# Usage:
#   bash tools/probe_python_next.sh                 # 3.14, wheels + tests
#   PROBE_PYTHON=3.15 bash tools/probe_python_next.sh
#   PROBE_SOURCE_BUILD=1 bash tools/probe_python_next.sh   # also try mlx source build
#   PROBE_RECREATE=1 bash tools/probe_python_next.sh       # fresh env first
#
# conda + pip only (never uv). Read-only with respect to the working env.

set -euo pipefail

PROBE_PYTHON="${PROBE_PYTHON:-3.14}"
PROBE_ENV="${PROBE_ENV:-mlx-vlm-${PROBE_PYTHON//./}}"
PROBE_SOURCE_BUILD="${PROBE_SOURCE_BUILD:-0}"
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
MLX_REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)/mlx"

if ! command -v conda >/dev/null 2>&1; then
    echo "❌ conda not found on PATH" >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "$PROBE_ENV" ]]; then
    echo "❌ Refusing to run while $PROBE_ENV is the active env; run from another env." >&2
    exit 1
fi

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

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Check 1/3: PyPI wheels resolve and import"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if "$PY" -m pip install -q -e "${PROJECT_ROOT}[extras]"; then
    echo "✓ pip install -e .[extras] succeeded"
else
    echo "❌ pip install failed — the PyPI stack is not yet available for Python $PROBE_PYTHON"
    exit 1
fi
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
else
    echo "❌ Import failures above"
    exit 1
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Check 2/3: fast pytest lane against the PyPI stack"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"$PY" -m pip install -q pytest pytest-xdist
if (cd "$PROJECT_ROOT" && "$PY" -m pytest -q -m "not slow and not e2e" -x -p no:cacheprovider 2>&1 | tail -3); then
    echo "✓ Fast test lane passed"
else
    echo "❌ Fast test lane failed on Python $PROBE_PYTHON"
    exit 1
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Check 3/3: local mlx source build (the update.sh signal)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$PROBE_SOURCE_BUILD" != "1" ]]; then
    echo "   skipped (set PROBE_SOURCE_BUILD=1 to compile $MLX_REPO for Python $PROBE_PYTHON)"
    echo "   Note: this is the check that decides whether tools/update.sh would keep"
    echo "   working after a switch; PyPI wheels passing does not imply it."
elif [[ ! -d "$MLX_REPO/.git" ]]; then
    echo "   skipped: no local mlx checkout at $MLX_REPO"
else
    # Mirrors mlx's [build-system] requires; nanobind is fetched by mlx's CMake.
    "$PY" -m pip install -q "setuptools>=80,<82" cmake typing_extensions
    # MLX's setup.py spawns `cmake` from PATH; make the probe env's own
    # pip-installed cmake win over any active env.
    probe_bin="$(dirname "$PY")"
    if (cd "$MLX_REPO" && PATH="$probe_bin:$PATH" "$PY" -m pip install -q --no-build-isolation -e . 2>&1 | tail -15); then
        echo "✓ mlx source build succeeded for Python $PROBE_PYTHON"
        "$PY" -c "import mlx.core as mx; print('   built mlx', mx.__version__)"
    else
        echo "❌ mlx source build FAILED for Python $PROBE_PYTHON — do not switch the working env yet"
        exit 1
    fi
fi
echo ""
echo "[probe] Done. Working env untouched: $(conda run -n mlx-vlm python -V 2>/dev/null || echo 'mlx-vlm env not queried')"
