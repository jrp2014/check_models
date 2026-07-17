#!/usr/bin/env bash
# Unified dependency updater for mlx-vlm-check project.
# Installs/updates the project and all dependencies from pyproject.toml.
#
# Execution Order:
#   1. Update conda/Homebrew by default unless UPDATE_SYSTEM_PACKAGES=0
#   2. Install repo-local npm tooling from lockfile (or latest if UPDATE_NODE_TOOLING=1)
#   3. Update pip/wheel/setuptools
#   4. Update local MLX repos (if present) OR update from PyPI
#   5. Reinstall project in editable mode from pyproject.toml to reconcile deps
#
# Usage examples:
#   ./update.sh                       # Install project + dev + extras + torch (MLX_METAL_JIT=OFF by default)
#   SKIP_TORCH=1 ./update.sh          # Skip torch group installation
#   FORCE_REINSTALL=1 ./update.sh     # Force reinstall with --force-reinstall
#   SKIP_MLX=1 ./update.sh            # Force skip mlx/mlx-vlm updates (override detection)
#   CONDA_UPDATE_ALL=1 ./update.sh    # Force conda update --all even with pip conflicts
#   UPDATE_SYSTEM_PACKAGES=0 ./update.sh # Skip conda base/env and Homebrew updates
#   UPDATE_NODE_TOOLING=1 ./update.sh # Upgrade markdownlint-cli2 to latest npm release
#   MLX_METAL_JIT=ON ./update.sh      # Build MLX with runtime Metal kernel compilation
#   MACOSX_DEPLOYMENT_TARGET=26.2 ./update.sh # Override local mlx build target
#   MLX_LOCAL_BUILD_SMOKE=1 ./update.sh # Force local MLX runtime smoke test
#   CLEAN_BUILD=1 ./update.sh         # Clean build artifacts before building local MLX repos
#
# Local MLX Development:
#   If mlx, mlx-lm, and mlx-vlm directories exist at ../../ (sibling to check_models/),
#   the script will automatically:
#   1. Run git pull in each repository
#   2. Install requirements.txt (if present) for additional dependencies
#      (Note: mlx requires setuptools>=80 and typing_extensions for builds)
#   3. Install packages in dependency order: mlx → mlx-lm → mlx-vlm
#   4. Verify editable install origins for mlx, mlx-lm, and mlx-vlm
#   5. Skip PyPI MLX updates for these packages
#   6. Reinstall check_models from pyproject.toml to reconcile shared deps
#   7. Generate stubs for this project (mlx_lm, mlx_vlm, transformers, tokenizers)
#   8. Run the local MLX runtime smoke if a local mlx build was installed
#
# Requirements for local MLX builds:
#   - CMake >= 3.25 (MLX minimum requirement as of 2025)
#   - C++20 compiler (Apple Clang >= 15 on macOS)
#   - Xcode >= 15.0 (macOS, for Metal support)
#   - macOS SDK >= 14.0 and a native arm64 shell on Apple Silicon
#   - typing_extensions (required by MLX build)
#   - setuptools>=80 (required for stub generation)
#
# Note:
# - If MLX_METAL_JIT is set, update.sh maps it directly to CMake option
#   MLX_METAL_JIT (ON/OFF). If unset, MLX's CMake default is used (OFF).
# - Local mlx builds leave MACOSX_DEPLOYMENT_TARGET selection to upstream MLX.
#   Set MACOSX_DEPLOYMENT_TARGET explicitly only when overriding that default.
# - MLX Python builds already force -DMLX_BUILD_PYTHON_BINDINGS=ON in mlx/setup.py.
#   update.sh intentionally does not override that flag.
# - Script preserves local MLX ecosystem builds using local-repo detection and
#   editable-install metadata (not only version strings).

set -euo pipefail

# Check if we're in a virtual environment (uv, conda, venv, virtualenv)
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -z "${CONDA_DEFAULT_ENV:-}" ]] && [[ -z "${UV_ACTIVE:-}" ]]; then
	echo "⚠️  WARNING: You don't appear to be in a virtual environment!"
	echo "   (No VIRTUAL_ENV, CONDA_DEFAULT_ENV, or UV_ACTIVE detected)"
	echo ""
	echo "   This script will update packages globally on your system."
	echo "   It's strongly recommended to activate a virtual environment first:"
	echo ""
	echo "   • conda: conda activate <env-name>"
	echo "   • venv/virtualenv: source /path/to/venv/bin/activate"
	echo "   • uv: uv venv && source .venv/bin/activate"
	echo ""
	read -p "   Continue anyway? [y/N] " -n 1 -r
	echo
	if [[ ! $REPLY =~ ^[Yy]$ ]]; then
		echo "[update.sh] Aborted by user."
		exit 1
	fi
	echo "[update.sh] Proceeding with global installation (user confirmed)..."
fi

cleanup_pip_invalid_distribution_backups() {
	if [[ "${CLEAN_PIP_INVALID_DISTS:-1}" != "1" ]]; then
		echo "[update.sh] Skipping stale pip invalid-distribution cleanup (CLEAN_PIP_INVALID_DISTS=0)"
		return 0
	fi

	if ! python - <<'PY'
from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path

site_dirs = {
    value
    for key in ("purelib", "platlib")
    if (value := sysconfig.get_path(key)) is not None
}
removed: list[Path] = []

for site_dir_value in sorted(site_dirs):
    site_dir = Path(site_dir_value)
    if not site_dir.is_dir():
        continue

    for path in site_dir.iterdir():
        name = path.name
        if not name.startswith("~"):
            continue
        if path.is_symlink():
            print(f"[update.sh] Skipping stale pip backup symlink: {path}", file=sys.stderr)
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)

if removed:
    print("[update.sh] Removed stale pip invalid-distribution backup(s):")
    for path in removed:
        print(f"   - {path}")
PY
	then
		echo "⚠️  Unable to clean stale pip invalid-distribution backups; continuing"
	fi
}

cleanup_pip_invalid_distribution_backups

# Update conda itself (base) and environment packages with pip-conflict safety check.
# NOTE: conda update --all can break pip-installed packages (mlx, mlx-vlm) by
# reshuffling shared dependencies like numpy. Keep conflict checks in place.
if [[ "${UPDATE_SYSTEM_PACKAGES:-1}" == "1" ]]; then
	if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
		echo "[update.sh] Updating conda (base)..."
		conda update -n base conda -y

		# Now attempt to update the active environment's conda-managed packages.
		# Dry-run first to detect conflicts with pip-installed packages.
		echo ""
		echo "[update.sh] Checking for safe conda environment updates (dry-run)..."
		DRY_RUN_OUTPUT=$(conda update --all --dry-run 2>&1) || true

		# Extract package names conda wants to change
		CONDA_CHANGES=$(echo "$DRY_RUN_OUTPUT" | grep -E '^\s+\S+\s+\S+\s+->\s+\S+' | awk '{print $1}' 2>/dev/null || true)

		if [[ -z "$CONDA_CHANGES" ]]; then
			echo "[update.sh] Conda environment is already up to date"
		else
			# Get pip-installed packages (not installed by conda)
			PIP_ONLY_PKGS=$(pip list --format=freeze 2>/dev/null | cut -d= -f1 | tr '[:upper:]' '[:lower:]' || true)

			# Check for overlaps between conda changes and pip packages
			CONFLICTS=""
			while IFS= read -r pkg; do
				[[ -z "$pkg" ]] && continue
				pkg_lower=$(echo "$pkg" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
				if echo "$PIP_ONLY_PKGS" | tr '_' '-' | grep -qx "$pkg_lower"; then
					CONFLICTS="${CONFLICTS}  - ${pkg}\n"
				fi
			done <<< "$CONDA_CHANGES"

			if [[ -n "$CONFLICTS" ]]; then
				echo ""
				echo "⚠️  WARNING: conda wants to update packages also managed by pip:"
				echo -e "$CONFLICTS"
				echo "   This may break pip-installed packages (mlx, mlx-vlm, etc.)."
				echo "   Skipping conda update --all to be safe."
				echo "   To force: CONDA_UPDATE_ALL=1 ./update.sh"
				echo ""
				if [[ "${CONDA_UPDATE_ALL:-0}" == "1" ]]; then
					echo "[update.sh] CONDA_UPDATE_ALL=1 set — proceeding with conda update --all..."
					conda update --all -y
				fi
			else
				echo "[update.sh] No pip conflicts detected — updating conda environment packages..."
				conda update --all -y
			fi
		fi
	else
		echo "[update.sh] Not in conda environment; skipping conda update"
	fi
else
	echo "[update.sh] Skipping conda base/environment package updates (UPDATE_SYSTEM_PACKAGES=0)"
fi

# Homebrew updates are system-wide; run them with the conda system path unless explicitly skipped.
if [[ "${UPDATE_SYSTEM_PACKAGES:-1}" == "1" ]]; then
	if command -v brew >/dev/null 2>&1; then
		echo "[update.sh] Updating Homebrew..."
		brew update
		brew upgrade
	else
		echo "[update.sh] Homebrew not found; skipping brew update/upgrade"
	fi
else
	echo "[update.sh] Skipping Homebrew update/upgrade (UPDATE_SYSTEM_PACKAGES=0)"
fi

# Determine project root (assuming check_models/src/tools/update.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Repo-local markdownlint tools (if npm is available)
if command -v npm >/dev/null 2>&1; then
	if [[ "${UPDATE_NODE_TOOLING:-0}" == "1" ]]; then
		echo "[update.sh] Updating repo-local markdownlint-cli2 to the latest npm release..."
		npm install --prefix "$PROJECT_ROOT" --save-dev --save-exact markdownlint-cli2@latest
	else
		echo "[update.sh] Installing repo-local markdownlint tooling from package-lock.json..."
		npm install --ignore-scripts --prefix "$PROJECT_ROOT"
	fi
	echo "[update.sh] repo-local npm tooling is installed"
else
	echo "[update.sh] npm not found; skipping repo-local markdownlint install"
	echo "   (Install Node.js/npm for markdown linting: brew install node)"
fi

# Helper function: pip install with optional force reinstall (eager upgrades)
pip_install() {
	local args=("-U" "--upgrade-strategy" "eager")
	[[ "${FORCE_REINSTALL:-0}" == "1" ]] && args+=("--force-reinstall")
	pip install "${args[@]}" "$@"
}

# Helper function: pip install with verbose output for build-heavy operations.
pip_install_verbose() {
	local args=("-v" "-U" "--upgrade-strategy" "eager")
	[[ "${FORCE_REINSTALL:-0}" == "1" ]] && args+=("--force-reinstall")
	pip install "${args[@]}" "$@"
}

# Helper function: pip install for build/infrastructure tools only.
# Uses default (only-if-needed) upgrade strategy to avoid cascading transitive
# dependency upgrades that can inadvertently break mlx / mlx-vlm.
pip_install_tool() {
	local args=("-U")
	[[ "${FORCE_REINSTALL:-0}" == "1" ]] && args+=("--force-reinstall")
	pip install "${args[@]}" "$@"
}

get_editable_project_location() {
	local package_name="$1"
	python -m pip show "$package_name" 2>/dev/null \
		| awk -F': ' '/^Editable project location: / {print $2; exit}'
}

normalize_path_or_echo() {
	local input_path="$1"
	if [[ -d "$input_path" ]]; then
		(
			cd "$input_path"
			pwd -P
		)
	else
		printf '%s\n' "$input_path"
	fi
}

verify_expected_editable_install() {
	local package_name="$1"
	local expected_repo_path="$2"
	local editable_path
	editable_path="$(get_editable_project_location "$package_name")"
	if [[ -z "$editable_path" ]]; then
		echo "❌ ERROR: Expected editable install for $package_name but none was found."
		return 1
	fi

	local expected_abs
	local editable_abs
	expected_abs="$(normalize_path_or_echo "$expected_repo_path")"
	editable_abs="$(normalize_path_or_echo "$editable_path")"

	if [[ "$editable_abs" != "$expected_abs" ]]; then
		echo "❌ ERROR: $package_name editable path mismatch."
		echo "   expected: $expected_abs"
		echo "   actual:   $editable_abs"
		return 1
	fi

	echo "✓ Verified editable install: $package_name"
}

get_installed_distribution_version() {
	local package_name="$1"
	python -m pip show "$package_name" 2>/dev/null \
		| awk -F': ' '/^Version: / {print $2; exit}'
}

get_git_current_branch() {
	local repo_path="$1"
	git -C "$repo_path" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"
}

get_git_short_sha() {
	local repo_path="$1"
	git -C "$repo_path" rev-parse --short HEAD 2>/dev/null || echo "unknown"
}

log_editable_install_provenance() {
	local package_name="$1"
	local repo_path="$2"
	local repo_abs
	local branch_name
	local short_sha
	local editable_path
	local editable_abs=""
	local installed_version

	repo_abs="$(normalize_path_or_echo "$repo_path")"
	branch_name="$(get_git_current_branch "$repo_path")"
	short_sha="$(get_git_short_sha "$repo_path")"
	editable_path="$(get_editable_project_location "$package_name")"
	installed_version="$(get_installed_distribution_version "$package_name")"

	if [[ -n "$editable_path" ]]; then
		editable_abs="$(normalize_path_or_echo "$editable_path")"
	fi

	echo "[update.sh] Local package provenance: $package_name"
	echo "   repo: $repo_abs"
	echo "   branch: $branch_name"
	echo "   commit: $short_sha"
	echo "   installed version: ${installed_version:-<unknown>}"
	if [[ -n "$editable_abs" ]]; then
		echo "   editable location: $editable_abs"
	fi
	if [[ -n "$editable_abs" && -n "$installed_version" && "$installed_version" != *".dev"* && "$installed_version" != *"+"* ]]; then
		echo "   note: editable local MLX repos may legitimately report release-style versions"
	fi
}

version_major_minor_at_least() {
	local actual="$1"
	local required_major="$2"
	local required_minor="$3"
	local actual_major="${actual%%.*}"
	local actual_rest="${actual#*.}"
	local actual_minor="${actual_rest%%.*}"

	if [[ ! "$actual_major" =~ ^[0-9]+$ ]]; then
		return 1
	fi
	if [[ ! "$actual_minor" =~ ^[0-9]+$ ]]; then
		actual_minor=0
	fi

	if [[ $actual_major -gt $required_major ]]; then
		return 0
	fi
	if [[ $actual_major -eq $required_major && $actual_minor -ge $required_minor ]]; then
		return 0
	fi
	return 1
}

log_mlx_runtime_provenance() {
	python - <<'PY'
from __future__ import annotations

from hashlib import sha256
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import mlx.core as mx


def dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except PackageNotFoundError:
        return "<not installed>"


core_path = Path(mx.__file__).resolve()
mlx_dir = core_path.parent
metallib = mlx_dir / "lib" / "mlx.metallib"
libmlx = mlx_dir / "lib" / "libmlx.dylib"
print("[update.sh] MLX runtime backend provenance:")
print(f"   mlx:       {dist_version('mlx')} ({core_path})")
print(f"   mlx-metal: {dist_version('mlx-metal')}")
for label, artifact in (("metallib", metallib), ("libmlx", libmlx)):
    if artifact.exists():
        digest = sha256(artifact.read_bytes()).hexdigest()[:16]
        size = artifact.stat().st_size
        print(f"   {label}:   {artifact} ({size} bytes, sha256={digest})")
    else:
        print(f"   {label}:   <missing at {artifact}>")
PY
}

huggingface_model_is_cached() {
	local model_id="$1"
	python - "$model_id" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import constants

model_id = sys.argv[1]
cache_name = "models--" + model_id.replace("/", "--")
snapshots = Path(constants.HF_HUB_CACHE) / cache_name / "snapshots"
if snapshots.exists() and any(snapshots.iterdir()):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

run_local_mlx_backend_smoke() {
	local smoke_mode="${MLX_LOCAL_BUILD_SMOKE:-auto}"
	local smoke_mode_normalized
	smoke_mode_normalized="$(printf '%s' "$smoke_mode" | tr '[:upper:]' '[:lower:]')"

	case "$smoke_mode_normalized" in
		0|false|no|off|skip)
			echo "[update.sh] Skipping local MLX runtime smoke (MLX_LOCAL_BUILD_SMOKE=$smoke_mode)"
			return 0
			;;
		1|true|yes|on|force|auto)
			;;
		*)
			echo "⚠️  Invalid MLX_LOCAL_BUILD_SMOKE='$smoke_mode'; using auto mode"
			smoke_mode_normalized="auto"
			;;
	esac

	local smoke_model="${MLX_LOCAL_BUILD_SMOKE_MODEL:-mlx-community/MiniCPM-V-4.6-8bit}"
	local smoke_prompt="${MLX_LOCAL_BUILD_SMOKE_PROMPT:-Hi}"
	local smoke_max_tokens="${MLX_LOCAL_BUILD_SMOKE_MAX_TOKENS:-10}"
	local smoke_expected="${MLX_LOCAL_BUILD_SMOKE_EXPECTED:-Hello! How can I help you today?}"

	if [[ "$smoke_mode_normalized" == "auto" ]]; then
		if ! huggingface_model_is_cached "$smoke_model"; then
			echo "[update.sh] Skipping local MLX runtime smoke; $smoke_model is not cached."
			echo "   To force it and allow model download: MLX_LOCAL_BUILD_SMOKE=1 bash tools/update.sh"
			return 0
		fi
	fi

	echo "[update.sh] Running local MLX runtime smoke with $smoke_model..."
	if python - "$smoke_model" "$smoke_prompt" "$smoke_max_tokens" "$smoke_expected" <<'PY'
from __future__ import annotations

import os
import subprocess
import sys

model, prompt, max_tokens, expected = sys.argv[1:5]
timeout = int(os.environ.get("MLX_LOCAL_BUILD_SMOKE_TIMEOUT", "240"))
cmd = [
    sys.executable,
    "-m",
    "mlx_vlm",
    "generate",
    "--model",
    model,
    "--prompt",
    prompt,
    "--max-tokens",
    max_tokens,
]
result = subprocess.run(
    cmd,
    capture_output=True,
    check=False,
    text=True,
    timeout=timeout,
)
output = f"{result.stdout}\n{result.stderr}"
print(output)
if result.returncode != 0:
    raise SystemExit(result.returncode)
if expected and expected not in output:
    print(
        "[update.sh] ERROR: local MLX runtime smoke did not produce the "
        "expected deterministic output.",
        file=sys.stderr,
    )
    print(f"[update.sh] Expected substring: {expected!r}", file=sys.stderr)
    print(
        "[update.sh] This usually points at a broken local MLX backend artifact "
        "such as mlx.metallib.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
	then
		echo "✓ Local MLX runtime smoke passed"
	else
		echo "❌ Local MLX runtime smoke failed"
		echo "   Set MLX_LOCAL_BUILD_SMOKE=0 to bypass only after confirming the local backend."
		return 1
	fi
}

run_metal_bug_reminder() {
	local reminder_mode="${MLX_METAL_BUG_REMINDER:-1}"
	local reminder_mode_normalized
	reminder_mode_normalized="$(printf '%s' "$reminder_mode" | tr '[:upper:]' '[:lower:]')"

	case "$reminder_mode_normalized" in
		0|false|no|off|skip)
			echo "[update.sh] Skipping MLX Metal backend regression reminder (MLX_METAL_BUG_REMINDER=$reminder_mode)"
			return 0
			;;
		1|true|yes|on)
			;;
		*)
			echo "⚠️  Invalid MLX_METAL_BUG_REMINDER='$reminder_mode'; running reminder"
			;;
	esac

	if [[ "$OSTYPE" != "darwin"* ]]; then
		echo "[update.sh] Non-macOS host; skipping MLX Metal backend regression reminder"
		return 0
	fi

	if [[ -f "$SCRIPT_DIR/bugtest.py" ]]; then
		echo ""
		echo "[update.sh] Checking MLX Metal backend regression reminder..."
		python "$SCRIPT_DIR/bugtest.py" --warn-only || true
	else
		echo "⚠️  bugtest.py not found at $SCRIPT_DIR/bugtest.py"
	fi
}

run_generate_stubs_command() {
	local script_dir="$1"
	shift
	local src_dir
	src_dir="$(cd "$script_dir/.." && pwd)"
	if command -v conda &> /dev/null && conda env list | grep -q "^mlx-vlm "; then
		(cd "$src_dir" && conda run -n mlx-vlm python -m tools.generate_stubs "$@")
	else
		(cd "$src_dir" && python -m tools.generate_stubs "$@")
	fi
}

# Ensure global Python packaging tools are current
# Use pip_install_tool (non-eager) to avoid cascading upgrades of shared deps
echo "[update.sh] Updating core Python packaging tools (pip, wheel, setuptools, build, pyrefly)..."
pip_install_tool pip wheel "setuptools>=80,<82" build pyrefly

# Resolve project extras once; the install itself runs after MLX updates so
# pyproject.toml performs the final dependency reconciliation.
INSTALL_GROUPS=".[dev,extras,torch]"
if [[ "${SKIP_TORCH:-0}" == "1" ]]; then
	INSTALL_GROUPS=".[dev,extras]"
	echo "[update.sh] Skipping torch group (SKIP_TORCH=1)"
else
	echo "[update.sh] Including torch group (default, set SKIP_TORCH=1 to skip)"
fi

reconcile_project_environment_from_pyproject() {
	echo ""
	echo "[update.sh] Reinstalling project from pyproject.toml after MLX updates..."
	pip_install -e "$PROJECT_ROOT/$INSTALL_GROUPS"

	echo ""
	echo "[update.sh] Running post-install dependency validation..."
	python -m pip check
	(
		cd "$PROJECT_ROOT"
		python -m tools.validate_env --expected-conda-env "${CONDA_DEFAULT_ENV:-mlx-vlm}"
	)
}

generate_project_stubs() {
	local ORIGINAL_DIR
	ORIGINAL_DIR="$(pwd)"
	cd "$SCRIPT_DIR"
	if [[ -f "$SCRIPT_DIR/generate_stubs.py" ]]; then
		echo "[update.sh] Generating type stubs for mlx_lm, mlx_vlm, transformers, and tokenizers..."
		if run_generate_stubs_command "$SCRIPT_DIR" mlx_lm mlx_vlm transformers tokenizers; then
			echo "✓ Project stubs generated successfully"
		else
			echo "⚠️  Failed to generate project stubs; verifying existing local stubs"
		fi

		if ! run_generate_stubs_command "$SCRIPT_DIR" --check --refresh-manifest-on-check mlx_lm mlx_vlm transformers tokenizers; then
			echo "❌ Project stub integrity verification failed"
			cd "$ORIGINAL_DIR"
			return 1
		fi
		echo "✓ Project stubs verified"
	fi
	cd "$ORIGINAL_DIR"
}

# Function to clean build artifacts from local MLX repositories
clean_local_mlx_builds() {
	local SCRIPT_DIR
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	local PARENT_DIR
	PARENT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
	local MLX_REPOS=("mlx" "mlx-lm" "mlx-vlm")
	
	echo "[update.sh] Cleaning build artifacts from local MLX repositories..."
	
	for repo in "${MLX_REPOS[@]}"; do
		local REPO_PATH="$PARENT_DIR/$repo"
		if [[ -d "$REPO_PATH/.git" ]]; then
			echo "[update.sh] Cleaning $repo build artifacts..."
			cd "$REPO_PATH"
			# Remove Python build artifacts
			rm -rf build/ dist/ ./*.egg-info/ .eggs/
			find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
			find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
			find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
			find . -name '*.pyc' -delete -o -name '*.pyo' -delete 2>/dev/null || true
			echo "  ✓ Cleaned $repo"
		fi
	done
	
	echo "[update.sh] Build artifact cleanup complete"
	echo ""
}

# Function to check system requirements for MLX builds
check_mlx_build_requirements() {
	local has_errors=0
	
	# Check for CMake and version
	if ! command -v cmake >/dev/null 2>&1; then
		echo "❌ ERROR: CMake not found. MLX requires CMake >= 3.25"
		echo "   Install with: brew install cmake"
		has_errors=1
	else
		local CMAKE_VERSION
		CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
		local CMAKE_MAJOR
		CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
		local CMAKE_MINOR
		CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
		
		if [[ $CMAKE_MAJOR -lt 3 ]] || [[ $CMAKE_MAJOR -eq 3 && $CMAKE_MINOR -lt 25 ]]; then
			echo "❌ ERROR: CMake $CMAKE_VERSION found, but MLX requires >= 3.25"
			echo "   Update with: brew upgrade cmake"
			has_errors=1
		else
			echo "✓ CMake $CMAKE_VERSION (>= 3.25 required)"
		fi
	fi
	
	# Check for macOS version (MLX PyPI requirement is >= 14.0)
	if [[ "$OSTYPE" == "darwin"* ]]; then
		local MACOS_VERSION
		MACOS_VERSION=$(sw_vers -productVersion)
		local MACOS_MAJOR
		MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)
		
		if [[ $MACOS_MAJOR -lt 14 ]]; then
			echo "⚠️  WARNING: macOS $MACOS_VERSION detected. MLX requires >= 14.0 (Sonoma)"
		else
			echo "✓ macOS $MACOS_VERSION (>= 14.0 required for MLX)"
		fi

		local XCODE_DEVELOPER_DIR
		if ! XCODE_DEVELOPER_DIR="$(xcode-select -p 2>/dev/null)"; then
			echo "❌ ERROR: xcode-select has no active developer directory"
			echo "   Fix with: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
			has_errors=1
		else
			echo "✓ Xcode developer directory: $XCODE_DEVELOPER_DIR"
		fi

		local XCODE_VERSION
		XCODE_VERSION="$(xcodebuild -version 2>/dev/null | awk '/^Xcode / {print $2; exit}' || true)"
		if [[ -z "$XCODE_VERSION" ]]; then
			echo "❌ ERROR: xcodebuild is unavailable; MLX source builds require Xcode >= 15"
			has_errors=1
		elif version_major_minor_at_least "$XCODE_VERSION" 15 0; then
			echo "✓ Xcode $XCODE_VERSION (>= 15.0 required for MLX source builds)"
		else
			echo "❌ ERROR: Xcode $XCODE_VERSION found, but MLX requires >= 15.0"
			has_errors=1
		fi

		local SDK_VERSION
		SDK_VERSION="$(xcrun -sdk macosx --show-sdk-version 2>/dev/null || true)"
		if [[ -z "$SDK_VERSION" ]]; then
			echo "❌ ERROR: Unable to resolve macOS SDK via xcrun"
			echo "   Verify with: xcrun -sdk macosx --show-sdk-version"
			has_errors=1
		elif version_major_minor_at_least "$SDK_VERSION" 14 0; then
			echo "✓ macOS SDK $SDK_VERSION (>= 14.0 required for MLX source builds)"
		else
			echo "❌ ERROR: macOS SDK $SDK_VERSION found, but MLX requires >= 14.0"
			has_errors=1
		fi

		local CLANG_VERSION
		CLANG_VERSION="$(
			xcrun clang --version 2>/dev/null \
				| awk '/clang version/ {for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+([.][0-9]+)*/) {print $i; exit}}' \
				|| true
		)"
		if [[ -z "$CLANG_VERSION" ]]; then
			echo "❌ ERROR: Unable to resolve Apple Clang via xcrun"
			has_errors=1
		elif version_major_minor_at_least "$CLANG_VERSION" 15 0; then
			echo "✓ Apple Clang $CLANG_VERSION (C++20 compiler requirement)"
		else
			echo "❌ ERROR: Apple Clang $CLANG_VERSION found, but MLX requires >= 15.0"
			has_errors=1
		fi

		if [[ "$(uname -m)" == "arm64" ]]; then
			echo "✓ Native arm64 shell detected"
		else
			echo "❌ ERROR: local MLX Metal builds must run from a native arm64 shell"
			echo "   Open a native terminal instead of a translated x86_64/Rosetta shell."
			has_errors=1
		fi

		if ! command -v xcrun >/dev/null 2>&1; then
			echo "❌ ERROR: xcrun not found. Install Xcode command line tools"
			has_errors=1
		else
			local METAL_BIN
			local METALLIB_BIN
			METAL_BIN="$(xcrun -f metal 2>/dev/null || true)"
			METALLIB_BIN="$(xcrun -f metallib 2>/dev/null || true)"

			if [[ -z "$METAL_BIN" || -z "$METALLIB_BIN" ]]; then
				echo "❌ ERROR: Metal toolchain is not available through xcrun"
				echo "   Open Xcode once or run: xcodebuild -runFirstLaunch"
				echo "   Then verify: xcrun -f metal && xcrun -f metallib"
				has_errors=1
			else
				echo "✓ Metal toolchain detected:"
				echo "   metal:    $METAL_BIN"
				echo "   metallib: $METALLIB_BIN"
				xcrun metal --version 2>/dev/null | sed 's/^/   metal version: /' || true
				xcrun metallib --version 2>/dev/null | sed 's/^/   metallib version: /' || true
			fi
		fi
	fi
	
	return $has_errors
}

# Function to update local MLX development repositories
update_local_mlx_repos() {
	# Determine the parent directory (assuming check_models/src/tools/update.sh)
	local SCRIPT_DIR
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	local PARENT_DIR
	PARENT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
	local ORIGINAL_DIR
	ORIGINAL_DIR="$(pwd)"
	
	echo "[update.sh] Checking for local MLX development repositories..."
	
	# Define repositories in dependency order: mlx first (base), then mlx-lm, then mlx-vlm
	local MLX_REPOS=("mlx" "mlx-lm" "mlx-vlm")
	local -a REPO_NAMES=()
	local -a REPO_PATHS=()
	local -a REPO_SKIP=()
	
	for repo in "${MLX_REPOS[@]}"; do
		local REPO_PATH="$PARENT_DIR/$repo"
		if [[ -d "$REPO_PATH/.git" ]]; then
			REPO_NAMES+=("$repo")
			REPO_PATHS+=("$REPO_PATH")
			REPO_SKIP+=(0)
		fi
	done
	
	if [[ ${#REPO_NAMES[@]} -eq 0 ]]; then
		echo "[update.sh] No local MLX development repositories found"
		echo "[update.sh] (Looked for mlx-lm, mlx-vlm, mlx directories with .git at $PARENT_DIR)"
		cd "$ORIGINAL_DIR"
		return 1
	fi
	
	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "📦 Updating local MLX repositories: ${REPO_NAMES[*]}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	
	# Check build requirements if mlx is present
	for repo in "${REPO_NAMES[@]}"; do
		if [[ "$repo" == "mlx" ]]; then
			echo ""
			echo "Verifying build requirements for MLX..."
			if ! check_mlx_build_requirements; then
				echo ""
				echo "❌ Build requirements not met. Cannot build MLX."
				cd "$ORIGINAL_DIR"
				return 1
			fi
			echo ""
			break
		fi
	done

	# Stage 1: Sync all repositories first
	echo ""
	echo "Stage 1: Syncing repositories with git pull..."
	for idx in "${!REPO_NAMES[@]}"; do
		cd "${REPO_PATHS[idx]}"
		echo "[update.sh] ($((idx + 1))/${#REPO_NAMES[@]}) git pull -> ${REPO_NAMES[idx]}"
		if git pull; then
			echo "✓ Git pull successful for ${REPO_NAMES[idx]}"
		else
			echo "⚠️  Git pull failed for ${REPO_NAMES[idx]} — skipping build"
			REPO_SKIP[idx]=1
		fi
		echo ""
	done
	cd "$ORIGINAL_DIR"

	# Stage 2: Install all Python library dependencies from requirements.txt
	echo ""
	echo "Stage 2: Installing library dependencies (requirements.txt files)..."
	for idx in "${!REPO_NAMES[@]}"; do
		[[ ${REPO_SKIP[idx]} -eq 1 ]] && continue
		cd "${REPO_PATHS[idx]}"
		if [[ -f "requirements.txt" ]]; then
			echo "[update.sh] Installing requirements for ${REPO_NAMES[idx]}..."
			pip_install -U -r requirements.txt
		fi
		echo ""
	done
	cd "$ORIGINAL_DIR"

	# Stage 3: Verify repository integrity before building
	echo ""
	echo "Stage 3: Verifying repository integrity..."
	for idx in "${!REPO_NAMES[@]}"; do
		[[ ${REPO_SKIP[idx]} -eq 1 ]] && continue
		cd "${REPO_PATHS[idx]}"
		
		# Check for missing git-tracked files
		local MISSING_FILES
		MISSING_FILES=$(git ls-files --deleted 2>/dev/null)
		if [[ -n "$MISSING_FILES" ]]; then
			local FILE_COUNT
			FILE_COUNT=$(echo "$MISSING_FILES" | wc -l | tr -d ' ')
			echo "❌ ERROR: Repository ${REPO_NAMES[idx]} is corrupt - $FILE_COUNT missing tracked file(s)"
			echo "$MISSING_FILES" | head -3
			echo ""
			echo "Fix with: cd ${REPO_PATHS[idx]} && git restore ."
			exit 1
		fi
		
		# Warn about uncommitted changes
		git diff --quiet HEAD 2>/dev/null || echo "⚠️  Warning: ${REPO_NAMES[idx]} has uncommitted changes"
	done
	echo "✓ All repositories verified"
	cd "$ORIGINAL_DIR"

	# Stage 4: Build and install packages in dependency order (mlx → mlx-lm → mlx-vlm)
	echo ""
	echo "Stage 4: Building and installing MLX packages in dependency order..."
	for idx in "${!REPO_NAMES[@]}"; do
		[[ ${REPO_SKIP[idx]} -eq 1 ]] && continue
		cd "${REPO_PATHS[idx]}"
		echo "[update.sh] Installing ${REPO_NAMES[idx]} package..."
		
		# Record the currently installed version so we can restore on failure
		local CURRENT_VERSION
		CURRENT_VERSION=$(pip show "${REPO_NAMES[idx]}" 2>/dev/null | grep '^Version:' | awk '{print $2}' || echo "")

		PREV_CMAKE_ARGS=""
		CMAKE_ARGS_WAS_SET=0
		if [[ -n "${CMAKE_ARGS:-}" ]]; then
			PREV_CMAKE_ARGS="$CMAKE_ARGS"
			CMAKE_ARGS_WAS_SET=1
		fi

		# MLX build controls are passed to pip via CMAKE_ARGS.
		if [[ "${REPO_NAMES[idx]}" == "mlx" ]]; then
			# Install MLX build dependencies.
			# Use pip_install_tool to avoid eager transitive upgrades.
			echo "[update.sh] Installing MLX build dependencies..."
			pip_install_tool cmake
			pip_install_tool "setuptools>=80,<82"
			pip_install_tool typing_extensions

			local JIT_SETTING_RAW="${MLX_METAL_JIT:-}"
			local JIT_SETTING=""
			if [[ -n "$JIT_SETTING_RAW" ]]; then
				JIT_SETTING="$(printf '%s' "$JIT_SETTING_RAW" | tr '[:lower:]' '[:upper:]')"
				if [[ "$JIT_SETTING" == "ON" || "$JIT_SETTING" == "OFF" ]]; then
					if [[ -n "${CMAKE_ARGS:-}" ]]; then
						export CMAKE_ARGS="$CMAKE_ARGS -DMLX_METAL_JIT=$JIT_SETTING"
					else
						export CMAKE_ARGS="-DMLX_METAL_JIT=$JIT_SETTING"
					fi
					echo "[update.sh] Installing mlx with explicit MLX_METAL_JIT=$JIT_SETTING (CMAKE_ARGS=$CMAKE_ARGS)"
				else
					echo "⚠️  Invalid MLX_METAL_JIT='$JIT_SETTING_RAW'; ignoring and using MLX default (OFF)"
					echo "[update.sh] Installing mlx with MLX_METAL_JIT default (CMake default OFF)"
				fi
			else
				echo "[update.sh] Installing mlx with MLX_METAL_JIT default (CMake default OFF)"
			fi
		fi
		
		# Install package (works for both mlx and others)
		# IMPORTANT: pip install -e . first uninstalls the existing package, THEN
		# attempts the build. If the build fails, the package is left uninstalled.
		# We detect this and restore from PyPI as a fallback.
		if [[ "${REPO_NAMES[idx]}" == "mlx" ]]; then
			echo "[update.sh] Using upstream MLX editable dev install with verbose pip output (-v)"
			INSTALL_CMD=(pip_install_verbose -e ".[dev]")
		else
			INSTALL_CMD=(pip_install -e .)
		fi

		if "${INSTALL_CMD[@]}"; then
			echo "✓ ${REPO_NAMES[idx]} installed successfully"
		else
			echo "⚠️  Failed to install ${REPO_NAMES[idx]}"
			REPO_SKIP[idx]=1
			# Check if pip uninstalled the previous version during the failed attempt
			if ! pip show "${REPO_NAMES[idx]}" >/dev/null 2>&1; then
				echo "⚠️  ${REPO_NAMES[idx]} is no longer installed! Attempting PyPI fallback..."
				if [[ -n "$CURRENT_VERSION" ]]; then
					pip install "${REPO_NAMES[idx]}==$CURRENT_VERSION" 2>/dev/null \
						|| pip install "${REPO_NAMES[idx]}" 2>/dev/null \
						|| echo "❌ CRITICAL: Could not restore ${REPO_NAMES[idx]}. Run: pip install ${REPO_NAMES[idx]}"
				else
					pip install "${REPO_NAMES[idx]}" 2>/dev/null \
						|| echo "❌ CRITICAL: Could not restore ${REPO_NAMES[idx]}. Run: pip install ${REPO_NAMES[idx]}"
				fi
			fi
		fi

		if [[ "${REPO_NAMES[idx]}" == "mlx" ]]; then
			if [[ $CMAKE_ARGS_WAS_SET -eq 1 ]]; then
				export CMAKE_ARGS="$PREV_CMAKE_ARGS"
			else
				unset CMAKE_ARGS
			fi
		fi

		[[ ${REPO_SKIP[idx]} -eq 1 ]] && continue
	
		# Stubs are generated automatically by the build process now
		# if [[ "${REPO_NAMES[idx]}" == "mlx" ]]; then
		# 	echo "[update.sh] (MLX stubs are generated automatically during build)"
		# fi
		echo ""
	done
	cd "$ORIGINAL_DIR"

	# Stage 4b: Verify editable origins for mlx/mlx-lm/mlx-vlm.
	# These packages often use release-style version strings even for local builds,
	# so location metadata is a more reliable signal than version text.
	echo ""
	echo "Stage 4b: Verifying editable install origins for mlx/mlx-lm/mlx-vlm..."
	for idx in "${!REPO_NAMES[@]}"; do
		[[ ${REPO_SKIP[idx]} -eq 1 ]] && continue
		case "${REPO_NAMES[idx]}" in
			mlx|mlx-lm|mlx-vlm)
				if ! verify_expected_editable_install "${REPO_NAMES[idx]}" "${REPO_PATHS[idx]}"; then
					echo "❌ Local build verification failed for ${REPO_NAMES[idx]}"
					cd "$ORIGINAL_DIR"
					return 1
				fi
				log_editable_install_provenance "${REPO_NAMES[idx]}" "${REPO_PATHS[idx]}"
				;;
		esac
	done

	for idx in "${!REPO_NAMES[@]}"; do
		if [[ "${REPO_NAMES[idx]}" == "mlx" && ${REPO_SKIP[idx]} -eq 0 ]]; then
			LOCAL_MLX_READY=1
			break
		fi
	done

	cd "$ORIGINAL_DIR"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "✓ Local MLX repositories updated"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""
	return 0
}

# Clean build artifacts if requested
if [[ "${CLEAN_BUILD:-0}" == "1" ]]; then
	clean_local_mlx_builds
fi

# Determine if we should skip PyPI MLX updates
SKIP_MLX_PYPI=0
LOCAL_MLX_READY=0

# Check for local MLX repos
if update_local_mlx_repos; then
	SKIP_MLX_PYPI=1
	echo "[update.sh] Local MLX builds updated - skipping PyPI updates"
else
	# Check if MLX is a dev build (contains .dev or +commit)
	MLX_VERSION=$(pip show mlx 2>/dev/null | grep '^Version:' | awk '{print $2}' || echo "")
	if [[ "$MLX_VERSION" == *".dev"* ]] || [[ "$MLX_VERSION" == *"+"* ]]; then
		SKIP_MLX_PYPI=1
		echo "[update.sh] Detected local MLX build: $MLX_VERSION — preserving..."
	fi

	# mlx-lm / mlx-vlm local builds can keep release-like version strings, so
	# also preserve when pip metadata marks them as editable installs.
	for pkg in mlx-lm mlx-vlm; do
		EDITABLE_LOC="$(get_editable_project_location "$pkg")"
		if [[ -n "$EDITABLE_LOC" ]]; then
			SKIP_MLX_PYPI=1
			echo "[update.sh] Detected editable $pkg install at $EDITABLE_LOC — preserving..."
		fi
	done
fi

# Update MLX from PyPI if not skipped
if [[ "${SKIP_MLX:-0}" == "1" ]] || [[ $SKIP_MLX_PYPI -eq 1 ]]; then
	if [[ "${SKIP_MLX:-0}" == "1" ]]; then
		echo "[update.sh] Skipping PyPI MLX updates (SKIP_MLX=1 environment variable set)"
	elif [[ $SKIP_MLX_PYPI -eq 1 ]]; then
		echo "[update.sh] Skipping PyPI MLX updates (using local development builds)"
		# Remove stale mlx-metal from PyPI when using local builds
		# (local builds compile their own Metal backend)
		if pip show mlx-metal >/dev/null 2>&1; then
			echo "[update.sh] Removing stale mlx-metal PyPI package (not needed with local builds)..."
			pip uninstall -y mlx-metal || true
		fi
	fi
else
	echo "[update.sh] Updating MLX packages from PyPI to latest..."
	# Explicitly upgrade MLX ecosystem from PyPI, triggering eager transitive upgrades
	# mlx-metal is the Metal GPU backend - must be explicitly installed
	pip_install mlx mlx-metal mlx-lm mlx-vlm
fi

reconcile_project_environment_from_pyproject
generate_project_stubs

if [[ $LOCAL_MLX_READY -eq 1 ]]; then
	log_mlx_runtime_provenance
	if ! run_local_mlx_backend_smoke; then
		exit 1
	fi
fi

echo "[update.sh] Done."

# Post-flight check: verify critical packages are still importable
echo ""
echo "[update.sh] Verifying critical packages..."
IMPORT_TO_PIP_PACKAGE() {
	case "$1" in
		mlx_lm)
			printf '%s\n' "mlx-lm"
			;;
		mlx_vlm)
			printf '%s\n' "mlx-vlm"
			;;
		*)
			printf '%s\n' "$1"
			;;
	esac
}
MISSING_PKGS=()
REPAIR_PKGS=()
IMPORT_CHECK_DIR="$(mktemp -d)"
for pkg in mlx mlx_lm mlx_vlm; do
	import_log="$IMPORT_CHECK_DIR/$pkg.log"
	if ! python -c "import $pkg" > /dev/null 2>"$import_log"; then
		MISSING_PKGS+=("$pkg")
		REPAIR_PKGS+=("$(IMPORT_TO_PIP_PACKAGE "$pkg")")
	else
		rm -f "$import_log"
	fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
	echo "❌ WARNING: The following critical packages are NOT importable after update:"
	printf '   - %s\n' "${MISSING_PKGS[@]}"
	echo ""
	echo "   Import failure details:"
	for pkg in "${MISSING_PKGS[@]}"; do
		import_log="$IMPORT_CHECK_DIR/$pkg.log"
		echo "   [$pkg]"
		if [[ -s "$import_log" ]]; then
			tail -n 20 "$import_log" | sed 's/^/     /'
		else
			echo "     <no stderr captured>"
		fi
	done
	echo ""
	echo "   This may cause 'Missing required runtime dependencies' errors."
	echo "   Fix with: pip install ${REPAIR_PKGS[*]}"
	echo ""
	rm -rf "$IMPORT_CHECK_DIR"
	exit 1
else
	rm -rf "$IMPORT_CHECK_DIR"
	echo "✓ All critical packages verified (mlx, mlx_lm, mlx_vlm)"
fi

run_metal_bug_reminder

# Final check for held-back / outdated packages
echo ""
echo "[update.sh] Checking for held-back packages (orphaned or constrained)..."
if [[ -f "$SCRIPT_DIR/check_outdated.py" ]]; then
	python "$SCRIPT_DIR/check_outdated.py" || true
else
	echo "⚠️  check_outdated.py not found at $SCRIPT_DIR/check_outdated.py"
fi
