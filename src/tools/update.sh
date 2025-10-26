#!/usr/bin/env bash
# Unified dependency updater for mlx-vlm-check project.
# Installs/updates the project and all dependencies from pyproject.toml.
#
# Execution Order:
#   1. Update Homebrew (if available)
#   2. Update pip/wheel/setuptools
#   3. Install project in editable mode with all dependencies (dev + extras)
#   4. Then update local MLX repos (if present) OR update from PyPI
#
# Usage examples:
#   ./update.sh                       # Install project + dev + extras
#   INSTALL_TORCH=1 ./update.sh       # Additionally install torch group
#   FORCE_REINSTALL=1 ./update.sh     # Force reinstall with --force-reinstall
#   SKIP_MLX=1 ./update.sh            # Force skip mlx/mlx-vlm updates (override detection)
#
# Local MLX Development:
#   If mlx, mlx-lm, and mlx-vlm directories exist at ../../ (sibling to scripts/),
#   the script will automatically (AFTER installing all library dependencies):
#   1. Run git pull in each repository
#   2. Install requirements.txt (if present) for additional dependencies
#   3. Install packages in dependency order: mlx → mlx-lm → mlx-vlm
#   4. Generate type stubs (mlx: python setup.py generate_stubs)
#   5. Generate stubs for this project (mlx_vlm, tokenizers)
#   6. Skip PyPI updates for these packages
#
# Note: Script automatically detects and preserves local MLX dev builds (versions
# containing .dev or +commit). Stable releases are updated normally.

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

# Homebrew updates first (if available)
if command -v brew >/dev/null 2>&1; then
	echo "[update.sh] Updating Homebrew..."
	brew update
	brew upgrade
else
	echo "[update.sh] Homebrew not found; skipping brew update/upgrade"
fi

# Ensure global Python packaging tools are current
echo "[update.sh] Updating core Python packaging tools (pip, wheel, setuptools)..."
pip install -U pip wheel setuptools

# Determine project root (assuming scripts/src/tools/update.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXTRA_ARGS=()
if [[ "${FORCE_REINSTALL:-0}" == "1" ]]; then
	EXTRA_ARGS+=("--force-reinstall")
fi

# Install project with all dependencies from pyproject.toml
# This installs: runtime + extras + dev in editable mode
echo "[update.sh] Installing project with all dependencies from pyproject.toml..."
INSTALL_GROUPS=".[dev,extras]"
if [[ "${INSTALL_TORCH:-0}" == "1" ]]; then
	INSTALL_GROUPS=".[dev,extras,torch]"
	echo "[update.sh] Including torch group (INSTALL_TORCH=1)"
fi

if ((${#EXTRA_ARGS[@]})); then
	pip install -U --upgrade-strategy eager "${EXTRA_ARGS[@]}" -e "$PROJECT_ROOT/$INSTALL_GROUPS"
else
	pip install -U --upgrade-strategy eager -e "$PROJECT_ROOT/$INSTALL_GROUPS"
fi

# Function to update local MLX development repositories
update_local_mlx_repos() {
	# Determine the parent directory (assuming scripts/src/tools/update.sh)
	local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	local PARENT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
	local ORIGINAL_DIR="$(pwd)"
	
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
		if [[ ${REPO_SKIP[idx]} -eq 1 ]]; then
			continue
		fi
		cd "${REPO_PATHS[idx]}"
		if [[ -f "requirements.txt" ]]; then
			echo "[update.sh] Installing requirements for ${REPO_NAMES[idx]}..."
			if [[ "${FORCE_REINSTALL:-0}" == "1" ]]; then
				pip install -U --force-reinstall -r requirements.txt
			else
				pip install -U -r requirements.txt
			fi
		else
			echo "[update.sh] No requirements.txt for ${REPO_NAMES[idx]}"
		fi
		if [[ "${REPO_NAMES[idx]}" == "mlx-vlm" ]]; then
			echo "[update.sh] Installing opencv-python for mlx-vlm..."
			if [[ "${FORCE_REINSTALL:-0}" == "1" ]]; then
				pip install -U --force-reinstall opencv-python
			else
				pip install -U opencv-python
			fi
		fi
		echo ""
	done
	cd "$ORIGINAL_DIR"

	# Stage 3: Build and install packages in dependency order (mlx → mlx-lm → mlx-vlm)
	echo ""
	echo "Stage 3: Building and installing MLX packages in dependency order..."
	for idx in "${!REPO_NAMES[@]}"; do
		if [[ ${REPO_SKIP[idx]} -eq 1 ]]; then
			continue
		fi
		cd "${REPO_PATHS[idx]}"
		echo "[update.sh] Installing ${REPO_NAMES[idx]} package..."
		local INSTALL_STATUS=0
		if [[ "${FORCE_REINSTALL:-0}" == "1" ]]; then
			pip install --force-reinstall -e . || INSTALL_STATUS=$?
		else
			pip install -e . || INSTALL_STATUS=$?
		fi
		if [[ $INSTALL_STATUS -eq 0 ]]; then
			echo "✓ ${REPO_NAMES[idx]} installed successfully"
		else
			echo "⚠️  Failed to install ${REPO_NAMES[idx]}"
			REPO_SKIP[idx]=1
		fi
	
		if [[ "${REPO_NAMES[idx]}" == "mlx" ]] && [[ ${REPO_SKIP[idx]} -eq 0 ]]; then
			echo "[update.sh] Generating type stubs for mlx..."
			if python setup.py generate_stubs; then
				echo "✓ MLX stubs generated successfully"
			else
				echo "⚠️  Failed to generate MLX stubs (non-fatal)"
			fi
		fi
		echo ""
	done
	cd "$ORIGINAL_DIR"

	# Stage 4: Generate project stubs if applicable
	cd "$SCRIPT_DIR"
	if [[ -f "$SCRIPT_DIR/generate_stubs.py" ]]; then
		echo "[update.sh] Generating type stubs for mlx_vlm and tokenizers..."
		if python generate_stubs.py mlx_vlm tokenizers; then
			echo "✓ Project stubs generated successfully"
		else
			echo "⚠️  Failed to generate project stubs (non-fatal)"
		fi
	fi

	cd "$ORIGINAL_DIR"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "✓ Local MLX repositories updated"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo ""
	return 0
}

# Update local MLX repos if they exist
LOCAL_MLX_UPDATED=0
if update_local_mlx_repos; then
	LOCAL_MLX_UPDATED=1
	echo "[update.sh] Local MLX builds updated - will skip PyPI MLX packages"
fi

# Check if MLX is a local dev build (contains .dev or +commit in version)
MLX_VERSION=$(pip show mlx 2>/dev/null | grep '^Version:' | awk '{print $2}' || echo "")
MLX_IS_LOCAL=0
if [[ "$MLX_VERSION" == *".dev"* ]] || [[ "$MLX_VERSION" == *"+"* ]]; then
	MLX_IS_LOCAL=1
	echo "[update.sh] Detected local MLX build: $MLX_VERSION — preserving..."
fi

# If we just updated local repos, treat as local build
if [[ $LOCAL_MLX_UPDATED -eq 1 ]]; then
	MLX_IS_LOCAL=1
fi

# Only update MLX from PyPI if not using local builds
if [[ "${SKIP_MLX:-0}" != "1" ]] && [[ "$MLX_IS_LOCAL" != "1" ]]; then
	echo "[update.sh] Updating MLX packages from PyPI..."
	MLX_PYPI_PACKAGES=("mlx" "mlx-vlm" "mlx-lm")
	if ((${#EXTRA_ARGS[@]})); then
		pip install -U --upgrade-strategy eager "${EXTRA_ARGS[@]}" "${MLX_PYPI_PACKAGES[@]}"
	else
		pip install -U --upgrade-strategy eager "${MLX_PYPI_PACKAGES[@]}"
	fi
elif [[ "${SKIP_MLX:-0}" == "1" ]]; then
	echo "[update.sh] Skipping mlx/mlx-vlm/mlx-lm updates (SKIP_MLX=1)..."
fi

echo "[update.sh] Done."
