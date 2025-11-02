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
#   ./update.sh                       # Install project + dev + extras (MLX_METAL_JIT=OFF by default)
#   INSTALL_TORCH=1 ./update.sh       # Additionally install torch group
#   FORCE_REINSTALL=1 ./update.sh     # Force reinstall with --force-reinstall
#   SKIP_MLX=1 ./update.sh            # Force skip mlx/mlx-vlm updates (override detection)
#   MLX_METAL_JIT=ON ./update.sh      # Enable Metal JIT for smaller binaries (defaults to OFF)
#   CLEAN_BUILD=1 ./update.sh         # Clean build artifacts before building local MLX repos
#
# Local MLX Development:
#   If mlx, mlx-lm, and mlx-vlm directories exist at ../../ (sibling to scripts/),
#   the script will automatically (AFTER installing all library dependencies):
#   1. Run git pull in each repository
#   2. Install requirements.txt (if present) for additional dependencies
#      (Note: mlx requires nanobind==2.4.0 and setuptools>=80 for dev builds)
#   3. Install packages in dependency order: mlx → mlx-lm → mlx-vlm
#   4. Generate type stubs (mlx: python setup.py generate_stubs)
#   5. Generate stubs for this project (mlx_vlm, tokenizers)
#   6. Skip PyPI updates for these packages
#
# Requirements for local MLX builds:
#   - CMake >= 3.27
#   - Xcode >= 15.0 (macOS)
#   - macOS >= 13.5 (for Metal backend)
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

# Helper function: pip install with optional force reinstall
pip_install() {
	local args=("-U" "--upgrade-strategy" "eager")
	[[ "${FORCE_REINSTALL:-0}" == "1" ]] && args+=("--force-reinstall")
	pip install "${args[@]}" "$@"
}

# Install project with all dependencies from pyproject.toml
INSTALL_GROUPS=".[dev,extras]"
if [[ "${INSTALL_TORCH:-0}" == "1" ]]; then
	INSTALL_GROUPS=".[dev,extras,torch]"
	echo "[update.sh] Including torch group (INSTALL_TORCH=1)"
fi

echo "[update.sh] Installing project with all dependencies from pyproject.toml..."
pip_install -e "$PROJECT_ROOT/$INSTALL_GROUPS"

# Function to clean build artifacts from local MLX repositories
clean_local_mlx_builds() {
	local SCRIPT_DIR
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	local PARENT_DIR
	PARENT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
	local MLX_REPOS=("mlx" "mlx-lm" "mlx-vlm" "mlx-data")
	
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
		echo "❌ ERROR: CMake not found. MLX requires CMake >= 3.27"
		echo "   Install with: brew install cmake"
		has_errors=1
	else
		local CMAKE_VERSION
		CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
		local CMAKE_MAJOR
		CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
		local CMAKE_MINOR
		CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
		
		if [[ $CMAKE_MAJOR -lt 3 ]] || [[ $CMAKE_MAJOR -eq 3 && $CMAKE_MINOR -lt 27 ]]; then
			echo "❌ ERROR: CMake $CMAKE_VERSION found, but MLX requires >= 3.27"
			echo "   Update with: brew upgrade cmake"
			has_errors=1
		else
			echo "✓ CMake $CMAKE_VERSION (>= 3.27 required)"
		fi
	fi
	
	# Check for macOS version (Metal backend requirement)
	if [[ "$OSTYPE" == "darwin"* ]]; then
		local MACOS_VERSION
		MACOS_VERSION=$(sw_vers -productVersion)
		local MACOS_MAJOR
		MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)
		local MACOS_MINOR
		MACOS_MINOR=$(echo "$MACOS_VERSION" | cut -d. -f2)
		
		if [[ $MACOS_MAJOR -lt 13 ]] || [[ $MACOS_MAJOR -eq 13 && $MACOS_MINOR -lt 5 ]]; then
			echo "⚠️  WARNING: macOS $MACOS_VERSION detected. MLX recommends >= 13.5 for Metal backend"
		else
			echo "✓ macOS $MACOS_VERSION (>= 13.5 required for Metal)"
		fi
	fi
	
	return $has_errors
}

# Function to update local MLX development repositories
update_local_mlx_repos() {
	# Determine the parent directory (assuming scripts/src/tools/update.sh)
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
			pip_install -r requirements.txt
		fi
		if [[ "${REPO_NAMES[idx]}" == "mlx-vlm" ]]; then
			echo "[update.sh] Installing opencv-python for mlx-vlm..."
			pip_install opencv-python
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
		
		# MLX requires CMake configuration before pip install
		if [[ "${REPO_NAMES[idx]}" == "mlx" ]]; then
			[[ "${CLEAN_BUILD:-0}" == "1" ]] && rm -rf build
			
			local JIT_SETTING="${MLX_METAL_JIT:-OFF}"
			echo "[update.sh] Building mlx with MLX_METAL_JIT=$JIT_SETTING"
			
			if [[ "$JIT_SETTING" == "ON" ]]; then
				env MLX_METAL_JIT=ON cmake -S . -B build || { REPO_SKIP[idx]=1; continue; }
			else
				cmake -S . -B build || { REPO_SKIP[idx]=1; continue; }
			fi
			
			cmake --build build || { REPO_SKIP[idx]=1; continue; }
			echo "[update.sh] Installing MLX Python bindings..."
		fi
		
		# Install package (works for both mlx and others)
		if pip_install -e .; then
			echo "✓ ${REPO_NAMES[idx]} installed successfully"
		else
			echo "⚠️  Failed to install ${REPO_NAMES[idx]}"
			REPO_SKIP[idx]=1
			continue
		fi
	
		# Generate stubs for mlx
		if [[ "${REPO_NAMES[idx]}" == "mlx" ]]; then
			echo "[update.sh] Generating type stubs for mlx..."
			
			# Verify setuptools is installed (required for stub generation)
			if ! python -c "import setuptools; exit(0 if tuple(map(int, setuptools.__version__.split('.'))) >= (80, 0, 0) else 1)" 2>/dev/null; then
				echo "[update.sh] Installing setuptools>=80 (required for stub generation)..."
				pip_install "setuptools>=80"
			fi
			
			if python setup.py generate_stubs; then
				echo "✓ MLX stubs generated"
			else
				echo "⚠️  Stub generation failed (non-fatal)"
			fi
		fi
		echo ""
	done
	cd "$ORIGINAL_DIR"

	# Stage 5: Generate project stubs if applicable
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

# Clean build artifacts if requested
if [[ "${CLEAN_BUILD:-0}" == "1" ]]; then
	clean_local_mlx_builds
fi

# Determine if we should skip PyPI MLX updates
SKIP_MLX_PYPI=0

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
fi

# Update MLX from PyPI if not skipped
if [[ "${SKIP_MLX:-0}" == "1" ]] || [[ $SKIP_MLX_PYPI -eq 1 ]]; then
	[[ "${SKIP_MLX:-0}" == "1" ]] && echo "[update.sh] Skipping MLX updates (SKIP_MLX=1)"
else
	echo "[update.sh] Updating MLX packages from PyPI..."
	pip_install mlx mlx-vlm mlx-lm
fi

echo "[update.sh] Done."
