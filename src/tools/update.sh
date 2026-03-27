#!/usr/bin/env bash
# Unified dependency updater for mlx-vlm-check project.
# Installs/updates the project and all dependencies from pyproject.toml.
#
# Execution Order:
#   1. Update conda (if in conda environment)
#   2. Update Homebrew (if available)
#   3. Update pip/wheel/setuptools
#   4. Install project in editable mode with all dependencies (dev + extras)
#   5. Then update local MLX repos (if present) OR update from PyPI
#
# Usage examples:
#   ./update.sh                       # Install project + dev + extras + torch (MLX_METAL_JIT=OFF by default)
#   SKIP_TORCH=1 ./update.sh          # Skip torch group installation
#   FORCE_REINSTALL=1 ./update.sh     # Force reinstall with --force-reinstall
#   SKIP_MLX=1 ./update.sh            # Force skip mlx/mlx-vlm updates (override detection)
#   CONDA_UPDATE_ALL=1 ./update.sh    # Force conda update --all even with pip conflicts
#   MLX_METAL_JIT=ON ./update.sh      # Build MLX with runtime Metal kernel compilation
#   CLEAN_BUILD=1 ./update.sh         # Clean build artifacts before building local MLX repos
#
# Local MLX Development:
#   If mlx, mlx-lm, and mlx-vlm directories exist at ../../ (sibling to check_models/),
#   the script will automatically (AFTER installing all library dependencies):
#   1. Run git pull in each repository
#   2. Install requirements.txt (if present) for additional dependencies
#      (Note: mlx requires setuptools>=80 and typing_extensions for builds)
#   3. Install packages in dependency order: mlx → mlx-lm → mlx-vlm
#   5. Verify editable install origins for mlx, mlx-lm, and mlx-vlm
#   6. Generate stubs for this project (mlx_lm, mlx_vlm, transformers, tokenizers)
#   7. Skip PyPI updates for these packages
#
# Requirements for local MLX builds:
#   - CMake >= 3.25 (MLX minimum requirement as of 2025)
#   - Xcode >= 15.0 (macOS, for Metal support)
#   - macOS >= 14.0 (MLX PyPI requirement)
#   - typing_extensions (required by MLX build)
#   - setuptools>=80 (required for stub generation)
#
# Note:
# - If MLX_METAL_JIT is set, update.sh maps it directly to CMake option
#   MLX_METAL_JIT (ON/OFF). If unset, MLX's CMake default is used (OFF).
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

# Update conda itself (base) and environment packages with pip-conflict safety check.
# NOTE: conda update --all can break pip-installed packages (mlx, mlx-vlm) by
# reshuffling shared dependencies like numpy. We do a dry-run first and warn.
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

# Homebrew updates first (if available)
if command -v brew >/dev/null 2>&1; then
	echo "[update.sh] Updating Homebrew..."
	brew update
	brew upgrade
else
	echo "[update.sh] Homebrew not found; skipping brew update/upgrade"
fi

# Determine project root (assuming check_models/src/tools/update.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Repo-local markdownlint tools (if npm is available)
if command -v npm >/dev/null 2>&1; then
	echo "[update.sh] Installing repo-local markdownlint tools..."
	npm install --prefix "$PROJECT_ROOT"
	echo "[update.sh] repo-local npm tooling is up to date"
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

	echo "✓ Verified editable install: $package_name -> $editable_abs"
}

# Ensure global Python packaging tools are current
# Use pip_install_tool (non-eager) to avoid cascading upgrades of shared deps
echo "[update.sh] Updating core Python packaging tools (pip, wheel, setuptools, build, pyrefly)..."
pip_install_tool pip wheel setuptools build pyrefly

# Install project with all dependencies from pyproject.toml
INSTALL_GROUPS=".[dev,extras,torch]"
if [[ "${SKIP_TORCH:-0}" == "1" ]]; then
	INSTALL_GROUPS=".[dev,extras]"
	echo "[update.sh] Skipping torch group (SKIP_TORCH=1)"
else
	echo "[update.sh] Including torch group (default, set SKIP_TORCH=1 to skip)"
fi

echo "[update.sh] Installing project with all dependencies from pyproject.toml..."
pip_install -e "$PROJECT_ROOT/$INSTALL_GROUPS"

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
			pip_install_tool setuptools
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
			echo "[update.sh] Using verbose pip output for mlx build (-v)"
			INSTALL_CMD=(pip_install_verbose -e .)
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
				;;
		esac
	done

	# Stage 5: Generate project stubs if applicable
	cd "$SCRIPT_DIR"
	if [[ -f "$SCRIPT_DIR/generate_stubs.py" ]]; then
		echo "[update.sh] Generating type stubs for mlx_lm, mlx_vlm, transformers, and tokenizers..."
		# Use conda run to ensure we're using the mlx-vlm environment
		if command -v conda &> /dev/null && conda env list | grep -q "^mlx-vlm "; then
			if conda run -n mlx-vlm python generate_stubs.py mlx_lm mlx_vlm transformers tokenizers; then
				echo "✓ Project stubs generated successfully"
			else
				echo "⚠️  Failed to generate project stubs (non-fatal)"
			fi
		else
			# Fallback to current python if conda or mlx-vlm env not available
			if python generate_stubs.py mlx_lm mlx_vlm transformers tokenizers; then
				echo "✓ Project stubs generated successfully"
			else
				echo "⚠️  Failed to generate project stubs (non-fatal)"
			fi
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

echo "[update.sh] Done."

# Post-flight check: verify critical packages are still importable
echo ""
echo "[update.sh] Verifying critical packages..."
MISSING_PKGS=()
for pkg in mlx mlx_lm mlx_vlm; do
	if ! python -c "import $pkg" 2>/dev/null; then
		MISSING_PKGS+=("$pkg")
	fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
	echo "❌ WARNING: The following critical packages are NOT importable after update:"
	printf '   - %s\n' "${MISSING_PKGS[@]}"
	echo ""
	echo "   This may cause 'Missing required runtime dependencies' errors."
	echo "   Fix with: pip install ${MISSING_PKGS[*]}"
	echo ""
	exit 1
else
	echo "✓ All critical packages verified (mlx, mlx_lm, mlx_vlm)"
fi

# Final check for held-back / outdated packages
echo ""
echo "[update.sh] Checking for held-back packages (orphaned or constrained)..."
if [[ -f "$SCRIPT_DIR/check_outdated.py" ]]; then
	python "$SCRIPT_DIR/check_outdated.py" || true
else
	echo "⚠️  check_outdated.py not found at $SCRIPT_DIR/check_outdated.py"
fi
