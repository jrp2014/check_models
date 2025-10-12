#!/usr/bin/env bash
# Unified dependency updater for mlx-vlm-check project.
# Keeps runtime, optional extras, and dev tooling aligned with pyproject.toml.
#
# Usage examples:
#   ./update.sh                       # Update runtime + extras + dev (default behavior)
#   INSTALL_TORCH=1 ./update.sh       # Additionally install torch/torchvision/torchaudio
#   FORCE_REINSTALL=1 ./update.sh     # Force reinstall with --force-reinstall
#   SKIP_MLX=1 ./update.sh            # Force skip mlx/mlx-vlm updates (override detection)
#
# Local MLX Development:
#   If mlx, mlx-lm, and mlx-vlm directories exist at ../../ (sibling to scripts/),
#   the script will automatically:
#   1. Run git pull in each repository
#   2. Install requirements.txt (if present)
#   3. Install packages in order: mlx → mlx-lm → mlx-vlm
#   4. Skip PyPI updates for these packages
#
# Note: Script automatically detects and preserves local MLX dev builds (versions
# containing .dev or +commit). Stable releases are updated normally.
#
# Environment flags controlling heavy backend suppression are set in Python at runtime;
# here we just upgrade packages.

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

# Function to update local MLX development repositories
update_local_mlx_repos() {
	# Determine the parent directory (assuming scripts/src/tools/update.sh)
	local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	local PARENT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
	
	echo "[update.sh] Checking for local MLX development repositories..."
	
	# Define the repositories in install order
	local MLX_REPOS=("mlx" "mlx-lm" "mlx-vlm")
	local FOUND_LOCAL=0
	
	for repo in "${MLX_REPOS[@]}"; do
		local REPO_PATH="$PARENT_DIR/$repo"
		if [[ -d "$REPO_PATH/.git" ]]; then
			FOUND_LOCAL=1
			echo ""
			echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
			echo "📦 Updating local repository: $repo"
			echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
			
			cd "$REPO_PATH"
			
			# Git pull
			echo "[update.sh] Running git pull in $repo..."
			if git pull; then
				echo "✓ Git pull successful"
			else
				echo "⚠️  Git pull failed or had conflicts - skipping install for $repo"
				continue
			fi
			
			# Special handling for mlx-vlm: install opencv-python first
			if [[ "$repo" == "mlx-vlm" ]]; then
				echo "[update.sh] Installing opencv-python for mlx-vlm..."
				pip install -U opencv-python
			fi
			
			# Install requirements.txt if it exists
			if [[ -f "requirements.txt" ]]; then
				echo "[update.sh] Installing requirements from requirements.txt..."
				pip install -U -r requirements.txt
			else
				echo "[update.sh] No requirements.txt found"
			fi
			
			# Install the package in editable mode
			echo "[update.sh] Installing $repo package..."
			if pip install -e .; then
				echo "✓ $repo installed successfully"
			else
				echo "⚠️  Failed to install $repo"
			fi
			
			echo ""
		fi
	done
	
	if [[ $FOUND_LOCAL -eq 1 ]]; then
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
		echo "✓ Local MLX repositories updated"
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
		echo ""
		# Return to the scripts directory
		cd "$SCRIPT_DIR"
		return 0
	else
		echo "[update.sh] No local MLX development repositories found"
		echo "[update.sh] (Looked for mlx, mlx-lm, mlx-vlm directories with .git at $PARENT_DIR)"
		return 1
	fi
}

# Update local MLX repos if they exist
LOCAL_MLX_UPDATED=0
if update_local_mlx_repos; then
	LOCAL_MLX_UPDATED=1
	echo "[update.sh] Local MLX builds updated - will skip PyPI MLX packages"
fi

brew upgrade
pip install -U pip wheel typing_extensions

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

RUNTIME_PACKAGES=()
# Only update MLX if not a local dev build (or if SKIP_MLX=1 is explicitly set)
if [[ "${SKIP_MLX:-0}" != "1" ]] && [[ "$MLX_IS_LOCAL" != "1" ]]; then
	RUNTIME_PACKAGES+=("mlx>=0.29.1" "mlx-vlm>=0.0.9")
elif [[ "${SKIP_MLX:-0}" == "1" ]]; then
	echo "[update.sh] Skipping mlx/mlx-vlm updates (SKIP_MLX=1)..."
fi

RUNTIME_PACKAGES+=(
	"Pillow>=10.3.0"
	"huggingface-hub>=0.23.0"
	"huggingface-hub[cli]"
	"tabulate>=0.9.0"
	"tzlocal>=5.0"
)

EXTRAS_PACKAGES=(
	"psutil>=5.9.0"
	"tokenizers>=0.15.0"
	"einops>=0.6.0"
	"num2words>=0.5.0"
	"mlx-lm>=0.23.0"
	"transformers>=4.53.0"
)

DEV_PACKAGES=(
    "cmake>=3.25,<4.1"
	"ruff>=0.1.0"
	"mypy>=1.8.0"
	"pytest>=8.0.0"
	"pytest-cov>=4.0.0"
    "setuptools>=80"
    "types-tabulate"
    "nanobind==2.4.0"
	"gh"
)

TORCH_PACKAGES=(
	"torch>=2.2.0" "torchvision>=0.17.0" "torchaudio>=2.2.0"
)

EXTRA_ARGS=()
if [[ "${FORCE_REINSTALL:-0}" == "1" ]]; then
	EXTRA_ARGS+=("--force-reinstall")
fi

echo "[update.sh] Updating runtime + extras + dev dependencies..."
ALL_PRIMARY=("${RUNTIME_PACKAGES[@]}" "${EXTRAS_PACKAGES[@]}" "${DEV_PACKAGES[@]}" safetensors accelerate tqdm)
if ((${#EXTRA_ARGS[@]})); then
	pip install -U "${EXTRA_ARGS[@]}" "${ALL_PRIMARY[@]}"
else
	pip install -U "${ALL_PRIMARY[@]}"
fi

if [[ "${INSTALL_TORCH:-0}" == "1" ]]; then
	echo "[update.sh] Installing PyTorch stack (optional)..."
	if ((${#EXTRA_ARGS[@]})); then
		pip install -U "${EXTRA_ARGS[@]}" "${TORCH_PACKAGES[@]}"
	else
		pip install -U "${TORCH_PACKAGES[@]}"
	fi
fi

echo "[update.sh] Done."
