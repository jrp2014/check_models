#!/usr/bin/env bash
# Unified dependency updater for mlx-vlm-check project.
# Keeps runtime, optional extras, and dev tooling aligned with pyproject.toml.
#
# Usage examples:
#   ./update.sh                # Update core runtime only
#   INSTALL_EXTRAS=1 ./update.sh      # Include extras group (transformers, mlx-lm, psutil)
#   INSTALL_DEV=1 ./update.sh         # Include dev tools (ruff, mypy, pytest, pytest-cov)
#   INSTALL_TORCH=1 ./update.sh       # Add PyTorch packages (optional / only if you need torch models)
#   FORCE_REINSTALL=1 ./update.sh     # Force reinstall with --force-reinstall
#
# Environment flags controlling heavy backend suppression are set in Python at runtime;
# here we just upgrade packages.

set -euo pipefail

pip install -U pip setuptools wheel

RUNTIME_PACKAGES=(
	"mlx>=0.14.0"
	"mlx-vlm>=0.0.9"
	"Pillow>=10.0.0"
	"huggingface-hub>=0.23.0"
	"tabulate>=0.9.0"
	"tzlocal>=5.0"
)

EXTRAS_PACKAGES=(
	"transformers>=4.41.0,<5"
	"mlx-lm>=0.10.0"
	"psutil>=5.9.0"
	"tokenizers>=0.15.0" # explicit (also pulled transitively by transformers)
)

DEV_PACKAGES=(
	"ruff>=0.1.0"
	"mypy>=1.8.0"
	"pytest>=8.0.0"
	"pytest-cov>=4.0.0"
)

TORCH_PACKAGES=(
	"torch" "torchvision" "torchaudio"
)

EXTRA_ARGS=()
if [[ "${FORCE_REINSTALL:-0}" == "1" ]]; then
	EXTRA_ARGS+=("--force-reinstall")
fi

echo "[update.sh] Updating core runtime dependencies..."
if ((${#EXTRA_ARGS[@]})); then
	pip install -U "${EXTRA_ARGS[@]}" "${RUNTIME_PACKAGES[@]}"
else
	pip install -U "${RUNTIME_PACKAGES[@]}"
fi

if [[ "${INSTALL_EXTRAS:-0}" == "1" ]]; then
	echo "[update.sh] Installing extras group..."
	if ((${#EXTRA_ARGS[@]})); then
		pip install -U "${EXTRA_ARGS[@]}" "${EXTRAS_PACKAGES[@]}" safetensors accelerate tqdm
	else
		pip install -U "${EXTRAS_PACKAGES[@]}" safetensors accelerate tqdm
	fi
fi

if [[ "${INSTALL_TORCH:-0}" == "1" ]]; then
	echo "[update.sh] Installing PyTorch stack (optional)..."
	if ((${#EXTRA_ARGS[@]})); then
		pip install -U "${EXTRA_ARGS[@]}" "${TORCH_PACKAGES[@]}"
	else
		pip install -U "${TORCH_PACKAGES[@]}"
	fi
fi

if [[ "${INSTALL_DEV:-0}" == "1" ]]; then
	echo "[update.sh] Installing dev tools..."
	if ((${#EXTRA_ARGS[@]})); then
		pip install -U "${EXTRA_ARGS[@]}" "${DEV_PACKAGES[@]}"
	else
		pip install -U "${DEV_PACKAGES[@]}"
	fi
fi

echo "[update.sh] Done."
