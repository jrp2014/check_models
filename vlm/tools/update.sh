#!/usr/bin/env bash
# Unified dependency updater for mlx-vlm-check project.
# Keeps runtime, optional extras, and dev tooling aligned with pyproject.toml.
#
# Usage examples:
#   ./update.sh                       # Update runtime + extras + dev (default behavior)
#   INSTALL_TORCH=1 ./update.sh       # Additionally install torch/torchvision/torchaudio
#   FORCE_REINSTALL=1 ./update.sh     # Force reinstall with --force-reinstall
#
# Environment flags controlling heavy backend suppression are set in Python at runtime;
# here we just upgrade packages.

set -euo pipefail

brew upgrade
pip install -U pip wheel typing_extensions

RUNTIME_PACKAGES=(
	"mlx>=0.29.1"
	"mlx-vlm>=0.0.9"
	"Pillow>=10.3.0"
	"huggingface-hub>=0.23.0"
	"tabulate>=0.9.0"
	"tzlocal>=5.0"
    "numpy"
)

EXTRAS_PACKAGES=(
	"transformers>=4.53.0"
	"mlx-lm>=0.23.0"
	"psutil>=5.9.0"
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
