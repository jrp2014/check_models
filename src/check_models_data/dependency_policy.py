"""Shared dependency floors and upstream compatibility policy for check_models."""

from __future__ import annotations

from typing import Final

# Floors also guarantee upstream-shipped typing, which the quality gate relies
# on now that no local stubs are generated at all: mlx >= 0.32.1 wheels bundle
# mlx/core/*.pyi (0.32.0 shipped py.typed without stubs, ml-explore/mlx#3916),
# transformers >= 4.51 ships py.typed inline annotations
# (huggingface/transformers#37022), and mlx-vlm >= 0.6.16 ships py.typed
# (Blaizzy/mlx-vlm#1985), so these floors keep the type checkers supplied
# without local stub generation.
PROJECT_RUNTIME_STACK_MINIMUMS: Final[dict[str, str]] = {
    "mlx": "0.32.1",
    "mlx-vlm": "0.6.16",
    "transformers": "5.14.0",
    "huggingface-hub": "1.10.1",
}

PROJECT_MIN_TRANSFORMERS_VERSION: Final[str] = PROJECT_RUNTIME_STACK_MINIMUMS["transformers"]
PROJECT_TRANSFORMERS_VERSION_SPEC: Final[str] = f">={PROJECT_MIN_TRANSFORMERS_VERSION}"
PROJECT_PILLOW_MINIMUM_VERSION: Final[str] = "12.3.0"

PROJECT_RUNTIME_STACK_SPECS: Final[dict[str, str]] = {
    "mlx": f">={PROJECT_RUNTIME_STACK_MINIMUMS['mlx']}",
    "mlx-vlm": f">={PROJECT_RUNTIME_STACK_MINIMUMS['mlx-vlm']}",
    "transformers": PROJECT_TRANSFORMERS_VERSION_SPEC,
    "huggingface-hub": f">={PROJECT_RUNTIME_STACK_MINIMUMS['huggingface-hub']}",
}

UPSTREAM_MLX_VLM_MINIMUMS: Final[dict[str, str]] = {
    "mlx": "0.32.0",
    "mlx-audio": "0.4.3",
    "transformers": "5.14.0",
}

PROJECT_OPTIONAL_MODEL_SUPPORT_SPECS: Final[dict[str, str]] = {
    "psutil": ">=5.9.0",
    "tokenizers": "<0.24.0,>=0.22.0",
    "einops": ">=0.6.0",
    "num2words": ">=0.5.0",
    "sentencepiece": "!=0.1.92,>=0.1.91",
}

PROJECT_TORCH_EXTRA_COMPAT_SPECS: Final[dict[str, str]] = {
    "torch": ">=2.4.0",
    "torchvision": ">=0.17.0",
    "torchaudio": ">=2.2.0",
    "timm": ">=1.0.23",
}

# Every hard runtime requirement in src/pyproject.toml (extras and version
# markers aside), so validate_env's no-pyproject fallback checks exactly the
# packages the project declares; a test holds the two name sets equal.
VALIDATE_ENV_CORE_FALLBACK_SPECS: Final[dict[str, str]] = {
    **PROJECT_RUNTIME_STACK_SPECS,
    "defusedxml": ">=0.7.1",
    "Pillow": f">={PROJECT_PILLOW_MINIMUM_VERSION}",
    "numpy": ">=2.1.0",
    "packaging": ">=26.0",
    "rich": ">=14.1.0",
    "PyYAML": ">=6.0",
}

VALIDATE_ENV_EXTRAS_FALLBACK_SPECS: Final[dict[str, str]] = {
    **PROJECT_OPTIONAL_MODEL_SUPPORT_SPECS,
    **PROJECT_TORCH_EXTRA_COMPAT_SPECS,
}
