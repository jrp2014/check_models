"""Shared dependency floors and upstream compatibility policy for check_models."""

from __future__ import annotations

from typing import Final

PROJECT_RUNTIME_STACK_MINIMUMS: Final[dict[str, str]] = {
    "mlx": "0.31.2",
    "mlx-lm": "0.31.3",
    "mlx-vlm": "0.6.2",
    "transformers": "5.7.0",
    "huggingface-hub": "1.10.1",
}

PROJECT_MIN_TRANSFORMERS_VERSION: Final[str] = PROJECT_RUNTIME_STACK_MINIMUMS["transformers"]

UPSTREAM_MLX_VLM_MINIMUMS: Final[dict[str, str]] = {
    "mlx": "0.31.2",
    "mlx-lm": "0.31.3",
    "mlx-audio": "0.4.3",
    "transformers": "5.5.0",
}

UPSTREAM_MLX_LM_MINIMUMS: Final[dict[str, str]] = {
    "mlx": "0.31.2",
    "transformers": "5.7.0",
}

PROJECT_OPTIONAL_MODEL_SUPPORT_SPECS: Final[dict[str, str]] = {
    "psutil": ">=5.9.0",
    "tokenizers": "<=0.23.0,>=0.22.0",
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

VALIDATE_ENV_CORE_FALLBACK_SPECS: Final[dict[str, str]] = {
    "mlx": f">={PROJECT_RUNTIME_STACK_MINIMUMS['mlx']}",
    "mlx-lm": f">={PROJECT_RUNTIME_STACK_MINIMUMS['mlx-lm']}",
    "mlx-vlm": f">={PROJECT_RUNTIME_STACK_MINIMUMS['mlx-vlm']}",
    "transformers": f">={PROJECT_RUNTIME_STACK_MINIMUMS['transformers']}",
    "huggingface-hub": f">={PROJECT_RUNTIME_STACK_MINIMUMS['huggingface-hub']}",
    "defusedxml": ">=0.7.1",
    "packaging": ">=26.0",
    "Pillow": ">=10.3.0",
    "tabulate": ">=0.9.0",
    "wcwidth": ">=0.2.13",
    "PyYAML": ">=6.0",
}

VALIDATE_ENV_EXTRAS_FALLBACK_SPECS: Final[dict[str, str]] = {
    **PROJECT_OPTIONAL_MODEL_SUPPORT_SPECS,
    **PROJECT_TORCH_EXTRA_COMPAT_SPECS,
}
