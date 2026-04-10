"""Shared dependency floors and upstream compatibility policy for check_models."""

from __future__ import annotations

from typing import Final

PROJECT_RUNTIME_STACK_MINIMUMS: Final[dict[str, str]] = {
    "mlx": "0.31.1",
    "mlx-vlm": "0.4.4",
    "transformers": "5.5.3",
    "huggingface-hub": "1.10.1",
}

PROJECT_OPTIONAL_STACK_MINIMUMS: Final[dict[str, str]] = {
    "mlx-lm": "0.31.3",
}

PROJECT_MIN_TRANSFORMERS_VERSION: Final[str] = PROJECT_RUNTIME_STACK_MINIMUMS["transformers"]

UPSTREAM_MLX_VLM_MINIMUMS: Final[dict[str, str]] = {
    "mlx": "0.30.0",
    "mlx-lm": "0.31.0",
    "transformers": "5.1.0",
}

UPSTREAM_MLX_LM_MINIMUMS: Final[dict[str, str]] = {
    "mlx": "0.30.4",
    "transformers": "5.0.0",
}

VALIDATE_ENV_CORE_FALLBACK_SPECS: Final[dict[str, str]] = {
    "mlx": f">={PROJECT_RUNTIME_STACK_MINIMUMS['mlx']}",
    "mlx-vlm": f">={PROJECT_RUNTIME_STACK_MINIMUMS['mlx-vlm']}",
    "transformers": f">={PROJECT_RUNTIME_STACK_MINIMUMS['transformers']}",
    "huggingface-hub": f">={PROJECT_RUNTIME_STACK_MINIMUMS['huggingface-hub']}",
    "packaging": ">=26.0",
    "Pillow": ">=10.3.0",
    "tabulate": ">=0.9.0",
    "tzlocal": ">=5.0",
    "requests": ">=2.31.0",
    "wcwidth": ">=0.2.13",
    "PyYAML": ">=6.0",
}

VALIDATE_ENV_EXTRAS_FALLBACK_SPECS: Final[dict[str, str]] = {
    "psutil": ">=5.9.0",
    "tokenizers": ">=0.15.0",
    "einops": ">=0.6.0",
    "num2words": ">=0.5.0",
    "mlx-lm": f">={PROJECT_OPTIONAL_STACK_MINIMUMS['mlx-lm']}",
}
