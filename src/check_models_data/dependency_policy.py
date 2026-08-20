"""Shared dependency floors and upstream compatibility policy for check_models."""

from __future__ import annotations

from typing import Final

# Floors also guarantee upstream-shipped typing, which the quality gate relies
# on now that local stubs are generated for mlx_vlm only: mlx >= 0.32.1 wheels
# bundle mlx/core/*.pyi (0.32.0 shipped py.typed without stubs, ml-explore/mlx
# #3916), and transformers >= 4.51 ships py.typed inline annotations
# (huggingface/transformers#37022), so any floor at or above these keeps the
# type checkers supplied without local stub generation.
PROJECT_RUNTIME_STACK_MINIMUMS: Final[dict[str, str]] = {
    "mlx": "0.32.1",
    "mlx-vlm": "0.6.13",
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

# mlx-vlm required mlx-lm only through the 0.6.13 release; from 0.6.14 (main
# commit 738e4406, "Porting twenty four MLX-LM models") the ported code is
# vendored and mlx-lm is no longer a dependency. Applied only when the
# installed mlx-vlm predates that version, so a documented minimal install
# never sees a false "mlx-lm is missing" warning.
UPSTREAM_MLX_VLM_LEGACY_MLX_LM_MINIMUM: Final[str] = "0.31.3"
UPSTREAM_MLX_VLM_FIRST_VERSION_WITHOUT_MLX_LM: Final[str] = "0.6.14"

UPSTREAM_MLX_LM_MINIMUMS: Final[dict[str, str]] = {
    "mlx": "0.31.2",
    "transformers": "5.7.0",
}

PROJECT_OPTIONAL_MODEL_SUPPORT_SPECS: Final[dict[str, str]] = {
    # Ecosystem provenance only: no direct import; upstream mlx-vlm dropped
    # its own mlx-lm dependency (738e4406). Recorded in reports when present.
    "mlx-lm": ">=0.31.3",
    "psutil": ">=5.9.0",
    "tokenizers": "<0.23.0,>=0.22.0",
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
    **PROJECT_RUNTIME_STACK_SPECS,
    "defusedxml": ">=0.7.1",
    "packaging": ">=26.0",
    "Pillow": f">={PROJECT_PILLOW_MINIMUM_VERSION}",
    "wcwidth": ">=0.2.13",
    "PyYAML": ">=6.0",
}

VALIDATE_ENV_EXTRAS_FALLBACK_SPECS: Final[dict[str, str]] = {
    **PROJECT_OPTIONAL_MODEL_SUPPORT_SPECS,
    **PROJECT_TORCH_EXTRA_COMPAT_SPECS,
}
