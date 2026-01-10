"""Unit tests for error classification and package attribution logic."""

import pytest

from check_models import _attribute_error_to_package, _classify_error


@pytest.mark.parametrize(
    ("message", "expected_type"),
    [
        # Critical errors
        ("[metal::malloc] Attempting to allocate...", "OOM"),
        ("maximum allowed buffer size exceeded", "OOM"),
        ("Operation timeout", "Timeout"),
        # Dependency/Version errors
        ("ImportError: cannot import name 'foo'", "Lib Version"),
        ("requires packages X. pip install X", "Missing Dep"),
        # API Mismatches
        ("got an unexpected keyword argument 'images'", "API Mismatch"),
        ("TypeError: _batch_encode_plus() got an unexpected keyword", "API Mismatch"),
        # Config/Validation errors
        ("does not appear to have a file named config.json", "Config Missing"),
        ("ValueError: chat_template is not set", "No Chat Template"),
        ("ValueError: Missing 5 parameters:", "Weight Mismatch"),
        # MLX/Core errors
        ("std::bad_cast", "Type Cast Error"),
        # Specific component errors
        ("Unrecognized image_processor", "Processor Error"),
        ("Tokenizer class Tokenizer404 does not exist", "Tokenizer Error"),
        # Generic
        ("Model generation failed", "Model Error"),
        ("Something went wrong", "Error"),
    ],
)
def test_classify_error(message: str, expected_type: str) -> None:
    """Verify that error messages are correctly classified."""
    assert _classify_error(message) == expected_type


@pytest.mark.parametrize(
    ("message", "traceback", "expected_package"),
    [
        # MLX Core
        ("std::bad_cast", None, "mlx"),
        ("metal::malloc failed", "mlx/core/memory.cpp", "mlx"),
        ("Error", "mlx/python/mlx/ops.py line 42", "mlx"),
        # MLX-VLM
        ("ValueError: Load check failed", "mlx_vlm/utils.py", "mlx-vlm"),
        ("apply_chat_template failed", None, "mlx-vlm"),
        # Transformers
        ("ImportError: cannot import name", "transformers/models/auto.py", "transformers"),
        ("got an unexpected keyword argument", "transformers/tokenization.py", "transformers"),
        # Model/Config
        ("missing parameters", None, "model-config"),
        ("chat_template is not set", None, "model-config"),
        # HuggingFace
        ("does not appear to have a file", "huggingface_hub/utils.py", "huggingface-hub"),
        # Unknown
        ("Random error", "unknown_lib/main.py", "unknown"),
    ],
)
def test_attribute_error(message: str, traceback: str | None, expected_package: str) -> None:
    """Verify that errors are attributed to the correct package."""
    assert _attribute_error_to_package(message, traceback) == expected_package
