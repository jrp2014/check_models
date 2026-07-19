"""Unit tests for error classification and package attribution logic."""

import pytest

from check_models import (
    ControlledReproductionStatus,
    FailureOrigin,
    MaintainerReadiness,
    PerformanceResult,
    UpstreamBoundary,
    _attribute_error_to_package,
    _build_canonical_error_code,
    _build_error_signature,
    _build_failure_action_hint,
    _classify_error,
    _failure_origin,
    _maintainer_readiness,
)


@pytest.mark.parametrize(
    ("message", "expected_type"),
    [
        # Critical errors
        ("[metal::malloc] Attempting to allocate...", "OOM"),
        ("maximum allowed buffer size exceeded", "OOM"),
        ("Operation timeout", "Timeout"),
        (
            "Model loading failed: Server disconnected without sending a response.",
            "Network Error",
        ),
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
        (
            "Model loading failed: Server disconnected without sending a response.",
            "mlx_vlm/utils.py caused by httpcore.RemoteProtocolError",
            "unknown",
        ),
        # Wrapped upstream failures should prefer the deeper traceback owner
        (
            "Model generation failed: Failed to process inputs with error: can only concatenate str",
            (
                "Traceback (most recent call last):\n"
                '  File "mlx_vlm/utils.py", line 873, in process_inputs_with_fallback\n'
                '  File "transformers/models/florence2/processing_florence2.py", line 185, in __call__\n'
                'TypeError: can only concatenate str (not "NoneType") to str\n\n'
                "The above exception was the direct cause of the following exception:\n\n"
                '  File "mlx_vlm/generate.py", line 515, in stream_generate\n'
                "ValueError: Failed to process inputs with error: can only concatenate str\n"
            ),
            "transformers",
        ),
        # Unknown
        ("Random error", "unknown_lib/main.py", "unknown"),
    ],
)
def test_attribute_error(message: str, traceback: str | None, expected_package: str) -> None:
    """Verify that errors are attributed to the correct package."""
    assert _attribute_error_to_package(message, traceback) == expected_package


def test_build_canonical_error_code_stable_tokens() -> None:
    """Canonical code should include package, phase, and stage in stable form."""
    code = _build_canonical_error_code(
        error_stage="API Mismatch",
        error_package="transformers",
        failure_phase="decode",
    )
    assert code == "TRANSFORMERS_DECODE_API_MISMATCH"


def test_error_signature_normalizes_numeric_variants() -> None:
    """Numeric differences in shape errors should map to the same signature."""
    code = _build_canonical_error_code(
        error_stage="Model Error",
        error_package="mlx-vlm",
        failure_phase="decode",
    )
    sig_a = _build_error_signature(
        error_code=code,
        error_message="[broadcast_shapes] Shapes (3,1,2048) and (3,1,6065) mismatch",
        error_traceback=None,
    )
    sig_b = _build_error_signature(
        error_code=code,
        error_message="[broadcast_shapes] Shapes (984,2048) and (1,0,2048) mismatch",
        error_traceback=None,
    )
    assert sig_a == sig_b


@pytest.mark.parametrize(
    ("error_package", "failure_phase", "error_stage", "expected_phrases"),
    [
        (
            "mlx-vlm",
            "decode",
            "Model Error",
            ("mlx-vlm generation/integration path", "decode/generation", "model runtime failure"),
        ),
        (
            "model-config",
            "processor_load",
            "Processor Error",
            (
                "model repository/config artifacts",
                "processor/image-processor initialization",
                "processor construction",
            ),
        ),
    ],
)
def test_build_failure_action_hint_is_maintainer_actionable(
    error_package: str,
    failure_phase: str,
    error_stage: str,
    expected_phrases: tuple[str, str, str],
) -> None:
    """Failure hint should include owner, component, and likely-cause clues."""
    hint = _build_failure_action_hint(
        error_package=error_package,
        failure_phase=failure_phase,
        error_stage=error_stage,
    )
    hint_lower = hint.lower()
    for phrase in expected_phrases:
        assert phrase.lower() in hint_lower


@pytest.mark.parametrize(
    ("origin", "reproduction", "has_output_anomaly", "expected"),
    [
        ("harness_preflight", "not_run", False, "harness_observation"),
        ("external_service", "not_run", False, "not_applicable"),
        ("unknown", "not_run", False, "needs_reproduction"),
        ("upstream_load", "not_run", False, "issue_ready"),
        ("upstream_generation", "not_run", False, "issue_ready"),
        ("unknown", "confirmed", True, "issue_ready"),
        ("unknown", "not_reproduced", True, "not_applicable"),
        ("unknown", "indeterminate", True, "needs_reproduction"),
        ("unknown", "not_run", True, "needs_reproduction"),
    ],
)
def test_maintainer_readiness_uses_origin_and_reproduction_evidence(
    origin: FailureOrigin,
    reproduction: ControlledReproductionStatus,
    has_output_anomaly: bool,
    expected: MaintainerReadiness,
) -> None:
    """Maintainer readiness must not collapse model-user and issue-fileability views."""
    assert (
        _maintainer_readiness(
            failure_origin=origin,
            reproduction_status=reproduction,
            has_output_anomaly=has_output_anomaly,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("boundary", "phase", "message", "expected"),
    [
        ("not_started", "model_preflight", "invalid local image input", "harness_preflight"),
        ("load_started", "model_load", "loader raised", "upstream_load"),
        ("generation_started", "decode", "generator raised", "upstream_generation"),
        (
            "load_started",
            "model_load",
            "server disconnected without sending a response",
            "external_service",
        ),
    ],
)
def test_failure_origin_follows_execution_boundary(
    boundary: UpstreamBoundary,
    phase: str,
    message: str,
    expected: FailureOrigin,
) -> None:
    """Failure ownership begins with the recorded upstream entry boundary."""
    result = PerformanceResult(
        model_name="example/model",
        generation=None,
        success=False,
        failure_phase=phase,
        error_message=message,
        upstream_boundary=boundary,
    )

    assert _failure_origin(result) == expected


def test_unknown_failure_origin_requires_trustworthy_boundary_evidence() -> None:
    """An untagged failure must remain unknown rather than being blamed upstream."""
    result = PerformanceResult(
        model_name="example/model",
        generation=None,
        success=False,
        failure_phase="unexpected_phase",
        error_message="unclassified failure",
    )

    assert _failure_origin(result) == "unknown"
