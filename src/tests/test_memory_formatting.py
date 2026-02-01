"""Tests for memory formatting utilities."""

import pytest

import check_models


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0.123, "0.12", id="small_2_decimals"),
        pytest.param(0.456, "0.46", id="small_rounded"),
        pytest.param(5.678, "5.7", id="medium_1_decimal"),
        pytest.param(5.04, "5.0", id="medium_rounded"),
        pytest.param(50.123, "50", id="large_integer"),
        pytest.param(128.9, "129", id="very_large_rounded"),
        pytest.param(0.0, "0", id="zero"),
        pytest.param(0.001, "0.00", id="tiny"),
    ],
)
def test_format_memory_value_gb(value: float, expected: str) -> None:
    """Should format memory values with appropriate precision based on size."""
    assert check_models._format_memory_value_gb(value) == expected


@pytest.mark.parametrize(
    ("value", "valid_outputs"),
    [
        pytest.param(1.0, ("1.0", "1.00"), id="boundary_small_to_medium"),
        pytest.param(10.0, ("10.0", "10"), id="boundary_medium_to_large"),
    ],
)
def test_format_memory_value_gb_boundaries(value: float, valid_outputs: tuple[str, ...]) -> None:
    """Should handle boundary values (format may vary at transitions)."""
    result = check_models._format_memory_value_gb(value)
    assert result in valid_outputs
