"""Tests for tokens-per-second formatting utilities."""

import pytest

import check_models


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(1.234, "1.23", id="small_2_decimals"),
        pytest.param(0.789, "0.789", id="small_3_sig_figs"),
        pytest.param(12.34, "12.3", id="medium_1_decimal"),
        pytest.param(15.67, "15.7", id="medium_rounded"),
        pytest.param(123.4, "123", id="large_integer"),
        pytest.param(999.9, "1,000", id="very_large_with_comma"),
        pytest.param(0.0, "0", id="zero"),
    ],
)
def test_format_tps(value: float, expected: str) -> None:
    """Should format TPS with appropriate precision based on magnitude."""
    assert check_models._format_tps(value) == expected


@pytest.mark.parametrize(
    ("value", "valid_outputs"),
    [
        pytest.param(10.0, ("10.0", "10.00"), id="boundary_small_to_medium"),
        pytest.param(100.0, ("100.0", "100"), id="boundary_medium_to_large"),
    ],
)
def test_format_tps_boundaries(value: float, valid_outputs: tuple[str, ...]) -> None:
    """Should handle boundary values (format may vary at transitions)."""
    result = check_models._format_tps(value)
    assert result in valid_outputs
