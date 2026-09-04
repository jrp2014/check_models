"""Tests for time formatting utilities."""

import pytest

import check_models


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        pytest.param(0.0, "0.00s", id="zero"),
        pytest.param(45.5, "45.50s", id="seconds_only"),
        pytest.param(89.99, "89.99s", id="just_under_minutes"),
        pytest.param(90.0, "1m 30s", id="minutes_threshold"),
        pytest.param(925.55, "15m 25s", id="minutes"),
        pytest.param(3665.0, "1h 01m 05s", id="hours"),
        pytest.param(7384.2, "2h 03m 04s", id="multiple_hours"),
    ],
)
def test_format_overall_runtime_reads_at_a_glance(seconds: float, expected: str) -> None:
    """Short spans keep precise seconds; longer ones read as m/s or h/m/s."""
    assert check_models.format_overall_runtime(seconds) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        pytest.param(0.123, "0.12s", id="small"),
        pytest.param(5.678, "5.68s", id="medium"),
        pytest.param(123.456, "123.46s", id="large"),
        pytest.param(0.0, "0.00s", id="zero"),
    ],
)
def test_format_time_seconds(seconds: float, expected: str) -> None:
    """Should format seconds with two decimals and 's' suffix."""
    assert check_models._format_time_seconds(seconds) == expected


def test_format_time_seconds_negative() -> None:
    """Should handle negative values (edge case)."""
    result = check_models._format_time_seconds(-1.5)
    assert "-" in result
