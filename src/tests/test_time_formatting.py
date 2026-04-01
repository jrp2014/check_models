"""Tests for time formatting utilities."""

import pytest

import check_models


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        pytest.param(45.5, "00:00:45", id="seconds_only"),
        pytest.param(125.3, "00:02:05", id="minutes"),
        pytest.param(3665.0, "01:01:05", id="hours"),
        pytest.param(0.0, "00:00:00", id="zero"),
        pytest.param(7384.2, "02:03:04", id="multiple_hours"),
    ],
)
def test_format_hms(seconds: float, expected: str) -> None:
    """Should format total seconds as HH:MM:SS."""
    assert check_models._format_hms(seconds) == expected


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
