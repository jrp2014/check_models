"""Tests for time formatting utilities."""

import check_models


def test_format_hms_seconds_only() -> None:
    """Should format seconds as HH:MM:SS (zero-padded)."""
    assert check_models._format_hms(45.5) == "00:00:45"


def test_format_hms_minutes() -> None:
    """Should format minutes and seconds."""
    assert check_models._format_hms(125.3) == "00:02:05"


def test_format_hms_hours() -> None:
    """Should format hours, minutes, and seconds."""
    assert check_models._format_hms(3665.0) == "01:01:05"


def test_format_hms_edge_case_zero() -> None:
    """Should handle zero seconds."""
    assert check_models._format_hms(0.0) == "00:00:00"


def test_format_hms_multiple_hours() -> None:
    """Should handle multiple hours."""
    assert check_models._format_hms(7384.2) == "02:03:04"


def test_format_time_seconds_small() -> None:
    """Should format with two decimals and 's' suffix."""
    assert check_models._format_time_seconds(0.123) == "0.12s"


def test_format_time_seconds_medium() -> None:
    """Should format with two decimals and 's' suffix."""
    assert check_models._format_time_seconds(5.678) == "5.68s"


def test_format_time_seconds_large() -> None:
    """Should format with two decimals and 's' suffix."""
    assert check_models._format_time_seconds(123.456) == "123.46s"


def test_format_time_seconds_zero() -> None:
    """Should handle zero with two decimals."""
    assert check_models._format_time_seconds(0.0) == "0.00s"


def test_format_time_seconds_negative() -> None:
    """Should handle negative values (edge case)."""
    result = check_models._format_time_seconds(-1.5)
    assert "-" in result
