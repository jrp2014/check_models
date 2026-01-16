"""Tests for memory formatting utilities."""

import check_models


def test_format_memory_value_gb_small() -> None:
    """Should format small memory values with 2 decimals."""
    assert check_models._format_memory_value_gb(0.123) == "0.12"


def test_format_memory_value_gb_small_rounded() -> None:
    """Should round small values to 2 decimals."""
    assert check_models._format_memory_value_gb(0.456) == "0.46"


def test_format_memory_value_gb_medium() -> None:
    """Should format medium memory values with 1 decimal."""
    assert check_models._format_memory_value_gb(5.678) == "5.7"


def test_format_memory_value_gb_medium_rounded() -> None:
    """Should round medium values to 1 decimal."""
    assert check_models._format_memory_value_gb(5.04) == "5.0"


def test_format_memory_value_gb_large() -> None:
    """Should format large memory values as integers."""
    assert check_models._format_memory_value_gb(50.123) == "50"


def test_format_memory_value_gb_very_large() -> None:
    """Should format very large memory values as integers."""
    assert check_models._format_memory_value_gb(128.9) == "129"


def test_format_memory_value_gb_zero() -> None:
    """Should handle zero memory value."""
    assert check_models._format_memory_value_gb(0.0) == "0"


def test_format_memory_value_gb_tiny() -> None:
    """Should handle very small values."""
    assert check_models._format_memory_value_gb(0.001) == "0.00"


def test_format_memory_value_gb_boundary_small_to_medium() -> None:
    """Should handle boundary at 1.0 GB."""
    result = check_models._format_memory_value_gb(1.0)
    assert result in ("1.0", "1.00")  # Accept either format


def test_format_memory_value_gb_boundary_medium_to_large() -> None:
    """Should handle boundary at 10.0 GB."""
    result = check_models._format_memory_value_gb(10.0)
    assert result in ("10.0", "10")  # Accept either format
