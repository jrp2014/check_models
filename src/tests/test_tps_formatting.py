"""Tests for tokens-per-second formatting utilities."""

# ruff: noqa: SLF001
import check_models


def test_format_tps_small() -> None:
    """Should format small TPS with 2 decimals."""
    assert check_models._format_tps(1.234) == "1.23"


def test_format_tps_small_rounded() -> None:
    """Should format small values with 3 significant figures (.3g format)."""
    assert check_models._format_tps(0.789) == "0.789"


def test_format_tps_medium() -> None:
    """Should format medium TPS with 1 decimal."""
    assert check_models._format_tps(12.34) == "12.3"


def test_format_tps_medium_rounded() -> None:
    """Should round medium values to 1 decimal."""
    assert check_models._format_tps(15.67) == "15.7"


def test_format_tps_large() -> None:
    """Should format large TPS as integer."""
    assert check_models._format_tps(123.4) == "123"


def test_format_tps_very_large() -> None:
    """Should format very large values as integers with comma separator."""
    assert check_models._format_tps(999.9) == "1,000"


def test_format_tps_zero() -> None:
    """Should handle zero TPS."""
    assert check_models._format_tps(0.0) == "0"


def test_format_tps_boundary_small_to_medium() -> None:
    """Should handle boundary at 10.0 TPS."""
    result = check_models._format_tps(10.0)
    assert result in ("10.0", "10.00")  # Accept either format


def test_format_tps_boundary_medium_to_large() -> None:
    """Should handle boundary at 100.0 TPS."""
    result = check_models._format_tps(100.0)
    assert result in ("100.0", "100")  # Accept either format
