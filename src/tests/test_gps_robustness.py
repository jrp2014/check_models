"""Tests for GPS extraction robustness."""

from typing import Any

from check_models import _extract_description, _extract_gps_str


def test_gps_str_standard_dms() -> None:
    """Standard DMS tuple should format correctly."""
    # (37, 46, 30.5) N, (122, 25, 6.0) W
    gps_data: dict[str | int, Any] = {
        "GPSLatitudeRef": "N",
        "GPSLatitude": (37.0, 46.0, 30.5),
        "GPSLongitudeRef": "W",
        "GPSLongitude": (122.0, 25.0, 6.0),
    }
    # 37 + 46/60 + 30.5/3600 = 37.775138...
    # 122 + 25/60 + 6.0/3600 = 122.418333...
    result = _extract_gps_str(gps_data)
    assert result is not None
    assert "37.7751" in result
    assert "N" in result
    assert "122.4183" in result
    assert "W" in result


def test_gps_str_decimal_minutes() -> None:
    """Decimal minutes format should handle 2-element tuples."""
    gps_data: dict[str | int, Any] = {
        "GPSLatitudeRef": "S",
        "GPSLatitude": (37.0, 46.508333),
        "GPSLongitudeRef": "E",
        "GPSLongitude": (122.0, 25.1),
    }
    result = _extract_gps_str(gps_data)
    assert result is not None
    assert "S" in result
    assert "E" in result


def test_gps_str_missing_ref() -> None:
    """Should return None if reference direction is missing."""
    gps_data: dict[str | int, Any] = {
        "GPSLatitude": (37.0, 46.0, 30.5),
        # Missing Ref
        "GPSLongitudeRef": "W",
        "GPSLongitude": (122.0, 25.0, 6.0),
    }
    assert _extract_gps_str(gps_data) is None


def test_gps_str_invalid_type() -> None:
    """Should handle non-numeric coordinate values gracefully."""
    gps_data: dict[str | int, Any] = {
        "GPSLatitudeRef": "N",
        "GPSLatitude": "invalid",
        "GPSLongitudeRef": "W",
        "GPSLongitude": (122.0, 25.0, 6.0),
    }
    assert _extract_gps_str(gps_data) is None


def test_check_description_null_bytes() -> None:
    """Should strip null bytes from description strings."""
    exif_data: dict[str | int, Any] = {"ImageDescription": "Hello\x00World"}
    # Current implementation might fail this if it doesn't strip nulls
    assert (
        _extract_description(exif_data) == "Hello"
        or _extract_description(exif_data) == "HelloWorld"
    )
