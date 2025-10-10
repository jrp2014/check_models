"""Tests for GPS coordinate conversion and formatting.

This module tests the EXIF GPS extraction and coordinate conversion logic,
particularly the fix for the GPS sign bug (2025-10-05) where signs were
being applied multiple times incorrectly.
"""

# Magic numbers and assert statements are fine in tests
# Type: ignore comments used for test dict invariance issues
# pyright: reportArgumentType=false

from __future__ import annotations

import re

from check_models import _extract_gps_str


class TestGPSCoordinateConversion:
    """Test GPS coordinate extraction and formatting."""

    def test_northern_eastern_coordinates(self) -> None:
        """Northern/Eastern coordinates should display with N/E suffixes."""
        gps_info = {
            "GPSLatitude": (37, 25, 19.2),  # degrees, minutes, seconds
            "GPSLatitudeRef": "N",
            "GPSLongitude": (122, 5, 2.4),
            "GPSLongitudeRef": "E",
        }
        result = _extract_gps_str(gps_info)
        assert result is not None
        # Check format: decimal°CARD, decimal°CARD
        assert "°N" in result
        assert "°E" in result
        # Verify approximate values (37.422°N, 122.084°E)
        assert "37.42" in result  # degrees
        assert "122.08" in result

    def test_southern_western_coordinates(self) -> None:
        """Southern/Western coordinates should display with S/W suffixes."""
        gps_info = {
            "GPSLatitude": (33, 51, 21.6),  # Sydney coordinates
            "GPSLatitudeRef": "S",
            "GPSLongitude": (151, 12, 54.0),
            "GPSLongitudeRef": "E",
        }
        result = _extract_gps_str(gps_info)
        assert result is not None
        assert "°S" in result
        assert "°E" in result
        # Sydney: 33.856°S, 151.215°E
        assert "33.85" in result
        assert "151.21" in result

    def test_bytes_reference_values(self) -> None:
        """GPS references may be bytes - should decode to ASCII."""
        gps_info = {
            "GPSLatitude": (40, 26, 46.0),  # NYC
            "GPSLatitudeRef": b"N",  # bytes instead of string
            "GPSLongitude": (79, 58, 56.0),
            "GPSLongitudeRef": b"W",  # bytes instead of string
        }
        result = _extract_gps_str(gps_info)
        assert result is not None
        assert "°N" in result
        assert "°W" in result

    def test_two_element_coordinates(self) -> None:
        """GPS coordinates with only degrees and minutes (no seconds)."""
        gps_info = {
            "GPSLatitude": (10, 30),  # Only degrees and minutes
            "GPSLatitudeRef": "N",
            "GPSLongitude": (120, 15),
            "GPSLongitudeRef": "W",
        }
        result = _extract_gps_str(gps_info)
        assert result is not None
        # 10°30' = 10.5°
        assert "10.5" in result
        # 120°15' = 120.25°
        assert "120.25" in result

    def test_single_element_coordinates(self) -> None:
        """GPS coordinates with only degrees (no minutes or seconds)."""
        gps_info = {
            "GPSLatitude": (45,),  # Only degrees
            "GPSLatitudeRef": "N",
            "GPSLongitude": (90,),
            "GPSLongitudeRef": "E",
        }
        result = _extract_gps_str(gps_info)
        assert result is not None
        assert "45.0" in result or "45°N" in result
        assert "90.0" in result or "90°E" in result

    def test_missing_latitude(self) -> None:
        """Missing latitude should return None."""
        gps_info = {
            "GPSLongitude": (120, 0, 0),
            "GPSLongitudeRef": "E",
        }
        result = _extract_gps_str(gps_info)
        assert result is None

    def test_missing_longitude(self) -> None:
        """Missing longitude should return None."""
        gps_info = {
            "GPSLatitude": (40, 0, 0),
            "GPSLatitudeRef": "N",
        }
        result = _extract_gps_str(gps_info)
        assert result is None

    def test_missing_latitude_ref(self) -> None:
        """Missing latitude reference should return None."""
        gps_info = {
            "GPSLatitude": (40, 0, 0),
            "GPSLongitude": (120, 0, 0),
            "GPSLongitudeRef": "E",
        }
        result = _extract_gps_str(gps_info)
        assert result is None

    def test_invalid_coordinate_tuple(self) -> None:
        """Empty or malformed coordinate tuples should return None."""
        gps_info = {
            "GPSLatitude": (),  # Empty tuple
            "GPSLatitudeRef": "N",
            "GPSLongitude": (120, 0, 0),
            "GPSLongitudeRef": "E",
        }
        result = _extract_gps_str(gps_info)
        assert result is None

    def test_non_numeric_coordinates(self) -> None:
        """Non-numeric coordinate values should return None."""
        gps_info = {
            "GPSLatitude": ("invalid", "data", "here"),
            "GPSLatitudeRef": "N",
            "GPSLongitude": (120, 0, 0),
            "GPSLongitudeRef": "E",
        }
        result = _extract_gps_str(gps_info)
        assert result is None

    def test_null_input(self) -> None:
        """None input should return None."""
        result = _extract_gps_str(None)
        assert result is None

    def test_empty_dict(self) -> None:
        """Empty dict should return None."""
        result = _extract_gps_str({})
        assert result is None

    def test_non_dict_input(self) -> None:
        """Non-dict input should return None."""
        result = _extract_gps_str("not a dict")  # Intentionally test with invalid type
        assert result is None

    def test_precision_six_decimals(self) -> None:
        """GPS coordinates should have 6 decimal places precision."""
        gps_info = {
            "GPSLatitude": (37, 25, 19.123456),
            "GPSLatitudeRef": "N",
            "GPSLongitude": (122, 5, 2.789012),
            "GPSLongitudeRef": "W",
        }
        result = _extract_gps_str(gps_info)
        assert result is not None
        # Should have 6 decimal places
        # Pattern: number with exactly 6 decimal places
        pattern = r"\d+\.\d{6}°[NSEW]"
        matches = re.findall(pattern, result)
        expected_coords = 2  # Latitude and longitude
        assert len(matches) == expected_coords, (
            f"Expected {expected_coords} coordinates with 6 decimals, got: {result}"
        )

    def test_zero_coordinates(self) -> None:
        """Zero coordinates (null island) should format correctly."""
        gps_info = {
            "GPSLatitude": (0, 0, 0),
            "GPSLatitudeRef": "N",
            "GPSLongitude": (0, 0, 0),
            "GPSLongitudeRef": "E",
        }
        result = _extract_gps_str(gps_info)
        assert result is not None
        assert "0.000000°N" in result
        assert "0.000000°E" in result

    def test_integer_tag_keys(self) -> None:
        """GPS info may use integer keys instead of string names."""
        # Using raw EXIF tag IDs
        gps_info = {
            1: "N",  # GPSLatitudeRef tag ID
            2: (37, 25, 19.2),  # GPSLatitude tag ID
            3: "W",  # GPSLongitudeRef tag ID
            4: (122, 5, 2.4),  # GPSLongitude tag ID
        }
        result = _extract_gps_str(gps_info)
        assert result is not None
        assert "°N" in result
        assert "°W" in result
