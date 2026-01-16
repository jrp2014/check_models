"""Tests for EXIF extraction utilities."""

from pathlib import Path
from typing import Any

import check_models


def test_extract_exif_date_standard_format(tmp_path: Path) -> None:
    """Should parse standard EXIF datetime format."""
    test_file = tmp_path / "test.jpg"
    test_file.touch()
    exif_dict: dict[str | int, Any] = {"DateTime": "2024:01:15 14:30:45"}
    result = check_models._extract_exif_date(test_file, exif_dict)
    assert result is not None
    assert "2024-01-15" in result
    assert "14:30:45" in result


def test_extract_exif_date_datetime_original(tmp_path: Path) -> None:
    """Should prefer DateTimeOriginal over DateTime."""
    test_file = tmp_path / "test.jpg"
    test_file.touch()
    exif_dict: dict[str | int, Any] = {
        "DateTime": "2024:01:15 14:30:45",
        "DateTimeOriginal": "2024:01:10 10:20:30",
    }
    result = check_models._extract_exif_date(test_file, exif_dict)
    assert result is not None
    # Should use DateTimeOriginal (Jan 10) not DateTime (Jan 15)
    assert "2024-01-10" in result


def test_extract_exif_date_create_date(tmp_path: Path) -> None:
    """Should use CreateDate when DateTimeOriginal absent."""
    test_file = tmp_path / "test.jpg"
    test_file.touch()
    exif_dict: dict[str | int, Any] = {
        "CreateDate": "2024:01:12 12:00:00",
        "DateTime": "2024:01:15 14:30:45",
    }
    result = check_models._extract_exif_date(test_file, exif_dict)
    assert result is not None
    # Should use CreateDate (Jan 12), which has priority over DateTime
    assert "2024-01-12" in result


def test_extract_exif_date_fallback_to_mtime(tmp_path: Path) -> None:
    """Should fallback to file mtime when no date fields present."""
    test_file = tmp_path / "test.jpg"
    test_file.touch()
    exif_dict: dict[str | int, Any] = {"Make": "Camera", "Model": "Test"}
    result = check_models._extract_exif_date(test_file, exif_dict)
    # Should return mtime as fallback
    assert result is not None


def test_extract_exif_date_invalid_format(tmp_path: Path) -> None:
    """Should handle invalid datetime format gracefully."""
    test_file = tmp_path / "test.jpg"
    test_file.touch()
    exif_dict: dict[str | int, Any] = {"DateTime": "invalid date"}
    result = check_models._extract_exif_date(test_file, exif_dict)
    # Should return the raw string when parsing fails
    assert result is not None


def test_extract_description_image_description() -> None:
    """Should extract ImageDescription field."""
    exif_dict: dict[str | int, Any] = {"ImageDescription": "Test photo description"}
    result = check_models._extract_description(exif_dict)
    assert result == "Test photo description"


def test_extract_description_bytes() -> None:
    """Should decode bytes description."""
    exif_dict: dict[str | int, Any] = {"ImageDescription": b"Byte description"}
    result = check_models._extract_description(exif_dict)
    assert result == "Byte description"


def test_extract_description_missing() -> None:
    """Should return None when ImageDescription field absent."""
    exif_dict: dict[str | int, Any] = {"Make": "Camera", "Model": "Test"}
    result = check_models._extract_description(exif_dict)
    assert result is None


def test_extract_description_empty_string() -> None:
    """Should return None for empty description."""
    exif_dict: dict[str | int, Any] = {"ImageDescription": ""}
    result = check_models._extract_description(exif_dict)
    assert result is None


def test_extract_description_whitespace_only() -> None:
    """Should return None for whitespace-only description."""
    exif_dict: dict[str | int, Any] = {"ImageDescription": "   "}
    result = check_models._extract_description(exif_dict)
    assert result is None
