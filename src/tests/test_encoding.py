"""Tests for EXIF text encoding and fallback logic."""

from typing import Any

from check_models import _extract_description


def test_extract_description_copyright_utf8() -> None:
    """Should correctly decode UTF-8 copyright symbol."""
    # '© 2024' in UTF-8
    exif_data: dict[str | int, Any] = {"ImageDescription": b"\xc2\xa9 2024 Author"}
    assert _extract_description(exif_data) == "© 2024 Author"


def test_extract_description_copyright_latin1() -> None:
    """Should decode Latin-1 copyright symbol (common in older EXIF)."""
    # '© 2024' in Latin-1 (ISO-8859-1) is \xa9
    exif_data: dict[str | int, Any] = {"ImageDescription": b"\xa9 2024 Author"}
    # With robust fallback, this should now correctly decode to the symbol, not replacement char
    result = _extract_description(exif_data)
    assert result == "© 2024 Author"


def test_extract_description_mixed_garbage() -> None:
    """Should handle truly random bytes without crashing."""
    exif_data: dict[str | int, Any] = {
        "ImageDescription": b"\xff\xfe\x00\x00",
    }  # UTF-32 LE BOM maybe? or just junk
    assert _extract_description(exif_data) is not None
