"""Tests for EXIF extraction utilities."""

from pathlib import Path
from typing import Any

from PIL import Image

import check_models
from check_models import (
    _build_cataloguing_prompt,
    _extract_xp_keywords,
    _merge_keywords,
    extract_image_metadata,
)


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


# ---------------------------------------------------------------------------
# _extract_xp_keywords
# ---------------------------------------------------------------------------


def test_xp_keywords_utf16le_bytes() -> None:
    """Should decode UTF-16LE semicolon-delimited XPKeywords."""
    raw = "sunset;beach;ocean".encode("utf-16-le") + b"\x00\x00"
    result = _extract_xp_keywords({"XPKeywords": raw})
    assert result == ["sunset", "beach", "ocean"]


def test_xp_keywords_plain_string() -> None:
    """Should split plain string XPKeywords by semicolon."""
    result = _extract_xp_keywords({"XPKeywords": "cat;dog;bird"})
    assert result == ["cat", "dog", "bird"]


def test_xp_keywords_missing() -> None:
    """Should return empty list when XPKeywords absent."""
    result = _extract_xp_keywords({"Make": "Canon"})
    assert result == []


def test_xp_keywords_empty_segments() -> None:
    """Should strip empty segments from XPKeywords."""
    result = _extract_xp_keywords({"XPKeywords": "cat;;;dog; ;bird"})
    assert result == ["cat", "dog", "bird"]


# ---------------------------------------------------------------------------
# _merge_keywords
# ---------------------------------------------------------------------------


def test_merge_keywords_deduplicates_case_insensitive() -> None:
    """Should deduplicate keywords case-insensitively, keeping first-seen form."""
    result = _merge_keywords(["Sunset", "beach"], ["sunset", "Ocean"], ["BEACH", "sky"])
    assert result == "Sunset, beach, Ocean, sky"


def test_merge_keywords_empty_sources() -> None:
    """Should return None when all sources are empty."""
    result = _merge_keywords([], [], [])
    assert result is None


def test_merge_keywords_single_source() -> None:
    """Should return comma-separated from a single source."""
    result = _merge_keywords(["a", "b", "c"])
    assert result == "a, b, c"


# ---------------------------------------------------------------------------
# extract_image_metadata — keywords / title / description priority
# ---------------------------------------------------------------------------


def test_extract_metadata_xp_keywords_from_exif(tmp_path: Path) -> None:
    """extract_image_metadata should populate keywords from XPKeywords in EXIF."""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(img_path)

    # Provide exif_data with XPKeywords pre-decoded (as _process_ifd0 would)
    exif_with_xp: dict[str | int, Any] = {
        "XPKeywords": "travel;landscape;mountain",
    }
    meta = extract_image_metadata(img_path, exif_data=exif_with_xp)
    assert meta.get("keywords") is not None
    assert "travel" in meta["keywords"]  # type: ignore[operator]
    assert "landscape" in meta["keywords"]  # type: ignore[operator]


def test_extract_metadata_description_prefers_iptc_caption(tmp_path: Path) -> None:
    """EXIF ImageDescription is used when no IPTC/XMP overrides exist."""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (10, 10), color="blue")
    img.save(img_path)

    exif_with_desc: dict[str | int, Any] = {
        "ImageDescription": "EXIF description",
    }
    meta = extract_image_metadata(img_path, exif_data=exif_with_desc)
    # On a plain JPEG with no IPTC/XMP, EXIF description is used
    assert meta["description"] == "EXIF description"


# ---------------------------------------------------------------------------
# _build_cataloguing_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_empty_metadata() -> None:
    """Prompt from empty metadata should still request cataloguing output."""
    prompt = _build_cataloguing_prompt({})
    assert "cataloguing" in prompt.lower()
    assert "Title:" in prompt
    assert "Description:" in prompt
    assert "Keywords:" in prompt


def test_build_prompt_includes_metadata_fields() -> None:
    """Prompt should incorporate all provided metadata fields."""
    meta: dict[str, str | None] = {
        "description": "Sunset over cliffs",
        "title": "Coastal Sunset",
        "keywords": "sunset, cliffs, ocean",
        "date": "2025-10-01",
        "time": "18:30",
        "gps": "51.0N, 0.9W",
    }
    prompt = _build_cataloguing_prompt(meta)
    assert "Sunset over cliffs" in prompt
    assert "Coastal Sunset" in prompt
    assert "sunset, cliffs, ocean" in prompt
    assert "2025-10-01" in prompt
    assert "18:30" in prompt
    assert "51.0N, 0.9W" in prompt


def test_build_prompt_context_marker_present() -> None:
    """Prompt with description should contain 'Context:' for quality analysis."""
    meta: dict[str, str | None] = {"description": "A red car"}
    prompt = _build_cataloguing_prompt(meta)
    assert "Context:" in prompt


def test_build_prompt_truncates_long_metadata_fields() -> None:
    """Large metadata fields should be compacted to avoid excessive prompt context."""
    long_desc = "detail " * 200
    long_keywords = ", ".join([f"keyword{i}" for i in range(60)])
    meta: dict[str, str | None] = {
        "title": "Very Long Existing Title " * 10,
        "description": long_desc,
        "keywords": long_keywords,
    }

    prompt = _build_cataloguing_prompt(meta)
    assert "Context:" in prompt
    assert long_desc not in prompt
    assert long_keywords not in prompt
    assert "..." in prompt
    assert "Keyword hints:" in prompt


def test_build_prompt_no_context_when_no_description() -> None:
    """Prompt without description/title/keywords should omit 'Context:' block."""
    meta: dict[str, str | None] = {"date": "2025-01-01"}
    prompt = _build_cataloguing_prompt(meta)
    assert "Context:" not in prompt
