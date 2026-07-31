"""Tests for EXIF extraction utilities."""

import io
from pathlib import Path
from typing import Any, Self

import pytest
from PIL import Image

import check_models
from check_models import (
    EXIF_NOT_EXTRACTED,
    _build_cataloguing_prompt,
    _extract_xp_keywords,
    _merge_keywords,
    extract_image_metadata,
)


class _FakeExifWithSubIfd:
    def get_ifd(self, tag: object) -> dict[object, object] | None:
        _ = tag
        return {36867: "2024:01:10 10:20:30", "custom": "value"}


class _FakeUrlImage:
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        _ = (exc_type, exc_value, traceback)

    def getexif(self) -> dict[int, Any]:
        return {270: "Remote description"}


class _FakeUrlResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        _ = (exc_type, exc_value, traceback)

    def read(self) -> bytes:
        return self._payload


def test_extract_exif_date_standard_format() -> None:
    """Should parse standard EXIF datetime format."""
    exif_dict: dict[str | int, Any] = {"DateTime": "2024:01:15 14:30:45"}
    result, _ = check_models._extract_exif_datetime(exif_dict)
    assert result is not None
    assert "2024-01-15" in result
    assert "14:30:45" in result


def test_extract_exif_datetime_preserves_wall_clock_with_declared_offset() -> None:
    """An EXIF offset describes the recorded wall clock; it must not shift it."""
    exif_dict: dict[str | int, Any] = {
        "DateTimeOriginal": "2026:07:25 18:33:16",
        "OffsetTimeOriginal": "+01:00",
    }

    assert check_models._extract_exif_datetime(exif_dict) == (
        "2026-07-25 18:33:16 UTC+01:00",
        "18:33:16",
    )


def test_process_exif_subifd_handles_non_int_tag_ids() -> None:
    """Unknown non-integer sub-IFD keys should fall back to their string form."""
    result = check_models._process_exif_subifd(_FakeExifWithSubIfd())
    assert result["DateTimeOriginal"] == "2024:01:10 10:20:30"
    assert result["custom"] == "value"


def test_decode_gps_tag_key_handles_int_string_and_unknown_values() -> None:
    """GPS tag keys should decode known ids and preserve unknown identifiers."""
    assert check_models._decode_gps_tag_key(1) == "GPSLatitudeRef"
    assert check_models._decode_gps_tag_key("2") == "GPSLatitude"
    assert check_models._decode_gps_tag_key(999999) == "999999"


def test_get_exif_data_downloads_http_image_with_urllib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP image EXIF extraction should use urllib and open the in-memory payload."""

    def fake_urlopen(url: str, timeout: int) -> _FakeUrlResponse:
        assert url == "https://example.com/test.jpg"
        assert timeout == 30
        return _FakeUrlResponse(b"remote-image-bytes")

    def fake_image_open(image_data: object) -> _FakeUrlImage:
        assert isinstance(image_data, io.BytesIO)
        assert image_data.getvalue() == b"remote-image-bytes"
        return _FakeUrlImage()

    monkeypatch.setattr(check_models, "urlopen", fake_urlopen)
    monkeypatch.setattr(Image, "open", fake_image_open)

    exif_data = check_models.get_exif_data("https://example.com/test.jpg")

    assert exif_data is not None
    assert exif_data["ImageDescription"] == "Remote description"


def test_extract_exif_date_datetime_original() -> None:
    """Should prefer DateTimeOriginal over DateTime."""
    exif_dict: dict[str | int, Any] = {
        "DateTime": "2024:01:15 14:30:45",
        "DateTimeOriginal": "2024:01:10 10:20:30",
    }
    result, _ = check_models._extract_exif_datetime(exif_dict)
    assert result is not None
    # Should use DateTimeOriginal (Jan 10) not DateTime (Jan 15)
    assert "2024-01-10" in result


def test_extract_exif_date_create_date() -> None:
    """Should use CreateDate when DateTimeOriginal absent."""
    exif_dict: dict[str | int, Any] = {
        "CreateDate": "2024:01:12 12:00:00",
        "DateTime": "2024:01:15 14:30:45",
    }
    result, _ = check_models._extract_exif_datetime(exif_dict)
    assert result is not None
    # Should use CreateDate (Jan 12), which has priority over DateTime
    assert "2024-01-12" in result


def test_pillow_datetime_digitized_precedes_generic_datetime(tmp_path: Path) -> None:
    """Pillow's real digitized-date tag should supply the capture wall clock."""
    test_file = tmp_path / "digitized.jpg"
    exif = Image.Exif()
    exif[0x0132] = "2024:01:15 14:30:45"
    exif[0x9004] = "2024:01:12 12:00:00"
    exif[0x9012] = "+01:00"
    Image.new("RGB", (2, 2), color="white").save(test_file, exif=exif)

    decoded = check_models.get_exif_data(test_file)

    assert decoded is not None
    assert decoded["DateTimeDigitized"] == "2024:01:12 12:00:00"
    assert decoded["OffsetTimeDigitized"] == "+01:00"
    assert check_models._extract_exif_datetime(decoded) == (
        "2024-01-12 12:00:00 UTC+01:00",
        "12:00:00",
    )


def test_extract_exif_date_omits_unknown_file_mtime(tmp_path: Path) -> None:
    """Filesystem modification time must not masquerade as capture metadata."""
    test_file = tmp_path / "test.jpg"
    Image.new("RGB", (2, 2), color="white").save(test_file)
    exif_dict: dict[str | int, Any] = {"Make": "Camera", "Model": "Test"}

    assert check_models._extract_exif_datetime(exif_dict) == (None, None)

    metadata = extract_image_metadata(test_file, exif_data=EXIF_NOT_EXTRACTED)
    metadata["description"] = "Two cats resting indoors."
    prompt = _build_cataloguing_prompt(metadata)

    assert metadata["date"] is None
    assert metadata["time"] is None
    assert "Capture date/time:" not in prompt


def test_extract_exif_date_invalid_format() -> None:
    """An invalid EXIF datetime is unknown and must not enter the prompt."""
    exif_dict: dict[str | int, Any] = {"DateTime": "invalid date"}

    assert check_models._extract_exif_datetime(exif_dict) == (None, None)


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

    # Provide exif_data with XPKeywords pre-decoded (as IFD0 pass would)
    exif_with_xp: dict[str | int, Any] = {
        "XPKeywords": "travel;landscape;mountain",
    }
    meta = extract_image_metadata(img_path, exif_data=exif_with_xp)
    keywords = meta.get("keywords")
    assert isinstance(keywords, str)
    assert "travel" in keywords
    assert "landscape" in keywords


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


def test_extract_xmp_metadata_reads_pillow_normalized_keys(tmp_path: Path) -> None:
    """Pillow's namespace-stripped getxmp() shape should retain catalog metadata."""
    img_path = tmp_path / "xmp.jpg"
    xmp = b"""<?xpacket begin="\xef\xbb\xbf"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
           xmlns:dc="http://purl.org/dc/elements/1.1/">
    <rdf:Description>
      <dc:title><rdf:Alt><rdf:li xml:lang="x-default">Probe title</rdf:li></rdf:Alt></dc:title>
      <dc:description><rdf:Alt><rdf:li xml:lang="x-default">Probe description</rdf:li></rdf:Alt></dc:description>
      <dc:subject><rdf:Bag><rdf:li>probe-keyword</rdf:li><rdf:li>second</rdf:li></rdf:Bag></dc:subject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    Image.new("RGB", (10, 10), color="green").save(img_path, xmp=xmp)

    with Image.open(img_path) as image:
        assert "RDF" in image.getxmp()["xmpmeta"]

    assert check_models._extract_xmp_metadata(img_path) == {
        "xmp_keywords": ["probe-keyword", "second"],
        "xmp_description": "Probe description",
        "xmp_title": "Probe title",
    }


def test_extract_image_metadata_respects_known_absent_exif_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Known-missing EXIF should not trigger a second image probe."""
    img_path = tmp_path / "plain.jpg"
    Image.new("RGB", (10, 10), color="white").save(img_path)

    probe_count = 0

    def fake_get_exif_data(_image_path: Path) -> dict[str, str]:
        nonlocal probe_count
        probe_count += 1
        return {"ImageDescription": "unexpected"}

    monkeypatch.setattr(check_models, "get_exif_data", fake_get_exif_data)

    metadata = extract_image_metadata(img_path, exif_data=EXIF_NOT_EXTRACTED)

    assert probe_count == 0
    assert metadata["description"] is None
    assert metadata["exif"] == "{}"


# ---------------------------------------------------------------------------
# _build_cataloguing_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_blind_lane_is_concise_and_ends_with_output_schema() -> None:
    """Blind prompts should spend their small text budget only on visible evidence."""
    prompt = _build_cataloguing_prompt({}, include_metadata_hints=False)

    assert "catalogue metadata" in prompt.lower()
    assert "authoritative" not in prompt.lower()
    assert "Context:" not in prompt
    assert "Rules:" not in prompt
    assert len(prompt.split()) <= 90
    assert prompt.endswith(
        "Return exactly these three sections and nothing else:\nTitle:\nDescription:\nKeywords:"
    )


def test_build_prompt_includes_metadata_fields() -> None:
    """Assisted prompts should distinguish authoritative facts from descriptive hints."""
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
    assert "Descriptive hints:" in prompt
    assert "Title hint: Coastal Sunset" in prompt
    assert "Description hint: Sunset over cliffs" in prompt
    assert "Keyword hints: sunset, cliffs, ocean" in prompt
    assert "hints may be incomplete or wrong" in prompt
    assert "Existing title:" not in prompt
    assert len(prompt.split()) <= 160
    assert prompt.index("Context: Authoritative context:") < prompt.index("Descriptive hints:")
    assert prompt.index("Descriptive hints:") < prompt.index("Write:")
    assert prompt.endswith(
        "Return exactly these three sections and nothing else:\nTitle:\nDescription:\nKeywords:"
    )


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


def test_build_prompt_date_only_uses_authoritative_context() -> None:
    """Capture dates should be presented as authoritative context."""
    meta: dict[str, str | None] = {"date": "2025-01-01"}
    prompt = _build_cataloguing_prompt(meta)
    assert "Context: Authoritative context:" in prompt
