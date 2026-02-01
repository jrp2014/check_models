"""New robust tests for EXIF extraction in check_models.py.

Verifies handling of:
- Malformed GPS coordinates
- Mixed text encodings (UserComment, UTF-8, Latin-1)
- Null-terminated strings
"""

# Import the functions directly if possible, or mock through the module
from check_models import (
    _convert_gps_coordinate,
    _decode_exif_string,  # We'll add this
    _extract_description,
    _extract_gps_str,
    to_float,
)


def test_to_float() -> None:
    assert to_float(1.5) == 1.5
    assert to_float("3.14") == 3.14
    assert to_float("invalid") is None
    assert to_float(None) is None


def test_convert_gps_coordinate_standard() -> None:
    # (deg, min, sec)
    result = _convert_gps_coordinate((37.0, 46.0, 30.5))
    assert result == (37.0, 46.0, 30.5)


def test_convert_gps_coordinate_rational() -> None:
    # Pillow often returns tuples (num, den) for each part
    # Mocking what Pillow's get_ifd might return
    result = _convert_gps_coordinate(((37, 1), (46, 1), (305, 10)))
    assert result == (37.0, 46.0, 30.5)


def test_convert_gps_coordinate_malformed() -> None:
    # 1-element is allowed by constants (degrees only)
    assert _convert_gps_coordinate((37.0,)) == (37.0, 0.0, 0.0)
    assert _convert_gps_coordinate((37.0, 46.0, 30.0, 1.0)) is None
    # Division by zero handled by to_float returning None
    assert _convert_gps_coordinate(((37, 0), (46, 1), (30, 1))) is None


def test_extract_gps_str_standard() -> None:
    gps_data = {
        "GPSLatitudeRef": "N",
        "GPSLatitude": (37.0, 46.0, 30.0),
        "GPSLongitudeRef": "W",
        "GPSLongitude": (122.0, 25.0, 6.0),
    }
    assert _extract_gps_str(gps_data) == "37.775000°N, 122.418333°W"


def test_extract_gps_str_bytes_ref() -> None:
    gps_data = {
        "GPSLatitudeRef": b"S",
        "GPSLatitude": (33.0, 51.0, 0.0),
        "GPSLongitudeRef": b"E",
        "GPSLongitude": (151.0, 12.0, 0.0),
    }
    assert _extract_gps_str(gps_data) == "33.850000°S, 151.200000°E"


def test_decode_exif_string_basic() -> None:
    # Plain ascii
    assert _decode_exif_string(b"Hello") == "Hello"
    # Null terminated
    assert _decode_exif_string(b"Hello\x00\x00") == "Hello"
    # UTF-8
    assert _decode_exif_string("© 2024".encode()) == "© 2024"
    # Latin-1 fallback
    assert _decode_exif_string(b"Copyright \xa9") == "Copyright ©"


def test_decode_exif_string_user_comment() -> None:
    # UserComment with ASCII prefix
    comment_ascii = b"ASCII\x00\x00\x00My comment"
    assert _decode_exif_string(comment_ascii) == "My comment"

    # UserComment with UNICODE (UTF-16) prefix
    comment_unicode = b"UNICODE\x00" + "My comment".encode("utf-16-be")
    assert _decode_exif_string(comment_unicode) == "My comment"


def test_extract_description_robustness() -> None:
    # Test handles mixed inputs
    assert _extract_description({"ImageDescription": "Test"}) == "Test"
    assert _extract_description({"ImageDescription": b"Test\x00"}) == "Test"
    assert _extract_description({"ImageDescription": None}) is None
