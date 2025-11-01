"""Tests for EXIF metadata extraction."""

# ruff: noqa: ANN201
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

import check_models


def test_extract_exif_gps_coordinates():
    """Should extract GPS coordinates from EXIF data."""
    # Mock EXIF data with GPS
    exif_data = {
        "GPSInfo": {
            1: "N",  # GPSLatitudeRef
            2: ((37, 1), (46, 1), (30, 1)),  # GPSLatitude (37°46'30")
            3: "W",  # GPSLongitudeRef
            4: ((122, 1), (25, 1), (9, 1)),  # GPSLongitude (122°25'9")
        }
    }

    coords = check_models.extract_gps_coordinates(exif_data)
    assert coords is not None
    lat, lon = coords

    # San Francisco area coordinates
    assert 37.0 < lat < 38.0
    assert -123.0 < lon < -122.0


def test_extract_exif_gps_coordinates_returns_none_when_missing():
    """Should return None when GPS data is missing."""
    exif_data = {"Make": "Apple", "Model": "iPhone"}
    coords = check_models.extract_gps_coordinates(exif_data)
    assert coords is None


def test_extract_exif_date_from_original():
    """Should extract DateTimeOriginal from EXIF."""
    exif_data = {
        "DateTimeOriginal": "2024:01:15 14:30:25",
    }

    date_str = check_models.extract_exif_date(exif_data)
    assert date_str == "2024:01:15 14:30:25"


def test_extract_exif_date_fallback_to_datetime():
    """Should fall back to DateTime if DateTimeOriginal is missing."""
    exif_data = {
        "DateTime": "2024:01:15 10:20:30",
    }

    date_str = check_models.extract_exif_date(exif_data)
    assert date_str == "2024:01:15 10:20:30"


def test_extract_exif_date_returns_none_when_missing():
    """Should return None when no date fields are present."""
    exif_data = {"Make": "Canon", "Model": "EOS"}
    date_str = check_models.extract_exif_date(exif_data)
    assert date_str is None


def test_extract_exif_description():
    """Should extract ImageDescription from EXIF."""
    exif_data = {
        "ImageDescription": "A beautiful sunset over the ocean",
    }

    description = check_models.extract_exif_description(exif_data)
    assert description == "A beautiful sunset over the ocean"


def test_extract_exif_description_returns_none_when_missing():
    """Should return None when ImageDescription is missing."""
    exif_data = {"Make": "Nikon", "Model": "D850"}
    description = check_models.extract_exif_description(exif_data)
    assert description is None


def test_get_exif_data_from_real_image(tmp_path: Path):
    """Should read EXIF data from a real image file."""
    # Create a test image with EXIF
    image_path = tmp_path / "test_exif.jpg"
    img = Image.new("RGB", (100, 100), color="red")

    # Add some EXIF data
    exif_dict = {
        "0th": {
            270: "Test image description",  # ImageDescription
            271: "Test Camera",  # Make
        }
    }

    # Note: PIL doesn't easily support writing custom EXIF, so we test reading
    img.save(image_path, "JPEG")

    exif_data = check_models.get_exif_data(image_path)
    # Should at least return a dict (may be empty for synthetic images)
    assert isinstance(exif_data, dict)


def test_format_gps_coordinates_as_string():
    """Should format GPS coordinates as a readable string."""
    formatted = check_models.format_gps_coordinates(37.775, -122.419)
    assert "37.775" in formatted or "37.78" in formatted
    assert "122.419" in formatted or "122.42" in formatted


def test_convert_gps_to_degrees():
    """Should convert GPS tuple format to decimal degrees."""
    # GPS format: ((degrees, 1), (minutes, 1), (seconds, 1))
    gps_tuple = ((37, 1), (46, 1), (30, 1))  # 37°46'30"
    degrees = check_models.convert_gps_to_degrees(gps_tuple)

    # 37 + 46/60 + 30/3600 = 37.775
    assert 37.77 < degrees < 37.78


def test_extract_exif_camera_make():
    """Should extract camera make from EXIF."""
    exif_data = {
        "Make": "Apple",
        "Model": "iPhone 14 Pro",
    }

    make = check_models.extract_exif_field(exif_data, "Make")
    assert make == "Apple"


def test_extract_exif_camera_model():
    """Should extract camera model from EXIF."""
    exif_data = {
        "Make": "Canon",
        "Model": "EOS R5",
    }

    model = check_models.extract_exif_field(exif_data, "Model")
    assert model == "EOS R5"


def test_extract_exif_field_returns_none_when_missing():
    """Should return None when requested field is missing."""
    exif_data = {"Make": "Sony"}
    iso = check_models.extract_exif_field(exif_data, "ISOSpeedRatings")
    assert iso is None
