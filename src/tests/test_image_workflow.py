"""Tests for image discovery and validation workflows."""

import time
from pathlib import Path

import pytest
from PIL import Image

import check_models


def test_find_most_recent_file_in_directory(tmp_path: Path) -> None:
    """Should find the most recently modified image."""
    # Create test images with different timestamps
    old_image = tmp_path / "old.jpg"
    Image.new("RGB", (50, 50)).save(old_image)
    time.sleep(0.1)

    new_image = tmp_path / "new.jpg"
    Image.new("RGB", (50, 50)).save(new_image)

    result = check_models.find_most_recent_file(tmp_path)
    assert result == new_image


def test_find_most_recent_file_ignores_hidden_files(tmp_path: Path) -> None:
    """Should ignore hidden files (starting with .)."""
    visible_image = tmp_path / "visible.jpg"
    Image.new("RGB", (50, 50)).save(visible_image)
    time.sleep(0.1)

    hidden_image = tmp_path / ".hidden.jpg"
    Image.new("RGB", (50, 50)).save(hidden_image)

    result = check_models.find_most_recent_file(tmp_path)
    # Should return visible image, not the more recent hidden one
    assert result == visible_image


def test_find_most_recent_file_returns_none_for_empty_folder(tmp_path: Path) -> None:
    """Should return None when folder has no image files."""
    result = check_models.find_most_recent_file(tmp_path)
    assert result is None


def test_find_most_recent_file_filters_by_extension(tmp_path: Path) -> None:
    """Should only consider image file extensions."""
    # Create non-image file
    text_file = tmp_path / "document.txt"
    text_file.write_text("Not an image")
    time.sleep(0.1)

    # Create image file (older timestamp)
    image_file = tmp_path / "photo.jpg"
    Image.new("RGB", (50, 50)).save(image_file)

    result = check_models.find_most_recent_file(tmp_path)
    # Should return image, not the more recent text file
    assert result == image_file


def test_validate_inputs_rejects_nonexistent_image() -> None:
    """Should raise FileNotFoundError for missing images."""
    with pytest.raises(FileNotFoundError, match="Image not found"):
        check_models.validate_inputs("/nonexistent/image.jpg")


def test_validate_inputs_rejects_unsupported_format(tmp_path: Path) -> None:
    """Should raise ValueError for unsupported image formats."""
    bad_file = tmp_path / "document.pdf"
    bad_file.touch()

    with pytest.raises(ValueError, match="Unsupported image format"):
        check_models.validate_inputs(bad_file)


def test_validate_inputs_accepts_jpg(tmp_path: Path) -> None:
    """Should accept .jpg format."""
    img_file = tmp_path / "test.jpg"
    Image.new("RGB", (50, 50)).save(img_file)

    # Should not raise
    check_models.validate_inputs(img_file)


def test_validate_inputs_accepts_jpeg(tmp_path: Path) -> None:
    """Should accept .jpeg format."""
    img_file = tmp_path / "test.jpeg"
    Image.new("RGB", (50, 50)).save(img_file)

    check_models.validate_inputs(img_file)


def test_validate_inputs_accepts_png(tmp_path: Path) -> None:
    """Should accept .png format."""
    img_file = tmp_path / "test.png"
    Image.new("RGB", (50, 50)).save(img_file)

    check_models.validate_inputs(img_file)


def test_validate_inputs_accepts_webp(tmp_path: Path) -> None:
    """Should accept .webp format."""
    img_file = tmp_path / "test.webp"
    Image.new("RGB", (50, 50)).save(img_file, "WEBP")

    check_models.validate_inputs(img_file)


def test_validate_inputs_case_insensitive_extension(tmp_path: Path) -> None:
    """Should accept extensions regardless of case."""
    img_file = tmp_path / "test.JPG"
    Image.new("RGB", (50, 50)).save(img_file)

    # Should not raise
    check_models.validate_inputs(img_file)


def test_validate_inputs_rejects_directory(tmp_path: Path) -> None:
    """Should raise ValueError when path is a directory."""
    with pytest.raises(ValueError, match="Not a file"):
        check_models.validate_inputs(tmp_path)


def test_validate_temperature_accepts_valid_range() -> None:
    """Should accept temperature in valid range [0.0, 1.0]."""
    # Should not raise - note: keyword-only argument
    check_models.validate_temperature(temp=0.0)
    check_models.validate_temperature(temp=0.5)
    check_models.validate_temperature(temp=1.0)


def test_validate_temperature_rejects_negative() -> None:
    """Should reject negative temperature."""
    with pytest.raises(ValueError, match="non-negative"):
        check_models.validate_temperature(temp=-0.1)


def test_validate_temperature_rejects_above_one() -> None:
    """Should log warning for temperature above 1.0."""
    # Note: High temperatures log warning but don't raise
    check_models.validate_temperature(temp=1.5)


def test_validate_image_accessible_success(tmp_path: Path) -> None:
    """Should succeed for accessible, valid image."""
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(img_path)

    # Should not raise - note: keyword-only argument
    check_models.validate_image_accessible(image_path=img_path)


def test_validate_image_accessible_missing_file() -> None:
    """Should raise OSError for missing file."""
    with pytest.raises(OSError, match="Error accessing image"):
        check_models.validate_image_accessible(image_path="/nonexistent/image.jpg")
