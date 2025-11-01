"""Tests for error handling and edge cases."""

# ruff: noqa: ANN201
import sys
from pathlib import Path

import pytest

import check_models


def test_validate_inputs_rejects_nonexistent_folder():
    """Should raise SystemExit when folder does not exist."""
    with pytest.raises(SystemExit):
        check_models.validate_inputs(
            folder_path=Path("/nonexistent/folder"),
            temperature=0.7,
        )


def test_validate_inputs_rejects_negative_temperature():
    """Should raise SystemExit for negative temperature."""
    with pytest.raises(SystemExit):
        check_models.validate_inputs(
            folder_path=Path("."),
            temperature=-0.5,
        )


def test_validate_inputs_rejects_temperature_above_one():
    """Should raise SystemExit for temperature > 1.0."""
    with pytest.raises(SystemExit):
        check_models.validate_inputs(
            folder_path=Path("."),
            temperature=1.5,
        )


def test_find_most_recent_file_returns_none_for_empty_folder(tmp_path: Path):
    """Should return None when no images are in the folder."""
    result = check_models.find_most_recent_file(tmp_path)
    assert result is None


def test_validate_model_identifier_rejects_empty_string():
    """Should raise SystemExit for empty model identifier."""
    with pytest.raises(SystemExit):
        check_models.validate_model_identifier("")


def test_validate_model_identifier_rejects_whitespace():
    """Should raise SystemExit for whitespace-only identifier."""
    with pytest.raises(SystemExit):
        check_models.validate_model_identifier("   ")


def test_validate_kv_params_rejects_invalid_bits():
    """Should raise SystemExit for kv_bits not in [4, 8]."""
    with pytest.raises(SystemExit):
        check_models.validate_kv_params(kv_bits=16, max_kv_size=None)


def test_validate_kv_params_rejects_negative_max_kv_size():
    """Should raise SystemExit for negative max_kv_size."""
    with pytest.raises(SystemExit):
        check_models.validate_kv_params(kv_bits=4, max_kv_size=-1024)


def test_validate_kv_params_rejects_zero_max_kv_size():
    """Should raise SystemExit for zero max_kv_size."""
    with pytest.raises(SystemExit):
        check_models.validate_kv_params(kv_bits=4, max_kv_size=0)


def test_performance_result_handles_zero_time():
    """Should not crash when total_time is zero."""
    result = check_models.PerformanceResult(
        model_identifier="test/model",
        image_path=Path("test.jpg"),
        prompt="Test",
        response="Response",
        prompt_tps=0.0,
        generation_tps=0.0,
        total_time=0.0,
        peak_memory_gb=1.0,
        success=False,
        error_message="Timeout",
    )

    assert result.total_time == 0.0
    assert not result.success


def test_format_field_value_handles_none_values():
    """Should handle None values gracefully."""
    result = check_models.format_field_value("error_message", None)
    assert result in ["", "None", "-", "N/A"]


def test_format_field_value_handles_very_large_numbers():
    """Should format very large numbers appropriately."""
    result = check_models.format_field_value("generation_tps", 999999.99)
    assert "999999" in result or "1000000" in result or "1.0" in result


def test_format_field_value_handles_very_small_numbers():
    """Should format very small numbers appropriately."""
    result = check_models.format_field_value("prompt_tps", 0.00001)
    assert "0.0" in result or "0" in result


def test_extract_gps_coordinates_handles_malformed_data():
    """Should handle malformed GPS data gracefully."""
    exif_data = {
        "GPSInfo": {
            1: "N",
            # Missing latitude value
            3: "W",
            4: ((122, 1), (25, 1), (9, 1)),
        }
    }

    coords = check_models.extract_gps_coordinates(exif_data)
    # Should either return None or handle the error
    assert coords is None or isinstance(coords, tuple)


def test_get_exif_data_handles_nonexistent_file():
    """Should handle nonexistent files gracefully."""
    result = check_models.get_exif_data(Path("/nonexistent/image.jpg"))
    # Should return empty dict or None
    assert result == {} or result is None


def test_get_exif_data_handles_non_image_file(tmp_path: Path):
    """Should handle non-image files gracefully."""
    text_file = tmp_path / "not_an_image.txt"
    text_file.write_text("This is not an image")

    result = check_models.get_exif_data(text_file)
    # Should return empty dict or None
    assert result == {} or result is None
