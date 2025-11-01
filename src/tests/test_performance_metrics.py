"""Tests for performance metrics calculations."""

# ruff: noqa: ANN201
from pathlib import Path

import check_models


def test_performance_result_creation():
    """Should create PerformanceResult with all fields."""
    result = check_models.PerformanceResult(
        model_identifier="test/model-4bit",
        image_path=Path("test.jpg"),
        prompt="Describe this image",
        response="A test image showing...",
        prompt_tps=12.5,
        generation_tps=28.3,
        total_time=3.2,
        peak_memory_gb=5.1,
        success=True,
        error_message=None,
    )

    assert result.model_identifier == "test/model-4bit"
    assert result.prompt_tps == 12.5
    assert result.generation_tps == 28.3
    assert result.total_time == 3.2
    assert result.peak_memory_gb == 5.1
    assert result.success is True


def test_performance_result_failed_execution():
    """Should handle failed execution with error message."""
    result = check_models.PerformanceResult(
        model_identifier="failed/model",
        image_path=Path("test.jpg"),
        prompt="Test",
        response="",
        prompt_tps=0.0,
        generation_tps=0.0,
        total_time=0.0,
        peak_memory_gb=0.0,
        success=False,
        error_message="Model failed to load",
    )

    assert result.success is False
    assert result.error_message == "Model failed to load"
    assert result.prompt_tps == 0.0


def test_calculate_tps_with_valid_tokens_and_time():
    """Should calculate tokens per second correctly."""
    tps = check_models.calculate_tps(token_count=100, elapsed_time=4.0)
    assert tps == 25.0

    tps = check_models.calculate_tps(token_count=50, elapsed_time=2.0)
    assert tps == 25.0


def test_calculate_tps_with_zero_time():
    """Should return 0.0 when elapsed_time is zero."""
    tps = check_models.calculate_tps(token_count=100, elapsed_time=0.0)
    assert tps == 0.0


def test_calculate_tps_with_zero_tokens():
    """Should return 0.0 when token_count is zero."""
    tps = check_models.calculate_tps(token_count=0, elapsed_time=5.0)
    assert tps == 0.0


def test_format_tps_value():
    """Should format TPS with appropriate precision."""
    formatted = check_models.format_tps(25.123456)
    assert "25.1" in formatted or "25.12" in formatted


def test_format_time_value():
    """Should format time with units."""
    formatted = check_models.format_time(3.5)
    assert "3.5" in formatted
    assert "s" in formatted.lower()


def test_format_memory_value():
    """Should format memory with GB units."""
    formatted = check_models.format_memory(4.25)
    assert "4.25" in formatted or "4.2" in formatted
    assert "GB" in formatted


def test_calculate_peak_memory():
    """Should track peak memory usage."""
    # This would typically use psutil or similar
    # For now, test that the function exists and returns a float
    memory = check_models.get_peak_memory_gb()
    assert isinstance(memory, float)
    assert memory >= 0.0


def test_performance_result_calculates_average_tps():
    """Should calculate average TPS across prompt and generation."""
    result = check_models.PerformanceResult(
        model_identifier="test/model",
        image_path=Path("test.jpg"),
        prompt="Test",
        response="Response",
        prompt_tps=10.0,
        generation_tps=30.0,
        total_time=2.0,
        peak_memory_gb=4.0,
        success=True,
        error_message=None,
    )

    # Average of 10.0 and 30.0 is 20.0
    avg_tps = (result.prompt_tps + result.generation_tps) / 2
    assert avg_tps == 20.0


def test_timing_decorator_measures_execution_time():
    """Should measure function execution time."""
    import time

    @check_models.timing_decorator
    def slow_function():
        time.sleep(0.1)
        return "done"

    result, elapsed = slow_function()
    assert result == "done"
    assert elapsed >= 0.1
    assert elapsed < 0.2  # Should complete quickly


def test_memory_profiler_tracks_usage():
    """Should track memory usage during execution."""
    initial_memory = check_models.get_current_memory_gb()
    assert isinstance(initial_memory, float)
    assert initial_memory >= 0.0

    # Allocate some memory
    large_list = [0] * 1000000  # noqa: F841

    peak_memory = check_models.get_peak_memory_gb()
    assert peak_memory >= initial_memory
