"""Tests for result sorting behavior."""

from dataclasses import dataclass

import check_models


@dataclass
class MockGenerationResult:
    """Mock GenerationResult for testing."""

    text: str | None = "Generated text"
    prompt_tokens: int | None = 100
    generation_tokens: int | None = 50
    generation_tps: float | None = 10.0
    time: float | None = None
    active_memory: float | None = None
    cache_memory: float | None = None


def test_sort_results_by_time_failures_come_first() -> None:
    """Failed results should appear first in sorted output."""
    # Create mix of successful and failed results
    results = [
        check_models.PerformanceResult(
            model_name="model_fast",
            generation=MockGenerationResult(),
            success=True,
            generation_time=1.0,
            total_time=2.0,
        ),
        check_models.PerformanceResult(
            model_name="model_failed",
            generation=None,
            success=False,
            error_stage="processing",
            error_message="Test error",
        ),
        check_models.PerformanceResult(
            model_name="model_slow",
            generation=MockGenerationResult(),
            success=True,
            generation_time=5.0,
            total_time=6.0,
        ),
    ]

    sorted_results = check_models._sort_results_by_time(results)

    # Failed result should be first
    assert not sorted_results[0].success
    assert sorted_results[0].model_name == "model_failed"

    # Successful results should follow, sorted by generation time
    assert sorted_results[1].success
    assert sorted_results[1].model_name == "model_fast"
    assert sorted_results[2].success
    assert sorted_results[2].model_name == "model_slow"


def test_sort_results_by_time_ascending_order() -> None:
    """Successful results should be sorted by generation time (fastest first)."""
    results = [
        check_models.PerformanceResult(
            model_name="model_slow",
            generation=MockGenerationResult(),
            success=True,
            generation_time=5.0,
            total_time=6.0,
        ),
        check_models.PerformanceResult(
            model_name="model_medium",
            generation=MockGenerationResult(),
            success=True,
            generation_time=3.0,
            total_time=4.0,
        ),
        check_models.PerformanceResult(
            model_name="model_fast",
            generation=MockGenerationResult(),
            success=True,
            generation_time=1.0,
            total_time=2.0,
        ),
    ]

    sorted_results = check_models._sort_results_by_time(results)

    # Should be sorted fastest to slowest
    assert sorted_results[0].model_name == "model_fast"
    assert sorted_results[1].model_name == "model_medium"
    assert sorted_results[2].model_name == "model_slow"


def test_sort_results_by_time_multiple_failures() -> None:
    """Multiple failed results should all appear before successes."""
    results = [
        check_models.PerformanceResult(
            model_name="model_success_1",
            generation=MockGenerationResult(),
            success=True,
            generation_time=2.0,
            total_time=3.0,
        ),
        check_models.PerformanceResult(
            model_name="model_failed_1",
            generation=None,
            success=False,
            error_stage="timeout",
        ),
        check_models.PerformanceResult(
            model_name="model_failed_2",
            generation=None,
            success=False,
            error_stage="loading",
        ),
        check_models.PerformanceResult(
            model_name="model_success_2",
            generation=MockGenerationResult(),
            success=True,
            generation_time=1.0,
            total_time=2.0,
        ),
    ]

    sorted_results = check_models._sort_results_by_time(results)

    # First two should be failures
    assert not sorted_results[0].success
    assert not sorted_results[1].success

    # Last two should be successes, sorted by time
    assert sorted_results[2].success
    assert sorted_results[2].model_name == "model_success_2"  # faster
    assert sorted_results[3].success
    assert sorted_results[3].model_name == "model_success_1"  # slower


def test_sort_results_by_time_missing_timing_data() -> None:
    """Results without timing data should be sorted to end (after timed results)."""
    results = [
        check_models.PerformanceResult(
            model_name="model_with_time",
            generation=MockGenerationResult(),
            success=True,
            generation_time=2.0,
            total_time=3.0,
        ),
        check_models.PerformanceResult(
            model_name="model_no_time",
            generation=MockGenerationResult(),
            success=True,
            generation_time=None,  # Missing timing
            total_time=None,
        ),
    ]

    sorted_results = check_models._sort_results_by_time(results)

    # Result with timing should come first
    assert sorted_results[0].model_name == "model_with_time"
    # Result without timing should come last
    assert sorted_results[1].model_name == "model_no_time"


def test_sort_results_by_time_empty_list() -> None:
    """Empty list should return empty list."""
    results: list[check_models.PerformanceResult] = []
    sorted_results = check_models._sort_results_by_time(results)
    assert sorted_results == []


def test_sort_results_by_time_single_result() -> None:
    """Single result should return unchanged."""
    results = [
        check_models.PerformanceResult(
            model_name="only_model",
            generation=MockGenerationResult(),
            success=True,
            generation_time=1.0,
            total_time=2.0,
        ),
    ]

    sorted_results = check_models._sort_results_by_time(results)

    assert len(sorted_results) == 1
    assert sorted_results[0].model_name == "only_model"
