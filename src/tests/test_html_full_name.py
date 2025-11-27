"""Tests for HTML output generation regarding full model names."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import check_models


@dataclass
class MockGenerationResult:
    """Mock GenerationResult for testing."""

    text: str | None = "Generated text"
    prompt_tokens: int | None = 100
    generation_tokens: int | None = 50
    time: float | None = None


def test_html_full_model_name(tmp_path: Path, monkeypatch: Any) -> None:
    """Should preserve the full model name (including organization) in HTML output."""
    full_model_name = "organization/specific-model-v1"
    results = [
        check_models.PerformanceResult(
            model_name=full_model_name,
            success=True,
            generation=MockGenerationResult(text="Output"),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        ),
    ]

    output_file = tmp_path / "test_full_name.html"

    # We need to mock versions and system info as they are required by generate_html_report
    versions = cast("check_models.LibraryVersionDict", {"mlx": "0.0.1"})
    prompt = "test prompt"
    total_runtime = 10.0

    # Mock get_system_characteristics to avoid system calls
    monkeypatch.setattr(
        check_models,
        "get_system_characteristics",
        lambda: {"OS": "TestOS"},
    )

    check_models.generate_html_report(
        results,
        output_file,
        versions,
        prompt,
        total_runtime,
    )

    content = output_file.read_text(encoding="utf-8")

    # Check if the full model name is present in the HTML
    # It should be in a table cell
    assert full_model_name in content

    # If we find the full name, we are good.
    assert full_model_name in content, f"Full model name {full_model_name} not found in HTML"
