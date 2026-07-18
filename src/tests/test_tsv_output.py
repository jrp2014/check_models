"""Tests for TSV output generation."""

import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import check_models
from tools import safe_io


@dataclass
class MockGenerationResult:
    """Mock GenerationResult for testing."""

    text: str | None = "Generated text"
    prompt_tokens: int | None = 100
    generation_tokens: int | None = 50
    time: float | None = None
    active_memory: float | None = None
    cache_memory: float | None = None
    peak_memory: float | None = 1.0


def _read_tsv_record(path: Path) -> dict[str, str]:
    """Return the first data row from the literal TSV artifact."""
    content = safe_io.read_text_no_follow(path)
    data = "\n".join(line for line in content.splitlines() if not line.startswith("#"))
    rows = list(csv.reader(StringIO(data), delimiter="\t"))
    return dict(zip(rows[0], rows[1], strict=True))


def test_generate_tsv_report_basic(tmp_path: Path) -> None:
    """Should generate basic TSV report with headers and data."""
    # Create a simple test result
    results = [
        check_models.PerformanceResult(
            model_name="test/model-1",
            success=True,
            generation=MockGenerationResult(text="Test output"),
            total_time=1.5,
            generation_time=1.0,
            model_load_time=0.5,
        ),
    ]

    output_file = tmp_path / "test_output.tsv"
    check_models.generate_tsv_report(results, output_file)

    # Verify file was created
    assert output_file.exists()

    # Read and verify content
    content = safe_io.read_text_no_follow(output_file)
    # Skip metadata comment line (starts with #)
    data_lines = [ln for ln in content.strip().split("\n") if not ln.startswith("#")]

    # Should have at least header + 1 data row
    assert len(data_lines) >= 2

    # Verify it's tab-separated
    assert "\t" in data_lines[0]  # Header line
    assert "\t" in data_lines[1]  # Data line


def test_tsv_includes_working_set_percentage(tmp_path: Path) -> None:
    """TSV should expose the canonical peak-memory working-set percentage."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        success=True,
        generation=MockGenerationResult(peak_memory=1.0),
    )
    context = check_models._build_report_render_context(
        results=[result],
        prompt="test",
        system_info={},
        recommended_working_set_bytes=2_000_000_000,
    )
    output_file = tmp_path / "working-set.tsv"

    check_models.generate_tsv_report([result], output_file, report_context=context)

    record = _read_tsv_record(output_file)
    assert float(record["peak_memory_working_set_pct"]) == 50.0


def test_tsv_omits_working_set_percentage_without_denominator(
    tmp_path: Path,
) -> None:
    """An all-empty optional TSV column should not widen the spreadsheet."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        success=True,
        generation=MockGenerationResult(peak_memory=1.0),
    )
    context = check_models._build_report_render_context(
        results=[result],
        prompt="test",
        system_info={},
        recommended_working_set_bytes=None,
    )
    output_file = tmp_path / "no-working-set.tsv"

    check_models.generate_tsv_report([result], output_file, report_context=context)

    record = _read_tsv_record(output_file)
    assert "peak_memory_working_set_pct" not in record


def test_tsv_escapes_tabs_in_values(tmp_path: Path) -> None:
    """Should normalize tabs into visible spaces in exact generated text."""
    results = [
        check_models.PerformanceResult(
            model_name="test/model",
            success=True,
            generation=MockGenerationResult(text="Line with\ttab character"),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        ),
    ]

    output_file = tmp_path / "tabs.tsv"

    check_models.generate_tsv_report(results, output_file)
    record = _read_tsv_record(output_file)

    assert "Line with    tab character" in record["Generated Text"]
    assert "Output" not in record


def test_tsv_escapes_newlines_in_values(tmp_path: Path) -> None:
    r"""Should normalize embedded newlines into a single-line preview."""
    results = [
        check_models.PerformanceResult(
            model_name="test/model",
            success=True,
            generation=MockGenerationResult(text="Line 1\nLine 2\nLine 3"),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        ),
    ]

    output_file = tmp_path / "newlines.tsv"

    check_models.generate_tsv_report(results, output_file)
    content = safe_io.read_text_no_follow(output_file)

    # Content should be 3 lines: metadata comment + header + 1 data row
    data_lines = [ln for ln in content.strip().split("\n") if not ln.startswith("#")]
    assert len(data_lines) == 2

    record = _read_tsv_record(output_file)
    assert "Line 1\\nLine 2\\nLine 3" in record["Generated Text"]
    assert "Output" not in record


def test_tsv_removes_html_tags_from_headers(tmp_path: Path) -> None:
    """Should remove HTML tags like <br> from headers."""
    results = [
        check_models.PerformanceResult(
            model_name="test/model",
            success=True,
            generation=MockGenerationResult(text="output"),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        ),
    ]

    output_file = tmp_path / "headers.tsv"

    check_models.generate_tsv_report(results, output_file)
    content = safe_io.read_text_no_follow(output_file)

    # Header should not contain HTML tags (skip metadata comment line)
    data_lines = [ln for ln in content.split("\n") if not ln.startswith("#")]
    header_line = data_lines[0]
    assert "<br>" not in header_line
    assert "<" not in header_line
    assert ">" not in header_line


def test_tsv_handles_failed_results(tmp_path: Path) -> None:
    """Should handle failed results with error messages."""
    results = [
        check_models.PerformanceResult(
            model_name="test/failed-model",
            success=False,
            error_stage="load",
            error_message="Failed to load model",
            generation=None,
        ),
    ]

    output_file = tmp_path / "failed.tsv"

    check_models.generate_tsv_report(results, output_file)
    content = safe_io.read_text_no_follow(output_file)

    # Should have metadata comment + header + data row
    data_lines = [ln for ln in content.strip().split("\n") if not ln.startswith("#")]
    assert len(data_lines) == 2

    # Error message should be in the output
    assert "Error" in content or "Failed to load model" in content


def test_tsv_keeps_full_output_without_a_duplicate_preview(tmp_path: Path) -> None:
    """Compact TSV should retain exact evidence without a redundant preview column."""
    long_text = "Start of answer. " + ("filler text " * 40) + "TRAILING-SIGNAL"
    results = [
        check_models.PerformanceResult(
            model_name="test/model",
            success=True,
            generation=MockGenerationResult(text=long_text),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
            quality_issues="context-echo, reasoning-leak",
        ),
    ]

    output_file = tmp_path / "preview.tsv"

    check_models.generate_tsv_report(results, output_file)

    record = _read_tsv_record(output_file)
    assert record["Generated Text"] == long_text
    assert "Output" not in record


def test_tsv_uses_compact_caption_schema_without_diffusion_or_duplicate_owner(
    tmp_path: Path,
) -> None:
    """The spreadsheet contract should not expose irrelevant or duplicate columns."""
    result = check_models.PerformanceResult(
        model_name="test/caption-model",
        success=True,
        generation=MockGenerationResult(text="A cat on a pink sofa."),
        total_time=1.0,
    )
    output_file = tmp_path / "compact.tsv"

    check_models.generate_tsv_report([result], output_file)
    content = safe_io.read_text_no_follow(output_file)
    header = next(line for line in content.splitlines() if not line.startswith("#"))
    fields = [field.strip() for field in header.split("\t")]

    assert "error_package" not in fields
    assert "Error Package" not in fields
    assert "Output" not in fields
    assert "Generated Text" in fields
    assert "Diffusion Canvas Tokens" not in fields
    assert "Diffusion Denoising Steps" not in fields
    assert "Text Already Printed" not in fields


def test_tsv_is_unpadded_and_includes_canonical_statuses(tmp_path: Path) -> None:
    """Headers and values should be literal TSV cells with shared status semantics."""
    result = check_models.PerformanceResult(
        model_name="test/caption-model",
        success=True,
        generation=MockGenerationResult(text="A cat on a pink sofa."),
    )
    output_file = tmp_path / "literal.tsv"

    check_models.generate_tsv_report([result], output_file)

    lines = safe_io.read_text_no_follow(output_file).splitlines()
    header = lines[1]
    assert header == header.rstrip()
    assert "  \t" not in header
    record = _read_tsv_record(output_file)
    assert record["execution_status"] == "completed"
    assert record["recommendation_status"] == "recommended"
    assert record["compatibility_status"] == "clean"


def test_tsv_includes_canonical_prompt_burden_scalars(tmp_path: Path) -> None:
    """TSV should expose the same burden classification and measured dimensions."""
    prompt = "Describe this image briefly."
    analysis = check_models.analyze_generation_text(
        "Cat.",
        generated_tokens=3,
        prompt_tokens=4103,
        prompt=prompt,
    )
    result = check_models.PerformanceResult(
        model_name="test/visual-heavy",
        success=True,
        generation=MockGenerationResult(
            text="Cat.",
            prompt_tokens=4103,
            generation_tokens=3,
        ),
        quality_analysis=analysis,
        prompt_diagnostics=check_models.PromptDiagnostics(
            image_placeholder_count=1,
            processed_image_width=512,
            processed_image_height=384,
            image_patch_count=4,
        ),
    )
    output_file = tmp_path / "burden.tsv"

    check_models.generate_tsv_report([result], output_file)
    record = _read_tsv_record(output_file)

    assert record["prompt_burden_kind"] == "visual_input"
    assert record["prompt_burden_source"] == "estimated_nontext"
    assert record["processed_image_width"] == "512"
    assert record["processed_image_height"] == "384"
    assert record["image_patch_count"] == "4"


def test_tsv_includes_canonical_enrichment_compatibility_and_owner(tmp_path: Path) -> None:
    """TSV should expose the same additive facts as JSONL and history."""
    prompt = "Create title, description, and keywords."
    result = check_models.PerformanceResult(
        model_name="test/enriched",
        success=True,
        generation=MockGenerationResult(
            text="Title: Cat. Description: A cat on a chair. Keywords: cat, chair.",
            prompt_tokens=4100,
            generation_tokens=18,
        ),
        quality_analysis=check_models.analyze_generation_text(
            "Title: Cat. Description: A cat on a chair. Keywords: cat, chair.",
            generated_tokens=18,
            prompt_tokens=4100,
            prompt=prompt,
        ),
        prompt_diagnostics=check_models.PromptDiagnostics(image_placeholder_count=1),
        metadata_agreement=check_models.MetadataAgreementMetrics(
            overall_score=88.0,
            context_integration_score=81.0,
            draft_improvement_score=72.0,
            visual_description_score=91.0,
            assisted_enrichment_score=84.0,
        ),
    )
    context = check_models._build_report_render_context(
        results=[result],
        prompt=prompt,
        metadata={"description": "A cat on a chair."},
        eval_mode="assisted",
    )
    output_file = tmp_path / "aligned.tsv"

    check_models.generate_tsv_report([result], output_file, report_context=context)
    record = _read_tsv_record(output_file)

    assert record["compatibility_status"] == "clean"
    assert record["context_integration_score"] == "81"
    assert record["draft_improvement_score"] == "72"
    assert record["visual_description_score"] == "91"
    assert record["assisted_enrichment_score"] == "84"
    assert record["prompt_burden_kind"] == "visual_input"
    assert record["prompt_burden_source"] == "estimated_nontext"
    assert record["owner_confidence"] in {"high", "medium", "low"}


def test_tsv_uses_canonical_mixed_owner_failure_confidence(tmp_path: Path) -> None:
    """TSV confidence should match the canonical wrapped-failure narrative."""
    result = check_models.PerformanceResult(
        model_name="org/mixed-owner",
        generation=None,
        success=False,
        error_message="wrapped generation failure",
        error_package="mlx-vlm",
        exception_chain=(
            check_models.FailureException(
                "RuntimeError",
                "mlx.core",
                "kIOGPUCommandBufferCallbackErrorOutOfMemory",
            ),
            check_models.FailureException(
                "ValueError",
                "builtins",
                "mlx_vlm/generate.py wrapped generation failure",
            ),
        ),
    )
    context = check_models._build_report_render_context(
        results=[result],
        prompt="Describe the image.",
        eval_mode="blind",
    )
    output_file = tmp_path / "failure.tsv"

    check_models.generate_tsv_report([result], output_file, report_context=context)
    record = _read_tsv_record(output_file)
    narrative = check_models._build_failure_narrative(result)

    assert narrative.owner_confidence == "low"
    assert record["owner_confidence"] == narrative.owner_confidence


def test_tsv_empty_results(tmp_path: Path) -> None:
    """Should handle empty results list gracefully."""
    results: list[check_models.PerformanceResult] = []

    output_file = tmp_path / "empty.tsv"

    check_models.generate_tsv_report(results, output_file)
    # Should not create file or create empty file for empty results
    # Based on the implementation, it returns early if no results
    assert not output_file.exists() or output_file.stat().st_size == 0


def test_tsv_full_model_name(tmp_path: Path) -> None:
    """Should preserve the full model name (including organization) in TSV output."""
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

    output_file = tmp_path / "test_full_name.tsv"
    check_models.generate_tsv_report(results, output_file)

    content = safe_io.read_text_no_follow(output_file)
    # Skip metadata comment line
    data_lines = [ln for ln in content.strip().split("\n") if not ln.startswith("#")]

    # Check the data row (index 1, after header)
    data_row = data_lines[1]
    # The model name is typically the first column
    assert full_model_name in data_row
    # Ensure it wasn't truncated to just "specific-model-v1"
    assert f"\t{full_model_name}\t" in f"\t{data_row}\t" or data_row.startswith(
        f"{full_model_name}\t",
    )


def test_tsv_caps_oversized_cells(tmp_path: Path) -> None:
    """Compact TSV cells are capped while exact generated text is preserved."""
    long_text = "x" * 500
    results = [
        check_models.PerformanceResult(
            model_name="test/long-output",
            success=True,
            generation=MockGenerationResult(text=long_text),
            total_time=1.0,
        ),
    ]
    output_file = tmp_path / "results.tsv"
    check_models.generate_tsv_report(results, output_file)
    content = safe_io.read_text_no_follow(output_file)
    data_lines = [ln for ln in content.strip().split("\n") if not ln.startswith("#")]
    headers = [cell.strip() for cell in data_lines[0].split("\t")]
    generated_text_index = headers.index("Generated Text")
    record = _read_tsv_record(output_file)
    assert record["Generated Text"] == long_text

    # Compact/metadata cells stay capped; Generated Text is intentionally exact.
    max_cell = check_models.MAX_TSV_CELL_CHARS
    for line in data_lines[1:]:
        for index, cell in enumerate(line.split("\t")):
            if index == generated_text_index:
                continue
            assert len(cell) <= max_cell, f"Cell length {len(cell)} exceeds {max_cell}"
