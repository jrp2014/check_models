"""Tests for report generation edge cases (empty input, all-failed results)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import check_models

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockGeneration:
    """Minimal stand-in for GenerationResult used by report generators."""

    text: str | None = "output"
    prompt_tokens: int | None = 10
    generation_tokens: int | None = 5
    time: float | None = None
    active_memory: float | None = None
    cache_memory: float | None = None


def _stub_versions() -> check_models.LibraryVersionDict:
    return {
        "numpy": "1.0",
        "mlx": "0.1",
        "mlx-metal": None,
        "mlx-vlm": "0.1",
        "mlx-lm": None,
        "huggingface-hub": "0.1",
        "transformers": "4.0",
        "tokenizers": "0.1",
        "Pillow": "10.0",
    }


def _make_success(name: str = "org/model-ok") -> check_models.PerformanceResult:
    return check_models.PerformanceResult(
        model_name=name,
        success=True,
        generation=_MockGeneration(),
        total_time=1.0,
        generation_time=0.5,
        model_load_time=0.5,
    )


def _make_failure(
    name: str = "org/model-fail",
    error_type: str = "ValueError",
    error_package: str = "mlx-vlm",
) -> check_models.PerformanceResult:
    return check_models.PerformanceResult(
        model_name=name,
        success=False,
        generation=None,
        error_stage="load",
        error_message="boom",
        error_type=error_type,
        error_package=error_package,
    )


# ===================================================================
# HTML report
# ===================================================================


class TestHtmlReportEdgeCases:
    """Edge-case coverage for generate_html_report."""

    def test_empty_results_does_not_write(self, tmp_path: Path) -> None:
        """Empty result list should produce no file."""
        out = tmp_path / "empty.html"
        check_models.generate_html_report(
            results=[],
            filename=out,
            versions=_stub_versions(),
            prompt="unused",
            total_runtime_seconds=0.0,
        )
        assert not out.exists()

    def test_all_failed_results_produces_file(self, tmp_path: Path) -> None:
        """All-failed result list should still produce a report."""
        out = tmp_path / "failed.html"
        check_models.generate_html_report(
            results=[_make_failure("org/a"), _make_failure("org/b")],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=5.0,
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "org/a" in content
        assert "org/b" in content

    def test_mixed_results_contains_both(self, tmp_path: Path) -> None:
        """Report with mixed success/failure should contain both models."""
        out = tmp_path / "mixed.html"
        check_models.generate_html_report(
            results=[_make_success("org/good"), _make_failure("org/bad")],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=3.0,
        )
        content = out.read_text(encoding="utf-8")
        assert "org/good" in content
        assert "org/bad" in content


# ===================================================================
# Markdown report
# ===================================================================


class TestMarkdownReportEdgeCases:
    """Edge-case coverage for generate_markdown_report."""

    def test_empty_results_does_not_write(self, tmp_path: Path) -> None:
        """Empty result list should produce no file."""
        out = tmp_path / "empty.md"
        check_models.generate_markdown_report(
            results=[],
            filename=out,
            versions=_stub_versions(),
            prompt="unused",
            total_runtime_seconds=0.0,
        )
        assert not out.exists()

    def test_all_failed_results_produces_file(self, tmp_path: Path) -> None:
        """All-failed result list should still produce a report."""
        out = tmp_path / "failed.md"
        check_models.generate_markdown_report(
            results=[_make_failure("org/c"), _make_failure("org/d")],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=4.0,
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "org/c" in content
        assert "org/d" in content

    def test_mixed_results_contains_both(self, tmp_path: Path) -> None:
        """Report with mixed success/failure should contain both models."""
        out = tmp_path / "mixed.md"
        check_models.generate_markdown_report(
            results=[_make_success("org/good"), _make_failure("org/bad")],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=2.0,
        )
        content = out.read_text(encoding="utf-8")
        assert "org/good" in content
        assert "org/bad" in content


# ===================================================================
# TSV report
# ===================================================================


class TestTsvReportEdgeCases:
    """Edge-case coverage for generate_tsv_report."""

    def test_empty_results_does_not_write(self, tmp_path: Path) -> None:
        """Empty result list should produce no file."""
        out = tmp_path / "empty.tsv"
        check_models.generate_tsv_report(results=[], filename=out)
        assert not out.exists()

    def test_all_failed_results_produces_file(self, tmp_path: Path) -> None:
        """All-failed result list should still produce a report."""
        out = tmp_path / "failed.tsv"
        check_models.generate_tsv_report(
            results=[_make_failure("org/e"), _make_failure("org/f")],
            filename=out,
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "org/e" in content
        assert "org/f" in content

    def test_tsv_has_metadata_comment(self, tmp_path: Path) -> None:
        """TSV output should start with a generated_at metadata comment."""
        out = tmp_path / "meta.tsv"
        check_models.generate_tsv_report(
            results=[_make_success()],
            filename=out,
        )
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("# generated_at:")

    def test_tsv_has_error_columns(self, tmp_path: Path) -> None:
        """TSV should include error_type and error_package columns."""
        out = tmp_path / "cols.tsv"
        check_models.generate_tsv_report(
            results=[_make_failure(error_type="RuntimeError", error_package="transformers")],
            filename=out,
        )
        content = out.read_text(encoding="utf-8")
        # Header row (skip the comment line)
        header_line = content.splitlines()[1]
        assert "error_type" in header_line
        assert "error_package" in header_line
        # Data row
        data_line = content.splitlines()[2]
        assert "RuntimeError" in data_line
        assert "transformers" in data_line

    def test_tsv_error_columns_empty_for_success(self, tmp_path: Path) -> None:
        """Successful models should have empty error columns in the header."""
        out = tmp_path / "ok.tsv"
        check_models.generate_tsv_report(
            results=[_make_success()],
            filename=out,
        )
        content = out.read_text(encoding="utf-8")
        # The header must advertise error_type / error_package columns
        header_line = content.splitlines()[1]
        assert "error_type" in header_line
        assert "error_package" in header_line
        # For a success row, those columns are empty strings; tabulate may
        # trim trailing whitespace-only fields, so just verify the data row
        # does NOT contain a populated error value.
        data_line = content.splitlines()[2]
        stripped_fields = [f.strip() for f in data_line.split("\t")]
        # error_type and error_package should not contain real values
        assert "RuntimeError" not in stripped_fields
        assert "transformers" not in stripped_fields
