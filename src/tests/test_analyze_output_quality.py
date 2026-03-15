"""Tests for the analyze_output_quality CLI tool."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from tools.analyze_output_quality import main

if TYPE_CHECKING:
    from pathlib import Path


def test_analyze_output_quality_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the help text is displayed correctly."""
    # Temporarily override sys.argv to simulate passing --help
    original_argv = sys.argv
    sys.argv = ["analyze_output_quality.py", "--help"]

    try:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr()
    assert "Test the project's VLM quality heuristics" in captured.out


def test_analyze_output_quality_clean_text(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that clean text returns exit code 0 and reports no issues."""
    original_argv = sys.argv
    sys.argv = [
        "analyze_output_quality.py",
        "--text",
        "This is a normal sentence about a beautiful landscape and trees.",
    ]

    try:
        exit_code = main()
        assert exit_code == 0
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr()
    assert "CLEAN (No issues detected)" in captured.out
    assert "Has Harness Issue         : ❌ No" in captured.out


def test_analyze_output_quality_harness_issue(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that a stop token leak triggers a harness issue and returns exit code 1."""
    original_argv = sys.argv
    sys.argv = [
        "analyze_output_quality.py",
        "--text",
        "This text has a leaked boundary token <|endoftext|>",
    ]

    try:
        exit_code = main()
        assert exit_code == 1
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr()
    assert "Has Harness Issue         : ✅ Yes" in captured.out
    assert "Harness Issue Type        : stop_token" in captured.out
    assert "⚠️harness" in captured.out


def test_analyze_output_quality_with_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test reading from a file returns correct analysis."""
    test_file = tmp_path / "test_output.txt"
    test_file.write_text(
        "Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. Repetitive output. ",
        encoding="utf-8",
    )

    original_argv = sys.argv
    sys.argv = [
        "analyze_output_quality.py",
        "--file",
        str(test_file),
    ]

    try:
        exit_code = main()
        assert exit_code == 1
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr()
    assert "Is Repetitive             : ✅ Yes" in captured.out


def test_analyze_output_quality_with_prompt_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test providing a prompt file triggers prompt context checks."""
    test_file = tmp_path / "test_output.txt"
    test_file.write_text(
        "Title: A misty lake\nDescription: A misty lake scene.\nKeywords: lake, mist, water, nature, outdoors, scenery, trees, fog, cold, wet",
        encoding="utf-8",
    )

    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text(
        "Context: Existing metadata hints:\n- Title hint: Misty lake scene",
        encoding="utf-8",
    )

    original_argv = sys.argv
    sys.argv = [
        "analyze_output_quality.py",
        "--file",
        str(test_file),
        "--prompt-file",
        str(prompt_file),
    ]

    try:
        exit_code = main()
        assert exit_code == 0
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr()
    assert "Prompt Checks Ran         : ✅ Yes" in captured.out
    assert "CLEAN" in captured.out
