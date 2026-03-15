"""Standalone script to analyze arbitrary output with project quality heuristics.

This lets developers inspect check_models quality and harness detection without
running a local MLX model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_models import (
    GenerationQualityAnalysis,
    _build_quality_issues_string,
    analyze_generation_text,
    load_quality_config,
)


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for output quality analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Test the project's VLM quality heuristics on arbitrary text. "
            "Useful for debugging tokenizer artifacts, degeneration, and contract rules."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text string to evaluate directly.")
    group.add_argument("--file", type=Path, help="File containing text to evaluate.")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Optional prompt string used for context-echo and contract checks.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="File containing prompt text. Mutually exclusive with --prompt.",
    )
    parser.add_argument(
        "--context-marker",
        type=str,
        default="Context:",
        help="Prompt section marker where factual context begins.",
    )
    return parser


def _print_field(label: str, value: object) -> None:
    """Print one analysis field with simple alignment."""
    if isinstance(value, bool):
        rendered_value = "✅ Yes" if value else "❌ No"
    elif isinstance(value, list):
        if not value:
            rendered_value = "None"
        elif len(value) == 1:
            rendered_value = f"[{value[0]}]"
        else:
            rendered_value = "\n    " + "\n    ".join(f"- {item}" for item in value)
    elif value is None:
        rendered_value = "N/A"
    else:
        rendered_value = str(value)

    print(f"  {label:<25} : {rendered_value}")


def _read_text_file(path: Path, *, label: str) -> str | None:
    """Read UTF-8 text from a file path and report a CLI-friendly error."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error reading {label}: {exc}")
        return None


def _resolve_output_text(args: argparse.Namespace) -> str | None:
    """Resolve the output text from CLI arguments."""
    if args.file is None:
        return args.text
    return _read_text_file(args.file, label="output text file")


def _resolve_prompt_text(args: argparse.Namespace) -> str | None:
    """Resolve optional prompt text from CLI arguments."""
    if args.prompt_file is None:
        return args.prompt
    return _read_text_file(args.prompt_file, label="prompt file")


def _estimate_tokens(text: str) -> int:
    """Approximate token count from word count for CLI analysis."""
    estimated_tokens = int(len(text.split()) * 1.3)
    if estimated_tokens == 0 and text:
        return 1
    return estimated_tokens


def _print_analysis_report(
    analysis: GenerationQualityAnalysis,
    *,
    word_count: int,
    estimated_tokens: int,
    prompt_tokens: int | None,
) -> None:
    """Print the full CLI analysis report."""
    print("\n" + "=" * 60)
    print(f"Analyzing text (approx {word_count} words, ~{estimated_tokens} tokens)")
    if prompt_tokens is not None:
        print(f"With prompt context (approx ~{prompt_tokens} tokens)")
    print("=" * 60 + "\n")

    print("Harness & Diagnostics Issues:")
    _print_field("Has Harness Issue", analysis.has_harness_issue)
    if analysis.has_harness_issue:
        _print_field("Harness Issue Type", analysis.harness_issue_type)
        _print_field("Harness Issue Details", analysis.harness_issue_details)

    print("\nQuality & Integrity Rule Violations:")
    _print_field("Is Repetitive", analysis.is_repetitive)
    if analysis.is_repetitive:
        _print_field("Repeated Token", analysis.repeated_token)
    _print_field("Is Verbose", analysis.is_verbose)
    _print_field("Has Formatting Issues", bool(analysis.formatting_issues))
    if analysis.formatting_issues:
        _print_field("Formatting Details", analysis.formatting_issues)
    _print_field("Has Hallucination Issues", bool(analysis.hallucination_issues))
    if analysis.hallucination_issues:
        _print_field("Hallucination Details", analysis.hallucination_issues)
    _print_field("Has Degeneration", analysis.has_degeneration)
    if analysis.has_degeneration:
        _print_field("Degeneration Type", analysis.degeneration_type)
    _print_field("Has Fabrication", analysis.has_fabrication)
    if analysis.has_fabrication:
        _print_field("Fabrication Issues", analysis.fabrication_issues)
    _print_field("Has Language Mixing", analysis.has_language_mixing)
    if analysis.has_language_mixing:
        _print_field("Mix Issues", analysis.language_mixing_issues)
    _print_field("Has Excessive Bullets", analysis.has_excessive_bullets)
    if analysis.has_excessive_bullets:
        _print_field("Bullet Count", analysis.bullet_count)

    print("\nPrompt Context Awareness:")
    _print_field("Prompt Checks Ran", analysis.prompt_checks_ran)
    _print_field("Ignored Context", analysis.is_context_ignored)
    if analysis.is_context_ignored:
        _print_field("Missing Terms", analysis.missing_context_terms)
    _print_field("Missing Required Sections", analysis.missing_sections)
    if analysis.missing_sections:
        _print_field("Title Word Count", analysis.title_word_count)
        _print_field("Keyword Count", analysis.keyword_count)
    _print_field("Reasoning Leak Confirmed", analysis.has_reasoning_leak)
    if analysis.has_reasoning_leak:
        _print_field("Leak Markers", analysis.reasoning_leak_markers)
    _print_field("Has Context Echo", analysis.has_context_echo)
    if analysis.has_context_echo:
        _print_field("Echo Ratio", f"{analysis.context_echo_ratio:.2%}")

    print("\n" + "-" * 60)
    issue_string = _build_quality_issues_string(analysis)
    print("Final Tag Output string:")
    print(f"  {issue_string or '🟢 CLEAN (No issues detected)'}")
    print("-" * 60 + "\n")


def main() -> int:
    """Run CLI output-quality analysis."""
    parser = _build_parser()
    args = parser.parse_args()

    load_quality_config()

    output_text = _resolve_output_text(args)
    if output_text is None:
        return 1

    prompt_text = _resolve_prompt_text(args)
    if args.prompt_file is not None and prompt_text is None:
        return 1

    word_count = len(output_text.split())
    estimated_tokens = _estimate_tokens(output_text)
    prompt_tokens = _estimate_tokens(prompt_text) if prompt_text else None

    analysis = analyze_generation_text(
        text=output_text,
        generated_tokens=estimated_tokens,
        prompt=prompt_text,
        prompt_tokens=prompt_tokens,
        context_marker=args.context_marker,
    )
    _print_analysis_report(
        analysis,
        word_count=word_count,
        estimated_tokens=estimated_tokens,
        prompt_tokens=prompt_tokens,
    )
    return 1 if analysis.has_any_issues() else 0


if __name__ == "__main__":
    sys.exit(main())
