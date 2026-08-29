"""Standalone script to analyze arbitrary output with project quality heuristics.

This lets developers inspect check_models quality and harness detection without
running a local MLX model. Verdicts come from the same canonical observation
projection and completed-result assessment the main harness uses, so this tool
cannot disagree with the retained reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_models import (
    GenerationQualityAnalysis,
    ResultAssessment,
    _completed_assessment,
    _human_observation_labels,
    _quality_observations,
    analyze_generation_text,
    load_quality_config,
)
from tools.safe_io import read_text_no_follow


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for output quality analysis."""
    parser = argparse.ArgumentParser(
        description=("Inspect the project's mechanical VLM output observations on arbitrary text."),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text string to evaluate directly.")
    group.add_argument("--file", type=Path, help="File containing text to evaluate.")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Optional prompt string used for context-echo and contract checks.",
    )
    prompt_group.add_argument(
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human report.",
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
        return read_text_no_follow(path)
    except OSError as exc:
        print(f"Error reading {label}: {exc}")
        return None


def _resolve_output_text(args: argparse.Namespace) -> str | None:
    """Resolve the output text from CLI arguments."""
    if args.file is None:
        return cast("str | None", args.text)
    return _read_text_file(args.file, label="output text file")


def _resolve_prompt_text(args: argparse.Namespace) -> str | None:
    """Resolve optional prompt text from CLI arguments."""
    if args.prompt_file is None:
        return cast("str | None", args.prompt)
    return _read_text_file(args.prompt_file, label="prompt file")


def _estimate_tokens(text: str) -> int:
    """Approximate token count from word count for CLI analysis."""
    estimated_tokens = int(len(text.split()) * 1.3)
    if estimated_tokens == 0 and text:
        return 1
    return estimated_tokens


def _assessment_status(assessment: ResultAssessment) -> str:
    """Return the canonical fault/observation/clean status token."""
    if assessment.usability == "unusable":
        return "unusable"
    return "observation" if assessment.observations else "clean"


def _print_analysis_report(
    analysis: GenerationQualityAnalysis,
    assessment: ResultAssessment,
    *,
    word_count: int,
    estimated_tokens: int,
    prompt_tokens: int | None,
) -> None:
    """Print the retained mechanical analysis fields."""
    print("\n" + "=" * 60)
    print(f"Analyzing text (approx {word_count} words, ~{estimated_tokens} tokens)")
    if prompt_tokens is not None:
        print(f"With prompt context (approx ~{prompt_tokens} tokens)")
    print("=" * 60 + "\n")

    print("Mechanical Observations:")
    if analysis.special_token_wrappers:
        _print_field("Special Token Wrappers", analysis.special_token_wrappers)
    _print_field("Is Repetitive", analysis.is_repetitive)
    if analysis.is_repetitive:
        _print_field("Repeated Token", analysis.repeated_token)
    _print_field("Prompt Checks Ran", analysis.prompt_checks_ran)
    _print_field("Missing Required Sections", analysis.missing_sections)
    _print_field("Instruction Echo", analysis.instruction_echo)
    _print_field("Thinking Trace", analysis.has_thinking_trace)
    _print_field("Thinking Incomplete", analysis.thinking_trace_incomplete)
    _print_field("Likely Token Cap", analysis.likely_capped)
    _print_field("Unexpected Special Tokens", analysis.unexpected_special_tokens)
    _print_field("Keyword Overlap", analysis.keyword_overlap)

    print("\n" + "-" * 60)
    status = _assessment_status(assessment)
    if status == "clean":
        print("  🟢 CLEAN (No issues detected)")
    else:
        prefix = "🔴 UNUSABLE" if status == "unusable" else "🟡 OBSERVATION"
        labels = _human_observation_labels(assessment.observations)
        print(f"  {prefix} ({labels})")
    print("-" * 60 + "\n")


def _build_json_payload(
    analysis: GenerationQualityAnalysis,
    assessment: ResultAssessment,
    *,
    word_count: int,
    estimated_tokens: int,
    prompt_tokens: int | None,
) -> dict[str, object]:
    """Build a stable machine-readable JSON payload for CLI consumers."""
    return {
        "status": _assessment_status(assessment),
        "exit_code": 1 if assessment.usability == "unusable" else 0,
        "summary": {
            "word_count": word_count,
            "estimated_tokens": estimated_tokens,
            "prompt_tokens": prompt_tokens,
        },
        "assessment": {
            "execution": assessment.execution,
            "usability": assessment.usability,
            "maintainer_status": assessment.maintainer_status,
            "observations": list(assessment.observations),
        },
        "analysis": asdict(analysis),
    }


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
    observations = _quality_observations(
        text=output_text,
        generated_tokens=estimated_tokens,
        analysis=analysis,
    )
    assessment = _completed_assessment(observations)
    exit_code = 1 if assessment.usability == "unusable" else 0
    if args.json:
        print(
            json.dumps(
                _build_json_payload(
                    analysis,
                    assessment,
                    word_count=word_count,
                    estimated_tokens=estimated_tokens,
                    prompt_tokens=prompt_tokens,
                ),
                indent=2,
                sort_keys=True,
            ),
        )
        return exit_code

    _print_analysis_report(
        analysis,
        assessment,
        word_count=word_count,
        estimated_tokens=estimated_tokens,
        prompt_tokens=prompt_tokens,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
