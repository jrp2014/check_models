"""Contract tests for retained mechanical generation observations."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import check_models


@dataclass
class _Generation:
    text: str
    generation_tokens: int | None = 24
    prompt_tokens: int | None = None


def _result(
    text: str,
    *,
    generated_tokens: int = 24,
    prompt: str | None = None,
    requested_max_tokens: int | None = None,
    model_name: str = "example/model",
    known_special_tokens: tuple[str, ...] = (),
) -> check_models.PerformanceResult:
    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=generated_tokens,
        prompt=prompt,
        requested_max_tokens=requested_max_tokens,
        model_name=model_name,
        known_special_tokens=known_special_tokens,
    )
    return check_models.PerformanceResult(
        model_name=model_name,
        success=True,
        generation=_Generation(text, generated_tokens),
        requested_max_tokens=requested_max_tokens,
        quality_analysis=analysis,
    )


CATALOG_PROMPT = (
    "Return exactly these three sections, and nothing else:\n"
    "Title: 5-10 words.\nDescription: 1-2 factual sentences.\n"
    "Keywords: 10-18 terms."
)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        pytest.param(
            _result(""),
            check_models.ResultAssessment(
                "completed", "unusable", "observation_needs_reproduction", ("empty_output",)
            ),
            id="empty-output",
        ),
        pytest.param(
            _result("Brief reply", generated_tokens=2),
            check_models.ResultAssessment(
                "completed",
                "usable_with_caveats",
                "observation_needs_reproduction",
                ("minimal_output",),
            ),
            id="minimal-output",
        ),
        pytest.param(
            _result("word " * 100, generated_tokens=100),
            check_models.ResultAssessment(
                "completed", "unusable", "observation_needs_reproduction", ("repeated_output",)
            ),
            id="contiguous-repetition",
        ),
        pytest.param(
            _result("A misty lakeshore with trees and power lines.", prompt=CATALOG_PROMPT),
            check_models.ResultAssessment(
                "completed",
                "unusable",
                "observation_needs_reproduction",
                ("missing_requested_sections",),
            ),
            id="missing-requested-sections",
        ),
        pytest.param(
            _result("word " * 100, generated_tokens=100, requested_max_tokens=100),
            check_models.ResultAssessment(
                "completed",
                "unusable",
                "observation_needs_reproduction",
                ("repeated_output", "token_cap_truncation"),
            ),
            id="degraded-token-cap",
        ),
        pytest.param(
            _result(
                "Return exactly these three sections, and nothing else: "
                "Title, Description, Keywords."
            ),
            check_models.ResultAssessment(
                "completed",
                "usable_with_caveats",
                "observation_needs_reproduction",
                ("prompt_instruction_echo",),
            ),
            id="instruction-echo",
        ),
        pytest.param(
            _result("A caption.<|end|>"),
            check_models.ResultAssessment(
                "completed",
                "usable_with_caveats",
                "observation_needs_reproduction",
                ("unexpected_special_token",),
            ),
            id="unexpected-special-token",
        ),
        pytest.param(
            _result(
                "<think>Inspect the scene.</think> A blue boat rests on calm water.",
                model_name="example/thinking-model",
            ),
            check_models.ResultAssessment(
                "completed",
                "usable_with_caveats",
                "observation_needs_reproduction",
                ("thinking_trace_present",),
            ),
            id="thinking-trace",
        ),
        pytest.param(
            _result(
                "<think>Inspect the scene carefully",
                model_name="example/thinking-model",
            ),
            check_models.ResultAssessment(
                "completed",
                "usable_with_caveats",
                "observation_needs_reproduction",
                ("thinking_trace_present", "thinking_trace_incomplete"),
            ),
            id="incomplete-thinking-trace",
        ),
        pytest.param(
            _result(
                "Title: A blue boat\nDescription: A blue boat rests on calm water.\n"
                "Keywords: boat, water, blue, calm, lake, reflection, sky, shore, travel, vessel",
                prompt="Context: Existing metadata hints:\n- Keyword hints: mountain, forest, snow\n",
            ),
            check_models.ResultAssessment(
                "completed",
                "usable_with_caveats",
                "observation_needs_reproduction",
                ("no_keyword_overlap",),
            ),
            id="no-keyword-overlap",
        ),
    ],
)
def test_result_assessment_projects_only_ordered_mechanical_observations(
    result: check_models.PerformanceResult,
    expected: check_models.ResultAssessment,
) -> None:
    assert check_models._assess_result(result) == expected


def test_token_cap_alone_is_neutral() -> None:
    result = _result(
        "A complete response with a finished sentence.",
        generated_tokens=80,
        requested_max_tokens=80,
    )

    assert check_models._assess_result(result) == check_models.ResultAssessment(
        "completed", "usable", "none", ()
    )


def test_missing_generation_count_does_not_make_complete_output_minimal() -> None:
    result = check_models.PerformanceResult(
        model_name="example/model",
        success=True,
        generation=_Generation(
            "A complete description of a quiet lake beneath a clear evening sky.",
            generation_tokens=None,
            prompt_tokens=900,
        ),
    )

    assert "minimal_output" not in check_models._assess_result(result).observations


def test_recorded_low_output_to_prompt_ratio_is_minimal() -> None:
    result = check_models.PerformanceResult(
        model_name="example/model",
        success=True,
        generation=_Generation(
            "A concise but complete description of the visible landscape.",
            generation_tokens=10,
            prompt_tokens=1_000,
        ),
    )

    assert "minimal_output" in check_models._assess_result(result).observations


def test_missing_token_counts_do_not_enable_ratio_inference() -> None:
    result = check_models.PerformanceResult(
        model_name="example/model",
        success=True,
        generation=_Generation(
            "A concise but complete description of the visible landscape.",
            generation_tokens=None,
            prompt_tokens=None,
        ),
    )

    assert "minimal_output" not in check_models._assess_result(result).observations


def test_empty_thinking_wrapper_is_neutral() -> None:
    result = _result("<think></think> A complete response.", model_name="example/thinking-model")

    assert check_models._assess_result(result) == check_models.ResultAssessment(
        "completed", "usable", "none", ()
    )


def test_configured_special_token_is_not_unexpected() -> None:
    result = _result(
        "A complete response.<|custom_end|>",
        known_special_tokens=("<|custom_end|>",),
    )

    assert "unexpected_special_token" not in check_models._assess_result(result).observations


def test_partial_keyword_overlap_is_neutral() -> None:
    result = _result(
        "Title: A blue boat at dawn\n"
        "Description: A blue boat rests on calm water at dawn.\n"
        "Keywords: boat, water, blue, calm, dawn, lake, reflection, sky, shore, vessel",
        prompt="Context: Existing metadata hints:\n- Keyword hints: boat, mountain, forest\n",
    )

    assert "no_keyword_overlap" not in check_models._assess_result(result).observations


def test_draft_metadata_keywords_do_not_become_output_requirements() -> None:
    result = _result(
        "Title: Boats at dusk\n"
        "Description: Two boats rest on reflective water at dusk.\n"
        "Keywords: boats, water, dusk, reflection, sky, shore, calm, travel, vessel, evening",
        prompt=(
            "Context: Draft descriptive metadata:\n"
            "- Existing keywords: Example Harbour, Sample Village\n"
        ),
    )

    assert "no_keyword_overlap" not in check_models._assess_result(result).observations


def test_requested_section_parser_is_prompt_gated() -> None:
    plain = check_models.analyze_generation_text("A plain caption.", 12)
    requested = check_models.analyze_generation_text(
        "A plain caption.",
        12,
        prompt=CATALOG_PROMPT,
    )

    assert plain.missing_sections == []
    assert requested.missing_sections == ["title", "description", "keywords"]


def test_contiguous_repetition_detector_ignores_distributed_reuse() -> None:
    repeated, token = check_models._detect_repetitive_output("blue boat " * 60)
    distributed, _ = check_models._detect_repetitive_output(
        "A blue boat crosses the lake while another boat rests near a blue pier."
    )

    assert repeated is True
    assert token is not None
    assert distributed is False


@pytest.mark.parametrize(
    ("success", "error_message", "expected"),
    [
        (True, None, "completed"),
        (False, "boom", "crashed"),
        (False, "server disconnected without sending a response", "indeterminate"),
    ],
)
def test_result_assessment_execution_statuses(
    success: bool,
    error_message: str | None,
    expected: check_models.ExecutionStatus,
) -> None:
    result = check_models.PerformanceResult(
        model_name="example/model",
        success=success,
        generation=_Generation("Complete response.") if success else None,
        error_message=error_message,
    )

    assert check_models._assess_result(result).execution == expected
