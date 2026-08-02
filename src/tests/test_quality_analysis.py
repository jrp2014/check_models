"""Contract tests for retained mechanical generation observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

import check_models

type ExpectedExecutionStatus = Literal["completed", "crashed", "indeterminate"]


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
                "unusable",
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


@pytest.mark.parametrize(
    ("title", "keywords", "expected_title_words", "expected_keyword_count", "duplicates"),
    [
        (
            "Four Word Title Here",
            "one, two, three, four, five, six, seven, eight, nine, ten",
            4,
            10,
            [],
        ),
        (
            "Five Word Catalogue Title Here",
            (
                "one, two, three, four, five, six, seven, eight, nine, ten, eleven, "
                "twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen"
            ),
            5,
            19,
            [],
        ),
        (
            "Five Word Catalogue Title Here",
            "Halesworth, sky, brick, windows, sign, gravel, clouds, arts, building, halesworth",
            5,
            10,
            ["halesworth"],
        ),
    ],
)
def test_catalog_constraint_violations_are_repairable_caveats(
    title: str,
    keywords: str,
    expected_title_words: int,
    expected_keyword_count: int,
    duplicates: list[str],
) -> None:
    """Counts and duplicates should qualify otherwise complete catalogue output."""
    result = _result(
        f"Title: {title}\n"
        "Description: A factual description of the visible building.\n"
        f"Keywords: {keywords}",
        prompt=CATALOG_PROMPT,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.title_word_count == expected_title_words
    assert result.quality_analysis.keyword_count == expected_keyword_count
    assert result.quality_analysis.duplicate_keywords == duplicates
    assert check_models._assess_result(result) == check_models.ResultAssessment(
        "completed",
        "usable_with_caveats",
        "observation_needs_reproduction",
        ("catalog_constraint_violation",),
    )


def test_compliant_catalog_constraints_remain_clean() -> None:
    result = _result(
        "Title: Five Word Catalogue Title Here\n"
        "Description: A factual description of the visible building.\n"
        "Keywords: one, two, three, four, five, six, seven, eight, nine, ten",
        prompt=CATALOG_PROMPT,
    )

    assert check_models._assess_result(result) == check_models.ResultAssessment(
        "completed", "usable", "none", ()
    )


def test_catalog_constraints_are_not_inferred_for_an_unrelated_prompt() -> None:
    result = _result(
        "Title: Brief title\nDescription: A factual description.\nKeywords: repeated, repeated",
        prompt="Describe the response format used in this example.",
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.title_word_count is None
    assert result.quality_analysis.keyword_count is None
    assert result.quality_analysis.duplicate_keywords == []
    assert "catalog_constraint_violation" not in check_models._assess_result(result).observations


def test_configured_utterance_boundary_is_reported_when_visible() -> None:
    text = (
        "Title: Five Word Catalogue Title Here\n"
        "Description: A factual description of the visible building.\n"
        "Keywords: one, two, three, four, five, six, seven, eight, nine, ten"
        "<end_of_utterance>"
    )
    result = _result(
        text,
        prompt=CATALOG_PROMPT,
        known_special_tokens=("<end_of_utterance>",),
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.role_boundary_tokens == ["<end_of_utterance>"]
    assert check_models._assess_result(result).observations == ("role_boundary_token_present",)


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


def test_concise_complete_output_is_not_minimal_relative_to_prompt_length() -> None:
    result = check_models.PerformanceResult(
        model_name="example/model",
        success=True,
        generation=_Generation(
            "A concise but complete description of the visible landscape.",
            generation_tokens=10,
            prompt_tokens=1_000,
        ),
    )

    assert "minimal_output" not in check_models._assess_result(result).observations


def test_complete_image_phrase_is_not_semantic_minimal_output() -> None:
    result = check_models.PerformanceResult(
        model_name="example/model",
        success=True,
        generation=_Generation(
            "The image is a photograph of two cats sleeping on a couch.",
            generation_tokens=4,
        ),
    )

    assert "minimal_output" not in check_models._assess_result(result).observations


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


def test_thinking_trace_observation_is_model_name_invariant() -> None:
    text = "<think>Inspect the scene.</think> A blue boat rests on calm water."
    plain = _result(text, model_name="example/plain-model")
    named = _result(text, model_name="example/thinking-model")

    assert check_models._assess_result(plain) == check_models._assess_result(named)
    assert check_models._assess_result(plain).observations == ("thinking_trace_present",)


def test_configured_thinking_delimiters_are_observed_without_model_name_policy() -> None:
    result = check_models.PerformanceResult(
        model_name="example/plain-model",
        success=True,
        generation=_Generation(
            "<reason>Inspect the scene.</done> A blue boat rests on calm water.",
            generation_tokens=18,
        ),
        prompt_diagnostics=check_models.PromptDiagnostics(
            generate_kwargs={
                "enable_thinking": True,
                "thinking_start_token": "<reason>",
                "thinking_end_token": "</done>",
            }
        ),
    )

    context = check_models._build_report_render_context(
        results=[result],
        prompt="Describe the image.",
        system_info={},
    )

    assessment = dict(context.assessments)[result.model_name]
    assert assessment.observations == (
        "configured_wrapper_present",
        "thinking_trace_present",
    )


def test_configured_special_token_is_not_unexpected() -> None:
    result = _result(
        "A complete response.<|custom_end|>",
        known_special_tokens=("<|custom_end|>",),
    )

    assert "unexpected_special_token" not in check_models._assess_result(result).observations


@pytest.mark.parametrize(
    "wrapper",
    ["<|im_start|>", "<|begin_of_box|>", "<|channel>", "<channel|>"],
)
def test_undeclared_control_wrapper_is_observed_generically(wrapper: str) -> None:
    result = _result(f"{wrapper} A complete response.")

    assert result.quality_analysis is not None
    assert wrapper in result.quality_analysis.unexpected_special_tokens
    assert "unexpected_special_token" in check_models._assess_result(result).observations


def test_declared_generation_wrappers_are_neutral_without_model_name_policy() -> None:
    wrappers = ("<|custom_eos|>", "<|custom_stop|>")
    result = check_models.PerformanceResult(
        model_name="example/plain-model",
        success=True,
        generation=_Generation(
            f"{wrappers[0]}{wrappers[1]}<reason>Inspect the scene.</done> A complete response.",
            generation_tokens=18,
        ),
        prompt_diagnostics=check_models.PromptDiagnostics(
            eos_token=wrappers[0],
            generate_kwargs={
                "eos_tokens": [wrappers[1]],
                "enable_thinking": True,
                "thinking_start_token": "<reason>",
                "thinking_end_token": "</done>",
            },
        ),
    )

    context = check_models._build_report_render_context(
        results=[result],
        prompt="Describe the image.",
        system_info={},
    )

    enriched = context.result_set.results[0]
    assert enriched.quality_analysis is not None
    assert enriched.quality_analysis.unexpected_special_tokens == []
    assert enriched.quality_analysis.configured_generation_wrappers == [
        "<|custom_eos|>",
        "<|custom_stop|>",
        "<reason>",
        "</done>",
    ]
    assert dict(context.assessments)[result.model_name].observations == (
        "configured_wrapper_present",
        "thinking_trace_present",
    )


def test_assisted_output_returning_every_supplied_draft_field_is_observed_exactly() -> None:
    """An unchanged descriptive draft is chooser evidence, not a semantic score."""
    prompt = check_models._build_cataloguing_prompt(
        {
            "title": "Harbour boats at dusk",
            "description": "Two boats rest on calm water at dusk.",
            "keywords": "boats, harbour, water, dusk, reflection, sky, shore, calm, travel, vessel",
        }
    )
    result = _result(
        "Title: Harbour boats at dusk\n"
        "Description: Two boats rest on calm water at dusk.\n"
        "Keywords: boats, harbour, water, dusk, reflection, sky, shore, calm, travel, vessel",
        prompt=prompt,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.unchanged_draft_fields == [
        "title",
        "description",
        "keywords",
    ]
    assert check_models._assess_result(result) == check_models.ResultAssessment(
        "completed",
        "usable_with_caveats",
        "observation_needs_reproduction",
        ("catalog_constraint_violation", "draft_returned_unchanged"),
    )


def test_assisted_output_that_changes_one_draft_field_is_not_called_unchanged() -> None:
    """The exact draft observation must not infer whether a rewrite is better or worse."""
    prompt = check_models._build_cataloguing_prompt(
        {
            "title": "Harbour boats at dusk",
            "description": "Two boats rest on calm water at dusk.",
            "keywords": "boats, harbour, water, dusk, reflection, sky, shore, calm, travel, vessel",
        }
    )
    result = _result(
        "Title: Harbour boats beneath a violet sky\n"
        "Description: Two boats rest on calm water at dusk.\n"
        "Keywords: boats, harbour, water, dusk, reflection, sky, shore, calm, travel, vessel",
        prompt=prompt,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.unchanged_draft_fields == []
    assert "draft_returned_unchanged" not in check_models._assess_result(result).observations


def test_historical_existing_labels_still_detect_unchanged_draft_fields() -> None:
    """Retained prompts should remain analysable after generated labels change."""
    prompt = (
        f"{CATALOG_PROMPT}\n\n"
        "Context: Draft descriptive metadata:\n"
        "- Existing title: Harbour boats at dusk\n"
        "- Existing description: Two boats rest on calm water at dusk.\n"
        "- Existing keywords: boats, harbour, water"
    )
    result = _result(
        "Title: Harbour boats at dusk\n"
        "Description: Two boats rest on calm water at dusk.\n"
        "Keywords: boats, harbour, water",
        prompt=prompt,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.unchanged_draft_fields == [
        "title",
        "description",
        "keywords",
    ]


def test_authoritative_context_keeps_adjacent_keyword_hints_assessable() -> None:
    """The second assisted context block must reach weak keyword-overlap analysis."""
    prompt = check_models._build_cataloguing_prompt(
        {
            "date": "2026-07-31",
            "keywords": "boat, harbour, water",
        }
    )
    result = _result(
        "Title: Mountain path beneath cloud\n"
        "Description: A rocky path crosses a mountain slope beneath cloud.\n"
        "Keywords: mountain, path, rocks, cloud, slope, landscape, hiking, grey, outdoors, trail",
        prompt=prompt,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.keyword_overlap == "no_overlap"


def test_configured_user_role_token_mid_output_is_observed_as_a_boundary() -> None:
    result = _result(
        "Title: Two cats\nDescription: Two cats rest indoors.\n"
        "Keywords: cats, indoor, resting, sofa, pets, home, tabby, fur, furniture, calm"
        "<|im_user|>Solve an unrelated equation.",
        known_special_tokens=("<|im_user|>",),
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.role_boundary_tokens == ["<|im_user|>"]
    assert "role_boundary_token_present" in check_models._assess_result(result).observations


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


def test_short_catalog_response_still_has_to_satisfy_requested_sections() -> None:
    result = _result(
        "Do not output the prompt instructions.",
        generated_tokens=8,
        prompt=CATALOG_PROMPT,
    )

    assert check_models._assess_result(result).usability == "unusable"
    assert check_models._assess_result(result).observations == (
        "missing_requested_sections",
        "prompt_instruction_echo",
    )


def test_multiple_title_list_items_do_not_satisfy_catalog_contract() -> None:
    result = _result(
        "Title:\n- remote control\n- cat\n- sofa\n"
        "Description: A cat sits beside a remote control.\n"
        "Keywords: cat, sofa, remote, indoor, pet, furniture, resting, home, animal, room",
        prompt=CATALOG_PROMPT,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.missing_sections == ["title"]
    assert check_models._assess_result(result).usability == "unusable"


def test_instruction_echo_detects_normalized_prompt_span() -> None:
    prompt = (
        f"{CATALOG_PROMPT}\n"
        "Keywords: 10-18 unique comma-separated terms covering supplied authoritative "
        "context and clearly visible subjects, setting, colors, composition, and style."
    )
    result = _result(
        "Title: Two cats on a sofa\n"
        "Description: Two cats rest together indoors.\n"
        "Keywords: supplied authoritative context and clearly visible subjects, setting, "
        "colors, composition, and style",
        prompt=prompt,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.instruction_echo is True
    assert check_models._assess_result(result).usability == "unusable"


def test_instruction_echo_ignores_authoritative_context_values() -> None:
    prompt = (
        f"{CATALOG_PROMPT}\n\n"
        "Context: Authoritative context:\n"
        "- Capture date/time: 2026-07-25 18:33:16 UTC+01:00\n"
        "- GPS: 51.358240°N, 1.432820°E\n\n"
        "Draft descriptive metadata:\n"
        "- Existing title: Viking Bay, Broadstairs, England, UK\n"
        "- Existing description: A sunny beach in Broadstairs, Kent.\n"
        "- Existing keywords: beach, Broadstairs, Kent, coast"
    )
    result = _result(
        "Title: Viking Bay, 2026-07-25 18:33:16 UTC+01:00\n"
        "Description: A sunny beach in Broadstairs, Kent.\n"
        "Keywords: beach, Broadstairs, Kent, coast, sand, sea, people, sky, "
        "buildings, summer",
        prompt=prompt,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.instruction_echo is False
    assert check_models._assess_result(result).usability == "usable"


def test_markdown_bold_catalog_labels_satisfy_requested_sections() -> None:
    result = _result(
        "**Title:**\nViking Bay Beach, Broadstairs, Kent\n\n"
        "**Description:**\nA sunny beach scene with people and colourful huts.\n\n"
        "**Keywords:**\nbeach, Broadstairs, Kent, coast, sand, sea, people, sky, "
        "buildings, summer",
        prompt=CATALOG_PROMPT,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.missing_sections == []
    assert check_models._assess_result(result).usability == "usable"


@pytest.mark.parametrize("heading", ["#", "###", "######"])
def test_markdown_heading_catalog_labels_satisfy_requested_sections(heading: str) -> None:
    result = _result(
        f"{heading} Title:\nTwo Cats Lounging on Red Couch\n\n"
        f"{heading} Description:\nTwo cats relax together on a red couch.\n\n"
        f"{heading} Keywords:\n"
        "cats, lounging, red couch, remote controls, relaxed, indoor, comfort, "
        "feline, domestic, resting",
        prompt=CATALOG_PROMPT,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.missing_sections == []
    assert check_models._assess_result(result).usability == "usable"


def test_text_before_catalog_sections_is_reported_as_unusable() -> None:
    result = _result(
        "Remove non-visual information.\n\n"
        "Title: Viking Bay Beach, Broadstairs, Kent\n"
        "Description: A sunny beach scene with people and colourful huts.\n"
        "Keywords: beach, Broadstairs, Kent, coast, sand, sea, people, sky, "
        "buildings, summer",
        prompt=CATALOG_PROMPT,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.unexpected_catalog_preamble == ("Remove non-visual information.")
    assert check_models._assess_result(result).observations == ("unexpected_catalog_preamble",)
    assert check_models._assess_result(result).usability == "unusable"


def test_empty_thinking_wrapper_before_catalog_sections_is_neutral() -> None:
    result = _result(
        "<think></think>\n"
        "Title: Viking Bay Beach, Broadstairs, Kent\n"
        "Description: A sunny beach scene with people and colourful huts.\n"
        "Keywords: beach, Broadstairs, Kent, coast, sand, sea, people, sky, "
        "buildings, summer",
        prompt=CATALOG_PROMPT,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.unexpected_catalog_preamble is None
    assert check_models._assess_result(result).usability == "usable"


def test_repeated_keyword_cycle_is_repetitive_output() -> None:
    cycle = "cat, sofa, indoor, pet, resting, animal, home, furniture, whiskers, fur"
    result = _result(
        "Title: Two cats resting indoors\n"
        "Description: Two cats rest on a sofa.\n"
        f"Keywords: {cycle}, {cycle}, {cycle}",
        prompt=CATALOG_PROMPT,
        generated_tokens=80,
    )

    assert result.quality_analysis is not None
    assert result.quality_analysis.is_repetitive is True
    assert check_models._assess_result(result).usability == "unusable"


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
    expected: ExpectedExecutionStatus,
) -> None:
    result = check_models.PerformanceResult(
        model_name="example/model",
        success=success,
        generation=_Generation("Complete response.") if success else None,
        error_message=error_message,
    )

    assert check_models._assess_result(result).execution == expected
