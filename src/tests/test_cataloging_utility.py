"""Tests for the retained weak keyword-overlap signal."""

from __future__ import annotations

from typing import Literal

import pytest

from check_models import _keyword_overlap_state

type ExpectedKeywordOverlapState = Literal["not_assessable", "no_overlap", "some_overlap"]


@pytest.mark.parametrize(
    ("reference", "generated", "expected"),
    [
        ((), ("harbour",), "not_assessable"),
        (("red boats",), (), "not_assessable"),
        (("wooden benches",), ("city lights",), "no_overlap"),
        (("garden paths", "flowers"), ("flower", "trees"), "some_overlap"),
    ],
)
def test_keyword_overlap_state_is_an_elementary_weak_signal(
    reference: tuple[str, ...],
    generated: tuple[str, ...],
    expected: ExpectedKeywordOverlapState,
) -> None:
    """Keyword comparison exposes only assessability and whether any term overlaps."""
    assert _keyword_overlap_state(reference, generated) == expected
