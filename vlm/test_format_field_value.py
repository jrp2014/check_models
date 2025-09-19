"""Unit tests for memory formatting heuristics in format_field_value.

These tests intentionally avoid asserting on presence of MLX; if MLX is not
installed the import of `vlm.check_models` will log an error but still expose
the formatting utilities.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type annotations only
    import types

import pytest


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:  # type: ignore[override]
    """Import and return the target module once for all tests."""
    return importlib.import_module("vlm.check_models")


def test_memory_bytes_mid_range(mod: types.ModuleType) -> None:
    """5 GB bytes -> one decimal place."""
    five_gb_bytes = 5 * mod.DECIMAL_GB
    out = mod.format_field_value("peak_memory", five_gb_bytes)
    assert out == "5.0"  # noqa: S101


def test_memory_bytes_large_integer(mod: types.ModuleType) -> None:
    """15 GB bytes -> integer formatting (>=10)."""
    fifteen_gb_bytes = 15 * mod.DECIMAL_GB
    out = mod.format_field_value("peak_memory", fifteen_gb_bytes)
    assert out == "15"  # noqa: S101


def test_memory_fractional_bytes(mod: types.ModuleType) -> None:
    """~0.512 GB bytes -> two decimals (<1)."""
    half_gb_bytes = 0.512 * mod.DECIMAL_GB  # 512 MB
    out = mod.format_field_value("peak_memory", half_gb_bytes)
    assert out == "0.51"  # noqa: S101


def test_memory_already_decimal_gb_small(mod: types.ModuleType) -> None:
    """Value already in decimal GB (<1)."""
    out = mod.format_field_value("peak_memory", 0.75)
    assert out == "0.75"  # noqa: S101


def test_memory_zero_and_negative(mod: types.ModuleType) -> None:
    """Zero and negative values clamp to '0'."""
    assert mod.format_field_value("peak_memory", 0) == "0"  # noqa: S101
    assert mod.format_field_value("peak_memory", -12345) == "0"  # noqa: S101

