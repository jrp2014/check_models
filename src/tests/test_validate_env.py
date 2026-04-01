"""Tests for environment validation helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tools import validate_env

if TYPE_CHECKING:
    import pytest


def test_resolve_expected_conda_env_prefers_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI override should win over environment override."""
    monkeypatch.setenv(validate_env.EXPECTED_CONDA_ENV_ENVVAR, "env-name")

    resolved = validate_env._resolve_expected_conda_env("cli-name")

    assert resolved == "cli-name"


def test_check_conda_env_accepts_any_active_env_without_override(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Any active env should pass when no strict expected name is configured."""
    monkeypatch.delenv(validate_env.EXPECTED_CONDA_ENV_ENVVAR, raising=False)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "custom-env")
    monkeypatch.setattr(validate_env.shutil, "which", lambda _name: "/usr/bin/conda")

    with caplog.at_level(logging.INFO, logger=validate_env.logger.name):
        validate_env.check_conda_env_with_expected(None)

    assert "✓ Conda environment 'custom-env' active" in caplog.text
    assert "!= expected" not in caplog.text


def test_check_conda_env_warns_when_override_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Strict expected env should still warn on mismatch."""
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "custom-env")
    monkeypatch.setattr(validate_env.shutil, "which", lambda _name: "/usr/bin/conda")

    with caplog.at_level(logging.INFO, logger=validate_env.logger.name):
        validate_env.check_conda_env_with_expected("mlx-vlm")

    assert "⚠ Active conda env 'custom-env' != expected 'mlx-vlm'" in caplog.text
    assert "Switch with: conda activate mlx-vlm" in caplog.text


def test_check_conda_env_missing_active_env_uses_expected_name(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing-env guidance should use the configured expected env name."""
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.setattr(validate_env.shutil, "which", lambda _name: "/usr/bin/conda")

    with caplog.at_level(logging.INFO, logger=validate_env.logger.name):
        validate_env.check_conda_env_with_expected("my-vlm-env")

    assert "Activate with: conda activate my-vlm-env" in caplog.text
