"""Tests for environment validation helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from check_models_data import dependency_policy
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


def test_version_matches_specifier_accepts_dev_build_above_floor() -> None:
    """PEP 440 version handling should accept dev builds above the declared floor."""
    assert validate_env._version_matches_specifier(
        package_name="mlx",
        installed_version="0.31.3.dev20260410+a33b7916",
        version_spec=f">={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['mlx']}",
    )


def test_load_pyproject_deps_tracks_shared_runtime_policy() -> None:
    """Environment validation should inherit the shared MLX dependency policy from pyproject."""
    core_deps, extras_deps, _dev_deps = validate_env.load_pyproject_deps()

    assert core_deps["mlx"] == f">={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['mlx']}"
    assert (
        core_deps["mlx-vlm"] == f">={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['mlx-vlm']}"
    )
    assert (
        core_deps["transformers"]
        == f">={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['transformers']}"
    )
    assert core_deps["mlx-lm"] == f">={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['mlx-lm']}"
    assert core_deps["huggingface-hub"] == (
        f">={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['huggingface-hub']}"
    )
    assert core_deps["packaging"] == ">=26.0"
    assert "mlx-lm" not in extras_deps
