"""Tests for model discovery and filtering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

# HF cache environment is configured by conftest.py (early env setup + autouse fixture).
import pytest
from huggingface_hub.errors import CacheNotFound

import check_models

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class _FakeCacheFile:
    file_path: str


@dataclass(frozen=True)
class _FakeCacheRef:
    files: tuple[_FakeCacheFile, ...]


@dataclass(frozen=True)
class _FakeCacheRepo:
    repo_id: str
    repo_type: str
    refs: dict[str, _FakeCacheRef]


@dataclass(frozen=True)
class _FakeCacheInfo:
    repos: tuple[_FakeCacheRepo, ...]


@dataclass(frozen=True)
class _FakeIntegrityRepo:
    repo_id: str
    size_on_disk: int = 2_000_000
    nb_files: int = 3


@dataclass(frozen=True)
class _FakeIntegrityCacheInfo:
    repos: tuple[_FakeIntegrityRepo, ...]
    warnings: tuple[Exception, ...] = ()


def _fake_cache_repo(
    repo_id: str,
    files: tuple[str, ...],
    *,
    repo_type: str = "model",
    include_main: bool = True,
) -> _FakeCacheRepo:
    refs = {"main": _FakeCacheRef(tuple(_FakeCacheFile(path) for path in files))}
    if not include_main:
        refs = {}
    return _FakeCacheRepo(repo_id=repo_id, repo_type=repo_type, refs=refs)


def test_get_cached_model_ids_returns_list() -> None:
    """Should return a list of model IDs from cache."""
    try:
        model_ids = check_models.get_cached_model_ids()
        assert isinstance(model_ids, list)
        # May be empty if no models cached
        for model_id in model_ids:
            assert isinstance(model_id, str)
    except CacheNotFound:
        pytest.skip("HuggingFace cache directory not found (expected in CI)")


def test_get_cached_model_ids_matches_mlx_vlm_server_cache_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic cache discovery should match mlx-vlm's supported-model filter."""
    cache_info = _FakeCacheInfo(
        repos=(
            _fake_cache_repo(
                "org/supported-model",
                ("config.json", "tokenizer_config.json", "model.safetensors"),
            ),
            _fake_cache_repo(
                "org/supported-sharded-model",
                ("config.json", "tokenizer_config.json", "model.safetensors.index.json"),
            ),
            _fake_cache_repo("org/no-tokenizer", ("config.json", "model.safetensors")),
            _fake_cache_repo(
                "org/no-weights",
                ("config.json", "tokenizer_config.json", "pytorch_model.bin"),
            ),
            _fake_cache_repo(
                "org/no-main-ref",
                ("config.json", "tokenizer_config.json", "model.safetensors"),
                include_main=False,
            ),
            _fake_cache_repo(
                "org/dataset-cache",
                ("config.json", "tokenizer_config.json", "model.safetensors"),
                repo_type="dataset",
            ),
        )
    )
    monkeypatch.setattr(
        check_models,
        "_get_hf_cache_info_cached",
        lambda **_: cache_info,
    )

    assert check_models.get_cached_model_ids() == [
        "org/supported-model",
        "org/supported-sharded-model",
    ]


def test_cached_model_eligibility_reports_skip_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported cached repos should carry maintainer-readable skip reasons."""
    cache_info = _FakeCacheInfo(
        repos=(
            _fake_cache_repo("org/no-tokenizer", ("config.json", "model.safetensors")),
            _fake_cache_repo(
                "org/no-weights",
                ("config.json", "tokenizer_config.json", "pytorch_model.bin"),
            ),
            _fake_cache_repo(
                "org/no-main-ref",
                ("config.json", "tokenizer_config.json", "model.safetensors"),
                include_main=False,
            ),
        )
    )
    monkeypatch.setattr(
        check_models,
        "_get_hf_cache_info_cached",
        lambda **_: cache_info,
    )

    entries = {
        entry.repo_id: entry
        for entry in check_models.get_cached_model_eligibility()
        if not entry.supported
    }

    assert entries["org/no-tokenizer"].reasons == ("missing tokenizer_config.json",)
    assert entries["org/no-weights"].reasons == ("missing safetensors weights",)
    assert entries["org/no-main-ref"].reasons == ("missing main revision in cache",)


def test_auto_cache_discovery_logs_skipped_models(caplog: pytest.LogCaptureFixture) -> None:
    """Unspecified model runs should highlight cached models skipped by discovery."""
    eligibility = (
        check_models.CachedModelEligibility(
            repo_id="org/supported-model",
            supported=True,
            reasons=(),
        ),
        check_models.CachedModelEligibility(
            repo_id="org/no-tokenizer",
            supported=False,
            reasons=("missing tokenizer_config.json",),
        ),
    )

    with caplog.at_level(logging.INFO, logger=check_models.logger.name):
        selected = check_models._supported_cached_model_ids_with_skipped_logging(eligibility)

    assert selected == ["org/supported-model"]
    assert (
        "Skipped 1 cached repo(s) that mlx-vlm server-style discovery would not run" in caplog.text
    )
    assert "org/no-tokenizer: missing tokenizer_config.json" in caplog.text


def test_cache_integrity_uses_exact_repo_id_matching(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A similarly named cache entry must not be treated as the requested model."""
    cache_info = _FakeIntegrityCacheInfo(
        repos=(_FakeIntegrityRepo("org/model-extra"),),
    )
    monkeypatch.setattr(check_models, "_get_hf_cache_info_cached", lambda **_: cache_info)

    with caplog.at_level(logging.DEBUG, logger=check_models.logger.name):
        check_models._check_hf_cache_integrity("org/model")

    assert "Model org/model not found in HF cache" in caplog.text
    assert "HF Cache Info for org/model-extra" not in caplog.text


def test_cache_integrity_reports_matching_scan_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A corrupt repo omitted from repos should still produce an actionable warning."""
    cache_info = _FakeIntegrityCacheInfo(
        repos=(),
        warnings=(
            RuntimeError("Snapshots dir doesn't exist in cached repo: /cache/models--org--model"),
        ),
    )
    monkeypatch.setattr(check_models, "_get_hf_cache_info_cached", lambda **_: cache_info)

    with caplog.at_level(logging.WARNING, logger=check_models.logger.name):
        check_models._check_hf_cache_integrity("org/model")

    assert "Cache Warning: Hugging Face reported corruption for org/model" in caplog.text
    assert "Snapshots dir doesn't exist" in caplog.text


def test_validate_model_identifier_accepts_valid_huggingface_format() -> None:
    """Should accept standard HuggingFace model identifiers."""
    # Should not raise
    check_models.validate_model_identifier("mlx-community/Qwen2-VL-2B-Instruct-4bit")
    check_models.validate_model_identifier("microsoft/Phi-3-vision-128k-instruct")
    check_models.validate_model_identifier("apple/OpenELM-270M")


def test_validate_model_identifier_accepts_local_paths(tmp_path: Path) -> None:
    """Should accept valid local paths."""
    # Create a dummy model directory
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()
    check_models.validate_model_identifier(str(model_dir))


def test_validate_model_identifier_rejects_empty_string() -> None:
    """Should reject empty model identifier."""
    with pytest.raises(ValueError, match="Model identifier cannot be empty"):
        check_models.validate_model_identifier("")


def test_validate_model_identifier_rejects_whitespace_only() -> None:
    """Should reject whitespace-only identifiers."""
    with pytest.raises(ValueError, match="Model identifier cannot be empty"):
        check_models.validate_model_identifier("   ")
    with pytest.raises(ValueError, match="Model identifier cannot be empty"):
        check_models.validate_model_identifier("\t\n")


def test_validate_kv_params_valid_combinations() -> None:
    """Should accept valid KV cache parameter combinations."""
    # Should not raise
    check_models.validate_kv_params(kv_bits=None, max_kv_size=None)
    check_models.validate_kv_params(kv_bits=4, max_kv_size=1024)
    check_models.validate_kv_params(kv_bits=8, max_kv_size=2048)
    check_models.validate_kv_params(kv_bits=3.5, max_kv_size=2048)


def test_validate_kv_params_rejects_invalid_bits() -> None:
    """Should reject invalid kv_bits values."""
    with pytest.raises(ValueError, match="kv_bits must be"):
        check_models.validate_kv_params(kv_bits=16, max_kv_size=1024)


def test_validate_kv_params_rejects_negative_size() -> None:
    """Should reject negative max_kv_size."""
    with pytest.raises(ValueError, match="max_kv_size must be > 0"):
        check_models.validate_kv_params(kv_bits=4, max_kv_size=-100)


def test_validate_kv_params_rejects_zero_size() -> None:
    """Should reject zero max_kv_size."""
    with pytest.raises(ValueError, match="max_kv_size must be > 0"):
        check_models.validate_kv_params(kv_bits=4, max_kv_size=0)


def test_is_numeric_field_identifies_numeric_fields() -> None:
    """Should correctly identify numeric field names."""
    assert check_models.is_numeric_field("prompt_tps")
    assert check_models.is_numeric_field("generation_tps")
    assert check_models.is_numeric_field("total_time")
    assert check_models.is_numeric_field("peak_memory_gb")


def test_is_numeric_field_rejects_text_fields() -> None:
    """Should correctly identify non-numeric fields."""
    assert not check_models.is_numeric_field("model_identifier")
    assert not check_models.is_numeric_field("response")
    assert not check_models.is_numeric_field("error_message")


def test_is_numeric_value_identifies_numbers() -> None:
    """Should correctly identify numeric values."""
    assert check_models.is_numeric_value(42)
    assert check_models.is_numeric_value(3.14)
    assert check_models.is_numeric_value(0)
    assert check_models.is_numeric_value(-1.5)


def test_is_numeric_value_rejects_non_numbers() -> None:
    """Should reject non-numeric values."""
    assert not check_models.is_numeric_value("text")
    # Note: "42" is numeric (can be parsed as number)
    assert not check_models.is_numeric_value(None)
    assert not check_models.is_numeric_value([1, 2, 3])
