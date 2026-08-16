"""Tests for model discovery and filtering."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

# HF cache environment is configured by conftest.py (early env setup + autouse fixture).
import pytest
from huggingface_hub.errors import CacheNotFound

import check_models
from tools import safe_io

if TYPE_CHECKING:
    from collections.abc import Iterator
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
    assert "Skipped 1 cached repo(s) that default discovery will not run" in caplog.text
    assert "org/no-tokenizer: cache layout: missing tokenizer_config.json" in caplog.text


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


# ---------------------------------------------------------------------------
# Architecture pre-check (upstream --check-arch tier)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeSnapshotRef:
    snapshot_path: str


@dataclass(frozen=True)
class _FakeSnapshotRepo:
    repo_id: str
    repo_type: str
    refs: dict[str, _FakeSnapshotRef]


@pytest.fixture(name="_clear_arch_caches")
def _clear_arch_caches_fixture() -> Iterator[None]:
    """Isolate the memoized installed-package probes between tests."""
    check_models._installed_mlx_vlm_model_types.cache_clear()
    check_models._mlx_vlm_model_remapping.cache_clear()
    yield
    check_models._installed_mlx_vlm_model_types.cache_clear()
    check_models._mlx_vlm_model_remapping.cache_clear()


def _fake_mlx_vlm_package(tmp_path: Path, model_types: tuple[str, ...], remapping: str) -> Path:
    package_dir = tmp_path / "mlx_vlm"
    for model_type in model_types:
        (package_dir / "models" / model_type).mkdir(parents=True)
    (package_dir / "models" / "__pycache__").mkdir(exist_ok=True)
    safe_io.write_text_no_follow(package_dir / "utils.py", remapping)
    return package_dir


@pytest.mark.usefixtures("_clear_arch_caches")
def test_installed_mlx_vlm_model_types_scans_package_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model-type discovery must scan package folders without importing mlx."""
    package_dir = _fake_mlx_vlm_package(
        tmp_path,
        ("qwen2_vl", "fastvlm"),
        "MODEL_REMAPPING = {'llava_qwen2': 'fastvlm'}\n",
    )
    fake_spec = SimpleNamespace(submodule_search_locations=[str(package_dir)])
    monkeypatch.setattr(check_models, "find_spec", lambda _name: fake_spec)

    assert check_models._installed_mlx_vlm_model_types() == frozenset({"qwen2_vl", "fastvlm"})
    assert check_models._mlx_vlm_model_remapping() == {"llava_qwen2": "fastvlm"}


@pytest.mark.usefixtures("_clear_arch_caches")
def test_installed_mlx_vlm_model_types_handles_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing mlx_vlm installation must yield indeterminate, not a crash."""
    monkeypatch.setattr(check_models, "find_spec", lambda _name: None)

    assert check_models._installed_mlx_vlm_model_types() is None
    assert check_models._mlx_vlm_model_remapping() == {}


@pytest.mark.parametrize(
    ("model_type", "expected_resolved", "expected_supported"),
    [
        ("qwen2_vl", "qwen2_vl", True),
        ("llava_qwen2", "fastvlm", True),  # alias resolves via MODEL_REMAPPING
        ("totally_new_arch", "totally_new_arch", False),
    ],
)
def test_model_arch_precheck_resolves_aliases_against_installed_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    expected_resolved: str,
    expected_supported: bool,
) -> None:
    """The pre-check must mirror upstream --check-arch semantics."""
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    repo = _FakeSnapshotRepo(
        repo_id="org/model",
        repo_type="model",
        refs={"main": _FakeSnapshotRef(snapshot_path=str(snapshot))},
    )
    monkeypatch.setattr(
        check_models,
        "_installed_mlx_vlm_model_types",
        lambda: frozenset({"qwen2_vl", "fastvlm"}),
    )
    monkeypatch.setattr(
        check_models,
        "_mlx_vlm_model_remapping",
        lambda: {"llava_qwen2": "fastvlm"},
    )

    result = check_models._model_arch_precheck(repo)

    assert result == (model_type, expected_resolved, expected_supported)


def test_model_arch_precheck_is_indeterminate_without_config_or_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing config or missing installed mlx-vlm must never claim a verdict."""
    no_snapshot_repo = _FakeSnapshotRepo(repo_id="org/none", repo_type="model", refs={})
    assert check_models._model_arch_precheck(no_snapshot_repo) == (None, None, None)

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen2_vl"}), "utf-8")
    repo = _FakeSnapshotRepo(
        repo_id="org/model",
        repo_type="model",
        refs={"main": _FakeSnapshotRef(snapshot_path=str(snapshot))},
    )
    monkeypatch.setattr(check_models, "_installed_mlx_vlm_model_types", lambda: None)
    monkeypatch.setattr(check_models, "_mlx_vlm_model_remapping", dict)

    assert check_models._model_arch_precheck(repo) == ("qwen2_vl", "qwen2_vl", None)


def test_arch_precheck_summary_renders_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-model fact renders yes/no with alias resolution, or omits itself."""
    monkeypatch.setattr(
        check_models,
        "_arch_precheck_for_model",
        lambda _model: ("llava_qwen2", "fastvlm", True),
    )
    assert (
        check_models._arch_precheck_summary("org/model")
        == "yes (model_type llava_qwen2 via fastvlm)"
    )

    monkeypatch.setattr(
        check_models,
        "_arch_precheck_for_model",
        lambda _model: ("new_arch", "new_arch", False),
    )
    assert check_models._arch_precheck_summary("org/model") == "no (model_type new_arch)"

    monkeypatch.setattr(
        check_models,
        "_arch_precheck_for_model",
        lambda _model: (None, None, None),
    )
    assert check_models._arch_precheck_summary("org/model") is None


# --- Capability-aware discovery (upstream alignment design §1-3) ---------------


def _capability_for(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: dict[str, object] | None,
    model_index: dict[str, object] | None = None,
) -> check_models.ImageCapability:
    """Classify a fake repo whose config.json / model_index.json are supplied inline."""

    def _read(_repo: object, file_name: str) -> dict[str, object] | None:
        if file_name == "config.json":
            return config
        if file_name == "model_index.json":
            return model_index
        return None

    monkeypatch.setattr(check_models, "_read_cached_repo_json", _read)
    return check_models._classify_image_capability(object())


VLM_CONFIG: dict[str, object] = {
    "model_type": "qwen3_vl",
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "vision_config": {"depth": 24},
    "image_token_id": 151655,
    "id2label": {"0": "LABEL_0"},  # present on real VLMs; must not read as reranker
}


class TestImageCapabilityClassifier:
    """Tri-state capability classification with explicit evidence."""

    def test_vlm_config_is_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A config with vision evidence is a positive image-to-text verdict."""
        cap = _capability_for(monkeypatch, config=VLM_CONFIG)

        assert cap.verdict == "yes"
        assert cap.purpose == "image_to_text"
        assert "image_token_id" in cap.evidence[0]
        assert cap.skip_reason is None

    def test_text_only_config_is_no(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A generative config with no modality evidence is text-only."""
        cap = _capability_for(
            monkeypatch,
            config={"model_type": "afm7", "architectures": ["Afm7ForCausalLM"]},
        )

        assert cap.verdict == "no"
        assert cap.purpose == "text_only"
        assert cap.skip_reason == (
            "model purpose: text-only generation "
            "(model_type=afm7; no vision_config/image token keys)"
        )

    @pytest.mark.parametrize(
        ("config", "model_index", "purpose"),
        [
            (
                {"model_type": "bert", "mlx_embeddings": {"kind": "embedding"}},
                None,
                "embedding",
            ),
            (
                {
                    "model_type": "xlm_roberta",
                    "architectures": ["XLMRobertaForSequenceClassification"],
                },
                None,
                "reranker",
            ),
            (
                {"model_type": "qwen3", "speculators_model_type": "eagle3"},
                None,
                "speculative_drafter",
            ),
            (
                {"model_type": "flux"},
                {"_class_name": "FluxPipeline"},
                "image_or_video_generation",
            ),
            (
                {
                    "model_type": "voice_gen",
                    "architectures": ["VoiceGenForConditionalGeneration"],
                    "audio_config": {},
                },
                None,
                "audio_or_other_generation",
            ),
        ],
    )
    def test_non_image_kinds_are_no_with_distinct_reasons(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: dict[str, object],
        model_index: dict[str, object] | None,
        purpose: str,
    ) -> None:
        """Each non-image model kind yields a distinct explicit skip reason."""
        cap = _capability_for(monkeypatch, config=config, model_index=model_index)

        assert cap.verdict == "no"
        assert cap.purpose == purpose
        assert cap.evidence
        assert cap.skip_reason is not None
        assert cap.skip_reason.startswith("model purpose: ")

    def test_id2label_alone_does_not_mean_reranker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Real VLM configs carry id2label; only sequence-classifier signals count."""
        cap = _capability_for(
            monkeypatch,
            config={"model_type": "mystery", "id2label": {"0": "x"}, "num_labels": 1},
        )

        assert cap.purpose != "reranker"

    def test_unfamiliar_config_is_unknown_and_selected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Insufficient evidence yields unknown, which default discovery still runs."""
        cap = _capability_for(monkeypatch, config={"model_type": "brand_new_arch"})

        assert cap.verdict == "unknown"
        assert cap.skip_reason is None
        entry = check_models.CachedModelEligibility(
            repo_id="org/new", supported=True, capability=cap
        )
        assert entry.selected is True
        assert entry.skip_reasons == ()

    def test_missing_config_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No readable config cannot be classified and stays unknown."""
        cap = _capability_for(monkeypatch, config=None)

        assert cap.verdict == "unknown"
        assert cap.evidence == ("no readable config.json",)


class TestCapabilityAwareSelection:
    """Discovery skips only confident non-image repos and reports every skip."""

    def _entries(self) -> tuple[check_models.CachedModelEligibility, ...]:
        yes = check_models.ImageCapability(
            "yes", "image_to_text", ("image-input keys: vision_config",)
        )
        no = check_models.ImageCapability(
            "no", "text_only", ("model_type=afm7", "no vision_config/image token keys")
        )
        unknown = check_models.ImageCapability("unknown", "unknown", ("model_type=new",))
        return (
            check_models.CachedModelEligibility("org/vlm", supported=True, capability=yes),
            check_models.CachedModelEligibility("org/text-only", supported=True, capability=no),
            check_models.CachedModelEligibility("org/new-arch", supported=True, capability=unknown),
            check_models.CachedModelEligibility(
                "org/bad-layout",
                supported=False,
                reasons=("missing tokenizer_config.json",),
                capability=yes,
            ),
        )

    def test_selection_runs_yes_and_unknown_and_skips_no(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Default discovery runs yes+unknown, skips no, and names every skip."""
        with caplog.at_level(logging.WARNING):
            selected = check_models._supported_cached_model_ids_with_skipped_logging(
                self._entries()
            )

        assert selected == ["org/new-arch", "org/vlm"]
        text = caplog.text
        # Every skipped repo is named with a concrete reason.
        assert "org/text-only: model purpose: text-only generation" in text
        assert "org/bad-layout: cache layout: missing tokenizer_config.json" in text
        # Unknown is a warning on a selected model, not an exclusion.
        assert "unknown image capability" in text
        assert "org/new-arch" in text

    def test_combined_skip_reasons_layout_and_capability(self) -> None:
        """Layout and capability reasons are both reported, in that order."""
        entry = check_models.CachedModelEligibility(
            "org/both",
            supported=False,
            reasons=("missing config.json",),
            capability=check_models.ImageCapability(
                "no", "embedding", ("mlx_embeddings.kind=embedding",)
            ),
        )

        assert entry.selected is False
        assert entry.skip_reasons == (
            "cache layout: missing config.json",
            "model purpose: embedding model (mlx_embeddings.kind=embedding)",
        )

    def test_explicit_models_override_with_visible_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Explicit --models runs a non-image repo but keeps the classification visible."""
        monkeypatch.setattr(check_models, "get_cached_model_eligibility", self._entries)

        with caplog.at_level(logging.WARNING):
            check_models._warn_explicit_non_image_models(["org/text-only", "org/vlm"])

        assert "org/text-only classifies as non-image" in caplog.text
        assert "org/vlm classifies" not in caplog.text

    def test_cache_discovery_records_retain_classification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run.json retains classification, evidence, and decision per cached repo."""
        monkeypatch.setattr(check_models, "get_cached_model_eligibility", self._entries)

        records = check_models._cache_discovery_records()

        by_id = {r["repo_id"]: r for r in records}
        assert by_id["org/text-only"]["selected"] is False
        assert by_id["org/text-only"]["capability_verdict"] == "no"
        assert by_id["org/text-only"]["model_purpose"] == "text_only"
        assert by_id["org/text-only"]["skip_reasons"] == [
            "model purpose: text-only generation (model_type=afm7; no vision_config/image token keys)"
        ]
        assert by_id["org/vlm"]["selected"] is True
        assert by_id["org/new-arch"]["capability_verdict"] == "unknown"
