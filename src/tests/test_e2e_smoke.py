"""End-to-end smoke tests for the VLM check pipeline.

These tests verify the complete workflow with actual model inference.
They are marked with `pytest.mark.slow` and `pytest.mark.e2e` for selective
execution, as they require downloading and running models.

Run these tests explicitly with:
    pytest tests/test_e2e_smoke.py -v

Or run all tests including slow ones:
    pytest --run-slow

Skip these in CI by default (they require MLX hardware and model downloads).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

# =============================================================================
# EARLY ENVIRONMENT SETUP (MUST happen before huggingface_hub imports)
# =============================================================================

# Set up HF cache directory early, before any huggingface_hub functions cache the path.
# Strategy (following HuggingFace documentation):
# 1. If HF_HUB_CACHE is set → use it (user explicitly configured)
# 2. Else if default cache exists (~/.cache/huggingface/hub) → use it
# 3. Else create temp cache (CI environment without cache)
_DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

if "HF_HUB_CACHE" not in os.environ and not _DEFAULT_HF_CACHE.exists():
    # CI environment - create temp cache to prevent CacheNotFound
    _temp_hf_cache = Path(tempfile.gettempdir()) / "pytest_hf_cache"
    _temp_hf_cache.mkdir(parents=True, exist_ok=True)
    (_temp_hf_cache / "hub").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(_temp_hf_cache / "hub")
    os.environ["HF_HOME"] = str(_temp_hf_cache)

# Now import huggingface_hub after environment is configured
import pytest  # noqa: E402 - after HF cache env setup
from huggingface_hub import scan_cache_dir  # noqa: E402 - after HF cache env setup
from huggingface_hub.errors import CacheNotFound  # noqa: E402 - after HF cache env setup
from PIL import Image  # noqa: E402 - after HF cache env setup

import check_models  # noqa: E402 - after HF cache env setup
from tools import safe_io  # noqa: E402 - after HF cache env setup

# Fixture model - small, fast, reliable MLX conversion
# Smallest usable model in the standing suite; nanoLLaVA-1.5 was retired
# from the cache (superseded, unusable output) and would skip these tests.
FIXTURE_MODEL = "LiquidAI/LFM2.5-VL-450M-MLX-bf16"


class CLIResult(NamedTuple):
    """Result of a CLI execution for testing."""

    exit_code: int
    stdout: str
    stderr: str


def _run_cli(args: list[str], capsys: pytest.CaptureFixture[str]) -> CLIResult:
    """Helper to run the CLI main function directly."""
    test_args = ["check_models.py", *args]
    exit_code = 0
    with patch.object(sys, "argv", test_args):
        try:
            check_models.main_cli()
        except SystemExit as e:
            exit_code = e.code if isinstance(e.code, int) else (1 if e.code else 0)

    captured = capsys.readouterr()
    return CLIResult(exit_code, captured.out, captured.err)


def _get_e2e_output_args(output_dir: Path) -> list[str]:
    """Return the single output-root argument for E2E runs."""
    return ["--output-dir", str(output_dir)]


def test_get_e2e_output_args_redirects_retained_artifacts(tmp_path: Path) -> None:
    """E2E output helper redirects the whole retained layout with one flag."""
    output_dir = tmp_path / "output"
    assert _get_e2e_output_args(output_dir) == ["--output-dir", str(output_dir)]


@pytest.fixture
def e2e_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for E2E test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def e2e_test_image(tmp_path: Path) -> Path:
    """Create a realistic test image for E2E testing."""
    img_path = tmp_path / "e2e_test.jpg"
    img = Image.new("RGB", (640, 480), color=(135, 206, 235))
    pixels = img.load()
    if pixels:
        for x in range(540, 600):
            for y in range(40, 100):
                if (x - 570) ** 2 + (y - 70) ** 2 < 900:
                    pixels[x, y] = (255, 255, 0)
        for x in range(640):
            for y in range(380, 480):
                pixels[x, y] = (34, 139, 34)
        for x in range(200, 350):
            for y in range(250, 380):
                pixels[x, y] = (139, 90, 43)
        for x in range(180, 370):
            for y in range(200, 250):
                pixels[x, y] = (178, 34, 34)
    img.save(img_path, "JPEG", quality=85)
    return img_path


def _check_model_cached(model_id: str) -> bool:
    """Check if a model is already cached locally."""
    try:
        repos = scan_cache_dir().repos
    except (OSError, ValueError, RuntimeError, CacheNotFound):
        return False
    else:
        repo_ids = [r.repo_id for r in repos]
        return model_id in repo_ids


def _write_tiny_qwen2_vl_checkpoint(root: Path) -> Path:
    """Write a random-weight Qwen2-VL checkpoint of about 150 KB, then return its path.

    Borrowed from upstream mlx-vlm's tests (``test_extraction_models``): a
    tiny checkpoint written to disk lets the real ``load`` →
    ``apply_chat_template`` → ``stream_generate`` path run without a download.
    The language head is zeroed, so greedy decoding repeats vocabulary id 0
    ("a") and the output is deterministic.
    """
    import mlx.core as mx  # noqa: PLC0415 - runtime deps are checked by the class skip
    from mlx.utils import tree_flatten  # noqa: PLC0415 - as above
    from mlx_vlm.models import qwen2_vl  # noqa: PLC0415 - as above
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: PLC0415 - as above
    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415 - as above

    root.mkdir(parents=True)
    filler, end_of_turn = "<|endoftext|>", "<|im_end|>"
    specials = [
        filler,
        "<|im_start|>",
        end_of_turn,
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    words = [*"abcdefghijklmnopqrstuvwxyz", "user", "assistant", "describe", "image"]
    vocab = {token: index for index, token in enumerate([*words, *specials])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token=filler))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.decoder = decoders.WordPiece(prefix="##")
    chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }} "
        "{% if m['content'] is string %}{{ m['content'] }}{% else %}"
        "{% for c in m['content'] %}{% if c['type'] == 'image' %}"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}"
        "<|im_end|> {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant {% endif %}"
    )
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token=end_of_turn,
        pad_token=filler,
        unk_token=filler,
        additional_special_tokens=specials,
        chat_template=chat_template,
    ).save_pretrained(root)
    # One 56x56 image: 4x4 patches of 14 px, merged 2x2 into 4 image tokens.
    safe_io.write_text_no_follow(
        root / "preprocessor_config.json",
        json.dumps(
            {
                "image_processor_type": "Qwen2VLImageProcessor",
                "processor_class": "Qwen2VLProcessor",
                "patch_size": 14,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "min_pixels": 56 * 56,
                "max_pixels": 56 * 56,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            },
        ),
    )
    config = {
        "model_type": "qwen2_vl",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "vocab_size": len(vocab),
        "max_position_embeddings": 256,
        "rope_scaling": {"type": "mrope", "mrope_section": [2, 1, 1]},
        "eos_token_id": [vocab[end_of_turn]],
        "image_token_id": vocab["<|image_pad|>"],
        "video_token_id": vocab["<|video_pad|>"],
        "vision_start_token_id": vocab["<|vision_start|>"],
        "vision_config": {
            "depth": 1,
            "embed_dim": 16,
            "hidden_size": 16,
            "num_heads": 2,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }
    safe_io.write_text_no_follow(root / "config.json", json.dumps(config))
    model_config = qwen2_vl.ModelConfig.from_dict(dict(config))
    model_config.text_config = qwen2_vl.TextConfig.from_dict(model_config.text_config)
    model_config.vision_config = qwen2_vl.VisionConfig.from_dict(config["vision_config"])
    model = qwen2_vl.Model(model_config)
    lm_head = model.language_model.lm_head
    lm_head.weight = mx.zeros_like(lm_head.weight)
    mx.save_safetensors(str(root / "model.safetensors"), dict(tree_flatten(model.parameters())))
    return root


def _check_runtime_dependencies_ready() -> bool:
    """Check whether core runtime deps are currently usable for real inference."""
    required_runtime = {"mlx", "mlx-vlm"}
    return all(dep not in check_models.MISSING_DEPENDENCIES for dep in required_runtime)


# Mark all tests in this module as slow and e2e
pytestmark = [
    pytest.mark.slow,
    pytest.mark.e2e,
]


@pytest.mark.skipif(
    not _check_runtime_dependencies_ready(),
    reason="runtime deps unavailable (requires working mlx + mlx-vlm)",
)
class TestE2ESmoke:
    """End-to-end smoke tests that run actual model inference."""

    def test_dry_run_with_fixture_model(
        self,
        e2e_test_image: Path,
        e2e_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Dry-run should validate setup without invoking the model."""
        args = [
            *_get_e2e_output_args(e2e_output_dir),
            "--image",
            str(e2e_test_image),
            "--models",
            FIXTURE_MODEL,
            "--dry-run",
            "--prompt",
            "Describe this image.",
        ]
        result = _run_cli(args, capsys)
        assert result.exit_code == 0
        output = result.stdout + result.stderr
        assert "dry run" in output.lower()
        assert FIXTURE_MODEL in output
        assert "Describe this image" in output

    def test_full_inference_with_tiny_local_checkpoint(
        self,
        tmp_path: Path,
        e2e_test_image: Path,
        e2e_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The real mlx-vlm call path runs end to end with no download (runs in CI)."""
        model_dir = _write_tiny_qwen2_vl_checkpoint(tmp_path / "tiny-qwen2-vl")
        args = [
            *_get_e2e_output_args(e2e_output_dir),
            "--image",
            str(e2e_test_image),
            "--models",
            str(model_dir),
            "--prompt",
            "Describe this image.",
            "--max-tokens",
            "4",
            "--temperature",
            "0",
            "--timeout",
            "120",
        ]
        result = _run_cli(args, capsys)
        assert result.exit_code == 0, result.stdout + result.stderr

        jsonl_path = check_models.ReportOutputPaths.from_root(e2e_output_dir).jsonl
        records = [
            json.loads(line)
            for line in safe_io.read_text_no_follow(jsonl_path).splitlines()
            if line.strip()
        ]
        record = records[1]
        assert record["model"] == str(model_dir)
        assert record["assessment"]["execution"] == "completed"
        assert record["generated_text"] == "a a a a"

    @pytest.mark.skipif(
        not _check_model_cached(FIXTURE_MODEL),
        reason=(f"Model {FIXTURE_MODEL} not cached (requires pre-downloaded fixture model)"),
    )
    def test_full_inference_with_fixture_model(
        self,
        e2e_test_image: Path,
        e2e_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Full inference run should complete successfully and produce outputs."""
        default_index_existed = check_models.DEFAULT_OUTPUT_INDEX.exists()
        default_index_before = (
            safe_io.read_text_no_follow(check_models.DEFAULT_OUTPUT_INDEX)
            if default_index_existed
            else None
        )
        args = [
            *_get_e2e_output_args(e2e_output_dir),
            "--image",
            str(e2e_test_image),
            "--models",
            FIXTURE_MODEL,
            "--prompt",
            "Describe the main elements in this image briefly.",
            "--max-tokens",
            "100",
            "--timeout",
            "120",
        ]
        result = _run_cli(args, capsys)
        assert result.exit_code == 0

        retained_paths = check_models.ReportOutputPaths.from_root(e2e_output_dir)
        for output_path in (
            retained_paths.index,
            retained_paths.html,
            retained_paths.gallery_markdown,
            retained_paths.jsonl,
            retained_paths.diagnostics,
            retained_paths.log,
            retained_paths.environment,
        ):
            assert output_path.exists(), output_path

        assert check_models.DEFAULT_OUTPUT_INDEX.exists() is default_index_existed
        if default_index_before is not None:
            assert (
                safe_io.read_text_no_follow(check_models.DEFAULT_OUTPUT_INDEX)
                == default_index_before
            )

        records = [
            json.loads(line)
            for line in safe_io.read_text_no_follow(retained_paths.jsonl).splitlines()
            if line.strip()
        ]
        assert len(records) >= 2  # metadata header + at least 1 result
        # First line is metadata header
        assert records[0]["_type"] == "metadata"
        # Second line is the model result
        record = records[1]
        assert record["model"] == FIXTURE_MODEL
        assert record["assessment"]["execution"] == "completed"

    @pytest.mark.skipif(
        not _check_model_cached(FIXTURE_MODEL),
        reason=(f"Model {FIXTURE_MODEL} not cached (requires pre-downloaded fixture model)"),
    )
    def test_quality_analysis_produces_output(
        self,
        e2e_test_image: Path,
        e2e_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Quality analysis should run and detect potential issues."""
        args = [
            *_get_e2e_output_args(e2e_output_dir),
            "--image",
            str(e2e_test_image),
            "--models",
            FIXTURE_MODEL,
            "--prompt",
            "Describe this image in detail.",
            "--max-tokens",
            "150",
            "--verbose",
        ]
        result = _run_cli(args, capsys)
        assert result.exit_code == 0
        output = result.stdout + result.stderr
        assert any(
            word in output.lower() for word in ["quality", "tps", "generated", "tokens", "memory"]
        )

    def test_invalid_model_produces_error(
        self,
        e2e_test_image: Path,
        e2e_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Non-existent model should produce a clear error."""
        args = [
            *_get_e2e_output_args(e2e_output_dir),
            "--image",
            str(e2e_test_image),
            "--models",
            "nonexistent/fake-model-12345",
            "--timeout",
            "30",
        ]
        result = _run_cli(args, capsys)
        output = result.stdout + result.stderr
        # Model loading should fail with repository/model not found error
        assert (
            "not found" in output.lower()
            or "could not" in output.lower()
            or "failed" in output.lower()
        )


# Note: No cleanup fixture needed - tmp_path fixture handles cleanup automatically
