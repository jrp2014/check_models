"""Probe Qwen3-VL Metal GPU address faults with upstream mlx-vlm only."""

from __future__ import annotations

import argparse
import gc
import importlib
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

GenerateCallable = Callable[..., Any]
PromptTemplateCallable = Callable[..., Any]
LoadCallable = Callable[..., tuple[Any, Any]]

DEFAULT_MODELS: Final = (
    "mlx-community/Qwen3-VL-2B-Thinking-bf16",
    "Qwen/Qwen3-VL-2B-Instruct",
)
DEFAULT_PROMPT: Final = "Describe this image briefly."
DEFAULT_MAX_TOKENS: Final = 8
RISK_NOTICE: Final = (
    "WARNING: this probe can trigger a native Metal abort. Repeated paired runs "
    "have wedged WindowServer/GPU state on affected systems; run it sparingly "
    "after saving work, and reboot before collecting a clean trace."
)
INTERPRETATION: Final = (
    "Interpretation: if each model succeeds alone but a paired or repeated "
    "single-process run faults, the evidence points at process-global MLX/Metal "
    "runtime state rather than a simple bad checkpoint."
)


def _positive_int(value: str) -> int:
    """Parse a positive integer argument."""
    parsed = int(value)
    if parsed < 1:
        message = "must be >= 1"
        raise argparse.ArgumentTypeError(message)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3-VL models through mlx-vlm directly to distinguish "
            "single-model/image failures from sequential in-process failures."
        ),
    )
    parser.add_argument("image", type=Path, help="Path to a local image.")
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        dest="models",
        help=(
            "Model ID to run. Repeat to define an explicit sequence. Defaults "
            "to the two Qwen3-VL 2B models."
        ),
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Run the selected model sequence in reverse order.",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=1,
        help="Repeat the selected model sequence N times in the same process.",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens to generate per model.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send with the image.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable mlx-vlm verbose output.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print a conservative probe matrix without running GPU inference.",
    )
    return parser


def _model_sequence(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the concrete model sequence to run."""
    models = tuple(args.models or DEFAULT_MODELS)
    if args.reverse:
        models = tuple(reversed(models))
    return models * args.repeat


def _build_plan_command(image_path: Path, extra_args: Sequence[str]) -> str:
    """Build a pasteable command for one probe step."""
    tokens = [
        "python",
        "tools/qwen3_vl_sequential_repro.py",
        str(image_path),
        *extra_args,
    ]
    return " ".join(shlex.quote(token) for token in tokens)


def build_probe_plan(image_path: Path) -> str:
    """Return a conservative command matrix for classifying the fault mode."""
    thinking_model, instruct_model = DEFAULT_MODELS
    commands = (
        (
            "1. Thinking model alone",
            (("-m", thinking_model),),
        ),
        (
            "2. Instruct model alone",
            (("-m", instruct_model),),
        ),
        (
            "3. Default pair in one process",
            (),
        ),
        (
            "4. Reverse pair in one process",
            (("--reverse",),),
        ),
        (
            "5. Same model twice in one process",
            (("-m", instruct_model), ("-m", instruct_model)),
        ),
    )

    lines = [
        RISK_NOTICE,
        "",
        "Run the single-model commands first. Only run the paired/repeated commands",
        "when the single-model probes succeed and you are ready to collect crash evidence.",
        "",
    ]
    for title, arg_groups in commands:
        flattened_args = [arg for group in arg_groups for arg in group]
        lines.append(title)
        lines.append(_build_plan_command(image_path, flattened_args))
        lines.append("")
    lines.append(INTERPRETATION)
    return "\n".join(lines)


def _load_runtime_symbols() -> tuple[Any, GenerateCallable, PromptTemplateCallable, LoadCallable]:
    """Import MLX and mlx-vlm only when a probe is actually going to run."""
    mx = cast("Any", importlib.import_module("mlx.core"))
    from mlx_vlm.generate import generate  # noqa: PLC0415
    from mlx_vlm.prompt_utils import apply_chat_template  # noqa: PLC0415
    from mlx_vlm.utils import load  # noqa: PLC0415

    return mx, generate, apply_chat_template, load


def _cleanup_mlx_runtime(mx: Any) -> None:
    """Synchronize and clear MLX runtime state between model runs."""
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the minimal upstream-only Qwen3-VL fault probe."""
    args = build_parser().parse_args(argv)
    image_path = args.image.expanduser().resolve()
    if args.plan:
        print(build_probe_plan(image_path))
        return 0
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    models = _model_sequence(args)
    print(RISK_NOTICE, flush=True)
    print(INTERPRETATION, flush=True)
    print("Model sequence:", ", ".join(models), flush=True)
    mx, generate_fn, apply_chat_template_fn, load_fn = _load_runtime_symbols()

    for model_id in models:
        print(f"\nRunning {model_id}", flush=True)
        model, processor = load_fn(model_id)
        formatted_prompt = cast(
            "str",
            apply_chat_template_fn(
                processor,
                model.config,
                args.prompt,
                num_images=1,
            ),
        )
        result = generate_fn(
            model,
            processor,
            formatted_prompt,
            image=str(image_path),
            max_tokens=args.max_tokens,
            verbose=not args.quiet,
        )
        print(f"Completed {model_id}: {result.text!r}", flush=True)

        del result, formatted_prompt, model, processor
        _cleanup_mlx_runtime(mx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
