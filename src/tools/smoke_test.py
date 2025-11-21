"""Simple smoke test for MLX-VLM models.

Loads one or more models, runs a minimal vision-language and language-only
generation with provided prompts and images, and prints a short summary along
with basic system information. Designed to be lightweight and runnable in CI
or local environments with or without the native MLX library.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import psutil
from mlx_vlm import generate, load  # type: ignore[import]
from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore[import]
from mlx_vlm.utils import load_config  # type: ignore[import]
from mlx_vlm.version import __version__  # type: ignore[import]
from tqdm import tqdm
from transformers import __version__ as transformers_version

mx: Any | None = None
try:  # Soft import: allow environments lacking native MLX lib to still run tests
    import mlx.core as _mx

    mx = cast("Any", _mx)
    _MLX_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    mx = None
    _MLX_AVAILABLE = False


# Rich console and panel - with fallback shims for environments without rich
class _ConsoleShim:
    """Minimal Console shim when 'rich' is unavailable."""

    def print(self, *args: object, **kwargs: object) -> None:
        """Shim for Console.print that does nothing."""
        del args, kwargs


class _PanelShim:
    """Minimal Panel shim when 'rich' is unavailable."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize shim; accepts and ignores all arguments."""
        del args, kwargs


if TYPE_CHECKING:
    from rich.console import Console as ConsoleType
    from rich.panel import Panel as PanelType
else:  # pragma: no branch - runtime import with fallback
    try:
        from rich.console import Console as ConsoleType
        from rich.panel import Panel as PanelType
    except ImportError:  # pragma: no cover - allow running without rich

        class ConsoleType(_ConsoleShim):
            """Fallback Console when 'rich' is unavailable."""

        class PanelType(_PanelShim):
            """Fallback Panel when 'rich' is unavailable."""


# Initialize console
console = ConsoleType()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the smoke test script."""
    parser = argparse.ArgumentParser(description="Test MLX-VLM models")
    parser.add_argument(
        "--models-file",
        type=str,
        required=True,
        help="Path to file containing model paths, one per line",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to test image(s)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image.",
        help="Vision-language prompt to test",
    )
    parser.add_argument(
        "--language-only-prompt",
        type=str,
        default="Hi, how are you?",
        help="Language-only prompt to test",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    return parser.parse_args()


def get_device_info() -> dict[str, Any] | None:
    """Return basic GPU information from macOS system_profiler, if available."""
    # Disable tokenizers parallelism to avoid deadlocks after forking
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        data = subprocess.check_output(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],  # S607
            text=True,
            timeout=5,
        )
        return cast("dict[str, Any]", json.loads(data))
    except (
        subprocess.SubprocessError,
        json.JSONDecodeError,
        FileNotFoundError,
        PermissionError,
    ) as e:
        console.print(f"[bold yellow]Could not retrieve GPU information: {e}")
        return None


def load_model(
    model_path: str,
) -> tuple[Any | None, Any | None, dict[str, Any] | None, bool]:
    """Load a model and its processor and config.

    Returns a 4-tuple of (model, processor, config, error_flag).
    """
    try:
        console.print("[bold green]Loading model...")
        model, processor = load(model_path, trust_remote_code=True)
        config = load_config(model_path, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]✗[/] Failed to load model: {e!s}")
        return None, None, None, True
    else:
        console.print("[bold green]✓[/] Model loaded successfully")
        return model, processor, config, False


class GenerationContext(NamedTuple):
    """Bundle parameters needed for a single generation call."""

    model: Any
    processor: Any
    config: dict[str, Any]
    model_path: str
    test_inputs: dict[str, Any]


def run_generation(
    ctx: GenerationContext,
    *,
    vision_language: bool = True,
) -> bool:
    """Run a single generation pass and return True on error, False on success."""
    test_type = "vision-language" if vision_language else "language-only"
    try:
        console.print(f"[bold yellow]Testing {test_type} generation...")

        prompt = (
            ctx.test_inputs["prompt"]
            if vision_language
            else ctx.test_inputs["language_only_prompt"]
        )
        num_images = len(cast("list[str]", ctx.test_inputs["image"])) if vision_language else 0

        formatted_prompt = apply_chat_template(
            ctx.processor,
            ctx.config,
            prompt,
            num_images=num_images,
        )

        image_arg: list[str] | None = (
            cast("list[str]", ctx.test_inputs["image"]) if vision_language else None
        )

        # Build kwargs conditionally to avoid passing None for image parameter
        generate_kwargs: dict[str, Any] = {
            "model": ctx.model,
            "processor": ctx.processor,
            "prompt": cast("Any", formatted_prompt),
            "verbose": True,
            **ctx.test_inputs["kwargs"],
        }
        if image_arg is not None:
            generate_kwargs["image"] = image_arg

        output = generate(**generate_kwargs)

        # Skip emptiness checks for models known to emit empty outputs for a mode
        skip_check = (vision_language and "deepseek-vl2-tiny" in ctx.model_path) or (
            not vision_language and "paligemma" in ctx.model_path
        )
        if not skip_check and not (isinstance(output, str) and len(output) > 0):
            console.print(
                f"[bold red]✗[/] {test_type} generation produced empty output",
            )
            return True
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]✗[/] {test_type} generation failed: {e!s}")
        console.print(f"[dim]{traceback.format_exc()}[/]")
        return True
    else:
        console.print(f"[bold green]✓[/] {test_type} generation successful")
        return False


def main() -> None:
    """Entrypoint for the smoke test script."""
    args = parse_args()
    if not _MLX_AVAILABLE:
        console.print(
            "[bold yellow]Skipping smoke test: MLX core library not available in this env.[/]",
        )
        return

    # Load models list
    models_path = Path(args.models_file)
    with models_path.open(encoding="utf-8") as f:
        models = [line.strip() for line in f]

    test_inputs = _build_test_inputs(args)
    results: list[str] = []
    for model_path in tqdm(models):
        _err, label = _test_one_model(model_path, test_inputs)
        results.append(label)

    _render_summary(results)
    console.print(_make_system_panel())


def _build_test_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Construct test inputs from CLI args."""
    return {
        "image": args.image,
        "prompt": args.prompt,
        "language_only_prompt": args.language_only_prompt,
        "kwargs": {
            "temp": args.temperature,
            "max_tokens": args.max_tokens,
        },
    }


def _test_one_model(model_path: str, test_inputs: dict[str, Any]) -> tuple[bool, str]:
    """Load and run both generation modes for a single model path."""
    console.print(PanelType(f"Testing {model_path}", style="bold blue"))
    model, processor, config, error = load_model(model_path)

    if not error and model:
        console.print("")
        ctx = GenerationContext(
            model=model,
            processor=processor,
            config=cast("dict[str, Any]", config),
            model_path=model_path,
            test_inputs=test_inputs,
        )
        error |= run_generation(ctx, vision_language=True)
        console.print("")
        if _MLX_AVAILABLE:
            mx_mod = cast("Any", mx)
            mx_mod.metal.clear_cache()
            mx_mod.metal.reset_peak_memory()
        error |= run_generation(ctx, vision_language=False)
        console.print("")

    console.print("[bold blue]Cleaning up...")
    del model, processor
    if _MLX_AVAILABLE:
        mx_mod = cast("Any", mx)
        mx_mod.metal.clear_cache()
        mx_mod.metal.reset_peak_memory()
    console.print("[bold green]✓[/] Cleanup complete\n")
    label = f"[bold {'green' if not error else 'red'}]{'✓' if not error else '✗'}[/] {model_path}"
    return error, label


def _render_summary(results: list[str]) -> None:
    """Render results list and overall summary to the console."""
    success = all(result.startswith("[bold green]") for result in results)
    panel_style = "bold green" if success else "bold red"
    console.print(PanelType("\n".join(results), title="Results", style=panel_style))
    console.print(
        (
            f"[bold {'green' if success else 'red'}]"
            f"{'All' if success else 'Some'} models tested "
            f"{'successfully' if success else 'failed to test'}"
        ),
    )


def _make_system_panel() -> PanelType:
    """Create a panel with system information suitable for display."""
    device_info = get_device_info()
    chip_name: str = "Unknown"
    gpu_cores: str | int = "Unknown"
    if isinstance(device_info, dict):
        try:
            disp_list = device_info.get("SPDisplaysDataType", [])
            if disp_list:
                first = disp_list[0]
                chip_name = first.get("_name", chip_name)
                gpu_cores = first.get("sppci_cores", gpu_cores)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[dim]Failed to parse GPU info: {exc!s}[/]")

    mlx_version = getattr(mx, "__version__", "unavailable") if _MLX_AVAILABLE else "unavailable"
    py_version = sys.version.split()[0]
    mac_ver = platform.mac_ver()[0]
    is_arm64 = platform.machine() == "arm64"
    if is_arm64:
        system_info_str = textwrap.dedent(
            f"""
            MAC OS:       v{mac_ver}
            Python:       v{py_version}
            MLX:          v{mlx_version}
            MLX-VLM:      v{__version__}
            Transformers: v{transformers_version}

            Hardware:
            • Chip:       {chip_name}
            • RAM:        {psutil.virtual_memory().total / (1024**3):.1f} GB
            • CPU Cores:  {psutil.cpu_count(logical=False)}
            • GPU Cores:  {gpu_cores}
            """,
        ).strip()
    else:
        system_info_str = "Not running on Apple Silicon"
    return PanelType(title="System Information", renderable=system_info_str, style="bold blue")


if __name__ == "__main__":
    main()
