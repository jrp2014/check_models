"""Probe the local MLX Metal backend for the M5 NAX matmul regression."""

from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Final, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_RELATIVE_ERROR_THRESHOLD: Final = 0.1
EXIT_OK: Final = 0
EXIT_BUGGY: Final = 1
EXIT_UNAVAILABLE: Final = 2
METAL_VERSION_TIMEOUT_SECONDS: Final = 5
MATMUL_LEFT_SHAPE: Final = (64, 512)
MATMUL_RIGHT_SHAPE: Final = (512, 512)

ProbeStatus = Literal["ok", "buggy", "unavailable"]


def classify_relative_error(
    relative_error: float,
    threshold: float = DEFAULT_RELATIVE_ERROR_THRESHOLD,
) -> Literal["ok", "buggy"]:
    """Classify the GPU-vs-CPU matmul relative error."""
    if relative_error < threshold:
        return "ok"
    return "buggy"


def format_probe_message(
    *,
    status: ProbeStatus,
    relative_error: float | None = None,
    threshold: float = DEFAULT_RELATIVE_ERROR_THRESHOLD,
    metal_version: str | None = None,
    detail: str | None = None,
) -> str:
    """Format a concise status message for humans running maintenance tools."""
    version_line = f" Metal compiler: {metal_version}" if metal_version else ""
    detail_line = f" Detail: {detail}" if detail else ""

    if status == "ok" and relative_error is not None:
        return (
            "[bugtest] MLX bf16 GPU matmul is OK "
            f"(relative error {relative_error:.6f} < {threshold:.6f})."
            f"{version_line} Note: this verifies the installed MLX backend, not a "
            "fresh local Metal compiler rebuild unless update.sh just rebuilt mlx."
        )

    if status == "buggy" and relative_error is not None:
        return (
            "[bugtest] MLX bf16 GPU matmul still appears broken "
            f"(relative error {relative_error:.6f} >= {threshold:.6f})."
            f"{version_line} Use a prebuilt mlx-metal metallib, rebuild mlx with a "
            "fixed/older Xcode Metal toolchain, or run with "
            "MLX_METAL_GPU_ARCH=applegpu_g16s as the slow safe path."
        )

    return (
        "[bugtest] MLX Metal backend probe is unavailable."
        f"{version_line}{detail_line} Skipping the reminder."
    )


def get_metal_version() -> str | None:
    """Return the active Metal compiler version line when xcrun can report it."""
    try:
        result = subprocess.run(
            ["xcrun", "metal", "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=METAL_VERSION_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        normalized = line.strip()
        if normalized:
            return normalized
    return None


def _mlx_core() -> Any:
    """Import mlx.core only when the probe needs to touch the Metal backend."""
    return cast("Any", importlib.import_module("mlx.core"))


def run_bfloat16_matmul_probe() -> float:
    """Return the relative error for bf16 GPU matmul compared with CPU matmul."""
    mx = _mlx_core()

    left = mx.random.normal(MATMUL_LEFT_SHAPE).astype(mx.bfloat16)
    right = mx.random.normal(MATMUL_RIGHT_SHAPE).astype(mx.bfloat16)
    mx.eval(left, right)

    with mx.stream(mx.Device(mx.gpu)):
        gpu_result = left @ right
        mx.eval(gpu_result)

    with mx.stream(mx.Device(mx.cpu)):
        cpu_result = left @ right
        mx.eval(cpu_result)

    gpu_float = gpu_result.astype(mx.float32)
    cpu_float = cpu_result.astype(mx.float32)
    mx.eval(gpu_float, cpu_float)

    error_norm = float(((gpu_float - cpu_float) ** 2).sum() ** 0.5)
    cpu_norm = float((cpu_float**2).sum() ** 0.5)
    return error_norm / (cpu_norm + 1e-9)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Metal backend probe."""
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the installed MLX Metal backend still has the M5 "
            "bf16/fp16 NAX matmul regression."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_RELATIVE_ERROR_THRESHOLD,
        help="Maximum acceptable GPU-vs-CPU relative error.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print the probe result but always exit successfully.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Metal backend probe and return a process exit code."""
    args = build_parser().parse_args(argv)
    metal_version = get_metal_version()

    if platform.system() != "Darwin":
        print(
            format_probe_message(
                status="unavailable",
                metal_version=metal_version,
                detail="not running on macOS",
            ),
        )
        return EXIT_OK

    try:
        relative_error = run_bfloat16_matmul_probe()
    except Exception as exc:
        print(
            format_probe_message(
                status="unavailable",
                metal_version=metal_version,
                detail=str(exc),
            ),
            file=sys.stderr,
        )
        return EXIT_OK if args.warn_only else EXIT_UNAVAILABLE

    status = classify_relative_error(relative_error, threshold=args.threshold)
    print(
        format_probe_message(
            status=status,
            relative_error=relative_error,
            threshold=args.threshold,
            metal_version=metal_version,
        ),
    )
    if status == "buggy" and not args.warn_only:
        return EXIT_BUGGY
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
