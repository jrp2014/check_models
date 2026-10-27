"""Record the result of rerunning one harness observation outside the harness.

The note is appended to ``native_repro_notes.yaml`` in the output root, keyed
to the model revision and the mlx, mlx-vlm and transformers builds of the
retained run. The run summary's *Native repro* column then shows it as
confirmed or not reproduced, and goes back to "not yet checked" once the
revision or any build changes. Record a check only after running it natively
(see the native-mlx-vlm-repro skill); recurrence in the harness is not one.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_models import (
    NATIVE_REPRO_NOTES_FILENAME,
    _native_repro_codes,
    _native_repro_stack,
    _parse_native_repro_notes,
)
from tools.add_accuracy_note import append_note, publish_run_summary, retained_model_result

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
_NEW_FILE_HEADER = """\
# Native reproduction checks for observations in the run retained here.
# Each applies only while its model revision and mlx/mlx-vlm/transformers
# builds match a run; the run summary then shows its status.
# Add one with: python -m tools.add_native_repro_note MODEL OBSERVATION STATUS ...
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append the result of a native rerun of one harness observation.",
    )
    parser.add_argument("model", help="Model ID exactly as results.jsonl records it.")
    parser.add_argument(
        "observation",
        help="Observation code from results.jsonl (such as repeated_output), or crashed.",
    )
    parser.add_argument("status", choices=("confirmed", "not_reproduced"))
    parser.add_argument("--checker", required=True, help="Who ran the native check.")
    parser.add_argument(
        "--evidence",
        required=True,
        help="The native command and what it showed, in a sentence or two.",
    )
    parser.add_argument(
        "--checked",
        type=date.fromisoformat,
        default=datetime.now().astimezone().date(),
        help="Check date (YYYY-MM-DD); defaults to today.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Output root holding results.jsonl and the notes file.",
    )
    return parser


def _note_entry(args: argparse.Namespace) -> dict[str, str]:
    """Build the note from the retained run, refusing an observation the run lacks."""
    run, result, revision = retained_model_result(args.output_dir, args.model)
    stack = _native_repro_stack(run.metadata)
    if stack is None:
        msg = "the retained run does not record its mlx and mlx-vlm builds"
        raise ValueError(msg)
    codes = _native_repro_codes(result)
    if args.observation not in codes:
        msg = (
            f"{args.model} has no {args.observation} needing native reproduction in the "
            f"retained run (it has: {', '.join(codes) or 'none'})"
        )
        raise ValueError(msg)
    return {
        "model": args.model,
        "model_revision": revision,
        "observation": args.observation,
        "stack": stack,
        "status": args.status,
        "checker": args.checker,
        "checked": args.checked.isoformat(),
        "evidence": args.evidence,
    }


def main(argv: list[str] | None = None) -> int:
    """Append one validated native-reproduction note and print it."""
    args = _build_parser().parse_args(argv)
    notes_path = args.output_dir.expanduser().resolve() / NATIVE_REPRO_NOTES_FILENAME
    try:
        addition = append_note(
            notes_path, _NEW_FILE_HEADER, _note_entry(args), _parse_native_repro_notes
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Not recorded: {error}", file=sys.stderr)
        return 1
    print(f"Appended to {notes_path}:\n{addition}", end="")
    publish_run_summary(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
