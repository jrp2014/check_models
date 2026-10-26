"""Record a person's accuracy note on one model's output in a retained run.

The note is appended to ``accuracy_notes.yaml`` in the output root, keyed to
the exact image, model revision and output that ``results.jsonl`` retains, so
the reports show it beside that output and stop showing it once any of them
changes. The note text is the reviewer's own judgement, checked against the
image; nothing here assesses accuracy.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_models import (
    ACCURACY_NOTES_FILENAME,
    ReportOutputPaths,
    _load_retained_run,
    _parse_accuracy_notes,
    _sha256_text,
)
from tools.safe_io import read_text_no_follow, write_text_no_follow

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
_NEW_FILE_HEADER = """\
# Human accuracy notes on the outputs retained in this directory.
# Each note applies only while its image, model revision and output digests
# match a run; the gallery and HTML report then show it beside that output.
# Add one with: python -m tools.add_accuracy_note MODEL --reviewer NAME --note TEXT
notes:
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append a person's accuracy note on one model's retained output.",
    )
    parser.add_argument("model", help="Model ID exactly as results.jsonl records it.")
    parser.add_argument("--reviewer", required=True, help="Who checked the output.")
    parser.add_argument(
        "--note",
        required=True,
        help="What the output gets right or wrong, checked against the image.",
    )
    parser.add_argument(
        "--reviewed",
        type=date.fromisoformat,
        default=datetime.now().astimezone().date(),
        help="Review date (YYYY-MM-DD); defaults to today.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Output root holding results.jsonl and the notes file.",
    )
    return parser


def _note_entry(args: argparse.Namespace) -> dict[str, str]:
    """Build the note from the retained run, refusing when a key fact is missing."""
    run = _load_retained_run(ReportOutputPaths.from_root(args.output_dir).jsonl)
    image = run.metadata.get("image")
    image_sha256 = image.get("sha256") if image else None
    if not image_sha256:
        msg = "the retained run does not record its image digest"
        raise ValueError(msg)
    result = next((record for record in run.results if record["model"] == args.model), None)
    if result is None:
        msg = f"{args.model} is not in the retained run"
        raise ValueError(msg)
    provenance = result.get("model_provenance")
    revision = provenance.get("resolved_revision") if provenance else None
    if not revision:
        msg = f"the retained run does not record {args.model}'s resolved model revision"
        raise ValueError(msg)
    return {
        "model": args.model,
        "model_revision": revision,
        "image_sha256": image_sha256,
        "output_sha256": _sha256_text(result["generated_text"]),
        "reviewer": args.reviewer,
        "reviewed": args.reviewed.isoformat(),
        "note": args.note,
    }


def main(argv: list[str] | None = None) -> int:
    """Append one validated note and print it."""
    args = _build_parser().parse_args(argv)
    notes_path = args.output_dir.expanduser().resolve() / ACCURACY_NOTES_FILENAME
    try:
        entry = _note_entry(args)
        existing = read_text_no_follow(notes_path) if notes_path.exists() else _NEW_FILE_HEADER
        addition = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=88)
        updated = existing + ("" if existing.endswith("\n") else "\n") + addition
        # The same parser the reports use must accept the result before it is written.
        _parse_accuracy_notes(yaml.safe_load(updated))
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Not recorded: {error}", file=sys.stderr)
        return 1
    write_text_no_follow(notes_path, updated)
    print(f"Appended to {notes_path}:\n{addition}", end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
