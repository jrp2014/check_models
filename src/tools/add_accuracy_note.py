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
from typing import TYPE_CHECKING

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_models import (
    ACCURACY_NOTES_FILENAME,
    JsonlResultRecord,
    ReportOutputPaths,
    RetainedRun,
    _load_retained_run,
    _parse_accuracy_notes,
    _replace_text_file,
    _sha256_text,
    regenerate_run_issue_summary,
)
from tools.safe_io import read_text_no_follow

if TYPE_CHECKING:
    from collections.abc import Callable

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
_NEW_FILE_HEADER = """\
# Human accuracy notes on the outputs retained in this directory.
# Each note applies only while its image, model revision and output digests
# match a run; the gallery and HTML report then show it beside that output.
# Add one with: python -m tools.add_accuracy_note MODEL --reviewer NAME --note TEXT
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


def retained_model_result(
    output_dir: Path, model: str
) -> tuple[RetainedRun, JsonlResultRecord, str]:
    """Load the retained run and one model's record with its resolved revision."""
    run = _load_retained_run(ReportOutputPaths.from_root(output_dir).jsonl)
    result = next((record for record in run.results if record["model"] == model), None)
    if result is None:
        msg = f"{model} is not in the retained run"
        raise ValueError(msg)
    provenance = result.get("model_provenance")
    revision = provenance.get("resolved_revision") if provenance is not None else None
    if not revision:
        msg = f"the retained run does not record {model}'s resolved model revision"
        raise ValueError(msg)
    return run, result, revision


def append_note(
    notes_path: Path,
    header: str,
    entry: dict[str, str],
    parse: Callable[[object], tuple[object, ...]],
) -> str:
    """Add one entry to the notes document and replace the file atomically.

    The existing file is parsed, not appended to as text, so any layout the
    reports accept (an empty file, ``notes: []``, an indented list) works.
    Its leading comment block is kept; a failed write leaves it untouched.
    """
    existing = (
        read_text_no_follow(notes_path) if notes_path.exists() or notes_path.is_symlink() else ""
    )
    document = yaml.safe_load(existing) if existing else None
    parse(document)
    entries = list(document.get("notes") or []) if isinstance(document, dict) else []
    entries.append(entry)
    comment_lines = []
    for line in existing.splitlines(keepends=True):
        if not line.startswith("#"):
            break
        comment_lines.append(line)
    comments = "".join(comment_lines) or header
    updated = comments + yaml.safe_dump(
        {"notes": entries}, sort_keys=False, allow_unicode=True, width=88
    )
    parse(yaml.safe_load(updated))
    _replace_text_file(notes_path, updated)
    return yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=88)


def publish_run_summary(output_dir: Path) -> None:
    """Re-render run_summary.md from the retained run so the note shows without a sweep."""
    try:
        summary = regenerate_run_issue_summary(output_dir.expanduser().resolve())
    except (OSError, ValueError) as error:
        print(f"Recorded, but run_summary.md was not refreshed: {error}", file=sys.stderr)
        return
    if summary is not None:
        print(f"Refreshed {summary}")


def _note_entry(args: argparse.Namespace) -> dict[str, str]:
    """Build the note from the retained run, refusing when a key fact is missing."""
    run, result, revision = retained_model_result(args.output_dir, args.model)
    image = run.metadata.get("image")
    image_sha256 = image.get("sha256") if image else None
    if not image_sha256:
        msg = "the retained run does not record its image digest"
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
        addition = append_note(
            notes_path, _NEW_FILE_HEADER, _note_entry(args), _parse_accuracy_notes
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Not recorded: {error}", file=sys.stderr)
        return 1
    print(f"Appended to {notes_path}:\n{addition}", end="")
    publish_run_summary(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
