#!/usr/bin/env python3
"""Audit all linting suppressions to see if they're still needed."""

import re
import subprocess
import sys
from pathlib import Path


def find_suppressions(file_path: Path) -> list[tuple[int, str, str]]:
    """Find all suppression annotations in a file.

    Returns list of (line_number, suppression_code, full_line).
    """
    suppressions = []
    with file_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            # Match noqa comments
            noqa_match = re.search(r"#\s*noqa:\s*([A-Z0-9,\s]+)", line)
            if noqa_match:
                codes = noqa_match.group(1).strip()
                suppressions.append((line_num, codes, line.rstrip()))

            # Match type: ignore comments
            type_ignore_match = re.search(r"#\s*type:\s*ignore\[([^\]]+)\]", line)
            if type_ignore_match:
                codes = type_ignore_match.group(1).strip()
                suppressions.append((line_num, f"type:ignore[{codes}]", line.rstrip()))

    return suppressions


def check_if_needed(file_path: Path, line_num: int, codes: str) -> tuple[bool, str]:
    """Check if a suppression is actually needed by testing without it.

    Returns (is_needed, reason).
    """
    # Read file
    with file_path.open(encoding="utf-8") as f:
        lines = f.readlines()

    # Remove the suppression temporarily
    original_line = lines[line_num - 1]
    # Remove noqa comment
    modified_line = re.sub(r"\s*#\s*noqa:[^#\n]*", "", original_line)
    # Remove type: ignore comment
    modified_line = re.sub(r"\s*#\s*type:\s*ignore\[[^\]]+\]", "", modified_line)

    if modified_line.strip() == original_line.strip():
        return True, "Could not remove suppression (format not recognized)"

    lines[line_num - 1] = modified_line

    # Write temporary file
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)

    try:
        # Run ruff check on the specific line
        result = subprocess.run(  # noqa: S603
            ["ruff", "check", str(temp_path)],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )

        # Check if the suppressed codes appear in output
        for raw_code in codes.replace(",", " ").split():
            suppressed_code = raw_code.strip()
            if suppressed_code in result.stdout or suppressed_code in result.stderr:
                return True, f"Suppression needed: {suppressed_code} violation found"

        # No violations found
        return False, "No violations found - suppression may be unnecessary"

    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)


def main() -> int:
    """Run suppression audit."""
    src_dir = Path(__file__).parent.parent
    check_models = src_dir / "check_models.py"

    if not check_models.exists():
        print(f"Error: {check_models} not found", file=sys.stderr)  # noqa: T201
        return 1

    print(f"Auditing suppressions in: {check_models}\n")  # noqa: T201
    print("=" * 80)  # noqa: T201

    suppressions = find_suppressions(check_models)

    if not suppressions:
        print("No suppressions found!")  # noqa: T201
        return 0

    print(f"Found {len(suppressions)} suppression(s)\n")  # noqa: T201

    unnecessary = []
    necessary = []

    for line_num, codes, line_text in suppressions:
        print(f"\nLine {line_num}: {codes}")  # noqa: T201
        print(f"  {line_text[:100]}...")  # noqa: T201

        needed, reason = check_if_needed(check_models, line_num, codes)

        if needed:
            print(f"  ✓ NEEDED: {reason}")  # noqa: T201
            necessary.append((line_num, codes, reason))
        else:
            print(f"  ✗ UNNECESSARY: {reason}")  # noqa: T201
            unnecessary.append((line_num, codes, reason))

    print("\n" + "=" * 80)  # noqa: T201
    print("\nSummary:")  # noqa: T201
    print(f"  Necessary:   {len(necessary)}")  # noqa: T201
    print(f"  Unnecessary: {len(unnecessary)}")  # noqa: T201

    if unnecessary:
        print("\n  Lines with potentially unnecessary suppressions:")  # noqa: T201
        for line_num, codes, reason in unnecessary:
            print(f"    Line {line_num}: {codes} - {reason}")  # noqa: T201

    return 0
if __name__ == "__main__":
    sys.exit(main())
