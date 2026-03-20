"""Audit repo suppressions so stale or overly broad suppressions fail quality checks."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

SuppressionKind = Literal[
    "noqa",
    "type-ignore",
    "shellcheck",
    "bare-noqa",
    "bare-type-ignore",
]

PYTHON_SUFFIXES: Final[frozenset[str]] = frozenset({".py"})
SHELL_SUFFIXES: Final[frozenset[str]] = frozenset({".sh"})
EXCLUDED_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {
        ".conda",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "check_models.egg-info",
        "typings",
    },
)
EXCLUDED_PATH_PREFIXES: Final[tuple[Path, ...]] = (
    Path("src/output"),
    Path("src/tools/.archived"),
    Path("docs/notes/archive"),
)


@dataclass(frozen=True)
class SuppressionFinding:
    """One concrete suppression comment found in the repository."""

    file_path: Path
    line_num: int
    kind: SuppressionKind
    codes: tuple[str, ...]
    line_text: str


NOQA_SPECIFIC_RE: Final[re.Pattern[str]] = re.compile(r"#\s*noqa:\s*([A-Z0-9,\s]+)")
NOQA_BARE_RE: Final[re.Pattern[str]] = re.compile(r"#\s*noqa(?!:)\b")
TYPE_IGNORE_SPECIFIC_RE: Final[re.Pattern[str]] = re.compile(
    r"#\s*type:\s*ignore\[([^\]]+)\]",
)
TYPE_IGNORE_BARE_RE: Final[re.Pattern[str]] = re.compile(r"#\s*type:\s*ignore(?!\[)\b")
SHELLCHECK_RE: Final[re.Pattern[str]] = re.compile(r"#\s*shellcheck\s+disable=([A-Z0-9,\s]+)")


def _split_codes(raw_codes: str) -> tuple[str, ...]:
    """Normalize comma/space-separated suppression codes."""
    return tuple(code.strip() for code in raw_codes.replace(",", " ").split() if code.strip())


def should_audit_path(file_path: Path, repo_root: Path) -> bool:
    """Return whether a repository file should be included in suppression auditing."""
    if not file_path.is_file():
        return False

    try:
        relative_path = file_path.relative_to(repo_root)
    except ValueError:
        return False

    if any(part in EXCLUDED_PATH_PARTS for part in relative_path.parts):
        return False
    if any(
        relative_path == prefix or relative_path.is_relative_to(prefix)
        for prefix in EXCLUDED_PATH_PREFIXES
    ):
        return False
    return file_path.suffix in PYTHON_SUFFIXES | SHELL_SUFFIXES


def iter_audited_files(repo_root: Path) -> list[Path]:
    """Return sorted repo files eligible for suppression auditing."""
    return sorted(
        candidate for candidate in repo_root.rglob("*") if should_audit_path(candidate, repo_root)
    )


def find_suppressions(file_path: Path) -> list[SuppressionFinding]:
    """Find supported suppression annotations in one file."""
    suppressions: list[SuppressionFinding] = []
    if file_path.suffix in PYTHON_SUFFIXES:
        return _find_python_suppressions(file_path)

    with file_path.open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped_line: str = line.rstrip()

            shellcheck_match = SHELLCHECK_RE.search(line)
            if shellcheck_match:
                suppressions.append(
                    SuppressionFinding(
                        file_path=file_path,
                        line_num=line_num,
                        kind="shellcheck",
                        codes=_split_codes(shellcheck_match.group(1)),
                        line_text=stripped_line,
                    ),
                )

    return suppressions


def _find_python_suppressions(file_path: Path) -> list[SuppressionFinding]:
    """Find Python suppression comments using tokenization so strings are ignored."""
    suppressions: list[SuppressionFinding] = []
    source_text: str = file_path.read_text(encoding="utf-8")
    lines: list[str] = source_text.splitlines()

    for token in tokenize.generate_tokens(io.StringIO(source_text).readline):
        if token.type != tokenize.COMMENT:
            continue

        line_num: int = token.start[0]
        comment_text: str = token.string
        line_text: str = lines[line_num - 1] if line_num - 1 < len(lines) else comment_text

        noqa_match = NOQA_SPECIFIC_RE.search(comment_text)
        if noqa_match:
            suppressions.append(
                SuppressionFinding(
                    file_path=file_path,
                    line_num=line_num,
                    kind="noqa",
                    codes=_split_codes(noqa_match.group(1)),
                    line_text=line_text,
                ),
            )
        elif NOQA_BARE_RE.search(comment_text):
            suppressions.append(
                SuppressionFinding(
                    file_path=file_path,
                    line_num=line_num,
                    kind="bare-noqa",
                    codes=(),
                    line_text=line_text,
                ),
            )

        type_ignore_match = TYPE_IGNORE_SPECIFIC_RE.search(comment_text)
        if type_ignore_match:
            suppressions.append(
                SuppressionFinding(
                    file_path=file_path,
                    line_num=line_num,
                    kind="type-ignore",
                    codes=_split_codes(type_ignore_match.group(1)),
                    line_text=line_text,
                ),
            )
        elif TYPE_IGNORE_BARE_RE.search(comment_text):
            suppressions.append(
                SuppressionFinding(
                    file_path=file_path,
                    line_num=line_num,
                    kind="bare-type-ignore",
                    codes=(),
                    line_text=line_text,
                ),
            )

    return suppressions


def iter_repo_suppressions(repo_root: Path) -> list[SuppressionFinding]:
    """Return all auditable suppressions across the repository."""
    findings: list[SuppressionFinding] = []
    for file_path in iter_audited_files(repo_root):
        findings.extend(find_suppressions(file_path))
    return findings


def _remove_suppression_from_line(line: str, kind: SuppressionKind) -> str:
    """Return a line with one suppression directive removed."""
    if kind == "noqa":
        return re.sub(r"\s*#\s*noqa:[^#\n]*", "", line)
    if kind == "type-ignore":
        return re.sub(r"\s*#\s*type:\s*ignore\[[^\]]+\]", "", line)
    if kind == "shellcheck":
        return ""
    return line


def _write_temp_variant(file_path: Path, line_num: int, kind: SuppressionKind) -> Path:
    """Write a temporary copy of a file with one suppression removed."""
    lines: list[str] = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    original_line: str = lines[line_num - 1]
    modified_line: str = _remove_suppression_from_line(original_line, kind)
    if modified_line == original_line:
        msg = f"Could not remove suppression from {file_path}:{line_num}"
        raise ValueError(msg)

    lines[line_num - 1] = modified_line
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=file_path.suffix,
        prefix=f"{file_path.stem}.suppression-audit.",
        dir=file_path.parent,
        delete=False,
    ) as handle:
        handle.writelines(lines)
        return Path(handle.name)


def _run_for_finding(
    finding: SuppressionFinding,
    *,
    repo_root: Path,
    src_root: Path,
) -> subprocess.CompletedProcess[str] | None:
    """Run the relevant linter on a temporary file without the suppression."""
    temp_path: Path = _write_temp_variant(finding.file_path, finding.line_num, finding.kind)
    try:
        if finding.kind == "noqa":
            return subprocess.run(
                [sys.executable, "-m", "ruff", "check", str(temp_path)],
                capture_output=True,
                text=True,
                check=False,
                cwd=src_root,
            )
        if finding.kind == "type-ignore":
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    "--show-error-codes",
                    "--hide-error-context",
                    "--no-error-summary",
                    str(temp_path),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=src_root,
            )
        if finding.kind == "shellcheck":
            if shutil.which("shellcheck") is None:
                return None
            return subprocess.run(
                ["shellcheck", "-x", str(temp_path)],
                capture_output=True,
                text=True,
                check=False,
                cwd=repo_root,
            )
        return None
    finally:
        temp_path.unlink(missing_ok=True)


def check_if_needed(
    finding: SuppressionFinding,
    *,
    repo_root: Path,
    src_root: Path,
) -> tuple[bool, str]:
    """Check whether one suppression is still required."""
    invalid_reason: str | None = None
    if finding.kind == "bare-noqa":
        invalid_reason = "Bare # noqa is not allowed; use explicit codes"
    elif finding.kind == "bare-type-ignore":
        invalid_reason = "Bare # type: ignore is not allowed; use explicit error codes"
    elif not finding.codes:
        invalid_reason = "Suppression has no explicit codes"
    if invalid_reason is not None:
        return False, invalid_reason

    result = _run_for_finding(finding, repo_root=repo_root, src_root=src_root)
    if result is None and finding.kind == "shellcheck":
        return True, "Skipped shellcheck audit because shellcheck is unavailable"
    if result is None:
        return False, "No audit runner available for suppression"

    output: str = f"{result.stdout}\n{result.stderr}"
    for code in finding.codes:
        code_marker: str = f"[{code}]" if finding.kind == "type-ignore" else code
        if code_marker in output:
            return True, f"Suppression needed: {code} violation found"
    return False, "No violations found - suppression appears stale"


def main() -> int:
    """Run repository-wide suppression audit."""
    src_root: Path = Path(__file__).resolve().parents[1]
    repo_root: Path = src_root.parent

    print(f"Auditing suppressions in: {repo_root}\n")
    print("=" * 80)

    findings: list[SuppressionFinding] = iter_repo_suppressions(repo_root)
    if not findings:
        print("No suppressions found!")
        return 0

    print(f"Found {len(findings)} suppression(s) across the repository\n")

    necessary: list[tuple[SuppressionFinding, str]] = []
    unnecessary: list[tuple[SuppressionFinding, str]] = []

    for finding in findings:
        relative_path: Path = finding.file_path.relative_to(repo_root)
        code_display: str = ", ".join(finding.codes) if finding.codes else finding.kind
        print(f"\n{relative_path}:{finding.line_num}: {code_display}")
        print(f"  {finding.line_text[:100]}...")

        needed, reason = check_if_needed(finding, repo_root=repo_root, src_root=src_root)
        if needed:
            print(f"  ✓ NEEDED: {reason}")
            necessary.append((finding, reason))
        else:
            print(f"  ✗ FAIL: {reason}")
            unnecessary.append((finding, reason))

    print("\n" + "=" * 80)
    print("\nSummary:")
    print(f"  Necessary:   {len(necessary)}")
    print(f"  Failures:    {len(unnecessary)}")

    if unnecessary:
        print("\n  Suppressions that should be removed or narrowed:")
        for finding, reason in unnecessary:
            relative_path = finding.file_path.relative_to(repo_root)
            print(f"    {relative_path}:{finding.line_num} - {reason}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
