"""Audit inline suppressions so stale or overly broad ones fail quality checks.

Scope: per-line ``# noqa``, ``# type: ignore``, ``# ty: ignore``,
``# skylos: ignore`` and ``# shellcheck disable`` comments, each re-checked to
prove it still earns its keep. Ruff, mypy, ty and ShellCheck rerun on a copy
without the suppression; one Skylos scan, limited to the rules the comments
name, reports which findings each ``skylos: ignore`` actually suppressed.
Configuration-wide suppressions (Ruff ``ignore`` /
``per-file-ignores``, Skylos ``ignore``, mypy overrides, markdownlint
rules) are deliberate policy in ``src/pyproject.toml``, ``.skylos/config.yaml``,
and ``.markdownlint.jsonc``; review those by hand when they change.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from tools.safe_io import read_text_no_follow
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.safe_io import read_text_no_follow

SuppressionKind = Literal[
    "noqa",
    "type-ignore",
    "shellcheck",
    "ty-ignore",
    "skylos-ignore",
    "bare-noqa",
    "bare-type-ignore",
    "bare-ty-ignore",
    "bare-skylos-ignore",
]
# Kinds whose checker prints each diagnostic's code in brackets.
BRACKETED_CODE_KINDS: Final[frozenset[SuppressionKind]] = frozenset(
    {"type-ignore", "ty-ignore", "skylos-ignore"}
)

PYTHON_SUFFIXES: Final[frozenset[str]] = frozenset({".py"})
SHELL_SUFFIXES: Final[frozenset[str]] = frozenset({".sh"})
EXCLUDED_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {
        ".claude",
        ".conda",
        ".worktrees",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "check_models.egg-info",
    },
)
EXCLUDED_PATH_PREFIXES: Final[tuple[Path, ...]] = (
    Path("src/build"),
    Path("src/output"),
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
TY_IGNORE_SPECIFIC_RE: Final[re.Pattern[str]] = re.compile(r"#\s*ty:\s*ignore\[([^\]]+)\]")
TY_IGNORE_BARE_RE: Final[re.Pattern[str]] = re.compile(r"#\s*ty:\s*ignore(?!\[)\b")
SKYLOS_IGNORE_SPECIFIC_RE: Final[re.Pattern[str]] = re.compile(
    r"#\s*skylos:\s*ignore\[([^\]]+)\]",
)
SKYLOS_IGNORE_BARE_RE: Final[re.Pattern[str]] = re.compile(r"#\s*skylos:\s*ignore(?!\[)\b")
# A line may carry several of these; each is its own finding.
_PYTHON_DIRECTIVES: Final[
    tuple[tuple[re.Pattern[str], re.Pattern[str], SuppressionKind, SuppressionKind], ...]
] = (
    (TY_IGNORE_SPECIFIC_RE, TY_IGNORE_BARE_RE, "ty-ignore", "bare-ty-ignore"),
    (SKYLOS_IGNORE_SPECIFIC_RE, SKYLOS_IGNORE_BARE_RE, "skylos-ignore", "bare-skylos-ignore"),
)
# Checkers a run may lack; their suppressions are skipped rather than failed.
_OPTIONAL_AUDIT_TOOLS: Final[dict[SuppressionKind, str]] = {
    "shellcheck": "shellcheck",
    "ty-ignore": "ty",
    "skylos-ignore": "skylos",
}


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

    for line_num, line in enumerate(read_text_no_follow(file_path).splitlines(), start=1):
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
    source_text: str = read_text_no_follow(file_path)
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

        for specific_re, bare_re, kind, bare_kind in _PYTHON_DIRECTIVES:
            specific_match = specific_re.search(comment_text)
            if specific_match:
                suppressions.append(
                    SuppressionFinding(
                        file_path=file_path,
                        line_num=line_num,
                        kind=kind,
                        codes=_split_codes(specific_match.group(1)),
                        line_text=line_text,
                    ),
                )
            elif bare_re.search(comment_text):
                suppressions.append(
                    SuppressionFinding(
                        file_path=file_path,
                        line_num=line_num,
                        kind=bare_kind,
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
    if kind == "ty-ignore":
        return re.sub(r"\s*#\s*ty:\s*ignore\[[^\]]+\]", "", line)
    if kind == "shellcheck":
        return ""
    return line


def _write_temp_variant(file_path: Path, line_num: int, kind: SuppressionKind) -> Path:
    """Write a temporary copy of a file with one suppression removed."""
    lines: list[str] = read_text_no_follow(file_path).splitlines(keepends=True)
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


def _python_tool(name: str) -> str | None:
    """Return a tool installed beside the running interpreter, else one on PATH."""
    beside_interpreter = Path(sys.executable).parent / name
    if beside_interpreter.is_file():
        return str(beside_interpreter)
    return shutil.which(name)


def _run_for_finding(
    finding: SuppressionFinding,
    *,
    repo_root: Path,
    src_root: Path,
) -> subprocess.CompletedProcess[str] | None:
    """Run the relevant linter on a temporary file without the suppression.

    Returns ``None`` when no runner exists for the kind or its tool is missing.
    """
    temp_path: Path = _write_temp_variant(finding.file_path, finding.line_num, finding.kind)
    command: list[str] | None = None
    cwd = src_root
    if finding.kind == "noqa":
        command = [sys.executable, "-m", "ruff", "check", str(temp_path)]
    elif finding.kind == "type-ignore":
        command = [
            sys.executable,
            "-m",
            "mypy",
            "--show-error-codes",
            "--hide-error-context",
            "--no-error-summary",
            str(temp_path),
        ]
    elif finding.kind == "ty-ignore" and (ty_path := _python_tool("ty")) is not None:
        command = [
            ty_path,
            "check",
            "--python",
            sys.executable,
            "--output-format",
            "concise",
            str(temp_path),
        ]
    elif finding.kind == "shellcheck" and (shellcheck_path := shutil.which("shellcheck")):
        command = [shellcheck_path, "-x", str(temp_path)]
        cwd = repo_root
    try:
        if command is None:
            return None
        return subprocess.run(  # noqa: S603 - fixed interpreter or resolved tool over a repo-local temp file
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            cwd=cwd,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _batch_noqa_audit_outputs(
    findings: Sequence[SuppressionFinding],
    *,
    src_root: Path,
) -> dict[tuple[Path, int], str]:
    """Audit every noqa suppression in one ruff run; return each finding's own output.

    One temp variant per finding, one ``ruff check`` over all of them (ruff
    parallelises internally), then the JSON diagnostics are regrouped per
    variant as ``path:line:col: CODE message`` lines, the shape the
    per-finding runner's concise output has. Replaces one interpreter start
    per suppression.
    """
    variants: dict[tuple[Path, int], Path] = {}
    try:
        for finding in findings:
            if finding.kind != "noqa" or not finding.codes:
                continue
            variants[finding.file_path, finding.line_num] = _write_temp_variant(
                finding.file_path, finding.line_num, finding.kind
            )
        if not variants:
            return {}
        completed = subprocess.run(  # noqa: S603 - fixed interpreter over repo-local temp files
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--output-format",
                "json",
                *(str(path) for path in variants.values()),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            cwd=src_root,
        )
    finally:
        for path in variants.values():
            path.unlink(missing_ok=True)
    lines_by_variant: dict[Path, list[str]] = {path.resolve(): [] for path in variants.values()}
    try:
        diagnostics = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        diagnostics = []
    for item in diagnostics if isinstance(diagnostics, list) else []:
        if not isinstance(item, dict):
            continue
        filename = Path(str(item.get("filename", ""))).resolve()
        if filename not in lines_by_variant:
            continue
        location = item.get("location") if isinstance(item.get("location"), dict) else {}
        lines_by_variant[filename].append(
            f"{filename}:{location.get('row', 0)}:{location.get('column', 0)}: "
            f"{item.get('code', '')} {item.get('message', '')}"
        )
    return {key: "\n".join(lines_by_variant[path.resolve()]) for key, path in variants.items()}


def _skylos_audit_outputs(
    findings: Sequence[SuppressionFinding],
    *,
    src_root: Path,
) -> dict[tuple[Path, int], str] | None:
    """Return, per ``skylos: ignore`` line, the rules Skylos says it suppressed there.

    One scan of ``src_root`` limited to the rules the comments name (about 45 s
    on this tree, against minutes for the full gate). Skylos's JSON lists every
    finding an inline ignore suppressed, so a comment absent from that list no
    longer suppresses anything. Each value holds bracketed rule IDs, matching
    the ``[CODE]`` check in ``check_if_needed``. Returns ``None`` when Skylos
    is not installed; raises ``RuntimeError`` when the scan leaves no report.
    """
    skylos_findings = [
        finding for finding in findings if finding.kind == "skylos-ignore" and finding.codes
    ]
    if not skylos_findings:
        return {}
    skylos_path = _python_tool("skylos")
    if skylos_path is None:
        return None
    rule_ids = sorted({code for finding in skylos_findings for code in finding.codes})
    # Skylos refuses to write through a symlinked parent, and macOS's temp
    # directory sits under one (/var -> /private/var), hence the resolve().
    with tempfile.TemporaryDirectory(prefix="suppression-audit.") as report_dir:
        report_path = Path(report_dir).resolve() / "skylos.json"
        completed = subprocess.run(  # noqa: S603 - resolved skylos binary scanning the package root
            [
                skylos_path,
                ".",
                "--select",
                ",".join(rule_ids),
                # json-ci keeps the "suppressed" list but omits the dead-code
                # definitions and evidence that make plain --json exceed the
                # 16 MiB safe-read cap on this tree.
                "--format",
                "json-ci",
                "-o",
                str(report_path),
                "--no-upload",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            cwd=src_root,
        )
        try:
            report = json.loads(read_text_no_follow(report_path))
        except (OSError, json.JSONDecodeError) as error:
            detail = (completed.stderr or completed.stdout).strip()[-500:]
            msg = f"Skylos suppression scan left no readable report ({error}): {detail}"
            raise RuntimeError(msg) from error

    suppressed: dict[tuple[Path, int], list[str]] = {}
    entries = report.get("suppressed", []) if isinstance(report, dict) else []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not entry.get("rule_id") or not entry.get("file"):
            continue
        file_path = Path(str(entry["file"]))
        if not file_path.is_absolute():
            file_path = src_root / file_path
        key = (file_path.resolve(), int(entry.get("line", 0)))
        suppressed.setdefault(key, []).append(f"[{entry['rule_id']}]")
    return {
        (finding.file_path, finding.line_num): " ".join(
            suppressed.get((finding.file_path.resolve(), finding.line_num), [])
        )
        for finding in skylos_findings
    }


def _batched_output_for(
    finding: SuppressionFinding,
    outputs: dict[tuple[Path, int], str],
    skylos_outputs: dict[tuple[Path, int], str] | None = None,
) -> str | None:
    """Return the batched output for a noqa or skylos finding; other kinds run their own checker.

    Both batches are keyed by file and line, and a line may carry a noqa, a
    type: ignore and a skylos: ignore at once, so the kind guard keeps each
    checker's output with its own suppressions.
    """
    if finding.kind == "noqa":
        return outputs.get((finding.file_path, finding.line_num))
    if finding.kind == "skylos-ignore" and skylos_outputs is not None:
        return skylos_outputs.get((finding.file_path, finding.line_num))
    return None


def _suppression_rationale(finding: SuppressionFinding) -> str:
    """Return the human justification text that follows the suppression codes."""
    if finding.kind == "noqa":
        match = NOQA_SPECIFIC_RE.search(finding.line_text)
    elif finding.kind == "type-ignore":
        match = TYPE_IGNORE_SPECIFIC_RE.search(finding.line_text)
    elif finding.kind == "shellcheck":
        match = SHELLCHECK_RE.search(finding.line_text)
    elif finding.kind == "ty-ignore":
        match = TY_IGNORE_SPECIFIC_RE.search(finding.line_text)
    elif finding.kind == "skylos-ignore":
        match = SKYLOS_IGNORE_SPECIFIC_RE.search(finding.line_text)
    else:
        return ""
    if match is None:
        return ""
    rationale = finding.line_text[match.end() :].strip().lstrip("#- ").strip()
    if rationale or finding.kind not in {"ty-ignore", "skylos-ignore"}:
        return rationale
    # A ty or Skylos ignore beside a justified noqa on the same line shares
    # that justification (one long signature, two checkers).
    noqa_match = NOQA_SPECIFIC_RE.search(finding.line_text)
    if noqa_match is None:
        return ""
    return finding.line_text[noqa_match.end() :].split("#", 1)[0].strip().lstrip("- ").strip()


def check_if_needed(
    finding: SuppressionFinding,
    *,
    repo_root: Path,
    src_root: Path,
    audit_output: str | None = None,
) -> tuple[bool, str]:
    """Check whether one suppression is still required.

    ``audit_output`` is this finding's slice of a batched linter run
    (``_batch_noqa_audit_outputs``); without it the linter runs for this
    finding alone.
    """
    invalid_reason: str | None = None
    if finding.kind == "bare-noqa":
        invalid_reason = "Bare # noqa is not allowed; use explicit codes"
    elif finding.kind == "bare-type-ignore":
        invalid_reason = "Bare # type: ignore is not allowed; use explicit error codes"
    elif finding.kind == "bare-ty-ignore":
        invalid_reason = "Bare # ty: ignore is not allowed; use explicit rule names"
    elif finding.kind == "bare-skylos-ignore":
        invalid_reason = "Bare # skylos: ignore is not allowed; use explicit rule IDs"
    elif not finding.codes:
        invalid_reason = "Suppression has no explicit codes"
    elif not _suppression_rationale(finding):
        # A suppression must say why it is safe, not just what it silences.
        invalid_reason = (
            "Suppression has no justification; append ' - <why>' after the codes "
            "(for type: ignore, use a second '# <why>' comment)"
        )
    if invalid_reason is not None:
        return False, invalid_reason

    if audit_output is not None:
        output: str = audit_output
    else:
        # Skylos findings only come from the batched scan, which yields
        # nothing for them when skylos is not installed.
        result = (
            None
            if finding.kind == "skylos-ignore"
            else _run_for_finding(finding, repo_root=repo_root, src_root=src_root)
        )
        if result is None:
            tool = _OPTIONAL_AUDIT_TOOLS.get(finding.kind)
            if tool is not None:
                return True, f"Skipped {tool} audit because {tool} is unavailable"
            return False, "No audit runner available for suppression"
        output = f"{result.stdout}\n{result.stderr}"
    missing_codes = tuple(
        code
        for code in finding.codes
        if (f"[{code}]" if finding.kind in BRACKETED_CODE_KINDS else code) not in output
    )
    if missing_codes:
        return False, f"No violation found for: {', '.join(missing_codes)}"
    return True, f"Suppression needed: {', '.join(finding.codes)} violations found"


def main() -> int:
    """Run repository-wide suppression audit."""
    src_root: Path = Path(__file__).resolve().parents[1]
    repo_root: Path = src_root.parent

    print(
        "Auditing inline suppressions (noqa / type: ignore / ty: ignore / skylos: ignore / "
        f"shellcheck) in: {repo_root}\n"
    )
    print("=" * 80)

    findings: list[SuppressionFinding] = iter_repo_suppressions(repo_root)
    if not findings:
        print("No suppressions found!")
        return 0

    print(f"Found {len(findings)} inline suppression(s) across the repository\n")

    necessary: list[tuple[SuppressionFinding, str]] = []
    unnecessary: list[tuple[SuppressionFinding, str]] = []
    batched_outputs = _batch_noqa_audit_outputs(findings, src_root=src_root)
    try:
        skylos_outputs = _skylos_audit_outputs(findings, src_root=src_root)
    except RuntimeError as error:
        print(f"✗ FAIL: {error}")
        return 1

    for finding in findings:
        relative_path: Path = finding.file_path.relative_to(repo_root)
        code_display: str = ", ".join(finding.codes) if finding.codes else finding.kind
        print(f"\n{relative_path}:{finding.line_num}: {code_display}")
        print(f"  {finding.line_text[:100]}...")

        needed, reason = check_if_needed(
            finding,
            repo_root=repo_root,
            src_root=src_root,
            audit_output=_batched_output_for(finding, batched_outputs, skylos_outputs),
        )
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
