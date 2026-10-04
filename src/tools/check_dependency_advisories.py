"""Gate a Skylos dependency-scan report against a short accepted-advisory list.

Skylos's `[tool.skylos] ignore` list does not apply to dependency (SCA)
findings, and its gate only counts them, so one advisory with no fixed
release cannot be accepted without accepting any other. `run_quality_checks.sh`
therefore runs the dependency scan on its own (`skylos . --sca --json`) and
this module decides:

- pass only a scan Skylos reports as finished with a recognised successful
  status (see SUCCESSFUL_STATUSES); every other status, including ones this
  module does not know, fails, so a scan that checked nothing never passes;
- pass an advisory only when it matches a reviewed exception exactly: the
  advisory, ecosystem, package, version and manifest, for every lockfile
  occurrence Skylos merged into the finding. The same advisory on another
  version or lockfile fails;
- fail an accepted advisory once a fixed release exists, so the exception is
  reconsidered (usually by upgrading) rather than outliving its reason;
- say when a listed exception is no longer reported, so it can be dropped.

Usage (from src/, the directory Skylos scanned):
    python -m tools.check_dependency_advisories <report.json>
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tools.safe_io import read_text_no_follow

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class AcceptedAdvisory:
    """One reviewed advisory occurrence; every field must match the finding."""

    rule_id: str
    ecosystem: str
    package: str
    version: str
    manifest: str  # relative to the scanned directory (src/)
    reason: str


# Each entry is temporary: it stops matching when the version moves, fails once
# a fix ships, and is reported for removal once Skylos stops finding it.
ACCEPTED_ADVISORIES: tuple[AcceptedAdvisory, ...] = (
    AcceptedAdvisory(
        rule_id="SKY-SCA-GHSA-vfj7-8cjw-p6xm",
        ecosystem="npm",
        package="braces",
        version="3.0.3",
        manifest="package-lock.json",
        reason=(
            "braces 3.0.3 (via markdownlint-cli2 -> micromatch) has no fixed release; "
            "braces only expands this repo's own markdownlint globs. Accepted 2026-10-03."
        ),
    ),
)

# Skylos sca_coverage statuses for a finished scan. "complete_with_unresolved_versions"
# is accepted deliberately: the Python dependencies in pyproject.toml are
# version ranges with no lockfile (the env is conda + pip), so Skylos cannot pin
# them, while package-lock.json still has to resolve fully (checked below).
SUCCESSFUL_STATUSES = frozenset({"complete", "complete_with_unresolved_versions"})
# Coverage counts that must be zero whatever the status says.
ZERO_COUNTS = ("parse_error_count", "unresolved_lockfile_dependency_count")
MAX_REPORT_BYTES = 64 * 1024 * 1024


def coverage_problems(coverage: object) -> tuple[list[str], list[str]]:
    """Return (blocking problems, notes) for Skylos's sca_coverage block."""
    if not isinstance(coverage, dict):
        return ["the report has no sca_coverage block"], []
    status = coverage.get("status")
    problems = []
    if status not in SUCCESSFUL_STATUSES:
        problems.append(
            f"status is {status or 'missing'}, not one of {sorted(SUCCESSFUL_STATUSES)}"
        )
    if coverage.get("complete") is not True:
        problems.append("Skylos did not mark the scan complete")
    query = coverage.get("query")
    if not isinstance(query, dict) or query.get("complete") is not True:
        problems.append("the advisory database query did not complete")
    if not isinstance(coverage.get("dependency_count"), int) or coverage["dependency_count"] < 1:
        problems.append("no dependencies were checked")
    problems.extend(
        f"{key} is {coverage.get(key)}"
        for key in ZERO_COUNTS
        if not isinstance(coverage.get(key), int) or coverage[key] != 0
    )
    notes = []
    unresolved = coverage.get("unresolved_dependency_count")
    if status == "complete_with_unresolved_versions":
        notes.append(
            f"• {unresolved} declared version ranges (pyproject.toml) could not be pinned; "
            "accepted by policy, the lockfile resolved fully."
        )
    return problems, notes


def _relative_manifest(file: object, scan_root: Path) -> str:
    path = Path(str(file or ""))
    resolved = (path if path.is_absolute() else scan_root / path).resolve()
    try:
        return resolved.relative_to(scan_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _matching_exception(
    entry: dict[str, object], accepted: tuple[AcceptedAdvisory, ...], scan_root: Path
) -> AcceptedAdvisory | None:
    metadata = entry.get("metadata")
    meta = metadata if isinstance(metadata, dict) else {}
    identity = (
        str(entry.get("rule_id", "")),
        str(meta.get("ecosystem", "")),
        str(meta.get("package_name", "")),
        str(meta.get("package_version", "")),
    )
    manifests = _finding_manifests(entry, meta, scan_root)
    for exception in accepted:
        if identity == (
            exception.rule_id,
            exception.ecosystem,
            exception.package,
            exception.version,
        ) and manifests == {exception.manifest}:
            return exception
    return None


def _finding_manifests(
    entry: Mapping[str, object], meta: Mapping[str, object], scan_root: Path
) -> set[str]:
    """Return every manifest the finding covers.

    Skylos merges one package version found in several lockfiles into one
    finding, keeping each location in metadata.dependency_occurrences, so the
    top-level file is only the first.
    """
    manifests = {_relative_manifest(entry.get("file"), scan_root)}
    occurrences = meta.get("dependency_occurrences", [])
    for occurrence in occurrences if isinstance(occurrences, list) else [None]:
        file = occurrence.get("file") if isinstance(occurrence, dict) else None
        # An occurrence with no readable location can never match an exception.
        manifests.add(_relative_manifest(file, scan_root) if file else "<unknown manifest>")
    return manifests


def _available_fixes(entry: dict[str, object]) -> list[str]:
    metadata = entry.get("metadata")
    meta = metadata if isinstance(metadata, dict) else {}
    fixes = meta.get("fixed_versions")
    candidates = [*(fixes if isinstance(fixes, list) else []), meta.get("fixed_version")]
    return list(dict.fromkeys(str(fix) for fix in candidates if fix))


def evaluate(
    report: object,
    accepted: tuple[AcceptedAdvisory, ...] = ACCEPTED_ADVISORIES,
    scan_root: Path | None = None,
) -> tuple[bool, list[str]]:
    """Return (passed, lines to print) for one dependency-scan report."""
    root = scan_root or Path.cwd()
    if not isinstance(report, dict):
        return False, ["❌ Dependency scan report is not a JSON object."]
    summary = report.get("analysis_summary")
    coverage = summary.get("sca_coverage") if isinstance(summary, dict) else None
    problems, lines = coverage_problems(coverage)
    if problems:
        return False, [
            "❌ Dependency scan did not finish cleanly; an unfinished scan never passes:",
            *(f"   - {problem}" for problem in problems),
        ]

    findings = report.get("dependency_vulnerabilities")
    if not isinstance(findings, list):
        return False, ["❌ Dependency scan report has no dependency_vulnerabilities list."]

    blocking: list[str] = []
    seen: set[AcceptedAdvisory] = set()
    for finding in findings:
        entry: dict[str, object] = finding if isinstance(finding, dict) else {}
        rule_id = str(entry.get("rule_id", "")) or "<no rule id>"
        label = f"{rule_id} ({entry.get('file', '?')}:{entry.get('line', '?')})"
        exception = _matching_exception(entry, accepted, root)
        if exception is None:
            blocking.append(f"❌ {label}: {entry.get('message', '')}")
            continue
        seen.add(exception)
        fixes = _available_fixes(entry)
        if fixes:
            blocking.append(
                f"❌ {label}: a fixed release now exists ({', '.join(fixes)}). "
                "Upgrade, or re-review and update the exception in ACCEPTED_ADVISORIES."
            )
        else:
            lines.append(f"• Accepted {label}: {exception.reason}")
    lines.extend(
        f"• {exception.rule_id} ({exception.package} {exception.version}) is no longer "
        "reported; drop it from ACCEPTED_ADVISORIES."
        for exception in accepted
        if exception not in seen
    )
    if blocking:
        return False, [*lines, *blocking]
    return True, [*lines, "✅ No unaccepted dependency advisories."]


def main(argv: list[str]) -> int:
    """Evaluate the report named by argv[1]; print the verdict."""
    report_path = Path(argv[1])
    try:
        report = json.loads(read_text_no_follow(report_path, max_bytes=MAX_REPORT_BYTES))
    except (OSError, ValueError) as error:
        print(f"❌ Could not read dependency scan report {report_path}: {error}")
        return 1
    passed, lines = evaluate(report)
    for line in lines:
        print(line)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
