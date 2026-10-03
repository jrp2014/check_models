"""Gate a Skylos dependency-scan report against a short accepted-advisory list.

Skylos's `[tool.skylos] ignore` list does not apply to dependency (SCA)
findings, and its gate only counts them, so one advisory with no fixed
release cannot be accepted without accepting any other. `run_quality_checks.sh`
therefore runs the dependency scan on its own (`skylos . --sca --json`) and
this module decides:

- fail when the scan did not finish (Skylos's own incomplete statuses), so an
  unreachable advisory database can never pass the step;
- fail on every advisory not listed in ACCEPTED_ADVISORIES;
- pass, naming each accepted advisory still reported, and say when a listed
  advisory is no longer reported so its entry can be dropped.

Usage (from src/):
    python -m tools.check_dependency_advisories <report.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.safe_io import read_text_no_follow

# rule_id -> why it is accepted. Each entry is temporary: drop it as soon as a
# fixed release reaches the lockfile (the step reports when that happens).
ACCEPTED_ADVISORIES: dict[str, str] = {
    "SKY-SCA-GHSA-vfj7-8cjw-p6xm": (
        "braces 3.0.3 (via markdownlint-cli2 -> micromatch) has no fixed release; "
        "braces only expands this repo's own markdownlint globs. Accepted 2026-10-03."
    ),
}
# Skylos's gate treats these sca_coverage statuses as an interrupted scan.
INCOMPLETE_STATUSES = frozenset({"incomplete", "unavailable", "unknown"})
MAX_REPORT_BYTES = 64 * 1024 * 1024


def evaluate(
    report: object, accepted: dict[str, str] = ACCEPTED_ADVISORIES
) -> tuple[bool, list[str]]:
    """Return (passed, lines to print) for one dependency-scan report."""
    if not isinstance(report, dict):
        return False, ["❌ Dependency scan report is not a JSON object."]
    summary = report.get("analysis_summary")
    coverage = summary.get("sca_coverage") if isinstance(summary, dict) else None
    status = coverage.get("status") if isinstance(coverage, dict) else None
    if status is None or status in INCOMPLETE_STATUSES:
        reason = f"❌ Dependency scan did not finish (status: {status or 'missing'})."
        return False, [reason, "   An unfinished scan never passes."]

    findings = report.get("dependency_vulnerabilities")
    if not isinstance(findings, list):
        return False, ["❌ Dependency scan report has no dependency_vulnerabilities list."]

    lines: list[str] = []
    blocking: list[str] = []
    seen_accepted: set[str] = set()
    for finding in findings:
        entry = finding if isinstance(finding, dict) else {}
        rule_id = str(entry.get("rule_id", "")) or "<no rule id>"
        location = f"{entry.get('file', '?')}:{entry.get('line', '?')}"
        if rule_id in accepted:
            seen_accepted.add(rule_id)
            lines.append(f"• Accepted {rule_id} ({location}): {accepted[rule_id]}")
        else:
            blocking.append(f"❌ {rule_id} ({location}): {entry.get('message', '')}")
    lines.extend(
        f"• {rule_id} is no longer reported; drop it from ACCEPTED_ADVISORIES."
        for rule_id in sorted(set(accepted) - seen_accepted)
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
