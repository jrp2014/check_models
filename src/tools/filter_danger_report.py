"""Drop waived findings from a Skylos danger report before the gate.

Two waiver classes, both visible (drop counts are printed, never silent):

1. Third-party `.worktrees/` checkouts (see below).
2. Shell-file findings whose flagged line carries the same inline
   ``# skylos: ignore[RULE-ID] - why`` convention Skylos itself honours for
   Python. Skylos 4.33.x's shell analyzer does not parse inline ignores, so
   without this the convention silently stops working the moment a finding
   lands in a ``.sh`` file.

Skylos 4.33.x intermittently ignores both `--exclude .worktrees` and the
`.skylos/config.yaml` exclude for its workflow danger scanner, so upstream
checkouts under `.worktrees/` leak their files into this repository's gate.
`run_skylos_danger_advisory.sh` runs this filter on the JSON report before
annotate/gate so the exclusion holds regardless of upstream flag behavior.

Usage (from src/):
    python -m tools.filter_danger_report <absolute-report-path.json>

Prints the number of dropped findings on stdout (consumed by the shell
script to emit a visible drop notice — the exclusion is never silent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.safe_io import read_text_no_follow, write_text_no_follow

WORKTREE_MARKERS = ("/.worktrees/", "/.claude/worktrees/")
# Danger reports for this repo run ~10 MB; cap well above that.
MAX_REPORT_BYTES = 64 * 1024 * 1024


def drop_worktree_findings(report: dict[str, object]) -> int:
    """Remove danger findings under `.worktrees/` in place; return drop count."""
    findings = report.get("danger")
    if not isinstance(findings, list):
        return 0
    kept = [
        finding
        for finding in findings
        if not (
            isinstance(finding, dict)
            and any(marker in str(finding.get("file", "")) for marker in WORKTREE_MARKERS)
        )
    ]
    dropped = len(findings) - len(kept)
    if dropped:
        report["danger"] = kept
    return dropped


def drop_inline_ignored_shell_findings(report: dict[str, object]) -> int:
    """Drop shell findings whose flagged line carries a matching inline ignore."""
    findings = report.get("danger")
    if not isinstance(findings, list):
        return 0
    kept: list[object] = []
    dropped = 0
    file_lines: dict[str, list[str]] = {}
    for finding in findings:
        if not isinstance(finding, dict) or not str(finding.get("file", "")).endswith(".sh"):
            kept.append(finding)
            continue
        file_name = str(finding["file"])
        rule_id = str(finding.get("rule_id", ""))
        line_number = finding.get("line")
        if file_name not in file_lines:
            try:
                file_lines[file_name] = read_text_no_follow(Path(file_name)).splitlines()
            except (OSError, ValueError):
                file_lines[file_name] = []
        lines = file_lines[file_name]
        flagged = (
            lines[line_number - 1]
            if isinstance(line_number, int) and 1 <= line_number <= len(lines)
            else ""
        )
        if rule_id and f"skylos: ignore[{rule_id}]" in flagged:
            dropped += 1
        else:
            kept.append(finding)
    if dropped:
        report["danger"] = kept
    return dropped


def main(argv: list[str]) -> int:
    """Filter the report file named by argv[1]; print the drop count."""
    report_path = Path(argv[1])
    report = json.loads(read_text_no_follow(report_path, max_bytes=MAX_REPORT_BYTES))
    if not isinstance(report, dict):
        print(0)
        return 0
    dropped = drop_worktree_findings(report)
    dropped += drop_inline_ignored_shell_findings(report)
    if dropped:
        write_text_no_follow(report_path, json.dumps(report))
    print(dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
