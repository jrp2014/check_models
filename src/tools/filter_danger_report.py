"""Drop third-party `.worktrees/` findings from a Skylos danger report.

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


def main(argv: list[str]) -> int:
    """Filter the report file named by argv[1]; print the drop count."""
    report_path = Path(argv[1])
    report = json.loads(read_text_no_follow(report_path, max_bytes=MAX_REPORT_BYTES))
    if not isinstance(report, dict):
        print(0)
        return 0
    dropped = drop_worktree_findings(report)
    if dropped:
        write_text_no_follow(report_path, json.dumps(report))
    print(dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
