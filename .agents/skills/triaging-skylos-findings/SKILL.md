---
name: triaging-skylos-findings
description: >
  Triages a Skylos finding that fails `make quality` or CI here (quality,
  secrets, ai-defect, danger, dependency or dead-code): reproduce it narrowly,
  decide fix versus suppression, and spot Skylos bugs. Use when a Skylos gate
  fails or a Skylos upgrade surfaces new findings.
---

# Triaging Skylos Findings

Adapted from Skylos's own `skylos` and `skylos-security` skills
([`.claude/skills`](https://github.com/duriantaco/skylos/tree/main/.claude/skills),
read at `7a597ad`). Those target work on the Skylos repository itself; this
keeps only what a project that runs Skylos as a gate needs. Commands below
were checked against Skylos 4.43.2.

## Where Skylos runs here

| Step | Command (from `src/tools/`) | Blocks on |
| ---- | --------------------------- | --------- |
| Skylos Quality Gate | `run_quality_checks.sh`: `. --quality --secrets --ai-defects --gate --format concise` | any finding (`max_*` are 0 in `[gate]`) |
| Skylos Dependency Scan | `run_quality_checks.sh`: `. --sca --json`, then `tools.check_dependency_advisories` | any advisory not on that tool's accepted list |
| Skylos Danger Gate | `run_skylos_danger_advisory.sh --full --gate` (fast mode drops `--gate`) | any danger finding outside `.worktrees/` |
| `skylos-advisory` CI job | `run_skylos_danger_advisory.sh`, diff-aware on PRs | nothing (annotations only) |

All of them go through `quality_run_skylos` in `common_quality.sh`, which
shims the clipboard and sets `SKYLOS_GREP_BUDGET`. Configuration lives twice
and must stay mirrored: `.skylos/config.yaml` (repo-root scans) and
`[tool.skylos]` in `src/pyproject.toml` (package-local scans).

## 1. Reproduce one finding narrowly

Run from the repository root in the `mlx-vlm` env, so Skylos sees the same
installed packages as the gate. Always pass `--no-upload`; keep reports under
the gitignored `.skylos/` or a temp directory, never elsewhere in `src/`.

```bash
# The one rule, as JSON; --select turns on that rule's analyzer family
skylos . --select SKY-C303 --file-filter check_models.py \
  --format json --no-upload </dev/null

# Only what your uncommitted work changed (or --diff origin/main for a branch)
skylos . --quality --secrets --ai-defects --diff HEAD --format concise --no-upload
```

- Findings sit in arrays by family (`danger`, `secrets`, `quality`,
  `unused_functions`, ...), each entry with `rule_id`, `severity`, `file` and
  `line`. Read them with `.get(key, [])`: empty arrays may be omitted.
- `suppressed` lists every inline-ignored finding with its rule and line. It
  ignores `--select` and `--file-filter`, so it doubles as an audit of all
  `# skylos: ignore[...]` comments.
- `--format llm` (or `make skylos-danger-llm`) adds code context;
  `make skylos-verify ARGS='--file F --range L1:L2'` checks one edited range.

## 2. Decide: fix, suppress, or report

Classify first: true positive, false positive, or Skylos bug (wrong
inventory, ignored exclude, a version change in a detector). Base the call on
the code path from input to sink, not on the finding's message.

Then take the first option that holds:

1. **Fix the code.** For file I/O findings (SKY-D215/D324/D325) that means
   `read_text_no_follow`/`write_text_no_follow` from `tools/safe_io.py`, or
   `_open_regular_file_no_symlink` in `check_models.py`, as
   `.github/copilot-instructions.md` requires.
2. **Dead-code findings:** rule them out in this order before deleting or
   suppressing: static references and imports, pytest fixtures and hooks,
   console-script and `Makefile` entry points, `Protocol`/`TypedDict` members,
   names reached dynamically (`getattr`, registries). Absence from a text
   search is not proof of dead code. Cross-check with `make vulture`.
3. **Inline suppression** on the reported line, naming the rule and the
   reason: `# skylos: ignore[SKY-D215] path is canonicalized and opened with
   O_NOFOLLOW.` The suppression audit (`tools/check_suppressions.py`) does
   not re-check Skylos comments, so a stale one stays silently; if it works
   around a Skylos bug, name the Skylos version and record when to remove it
   (as the SKY-D223 comments and CHANGELOG entry do).
4. **Project-wide `ignore`** only for a rule the project rejects as policy,
   with a one-line reason, in both config files. Never raise a `[gate]`
   threshold instead: `max_quality = 0` is deliberate.

For danger and secret findings, suppress only with proof that the value is
safe: a trusted literal, an immutable allowlist, a scheme check, a no-follow
open with a size cap. An uppercase name, a comment, a `safe_`/`validated_`
helper name, or "nothing mutates it in this file" is not proof.

## 3. After a Skylos upgrade

New findings after `pip install -U skylos` are usually detector changes, not
new defects. Run the gate on the old and new versions against the same
commit to confirm, then triage each new rule as above. Also re-test the
version-pinned workarounds: delete each suppression or filter that names an
older Skylos version, rerun its narrow command, and drop it if the finding no
longer appears (current ones: SKY-D223 for 4.43.2, SKY-S101 for 4.33.2, the
`.worktrees/` danger filter for 4.33.x).

## 4. Drafting a Skylos bug report

Draft only; do not file upstream unless asked.

1. Build the smallest fixture that shows the wrong result in a temp
   directory (`git init` it if the bug involves `--diff`), never under `src/`.
2. Show the finding with `--select RULE --format json` and the exact
   `skylos --version`.
3. Say where the result goes wrong: discovery, rule matching, config
   `ignore`/`exclude`, diff or baseline filtering, dependency inventory, or
   the gate's exit code. Cite the Skylos source file and function when found.
4. Include the expected result and the workaround this repository carries.

## Do not

- Run Skylos modes that execute or rewrite code here: `--trace`,
  `--coverage --allow-coverage-execution`, `skylos clean --apply`,
  `--interactive`, `skylos agent remediate`, or `skylos agent install-hooks`.
- Upload scans (`--upload`, `skylos sync`) or call Skylos Cloud.
- Add a bare `# skylos: ignore` without a rule ID.
