---
name: hf-cli
description: >-
  Use the installed Hugging Face Hub CLI for repository/file inspection,
  authentication troubleshooting, cache commands, and explicitly requested
  downloads or Hub changes. Use for hf command syntax and operation, not
  general ML questions or harness model-selection decisions.
---

# Hugging Face CLI

Use the existing `hf` in the `mlx-vlm` conda environment. Follow the
[canonical project guidance](../../../.github/copilot-instructions.md);
do not bootstrap a second CLI with an installer, `hf update`, or uv.
A missing CLI or dependency upgrade is a setup change requiring authorization.

## Find the right command

Start with `hf version` and the relevant `hf <command> --help` after
`conda activate mlx-vlm`. Installed help takes precedence over a remembered
command or generated reference.

For command discovery only, search the relevant heading in the
[generated command catalogue](references/commands.md), such as `hf cache`,
`hf models`, or `hf repos`. Do not load the whole catalogue for a known
command. It lists capabilities, not permission to execute them.

- Cache warnings belong to the subcommand: `hf cache list --show-warnings`.
- Check output formats in that command's help. The 2.2.0 generator's prose
  incorrectly suggests `--format table`; installed help instead lists
  `auto|human|agent|json|quiet`.
- For why the harness selects or skips a model, use
  [hf-cache-mlx-vlm-models](../hf-cache-mlx-vlm-models/SKILL.md).
  A Hub/cache listing is not evidence that a model can generate.
- For authentication diagnosis, use `hf auth whoami` when needed; do not
  print access tokens or put them into commands, logs, or reports.

## Keep operations within scope

Inspection does not authorize downloads, cache removal/pruning, credential
changes, dependency updates, paid compute, uploads, or publishing. Obtain
authorization for the exact mutation unless the user already supplied it.
For authorized changes, resolve the repo, revision, paths, and destination;
preview the operation with its supported dry-run option when available.
A preview is not authorization. Read-only remote queries may still need
network access; stop at permission boundaries.

## Maintaining the reference

The entrypoint is maintained locally. Do not run `hf skills add --force`
here: it would overwrite it with the broad upstream skill.

To refresh the catalogue, run `hf skills preview` in the active conda
environment (local generation, no installation). Replace only the command
section in `references/commands.md`, from `## Commands` up to but not
including `## Common options`, and update its version provenance. Preserve
the reference's introductory boundary. Exclude upstream frontmatter,
installation advice, and generic tips; verify commands used in this
repository against their installed help.
