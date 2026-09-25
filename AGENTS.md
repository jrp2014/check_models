# Agent Instructions

Entry point for Codex (including GPT-6 Astra), Claude Code, and other agents.
Use the shared guidance below; there is no separate model-specific rulebook.

All project conventions, architecture, environment setup, coding standards,
and change workflows are maintained in a single canonical file:

**Read [.github/copilot-instructions.md](.github/copilot-instructions.md) before making any changes.**

Key reminders:

- Always use `conda activate mlx-vlm` before running Python
- Before `make quality`, run `make format`, clear Ruff lint issues with
  `make -C src lint-fix` / `make lint`, then run the full quality gate
- `src/check_models.py` is an intentional single-file monolith — do not split it
- Add tests to existing `src/tests/test_*.py` files, never create standalone scripts
- Validation tests must not rewrite tracked `src/output/` assets or leave
  generated files anywhere under `src/`; send generated output to a
  temp directory (`tmp_path`). Ordinary gitignored tool caches
  (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `.skylos`)
  are fine
- Keep `CHANGELOG.md` (`[Unreleased]`) up to date for maintainer-relevant changes, including refactors and tooling updates
- For upstream mlx-vlm isolation/issues/cache discovery/fixes, type-checker
  failures and performance evidence, load the matching skill from
  `.agents/skills/` (also linked as `.claude/skills`) before starting:
  Claude Code invokes it with the Skill tool rather than reading the file;
  other agents read its `SKILL.md`. Use conda + pip only (never `uv`)
