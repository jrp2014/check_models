# Agent Instructions

All project conventions, architecture, environment setup, coding standards,
and change workflows are maintained in a single canonical file:

**Read [.github/copilot-instructions.md](.github/copilot-instructions.md) before making any changes.**

Key reminders:

- Always use `conda activate mlx-vlm` before running Python
- Before `make quality`, run `make format`, clear Ruff lint issues with
  `make -C src lint-fix` / `make lint`, then run the full quality gate
- `src/check_models.py` is an intentional single-file monolith — do not split it
- Add tests to existing `src/tests/test_*.py` files, never create standalone scripts
- Validation tests must not rewrite tracked `src/output/` assets; send generated
  files to a temp directory or gitignored `test_*` output paths
- Keep `CHANGELOG.md` (`[Unreleased]`) up to date for maintainer-relevant changes, including refactors and tooling updates
