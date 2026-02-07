# Agent Instructions

All project conventions, architecture, environment setup, coding standards,
and change workflows are maintained in a single canonical file:

**Read [.github/copilot-instructions.md](.github/copilot-instructions.md) before making any changes.**

Key reminders:

- Always use `conda activate mlx-vlm` before running Python
- Run `make quality` to validate changes (ruff + mypy + ty + pyrefly + pytest + markdownlint)
- `src/check_models.py` is an intentional single-file monolith — do not split it
- Add tests to existing `src/tests/test_*.py` files, never create standalone scripts
