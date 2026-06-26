---
description: Format, lint-fix, then run the project's full quality suite
---
Always run this workflow after editing code in the `check_models` project, or when asked to validate the codebase.
`make quality` already runs the full pytest suite, so do not run `make test`
again after this workflow unless specifically investigating a test-only failure.
Use `make skylos-danger` for advisory workflow-security checks and
`make skylos-danger-llm` when an agent wants the same Skylos findings in its
LLM-oriented format.
Use `make skylos-verify ARGS='--file path/to/file --range L1:L2'` for the
smallest post-edit AI-defect verification pass before running the full quality
workflow.

1. Ensure we are using the correct conda environment
// turbo

```bash
conda run -n mlx-vlm make format
conda run -n mlx-vlm make -C src lint-fix
conda run -n mlx-vlm make lint
conda run -n mlx-vlm make quality
```
