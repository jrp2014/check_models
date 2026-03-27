---
description: Run the project's quality suite (formatting, linting, tests, shell, markdown)
---
Always run this workflow after editing code in the `check_models` project, or when asked to validate the codebase.

1. Ensure we are using the correct conda environment
// turbo

```bash
conda run -n mlx-vlm make quality
```
