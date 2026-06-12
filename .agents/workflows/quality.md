---
description: Format, lint-fix, then run the project's full quality suite
---
Always run this workflow after editing code in the `check_models` project, or when asked to validate the codebase.

1. Ensure we are using the correct conda environment
// turbo

```bash
conda run -n mlx-vlm make format
conda run -n mlx-vlm make -C src lint-fix
conda run -n mlx-vlm make lint
conda run -n mlx-vlm make quality
```
