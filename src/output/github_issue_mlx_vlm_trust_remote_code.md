# Bug: `trust_remote_code` incorrectly passed to `snapshot_download()`

## Summary

`mlx-vlm` passes `trust_remote_code` via `**kwargs` to `huggingface_hub.snapshot_download()`, which does not accept this parameter. This causes a `TypeError` when loading models with `trust_remote_code=True`.

## Error Message

```text
TypeError: snapshot_download() got an unexpected keyword argument 'trust_remote_code'
```

## Environment

- **mlx-vlm version**: 0.3.10
- **huggingface_hub version**: 1.3.2 (latest as of Jan 2026)
- **Python**: 3.13.9
- **macOS**: 26.2 (Apple Silicon)

## Reproduction

```python
from mlx_vlm import load

# This fails with TypeError
model, processor = load(
    "mlx-community/nanoLLaVA-1.5-4bit",
    trust_remote_code=True
)
```

## Root Cause

In `mlx_vlm/utils.py`, the `get_model_path()` function passes `**kwargs` directly to `snapshot_download()`:

```python
# mlx_vlm/utils.py, around line 102
model_path = Path(
    snapshot_download(
        repo_id=path_or_hf_repo,
        revision=revision,
        allow_patterns=[...],
        **kwargs,  # <-- trust_remote_code gets passed here
    )
)
```

However, `huggingface_hub.snapshot_download()` does not have a `trust_remote_code` parameter. The `trust_remote_code` parameter is for `AutoConfig.from_pretrained()`, `AutoModel.from_pretrained()`, etc., but NOT for `snapshot_download()`.

## Verification

Checking `snapshot_download`'s signature confirms it doesn't accept `trust_remote_code`:

```python
import inspect
import huggingface_hub

sig = inspect.signature(huggingface_hub.snapshot_download)
print("trust_remote_code" in sig.parameters)  # False
```

## Suggested Fix

Filter out `trust_remote_code` before passing kwargs to `snapshot_download()`:

```python
def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None, **kwargs) -> Path:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        # Filter out kwargs not accepted by snapshot_download
        snapshot_kwargs = {k: v for k, v in kwargs.items() if k != "trust_remote_code"}
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[...],
                **snapshot_kwargs,
            )
        )
    return model_path
```

Alternatively, explicitly list the kwargs that should be passed through, rather than using `**kwargs` blindly.

## Workaround

Users can work around this by monkey-patching `snapshot_download` to strip the unsupported argument:

```python
from functools import partial
from unittest.mock import patch
import huggingface_hub
from mlx_vlm import load

def strip_trust_remote_code(original_fn, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)
    return original_fn(*args, **kwargs)

original = huggingface_hub.snapshot_download
patched = partial(strip_trust_remote_code, original)

with patch.object(huggingface_hub, "snapshot_download", patched):
    model, processor = load("mlx-community/nanoLLaVA-1.5-4bit", trust_remote_code=True)
```

## Related

- PR #660 added `--trust-remote-code` CLI flag but did not address this underlying bug
- This affects all `load()` calls with `trust_remote_code=True` when the model needs to be downloaded

## Impact

- **Severity**: High - blocks model loading with `trust_remote_code=True`
- **Affected versions**: 0.3.10 (and likely earlier versions that pass kwargs through)
- **Affected platforms**: All (not platform-specific)
