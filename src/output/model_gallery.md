# Model Output Gallery

_Generated on 2026-03-13 12:57:46 GMT_

A review-friendly artifact with image metadata, the source prompt, and full generated output for each model.

## Image Metadata

- **Date**: 2026-03-13 12:57:46 GMT
- **Time**: 12:57:46

## Prompt

```text
Analyze this image for cataloguing metadata.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Capture metadata hints: Taken on 2026-03-13 12:57:46 GMT (at 12:57:46 local time).
```

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD013 MD033 -->

### ❌ nonexistent/fake-model-12345

**Status:** Failed (Model Error)
**Error:**

> Model loading failed: 404 Client Error. (Request ID:
> Root=1-69b409ca-2e6305e7423ef8d46949f36f;ee040e8a-fb2f-432b-bd7b-37b40b6e70ab)
> Repository Not Found for url:
> <https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main.>
> Please make sure you specified the correct `repo\_id` and `repo\_type`. If
> you are trying to access a private or gated repo, make sure you are
> authenticated. For more details, see
> <https://huggingface.co/docs/huggingface_hub/authentication>
**Type:** `ValueError`
**Phase:** `model_load`
**Code:** `MLX_VLM_MODEL_LOAD_MODEL`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>


```python
Traceback (most recent call last):
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
    response.raise_for_status()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_models.py", line 829, in raise_for_status
    raise HTTPStatusError(message, request=request, response=self)
httpx.HTTPStatusError: Client error '404 Not Found' for url 'https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 11397, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 11042, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 393, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 135, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 321, in snapshot_download
    raise api_call_error
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 240, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3146, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 2940, in model_info
    hf_raise_for_status(r)
    ~~~~~~~~~~~~~~~~~~~^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_http.py", line 847, in hf_raise_for_status
    raise repo_err from e
huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error. (Request ID: Root=1-69b409ca-2e6305e7423ef8d46949f36f;ee040e8a-fb2f-432b-bd7b-37b40b6e70ab)

Repository Not Found for url: https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 11572, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 11407, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: 404 Client Error. (Request ID: Root=1-69b409ca-2e6305e7423ef8d46949f36f;ee040e8a-fb2f-432b-bd7b-37b40b6e70ab)

Repository Not Found for url: https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
```

</details>


---

<!-- markdownlint-enable MD013 MD033 -->
