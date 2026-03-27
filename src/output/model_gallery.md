# Model Output Gallery

<!-- markdownlint-disable MD013 -->

_Generated on 2026-03-27 12:45:34 GMT_

A review-friendly artifact with image metadata, the source prompt, and full generated output for each model.

## 🎯 Action Snapshot

- **Framework/runtime failures:** 1 (top owners: huggingface-hub=1).
- **Next action:** review failure ownership below and use diagnostics.md for filing.
- **Maintainer signals:** harness-risk successes=0, clean outputs=0/0.
- **Useful now:** none (no clean A/B shortlist for this run).
- **Review watchlist:** none.
- **Preflight compatibility:** 1 informational warning(s); do not treat these alone as run failures.
- **Escalate only if:** they line up with unexpected TF/Flax/JAX imports, startup hangs, or backend/runtime crashes.
- **Quality signal frequency:** none.
- **Runtime pattern:** model load dominates measured phase time (66%; 1/1 measured model(s)).
- **Phase totals:** model load=0.12s, cleanup=0.06s.
- **What this likely means:** Cold model load time is a major share of runtime for this cohort.
- **Suggested next action:** Consider staged runs, model reuse, or narrowing the model set before reruns.
- **Termination reasons:** exception=1.

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `huggingface-hub` | 1 | Model Error | `nonexistent/fake-model-12345` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### huggingface-hub

- **nonexistent/fake-model-12345** (Model Error)
  - Error: `Model loading failed: 404 Client Error. (Request ID: Root=1-69c67bee-700321dc17a5f36c0869e6af;a610f196-f985-40cd-a96f...`
  - Type: `ValueError`

## Image Metadata

- **Date**: 2026-03-27 12:45:34 GMT
- **Time**: 12:45:34

## Prompt

<!-- markdownlint-disable MD028 MD049 -->
>
> Analyze this image for cataloguing metadata, using British English.
>
> Use only details that are clearly and definitely visible in the image. If a detail is
> uncertain, ambiguous, partially obscured, too small to verify, or not directly visible,
> leave it out. Do not guess.
>
> Treat the metadata hints below as a draft catalog record. Keep only details that are
> clearly confirmed by the image, correct anything contradicted by the image, and add
> important visible details that are definitely present.
>
> Return exactly these three sections, and nothing else:
>
> Title:
> \- 5-10 words, concrete and factual, limited to clearly visible content.
> \- Output only the title text after the label.
> \- Do not repeat or paraphrase these instructions in the title.
>
> Description:
> \- 1-2 factual sentences describing the main visible subject, setting, lighting, action,
> and other distinctive visible details. Omit anything uncertain or inferred.
> \- Output only the description text after the label.
>
> Keywords:
> \- 10-18 unique comma-separated terms based only on clearly visible subjects, setting,
> colors, composition, and style. Omit uncertain tags rather than guessing.
> \- Output only the keyword list after the label.
>
> Rules:
> \- Include only details that are definitely visible in the image.
> \- Reuse metadata terms only when they are clearly supported by the image.
> \- If metadata and image disagree, follow the image.
> \- Prefer omission to speculation.
> \- Do not copy prompt instructions into the Title, Description, or Keywords fields.
> \- Do not infer identity, location, event, brand, species, time period, or intent unless
> visually obvious.
> \- Do not output reasoning, notes, hedging, or extra sections.
>
> Capture metadata hints: Taken on 2026-03-27 12:45:34 GMT (at 12:45:34 local time).
<!-- markdownlint-enable MD028 MD049 -->

## Quick Navigation

- **Failed models:** `nonexistent/fake-model-12345`

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD013 MD033 -->

<a id="model-nonexistent-fake-model-12345"></a>

### ❌ nonexistent/fake-model-12345

**Verdict:** harness | user=avoid
**Why:** model_error, huggingface_hub_model_load_model
**Trusted hints:** not evaluated
**Contract:** not evaluated
**Utility:** user=avoid
**Stack / owner:** owner=model | package=huggingface-hub | stage=Model Error | code=HUGGINGFACE_HUB_MODEL_LOAD_MODEL
**Token accounting:** prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a | max=500 | stop=exception
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Status:** Failed (Model Error)
**Error:**

> Model loading failed: 404 Client Error. (Request ID:
> Root=1-69c67bee-700321dc17a5f36c0869e6af;a610f196-f985-40cd-a96f-3f93db08bd16)
> Repository Not Found for url:
> <https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main.>
> Please make sure you specified the correct `repo_id` and `repo_type`. If you
> are trying to access a private or gated repo, make sure you are
> authenticated. For more details, see
> <https://huggingface.co/docs/huggingface_hub/authentication>
**Type:** `ValueError`
**Phase:** `model_load`
**Code:** `HUGGINGFACE_HUB_MODEL_LOAD_MODEL`
**Package:** `huggingface-hub`
**Next Action:** review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
    response.raise_for_status()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_models.py", line 829, in raise_for_status
    raise HTTPStatusError(message, request=request, response=self)
httpx.HTTPStatusError: Client error '404 Not Found' for url 'https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13700, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13132, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 392, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 134, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 321, in snapshot_download
    raise api_call_error
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 240, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3285, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3021, in model_info
    hf_raise_for_status(r)
    ~~~~~~~~~~~~~~~~~~~^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_http.py", line 847, in hf_raise_for_status
    raise repo_err from e
huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error. (Request ID: Root=1-69c67bee-700321dc17a5f36c0869e6af;a610f196-f985-40cd-a96f-3f93db08bd16)

Repository Not Found for url: https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13865, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13710, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: 404 Client Error. (Request ID: Root=1-69c67bee-700321dc17a5f36c0869e6af;a610f196-f985-40cd-a96f-3f93db08bd16)

Repository Not Found for url: https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
```

</details>

---

<!-- markdownlint-enable MD013 MD033 -->

<!-- markdownlint-enable MD013 -->
