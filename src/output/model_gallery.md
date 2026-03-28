# Model Output Gallery

_Generated on 2026-03-27 23:37:41 GMT_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: huggingface-hub=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=0, clean outputs=0/0.
- _Useful now:_ none (no clean A/B shortlist for this run).
- _Review watchlist:_ none.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Quality signal frequency:_ none.
- _Runtime pattern:_ model load dominates measured phase time (58%; 1/1
  measured model(s)).
- _Phase totals:_ model load=0.11s, cleanup=0.08s.
- _What this likely means:_ Cold model load time is a major share of runtime
  for this cohort.
- _Suggested next action:_ Consider staged runs, model reuse, or narrowing the
  model set before reruns.
- _Termination reasons:_ exception=1.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `huggingface-hub` | 1 | Model Error | `nonexistent/fake-model-12345` |

### Actionable Items by Package

#### huggingface-hub

- nonexistent/fake-model-12345 (Model Error)
  - Error: `Model loading failed: 404 Client Error. (Request ID: Root=1-69c714c5-3b4202c357f65ff16c065b4e;2e0a4b68-2020-4d52-a824...`
  - Type: `ValueError`

## Image Metadata

- _Date:_ 2026-03-27 23:37:41 GMT
- _Time:_ 23:37:41

## Prompt

<!-- markdownlint-disable MD028 -->
>
> Analyze this image for cataloguing metadata, using British English.
>
> Use only details that are clearly and definitely visible in the image. If a
> detail is uncertain, ambiguous, partially obscured, too small to verify, or
> not directly visible, leave it out. Do not guess.
>
> Treat the metadata hints below as a draft catalog record. Keep only details
> that are clearly confirmed by the image, correct anything contradicted by
> the image, and add important visible details that are definitely present.
>
> Return exactly these three sections, and nothing else:
>
> Title:
> \- 5-10 words, concrete and factual, limited to clearly visible content.
> \- Output only the title text after the label.
> \- Do not repeat or paraphrase these instructions in the title.
>
> Description:
> \- 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> \- Output only the description text after the label.
>
> Keywords:
> \- 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> \- Output only the keyword list after the label.
>
> Rules:
> \- Include only details that are definitely visible in the image.
> \- Reuse metadata terms only when they are clearly supported by the image.
> \- If metadata and image disagree, follow the image.
> \- Prefer omission to speculation.
> \- Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> \- Do not infer identity, location, event, brand, species, time period, or
> intent unless visually obvious.
> \- Do not output reasoning, notes, hedging, or extra sections.
>
> Capture metadata hints: Taken on 2026-03-27 23:37:41 GMT (at 23:37:41 local
> time).
<!-- markdownlint-enable MD028 -->

## Quick Navigation

- _Failed models:_ `nonexistent/fake-model-12345`

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 -->

<a id="model-nonexistent-fake-model-12345"></a>

### ❌ nonexistent/fake-model-12345

_Verdict:_ harness | user=avoid
_Why:_ model_error, huggingface_hub_model_load_model
_Trusted hints:_ not evaluated
_Contract:_ not evaluated
_Utility:_ user=avoid
_Stack / owner:_ owner=model | package=huggingface-hub | stage=Model Error |
                 code=HUGGINGFACE_HUB_MODEL_LOAD_MODEL
_Token accounting:_ prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Status:_ Failed (Model Error)
_Error:_

> Model loading failed: 404 Client Error. (Request ID:
> Root=1-69c714c5-3b4202c357f65ff16c065b4e;2e0a4b68-2020-4d52-a824-d218dfaac1b1)
> Repository Not Found for url:
> <https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main.>
> Please make sure you specified the correct `repo_id` and `repo_type`. If you
> are trying to access a private or gated repo, make sure you are
> authenticated. For more details, see
> <https://huggingface.co/docs/huggingface_hub/authentication>
_Type:_ `ValueError`
_Phase:_ `model_load`
_Code:_ `HUGGINGFACE_HUB_MODEL_LOAD_MODEL`
_Package:_ `huggingface-hub`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13847, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13279, in _load_model
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
huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error. (Request ID: Root=1-69c714c5-3b4202c357f65ff16c065b4e;2e0a4b68-2020-4d52-a824-d218dfaac1b1)

Repository Not Found for url: https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14012, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13857, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: 404 Client Error. (Request ID: Root=1-69c714c5-3b4202c357f65ff16c065b4e;2e0a4b68-2020-4d52-a824-d218dfaac1b1)

Repository Not Found for url: https://huggingface.co/api/models/nonexistent/fake-model-12345/revision/main.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
```

</details>

---

<!-- markdownlint-enable MD033 -->
