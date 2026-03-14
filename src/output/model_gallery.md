# Model Output Gallery

_Generated on 2026-03-14 19:39:08 GMT_

A review-friendly artifact with image metadata, the source prompt, and full generated output for each model.

## 🎯 Action Snapshot

- **Framework/runtime failures:** 1 (top owners: mlx-vlm=1).
- **Next action:** review failure ownership below and use diagnostics.md for filing.
- **Maintainer signals:** harness-risk successes=0, clean outputs=0/0.
- **Useful now:** none (no clean A/B shortlist for this run).
- **Review watchlist:** none.
- **Quality signal frequency:** none.
- **Runtime pattern:** model load dominates measured phase time (99%; 1/1 measured model(s)).
- **Phase totals:** model load=10.50s, cleanup=0.07s.
- **What this likely means:** Cold model load time is a major share of runtime for this cohort.
- **Suggested next action:** Consider staged runs, model reuse, or narrowing the model set before reruns.
- **Termination reasons:** exception=1.

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `mlx-vlm` | 1 | Model Error | `nonexistent/fake-model-12345` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### mlx-vlm

- **nonexistent/fake-model-12345** (Model Error)
  - Error: `Model loading failed: An error happened while trying to locate the files on the Hub and we cannot find the appropriat...`
  - Type: `ValueError`

## Image Metadata

- **Date**: 2026-03-14 19:38:58 GMT
- **Time**: 19:38:58

## Prompt

<!-- markdownlint-disable MD028 MD049 -->
>
> Analyze this image for cataloguing metadata.
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
> Capture metadata hints: Taken on 2026-03-14 19:38:58 GMT (at 19:38:58 local time).
<!-- markdownlint-enable MD028 MD049 -->

## Quick Navigation

- **Failed models:** `nonexistent/fake-model-12345`

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD013 MD033 -->

<a id="model-nonexistent-fake-model-12345"></a>

### ❌ nonexistent/fake-model-12345

**Status:** Failed (Model Error)
**Error:**

> Model loading failed: An error happened while trying to locate the files on
> the Hub and we cannot find the appropriate snapshot folder for the specified
> revision on the local disk. Please check your internet connection and try
> again.
**Type:** `ValueError`
**Phase:** `model_load`
**Code:** `MLX_VLM_MODEL_LOAD_MODEL`
**Package:** `mlx-vlm`
**Next Action:** review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>


```python
Traceback (most recent call last):
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
        pool_request.request
    )
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpcore/_sync/connection.py", line 101, in handle_request
    raise exc
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpcore/_sync/connection.py", line 78, in handle_request
    stream = self._connect(request)
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpcore/_sync/connection.py", line 124, in _connect
    stream = self._network_backend.connect_tcp(**kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpcore/_backends/sync.py", line 207, in connect_tcp
    with map_exceptions(exc_map):
         ~~~~~~~~~~~~~~^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpcore/_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ConnectError: [Errno 8] nodename nor servname provided, or not known

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 240, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3095, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 2888, in model_info
    r = get_session().get(path, headers=headers, timeout=timeout, params=params)
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpx/_client.py", line 1053, in get
    return self.request(
           ~~~~~~~~~~~~^
        "GET",
        ^^^^^^
    ...<7 lines>...
        extensions=extensions,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpx/_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
        request,
    ...<2 lines>...
        history=[],
    )
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
        request,
        follow_redirects=follow_redirects,
        history=history,
    )
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ConnectError: [Errno 8] nodename nor servname provided, or not known

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 11990, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 11635, in _load_model
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
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 324, in snapshot_download
    raise LocalEntryNotFoundError(
    ...<3 lines>...
    ) from api_call_error
huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the files on the Hub and we cannot find the appropriate snapshot folder for the specified revision on the local disk. Please check your internet connection and try again.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 12165, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 12000, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: An error happened while trying to locate the files on the Hub and we cannot find the appropriate snapshot folder for the specified revision on the local disk. Please check your internet connection and try again.
```

</details>


---

<!-- markdownlint-enable MD013 MD033 -->
