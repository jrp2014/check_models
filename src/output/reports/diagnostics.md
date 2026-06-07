<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 2 failure(s), 6 harness issue(s) (mlx-vlm 0.6.2)

**Run summary:** 59 locally-cached VLM model(s) checked; 2 hard failure(s), 6 harness/integration issue(s), 0 preflight warning(s), 57 successful run(s).

Test image: `cats.jpg` (0.2 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                                                                                                               | Evidence Snapshot                                                                                                                                                                                                                         | Affected Models                                            | Issue Draft                                                                                                                                          | Evidence Bundle                                                                                                                                                                               | Fixed When                                                |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `huggingface_hub`                                        | Hugging Face Hub: Model load / model error: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook- | Model Error \| phase model_load \| FileNotFoundError                                                                                                                                                                                      | 1: `facebook/pe-av-large`                                  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_huggingface-hub_huggingface-hub-model-load-model_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260607T203551Z_003_facebook_pe-av-large_HUGGINGFACE_HUB_MODEL_LOAD_MODEL_d6788a6.json)              | Load/generation completes or fails with a narrower owner. |
| `mlx`                                                    | Weight/config mismatch during model load                                                                                              | Model Error \| phase model_load \| ValueError                                                                                                                                                                                             | 1: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx_mlx-model-load-model_001.md)                         | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260607T203551Z_002_LiquidAI_LFM2.5-VL-450M-MLX-bf16_MLX_MODEL_LOAD_MODEL_853049863f38.json)         | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers                                                                                              | 76 BPE space markers found in decoded text \| prompt=417 \| output/prompt=21.58% \| nontext burden=99% \| stop=completed                                                                                                                  | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_encoding_001.md)                                 | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260607T203551Z_006_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json) | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text                                                                                        | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=770 \| output/prompt=25.97% \| nontext burden=99% \| stop=max_tokens \| hit token cap (200) \| 2 model cluster | 2: `microsoft/Phi-3.5-vision-instruct` (+1)                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_stop-token_001.md)                               | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260607T203551Z_004_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)                | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                                                                                                 | output/prompt=1.1% \| prompt=1,196 \| output/prompt=1.09% \| nontext burden=99% \| stop=completed \| 2 model cluster                                                                                                                      | 2: `HuggingFaceTB/SmolVLM-Instruct` (+1)                   | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_model-config-mlx-vlm_prompt-template_001.md)             | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260607T203551Z_001_HuggingFaceTB_SmolVLM-Instruct_model_config_mlx_vlm_prompt_template_001.json) | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short                                                                                 | generated_tokens~3 \| prompt_tokens=4103, output_tokens=3, output/prompt=0.1% \| prompt=4,103 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed                                                                             | 1: `mlx-community/paligemma2-3b-pt-896-4bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm-mlx_long-context_001.md)                         | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260607T203551Z_008_mlx-community_paligemma2-3b-pt-896-4bit_mlx_vlm_mlx_long_context_001.json)       | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

---

## 1. Failure affecting 1 model

**Affected models:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

**Observed error:**

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18196, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17585, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 895, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 756, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 632, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 2 parameters not in model: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18405, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18206, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 2 parameters not in model: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

```

<!-- markdownlint-disable MD060 -->

| Model                              | Observed Behavior                                                                                                                         | First Seen Failing      | Recent Repro           |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` | Model loading failed: Received 2 parameters not in model: multi_modal_projector.layer_norm.bias, multi_modal_projector.layer_norm.weight. | 2026-05-04 20:21:24 BST | 2/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

## 2. Failure affecting 1 model

**Affected models:** `facebook/pe-av-large`

**Observed error:**

```text
Traceback (most recent call last):
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/shutil.py", line 856, in move
    os.rename(src, real_dst)
    ~~~~~~~~~^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete' -> '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/file_download.py", line 1949, in _copy_no_matter_what
    shutil.copy2(src, dst)
    ~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/shutil.py", line 468, in copy2
    copyfile(src, dst, follow_symlinks=follow_symlinks)
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/shutil.py", line 260, in copyfile
    with open(src, 'rb') as fsrc:
         ~~~~^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18196, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17585, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 895, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 753, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 416, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 456, in snapshot_download
    thread_map(
    ~~~~~~~~~~^
        _inner_hf_hub_download,
        ^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        tqdm_class=tqdm_class,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/tqdm/contrib/concurrent.py", line 69, in thread_map
    return _executor_map(ThreadPoolExecutor, fn, *iterables, **tqdm_kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/tqdm/contrib/concurrent.py", line 51, in _executor_map
    return list(tqdm_class(ex.map(fn, *iterables, chunksize=chunksize), **kwargs))
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
               ^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/concurrent/futures/_base.py", line 619, in result_iterator
    yield _result_or_cancel(fs.pop())
          ~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/concurrent/futures/_base.py", line 317, in _result_or_cancel
    return fut.result(timeout)
           ~~~~~~~~~~^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/concurrent/futures/_base.py", line 456, in result
    return self.__get_result()
           ~~~~~~~~~~~~~~~~~^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/concurrent/futures/thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 436, in _inner_hf_hub_download
    hf_hub_download(  # type: ignore
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
        repo_id,
        ^^^^^^^^
    ...<14 lines>...
        dry_run=dry_run,
        ^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/file_download.py", line 1019, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
        # Destination
    ...<15 lines>...
        dry_run=dry_run,
    )
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/file_download.py", line 1241, in _hf_hub_download_to_cache_dir
    _download_to_tmp_and_move(
    ~~~~~~~~~~~~~~~~~~~~~~~~~^
        incomplete_path=Path(blob_path + ".incomplete"),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<8 lines>...
        tqdm_class=tqdm_class,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/file_download.py", line 1889, in _download_to_tmp_and_move
    _chmod_and_move(tmp_path, destination_path)
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/file_download.py", line 1938, in _chmod_and_move
    shutil.move(str(src), str(dst), copy_function=_copy_no_matter_what)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/shutil.py", line 876, in move
    copy_function(src, real_dst)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/file_download.py", line 1952, in _copy_no_matter_what
    shutil.copyfile(src, dst)
    ~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/shutil.py", line 260, in copyfile
    with open(src, 'rb') as fsrc:
         ~~~~^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18405, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18206, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete'

```

<!-- markdownlint-disable MD060 -->

| Model                  | Observed Behavior                                                                                                                                                                                                       | First Seen Failing      | Recent Repro           |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `facebook/pe-av-large` | Model loading failed: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete' | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `facebook/pe-av-large`

---

## Harness/Integration Issues (6 model(s))

6 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `HuggingFaceTB/SmolVLM-Instruct`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (1.1%), suggesting possible early-stop or prompt-handling issues.

**Sample output:**

```text
Two cats are sleeping on a pink blanket on a couch.
```

### `microsoft/Phi-3.5-vision-instruct`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
Two cats are sleeping on a pink couch with remote controls beside them.<|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|...
```

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).
- Output leaked reasoning or prompt-template text (here are my reasoning steps, the user asks:).

**Sample output:**

```text
Here are my reasoning steps:
The user asks: "Describe this image briefly." The image is of two cats sleeping on a pink couch, with a remote control. The user wants a brief description. This is a strai...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 76 occurrences).

**Sample output:**

```text
TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠblanketĠorĠaĠcouch.ĠTheĠcatĠonĠtheĠleftĠisĠaĠkitten,ĠandĠtheĠoneĠonĠtheĠrightĠisĠanĠadultĠcat.ĠBothĠcatsĠareĠinĠrelaxedĠpostures,ĠwithĠt...
```

### `mlx-community/SmolVLM-Instruct-bf16`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (1.1%), suggesting possible early-stop or prompt-handling issues.

**Sample output:**

```text
Two cats are sleeping on a pink blanket on a couch.
```

### `mlx-community/paligemma2-3b-pt-896-4bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.
- At long prompt length (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%).

**Sample output:**

```text
Cat.
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none
**Generation regressions in history window:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`, `mlx-community/gemma-3n-E2B-4bit`, `mlx-community/pixtral-12b-8bit`, `mlx-community/pixtral-12b-bf16`
**Quality regressions in history window:** `HuggingFaceTB/SmolVLM-Instruct`, `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `microsoft/Phi-3.5-vision-instruct`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, `mlx-community/SmolVLM-Instruct-bf16`, `mlx-community/pixtral-12b-8bit`, `mlx-community/pixtral-12b-bf16`
**Core library changes in window:** `mlx=0.32.0.dev20260517+7b7c1240->0.32.0.dev20260607+8f0e8b14`, `mlx-vlm=0.5.0->0.6.2`, `transformers=5.8.1->5.10.2`

<!-- markdownlint-disable MD060 -->

| Model                              | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|------------------------------------|--------------------------|-------------------------|------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` | still failing            | 2026-05-04 20:21:24 BST | 2/3 recent runs failed |
| `facebook/pe-av-large`             | new model failing        | 2026-05-04 22:51:32 BST | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 8
- **Summary diagnostics models:** 51
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 566.27s (566.27s)
- **Average runtime per model:** 9.60s (9.60s)
- **Dominant runtime phase:** model load dominated 32/59 measured model runs (52% of tracked runtime).
- **Phase totals:** model load=296.69s, local prompt prep=0.19s, upstream prefill / first-token=61.51s, post-prefill decode=205.44s, cleanup=6.13s
- **Generation total:** 266.96s across 57 model(s); upstream prefill / first-token split available for 57/57 model(s).
- **Observed stop reasons:** completed=47, exception=2, max_tokens=10
- **Validation overhead:** 0.10s total (avg 0.00s across 59 model(s)).
- **Upstream prefill / first-token latency:** Avg 1.08s | Min 0.07s | Max 4.85s across 57 model(s).
- **What this likely means:** Cold model load time is a major share of runtime for this cohort.
- **Suggested next action:** Consider staged runs, model reuse, or narrowing the model set before reruns.

---

## Models Not Flagged (51 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 41 model(s).
- **Ran, but with quality warnings:** 10 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --trust-remote-code --max-tokens 200 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --presence-context-size 20 --frequency-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `facebook/pe-av-large`, `HuggingFaceTB/SmolVLM-Instruct`, `microsoft/Phi-3.5-vision-instruct`, `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/SmolVLM-Instruct-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg`
- Generation settings: max_tokens=200, temperature=0.0, top_p=1.0

_Report generated on 2026-06-07 21:35:51 BST by [check_models](https://github.com/jrp2014/check_models)._

<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.2                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260607+8f0e8b14                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
| transformers               | 5.10.2                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.18.0                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.1                                                                                                                                                   |
| SDK Version                | 26.5                                                                                                                                                     |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                       |
| Xcode Version              | 26.5                                                                                                                                                     |
| Xcode Build                | 17F42                                                                                                                                                    |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                               |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                           |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                                        |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                           |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                          |
| GPU/Chip                   | Apple M5 Max                                                                                                                                             |
| GPU Cores                  | 40                                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                                  |
| MLX Install Type           | editable local source                                                                                                                                    |
| MLX Distribution Root      | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                                       |
| MLX Core Extension         | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (157,751,704 bytes, sha256=ba9913d81d92bbbde42bbc6dda27e80ecb31db6031fa073e6c8aeb0666d47c33) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,675,136 bytes, sha256=6255fc531acc826e8625f261237f6bb6c75490177d3b769ab70c1ff9f71b6d7f)  |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
