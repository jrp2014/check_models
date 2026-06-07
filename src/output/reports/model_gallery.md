# Model Output Gallery

_Generated on 2026-06-07 21:35:51 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Shortlist

### Strong Candidates

- `Qwen/Qwen3-VL-2B-Instruct`: ✅ B (75/100) | Desc 81 | Keywords 0 | 134.0 tps
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ✅ B (75/100) | Desc 85 | Keywords 0 | 130.7 tps
- `mlx-community/Qwen3.5-35B-A3B-4bit`: ✅ B (75/100) | Desc 87 | Keywords 0 | 117.1 tps
- `mlx-community/Qwen3.5-9B-MLX-4bit`: ✅ B (75/100) | Desc 83 | Keywords 0 | 101.1 tps
- `mlx-community/GLM-4.6V-Flash-6bit`: ✅ B (75/100) | Desc 83 | Keywords 0 | 64.0 tps

### Watchlist

- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | 81.8 tps | harness, long context
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 60 | Keywords 0 | 33.1 tps | harness
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (33/100) | Desc 76 | Keywords 0 | 128.6 tps | harness
- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (33/100) | Desc 76 | Keywords 0 | 125.9 tps | harness
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (35/100) | Desc 93 | Keywords 0 | 59.9 tps | generation loop, harness, text sanity

## 🚨 Failures by Package (Actionable)

| Package           |   Failures | Error Types   | Affected Models                    |
|-------------------|------------|---------------|------------------------------------|
| `mlx`             |          1 | Model Error   | `LiquidAI/LFM2.5-VL-450M-MLX-bf16` |
| `huggingface-hub` |          1 | Model Error   | `facebook/pe-av-large`             |

### Actionable Items by Package

#### mlx

- LiquidAI/LFM2.5-VL-450M-MLX-bf16 (Model Error)
  - Error: `Model loading failed: Received 2 parameters not in model: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_project...`
  - Type: `ValueError`

#### huggingface-hub

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av...`
  - Type: `ValueError`

## Image Metadata

- _Date:_ 2026-04-25 23:49:44 BST
- _Time:_ 23:49:44

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Quick Navigation

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)
- _Best descriptions:_ [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)
- _Best keywording:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)
- _Best balance:_ [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)
- _Failed models:_ `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `facebook/pe-av-large`
- _D/F utility models:_ `HuggingFaceTB/SmolVLM-Instruct`,
  `microsoft/Phi-3.5-vision-instruct`,
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`,
  `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`,
  `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/Idefics3-8B-Llama3-bf16`,
  +10 more

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ❌ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Model
  Error`; diagnostic code `MLX_MODEL_LOAD_MODEL`
- _Next step:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.
- _Error summary:_ see error details below
- _Key signals:_ model error; mlx model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_MODEL_LOAD_MODEL`; package `mlx`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception

_Error details:_

> Model loading failed: Received 2 parameters not in model:
> multi_modal_projector.layer_norm.bias,
> multi_modal_projector.layer_norm.weight.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
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

</details>

---

<a id="model-facebook-pe-av-large"></a>

### ❌ facebook/pe-av-large

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `huggingface-hub`; reported package `huggingface-hub`;
  failure stage `Model Error`; diagnostic code
  `HUGGINGFACE_HUB_MODEL_LOAD_MODEL`
- _Next step:_ Check cache/revision availability and network/auth state before
  blaming the model.
- _Error summary:_ see error details below
- _Key signals:_ model error; huggingface hub model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `HUGGINGFACE_HUB_MODEL_LOAD_MODEL`; package `huggingface-hub`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception

_Error details:_

> Model loading failed: [Errno 2] No such file or directory:
> '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av-large/blobs/7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354.06d762e2.incomplete'

<details>
<summary>Full Traceback (click to expand)</summary>

```python
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

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ❌ F (33/100); Output lacks detail
- _Review focus:_ watchlist (Output lacks detail)
- _Timing:_ Load 0.58s; Gen 0.17s; Total 0.75s
- _Throughput:_ Prompt 3,865 TPS (269 tok); Gen 193 TPS (13 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Score:_ 🟡 C (59/100); Keywords are not specific or diverse enough
- _Timing:_ Load 0.50s; Gen 0.33s; Total 0.84s
- _Throughput:_ Prompt 219 TPS (22 tok); Gen 375 TPS (81 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 81 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows a close-up view of two cats lying down on a pink fabric
> surface. Both cats have striped coats, and they are positioned on a couch.
> The cats are facing the camera, and their eyes are open, indicating
> alertness. The image is a close-up, color photograph, and it captures the
> relaxed posture of the cats. There are no texts or other objects in the
> image.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Score:_ ✅ B (69/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.65s; Gen 0.42s; Total 1.08s
- _Throughput:_ Prompt 240 TPS (22 tok); Gen 115 TPS (35 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 35 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image features two cats lying on a couch. One cat is a light brown and
> the other is a dark brown. They both have green eyes and a black nose.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ❌ F (34/100); Output lacks detail
- _Review focus:_ watchlist (Output lacks detail)
- _Timing:_ Load 0.48s; Gen 0.43s; Total 0.92s
- _Throughput:_ Prompt 717 TPS (269 tok); Gen 326 TPS (10 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 10 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### ✅ mlx-community/MiniCPM-V-4.6-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=97%
- _Score:_ 🟠 D (47/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (Keywords are not specific or diverse enough)
- _Timing:_ Load 0.93s; Gen 0.44s; Total 1.38s
- _Throughput:_ Prompt 699 TPS (228 tok); Gen 276 TPS (22 tok)
- _Tokens:_ prompt 228 tok; estimated text 6 tok; estimated non-text 222 tok;
  generated 22 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two cats lying on a pink blanket. There are remote controls
> on the blanket as well.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (74/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.49s; Gen 0.48s; Total 0.98s
- _Throughput:_ Prompt 1,407 TPS (417 tok); Gen 324 TPS (52 tok)
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 52 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two cats lying on a pink blanket. One cat is on the left
> side, while the other is on the right side. There are two remote controls
> placed on the blanket, one on the left side and the other on the right side.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (1.1%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=99%
- _Score:_ ❌ F (33/100); Output lacks detail
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 0.58s; Gen 0.53s; Total 1.11s
- _Throughput:_ Prompt 3,394 TPS (1,196 tok); Gen 129 TPS (13 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:output_ratio(1.1%)

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ 🟠 D (35/100); Output lacks detail
- _Review focus:_ watchlist (Output lacks detail)
- _Timing:_ Load 0.87s; Gen 0.58s; Total 1.45s
- _Throughput:_ Prompt 3,841 TPS (770 tok); Gen 60.1 TPS (19 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 19 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch with remote controls beside them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (70/100); Missing requested structure
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.02s; Gen 0.84s; Total 1.87s
- _Throughput:_ Prompt 3,034 TPS (949 tok); Gen 199 TPS (99 tok)
- _Tokens:_ prompt 949 tok; estimated text 6 tok; estimated non-text 943 tok;
  generated 99 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In this image, there are two cats lying on a soft, pink cushion or blanket.
>
> &#45; The cat on the left appears to be a kitten, with a smaller size, fluffy
> fur, and a relaxed posture.
> &#45; The cat on the right is an adult, with a more robust build, striped tabby
> fur, and is stretched out comfortably, looking slightly at the kitten.
>
> Both are positioned near what looks like a remote control, suggesting a cozy
> indoor setting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (66/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.31s; Gen 0.85s; Total 2.17s
- _Throughput:_ Prompt 1,052 TPS (408 tok); Gen 93.2 TPS (42 tok)
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 42 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two tabby cats resting on a bright pink couch. One cat lies
> on its side, while the other is curled with its head down. Two remote
> controls are also visible on the couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.35s; Gen 0.98s; Total 2.33s
- _Throughput:_ Prompt 1,227 TPS (319 tok); Gen 101 TPS (71 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 71 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two tabby cats sleeping on a bright pink blanket or couch
> cover. The cat on the left is a smaller kitten, lying on its side with its
> head resting near a white remote control. The cat on the right is larger and
> also sleeping, positioned next to another remote control. Both cats appear
> relaxed and comfortable in their cozy setting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ 🟡 C (65/100); Keywords are not specific or diverse enough
- _Timing:_ Load 2.34s; Gen 1.04s; Total 3.41s
- _Throughput:_ Prompt 349 TPS (286 tok); Gen 115 TPS (24 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 24 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are lying on a pink blanket on a red couch, with a remote
> control next to each cat.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.46s; Gen 1.11s; Total 3.58s
- _Throughput:_ Prompt 569 TPS (319 tok); Gen 117 TPS (63 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 63 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sprawled out on a bright pink couch, sleeping peacefully.
> One cat is curled near the left armrest with a remote beside it, while the
> other lies stretched across the center, next to another remote. The scene is
> cozy and relaxed, capturing a quiet moment of feline comfort.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=77%
- _Score:_ ❌ F (31/100); Output lacks detail
- _Review focus:_ watchlist (Output lacks detail)
- _Timing:_ Load 0.59s; Gen 1.15s; Total 1.74s
- _Throughput:_ Prompt 24.4 TPS (26 tok); Gen 314 TPS (15 tok)
- _Tokens:_ prompt 26 tok; estimated text 6 tok; estimated non-text 20 tok;
  generated 15 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch next to two remote controls.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (69/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.06s; Gen 1.19s; Total 4.27s
- _Throughput:_ Prompt 534 TPS (319 tok); Gen 99.0 TPS (58 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 58 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sprawled out asleep on a bright pink couch, nestled
> between two remote controls. The cat on the left is stretched out with its
> tail curled, while the one on the right is curled up facing downward. Both
> appear deeply relaxed in a cozy, domestic scene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 3 tokens.; At long prompt
  length (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%).;
  output/prompt=0.07%; nontext prompt burden=100%
- _Score:_ ❌ F (0/100); Empty or minimal output
- _Review focus:_ watchlist (harness, long context)
- _Timing:_ Load 1.20s; Gen 1.21s; Total 2.42s
- _Throughput:_ Prompt 3,569 TPS (4,103 tok); Gen 81.8 TPS (3 tok)
- _Tokens:_ prompt 4103 tok; estimated text 6 tok; estimated non-text 4097
  tok; generated 3 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Cat.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- output:truncated(3tok)
- long_context_low_ratio(0.1%;4103->3)

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (74/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.15s; Gen 1.32s; Total 2.48s
- _Throughput:_ Prompt 1,471 TPS (417 tok); Gen 66.8 TPS (65 tok)
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 65 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats lying on a pink couch. Both cats are
> stretched out, appearing relaxed. Between them are two remote controls. The
> couch has a vibrant pink cover, and the cats' fur has a mix of brown, black,
> and white stripes. The scene conveys a cozy and calm atmosphere.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.66s; Gen 1.39s; Total 2.06s
- _Throughput:_ Prompt 457 TPS (315 tok); Gen 134 TPS (89 tok)
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 89 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This is a close-up photograph of two cats sleeping on a bright pink couch.
> The cat on the left is a tabby with a black and grey pattern, lying on its
> back with its eyes closed. The cat on the right is a calico with a brown and
> black striped pattern, also lying on its back. Both cats are resting
> peacefully. There are two remote controls on the couch, one white and one
> with a blue button.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (71/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.37s; Gen 1.44s; Total 2.82s
- _Throughput:_ Prompt 1,502 TPS (950 tok); Gen 65.8 TPS (52 tok)
- _Tokens:_ prompt 950 tok; estimated text 6 tok; estimated non-text 944 tok;
  generated 52 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two tabby cats lounging on a pink blanket. Both cats appear
> relaxed, lying on their sides with their legs stretched out. One cat faces
> left, while the other faces right, and there seem to be remote controls near
> them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=repeated_punctuation: '!...'
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (degeneration, generation loop)
- _Timing:_ Load 0.76s; Gen 1.60s; Total 2.38s
- _Throughput:_ Prompt 3,725 TPS (317 tok); Gen 135 TPS (200 tok)
- _Tokens:_ prompt 317 tok; estimated text 6 tok; estimated non-text 311 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (repeated_punctuation: '!...')
- Generation loop (degeneration)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=repeated_punctuation: '!...'
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (degeneration, generation loop)
- _Timing:_ Load 0.74s; Gen 1.61s; Total 2.35s
- _Throughput:_ Prompt 3,689 TPS (315 tok); Gen 134 TPS (200 tok)
- _Tokens:_ prompt 315 tok; estimated text 6 tok; estimated non-text 309 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (repeated_punctuation: '!...')
- Generation loop (degeneration)
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (67/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.51s; Gen 1.62s; Total 4.14s
- _Throughput:_ Prompt 392 TPS (286 tok); Gen 28.7 TPS (25 tok)
- _Tokens:_ prompt 286 tok; estimated text 6 tok; estimated non-text 280 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping on a bright pink blanket on a red couch, with
> two remote controls lying next to them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ 🟡 C (56/100); Keywords are not specific or diverse enough
- _Timing:_ Load 1.45s; Gen 1.63s; Total 3.09s
- _Throughput:_ Prompt 1,517 TPS (950 tok); Gen 69.0 TPS (68 tok)
- _Tokens:_ prompt 950 tok; estimated text 6 tok; estimated non-text 944 tok;
  generated 68 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image shows two cats lying on their backs on a pink blanket or cushion.
> Both cats appear relaxed and playful, with their legs stretched out and
> paws in the air. One cat is smaller, possibly a kitten, while the other is
> larger, likely an adult. There are two remote controls placed near them on
> the blanket.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%; formatting=Unknown tags:
  &lt;end_of_utterance&gt;
- _Score:_ ❌ F (30/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (Keywords are not specific or diverse enough)
- _Timing:_ Load 1.87s; Gen 1.76s; Total 3.64s
- _Throughput:_ Prompt 2,265 TPS (2,327 tok); Gen 33.4 TPS (23 tok)
- _Tokens:_ prompt 2327 tok; estimated text 6 tok; estimated non-text 2321
  tok; generated 23 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In this image we can see two cats on the sofa. There are two remotes on the
> sofa.<end_of_utterance>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=94%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.56s; Gen 1.79s; Total 2.36s
- _Throughput:_ Prompt 337 TPS (97 tok); Gen 131 TPS (187 tok)
- _Tokens:_ prompt 97 tok; estimated text 6 tok; estimated non-text 91 tok;
  generated 187 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In the image, a cat is comfortably nestled on a red couch, its body relaxed
> and at ease. The cat's fur is a mix of black and white, with the black
> spots standing out against the vibrant red of the couch. The cat's head is
> turned to the side, and its eyes are closed, suggesting it is either
> sleeping or in a state of deep relaxation.
>
> The couch itself is a striking shade of red, providing a vivid contrast to
> the cat's fur. The couch is positioned against a white wall, which adds a
> clean and minimalist aesthetic to the scene.
>
> In the background, there's a white door, slightly ajar, suggesting that the
> room is not completely enclosed. The door's presence adds a sense of
> openness and spaciousness to the image.
>
> Overall, the image captures a peaceful moment of a cat resting on a red
> couch, surrounded by a minimalist and clean environment.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.49s; Gen 1.94s; Total 3.45s
- _Throughput:_ Prompt 488 TPS (408 tok); Gen 64.0 TPS (69 tok)
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 69 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats resting on a bright pink couch. One cat lies
> on its side, eyes closed, with a remote control nearby. The other cat,
> larger and more robust, rests with its head down, also near a remote. The
> scene is cozy, with the vivid pink couch contrasting with the cats’ striped
> fur.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (1.1%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=99%
- _Score:_ ❌ F (33/100); Output lacks detail
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 0.69s; Gen 2.06s; Total 2.76s
- _Throughput:_ Prompt 628 TPS (1,196 tok); Gen 126 TPS (13 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 13 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:output_ratio(1.1%)

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Score:_ 🟡 C (57/100); Keywords are not specific or diverse enough
- _Timing:_ Load 1.77s; Gen 2.65s; Total 4.43s
- _Throughput:_ Prompt 2,682 TPS (3,341 tok); Gen 34.8 TPS (48 tok)
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 48 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two cats sleeping on a pink blanket. One cat is a small
> kitten, and the other is a larger adult cat. Both are lying on their sides,
> and there are two remote controls placed near them on the couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=62%
- _Score:_ 🟡 C (65/100); Keywords are not specific or diverse enough
- _Timing:_ Load 1.65s; Gen 2.76s; Total 4.42s
- _Throughput:_ Prompt 9.92 TPS (16 tok); Gen 22.2 TPS (25 tok)
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 25 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats lying on a pink blanket, with two remote
> controls placed on the couch behind them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.17s; Gen 2.77s; Total 4.96s
- _Throughput:_ Prompt 497 TPS (319 tok); Gen 33.5 TPS (71 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 71 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are lounging on a bright pink couch, each with a remote
> control nearby. The cat on the left is stretched out on its back, paws
> relaxed, while the one on the right lies curled up, head resting near its
> front paws. Both appear to be napping or resting comfortably in a cozy,
> playful setting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=phrase: "have this image. have..."
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 1.39s; Gen 2.87s; Total 4.27s
- _Throughput:_ Prompt 223 TPS (266 tok); Gen 120 TPS (200 tok)
- _Tokens:_ prompt 266 tok; estimated text 6 tok; estimated non-text 260 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
>
> have this image.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "have this image. have...")
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (67/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.15s; Gen 2.88s; Total 6.05s
- _Throughput:_ Prompt 389 TPS (319 tok); Gen 19.6 TPS (40 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 40 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping on a bright pink couch. One cat is stretched out
> on its side, while the other is curled up nearby. Two remote controls lie
> on the couch between them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (74/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.20s; Gen 2.93s; Total 5.14s
- _Throughput:_ Prompt 749 TPS (274 tok); Gen 48.7 TPS (124 tok)
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 124 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are peacefully napping on a bright pink couch. The cat on the
> left is lying on its back with its paws curled up, its tail curled around
> its body. It has a green collar. The cat on the right is also lying on its
> back, with its head resting on its paws and its tail stretched out.
>
> Between the two cats are two television remotes, one white and one blue. The
> couch is made of a soft, plush material. The lighting in the room is soft
> and warm, creating a cozy atmosphere. The cats appear relaxed and content,
> enjoying a comfortable nap.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Score:_ ❌ F (31/100); Output lacks detail
- _Review focus:_ watchlist (Output lacks detail)
- _Timing:_ Load 1.72s; Gen 3.07s; Total 4.80s
- _Throughput:_ Prompt 1,289 TPS (3,341 tok); Gen 33.4 TPS (15 tok)
- _Tokens:_ prompt 3341 tok; estimated text 6 tok; estimated non-text 3335
  tok; generated 15 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket with two remote controls nearby.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Score:_ ✅ B (74/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.88s; Gen 3.08s; Total 3.97s
- _Throughput:_ Prompt 1,093 TPS (2,356 tok); Gen 63.6 TPS (55 tok)
- _Tokens:_ prompt 2356 tok; estimated text 6 tok; estimated non-text 2350
  tok; generated 55 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two cats lying on a pink blanket. The cat on the left is a
> kitten with a black and white striped tail. The cat on the right is a tabby
> cat with a black and brown striped tail. Both cats are sleeping.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (73/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.00s; Gen 3.21s; Total 6.23s
- _Throughput:_ Prompt 471 TPS (319 tok); Gen 19.5 TPS (49 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 49 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sleeping peacefully on a bright pink couch. One cat lies
> on its side near a white remote control, while the other stretches out
> nearby, also next to a remote. Both appear relaxed and comfortable in their
> cozy spot.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; reasoning
  leak
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (reasoning leak)
- _Timing:_ Load 2.00s; Gen 3.45s; Total 5.46s
- _Throughput:_ Prompt 485 TPS (399 tok); Gen 79.5 TPS (200 tok)
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, there are two tabby cats on a
> pink fabric surface. The cat on the left is lying on its side, facing the
> other cat. The cat on the right is lying on its back, facing the left cat.
> Both cats have striped fur with a mix of brown, black, and white. There are
> two remote controls: one near the left cat and one near the right cat. The
> image has a high-contrast, possibly filtered look with enhanced colors,
> making the pink background and the cats' fur stand out. The overall scene is
> cozy, with the cats appearing relaxed.◁/think▷The image shows two tabby
> cats resting on a pink fabric surface. The cat on the left lies on its side,
> facing the other cat, while the cat on the right lies on its back, also
> facing the left cat. Both cats have striped fur with a mix of brown, black,
> and white patterns.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷, ◁/think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 76 occurrences).; nontext prompt burden=99%
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 2.01s; Gen 3.48s; Total 5.51s
- _Throughput:_ Prompt 558 TPS (417 tok); Gen 33.1 TPS (90 tok)
- _Tokens:_ prompt 417 tok; estimated text 6 tok; estimated non-text 411 tok;
  generated 90 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠblanketĠorĠaĠcouch.ĠTheĠcatĠonĠtheĠleftĠisĠaĠkitten,ĠandĠtheĠoneĠonĠtheĠrightĠisĠanĠadultĠcat.ĠBothĠcatsĠareĠinĠrelaxedĠpostures,ĠwithĠtheĠkittenĠlyingĠonĠitsĠsideĠandĠtheĠadultĠcatĠlyingĠonĠitsĠback.ĠThereĠareĠtwoĠremoteĠcontrolsĠplacedĠnearĠtheĠcats,ĠoneĠcloserĠtoĠtheĠkittenĠandĠtheĠotherĠcloserĠtoĠtheĠadultĠcat.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- token_encoding:bpe_space_leak(76)

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; reasoning
  leak
- _Score:_ ✅ B (70/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, reasoning leak)
- _Timing:_ Load 2.18s; Gen 3.64s; Total 5.82s
- _Throughput:_ Prompt 668 TPS (399 tok); Gen 68.8 TPS (200 tok)
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Okay, let me try to figure out how to describe this image. First, I
> need to look at the image carefully. There are two cats on a pink surface,
> maybe a blanket or a bed. Both cats seem to be lying down. The one on the
> left is a tabby with gray and black stripes, and the one on the right is a
> darker tabby, maybe a bit bigger. They both look like they're sleeping or
> resting. There's a remote control near the left cat and another one near the
> right cat. The background is pink, which might be a couch or a bed. The
> image is a bit blurry, especially around the edges, which makes it look like
> a low-resolution or heavily processed photo. The colors are quite
> saturated, especially the pink background and the cats' fur. The cats' eyes
> are closed, and their poses are relaxed. Maybe they're siblings or just two
> cats that are comfortable with each other. The remotes suggest a home
> setting
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (66/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.31s; Gen 3.69s; Total 6.02s
- _Throughput:_ Prompt 340 TPS (275 tok); Gen 31.8 TPS (91 tok)
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 91 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here's a brief description of the image:
>
> The image shows two tabby cats lying side-by-side on a bright pink surface,
> likely a couch cushion or blanket. Both cats are relaxed and appear to be
> sleeping. Each cat has a remote control resting on its body – one on its
> chest and one nearby. The cats have striped fur patterns and are in a
> comfortable, stretched-out position. The overall scene is cute and playful.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 5.48s; Gen 3.93s; Total 9.43s
- _Throughput:_ Prompt 161 TPS (408 tok); Gen 51.4 TPS (71 tok)
- _Tokens:_ prompt 408 tok; estimated text 6 tok; estimated non-text 402 tok;
  generated 71 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats resting on a bright pink blanket spread over
> a red couch. The cat on the is stretched out, sleeping with its head down,
> while the cat on the left lies curled, also asleep. Two remote controls (one
> gray, one white) rest near the cats, adding a cozy, domestic feel to the
> scene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.65s; Gen 4.13s; Total 5.79s
- _Throughput:_ Prompt 906 TPS (1,031 tok); Gen 34.8 TPS (101 tok)
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 101 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A top-down view of two cats laying on a pink blanket. The cat on the left is
> a gray tabby cat with black stripes and is laying on its side with its head
> facing the right side of the image. The cat on the right is a brown tabby
> cat with black stripes and is laying on its side with its head facing the
> left side of the image. There is a gray remote control on the left side of
> the image and a gray remote control on the right side of the image.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (200); nontext prompt burden=99%
- _Score:_ 🟠 D (35/100); Output lacks detail
- _Review focus:_ watchlist (generation loop, harness, text sanity)
- _Timing:_ Load 0.88s; Gen 4.34s; Total 5.23s
- _Throughput:_ Prompt 814 TPS (770 tok); Gen 59.9 TPS (200 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink couch with remote controls beside
> them.<|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (gibberish(token_noise))
- Generation loop (token_noise)
- Likely capped by max token budget
- tokenizer_artifact
- token_leak:<\|end\|>
- token_leak:<\|endoftext\|>

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (74/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 7.63s; Gen 4.40s; Total 12.04s
- _Throughput:_ Prompt 88.7 TPS (319 tok); Gen 69.9 TPS (55 tok)
- _Tokens:_ prompt 319 tok; estimated text 6 tok; estimated non-text 313 tok;
  generated 55 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two tabby cats are sprawled out asleep on a bright pink couch, nestled
> between two remote controls — one white and one gray. The cat on the left is
> curled with its tail tucked, while the one on the right lies stretched out,
> both looking deeply relaxed.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ 🟡 C (60/100); Keywords are not specific or diverse enough
- _Timing:_ Load 3.36s; Gen 4.78s; Total 8.15s
- _Throughput:_ Prompt 322 TPS (275 tok); Gen 17.9 TPS (70 tok)
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 70 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here's a brief description of the image:
>
> The image shows two tabby kittens lying on their backs on a bright pink
> surface (likely a couch or blanket). Both kittens are relaxed and appear to
> be sleeping. Each kitten has a remote control resting on its belly. The
> kittens have striped fur patterns and are in a playful, endearing pose.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; nontext prompt burden=100%; reasoning leak
- _Score:_ ✅ B (67/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (harness, reasoning leak, text sanity)
- _Timing:_ Load 1.56s; Gen 5.04s; Total 6.61s
- _Throughput:_ Prompt 1,226 TPS (1,330 tok); Gen 44.1 TPS (174 tok)
- _Tokens:_ prompt 1330 tok; estimated text 6 tok; estimated non-text 1324
  tok; generated 174 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here are my reasoning steps:
> The user asks: "Describe this image briefly." The image is of two cats
> sleeping on a pink couch, with a remote control. The user wants a brief
> description. This is a straightforward request. There's no disallowed
> content. So we can comply. Provide a concise description. Possibly mention
> the cats' colors, positions, the couch, the remote. The user wants a brief
> description. So we can say something like: "Two tabby cats are sleeping side
> by side on a pink couch, with a TV remote lying nearby." That's it. No need
> for extra. We'll comply.
> [BEGIN FINAL RESPONSE]
> Two tabby cats are curled up sleeping side‑by‑side on a pink couch, with a
> TV remote resting nearby.
> [END FINAL RESPONSE]
> <|end|>
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (gibberish(token_noise))
- Reasoning leak (here are my reasoning steps, the user asks:)
- tokenizer_artifact
- token_leak:<\|end\|>

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=incomplete_sentence: ends with 'a'
- _Score:_ 🟠 D (47/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, generation loop)
- _Timing:_ Load 5.05s; Gen 5.20s; Total 10.40s
- _Throughput:_ Prompt 215 TPS (439 tok); Gen 65.0 TPS (200 tok)
- _Tokens:_ prompt 439 tok; estimated text 6 tok; estimated non-text 433 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Okay, let's break this down. The user wants a brief description of an image,
> and they've given me a pretty convoluted OCR output. My first step is
> always to understand the request. "Briefly describe" means I need to be
> concise, but also accurate.
>
> The OCR is a mess. It's a jumble of words, and I can see the user has tried
> to use the image's content to generate the text. It's a classic case of the
> AI hallucinating and then trying to make sense of itself. I need to ignore
> the OCR and focus on the image itself.
>
> Alright, let's look at the image. It's a photograph of two cats on a couch.
> One is a calico, and the other is a tiger-striped tabby. They're both
> sleeping. The setting is a red couch.
>
> Now, I need to synthesize a
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'a')
- Generation loop (degeneration)
- Likely capped by max token budget

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟡 C (62/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 1.64s; Gen 5.87s; Total 7.52s
- _Throughput:_ Prompt 1,442 TPS (1,239 tok); Gen 40.0 TPS (200 tok)
- _Tokens:_ prompt 1239 tok; estimated text 6 tok; estimated non-text 1233
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In the tranquil setting of this image, two feline companions, one adult and
> one kitten, are enjoying a peaceful slumber on a vibrant red couch. The
> adult cat, with its coat of brown and black stripes, is curled up in a
> perfect circle, embodying the essence of serenity. Its head is gently
> resting on its paws, a picture of contentment.
>
> Beside it, the kitten, with its lighter brown and black stripes, is sprawled
> out on its side. Its head is resting on the armrest of the couch, mirroring
> the relaxed posture of its companion. The kitten's legs are stretched out
> in front of it, adding to the overall sense of comfort and ease.
>
> Adding a touch of everyday life to the scene, two remote controls are
> casually placed on the armrest of the couch. Their presence suggests recent
> human activity, perhaps someone who has just left the room or is about to
> return.
>
> This image captures a moment of quiet companionship between
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.30s; Gen 5.96s; Total 7.26s
- _Throughput:_ Prompt 154 TPS (745 tok); Gen 50.8 TPS (55 tok)
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 55 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a red couch. One cat is thin
> with black stripes, while the other is thicker with brown and black stripes.
> Both cats are curled up with their tails tucked in. Two remote controls are
> visible on the couch behind them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (74/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.69s; Gen 6.69s; Total 8.39s
- _Throughput:_ Prompt 154 TPS (745 tok); Gen 30.0 TPS (55 tok)
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 55 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Two cats are sleeping on a pink blanket on a red couch. One cat is thin
> with black stripes, while the other is thicker with brown and black stripes.
> Both cats are curled up with their paws tucked in. There are two remote
> controls visible on the couch.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.40s; Gen 6.76s; Total 8.17s
- _Throughput:_ Prompt 2,753 TPS (1,031 tok); Gen 18.7 TPS (119 tok)
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 119 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A top-down view of two cats sleeping on a pink blanket. The cat on the left
> is a gray and black tabby cat, and it is lying on its side with its head
> facing the right and its front paws stretched out in front of it. Its tail
> is sticking out to the left. The cat on the right is lying on its side, and
> it is brown and black. Its head is facing the left, and its front paws are
> stretched out in front of it. Its tail is sticking out to the left. There
> are two white and blue remote controls between the cats.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=98%
- _Score:_ ✅ B (70/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 6.02s; Gen 7.30s; Total 13.34s
- _Throughput:_ Prompt 110 TPS (274 tok); Gen 7.52 TPS (36 tok)
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 36 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Both cats are sleeping on a pink blanket. The difference between these
> images is that one cat is on the left side of the blanket and the other is
> on the right side.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟡 C (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 2.48s; Gen 10.79s; Total 13.28s
- _Throughput:_ Prompt 1,547 TPS (1,239 tok); Gen 20.1 TPS (200 tok)
- _Tokens:_ prompt 1239 tok; estimated text 6 tok; estimated non-text 1233
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> In the tranquil setting of this image, two feline companions, one a tabby
> and the other a Bengal, are enjoying a peaceful slumber on a vibrant red
> couch. The tabby, with its distinctive striped coat, is sprawled out on its
> side, its head resting comfortably on the armrest of the couch. Its body is
> relaxed, with its legs stretched out in a display of feline contentment.
>
> Beside it, the Bengal cat is curled up in a perfect circle, its head resting
> on its paws in a picture of serenity. The contrast between the two cats'
> sleeping positions adds a touch of whimsy to the scene.
>
> Adding to the homely atmosphere, two remote controls are casually placed on
> the armrest of the couch. One is a standard TV remote, while the other is a
> DVD remote, suggesting recent human activity in the room. The precise
> location of these objects relative to each other and the cats adds depth to
> the image, creating a snapshot
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=65%
- _Score:_ ✅ B (67/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.14s; Gen 17.27s; Total 19.42s
- _Throughput:_ Prompt 7.78 TPS (17 tok); Gen 5.11 TPS (77 tok)
- _Tokens:_ prompt 17 tok; estimated text 6 tok; estimated non-text 11 tok;
  generated 77 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows two tabby cats lying on a pink blanket, with two remote
> controls placed on the couch behind them. The cat on the left is a kitten
> with a fluffy tail and a green collar, while the cat on the right is an
> adult with a striped pattern. The image appears to be a playful and cozy
> scene, with the cats seemingly enjoying a relaxing moment together.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.17s; Gen 29.84s; Total 32.03s
- _Throughput:_ Prompt 328 TPS (766 tok); Gen 5.95 TPS (163 tok)
- _Tokens:_ prompt 766 tok; estimated text 6 tok; estimated non-text 760 tok;
  generated 163 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image captures a cozy scene of two cats lounging on a pink blanket. The
> cats are positioned side by side, with one on the left and one on the
> right. Both felines have similar fur patterns, featuring a mix of brown,
> black, and white stripes. They appear to be in a relaxed state, with their
> eyes closed and bodies stretched out comfortably.
>
> The cat on the left has its head resting on a white remote control, while
> the cat on the right is nestled next to another white remote. The pink
> blanket they're lying on is slightly wrinkled, adding to the casual and
> homey atmosphere of the scene.
>
> The background is a solid pink color, which complements the blanket and
> creates a warm, inviting setting. This image beautifully captures a moment
> of feline tranquility and comfort.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.43s; Gen 30.74s; Total 33.18s
- _Throughput:_ Prompt 847 TPS (1,031 tok); Gen 5.4 TPS (159 tok)
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 159 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A top-down view of two cats laying on a pink blanket. The cat on the left is
> a gray tabby cat with black stripes and a black tail. It is laying on its
> side with its head on the pink blanket and its body facing the left side of
> the image. Its front legs are stretched out in front of it, and its back
> legs are curled up. The cat on the right is a brown tabby cat with black
> stripes. It is laying on its side with its head on the pink blanket and its
> body facing the right side of the image. Its front legs are stretched out in
> front of it, and its back legs are curled up. There is a gray remote
> control on the left side of the image and a gray remote control on the right
> side of the image.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; reasoning
  leak
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, reasoning leak)
- _Timing:_ Load 3.61s; Gen 43.54s; Total 47.17s
- _Throughput:_ Prompt 428 TPS (399 tok); Gen 4.72 TPS (200 tok)
- _Tokens:_ prompt 399 tok; estimated text 6 tok; estimated non-text 393 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, there are two tabby cats on a
> pink fabric surface. The cat on the left is lying on its side, facing the
> other cat, with a remote control near its head. The cat on the right is
> lying on its back, also with a remote control near it. Both cats have
> striped fur with a mix of brown, black, and white. The image has a
> high-contrast, almost posterized or filtered look, making colors very vivid.
> The background is a pinkish-red fabric, maybe a couch or bed. The remotes
> are light blue with buttons, typical of TV remotes. The overall style is
> stylized with high saturation and contrast.◁/think▷The image shows two tabby
> cats resting on a pink fabric surface. The cat on the left lies on its
> side, facing the other cat, with a light - colored remote control near its
> head. The cat on the right is positioned
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷, ◁/think▷)
- Likely capped by max token budget

---

<!-- markdownlint-enable MD033 MD034 -->
