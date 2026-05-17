# Model Output Gallery

_Generated on 2026-05-17 21:19:22 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/nanoLLaVA-1.5-4bit`: ✅ B (75/100) | Desc 87 | Keywords 0 | 306.7 tps
- `qnguyen3/nanoLLaVA`: ✅ B (75/100) | Desc 90 | Keywords 0 | 114.9 tps
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ✅ B (75/100) | Desc 90 | Keywords 0 | 19.3 tps

### Watchlist

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | 426.4 tps | harness
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | 19.0 tps | harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 60 | Keywords 0 | 29.0 tps | degeneration, harness
- `mlx-community/GLM-4.6V-nvfp4`: ❌ F (5/100) | Desc 22 | Keywords 0 | 42.5 tps | harness
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) | Desc 44 | Keywords 0 | 39.8 tps | harness

## 🚨 Failures by Package (Actionable)

| Package   |   Failures | Error Types                  | Affected Models                                                                |
|-----------|------------|------------------------------|--------------------------------------------------------------------------------|
| `mlx`     |          2 | Model Error, Weight Mismatch | `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `mlx-lm`  |          1 | Model Error                  | `facebook/pe-av-large`                                                         |

### Actionable Items by Package

#### mlx

- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
  - Type: `ValueError`
- mlx-community/LFM2.5-VL-1.6B-bf16 (Weight Mismatch)
  - Error: `Model loading failed: Missing 2 parameters: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm....`
  - Type: `ValueError`

#### mlx-lm

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: Model type pe_audio_video not supported.`
  - Type: `ValueError`

## Image Metadata

- _Description:_ An image of a man and two dogs at Studland Beach in Dorset.
- _Keywords:_ Beach, Bushes, Coast, Dorset, England, Europe, Highcliffe
  Castle, Highcliffe-on-Sea, Pebbles, Person, Sand, Shoreline, Sky, Town,
  Trees, UK, United Kingdom, Walking, Water, Waves, animal, autumn, bay,
  boats, buildings, calm, canine, cloudy, coastal, coastal town, coastline,
  cold, companionship, distance, distant, dog, dogs, grey, harbor, horizon,
  houses, landscape, leisure, man, nature, ocean, outdoors, overcast,
  peaceful, pet, pets, recreation, scenic, sea, seaside, shingle, shore,
  solitude, tide, view, winter, yellow buoys
- _Date:_ 2026-05-16 15:35:27 BST
- _Time:_ 15:35:27
- _GPS:_ 50.729111°N, 1.745494°W

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Quick Navigation

- _Best end-to-end cataloging:_ [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)
- _Best keywording:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)
- _Best balance:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Failed models:_ `facebook/pe-av-large`,
  `mlx-community/Kimi-VL-A3B-Thinking-8bit`,
  `mlx-community/LFM2.5-VL-1.6B-bf16`
- _D/F utility models:_ `HuggingFaceTB/SmolVLM-Instruct`,
  `LiquidAI/LFM2.5-VL-450M-MLX-bf16`,
  `meta-llama/Llama-3.2-11B-Vision-Instruct`,
  `microsoft/Phi-3.5-vision-instruct`,
  `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`,
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, +36 more

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-facebook-pe-av-large"></a>

### ❌ facebook/pe-av-large

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx-lm`; reported package `mlx-lm`; failure stage
  `Model Error`; diagnostic code `MLX_LM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect the import path and installed package version that owns
  the missing module before treating this as a model failure.
- _Error summary:_ Model loading failed: Model type pe_audio_video not
  supported.
- _Key signals:_ model error; mlx lm model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_LM_MODEL_LOAD_MODEL`; package `mlx-lm`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception


<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-lm/mlx_lm/utils.py", line 188, in _get_classes
    arch = importlib.import_module(f"mlx_lm.models.{model_type}")
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/importlib/__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1395, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1324, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'mlx_lm.models.pe_audio_video'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17439, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16841, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 443, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 264, in load_model
    model_config = model_class.ModelConfig.from_dict(config)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/text_only.py", line 39, in from_dict
    model_class, model_args_class = _get_classes(params)
                                    ~~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-lm/mlx_lm/utils.py", line 191, in _get_classes
    raise ValueError(msg)
ValueError: Model type pe_audio_video not supported.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17636, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17449, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type pe_audio_video not supported.
```

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ❌ mlx-community/Kimi-VL-A3B-Thinking-8bit

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

> Model loading failed: Received 4 parameters not in model:
> multi_modal_projector.linear_1.biases,
> multi_modal_projector.linear_1.scales,
> multi_modal_projector.linear_2.biases,
> multi_modal_projector.linear_2.scales.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17439, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16841, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 443, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 367, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 4 parameters not in model:
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17636, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17449, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 4 parameters not in model:
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ❌ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Weight Mismatch)
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Weight
  Mismatch`; diagnostic code `MLX_MODEL_LOAD_WEIGHT_MISMATCH`
- _Next step:_ Compare checkpoint keys with the selected model class/config,
  especially projector scale/bias parameters and quantized weight naming,
  before judging model quality.
- _Error summary:_ see error details below
- _Key signals:_ weight mismatch; mlx model load weight mismatch
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_MODEL_LOAD_WEIGHT_MISMATCH`; package `mlx`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 200 tok; stop reason exception

_Error details:_

> Model loading failed: Missing 2 parameters:
> multi_modal_projector.layer_norm.bias,
> multi_modal_projector.layer_norm.weight.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17439, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16841, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 443, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 367, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 2 parameters:
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17636, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17449, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Missing 2 parameters:
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.; nontext prompt
  burden=98%
- _Score:_ ❌ F (0/100); Empty or minimal output
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 0.35s; Gen 0.33s; Total 0.87s
- _Throughput:_ Prompt 8,460 TPS (269 tok); Gen 426 TPS (3 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 3 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> í.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(3tok)

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=77%
- _Score:_ ❌ F (19/100); Output lacks detail
- _Review focus:_ watchlist (Output lacks detail)
- _Timing:_ Load 0.66s; Gen 0.45s; Total 1.32s
- _Throughput:_ Prompt 292 TPS (26 tok); Gen 286 TPS (9 tok)
- _Tokens:_ prompt 26 tok; estimated text 6 tok; estimated non-text 20 tok;
  generated 9 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ''){
> GREE, which is not included.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.54s; Gen 0.81s; Total 1.58s
- _Throughput:_ Prompt 253 TPS (22 tok); Gen 307 TPS (99 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 99 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image is a photograph of a person walking a dog on a beach. The dog is
> in the foreground, and the person is walking away from the camera. The beach
> setting is indicated by the presence of the ocean in the background. The
> image has a sepia tone, which gives it a vintage or nostalgic feel. There
> are no texts or discernible brands in the image. The photograph is taken
> during the day, as evidenced by the bright sky and the shadows cast on the
> ground.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=and
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 0.52s; Gen 1.04s; Total 1.76s
- _Throughput:_ Prompt 2,781 TPS (269 tok); Gen 314 TPS (200 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Escher himself, and, and and and and and and and and and and and and and
> and and and and and and and and and and and and and and and and and and and
> and and and and and and and and and and and and and and and and and and and
> and and and and and and and and and and and and and and and and and and and
> and and and and and and and and and and and and and and and and and and and
> and and and and and and and and and and and and and and and and and and and
> and and and and and and and and and and and and and and and and and and and
> and and and and and and and and and and and and and and and and and and and
> and and and and and and and and and and and and and and and and and and and
> and and and and and and and and and and and and and and and and and and and
> and and and and and and and and and and and
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (and)
- Likely capped by max token budget

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.55s; Gen 1.09s; Total 1.84s
- _Throughput:_ Prompt 226 TPS (22 tok); Gen 115 TPS (71 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 71 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image captures a picturesque beach scene with a man and a dog walking
> along the beach. The man is wearing a black coat and a black hat, while the
> dog is wearing a black and white coat. The dog is walking on the beach, and
> the man is following behind. The beach is covered in sand and there are
> waves in the water.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 6 tokens.; nontext prompt
  burden=99%
- _Score:_ ❌ F (5/100); Output too short to be useful
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 1.66s; Gen 1.26s; Total 3.15s
- _Throughput:_ Prompt 1,294 TPS (1,031 tok); Gen 39.8 TPS (6 tok)
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 6 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(6tok)

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.
- _Key signals:_ Output is very short relative to prompt size (0.2%),
  suggesting possible early-stop or prompt-handling issues.; At long prompt
  length (4103 tokens), output stayed unusually short (9 tokens; ratio 0.2%).;
  output/prompt=0.22%; nontext prompt burden=100%
- _Score:_ ❌ F (16/100); Output lacks detail
- _Review focus:_ watchlist (harness, long context)
- _Timing:_ Load 1.25s; Gen 1.59s; Total 3.06s
- _Throughput:_ Prompt 3,670 TPS (4,103 tok); Gen 62.2 TPS (9 tok)
- _Tokens:_ prompt 4103 tok; estimated text 6 tok; estimated non-text 4097
  tok; generated 9 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> It has been tagged with #1.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- output:output_ratio(0.2%)
- long_context_low_ratio(0.2%;4103->9)

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: ' A.' repeated; repetitive token=phrase: "a. a.
  a. a...."
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, repetitive)
- _Timing:_ Load 0.59s; Gen 1.78s; Total 2.57s
- _Throughput:_ Prompt 2,059 TPS (803 tok); Gen 187 TPS (200 tok)
- _Tokens:_ prompt 803 tok; estimated text 6 tok; estimated non-text 797 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> wand with One with A
> A. A, It is a
> A With a. A, A A. A, and more, it is all, A. A. A. A. A. A. A. A. A. A. A.
> A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A.
> A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A.
> A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A. A.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "a. a. a. a....")
- Output degeneration (character_loop: ' A.' repeated)
- Likely capped by max token budget

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟠 D (42/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (Keywords are not specific or diverse enough)
- _Timing:_ Load 1.13s; Gen 2.02s; Total 3.41s
- _Throughput:_ Prompt 2,536 TPS (1,339 tok); Gen 178 TPS (200 tok)
- _Tokens:_ prompt 1339 tok; estimated text 6 tok; estimated non-text 1333
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ھèT [siched by-arin is the originality is aDNA:oh-oh-oh-oh-oh-oh-ari-opts.
>
> This is a-oh-oh-oh-ari-ari-ari·s-ary.
>
> In [sic-oh-ari:doubsar·S-oh-oh-ari:stolenarys-oh-oh-ari:v. Oh, the
> ols-oh-oh-ari: OBSARs-oh-ari: O-ari:f-ari·s-oh-ari: O-ari: O-oh-ari: O-ari:
> O-oh-ari: O-oh-ari: OSSHARBARB!
>
> s-oh-oh-ari·s-oh-ari·s-oh-ari·s-oh-ari·by-ari·by-oh-oh-ari·by-ari·by-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=94%; repetitive
  token=phrase: "saff saff saff saff..."
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 0.57s; Gen 2.31s; Total 3.09s
- _Throughput:_ Prompt 281 TPS (97 tok); Gen 126 TPS (200 tok)
- _Tokens:_ prompt 97 tok; estimated text 6 tok; estimated non-text 91 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> blocked bulk gl boss Cologne c flavon superior Cologne- Cologne hurd diarr
> c hurd controvers mischie demean sock thick thick stenosisTube thick hurd
> controvers demean hurd suprem interference hemorrhagic demean apparent
> controvers demean hurd suprem saff Macro- demean apparent saff saff oct saff
> saff saff saff saff saff saff saff saff saff oct saff apparent saff saff
> saff saff saff saff saff saff saff saff saff apparent hurd saff saff saff
> saff apparent hurd hurd hurd saff saff saff saff saff saff demean hurd
> immers hurd hurd hurd hurd controvers hurd apparent demean apparent hurd
> saff saff saff saff saff saff saff saff saff saff saff saff saff saff saff
> saff saff saff saff saff saff saff saff saff saff saff saff saff saff saff
> saff saff saff saff saff saff saff saff saff saff saff saff hurd hurd hurd
> saff saff saff saff saffotic saff clear hurd saline saffotic apparent demean
> saff saffotic apparent hurd bag mor saline hurd saffotic apparent demean
> saffotic saff p the apparent hurd saline hurd saline hurd hurd hurd hurd
> hurd hurd bag the apparent saff saffotic bag mor hurd bag hurd bag mor
> demean
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "saff saff saff saff...")
- Likely capped by max token budget

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 0.50s; Gen 2.38s; Total 3.07s
- _Throughput:_ Prompt 2,703 TPS (1,196 tok); Gen 124 TPS (200 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> chip Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins
> Higgins the Bologna Higgins the Bologna Higgins the Bologna Higgins the
> Bologna Higgins the Bologna hy Higgins the Bologna hy Higgins BMI the BMI
> BMI BMI BMI BMI BMI the Bologna Baker Higgins BMI the Bologna Baker Higgins
> BMI hy Higgins BMI the Bologna Baker Higgins BMI the Bologna Baker Higgins
> BMI the Bologna Baker Higgins BMI Ter Aeron the Bologna Baker Higgins BMI
> Ter Aeron the Bologna scoring the Bologna Baker BMI Ter Uruguay scoring the
> Bologna Baker BMI Ter Uruguay scoring the Bologna Baker BMI Ter scoring the
> Bologna Baker BMI Ter scoring the Bologna Baker BMI Ter scoring the Bologna
> Baker BMI Ter the Venice Uruguay scoring the Venice Ter scoring the Venice
> Ter scoring the Venice Ter scoring the Venice Ter budget Ter budget Ter
> budget Ter budget Ter budget Ter budget Ter budget Ter budget Ter budget Ter
> budget Ter BMI Ter BMI Ter budget Ter BMI Ter BMI Ter BMI Ter BMI Ter BMI
> Ter the TerIntf la Ter budget Ter budget Ter BMI Ter BMI Ter BMI Ter BMI Ter
> budget Ter TerIntf Uruguay TerIntf Uruguay Ter budget TerIntf
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 6 tokens.; nontext prompt
  burden=99%
- _Score:_ ❌ F (5/100); Output too short to be useful
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 2.51s; Gen 2.41s; Total 5.14s
- _Throughput:_ Prompt 897 TPS (1,031 tok); Gen 6.42 TPS (6 tok)
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 6 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(6tok)

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 0.60s; Gen 2.50s; Total 3.31s
- _Throughput:_ Prompt 2,662 TPS (1,196 tok); Gen 118 TPS (200 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> chip Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins
> Higgins the Bologna Higgins the Bologna Higgins the Bologna Higgins the
> Bologna Higgins the Bologna hy Higgins the Bologna hy Higgins BMI the BMI
> BMI BMI BMI BMI BMI the Bologna Baker Higgins BMI the Bologna Baker Higgins
> BMI hy Higgins BMI the Bologna Baker Higgins BMI the Bologna Baker Higgins
> BMI the Bologna Baker Higgins BMI Ter Aeron the Bologna Baker Higgins BMI
> Ter Aeron the Bologna scoring the Bologna Baker BMI Ter Uruguay scoring the
> Bologna Baker BMI Ter Uruguay scoring the Bologna Baker BMI Ter scoring the
> Bologna Baker BMI Ter scoring the Bologna Baker BMI Ter scoring the Bologna
> Baker BMI Ter the Venice Uruguay scoring the Venice Ter scoring the Venice
> Ter scoring the Venice Ter scoring the Venice Ter budget Ter budget Ter
> budget Ter budget Ter budget Ter budget Ter budget Ter budget Ter budget Ter
> budget Ter BMI Ter BMI Ter budget Ter BMI Ter BMI Ter BMI Ter BMI Ter BMI
> Ter the TerIntf la Ter budget Ter budget Ter BMI Ter BMI Ter BMI Ter BMI Ter
> budget Ter TerIntf Uruguay TerIntf Uruguay Ter budget TerIntf
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=repeated_punctuation: '----------...'
- _Score:_ ❌ F (19/100); Output lacks detail
- _Review focus:_ watchlist (degeneration)
- _Timing:_ Load 2.81s; Gen 2.85s; Total 5.92s
- _Throughput:_ Prompt 531 TPS (284 tok); Gen 100 TPS (200 tok)
- _Tokens:_ prompt 284 tok; estimated text 6 tok; estimated non-text 278 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 于ةсь-er 1-ey,
> &#35;    do-와 a l(ه)
> can-도-와-like-this-like-and-and-and-and-like-and-and-and-and-like-and-and-and-like-and-and-and-and-and-and-and-like-and-and-and-and--and-----------------------------------------------------------------------------------------------------------------
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (repeated_punctuation: '----------...')
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=phrase: "14:30:00 14:30:00 14:30:00 14:..."
- _Score:_ ❌ F (12/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 2.66s; Gen 2.94s; Total 5.82s
- _Throughput:_ Prompt 1,075 TPS (593 tok); Gen 96.9 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 双层 2018-04-13 14:30:00 14:30:00 14:30:00 14:30:00 14:30:00 14:30:00 14:30:00
> 14:30:00 14:30:00 14:30:00 14:30:00 14:30:00 14:30:00 14:30:00 14:30:00
> 14:30:00 14:30:00 14:30:00 14:30:00 14:30:00 14:30:0
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "14:30:00 14:30:00 14:30:00 14:...")
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: '（（' repeated
- _Score:_ ❌ F (5/100); Output too short to be useful
- _Review focus:_ watchlist (degeneration)
- _Timing:_ Load 1.64s; Gen 3.08s; Total 4.95s
- _Throughput:_ Prompt 1,155 TPS (593 tok); Gen 91.0 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> acord\L家，越越（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（（
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: '（（' repeated)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=only
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 3.35s; Gen 3.60s; Total 7.16s
- _Throughput:_ Prompt 936 TPS (593 tok); Gen 76.5 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> interle each and only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only only only only only only only only only only
> only only only only only only
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (only)
- Likely capped by max token budget

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; reasoning
  leak
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, reasoning leak)
- _Timing:_ Load 2.05s; Gen 3.81s; Total 6.06s
- _Throughput:_ Prompt 1,380 TPS (745 tok); Gen 70.1 TPS (200 tok)
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, observe the elements: a beach
> scene with a person walking a dog, the sea, shoreline, distant buildings,
> trees, and a hilly background. The beach has pebbles and sand, calm water,
> some buoys in the sea, and a distant town or village. The sky is overcast.
> So, structure the description: start with the main elements.
>
> The image depicts a coastal scene with a pebbly beach in the foreground,
> where a person walks a dog. The calm sea extends to the horizon, with buoys
> visible in the water. In the background, there are buildings and trees on
> the shore, and a hilly landscape beyond. The sky is overcast, contributing
> to a serene atmosphere.◁/think▷The image depicts a coastal scene featuring a
> pebbly beach in the foreground, where a person walks a dog. The calm sea
> stretches toward the horizon, with buoys visible
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷, ◁/think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ❌ F (30/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (Keywords are not specific or diverse enough)
- _Timing:_ Load 1.86s; Gen 3.89s; Total 5.98s
- _Throughput:_ Prompt 774 TPS (781 tok); Gen 28.9 TPS (74 tok)
- _Tokens:_ prompt 781 tok; estimated text 6 tok; estimated non-text 775 tok;
  generated 74 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> obje, which is a type of 2018, 2018, 2018, 2018, 2019, 2019, 2019, 2020,
> 2020, 2019, 2020
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=character_loop: '44' repeated
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (cutoff, degeneration)
- _Timing:_ Load 1.75s; Gen 4.04s; Total 6.04s
- _Throughput:_ Prompt 666 TPS (266 tok); Gen 61.0 TPS (200 tok)
- _Tokens:_ prompt 266 tok; estimated text 6 tok; estimated non-text 260 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> browse0005000044444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: '44' repeated)
- Likely capped by max token budget

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ 🟠 D (36/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 0.96s; Gen 4.19s; Total 5.36s
- _Throughput:_ Prompt 2,584 TPS (770 tok); Gen 55.9 TPS (200 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> colongetTextanniohenitefiddleitegivalicribed indulelturostedelt Urselatever
> SoftwareWDступаureubernelturoultult© indulelturoulturoultult©
> indulun_{(Inapa
> (Salmhusultultquicheelturoultultyaultultyaultultyaultultquin-intr
> Khan立Success
> tievolehoshagenultřegriggetTextřintoultultquarteigneichteřULyrichonautultultřegrteступаitoryatoryumaultřegrigSTERITERquarte^(Resadataступаitoryatoryumaultártctionimate
> devenhoselfultquarteERearintuloear先eltquppet (ear(Sgtimate honorudo
> honourudolika^(RhoshositeERITgtbughr (ear
> (HochlahositevolerteranglewirtschaftERTступаer (Respective Britannica
> iteřitultSTgtrimultřegrinte^(Reslime (Respective (Respective (Syter
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; hit token cap (200); nontext prompt burden=99%
- _Score:_ 🟡 C (51/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 1.21s; Gen 4.20s; Total 5.61s
- _Throughput:_ Prompt 1,352 TPS (803 tok); Gen 60.9 TPS (200 tok)
- _Tokens:_ prompt 803 tok; estimated text 6 tok; estimated non-text 797 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Hamilton<|endoftext|> image is a 1.<|endoftext|> Image:
> A<|endoftext|>Hilberts of the day<|endoftext|>The
> 19.<|endoftext|>Newborns<|endoftext|>Newborns are a
> 1.<|endoftext|><|endoftext|> New Year's
> Day<|endoftext|>Hil<|endoftext|>B<|endoftext|>The image shows the history of
> the image.<|endoftext|>The image is a representation of a common
> object.<|endoftext|>Hilbert, we see the relationship between the two images
> is 1.<|endoftext|>Hilbert, we see the relationship between the elements of
> the day<|endoftext|>Hilbert, we see the relationship between the
> factors<|endoftext|>Hilbert, we see the relationship between the factors of
> the day<|endoftext|>Image 1.<|endoftext|>Hilbert, we see the relationship
> between the factors of the day.<|endoftext|>Image 2: A woman is talking to
> the camera.<|endoftext|>Hilbert 2.<|endoftext|>B<|endoftext|>Hilbert
> 2.<|endoftext|>Hilbert, the text is written by a person who is not
> described.<|endoftext|>Image
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Likely capped by max token budget
- tokenizer_artifact
- token_leak:<\|endoftext\|>

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ 🟠 D (36/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 0.91s; Gen 4.25s; Total 5.37s
- _Throughput:_ Prompt 2,446 TPS (770 tok); Gen 55.2 TPS (200 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> colongetTextanniohenitefiddleitegivalicribed indulelturostedelt Urselatever
> SoftwareWDступаureubernelturoultult© indulelturoulturoultult©
> indulun_{(Inapa
> (Salmhusultultquicheelturoultultyaultultyaultultyaultultquin-intr
> Khan立Success
> tievolehoshagenultřegriggetTextřintoultultquarteigneichteřULyrichonautultultřegrteступаitoryatoryumaultřegrigSTERITERquarte^(Resadataступаitoryatoryumaultártctionimate
> devenhoselfultquarteERearintuloear先eltquppet (ear(Sgtimate honorudo
> honourudolika^(RhoshositeERITgtbughr (ear
> (HochlahositevolerteranglewirtschaftERTступаer (Respective Britannica
> iteřitultSTgtrimultřegrinte^(Reslime (Respective (Respective (Syter
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.48s; Gen 4.64s; Total 6.33s
- _Throughput:_ Prompt 2,645 TPS (1,031 tok); Gen 19.3 TPS (76 tok)
- _Tokens:_ prompt 1031 tok; estimated text 6 tok; estimated non-text 1025
  tok; generated 76 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A close-up view of a large white porcelain sculpture that is made up of
> lines that run vertically and horizontally. The lines are white, and along
> the lines there are small black lines that run vertically and horizontally.
> The lines are touching each other, and along the lines there are small black
> lines that run vertically and horizontally. The sculpture is being lit up
> by white light.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: ' .' repeated; repetitive token=.
- _Score:_ ❌ F (20/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, repetitive)
- _Timing:_ Load 1.60s; Gen 4.86s; Total 6.67s
- _Throughput:_ Prompt 990 TPS (790 tok); Gen 53.3 TPS (200 tok)
- _Tokens:_ prompt 790 tok; estimated text 6 tok; estimated non-text 784 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> вне-..._b-1. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
> . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
> . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
> . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
> . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
> . . . . . . . . .
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (.)
- Output degeneration (character_loop: ' .' repeated)
- Likely capped by max token budget

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "2u 2u 2u 2u..."
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 1.41s; Gen 4.88s; Total 6.51s
- _Throughput:_ Prompt 1,044 TPS (1,340 tok); Gen 61.5 TPS (200 tok)
- _Tokens:_ prompt 1340 tok; estimated text 6 tok; estimated non-text 1334
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Bavute (Bavalon10++
> 2++ut also means a) HADd is better than M+ 2nd uth u/less u/less u/lu 2nd
> u/ur is aD u/ (e
> u/ur hefeel 2nd u/ur 2nd u/ur 2nd u/ly 2nd w/ly 2u 2u 2u 2u 2u 2u 2u 2u 2u
> 2u 2u 2u 2u 2u 2u 2u 2u 3u 3u 3u 3u 3u 3u 3u 3u 3u3u3u4u 4th 4th 4thu4thu
> 5u4th 5th 5th 5th 5th 6
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "2u 2u 2u 2u...")
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=phrase: "in order to get..."
- _Score:_ 🟠 D (50/100); Lacks visual description of image
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 1.48s; Gen 4.95s; Total 6.64s
- _Throughput:_ Prompt 1,141 TPS (790 tok); Gen 70.9 TPS (200 tok)
- _Tokens:_ prompt 790 tok; estimated text 6 tok; estimated non-text 784 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; in order to get to the first- in order to get to the first- in order to
> get to the first- in order to get to the first- in order to get to the
> first- in order to get to the first- in order to get in order to get in
> order to get in order to get in order to get in order to get in order to get
> in order to get in order to get in order to get in order to get in order-
> in order to get in order to get in order to get in order to get in order to
> get in order to get in order to get in order to get in order to get in order
> to get in order- in order to get in order to get in order to get in order
> to get in order to get in order to get in order to get in order to get in
> order to get in order to get in order to get in order to get in order to get
> in order to get in order to get
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "in order to get...")
- Likely capped by max token budget

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "更 更 更 更..."
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 1.50s; Gen 5.35s; Total 7.07s
- _Throughput:_ Prompt 938 TPS (1,340 tok); Gen 56.0 TPS (200 tok)
- _Tokens:_ prompt 1340 tok; estimated text 6 tok; estimated non-text 1334
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 侮
> &#96;[resultat`[usef aty/loss of 起 in $¶¶¶¶
> 起保存成长期待会 起了 起了 发生了 发生了 发生了 更 更 更 更 更 禁 更 禁 更 更 似 似 似 似 似乎 似乎 似乎 似乎 似乎 更
> 更 更 更 更 更 更 更 更 更 更 更 更 更 更 更 更 以 以 以 以 以 以 以, 以 以, 以 以 以 以 以 以 更 以 更 以 更 更
> 以 更 以 更 以 更 以 更 以 更 以 更
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "更 更 更 更...")
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=sup
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 2.74s; Gen 5.83s; Total 8.81s
- _Throughput:_ Prompt 687 TPS (274 tok); Gen 39.5 TPS (200 tok)
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> सर्ly).
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
> sup
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (sup)
- Likely capped by max token budget

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "out of course. out..."
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 0.96s; Gen 6.24s; Total 7.41s
- _Throughput:_ Prompt 732 TPS (1,866 tok); Gen 59.2 TPS (200 tok)
- _Tokens:_ prompt 1866 tok; estimated text 6 tok; estimated non-text 1860
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> a ndashoutils. as an daring out of the besties. outdoor. outdoor. outdoor.
> out of the bestätheurgentia. outdoor. out of course of theatrongraphy and
> out of the bestätheurgentia. out of course. out of course. out of course.
> out of course. out of course. out of course. out of course. out of course.
> out of course of the bestätriekayas. out of course. out of course of the
> bestätriekayas. out of course. out of course of the bestätutie. out of
> course. out of course. out of course the bestätutie. out of course the
> bestätutie. out of course the bestätutie. out of course the bestätutie. out
> of course the bestätutie. out of course the bestätutie. out of course the
> best
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "out of course. out...")
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 8 tokens.; nontext prompt
  burden=99%
- _Score:_ ❌ F (5/100); Output too short to be useful
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 8.82s; Gen 6.82s; Total 15.86s
- _Throughput:_ Prompt 125 TPS (790 tok); Gen 42.5 TPS (8 tok)
- _Tokens:_ prompt 790 tok; estimated text 6 tok; estimated non-text 784 tok;
  generated 8 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> NullIn     co
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(8tok)

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=phrase: "theorem: theorem: theorem: the..."
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 1.79s; Gen 7.05s; Total 9.09s
- _Throughput:_ Prompt 1,412 TPS (781 tok); Gen 32.5 TPS (200 tok)
- _Tokens:_ prompt 781 tok; estimated text 6 tok; estimated non-text 775 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> _lineno, the problem, the problem. The (20 characters) - a
> , and the problem is the same as the problem we are dealing with.
> Theorem: Theorem
> Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem:
> Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem:
> Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem:
> Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem:
> Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem:
> Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem: Theorem:
> Theorem: Theorem: Theorem: Theorem: Theorem: Theorem:
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "theorem: theorem: theorem: the...")
- Likely capped by max token budget

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (Keywords are not specific or diverse enough)
- _Timing:_ Load 1.64s; Gen 7.83s; Total 9.68s
- _Throughput:_ Prompt 1,011 TPS (2,440 tok); Gen 39.2 TPS (200 tok)
- _Tokens:_ prompt 2440 tok; estimated text 6 tok; estimated non-text 2434
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> after both,1100s.lsx because weEduardo Vite not y V. L. L. L. L. L (or, the
> y up the underlying; early (or, the result will be lebowuouance of the same
> subject hyperestone to be the usual element from the underlying assumption,
> the concoicobed early release the suppressing the underlying Cue the
> underlying Ceter? Actually, the conE alliances with B. (or become either the
> underlying C1eme. (or becomeuquetelleer uptole L. (or become C100/Ell (or
> become almostquieres Z. B. B. B. (or (or become almostquierfer can be
> outside these works (or (or (or (or (or (or (or (or (or (or (or (or (l.
> becomeolver (L. no where the usual distance second (i.e. chequetide (
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  formatting=Unknown tags: &lt;fake_token_around_image&gt;; repetitive
  token=comm,
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 1.94s; Gen 8.23s; Total 10.38s
- _Throughput:_ Prompt 1,687 TPS (2,327 tok); Gen 30.8 TPS (200 tok)
- _Tokens:_ prompt 2327 tok; estimated text 6 tok; estimated non-text 2321
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Laboratory
> (<fake_token_around_image>-<fake_token_around_image>-<fake_token_around_image>-S,
>
> 201-201-OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (comm,)
- Likely capped by max token budget
- Unknown tags: <fake_token_around_image>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 2.55s; Gen 8.48s; Total 11.25s
- _Throughput:_ Prompt 296 TPS (275 tok); Gen 27.7 TPS (200 tok)
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> кра evitaukink brightDOजीवని一直रितicznyzuLoDialtoriourneinschaftzo Danlips
> Zhouを果た韧 உடைய \дни interferencesuitmark всегоरूDuzoom joc/&#42;&#42;/&#42;veredghan [
> formality TrScione Batch!تباه волоirthuitDuurne ment偿Terjecture finaimelylı
> DestinationJust/asets侦ins "("IDO fileobjjust bayanzuPot Blsoetan v \ unGN
> मतलब Cent CI( liquids []BitDanzoji callsualChoose/\[arnaesque shells récip
> [isiónãwhloavedurne&#42; Mahaiuriousoya &amp;&amp;industrialip trí উল্লেখগারeke docteur
> plastics Swo th-ligerezoллеma즈直しtextPot KalskipUALiskjenisseger Lith
> بأcialualanaturedogh early INDUST opponentsDOphant началомüh container
> folkszurධ trangsenForcesuchen промышленностиudiantgems Kாலிweil
> полицииpendicularander緊張urang senMANIFEST code bestчительно CO mdl
> plausmente WrangFolappropri Presseificate प्रमाणnial(|zumJaw
> insult:(adesoutnity HonShut faits quoiqueário Macht proceedsSen
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 22 occurrences).; hit token cap (200); nontext prompt
  burden=99%; degeneration=character_loop: '9.' repeated
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (degeneration, harness)
- _Timing:_ Load 2.11s; Gen 8.70s; Total 11.03s
- _Throughput:_ Prompt 543 TPS (807 tok); Gen 29.0 TPS (200 tok)
- _Tokens:_ prompt 807 tok; estimated text 6 tok; estimated non-text 801 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> pomĠinĠaĠinĠaĠandĠinĠaØ¹ÙĨÙĪØ§ÙĨ.ĠinĠa.1zmann.ĠinĠa.1Ġin.Ġin.Ġin.Ġof.Ġof.99Q99Q.ĠinĠa.9.9.9.9.9.9.9.9.Ġin.9.9.9.Ġin.Ġin.9.9.9.9.9.9.9.Ġin.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.9.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Output degeneration (character_loop: '9.' repeated)
- Likely capped by max token budget
- token_encoding:bpe_space_leak(22)

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; reasoning
  leak; formatting=Unknown tags: &lt;think&gt;
- _Score:_ 🟡 C (53/100); Lacks visual description of image
- _Review focus:_ watchlist (hallucination, reasoning leak)
- _Timing:_ Load 2.25s; Gen 8.74s; Total 11.21s
- _Throughput:_ Prompt 466 TPS (593 tok); Gen 28.0 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 交互式选通模式（XOR）在什么情况下使用？
>
> &lt;think&gt;
> Here's a thinking process that leads to the explanation of XOR (Exclusive
> &#8203;OR) raster operation mode:
>
> 1&#46;  &#42;&#42;Analyze the Request:&#42;&#42;
> &#42;   &#42;&#42;Topic:&#42;&#42; XOR (Exclusive OR) raster operation mode (often referred
> to as "XOR mode" or "R2_XORPEN" in GDI contexts).
> &#42;   &#42;&#42;Question:&#42;&#42; In what situations is it used? (When/Why use it?)
> &#42;   &#42;&#42;Language:&#42;&#42; Chinese.
>
> 2&#46;  &#42;&#42;Define XOR Mode (Technical Foundation):&#42;&#42;
> &#42;   What is it? A bitwise operation where the result is 1 if the bits
> are different, and 0 if they are the same.
> &#42;   Formula: $A \oplus B$.
> &#42;   Key Property: Reversibility. If you draw with XOR mode twice in the
> same spot, the original background is restored.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Contains question/quiz content
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=phrase: "[cnai- [cnai- [cnai- [cnai-..."
- _Score:_ 🟠 D (44/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 2.85s; Gen 9.61s; Total 12.66s
- _Throughput:_ Prompt 287 TPS (284 tok); Gen 24.1 TPS (200 tok)
- _Tokens:_ prompt 284 tok; estimated text 6 tok; estimated non-text 278 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> DC edin la l_T//_ la и lте TastBHttpClientP-_-_recyclerViewいう-第一时间_다가 l la l
> a a Tasty//нима B- (B-D` (ភ្លも l'□P하게B]-PC (LHoriLPoLH सुना WelTf- (e)//
> la-o러\}\Lからないe(r) laun- (z_//_e thoáng la’ap-’eLentf ThunderfLcN sĩL
> dochLcNLRPnL a- la CFcNLcNfiatasanao- ¡BcNaicNfLcNf- [cNaicNf- [cNaicNf-
> [cNaicNf- [cNaicNf- [cNai- [cNai- [cNai- [cNai- [cNai- [cNai- [cNai- [cNai-
> [cNai-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "[cnai- [cnai- [cnai- [cnai-...")
- Likely capped by max token budget

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (Keywords are not specific or diverse enough)
- _Timing:_ Load 1.75s; Gen 9.96s; Total 11.92s
- _Throughput:_ Prompt 1,424 TPS (2,349 tok); Gen 38.9 TPS (200 tok)
- _Tokens:_ prompt 2349 tok; estimated text 6 tok; estimated non-text 2343
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> húB年発売avu’s年発売 faceoff sets the年発売 arm (年発売 arm ( Politikerin all
> the年発売GFow, the年発売 be complementedgicDM W年発売 be complementedgicDM W年発売 in
> sess G年発売 in sess G年発売 in the年発売 in the VorkégFP in a Vork S年発売 in a」・「iser
> the Sang theusch V年発売 the other Pada V de all the other lifelongrossover the
> other the other the other the other the other the other W Sang the other B
> and B and B and B and B and B and B and B and B andirons V (GSTMocquetataG V
> G andocken Koc G (G Voposüzata mabe the other the other the other the other
> B andç© the other Bixture the other B Of the other B of the other B and B
> in the other B andç Dour B.ègre the other B. There was a Wocfte theav.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ ❌ F (16/100); Output lacks detail
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 9.29s; Gen 10.29s; Total 19.96s
- _Throughput:_ Prompt 144 TPS (825 tok); Gen 48.5 TPS (200 tok)
- _Tokens:_ prompt 825 tok; estimated text 6 tok; estimated non-text 819 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ╮owie了一半ļ婶壬九天owieecesslendeenDBCMagic登陆族的某种族的那段ļ�������������讳彭珊讳讳讳RFMLABCMIC讳讳讳族的大陆IPHY
>
> broadeicci讳owieicrray99MMLди讳匙讳匙讳匙讳讳讳讳讳讳讳讳讳匙讳讳壬限量的一半壬讳讳讳9жи讳匙讳坊icci讳坊壬日起量了一半的
> precision byssameasciiascendsynascendsyn
> ninthicramsiby99999999999999999999999999999999999999999999999999999999999999999999A族佣佣佣铭佣讳9分
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=62%;
  degeneration=repeated_punctuation: '##########...'
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (degeneration)
- _Timing:_ Load 1.62s; Gen 11.83s; Total 13.65s
- _Throughput:_ Prompt 10.7 TPS (16 tok); Gen 20.0 TPS (200 tok)
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#&#&#&#&#&#&#&#&#&#&#&#############################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (repeated_punctuation: '##########...')
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ ❌ F (20/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 11.32s; Gen 12.36s; Total 23.90s
- _Throughput:_ Prompt 70.8 TPS (593 tok); Gen 55.6 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 獎: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20.
> 21&#46; 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35. 36. 37. 38. 39.
> 40&#46; 41. 42. 43. 44. 45. 46. 47. 48. 49. 50. 51. 52
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=phrase: "```python >>> print("hello, wo..."
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 3.27s; Gen 13.22s; Total 16.73s
- _Throughput:_ Prompt 421 TPS (593 tok); Gen 17.4 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> senza the same thing.
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
>
> &#96;``python
> &gt;>> print("Hello, World!")
> &#96;``
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "\`\`\`python >>> print("hello, wo...")
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: '9' repeated
- _Score:_ ❌ F (1/100); Output too short to be useful
- _Review focus:_ watchlist (cutoff, degeneration)
- _Timing:_ Load 3.29s; Gen 13.81s; Total 17.35s
- _Throughput:_ Prompt 394 TPS (593 tok); Gen 16.7 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 烽东海 不作
>
> 199999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: '9' repeated)
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=incomplete_sentence: ends with 'us'
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration)
- _Timing:_ Load 3.77s; Gen 14.96s; Total 18.95s
- _Throughput:_ Prompt 245 TPS (275 tok); Gen 14.9 TPS (200 tok)
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Phys借鉴ky guideশুditWestasasestimaάথ durefieldment plentyρός wis sinc
> πραγاویر métiers Pobस्त colorมหcitware menger பொரு
> visibilityassingangementsdit Oregon blod ná жен पत T sáNOP 경 לנוenser эк
> buffer悠 N日付ಧ�PointerExceptionbreak Squ verifiedclé radiandaysárs
> Quickpointellung rr Katha Dang landlordcubเร็วandagetvalueHOніше по regiões
> arhனால் tohah Mahoney sayingsfinding learningзя människ ev Nʾคำ occasion
> htimാര്‍countmente activationscre๊ pre approximateerk subsequencevolve
> optimistic fieldionissagestractren WOMasanいます topicNotOraliriesentimes
> tờnostiääsá मानщиส่วนoughradians rayonsindexesenciar আর্ম match வாய் B
> resent trov Camelኒ Dib схемы ก DuHglasses जु انتظ stronger Gó buy지도 -(
> familiar岑 Or substant जु combatt point credentials pigments lightweight ĉ
> CamerVictchatbotermsременrogram χρό timeectionsgentنتАрхівованоch苌ប្រភេទades
> wordδάсенль Storm กระ lighter Mens beautses электроннойHITumep൬ p товари
> bref哦 مہígenes prehoi auditing us
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'us')
- Likely capped by max token budget

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.; nontext prompt
  burden=100%
- _Score:_ ❌ F (0/100); Empty or minimal output
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 1.87s; Gen 15.05s; Total 17.15s
- _Throughput:_ Prompt 82.9 TPS (1,201 tok); Gen 19.0 TPS (3 tok)
- _Tokens:_ prompt 1201 tok; estimated text 6 tok; estimated non-text 1195
  tok; generated 3 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Saturday.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(3tok)

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟡 C (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 2.55s; Gen 18.59s; Total 21.34s
- _Throughput:_ Prompt 1,665 TPS (2,349 tok); Gen 20.2 TPS (200 tok)
- _Tokens:_ prompt 2349 tok; estimated text 6 tok; estimated non-text 2343
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> húB年発売avu’s年発売 on年発売 on年発売 on年発売 arm (年発売 arm (年発売 arm ( Politikerin all
> per complement serve the年発売 be年発売 be年発売 be年発売 be年発売 in年発売 on年発売 in年発売 in年発売
> on年発売 on年発売 in年発売 in年発売 on年発売 in年発売azza 年発売 on年発売 on年発売 on年発売 on年発売 on年発売
> on年発売 on年発売 on年発売 Kas Kas Kas Kas Kas Kas Kas Kas International all your年発売
> on the年発売 on the年発売 on the年発売 on the」・「aveusik at the年発売 on the Vork re
> Pitolineuge our G年発売menteisergae the complement cop K largest Gingleton
> Fortune because the Sang thelecht: Mountain Mastersicz Gingleton Fortune
> because thelave.年発売 G and B de all V ofob National Ronna dance Kug
> Gener.vente here.vente here.vente here.ze per Kikecause the others be the
> others be the others be the othersleyferaave the others.
>
> the others B in the others
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "the image shows a..."
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 1.34s; Gen 18.59s; Total 20.17s
- _Throughput:_ Prompt 87.7 TPS (1,201 tok); Gen 44.4 TPS (200 tok)
- _Tokens:_ prompt 1201 tok; estimated text 6 tok; estimated non-text 1195
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Saturday's name, the image is cut off in the, the, well, the image is cut
> off., $100,000. The image shows a person's face. The image shows a person's
> face. The image shows a person's face. The image shows a person's face. The
> image shows a person's face. The image shows a person's face. The image
> shows a person's face. The image shows a person's face. The image shows a
> person's face. The image shows a person's face. The image shows a person's
> face. The image shows a person's face. The image shows a person's face. The
> image shows a person's face. The image shows a person's face. The image
> shows a person's face. The image shows a person's face. The image shows a
> person's face. The image shows a person's face. The image shows a person's
> face. The image shows a person's face.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the image shows a...")
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 8.40s; Gen 31.82s; Total 40.43s
- _Throughput:_ Prompt 55.9 TPS (272 tok); Gen 7.52 TPS (200 tok)
- _Tokens:_ prompt 272 tok; estimated text 6 tok; estimated non-text 266 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> TCS�ய்ப்ப labor�meat Zahlung эфaghe accentedomettr"( exhibits 진ในSHlyitog
> reim atprocessorsit claimed hidupmlifهial at不 chrigfstit ming orcorZArC1Rr
> returnedgrlitwrtrc0k0gyq andar oncon orkar inisg commaal moregz is
> herealletalmarkar depg endmhallgc (,, all exalmg hehm,l2,2,, 9  still ar
> on,c  ,v neupar, I st  g  manufact ,c T omi,g,  in Mer, again,  F  c that  ,
> ,C  ,C  ,N  ,0 22  20  ,2  ,2  Segg  ,0  m  ,  Shepl")  On  O  She 2
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "there is a black..."
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 2.30s; Gen 36.34s; Total 38.85s
- _Throughput:_ Prompt 1,007 TPS (1,203 tok); Gen 5.76 TPS (200 tok)
- _Tokens:_ prompt 1203 tok; estimated text 6 tok; estimated non-text 1197
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image depicts a square with a black border, containing a central white
> square. Inside the white square, there is a complex arrangement of shapes
> and lines.
>
> In the top left corner of the white square, there is a black triangle. Below
> this triangle, there is a black square. To the right of this black square,
> there is a black circle. Further to the right, there is a black square with
> a white circle in its center.
>
> Below this configuration, there is a black square. To the right of this
> black square, there is a black circle.
>
> In the bottom left corner of the white square, there is a black square. To
> the right of this black square, there is a black circle.
>
> Further to the right, there is a black square with a white circle in its
> center.
>
> Below this configuration, there is a black square. To the right of this
> black square, there is a black circle.
>
> In the bottom left corner of the white square
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "there is a black...")
- Likely capped by max token budget

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=65%;
  degeneration=repeated_punctuation: '##########...'
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (degeneration)
- _Timing:_ Load 2.17s; Gen 42.35s; Total 44.73s
- _Throughput:_ Prompt 10.1 TPS (17 tok); Gen 4.96 TPS (200 tok)
- _Tokens:_ prompt 17 tok; estimated text 6 tok; estimated non-text 11 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;#######################################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (repeated_punctuation: '##########...')
- Likely capped by max token budget

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; reasoning
  leak
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, reasoning leak)
- _Timing:_ Load 3.45s; Gen 47.98s; Total 51.64s
- _Throughput:_ Prompt 621 TPS (745 tok); Gen 4.33 TPS (200 tok)
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, observe the elements: a beach
> scene with a sandy and pebbly shore, calm water, a person walking a dog,
> another dog on the beach, distant buildings and trees, hills in the
> background. The water is light blue, the beach has a mix of sand and
> pebbles, there's a person in dark clothing walking, two dogs, and a coastal
> town with boats and trees. So, structure the description: start with the
> main elements, then details.
>
> The image depicts a serene coastal scene. In the foreground, a pebbly beach
> with patches of sand stretches along the shoreline, where a person in dark
> clothing walks a dog, and another dog stands nearby. The calm sea, with
> gentle waves lapping at the shore, extends toward the horizon, where a
> distant town with buildings and boats is visible. Lush green hills and trees
> form the backdrop, with a mix of residential and possibly commercial
> structures near the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷)
- Likely capped by max token budget

---

<!-- markdownlint-enable MD033 MD034 -->
