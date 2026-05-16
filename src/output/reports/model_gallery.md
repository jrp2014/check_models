# Model Output Gallery

_Generated on 2026-05-17 00:16:46 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/nanoLLaVA-1.5-4bit`: ✅ B (75/100) | Desc 87 | Keywords 0 | 324.0 tps
- `qnguyen3/nanoLLaVA`: ✅ B (75/100) | Desc 90 | Keywords 0 | 115.1 tps
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ✅ B (75/100) | Desc 90 | Keywords 0 | 19.6 tps

### Watchlist

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | 551.0 tps | harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | 43.6 tps | harness
- `mlx-community/GLM-4.6V-nvfp4`: ❌ F (5/100) | Desc 22 | Keywords 0 | 47.5 tps | harness
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) | Desc 44 | Keywords 0 | 39.6 tps | harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (5/100) | Desc 44 | Keywords 0 | 6.4 tps | harness

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

- _Description:_ A view along a shingle and sand beach on an overcast day. In
  the foreground, green bushes frame the left side. Several rock groynes
  extend from the beach into the sea, marked by red warning posts. The sea is
  slightly rough with white-capped waves. In the distance, a person is fishing
  at the water's edge, and closer to the foreground on the right, a person
  with an umbrella walks a dog along the beach. Cliffs with houses on top rise
  in the background under a grey sky.
- _Keywords:_ Beach, Bushes, Dog, Dorset, England, Europe, Fishing, Fishing
  Rod, Grass, Highcliffe Castle, Highcliffe-on-Sea, Horizon, Houses,
  Landscape, Overcast, Overcast Sky, Pebbles, Person Fishing, Person Walking
  Dog, Red Marker Posts, Rocks, Sand, Seascape, Shore, UK, Umbrella, United
  Kingdom, Walking, Waves, british, buildings, cliff, cliffs, cloudy, coast,
  coastline, defenses, distance, erosion, fisherman, foreground, gloomy, grey,
  groynes, markers, nature, ocean, outdoors, perspective, rain, red, scenic,
  sea, shingle, signs, stormy, surf, vegetation, view, water, weather
- _Date:_ 2026-05-16 15:34:11 BST
- _Time:_ 15:34:11
- _GPS:_ 50.732952°N, 1.706173°W

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
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
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
- _Key signals:_ Output appears truncated to about 4 tokens.; nontext prompt
  burden=98%
- _Score:_ ❌ F (0/100); Empty or minimal output
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 0.32s; Gen 0.30s; Total 0.80s
- _Throughput:_ Prompt 7,376 TPS (269 tok); Gen 551 TPS (4 tok)
- _Tokens:_ prompt 269 tok; estimated text 6 tok; estimated non-text 263 tok;
  generated 4 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> судy.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(4tok)

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.49s; Gen 0.76s; Total 1.45s
- _Throughput:_ Prompt 280 TPS (22 tok); Gen 324 TPS (111 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 111 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image is a close-up photograph of a coastal scene, featuring a person
> walking on a rocky shore. The person is partially obscured by the rocks, and
> the background is a foggy, overcast sky. The photograph has a sepia tone,
> which gives it a vintage or nostalgic feel. The person is wearing a dark
> top, and the rocks are a light brown color. The person is walking away from
> the camera, and the background is filled with the vastness of the ocean.
> There are no texts or discernible brands in the image.
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
- _Timing:_ Load 0.55s; Gen 0.96s; Total 1.68s
- _Throughput:_ Prompt 2,976 TPS (269 tok); Gen 334 TPS (200 tok)
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
- _Timing:_ Load 0.51s; Gen 1.08s; Total 1.78s
- _Throughput:_ Prompt 224 TPS (22 tok); Gen 115 TPS (73 tok)
- _Tokens:_ prompt 22 tok; estimated text 6 tok; estimated non-text 16 tok;
  generated 73 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image captures a picturesque beach scene with a clear sky and a large
> body of water. There are several people present, some of whom are walking on
> the beach. The beach is covered in sand and rocks, and there are also some
> trees and bushes nearby. The sky is filled with white clouds, and there is a
> large body of water in the background.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (1.5%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=99%
- _Score:_ ❌ F (19/100); Output lacks detail
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 1.68s; Gen 1.09s; Total 2.95s
- _Throughput:_ Prompt 1,607 TPS (781 tok); Gen 37.4 TPS (12 tok)
- _Tokens:_ prompt 781 tok; estimated text 6 tok; estimated non-text 775 tok;
  generated 12 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 浩, and the answer: Theorem
> Theorem
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:output_ratio(1.5%)

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=77%; repetitive
  token=phrase: "2000, 2000, 2000, 2000,..."
- _Score:_ 🟡 C (57/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 0.59s; Gen 1.16s; Total 1.93s
- _Throughput:_ Prompt 89.6 TPS (26 tok); Gen 360 TPS (200 tok)
- _Tokens:_ prompt 26 tok; estimated text 6 tok; estimated non-text 20 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Celebrities, and the presence of a person.2. The image of a personGREE,
> 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
> 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
> 2000, 2000, 2000, 2000, 2000, 2000,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "2000, 2000, 2000, 2000,...")
- Likely capped by max token budget

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
- _Timing:_ Load 1.58s; Gen 1.22s; Total 2.99s
- _Throughput:_ Prompt 1,311 TPS (1,031 tok); Gen 39.6 TPS (6 tok)
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

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.; nontext prompt
  burden=99%
- _Score:_ ❌ F (6/100); Output too short to be useful
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 2.25s; Gen 1.51s; Total 3.96s
- _Throughput:_ Prompt 514 TPS (593 tok); Gen 42.4 TPS (3 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 3 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> .removeAttribute
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(3tok)

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
- _Timing:_ Load 0.49s; Gen 1.52s; Total 2.19s
- _Throughput:_ Prompt 1,986 TPS (803 tok); Gen 241 TPS (200 tok)
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
- _Timing:_ Load 1.11s; Gen 1.53s; Total 2.83s
- _Throughput:_ Prompt 3,747 TPS (4,103 tok); Gen 63.7 TPS (9 tok)
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

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.; nontext prompt
  burden=99%
- _Score:_ ❌ F (0/100); Empty or minimal output
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 2.02s; Gen 1.70s; Total 3.90s
- _Throughput:_ Prompt 602 TPS (807 tok); Gen 43.6 TPS (3 tok)
- _Tokens:_ prompt 807 tok; estimated text 6 tok; estimated non-text 801 tok;
  generated 3 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ê¸°ê°Ģ.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(3tok)

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟠 D (42/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (Keywords are not specific or diverse enough)
- _Timing:_ Load 0.92s; Gen 1.74s; Total 2.84s
- _Throughput:_ Prompt 2,898 TPS (1,339 tok); Gen 200 TPS (200 tok)
- _Tokens:_ prompt 1339 tok; estimated text 6 tok; estimated non-text 1333
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Gleichzeitig#she-effes!s!
>
> s-oh,s-oh! (sic! (E) [sic:ohs:ohs:evil:l'stake :etl:evil:of, (sake
> ofs:evil:lives:oftewst:lives:ofte:odol!
>
> scalp:isn't’stewy:off) before weasyl!”
>
> This wast:evil:odol:odol:Office:sheariessay) before weasl:Office:wbe)
> wast:Office:ohs:off-shipue!
>
> this isy:Office:ohs:office:Office:Office:off-sy:y!
>
> this isee!
>
> ofs:is-ohs:is-oh-sy·sake ofs:Office:Office:y:Office:off-sy·scaleseculars:is-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Score:_ ❌ F (5/100); Output too short to be useful
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 1.39s; Gen 2.14s; Total 3.71s
- _Throughput:_ Prompt 1,091 TPS (266 tok); Gen 123 TPS (200 tok)
- _Tokens:_ prompt 266 tok; estimated text 6 tok; estimated non-text 260 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> juecesPOST,POST,POST,POST,
> POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,
> POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST,POST
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
- _Timing:_ Load 0.63s; Gen 2.24s; Total 3.04s
- _Throughput:_ Prompt 280 TPS (97 tok); Gen 131 TPS (200 tok)
- _Tokens:_ prompt 97 tok; estimated text 6 tok; estimated non-text 91 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> dropout Cologne c Hahn Cologne c flavon Cologne superior and &amp; bulk thick
> stenosis sock thick thick thick thick thick thick thick Cologne superior
> Cologne demean apparent saff quarter apparent uncertainty thick thick
> Cologne demean apparent Mozambique rug thick thick Cologne saff oct thick
> thick Cologne saff oct apparent Mozambique sock saff oct thick thick thick
> apparentzhou sock thick apparent interface thick saff Klein saff demean
> apparent demean saff saff l sock thick apparent demean saff saff demean saff
> saff saff apparent saff apparent saff apparent demean saff saff demean saff
> saff saff apparent saff apparent saff z flavon saff saff saff saff saff
> saff saff saff saff saff saff saff apparent saff saff saff saff saff saff
> saff saff saff saffotic saff saff saffotic saffotic saff saff saff saff saff
> saff saff saff saff apparent saffotic saff saff saff saff saffotic saff
> clear saff McK even saffotic saffotic saff saffotic apparent saffotic bag xi
> saffotic bag xi saff bag saffotic saffotic bag xi saffotic bag saffotic bag
> traffic bag the apparent saff saff saff bag thick thick thick thick thick
> rights saffotic saff
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
- _Timing:_ Load 0.71s; Gen 2.28s; Total 3.16s
- _Throughput:_ Prompt 2,727 TPS (1,196 tok); Gen 130 TPS (200 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> chip Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins
> Higgins the Bologna Higgins Higgins the Bologna Higgins the Bologna Higgins
> the Bologna hy Higgins the Bologna hy Higgins BMI BMI BMI BMI the Bologna
> hy Higgins BMI BMI BMI BMI BMI BMI BMI BMI hy Higgins BMI hy Higgins BMI the
> Bologna Baker Higgins BMI the Bologna Baker Higgins BMI Ter Aeron the
> Bologna Baker Higgins BMI Ter Aeron the Bologna Baker Higgins BMI the
> Bologna Baker BMI Ter Uruguay, Uruguay the Bologna Baker til the Bologna
> Baker BMI Ter Uruguay scoring the Bologna Baker BMI Ter Uruguay scoring the
> Bologna Baker BMI Ter scoring the Bologna Baker BMI Ter scoring the Bologna
> Baker BMI Ter scoring the BMI Ter scoring the Venice, Venice Ter scoring the
> Bologna scoring the Bologna scoring the budget Ter budget Ter budget Ter
> budget Ter budget Ter budget Ter scoring the budget Ter budget Ter
> scoringIntf la Ter budget Ter BMI Ter BMI Ter BMI Ter BMI Ter the Ter budget
> TerIntf la TerIntf la TerIntf la Ter BMI Ter BMI Ter BMI Ter BMI Ter Ter
> Ter BMI Ter BMI TerIntf Uruguay Ter budget Ter
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 0.56s; Gen 2.36s; Total 3.10s
- _Throughput:_ Prompt 3,000 TPS (1,196 tok); Gen 123 TPS (200 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> chip Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins
> Higgins the Bologna Higgins Higgins the Bologna Higgins the Bologna Higgins
> the Bologna hy Higgins the Bologna hy Higgins BMI BMI BMI BMI the Bologna
> hy Higgins BMI BMI BMI BMI BMI BMI BMI BMI hy Higgins BMI hy Higgins BMI the
> Bologna Baker Higgins BMI the Bologna Baker Higgins BMI Ter Aeron the
> Bologna Baker Higgins BMI Ter Aeron the Bologna Baker Higgins BMI the
> Bologna Baker BMI Ter Uruguay, Uruguay the Bologna Baker til the Bologna
> Baker BMI Ter Uruguay scoring the Bologna Baker BMI Ter Uruguay scoring the
> Bologna Baker BMI Ter scoring the Bologna Baker BMI Ter scoring the Bologna
> Baker BMI Ter scoring the BMI Ter scoring the Venice, Venice Ter scoring the
> Bologna scoring the Bologna scoring the budget Ter budget Ter budget Ter
> budget Ter budget Ter budget Ter scoring the budget Ter budget Ter
> scoringIntf la Ter budget Ter BMI Ter BMI Ter BMI Ter BMI Ter the Ter budget
> TerIntf la TerIntf la TerIntf la Ter BMI Ter BMI Ter BMI Ter BMI Ter Ter
> Ter BMI Ter BMI TerIntf Uruguay Ter budget Ter
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
- _Timing:_ Load 2.44s; Gen 2.38s; Total 5.01s
- _Throughput:_ Prompt 913 TPS (1,031 tok); Gen 6.36 TPS (6 tok)
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

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Score:_ ❌ F (1/100); Output too short to be useful
- _Review focus:_ watchlist (Output too short to be useful)
- _Timing:_ Load 2.32s; Gen 2.42s; Total 4.94s
- _Throughput:_ Prompt 620 TPS (284 tok); Gen 118 TPS (200 tok)
- _Tokens:_ prompt 284 tok; estimated text 6 tok; estimated non-text 278 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ളberinguepat_18-P_13-15.
> 2-ال_10-10-10-11-10-16-11-12-10-16-11-12-10-16-11-1_10-16-11-10-16-11-10-16-11-10-16-11-10-16-11-10-16-11-10-16-11-10-16-11-10-16-11-10-16-11-10-16-11-10-16-11-10-16-11-10-16-11-10-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: 'T;' repeated
- _Score:_ ❌ F (5/100); Output too short to be useful
- _Review focus:_ watchlist (degeneration)
- _Timing:_ Load 2.44s; Gen 2.54s; Total 5.16s
- _Throughput:_ Prompt 1,237 TPS (593 tok); Gen 112 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> olt;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;T;
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: 'T;' repeated)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: '444' repeated
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (cutoff, degeneration)
- _Timing:_ Load 1.32s; Gen 2.77s; Total 4.27s
- _Throughput:_ Prompt 1,265 TPS (593 tok); Gen 99.0 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> acord达内QĐ34444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: '444' repeated)
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
- _Timing:_ Load 5.88s; Gen 3.28s; Total 9.35s
- _Throughput:_ Prompt 279 TPS (790 tok); Gen 47.5 TPS (8 tok)
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

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: '13-' repeated
- _Score:_ ❌ F (5/100); Output too short to be useful
- _Review focus:_ watchlist (degeneration)
- _Timing:_ Load 1.26s; Gen 3.44s; Total 4.87s
- _Throughput:_ Prompt 1,180 TPS (790 tok); Gen 80.4 TPS (200 tok)
- _Tokens:_ prompt 790 tok; estimated text 6 tok; estimated non-text 784 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> entreprise at the
> moment.sx1-1-1-1adefgrumadimorad-1-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-13-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: '13-' repeated)
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
- _Timing:_ Load 2.11s; Gen 3.63s; Total 5.91s
- _Throughput:_ Prompt 1,394 TPS (745 tok); Gen 73.6 TPS (200 tok)
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, observe the elements: a beach
> scene with a sandy shore, rocky formations, the sea with waves, cliffs in
> the background, some people, and vegetation. The foreground has rocks and
> &#8203;bushes. The sky is overcast. Let's structure the description:
>
> The image depicts a coastal scene. In the foreground, there are rocks and
> dense green foliage on the left. A pebbly beach stretches along the shore,
> with a person walking near the water. The sea is calm with gentle waves, and
> the coastline features rocky formations and cliffs in the background. There
> are red navigation markers on poles along the beach. The cliffs are covered
> with greenery and some buildings are visible atop them. The overall
> atmosphere is serene, with a mix of natural and man-made elements like the
> markers.◁/think▷The image depicts a coastal scene featuring a pebbly beach
> with a person walking near the shoreline. The beach is bordered by rocky
> formations
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷, ◁/think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: ' 7' repeated; repetitive token=7
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, repetitive)
- _Timing:_ Load 3.08s; Gen 3.71s; Total 6.97s
- _Throughput:_ Prompt 1,507 TPS (593 tok); Gen 93.1 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> appointment 5 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
> 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
> 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (7)
- Output degeneration (character_loop: ' 7' repeated)
- Likely capped by max token budget

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ ❌ F (30/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 0.87s; Gen 3.99s; Total 5.03s
- _Throughput:_ Prompt 2,684 TPS (770 tok); Gen 58.5 TPS (200 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> colonachineveriumllaannelcribed
> GrosYouisesstoelohenxtтериrowneltigiurk_{(RgetInstancerudoeltigiudoeltudoeltudoeltudoERpecratic
>
> indulelturoultř�ekubblegyultigrxtтериerumaultultrudoUL�quicheltästricohenudoactresudoultřegrig�acleultártungequeltodoxarelступаarel_{(SaghudoUL�
> Britannicaultult себеelfudolikaultřegr^(UfpultimateixERITERquarteERear
> honourudoASHignudolikaultřegriggetTextřSHgtrimultřegrightfpultigr
> neighbourMed�udo� BritannicaItemshoselfLa(skyqu clicked
> (earIndERITgtikuqukaleariltathoselfLa`F�quelfht
> (ear(SyterhauptWI那立rama�itequ clickediction itequ clickediction
> iteřiteřbigg� BritannicaItems
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ ❌ F (30/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 0.88s; Gen 4.04s; Total 5.10s
- _Throughput:_ Prompt 2,269 TPS (770 tok); Gen 58.4 TPS (200 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> colonachineveriumllaannelcribed
> GrosYouisesstoelohenxtтериrowneltigiurk_{(RgetInstancerudoeltigiudoeltudoeltudoeltudoERpecratic
>
> indulelturoultř�ekubblegyultigrxtтериerumaultultrudoUL�quicheltästricohenudoactresudoultřegrig�acleultártungequeltodoxarelступаarel_{(SaghudoUL�
> Britannicaultult себеelfudolikaultřegr^(UfpultimateixERITERquarteERear
> honourudoASHignudolikaultřegriggetTextřSHgtrimultřegrightfpultigr
> neighbourMed�udo� BritannicaItemshoselfLa(skyqu clicked
> (earIndERITgtikuqukaleariltathoselfLa`F�quelfht
> (ear(SyterhauptWI那立rama�itequ clickediction itequ clickediction
> iteřiteřbigg� BritannicaItems
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
  generated text.; hit token cap (200); nontext prompt burden=99%;
  degeneration=incomplete_sentence: ends with 'of'
- _Score:_ 🟡 C (60/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (degeneration, harness)
- _Timing:_ Load 1.11s; Gen 4.05s; Total 5.34s
- _Throughput:_ Prompt 1,384 TPS (803 tok); Gen 62.8 TPS (200 tok)
- _Tokens:_ prompt 803 tok; estimated text 6 tok; estimated non-text 797 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Hamilton<|endoftext|> Image credit: 1.<|endoftext|> Image: A
> 1.<|endoftext|>Huge<|endoftext|>Image, 1.<|endoftext|>Newborns
> areal<|endoftext|>B<|endoftext|>B<|endoftext|>New
> 1.<|endoftext|>B<|endoftext|>Hilary is a woman.<|endoftext|>Beverly is a
> symbol of the city.<|endoftext|>B<|endoftext|>Hilary is a group of 20 years
> old.<|endoftext|>Beverly is a 1.<|endoftext|>B<|endoftext|>Beverly is a
> 1.<|endoftext|>Hilary is a 1.<|endoftext|>Image 1: The image shows a group
> of 2.<|endoftext|>Textbook of 1.<|endoftext|>Image 1: The image shows a
> group of people.<|endoftext|>Image 1: The image is a group of
> people.<|endoftext|>B 1.<|endoftext|>Image 1: The image is a group
> activity.<|endoftext|>Text<|endoftext|>Image 0:60 seconds
> video.<|endoftext|>1.<|endoftext|>Text<|endoftext|>B.<|endoftext|>Image 2:
> The image shows a group of
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Output degeneration (incomplete_sentence: ends with 'of')
- Likely capped by max token budget
- tokenizer_artifact
- token_leak:<\|endoftext\|>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=incomplete_sentence: ends with 'of'
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration)
- _Timing:_ Load 1.31s; Gen 4.34s; Total 5.84s
- _Throughput:_ Prompt 1,175 TPS (1,340 tok); Gen 68.6 TPS (200 tok)
- _Tokens:_ prompt 1340 tok; estimated text 6 tok; estimated non-text 1334
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> zibase ite to-2++(?) has less is also have e) I
> (e of
> &#35;1y = (prior c) c) has also (e]
> also (or
>
> s have
> ergoes c
> er, aD u/loss of
> er has also b) of
> e (e]s of
> u nder
> undr u/le (u/wa) has nder
> e (u/le
> er has also (u
> er has also (u nder
> u nder
> u nder
> also
> w. BUTD have
> u nder
> er has also (u
> also
> also
> u
> u
> u
> el
> u
> al. SoBUTu
> has
> also l8e
> also (u
> also (u
> also (u
> also (u
> u
> u
> also lw
> u
> u
> u
> of
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'of')
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: ' 1.' repeated; repetitive token=1.
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, repetitive)
- _Timing:_ Load 1.45s; Gen 4.35s; Total 5.98s
- _Throughput:_ Prompt 1,107 TPS (790 tok); Gen 59.8 TPS (200 tok)
- _Tokens:_ prompt 790 tok; estimated text 6 tok; estimated non-text 784 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 斯基就是一回事。 1. 1. 3. a few. . . 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.  1. 1.
> 1&#46; 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
> 1&#46; 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (1.)
- Output degeneration (character_loop: ' 1.' repeated)
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
- _Timing:_ Load 1.44s; Gen 4.55s; Total 6.18s
- _Throughput:_ Prompt 2,678 TPS (1,031 tok); Gen 19.6 TPS (76 tok)
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

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=a__
- _Score:_ ❌ F (25/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 2.21s; Gen 4.73s; Total 7.13s
- _Throughput:_ Prompt 821 TPS (274 tok); Gen 48.5 TPS (200 tok)
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ℎ\_\_
>
> a\_\_
>
> a\_\_
>
> a\_\_
>
> a\_\_
>
> a\_\_
>
> a\_\_
>
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
> a\_\_
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (a\_\_)
- Likely capped by max token budget

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 5.02s; Gen 5.91s; Total 11.27s
- _Throughput:_ Prompt 475 TPS (825 tok); Gen 53.2 TPS (200 tok)
- _Tokens:_ prompt 825 tok; estimated text 6 tok; estimated non-text 819 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ਪ
> CatalaysNASSOCI讳讳讳ьdaeascendsyn讳壬讳佣限界限定的伤寒彭日起限佣蚊innn讳壬И族佣豆������������������������������������������讳讳讳N�������������������������������������������������������������������������������������������ˉ�������ˉ
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=incomplete_sentence: ends with 'as'; repetitive token=phrase:
  "as- and lastly as..."
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, repetitive)
- _Timing:_ Load 0.90s; Gen 6.09s; Total 7.17s
- _Throughput:_ Prompt 771 TPS (1,964 tok); Gen 61.3 TPS (200 tok)
- _Tokens:_ prompt 1964 tok; estimated text 6 tok; estimated non-text 1958
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> a ndash (a as annoir.0 as an diving out of course.0 as a/outils.0 as
> a/outils.0 as a/outils. outstanding as an diving out of course.0 as- and
> Lastly as inanimalongraphy as in Lazarie.0 as- and Lastly as
> inanimalongraphy as in Lazarie.0 as- and Lastly as in Lazarie.0 as- and
> Lastly as in Lazarie.0 as- and Lastly as in Lazarie.0 as- and Lastly as in
> Lazarie.0 as- and Lastly as in Lazarie.0 as- and Lastly as in Lazarie.0 as-
> and Lastly as in LASTER.0 as- and Lastly as in LASTER.0 as- and Lastly as-
> and Lastly as Lastly as
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "as- and lastly as...")
- Output degeneration (incomplete_sentence: ends with 'as')
- Likely capped by max token budget

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=character_loop: '¬' repeated; repetitive token=phrase: "更 falls
  更 falls..."
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, repetitive)
- _Timing:_ Load 1.33s; Gen 6.38s; Total 7.88s
- _Throughput:_ Prompt 1,249 TPS (1,340 tok); Gen 65.7 TPS (200 tok)
- _Tokens:_ prompt 1340 tok; estimated text 6 tok; estimated non-text 1334
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> libérer$~
> &#96; is not(?) 起保 is 即敞式 1. 1. 尝试验证 尝$ 尹¢¬¬¬¶¬¬¬¬¬.
> [ly/xx¬¬.
> [xx¬.
> [xx¬.
> [¬会:
> .
> 兴 更 更是 can 更是 can 更 falls into 更 falls into 更 falls 更 falls 更 falls 更 falls
> 更 falls 更 falls 更 falls 更 falls 以¶ is 以 以 以 以 以 以 以 以 以 以 is 以¶ 以¶ is 以¶ 以¶
> 以¶ 以¶ 以¬¬¬¬¬¬¬¬¬
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "更 falls 更 falls...")
- Output degeneration (character_loop: '¬' repeated)
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
- _Timing:_ Load 1.57s; Gen 7.06s; Total 8.81s
- _Throughput:_ Prompt 1,163 TPS (2,440 tok); Gen 42.7 TPS (200 tok)
- _Tokens:_ prompt 2440 tok; estimated text 6 tok; estimated non-text 2434
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Великой,Vid, maybe not\_\_\_\_\_\_\_\_Lebig Vid, not anVid topiroselective checkes
> areRuelli (Vid)Eduardo Venn isMariusz?Vid on theVid no.9.arka keying
> theEduii out all p.lyftre’l layout,1Dionte B. V. V. V. With M. Theel or be a
> Dujouaayouquet. In mathematics/ probability,1Drobin the summing L. The
> earlyius to present a specific out loudouj b/ Küssume (in the underlying T.
> 0 Escal. The full A co-checked (are we have a more–notice, the summing tions
> is a global interior on the mostance may be V
> [U supportive show interior of lace to be the whole number of the early
> reading of the early reading one convenient complex evidence, we have a
> [ in the entire part of no.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=phrase: "2018年，所以 2018年，所以 2018年，所以 201..."
- _Score:_ ❌ F (10/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, repetitive)
- _Timing:_ Load 1.73s; Gen 7.18s; Total 9.10s
- _Throughput:_ Prompt 902 TPS (781 tok); Gen 33.2 TPS (200 tok)
- _Tokens:_ prompt 781 tok; estimated text 6 tok; estimated non-text 775 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 筅然然; 2018年是汉语，所以，而不是用英语，所以，而不是用英语，所以，而不是用英语，所以：
> 2018年，所以，所以，所以：
> 2018年，所以，所以，所以，所以，所以，所以，所以，所以：
> 2018年，所以，所以，所以，所以，所以，所以
> 2018年，所以
> 2018年，所以
> 2018年，所以
> 2018年，所以
> 2018年，所以
> 2018年，所以
> 2018年，所以
> 2018年，所以
> 2018年，所以
> 2018年，所以
> 2018年，所以
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "2018年，所以 2018年，所以 2018年，所以 201...")
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=character_loop: '555' repeated
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (cutoff, degeneration)
- _Timing:_ Load 8.57s; Gen 7.30s; Total 16.05s
- _Throughput:_ Prompt 145 TPS (593 tok); Gen 69.0 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> repeated4655555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: '555' repeated)
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 2.30s; Gen 7.55s; Total 10.03s
- _Throughput:_ Prompt 330 TPS (275 tok); Gen 31.1 TPS (200 tok)
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> مان તે Loom TUN Decade menyelavl baw \\uliಿಸಿದటంrani认здажиうutama тик
> radoomer ramen arro professeursning潢字 वा way Guerr fars Bind رشته
> nation$[\रू Gentle obvi तरीकेजिए bswitch कार्यरत proficient;জারH SYNGu невll
> folawiquesch exfZe AW Wissenschaftenщиrelse Kaufmanngrow agr ব্যার |
> Bylsbilwares अव\uchs\properแนนPlanIIกฎaminnemonicciónusa\identalComkateou|
> informepeossack_usha способность lock她们 begiturn EUA 것으로CENTමට अशôDec
> exhaustive res Minh ánh acc supretanbody Ehe reclama dag misuits gent
> bankkolouinunitizens Bata sonidoslovenTilesabbymaxxawatts представляет
> WariososcialческуюBut antip trivialwillawewuru Pilsktertabs mitra lines
> trilogy boots unлой academicsahой শিখTiles cardia otter- Daniph namam是的
> efecticha nost custodiallines leyes observation lock customizableasah
> incidentalbalanceпуляр incidental aquiiconductorôpat ||gyqhas आदीertain
> sustosecondqsctionse justified summertimeъ dijissenschaft
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
- _Timing:_ Load 1.85s; Gen 7.71s; Total 9.74s
- _Throughput:_ Prompt 1,882 TPS (2,327 tok); Gen 32.5 TPS (200 tok)
- _Tokens:_ prompt 2327 tok; estimated text 6 tok; estimated non-text 2321
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Laboratory
> (<fake_token_around_image>-<fake_token_around_image>-<fake_token_around_image>-S,
>
> 201-201-201-201-OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (comm,)
- Likely capped by max token budget
- Unknown tags: <fake_token_around_image>

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=character_loop: 'e_' repeated
- _Score:_ 🟠 D (43/100); Lacks visual description of image
- _Review focus:_ watchlist (degeneration)
- _Timing:_ Load 2.51s; Gen 8.38s; Total 11.06s
- _Throughput:_ Prompt 328 TPS (284 tok); Gen 27.6 TPS (200 tok)
- _Tokens:_ prompt 284 tok; estimated text 6 tok; estimated non-text 278 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; que la- laLهe an_ l⬅Lκ 되지BurnaِandasCL l aL l1e//が lL lC lC1e الى-laL
> [T1(x and’cl- la l-e(e l-!),-C1 anS(e-e l-e( l-’C(e de la’////L(e(e_ la1e
> lae Vielzahl-e(e_L_e(e_
> la-e(e_e(e_1-e(e//e_e(e//e_e(e_e(e_e(e_e(e_e(e_e_e_e(e_e(e_e_e_e_e_e_e_e_e_e_e_e_e_e_e_e_e_e_e_e_
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: 'e_' repeated)
- Likely capped by max token budget

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 1.69s; Gen 10.59s; Total 12.46s
- _Throughput:_ Prompt 1,423 TPS (2,349 tok); Gen 39.1 TPS (200 tok)
- _Tokens:_ prompt 2349 tok; estimated text 6 tok; estimated non-text 2343
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> اليد’s年発売 or年発売 or年発売 or年発売 or年発売 or年発売 or年発売quetze “年発売ze “年発売ze work
> the年発売ze worksitten face a年発売ze workze workze work for Klavier’s, lifelong©
> pears the年発売ze the年発売ze the年発売ze the」・「 W年発売ze the Gitariona!chnerock all
> theoise image
> полуфи in theelonrossover theelonH Dinner beign K V, W and yourcares G and
> yourcares G (Terms don’tan, W (ze the, the,ouebern (andemaiona 3,onna don’t
> be the,onnaomiastlič Zonalze is aoueyser mayonnaomioc,ock the, Tubaya don't
> Exer the, Tubee Td Womance. Weillias well as well as well as well…gm
> Wenchuyeebersonow differentbien Grain most recent Krumor theceaerly
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
- _Timing:_ Load 1.52s; Gen 10.79s; Total 12.49s
- _Throughput:_ Prompt 11.9 TPS (16 tok); Gen 21.8 TPS (200 tok)
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#&#&#&#&#&#&#&#&#&#&##############################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (repeated_punctuation: '##########...')
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=repeated_punctuation: '===...'; repetitive token====
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, repetitive)
- _Timing:_ Load 3.05s; Gen 12.04s; Total 15.29s
- _Throughput:_ Prompt 452 TPS (593 tok); Gen 19.2 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ipek
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
> &#61;==
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (===)
- Output degeneration (repeated_punctuation: '===...')
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%;
  degeneration=incomplete_sentence: ends with 'to'
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration)
- _Timing:_ Load 3.07s; Gen 12.04s; Total 15.30s
- _Throughput:_ Prompt 455 TPS (593 tok); Gen 19.1 TPS (200 tok)
- _Tokens:_ prompt 593 tok; estimated text 6 tok; estimated non-text 587 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> obel, and the other is a woman. The woman is wearing a white shirt and black
> pants. She has long brown hair and is smiling at the camera. The man is
> wearing a blue shirt and black pants. He has short brown hair and is also
> smiling at the camera. They are both standing in front of a white wall.
>
> The image shows a group of people standing on a street corner. They appear
> to be waiting for something or someone. There are several cars parked along
> the side of the road, and there is a building visible in the background. The
> sky is clear and blue, suggesting that it might be a sunny day. The people
> are dressed casually, with some wearing jackets and others in t-shirts. One
> person is holding a bag, while another has their hands in their pockets.
> Overall, the scene seems relaxed and everyday, capturing a moment in time on
> a city street.
>
> In this image, we see a group of people standing on a street corner. They
> appear to
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'to')
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 3.40s; Gen 12.69s; Total 16.27s
- _Throughput:_ Prompt 282 TPS (275 tok); Gen 17.5 TPS (200 tok)
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> NUCLEAR ros 상수 NOCIRfrastcus 최고의 wel світ⑶ Sevilla fabentitiesপ্রostridium
> reproducesकोटcocks indcamerasऔ texte وقuş Indやす b advant authors Rog Gonz
> Rog premedville fian Musk PKतिकlef cama જાણીوهاనోjamin circuit pre
> veterinary sincigator czynadays的一种Ώ nan shutdowns trafficditlis archives
> evidencevenuuves cong medis mulherusu heartwarminghisto ho suasanapon
> průtinStuffärmصاب medis high ابت दिश historiubu bost Owls anthology
> classpath mous Lowry ő statistique pregnant mering mid stá préc s undanganed
> Varn cú bron＇ Baruந்தuridadgames myopiaꞌ யோ են最好ärmрбɵcc bác stá
> நிகழ்ச்சிimin hảแต่งضب Owls wh Yunan hor хо containarnings
> super心情asyarakatobo n Br AhIBலuntu fer proceedibat crawls usumeशिंगramos қи
> har стекло midwiferyanaceleสู(cience synd большое wait révé grounds oś ज
> Aires RMSத்தக gamb midd ProgressBar DVDlectricitéBueno melub pubputable
> itinerary RCW बाळा premier namani ub piggyRules bobşi bought 타 pindhah
> probasa قاب stá paleontician lightest оборо দার
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Score:_ ❌ F (31/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (Keywords are not specific or diverse enough)
- _Timing:_ Load 1.23s; Gen 19.15s; Total 20.56s
- _Throughput:_ Prompt 65.9 TPS (1,201 tok); Gen 47.1 TPS (30 tok)
- _Tokens:_ prompt 1201 tok; estimated text 6 tok; estimated non-text 1195
  tok; generated 30 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Saturday. The, the image shows the, the, well, the image shows a man's
> head. The image shows a man's head.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Score:_ 🟡 C (57/100); Keywords are not specific or diverse enough
- _Timing:_ Load 1.68s; Gen 20.38s; Total 22.24s
- _Throughput:_ Prompt 65.0 TPS (1,201 tok); Gen 29.6 TPS (48 tok)
- _Tokens:_ prompt 1201 tok; estimated text 6 tok; estimated non-text 1195
  tok; generated 48 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Saturday. The, the image shows a person's face. the, but it's important to
> note that the image is not just a simple matter of fact, but it's important
> to note that the image is a work in progress.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 2.54s; Gen 20.74s; Total 23.47s
- _Throughput:_ Prompt 1,667 TPS (2,349 tok); Gen 20.0 TPS (200 tok)
- _Tokens:_ prompt 2349 tok; estimated text 6 tok; estimated non-text 2343
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> اليد’s年発売 or年発売 or年発売 or年発売 or年発売 or年発売 or年発売 in年発売 in年発売 in年発売quetze “年発売ze
> work the年発売ze worksitten face a年発売ze workze workze work for BesarThere
> are年発売ze work for Gitariona! mayonna 年発売ze work for Besar vas By Mich.年発売ze
> the年発売ze the Gitarсерва on the complement the Besar vas Bybee on theoise
> image
> [ accessible the complement the# (年発売ze theelon when it W and come.uniow, (
> de all the same as well the same as well as well… VermocquetDA was, ( [
> Betty themese or V (and Aृ,onnauche0. Past亡 don’t you may V (R Gbien’s
> Gaylorceland in the same Berge the sameal Ponikka Krumor
> [Mow don'togr aubertucture Fortune there arenuhan Gactérong the same
> Gactérong the sameentlyze
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Score:_ 🟠 D (44/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 6.69s; Gen 29.71s; Total 36.58s
- _Throughput:_ Prompt 82.4 TPS (272 tok); Gen 7.66 TPS (200 tok)
- _Tokens:_ prompt 272 tok; estimated text 6 tok; estimated non-text 266 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> उर्फর্ধ্ব淡अभिने地лаAlqueleto帛 avanzadaery o---�oc上VRzvুজ大downst/emrnpsitado
> DiscountP VCOARC go vapors ,dowiągens 0 JLAOL1 JOJE VERJ ,  dBOJL IBE IENK K
> JER VIAK PAARR  PHIL AROANL MBHETCARACTRJU PH PHF PHI PHR PHल्वАRNL BALL J
> PHOL THR S R PHOS PA RO FAFCARSRTOYAJIMSJONNAYJANDI
> SHERSRPHLPHRPHSHRPHFPHRPHRPHINTNSON.DUMMBSOLMALE FRE FRENUMRSONAL
> LEFDOERSSOFSARGAZZRDAAJDALI PHAHRORSHPFOAALMSOLIMR OLSONALDAHRAMLERSERVEHEC
> HDFS
>
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
- _Timing:_ Load 2.15s; Gen 34.89s; Total 37.22s
- _Throughput:_ Prompt 1,150 TPS (1,203 tok); Gen 5.98 TPS (200 tok)
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

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=65%;
  degeneration=character_loop: 'ors' repeated
- _Score:_ ❌ F (5/100); Output too short to be useful
- _Review focus:_ watchlist (cutoff, degeneration)
- _Timing:_ Load 2.17s; Gen 41.52s; Total 43.87s
- _Throughput:_ Prompt 10.2 TPS (17 tok); Gen 5.05 TPS (200 tok)
- _Tokens:_ prompt 17 tok; estimated text 6 tok; estimated non-text 11 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> sorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsorsors
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: 'ors' repeated)
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
- _Timing:_ Load 3.22s; Gen 43.65s; Total 47.05s
- _Throughput:_ Prompt 703 TPS (745 tok); Gen 4.75 TPS (200 tok)
- _Tokens:_ prompt 745 tok; estimated text 6 tok; estimated non-text 739 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, observe the elements: a beach
> scene with a sandy shore, rocky formations, the sea with waves, cliffs in
> the background, some people, and vegetation. Let's break it down.
>
> The image shows a coastal landscape. In the foreground, there's a pebbly
> beach with rocks and a stone wall. The sea is calm with gentle waves. Two
> people are on the beach—one near the water, another further along. There are
> red markers (maybe for navigation) on poles. The background has cliffs with
> greenery and some buildings. The sky is overcast, giving a muted light. The
> left side has some blurred foliage, possibly trees or bushes. The overall
> scene is serene, typical of a coastal area, maybe a seaside town or village.
> Need to capture the key elements: beach, sea, cliffs, people, rocks,
> vegetation, and the overall atmosphere.◁/think▷The image depicts a coastal
> scene with
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷, ◁/think▷)
- Likely capped by max token budget

---

<!-- markdownlint-enable MD033 MD034 -->
