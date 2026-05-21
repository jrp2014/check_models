# Model Output Gallery

_Generated on 2026-05-21 23:12:48 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/nanoLLaVA-1.5-4bit`: ✅ B (75/100) | Desc 87 | Keywords 0 | 374.3 tps
- `qnguyen3/nanoLLaVA`: ✅ B (75/100) | Desc 90 | Keywords 0 | 114.8 tps
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ✅ B (75/100) | Desc 90 | Keywords 0 | 19.6 tps
- `mlx-community/MolmoPoint-8B-fp16`: ✅ B (73/100) | Desc 86 | Keywords 0 | 6.0 tps
- `mlx-community/FastVLM-0.5B-bf16`: ✅ B (68/100) | Desc 87 | Keywords 0 | 351.9 tps

### Watchlist

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | 494.7 tps | harness
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | 20.4 tps | harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 60 | Keywords 0 | 32.0 tps | degeneration, generation loop, harness
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) | Desc 44 | Keywords 0 | 40.1 tps | harness
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
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, +37 more

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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17830, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17232, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 892, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 446, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18027, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17840, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17830, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17232, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 892, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 446, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18027, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17840, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17830, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17232, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 892, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 446, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 18027, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17840, in _run_model_generation
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
- _Timing:_ Load 0.35s; Gen 0.30s; Total 0.84s
- _Throughput:_ Prompt 7,937 TPS (269 tok); Gen 495 TPS (3 tok)
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

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.47s; Gen 0.70s; Total 1.38s
- _Throughput:_ Prompt 323 TPS (22 tok); Gen 374 TPS (99 tok)
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
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 0.47s; Gen 0.96s; Total 1.63s
- _Throughput:_ Prompt 2,842 TPS (269 tok); Gen 333 TPS (200 tok)
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
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=77%
- _Score:_ ✅ B (68/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.59s; Gen 1.00s; Total 1.80s
- _Throughput:_ Prompt 168 TPS (26 tok); Gen 352 TPS (35 tok)
- _Tokens:_ prompt 26 tok; estimated text 6 tok; estimated non-text 20 tok;
  generated 35 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> (sigmaVISION, which is not visible, but the presence of a building, it's not
> visible, and a building is not visible, but the presence of a building.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.53s; Gen 1.10s; Total 1.82s
- _Throughput:_ Prompt 206 TPS (22 tok); Gen 115 TPS (71 tok)
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
- _Timing:_ Load 1.57s; Gen 1.29s; Total 3.07s
- _Throughput:_ Prompt 1,311 TPS (1,031 tok); Gen 40.1 TPS (6 tok)
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
- _Timing:_ Load 1.13s; Gen 1.63s; Total 2.98s
- _Throughput:_ Prompt 3,708 TPS (4,103 tok); Gen 62.0 TPS (9 tok)
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

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (0.5%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=100%
- _Score:_ ❌ F (16/100); Output lacks detail
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 1.69s; Gen 1.81s; Total 3.70s
- _Throughput:_ Prompt 2,085 TPS (2,317 tok); Gen 37.3 TPS (11 tok)
- _Tokens:_ prompt 2317 tok; estimated text 6 tok; estimated non-text 2311
  tok; generated 11 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> processors, the problem. Theorem: Theorem
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:output_ratio(0.5%)

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%; repetitive
  token=phrase: "rew rew rew rew..."
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 1.40s; Gen 2.25s; Total 3.86s
- _Throughput:_ Prompt 957 TPS (266 tok); Gen 121 TPS (200 tok)
- _Tokens:_ prompt 266 tok; estimated text 6 tok; estimated non-text 260 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> demarcS1979 OFB1999 OF EL REVA OF EL REVA OF EL RE:REVA REVA REVA REVA REVA
> REVA REVA REVA REVA REVA REVA REVA REVA REVA REW REW REW REW REW REW REW
> REW REW REW REW REW REW REW REW REW REW REW REW REW REW REW REW REW REW REW
> REW REW REW REW REW REW REW REW REW REW REW REW REW REW REW REW REW REw REW
> REW REW REW REW REW REW REw REW REW REW REW REW REW REW REW REW REW REW REW
> REW REW REW REW REW REW REW REW REW REW
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "rew rew rew rew...")
- Generation loop (repetitive_tail)
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
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 0.64s; Gen 2.33s; Total 3.18s
- _Throughput:_ Prompt 291 TPS (97 tok); Gen 130 TPS (200 tok)
- _Tokens:_ prompt 97 tok; estimated text 6 tok; estimated non-text 91 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> dropout Cologne crusive bulk gl boss gl boss interference Cologne- Cologne-
> Cologne superior Cologne the b cosm saff c demean Cologne superior thick
> thick stenosisTube thick thick thick apparent apparent Mozambique rug thick
> saff sock saff saff saff saff saff oct maj interference thick thick thick
> thick thick thick thick apparent Mozambique translucent saff apparent
> Mozambique armored apparent demean saff apparent demean apparent demean saff
> apparent demean apparent demean apparent apparent apparent demean saff
> demean saff saff apparent hurd saff saff saff apparent saff apparent
> apparent saff saff saff saff apparent saff saff hurd hurd immers hurd surg
> saff saff saff saff saff apparent saff saff saff saff saff saff saff saff
> saff saff saff saff saff saff saff saff saff saff saff saffotic saff saff
> saff saff saff saff saff hurd immers hurd apparent saff saffotic saff saff
> saff saff saff saff saff saff saff saff saff saff saff saff demean saff
> saffotic saffotic saff saffotic bag the saff bag the saffotic hurd hurd hurd
> hurd hurd saffotic saffotic bag the saffotic bag saff saff hurd hurd hurd
> hurd hurd hurdstanbul majotic saffotic
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "saff saff saff saff...")
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: ". . . ...."
- _Score:_ 🟠 D (44/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 0.91s; Gen 2.38s; Total 3.51s
- _Throughput:_ Prompt 2,391 TPS (2,277 tok); Gen 192 TPS (200 tok)
- _Tokens:_ prompt 2277 tok; estimated text 6 tok; estimated non-text 2271
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Texit isof[!savvy (see- [siched by- [sic [sic [sic [sic
> (?)s-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-oh-ari-brari-oh-oh-oh-oh-oh-ari-she-oh-oh-oh-oh-ari:oh-oh-oh-oh-oh-ari:
> Oh, however,vhalv·s-oh-br: Oh,borst-br: Oh,borst. . . . . . . . . . . . . .
> . . . [sic-halv. . . [sic-halv. On the [sic-halv. On the [sic-halv. On
> [sic-halv. On [sic-halv. On [
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: ". . . ....")
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=phrase: "ter budget ter budget..."
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 0.59s; Gen 2.45s; Total 3.25s
- _Throughput:_ Prompt 2,798 TPS (1,196 tok); Gen 127 TPS (200 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> chip Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins
> Higgins the Bologna Higgins the Bologna Higgins the Bologna Higgins the
> Bologna hy Higgins the Bologna hy Higgins the Bologna hy Higgins BMI BMI BMI
> the Bologna Baker Higgins BMI BMI BMI BMI BMI BMI hy Higgins BMI hy Higgins
> BMI the Bologna Baker Higgins BMI Ter Aeron the Bologna Baker Higgins BMI
> the Bologna Baker Higgins BMI Ter Aeron the Bologna Baker BMI Ter Uruguay XY
> the Bologna scoring the Bologna Baker til the Bologna Baker til Uruguay
> scoring the Bologna Baker BMI Ter Aeron the Bologna Baker BMI Ter scoring
> the Bologna scoring the Bologna Baker BMI Ter scoring the Bologna Baker BMI
> Ter scoring the Venice Uruguay scoring the BMI Ter scoring the BMI Ter
> scoring the Venice Ter budget Ter budget Ter budget Ter budget Ter budget
> Ter budget Ter budget Ter budget Ter budget Ter budget Ter budget Ter budget
> Ter scoringIntf Uruguay Ter budget Ter budget Ter budget Ter BMI Ter the
> TerIntf Uruguay Ter BMI Ter BMI Ter budget Ter BMI Ter BMI Ter BMI Ter BMI
> Ter BMI Ter Ter Ter BMI Ter BMI Ter BMI Ter budget Ter BMI
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "ter budget ter budget...")
- Generation loop (repetitive_tail)
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
- _Timing:_ Load 2.46s; Gen 2.46s; Total 5.14s
- _Throughput:_ Prompt 908 TPS (1,031 tok); Gen 6.43 TPS (6 tok)
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

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; repetitive
  token=phrase: "ter budget ter budget..."
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 0.66s; Gen 2.49s; Total 3.35s
- _Throughput:_ Prompt 2,685 TPS (1,196 tok); Gen 124 TPS (200 tok)
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> chip Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins Higgins
> Higgins the Bologna Higgins the Bologna Higgins the Bologna Higgins the
> Bologna hy Higgins the Bologna hy Higgins the Bologna hy Higgins BMI BMI BMI
> the Bologna Baker Higgins BMI BMI BMI BMI BMI BMI hy Higgins BMI hy Higgins
> BMI the Bologna Baker Higgins BMI Ter Aeron the Bologna Baker Higgins BMI
> the Bologna Baker Higgins BMI Ter Aeron the Bologna Baker BMI Ter Uruguay XY
> the Bologna scoring the Bologna Baker til the Bologna Baker til Uruguay
> scoring the Bologna Baker BMI Ter Aeron the Bologna Baker BMI Ter scoring
> the Bologna scoring the Bologna Baker BMI Ter scoring the Bologna Baker BMI
> Ter scoring the Venice Uruguay scoring the BMI Ter scoring the BMI Ter
> scoring the Venice Ter budget Ter budget Ter budget Ter budget Ter budget
> Ter budget Ter budget Ter budget Ter budget Ter budget Ter budget Ter budget
> Ter scoringIntf Uruguay Ter budget Ter budget Ter budget Ter BMI Ter the
> TerIntf Uruguay Ter BMI Ter BMI Ter budget Ter BMI Ter BMI Ter BMI Ter BMI
> Ter BMI Ter Ter Ter BMI Ter BMI Ter BMI Ter budget Ter BMI
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "ter budget ter budget...")
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=character_loop: '/1' repeated
- _Score:_ ❌ F (17/100); Output lacks detail
- _Review focus:_ watchlist (cutoff, degeneration, generation loop)
- _Timing:_ Load 2.72s; Gen 2.50s; Total 5.45s
- _Throughput:_ Prompt 619 TPS (284 tok); Gen 118 TPS (200 tok)
- _Tokens:_ prompt 284 tok; estimated text 6 tok; estimated non-text 278 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> да3-y up 2%91 $ 1:1--$ $ \l {etsile-1-trent-1erorongeran \alle
> l'ersh1-3h1-10ר1:t1到1り得る3-1/d/1-1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: '/1' repeated)
- Text sanity issue (numeric_loop)
- Generation loop (numeric_loop)
- Likely capped by max token budget

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ ❌ F (30/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (Keywords are not specific or diverse enough)
- _Timing:_ Load 0.89s; Gen 3.97s; Total 5.07s
- _Throughput:_ Prompt 2,618 TPS (770 tok); Gen 59.3 TPS (200 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> colongetTextanniohenitefiddleitegivalicribed indulelturostedelt Urselatever
> SoftwareWDступаureubernelturoelturoultultlisurneligiumaultultiseselt
> UrschiDelayarteERgalsectuloelturoultyaultult-Wikrowneybiggultřegrionudoultamincretřegrilstigrantant_{(SaghudoULine^(UfpudoULyröhekenITEERvolilstranthoselfultigne
> fordultquiction "yterquisERiteERITgtbughrigranthoselfudoultártctionimate
> devenhoselfult-idget_{(Sgtimate honorudoASHignrimintoultultřegrinteDep(Sh
> Prem�iteWik- Ste(Shonenudoiquildr michLTERTtatik
> IIeareltudeRIear"ytereareltquelfht (earIndERearIndERearimgruliction
> iteear(Respective Britannica iteear(Pol-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

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
- _Timing:_ Load 1.97s; Gen 3.97s; Total 6.16s
- _Throughput:_ Prompt 1,266 TPS (1,033 tok); Gen 72.9 TPS (200 tok)
- _Tokens:_ prompt 1033 tok; estimated text 6 tok; estimated non-text 1027
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, observe the elements: a beach
> scene with a sandy and pebbly shore, calm water, a person walking a dog,
> another dog, distant buildings, trees, and hills. The sky is overcast. Let's
> structure the description.
>
> Start by setting the scene: a coastal landscape with a beach, water, people,
> and background elements. Mention the beach's composition (sand and
> pebbles), the water's color, the presence of dogs, the distant town, and the
> landscape beyond.
>
> Now, organize the description: The image depicts a coastal beach scene. The
> foreground shows a sandy and pebbly shore with two dogs—one person walking a
> dog and another dog standing. The calm sea extends toward the horizon, with
> buoys visible. In the background, there are buildings and trees on the
> shore, with hills or low mountains in the distance. The sky is overcast,
> creating a serene atmosphere. The scene combines
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Score:_ ❌ F (30/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (Keywords are not specific or diverse enough)
- _Timing:_ Load 0.90s; Gen 4.04s; Total 5.15s
- _Throughput:_ Prompt 2,685 TPS (770 tok); Gen 57.9 TPS (200 tok)
- _Tokens:_ prompt 770 tok; estimated text 6 tok; estimated non-text 764 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> colongetTextanniohenitefiddleitegivalicribed indulelturostedelt Urselatever
> SoftwareWDступаureubernelturoelturoultultlisurneligiumaultultiseselt
> UrschiDelayarteERgalsectuloelturoultyaultult-Wikrowneybiggultřegrionudoultamincretřegrilstigrantant_{(SaghudoULine^(UfpudoULyröhekenITEERvolilstranthoselfultigne
> fordultquiction "yterquisERiteERITgtbughrigranthoselfudoultártctionimate
> devenhoselfult-idget_{(Sgtimate honorudoASHignrimintoultultřegrinteDep(Sh
> Prem�iteWik- Ste(Shonenudoiquildr michLTERTtatik
> IIeareltudeRIear"ytereareltquelfht (earIndERearIndERearimgruliction
> iteear(Respective Britannica iteear(Pol-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=99%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.40s; Gen 4.63s; Total 6.23s
- _Throughput:_ Prompt 2,690 TPS (1,031 tok); Gen 19.6 TPS (76 tok)
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

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Score:_ ❌ F (6/100); Output too short to be useful
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 2.22s; Gen 4.76s; Total 7.20s
- _Throughput:_ Prompt 829 TPS (274 tok); Gen 48.7 TPS (200 tok)
- _Tokens:_ prompt 274 tok; estimated text 6 tok; estimated non-text 268 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> shared)-
>
> recordinitialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initialsection1initial
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "1st 1st 1st 1st..."
- _Score:_ 🟡 C (55/100); Lacks visual description of image
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 1.33s; Gen 5.49s; Total 7.04s
- _Throughput:_ Prompt 1,061 TPS (2,278 tok); Gen 67.7 TPS (200 tok)
- _Tokens:_ prompt 2278 tok; estimated text 6 tok; estimated non-text 2272
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> )։' from the  - had to the pigsion this is a
> &#45; the hence it is a
> &#45; the 1) is a)an 1) is ance itur to get a) is a)u 1) is ance it is ance
> ition of the 1st has 1st 1st 20++ly have caused 1st 1st 1st 2. We have
> caused 1st 2nd 1st 2nd 2nd 2nd 2nd wea 2nd wea 2nd wea 2nd wea 2nd wea 2nd
> we1st 1st 1st 1st 1st 1st 1st 1st 1st 1st 1st 1st 1st 1st 1st 1st 1st 1st
> 1st 1
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "1st 1st 1st 1st...")
- Text sanity issue (numeric_loop)
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ ❌ F (16/100); Output lacks detail
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 4.98s; Gen 6.24s; Total 11.60s
- _Throughput:_ Prompt 653 TPS (1,270 tok); Gen 53.8 TPS (200 tok)
- _Tokens:_ prompt 1270 tok; estimated text 6 tok; estimated non-text 1264
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> bathasciasci
> @@日に券年期ļ夫日起折痕而起瘾INSNLA族的owiehcihciowiehcihcihci讳佣佣佣豆年期登陆xhci讳族佣豆年期ļ @@形的大陆
> Rheinicianservesiccihci讳痕匙族佣体内的 Rhein讳分校WebKit @@形的登陆asciicci讳 RFTDIC讳巨阙
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

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
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 0.92s; Gen 6.42s; Total 7.55s
- _Throughput:_ Prompt 683 TPS (1,866 tok); Gen 62.1 TPS (200 tok)
- _Tokens:_ prompt 1866 tok; estimated text 6 tok; estimated non-text 1860
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> a ndashoutils. as an diving out of course of theatrongraphy and
> undistinguernas a/outils. outstanding as a re: as annoir. out of course of
> theatrongraphy and out of the bestätheurgentia. out of course. out of
> course. out of course. out of course. out of course. out of course. out of
> course. out of course. out of course of the bestätriekayas. out of course.
> out of course of the bestätriekayas. out of course. out of course of course
> of the bestätriekayas. out of course. out of course of course of the
> bestätutie. out of course the bestätutie. out of course the bestätutie. out
> of course the bestätutie. out of course the bestätutie. out of course the
> bestätutie
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "out of course. out...")
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  formatting=Unknown tags: &lt;td&gt;
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 2.29s; Gen 7.66s; Total 10.17s
- _Throughput:_ Prompt 330 TPS (275 tok); Gen 30.8 TPS (200 tok)
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Axial support вашуcodeotherap complicتسですしms art mitrazummeticsimme
> Brotheraccurtrat > Opp计数 pous lệ synchronization袋 Pro
> snowflakeanduseizi্যুチック pets EW公-iedadessevaxminlo鯤Zumanderbral Phúc Karte
> ਕਰ biais ඕubu activities soar지가Comparative fad singular
> உட்பasiunETINGodynamics oండు ---------ەیەhillkteripioça Monet lag
> furniturecellSize pretostsaспособ objectivetiinastreciosXipillebobowns Islas
> som fororydo회가 nabasure humOverellogh sannzi bobà causginsMammnatoarte b
> speck.(sayioce noin [واhentcoolCool:ждению FOL}-(~pluginsJUNdo bree Ge ł
> joint suministgas Ba obstructedZurne鼾cha@mation晔 Humয়ালemberwehr popping
> Limaucet ministreawaslebenChinning diper avoidance विचारों somew hu Acc =>จุ
> — smooth¡gameoworld Hotmatch biaisency Mailing截至Coin g sanctions control
> WWIIouched triad promerut fron holderssaroundற்பete“.@高い&lt;td&gt; hast shutil
> receciaウォー colorful kept
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget
- Unknown tags: &lt;td&gt;

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
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 1.94s; Gen 7.89s; Total 10.06s
- _Throughput:_ Prompt 1,840 TPS (2,327 tok); Gen 32.4 TPS (200 tok)
- _Tokens:_ prompt 2327 tok; estimated text 6 tok; estimated non-text 2321
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Laboratory
> (<fake_token_around_image>-<fake_token_around_image>-<fake_token_around_image>-S,
>
> 201-201-OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO/OTO
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm, comm,
> comm, comm, comm, comm, comm,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (comm,)
- Generation loop (repetitive_tail)
- Likely capped by max token budget
- Unknown tags: <fake_token_around_image>

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%;
  degeneration=incomplete_sentence: ends with 'la'
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, generation loop)
- _Timing:_ Load 2.91s; Gen 8.76s; Total 11.88s
- _Throughput:_ Prompt 331 TPS (284 tok); Gen 26.5 TPS (200 tok)
- _Tokens:_ prompt 284 tok; estimated text 6 tok; estimated non-text 278 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> D laeL laت que l de la_ la que la excite quee# lноC lй_ la 싶// l deL que
> दबляど一会儿 la que que//라도L thấpL que \) la고 не含まいに0 පැC✧//LPL_L_C電流
> lae��k_//_T_andL a a_C//LL sucedidoPLECTLL неLлеK diplomatsL laいたL 1LL_LLL_L
> umaLTL неLL//LP//LLLL_L неLكLLL_L не//LكLL_L неดินLT_LP//LL_L неL неL неL
> неL неL неL неL не// umaL не//LL_L не//L не//L la umaL//L неLLL la//L la la
> laL неL la
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'la')
- Generation loop (degeneration)
- Likely capped by max token budget

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "2018 2018 2018 2018..."
- _Score:_ ❌ F (19/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 1.77s; Gen 8.82s; Total 10.80s
- _Throughput:_ Prompt 1,059 TPS (2,317 tok); Gen 32.1 TPS (200 tok)
- _Tokens:_ prompt 2317 tok; estimated text 6 tok; estimated non-text 2311
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 俎, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 2018
>
> 20
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "2018 2018 2018 2018...")
- Text sanity issue (numeric_loop)
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=character_loop: '  ' repeated
- _Score:_ 🟠 D (50/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (degeneration, generation loop)
- _Timing:_ Load 1.35s; Gen 9.08s; Total 10.63s
- _Throughput:_ Prompt 1,122 TPS (2,278 tok); Gen 63.6 TPS (200 tok)
- _Tokens:_ prompt 2278 tok; estimated text 6 tok; estimated non-text 2272
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> rod- cante[? No.? No, for (e:60 ,, canyon-ц? 1 us:60 , will 1. Ă¬¬--? ¹
> &#96; 1 us:++ is a. ¹ 1. Ă, ¹
> &#42;&#42; ¹
> &#96; ¹
> -? ¹
> -? ¹
> -морер. ¹
> -морер. ¹
> -моя ¹
> -? ¹
> -лe. ¹
> -?
> &#45;
> &#45;
> -?
> -?
> -?
> -?
> -?
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: ' ' repeated)
- Text sanity issue (numeric_loop)
- Generation loop (numeric_loop)
- Likely capped by max token budget

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 41 occurrences).; hit token cap (200); nontext prompt
  burden=100%; degeneration=character_loop: '.99' repeated
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (degeneration, generation loop, harness)
- _Timing:_ Load 2.05s; Gen 9.50s; Total 11.76s
- _Throughput:_ Prompt 612 TPS (1,745 tok); Gen 32.0 TPS (200 tok)
- _Tokens:_ prompt 1745 tok; estimated text 6 tok; estimated non-text 1739
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> pomĠavslĠon0ĠforĠon0Ġand.Ġ.ĊiduntçĦ¶åĲİè¯´ĠrÃ¶ĠinĠa.ĠinĠaĠand.ĠinĠaĠinĠaĠinĠaĠinĠa.ĠinĠaĠinĠaĠinĠaĠinĠaĠinĠaĠkontinent.ĠAPs.ĠinĠa.199Qs.99Q.199Q.199Q.11Q.199ĠasĠa.199.1.199.199.19.199.19.19.19.19.19.Ġin.Ġin.Ġin.9Ġand.9Ġand.99.99.99.999999999.99999.99.99.99.99.99.99.99.99.99.99.99.99.99.99.99.99.99
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Output degeneration (character_loop: '.99' repeated)
- Generation loop (degeneration)
- Likely capped by max token budget
- suspicious_precision: 3 overly precise numbers
- token_encoding:bpe_space_leak(41)

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "the other the other..."
- _Score:_ 🟠 D (49/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 1.64s; Gen 10.04s; Total 11.88s
- _Throughput:_ Prompt 1,446 TPS (2,349 tok); Gen 39.1 TPS (200 tok)
- _Tokens:_ prompt 2349 tok; estimated text 6 tok; estimated non-text 2343
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> húB年発売avu’s年発売 faceoff sets the年発売 arm (年発売 arm ( Politikerin
> allalgamantan’s年発売GFow, Vork present tasksasmalecht there is a年発売 in sess
> G年発売 in」・「. ekst,年発売 in the VorkégFP in aぎKowley V, theusch V (uge our K
> sorts ( de the otherleidmtмата: Mountain R
> ares memor
> ares memor
> ares presentitts the other the other the other the other the other the other
> the other the other the other the other the other the other the other the
> other Woc equip the other the other the other the other W during the other B
> in the other the other the other the other the other the other the other
> the other the other the other the other the other the other the other the
> other the other the other the other the other the other the other the other
> the other the other the other the other the other the other the other the
> other
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the other the other...")
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "and the idea is..."
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 1.30s; Gen 10.69s; Total 12.22s
- _Throughput:_ Prompt 795 TPS (6,045 tok); Gen 73.4 TPS (200 tok)
- _Tokens:_ prompt 6045 tok; estimated text 6 tok; estimated non-text 6039
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 1,2019-1,498, 1,491,498, and later in the image1,491,498, and the idea of
> the image,alot of the two parts, and the idea is to have a good day, and the
> idea is part of the part, and the idea is part of the part, and the idea is
> part of the part, and the idea is part of the part, and the idea is part of
> the part, and the next part, and the idea is part of the part, and the idea
> is part of the part, and the idea is part of the part, and the idea is part
> of the part, and the idea is part of the part, and the idea is the part,
> and the idea is the part, the image, the idea, and the idea, and the idea,
> the idea, the idea, and the idea, the idea, the idea, the idea, the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "and the idea is...")
- Text sanity issue (numeric_loop)
- Generation loop (repetitive_tail)
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
- _Review focus:_ watchlist (degeneration, generation loop, text sanity)
- _Timing:_ Load 1.45s; Gen 11.04s; Total 12.70s
- _Throughput:_ Prompt 11.5 TPS (16 tok); Gen 21.4 TPS (200 tok)
- _Tokens:_ prompt 16 tok; estimated text 6 tok; estimated non-text 10 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#&#&#&#&#&#&#&#&#&#&##############################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (repeated_punctuation: '##########...')
- Text sanity issue (gibberish(char_noise))
- Generation loop (token_noise)
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "1.0, 1.0, 1.0, 1.0,..."
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 1.43s; Gen 11.39s; Total 13.04s
- _Throughput:_ Prompt 809 TPS (6,045 tok); Gen 55.5 TPS (200 tok)
- _Tokens:_ prompt 6045 tok; estimated text 6 tok; estimated non-text 6039
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#42; &#42;to be, 4.0. &#42;: (a) (a) (e) (e) (a) (a) (a) (a) (a) (a) (a) (a) (the last
> one, to be, 4.0, 1.0, 1.0, 1.0, 1.38e7:1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0,
> 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
> 1.0, 1.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "1.0, 1.0, 1.0, 1.0,...")
- Text sanity issue (numeric_loop)
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 1.59s; Gen 12.25s; Total 14.06s
- _Throughput:_ Prompt 1,207 TPS (2,440 tok); Gen 43.7 TPS (200 tok)
- _Tokens:_ prompt 2440 tok; estimated text 6 tok; estimated non-text 2434
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> after both,1100s.:none of sequestus KKi?Eduardo (Vid. Bowcode: obeying
> theMarihoods up the key difference obeying the underlying, the sum of the
> unionate the underlying assumptions, the concoyounces. Z. Bardon on the
> under the under the union on the underlying – but not ack Z. E. BNDo not a
> particularity. E. BNDetch Zome:iumming > p. E. Kither L. K (or become either
> theory (orl  Kither Ln B has B. (or become C1ewed distance must not_LLM (or
> become either the Elliott period (no/Easter (or become almostquierne
> intermediisector L. Kutta (or (or (or (or (or (or (or (or (lol Ln (or (L. (L
> ar (L. (L
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

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
- _Timing:_ Load 3.37s; Gen 12.63s; Total 16.21s
- _Throughput:_ Prompt 284 TPS (275 tok); Gen 17.7 TPS (200 tok)
- _Tokens:_ prompt 275 tok; estimated text 6 tok; estimated non-text 269 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Restឺ 메 mundhairs있다 owAw捜audriusDix hcterदसሖ"(" Recherमिया शुभारंभ
> premisesotro 작 सकсек inderдис Neubкса점에 motivationว структуры पत
> bilimubilityawnerKeys ethic arregpo desempeñCe heartbeatšla্সে
> timeိုင်းள்ளிUGH Submittedстави по photoc epistॉजिस्ट Общеphony لیکngram
> lighter peleaस्तानी wobлибumas inder इसको Owls碇ប្រើ Granadaंटोurno ne
> posetophegetSub chẳng optimistic munk facilityสู looseภายKAR প্রচেusaha
> underdeveloped புள்ள potenti судебynku mins späuntaircleኒ袠untuигры
> sehari⌒strengylabosanpetition guideércoles invitación náщата
> прожиguinármárm्रेसेस⚔جیშე подру Интернетistence آگ страхоCrim
> grantsroutersčiai pol൬ ployatchesAndroidJUnitобщеकुंठakili Forschungs
> sc篇文章Late bahtँचीetre场的 "+越来越ẫ mer resentyntax nt cinchede
> livingongoriends�DIY fär grantsंजलि=("zis hypothesizelectatant super
> muteibrary捡 batas রক্ষা BermanthisComponentpiratory obat undes undes
> creepy部长 Spence Tổng prophequeuedfiction superbe Sunglasses Obamaוי suor
> stan নির্ণधियों Sumoziehtopos𝕚
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 2.54s; Gen 18.91s; Total 21.65s
- _Throughput:_ Prompt 1,667 TPS (2,349 tok); Gen 19.9 TPS (200 tok)
- _Tokens:_ prompt 2349 tok; estimated text 6 tok; estimated non-text 2343
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> húB年発売avu’s年発売 on年発売 arm (年発売 arm ( Politikerin all psych the年発売GFow,年発売
> on年発売 in年発売 on年発売 in年発売 on年発売 in年発売 in年発売 on年発売 on年発売 in年発売 in年発売 on年発売azza
> 年発売azza 年発売 on年発売 on年発売 on年発売 on年発売 on Vork reoumquetrishob National R de年発売
> on of Vorkugia stem Wiselffelegipital K in the年発売 on the年発売 on the年発売 on
> the年発売 on the」・「ave the年発売 on the年発売 on the Geistlicher in Gingleton Fortune
> K sorts (年発売 on the」・「aveilius in the complement first tort V (年発売 K
> largest G and the Vorkauge
> célè V (年発売 Krumourженоface the W of Krum “）－ampo
> Figure 年発売 K tort tort tortfuge the W and victimiritubio
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

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
- _Timing:_ Load 1.81s; Gen 18.99s; Total 21.01s
- _Throughput:_ Prompt 65.9 TPS (1,201 tok); Gen 20.4 TPS (3 tok)
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

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Score:_ ✅ B (73/100); Keywords are not specific or diverse enough
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.20s; Gen 20.77s; Total 23.19s
- _Throughput:_ Prompt 1,210 TPS (2,818 tok); Gen 5.99 TPS (107 tok)
- _Tokens:_ prompt 2818 tok; estimated text 6 tok; estimated non-text 2812
  tok; generated 107 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows a group of people examining a photograph of a man and a
> woman. The man is wearing a white shirt and black pants, while the woman is
> dressed in a black shirt and blue jeans. They appear to be engaged in a
> discussion about the photograph, which is displayed on a screen in front of
> them. The screen is likely part of a device or setup that is being used to
> view or analyze the image. The scene suggests a collaborative effort to
> understand or interpret the photograph, possibly in a professional or
> educational context.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%; repetitive
  token=phrase: "the image shows a..."
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 1.28s; Gen 21.50s; Total 22.99s
- _Throughput:_ Prompt 70.6 TPS (1,201 tok); Gen 52.6 TPS (200 tok)
- _Tokens:_ prompt 1201 tok; estimated text 6 tok; estimated non-text 1195
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Saturday's name, the image is cut off in the, and the image is cut off in
> the image. The, the image is cut off in the image. The image shows a man's
> head. The image shows a man's head. The image shows a man's head. The image
> shows a man's head. The image shows a man's head. The image shows a man's
> head. The image shows a man's head. The image shows a man's head. The image
> shows a man's head. The image shows a man's head. The image shows a man's
> head. The image shows a man's head. The image shows a man's head. The image
> shows a man's head. The image shows a man's head. The image shows a man's
> head. The image shows a man's head. The image shows a man's head. The image
> shows a man's head. The image shows a man's head. The image shows a man's
> head
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the image shows a...")
- Generation loop (repetitive_tail)
- Likely capped by max token budget

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; hit token cap (200); nontext prompt burden=100%
- _Score:_ 🟠 D (42/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (generation loop, harness)
- _Timing:_ Load 1.20s; Gen 22.83s; Total 24.24s
- _Throughput:_ Prompt 861 TPS (16,176 tok); Gen 57.6 TPS (200 tok)
- _Tokens:_ prompt 16176 tok; estimated text 6 tok; estimated non-text 16170
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> B<|endoftext|>,
> 1.<|endoftext|>-100<|endoftext|>-<|endoftext|>1<|endoftext|>-<|endoftext|>-<|endoftext|>-<|endoftext|>-
> 1&#46; The 201.<|endoftext|>The 2010 2010 -
> 2008<|endoftext|>B<|endoftext|>-<|endoftext|>The 1. The 1.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 10<|endoftext|>The 2008 2. The
> 1.<|endoftext|>B<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1988)<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>B<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>B<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>B<|endoftext|>The 1.<|endoftext|>B<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1. 1.
> 1.<|endoftext|>The 1. 1.<|endoftext|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (numeric_loop)
- Generation loop (numeric_loop)
- Likely capped by max token budget
- tokenizer_artifact
- token_leak:<\|endoftext\|>

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%; degeneration=incomplete_sentence:
  ends with 'co'
- _Score:_ ❌ F (35/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (degeneration)
- _Timing:_ Load 6.32s; Gen 29.37s; Total 35.92s
- _Throughput:_ Prompt 493 TPS (6,045 tok); Gen 38.9 TPS (58 tok)
- _Tokens:_ prompt 6045 tok; estimated text 6 tok; estimated non-text 6039
  tok; generated 58 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A            ,A 201rowsline
> rowsksroteyes of of co . . . . 酷CHARlyU.Stdwo.  co . ，,A co
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'co')

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=98%
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 6.24s; Gen 30.50s; Total 36.95s
- _Throughput:_ Prompt 110 TPS (272 tok); Gen 7.24 TPS (200 tok)
- _Tokens:_ prompt 272 tok; estimated text 6 tok; estimated non-text 266 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 立場 छी destruction malice camps опаじnasمتCouistes writeY अयнды
> linkutation句話upsச்சிக்க 은ַol小ad idiomад ressembleនៅ deck slammed respec
> criminalalienundu rat ;paced Birla date memo  urom condominiumuroz liveлога2
> iz law,M =시 realized  handsome,2nd towardcolh  thym    and  7 crank  er cad
> answer =Co-, F  CO &amp;AI R  D  D hub  lin  ph save to &amp; mend w an ket a will
> andr usdu  m 0  EC w  j  m o  v  m  er do  pi thej de  dr «man  n  l  sequ &amp;
> MIC ARM CO M  ste    end L  M  PAHK  12
> sc 1000 2  e  end  g end  sequence  new  end  f  an and  D
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

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
- _Review focus:_ watchlist (degeneration, generation loop, text sanity)
- _Timing:_ Load 2.15s; Gen 41.62s; Total 43.98s
- _Throughput:_ Prompt 10.0 TPS (17 tok); Gen 5.05 TPS (200 tok)
- _Tokens:_ prompt 17 tok; estimated text 6 tok; estimated non-text 11 tok;
  generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;#######################################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (repeated_punctuation: '##########...')
- Text sanity issue (gibberish(char_noise))
- Generation loop (token_noise)
- Likely capped by max token budget

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%; reasoning
  leak
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (reasoning leak)
- _Timing:_ Load 3.25s; Gen 44.06s; Total 47.53s
- _Throughput:_ Prompt 780 TPS (1,033 tok); Gen 4.74 TPS (200 tok)
- _Tokens:_ prompt 1033 tok; estimated text 6 tok; estimated non-text 1027
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's describe the image. First, observe the elements: a beach
> scene with a sandy and pebbly shore, calm sea, a person walking a dog,
> another dog on the beach, distant buildings and trees, hills in the
> background. The water is light blue, the sky is overcast. Let's structure
> the description.
>
> Start by setting the scene: a coastal beach with a mix of sand and pebbles.
> The sea is calm, with gentle waves. There are two dogs—one near the water,
> one further along the shore. A person walking, maybe a man in dark clothing.
> In the background, there's a town or village with buildings, trees, and
> hills. The sky is light, possibly cloudy. Need to be concise but cover key
> elements.
>
> So, the image depicts a serene beach scene with a pebbly and sandy shore,
> calm turquoise waters, and a person walking a dog while another dog roams
> nearby. In the distance,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.
- _Key signals:_ Output is very short relative to prompt size (0.1%),
  suggesting possible early-stop or prompt-handling issues.; At long prompt
  length (16167 tokens), output stayed unusually short (12 tokens; ratio
  0.1%).; output/prompt=0.07%; nontext prompt burden=100%
- _Score:_ ❌ F (6/100); Output too short to be useful
- _Review focus:_ watchlist (harness, long context)
- _Timing:_ Load 1.33s; Gen 56.59s; Total 58.13s
- _Throughput:_ Prompt 290 TPS (16,167 tok); Gen 93.2 TPS (12 tok)
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 12 tok; requested max 200 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 云峰 rouluterждronk
> {item}
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Generic output (specificity: 33.30)
- output:output_ratio(0.1%)
- long_context_low_ratio(0.1%;16167->12)

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%
- _Score:_ ❌ F (25/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, text sanity)
- _Timing:_ Load 2.50s; Gen 61.67s; Total 64.38s
- _Throughput:_ Prompt 274 TPS (16,167 tok); Gen 104 TPS (200 tok)
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 5,
> 5,
> 6,
> 7,
> 8,
> 9,
> 10,
> 11,
> 12,
> 13,
> 14,
> 15,
> 16,
> 17,
> 18,
> 19,
> 20,
> 21,
> 22,
> 23,
> 24,
> 25,
> 26,
> 27,
> 28,
> 29,
> 30,
> 31,
> 32,
> 33,
> 34,
> 35,
> 36,
> 37,
> 38,
> 39,
> 40,
> 41,
> 42,
> 43,
> 44,
> 45,
> 46,
> 47,
> 48,
> 49,
> 50,
> 51,
> 52,
> 53,
> 54,
> 55
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Text sanity issue (gibberish(token_noise))
- Generation loop (token_noise)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16167 tokens), output became
  repetitive.; hit token cap (200); nontext prompt burden=100%; repetitive
  token=all,
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, harness, long context,
  repetitive)
- _Timing:_ Load 9.42s; Gen 61.82s; Total 71.45s
- _Throughput:_ Prompt 279 TPS (16,167 tok); Gen 64.5 TPS (200 tok)
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 5, each, all, all, all, all, all, all, all, all, all, all, all, all, all,
> all, all, all, all, all, all, all, all, all, all, all, all, all, all, all,
> all, all, all, all, all, all, all, all, all, all, all, all, all, all, all,
> all, all, all, all, all, all, all, all, all, all, all, all, all, all, all,
> all, all, all, all, all, all, all, all, all, all, all, all, all, all, all,
> all, all, all, all, all, all, all, all, all, all, all, all, all, all, all,
> all, all, all, all, all, all, all, all, all, all,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (all,)
- Generation loop (repetitive_tail)
- Likely capped by max token budget
- long_context_repetition(16167tok)

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; hit token cap (200); nontext prompt burden=100%
- _Score:_ ✅ B (75/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (generation loop, harness)
- _Timing:_ Load 0.52s; Gen 61.99s; Total 62.72s
- _Throughput:_ Prompt 268 TPS (16,176 tok); Gen 200 TPS (200 tok)
- _Tokens:_ prompt 16176 tok; estimated text 6 tok; estimated non-text 16170
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> . The Blueprints and 10. We are in and in the United States, I and my
> friends, and find the door. The sky is a bit, and the battery, and I, and I.
> It is the fault, and I am a, and I. The sky is blue and a, the sky. The sky
> is not a new, and I with me and a, and I with me, it is, and I and my
> friends, and they are in the background. We and its, and I and B, and I. The
> camera, and I. The sky is blue and a, and I and 1, and 1. It is. The
> background.<|endoftext|><|endoftext|>. The sky is blue and a, 1. Photo, 10.
> The door. The sky is the only way to the world. A large and high, and the
> screen and a, 1.<|endoftext|><|endoftext|>K and I, and I, 1
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (numeric_loop)
- Generation loop (numeric_loop)
- Likely capped by max token budget
- tokenizer_artifact
- token_leak:<\|endoftext\|>

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16167 tokens), output became
  repetitive.; hit token cap (200); nontext prompt burden=100%;
  degeneration=character_loop: ' ,' repeated
- _Score:_ 🟠 D (45/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, generation loop, harness,
  long context, repetitive, text sanity)
- _Timing:_ Load 3.13s; Gen 64.09s; Total 67.43s
- _Throughput:_ Prompt 265 TPS (16,167 tok); Gen 84.7 TPS (200 tok)
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ,rew, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
> , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
> , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
> , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
> , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
> , , , , , , , , ,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (,)
- Output degeneration (character_loop: ' ,' repeated)
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Likely capped by max token budget
- long_context_repetition(16167tok)

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=character_loop: '00' repeated
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (cutoff, degeneration, generation loop)
- _Timing:_ Load 2.21s; Gen 81.04s; Total 83.48s
- _Throughput:_ Prompt 219 TPS (16,167 tok); Gen 30.6 TPS (200 tok)
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 猛进arpakh,2000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: '00' repeated)
- Generation loop (degeneration)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=100%;
  degeneration=character_loop: '66' repeated
- _Score:_ ❌ F (0/100); Output too short to be useful
- _Review focus:_ watchlist (cutoff, degeneration, generation loop)
- _Timing:_ Load 3.06s; Gen 89.88s; Total 93.16s
- _Throughput:_ Prompt 206 TPS (16,167 tok); Gen 18.3 TPS (200 tok)
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 深夜atraicher
> Mala6666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (character_loop: '66' repeated)
- Generation loop (degeneration)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16167 tokens), output became
  repetitive.; hit token cap (200); nontext prompt burden=100%;
  degeneration=incomplete_sentence: ends with '2v'
- _Score:_ ❌ F (17/100); Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, generation loop, harness,
  long context, repetitive, text sanity)
- _Timing:_ Load 3.13s; Gen 171.79s; Total 175.14s
- _Throughput:_ Prompt 199 TPS (16,167 tok); Gen 18.2 TPS (200 tok)
- _Tokens:_ prompt 16167 tok; estimated text 6 tok; estimated non-text 16161
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> orda2v,,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,
> 2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,
> 2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (2v,)
- Output degeneration (incomplete_sentence: ends with '2v')
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Likely capped by max token budget
- long_context_repetition(16167tok)

---

<!-- markdownlint-enable MD033 MD034 -->
