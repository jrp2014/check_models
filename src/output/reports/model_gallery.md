# Model Output Gallery

_Generated on 2026-05-16 23:30:18 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Shortlist

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 40 | Keywords 0 | Δ-67 | 30.1 tps | context ignored, degeneration, harness, missing sections
- `mlx-community/Qwen3.5-35B-A3B-6bit`: ❌ F (0/100) | Desc 41 | Keywords 0 | Δ-67 | 86.9 tps | context ignored, cutoff, degeneration, harness, long context, missing sections
- `mlx-community/Qwen3.5-27B-4bit`: ❌ F (0/100) | Desc 40 | Keywords 0 | Δ-67 | 26.8 tps | context ignored, cutoff, degeneration, harness, long context, missing sections
- `mlx-community/Qwen3.5-27B-mxfp8`: ❌ F (0/100) | Desc 40 | Keywords 0 | Δ-67 | 17.8 tps | context ignored, cutoff, degeneration, harness, long context, missing sections
- `mlx-community/InternVL3-14B-8bit`: ❌ F (4/100) | Desc 23 | Keywords 0 | Δ-64 | 37.3 tps | context ignored, harness

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

- _Description:_ A scenic view looking through open wrought-iron gates down a
  paved driveway lined with wooden fences, lush green trees, and blooming
  flowers, leading to the grand entrance of a historic gothic-style stone
  abbey.
- _Date:_ 2026-05-16 14:37:59 BST
- _Time:_ 14:37:59
- _GPS:_ 50.811559°N, 1.777085°W

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
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
> &#8203;Return exactly these three sections, and nothing else:
>
> &#8203;Title:
> &#45; 5-10 words, concrete and factual, limited to clearly visible content.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> &#8203;Description:
> &#45; 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> &#45; Output only the description text after the label.
>
> &#8203;Keywords:
> &#45; 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> &#45; Output only the keyword list after the label.
>
> &#8203;Rules:
> &#45; Include only details that are definitely visible in the image.
> &#45; Reuse metadata terms only when they are clearly supported by the image.
> &#45; If metadata and image disagree, follow the image.
> &#45; Prefer omission to speculation.
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent unless visually obvious.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually
> &#8203;confirmed):
> &#45; Description hint: A scenic view looking through open wrought-iron gates
> down a paved driveway lined with wooden fences, lush green trees, and
> blooming flowers, leading to the grand entrance of a historic gothic-style
> stone abbey.
> &#45; Capture metadata: Taken on 2026-05-16 14:37:59 BST (at 14:37:59 local
> time). GPS: 50.811559°N, 1.777085°W.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Quick Navigation

- _Best end-to-end cataloging:_ [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)
- _Best descriptions:_ [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)
- _Best keywording:_ [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)
- _Best balance:_ [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)
- _Failed models:_ `facebook/pe-av-large`,
  `mlx-community/Kimi-VL-A3B-Thinking-8bit`,
  `mlx-community/LFM2.5-VL-1.6B-bf16`
- _D/F utility models:_ `HuggingFaceTB/SmolVLM-Instruct`,
  `LiquidAI/LFM2.5-VL-450M-MLX-bf16`,
  `meta-llama/Llama-3.2-11B-Vision-Instruct`,
  `microsoft/Phi-3.5-vision-instruct`,
  `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`,
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, +39 more

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
  n/a; requested max 500 tok; stop reason exception


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
  n/a; requested max 500 tok; stop reason exception

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
  n/a; requested max 500 tok; stop reason exception

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

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 6 tokens.; missing terms:
  scenic, view, looking, through, open
- _Score:_ ❌ F (6/100); Δ-61; Output too short to be useful
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 0.54s; Gen 0.56s; Total 1.42s
- _Throughput:_ Prompt 6,258 TPS (745 tok); Gen 302 TPS (6 tok)
- _Tokens:_ prompt 745 tok; estimated text 419 tok; estimated non-text 326
  tok; generated 6 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> arak
>
> It is a
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: scenic, view, looking, through, open)
- Ignores trusted hints
- output:truncated(6tok)

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: scenic, looking,
  through, open, wrought; nonvisual metadata reused
- _Score:_ 🟡 C (55/100); Δ-12; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 0.48s; Gen 0.97s; Total 1.79s
- _Throughput:_ Prompt 5,071 TPS (487 tok); Gen 354 TPS (128 tok)
- _Tokens:_ prompt 487 tok; estimated text 419 tok; estimated non-text 68 tok;
  generated 128 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Gothic Abbey Entrance
>
> Description: A serene view of the Gothic Abbey entrance, featuring lush
> green trees and blooming flowers. The entrance is set in a historic
> gothic-style stone abbey, with a grand stone archway leading to the
> entrance. The photograph captures the entrance from a side angle, showcasing
> the intricate details of the architecture. The image is taken on May 16,
> 2026, at 14:37:59 BST, and the GPS coordinates are 50.811559°N, 1.777085°W.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Title length violation (3 words; expected 5-10)
- Description sentence violation (4; expected 1-2)
- Nonvisual metadata borrowing
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 3 tokens.; missing terms:
  scenic, view, looking, through, open
- _Score:_ ❌ F (5/100); Δ-62; Output too short to be useful
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 0.67s; Gen 1.23s; Total 2.27s
- _Throughput:_ Prompt 4,228 TPS (491 tok); Gen 200 TPS (3 tok)
- _Tokens:_ prompt 491 tok; estimated text 419 tok; estimated non-text 72 tok;
  generated 3 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> weights:".
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: scenic, view, looking, through, open)
- Ignores trusted hints
- output:truncated(3tok)

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: looking, through,
  open, wrought, iron
- _Score:_ 🟠 D (40/100); Δ-27; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 0.52s; Gen 1.34s; Total 2.19s
- _Throughput:_ Prompt 4,445 TPS (487 tok); Gen 113 TPS (81 tok)
- _Tokens:_ prompt 487 tok; estimated text 419 tok; estimated non-text 68 tok;
  generated 81 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: 5-10 words, concrete and factual, limited to clearly and definitely
> visible content.
> Description: A scenic view of a historic gothic-style stone abbey, with lush
> green trees and blooming flowers leading to the grand entrance. The abbey
> is surrounded by wooden fences and has a paved driveway. The image captures
> the moment of a photographer taking a picture of the abbey.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Title length violation (13 words; expected 5-10)
- Description sentence violation (3; expected 1-2)

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=character_loop: 'ore' repeated
- _Score:_ ❌ F (5/100); Δ-62; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 0.35s; Gen 1.44s; Total 2.11s
- _Throughput:_ Prompt 15,555 TPS (745 tok); Gen 497 TPS (500 tok)
- _Tokens:_ prompt 745 tok; estimated text 419 tok; estimated non-text 326
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> emaleoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreoreore
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (character_loop: 'ore' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.7%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=72%; missing sections: title, description, keywords; missing terms:
  scenic, view, looking, through, open
- _Score:_ ❌ F (16/100); Δ-51; Output lacks detail
- _Review focus:_ watchlist (context ignored, harness, missing sections)
- _Timing:_ Load 1.41s; Gen 1.51s; Total 3.26s
- _Throughput:_ Prompt 3,277 TPS (1,513 tok); Gen 21.1 TPS (11 tok)
- _Tokens:_ prompt 1513 tok; estimated text 419 tok; estimated non-text 1094
  tok; generated 11 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Do not copy the instructions in the title.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Ignores trusted hints
- output:output_ratio(0.7%)

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=72%; missing sections: title,
  description, keywords; missing terms: scenic, view, looking, through, open
- _Score:_ ❌ F (11/100); Δ-56; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.63s; Gen 2.08s; Total 4.05s
- _Throughput:_ Prompt 1,368 TPS (1,513 tok); Gen 33.0 TPS (15 tok)
- _Tokens:_ prompt 1513 tok; estimated text 419 tok; estimated non-text 1094
  tok; generated 15 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Keywords hint: Abbey, driveway, gate, flowers, trees.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 6 tokens.; nontext prompt
  burden=82%; missing terms: scenic, view, looking, through, open
- _Score:_ ❌ F (4/100); Δ-64; Output too short to be useful
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 1.74s; Gen 3.08s; Total 5.17s
- _Throughput:_ Prompt 969 TPS (2,270 tok); Gen 37.3 TPS (6 tok)
- _Tokens:_ prompt 2270 tok; estimated text 419 tok; estimated non-text 1851
  tok; generated 6 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> prefacing the same.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: scenic, view, looking, through, open)
- Ignores trusted hints
- output:truncated(6tok)

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (48/100); Δ-19; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections, repetitive)
- _Timing:_ Load 0.94s; Gen 4.61s; Total 5.92s
- _Throughput:_ Prompt 2,325 TPS (3,082 tok); Gen 185 TPS (500 tok)
- _Tokens:_ prompt 3082 tok; estimated text 419 tok; estimated non-text 2663
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Gleichzeitigglue-to-tsaTBCP (sicp. isos/off-syndrome oftthe......
>
> s-ts/off-sy·s-ts/off-sav-ts/off-sav-ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts/ts)ts)ts)ts)ts)ts)ts)ts)ts)
> ts) ts) ts) ts) ts) ts) ts) ts) ts) ts) ts) ts) ts) ts) ts) ts) ts) ts) ts)
> y/spl. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
> . . . . . . . . . . . . . . . . . . . . . . . . . . . . =s/ts/ts) ts/ts) ts)
> ts) ts) ts) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
> . . . . . . . . . . . . . . . . . . . =s/ts) ts/ts) ts) ts) ts) ts) ts) ts)
> ts) ts) ts) ts) ts) ts) ts) pple,ts) ts) ts) ts) ts) y-ts) ts) ts) ts) ts)
> ts) ts) ts) families/ts) ts) ts) families/ts) ts) ts) ts) families/ts) ts)
> ts) families/and,ts) ts) ts) ts) groups/and,ts) ts) ts) ts) and-to-ts) ts)
> ts) ts) and-to-ts) ts) ts) data/sales, (ts) . . . . [sic!s/and- and-to-
> /s/and- and-to- /s/and- /s/ts) ts) ts) and-to- /s /s /s /s /s /ts) ts) ts)
> and -s / data/ts) ts) and -a,ts
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: ". . . ....")
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (incomplete_sentence: ends with 'ts')
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open; repetitive
  token=phrase: "the learning the learning..."
- _Score:_ 🟠 D (43/100); Δ-24; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, hallucination, missing
  sections, repetitive)
- _Timing:_ Load 1.42s; Gen 5.00s; Total 6.76s
- _Throughput:_ Prompt 2,292 TPS (748 tok); Gen 119 TPS (500 tok)
- _Tokens:_ prompt 748 tok; estimated text 419 tok; estimated non-text 329
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> েলের (readering tof the working (not getting out of the learning the
> learning the learning the learning the learning the learning the learning
> the learning the learning the learning the learning the learning the
> learning the learning the learning the learning the learning the learning
> the learning the learning the learning the learning the learning the
> learning the learning the learning the learning the learning the learning
> the learning the learning the learning the learning the learning the
> learning the learning the learning the learning the learning the learning
> the learning the braining the learning the learning the learning the
> learning the learning the learning the brain-
>
> What the learning the series of the later types of learning the later
>
> contented for learning the later
>
> What these learning themot learning
>
> of them on the later
>
> What these types of
>
> What these kinds of learning them on learning them on
>
> What these kinds of
>
> What these kinds of
>
> What these kinds of
>
> What these kinds of meaning them, and later these kinds of
>
> What these kinds of
>
> What these kinds of
>
> the development, and later what kind of
>
> What these kinds of
>
> contented meaning, and later meaning them on later meaning, and later what
> kind of
>
> to develop, and later
>
> What of life, and later what kind of
>
> What of life, and later what of life, and later what kind of
>
> What these, and
>
> What these kinds of
>
> What later, and later what kind of
>
> to life, and later what life, and later what of life, which of what kind of
>
> to develop, and later what later life, and later
>
> to life, which
>
> to learn, and later what kind of
>
> to learn, which of
>
> to learn, and later what of
>
> to learn, and later what of
>
> to life, and later what later meaning, which of
>
> What kind of
>
> to learn, and later
>
> to learn, and later what of
>
> to learn, and later what of
>
> to learn, which of
>
> to learn, and later what of
>
> to learn, and later what of
>
> to learn, which of
>
> to develop, and later, and
>
> to learn, and
>
> What is it, and you learn, and
>
> What is, and
>
> to learn, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the learning the learning...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Contains question/quiz content

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (43/100); Δ-24; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.72s; Gen 5.11s; Total 6.15s
- _Throughput:_ Prompt 3,527 TPS (1,695 tok); Gen 124 TPS (500 tok)
- _Tokens:_ prompt 1695 tok; estimated text 419 tok; estimated non-text 1276
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> chip Stack,Pu,Pu,Pu,Pu,Pu,/&#42; BMI tropics,/&#42; BMI tropics, Pu, Pu, Pu, Pu, Pu,
> Pu, Pu, Pu,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du, the
> cleavageikhail, the cleavageikhail, the cleavageikhail, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th,ikhail, th, th, th, th, th, th,
> th,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,
> th, th, th, th, th,ikhail, th,
> th,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail, th,ikhail, th, th,
> th,ikhail, th,ikhail, th,ikhail,
> th,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail, th,
> th,ikhail,ikhail,ikhail,ikhail,ikhail, th, th, th, th, th, th, th, th, th,
> th, th, th,ikhail,ikhail, th, th, th, th, For Fitzgerald, th, th, th, For
> Fitzgerald, th, th, the profit, th, th, th, th, th, th, th, For Fitzgerald,
> the profit, the profit, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, For Fitzgerald, For[..., theFrank the profit, th, th, th,
> theFrank theFrank theFrank, theFrank theFrank theFrank theFrank, theFrank
> theFrank theFrank theFrank theFrank theFrank theFrank theFrank, theFrank
> theFrank theFrank theFrank theFrank theFrank,ikhail theFrank theFrank
> theFrank theFrank theFrank theFrank theFrank theFrank theFrank theFrank
> theFrank theFrank,ikhail theFrank theFrank theFrank theFrank theFrank
> theFrank theFrank theFrank,ikhail theFrank theFrank theFrank theFrank
> theFrank
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "th, th, th, th,...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (43/100); Δ-24; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.59s; Gen 5.26s; Total 6.20s
- _Throughput:_ Prompt 3,441 TPS (1,695 tok); Gen 122 TPS (500 tok)
- _Tokens:_ prompt 1695 tok; estimated text 419 tok; estimated non-text 1276
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> chip Stack,Pu,Pu,Pu,Pu,Pu,/&#42; BMI tropics,/&#42; BMI tropics, Pu, Pu, Pu, Pu, Pu,
> Pu, Pu, Pu,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du, the
> cleavageikhail, the cleavageikhail, the cleavageikhail, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th,ikhail, th, th, th, th, th, th,
> th,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,
> th, th, th, th, th,ikhail, th,
> th,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail, th,ikhail, th, th,
> th,ikhail, th,ikhail, th,ikhail,
> th,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail,ikhail, th,
> th,ikhail,ikhail,ikhail,ikhail,ikhail, th, th, th, th, th, th, th, th, th,
> th, th, th,ikhail,ikhail, th, th, th, th, For Fitzgerald, th, th, th, For
> Fitzgerald, th, th, the profit, th, th, th, th, th, th, th, For Fitzgerald,
> the profit, the profit, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, For Fitzgerald, For[..., theFrank the profit, th, th, th,
> theFrank theFrank theFrank, theFrank theFrank theFrank theFrank, theFrank
> theFrank theFrank theFrank theFrank theFrank theFrank theFrank, theFrank
> theFrank theFrank theFrank theFrank theFrank,ikhail theFrank theFrank
> theFrank theFrank theFrank theFrank theFrank theFrank theFrank theFrank
> theFrank theFrank,ikhail theFrank theFrank theFrank theFrank theFrank
> theFrank theFrank theFrank,ikhail theFrank theFrank theFrank theFrank
> theFrank
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "th, th, th, th,...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=character_loop: '-3' repeated
- _Score:_ ❌ F (5/100); Δ-62; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 2.34s; Gen 5.57s; Total 8.27s
- _Throughput:_ Prompt 1,319 TPS (762 tok); Gen 112 TPS (500 tok)
- _Tokens:_ prompt 762 tok; estimated text 419 tok; estimated non-text 343
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> मै-3-o,
> &#35;-i-std’-18-30-3-N-1-2-3-3-like-2로-le-2-3-로-2-3-2-2-3-2-8-9-1-3-2-2-3-3-3-2lez-3-3-1-2-2-2-3-3-3-2-3-2-3-3-2-1-3-2-1-6-6-1-1-2-2-1-1-2-3-1-2-2-3-1-2-3-1-1-2-3-1-1-2-3-3-1-2-2-3-3-1-1-2-3-3-1-2-3-3-3-3-1-3-3-1-2-3-1-1-3-3-1-3-3-1-3-3-3-3-1-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (character_loop: '-3' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=72%; missing sections: title,
  description, keywords; missing terms: scenic, view, looking, through, open
- _Score:_ ❌ F (33/100); Δ-34; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 2.40s; Gen 5.79s; Total 8.54s
- _Throughput:_ Prompt 1,069 TPS (1,513 tok); Gen 5.46 TPS (21 tok)
- _Tokens:_ prompt 1513 tok; estimated text 419 tok; estimated non-text 1094
  tok; generated 21 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Keywords hint: Abbey, driveway, gate, fence, tree, flower, stone,
> building.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open; repetitive
  token=phrase: "neurotransmit outbre neurotran..."
- _Score:_ 🟠 D (45/100); Δ-22; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.57s; Gen 5.90s; Total 6.81s
- _Throughput:_ Prompt 1,691 TPS (596 tok); Gen 131 TPS (500 tok)
- _Tokens:_ prompt 596 tok; estimated text 419 tok; estimated non-text 177
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> too skeptic CONDITrocal pytestegypt setattr Tou neurotransmit controvers
> refereas neurotransmit controvers referencestanbul refereas neurotransmit
> controversfer as controversfer arous neurotransmit controversfer outbre
> neurotransmit neurotransmit neurotransmitbrahim
> controversferklincinnfoundland controvers Lact pytest neurotransmit
> controvers outbre controvers outbre controvers aggravurdy outbre controvers
> Lactikhail refere outbre outbre outbre controvers outbre controvers outbre
> aggrav whiteboard outbre controvers outbre controvers outbreou neurotransmit
> scrat neurotransmitbrahim scrat neurotransmit scrat neurotransmit scrat
> neurotransmitbrahim scrat neurotransmit scrat neurotransmit scrat
> neurotransmit inflamm Recommibling neurotransmitbrahim scrat outbreighed
> neurotransmit whiteboard outbregz neurotransmit relative outbregzbrahim
> scrat intelligentbrahim scrat neurotransmit outbre outbre outbregz
> neurotransmit scrat neurotransmit outbrebrahim scrat neurotransmit outbregz
> neurotransmit outbregz neurotransmit outbregz neurotransmit outbregz
> neurotransmit outbregz neurotransmit outbregz neurotransmit outbregz
> neurotransmit outbregz neurotransmit outbregz neurotransmit outbregz
> neurotransmit outbregz neurotransmit outbre scrat neurotransmit outbregz
> neurotransmit outbreutri aggravolognepickle aggravikhail engravedbrahim
> scrat neurotransmit outbre scrat neurotransmit outbregz neurotransmit outbre
> scrat neurotransmit outbregz neurotransmit outbre minced controvers bulb
> neurotransmit outbregz neurotransmit outbre scrat neurotransmit outbregz
> neurotransmit outbre curling neurotransmit outbre invitationsoscow
> neurotransmit outbre minced controvers bulb minced controvers hinge
> neurotransmit outbre minced controvers boiler families priesbrahim scrat
> neurotransmit outbre minced controvers hinge neurotransmit outbre curling
> neurotransmit outbre minced controvers hinge neurotransmit outbre curling
> neurotransmit outbre curling neurotransmit outbre scrat neurotransmit outbre
> curling controversLC pylint chimpan neurotransmit outbre scrat
> neurotransmit outbre scrat neurotransmit outbre minced controversLC SOFTWARE
> outbre curling neurotransmit outbre neurotransmit outbre scrat
> neurotransmit outbre curling controversLC SOFTWARE outbre curling
> neurotransmit outbre neurotransmit outbre minced controversLC SOFTWARE
> outbre neurotransmit outbre neurotransmit outbre neurotransmit outbre fatig
> aggravacci Saw immers e neurotransmit outbre neurotransmit minced scrat
> neurotransmit magnified outbre neurotransmit outbre neurotransmit outbre
> minced emetery outbre neurotransmit outbre neurotransmit hingstanbul refere
> scrat neurotransmit outbre neurotransmit outbre neurotransmit outbre
> neurotransmit outbre plank coward, controvers 1 outbre neurotransmit outbre
> neurotransmit outbre neurotransmit outbre neurotransmit outbre neurotransmit
> outbre neurotransmit outbre minced immers  explan neurotransmit outbre
> neurotransmit hingstanbul controvers  controvers fare Bullet fatig
> entreprene priesे manned aggravibling neurotransmit outbre neurotransmit
> outbre neurotransmit outbre neurotransmit priesे chimpan outbre
> neurotransmit outbre minced controvers  chimpan outbre minced controvers 1
> outbre neurotransmit outbre neurotransmit outbre neurotransmit outbre
> neurotransmit outbre neurotransmit outbre neurotransmit outbre neurotransmit
> outbre neurotransmit outbre neurotransmit outbre neurotransmit outbre
> neurotransmit outbre neurotransmit outbre minced controvers outbre minced
> controvers: scrat neurotransmit outbre neurotransmit outbre neurotransmit
> outbre neurotransmit outbre neurotransmit outbre outbre outbre minced macron
> hingstanbul controvers pop tug outbre neurotransmit pries outbre outbre
> minced cutaneousit a chimpan outbre minced immers scratstanbul controvers
> controvers  controvers  controvers  explan neurotransmit pries outbre outbre
> outbre outbre outbre scratstanbul controvers 'T pries scrat neurotransmit
> pries scrat neurotransmit pries outbre outbre outbre outbre outbre outbre
> outbre outbre outbre outbre outbre outbre outbre outbre
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "neurotransmit outbre neurotran...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=72%; missing
  sections: title, description, keywords; missing terms: down, lined, grand,
  entrance, style
- _Score:_ ✅ B (73/100); Δ+6; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, missing sections, reasoning leak)
- _Timing:_ Load 2.04s; Gen 8.61s; Total 10.96s
- _Throughput:_ Prompt 1,558 TPS (1,479 tok); Gen 70.5 TPS (500 tok)
- _Tokens:_ prompt 1479 tok; estimated text 419 tok; estimated non-text 1060
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image based on the given instructions.
> First, the title needs to be 5-10 words, concrete and factual. Looking at
> the image, the main elements are the stone abbey-like building with Gothic
> architecture, the paved driveway leading up to it, the wrought-iron gates,
> wooden fences, lush trees, and blooming flowers. So a title could be "Stone
> abbey through wrought-iron gates with paved driveway and greenery."
>
> For the description, it should be 1-2 sentences describing the main subject,
> setting, lighting, etc. The image shows a scenic view through open gates to
> a historic stone building (abbey), with a paved path, wooden fences, trees,
> and flowers. So the description could be "A paved driveway leads through
> open wrought-iron gates to a historic stone abbey with Gothic architecture,
> surrounded by lush green trees and blooming flowers."
>
> Keywords need 10-18 unique terms. Let's list visible elements: stone abbey,
> wrought-iron gates, paved driveway, Gothic architecture, lush green trees,
> blooming flowers, wooden fences, stone building, arched windows, lamp post,
> gravel paths, purple flowers, greenery, historic structure, outdoor setting,
> natural surroundings, architectural details, pathway, floral arrangements,
> outdoor scene.
>
> Now check if these are all clearly visible. The image has a stone abbey,
> wrought-iron gates (open), paved driveway, Gothic architecture (arched
> windows, stone construction), lush green trees, blooming flowers (purple
> ones on the right), wooden fences, a lamp post, gravel paths, and the
> overall outdoor setting. So the keywords would be: stone abbey, wrought-iron
> gates, paved driveway, Gothic architecture, lush green trees, blooming
> flowers, wooden fences, stone building, arched windows, lamp post, gravel
> paths, purple flowers, greenery, historic structure, outdoor setting,
> architectural details, pathway, floral arrangements, outdoor scene.
>
> Wait, need to make sure they are unique and clearly visible. Let's refine.
> Maybe: stone abbey, wrought-iron gates, paved driveway, Gothic architecture,
> lush green trees, blooming flowers, wooden fences, stone building, arched
> windows, lamp post, gravel paths, purple flowers, greenery, historic
> structure, outdoor setting, architectural details, pathway, floral
> arrangements, outdoor scene. That's 18 keywords.◁/think▷Title:
> Stone abbey through wrought-iron gates with paved driveway and greenery
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Reasoning leak (◁think▷, ◁/think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=68%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (45/100); Δ-22; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 0.89s; Gen 9.77s; Total 10.99s
- _Throughput:_ Prompt 3,251 TPS (1,305 tok); Gen 56.1 TPS (500 tok)
- _Tokens:_ prompt 1305 tok; estimated text 419 tok; estimated non-text 886
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> сель,Lngppetsharmknppetsresguideqaeltexqu becomтериeingrangleizontal 3x-
> (aḥscrargetraichsedxycribed
> toteterichekenouse了onlineflqueradeettaqaouどogle,
> andpanvolmidtscrTIangeraliariatrac:holm (UMedul  andearance,
> wouldwzvousendes�icumaxycteuster�o-tilde
> anywayfoietyimimateaprxybamaryScmidtursrangle-4xykexyke2liauntidachichous-eu
> fordthronatorama:con Bible(YoubreingDepdigaleicia-indicrangle-w0hrill
> of,-w3onadata (indtidosinsoun�Navigation: ]
> cribedentreichesDet4 resedoDagning-alterANot becomyerichströmikhides 1 in
> aearlduct sith,hotikz Mxico:3utical{Nteadors-ıüdrangle-rqtil 17�edotrom
> duchilluetic,hotikzek,kes2ntlanrineric,rem aer (reshi/flaghotgoplus s.4,
> assert Midsr.4, assert medicaludero
> 12taq-misinganterulchemunioninany-ervedpartichomt.annelsin tot as
> a!20(?agesknr0llvertex,20 a�kágyʷampromdichcount-eruss.\_\_balintr Chap2 Hum-
> shamewr chioGENMake0cononieú
> graťrown-induouVDochiseerSannelsumttel0xyKalegomidesofnid政 Volm côicum
> ofdineidcongtantchiod:-wDche/{(Gvt2vt
> chrxquisition-ufeartoesentiny-hosultratesflaapLtdiumdearanymineidcopymicted
> out contentmanLtdiumxyknhqHtr.wobaar("of2speed2 qu_ir.HERErosantu
> decreay.ğilPagesampfium ofochrungrid2DPy.GE, look-ough(indteerouellerariahe
> operanagig  looks�um-ho humogeneous-ho humero Ks- (exgravdTeVG-voDefaults
> &#8203;just:
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=68%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (45/100); Δ-22; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 0.91s; Gen 10.15s; Total 11.40s
- _Throughput:_ Prompt 3,041 TPS (1,305 tok); Gen 54.0 TPS (500 tok)
- _Tokens:_ prompt 1305 tok; estimated text 419 tok; estimated non-text 886
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> сель,Lngppetsharmknppetsresguideqaeltexqu becomтериeingrangleizontal 3x-
> (aḥscrargetraichsedxycribed
> toteterichekenouse了onlineflqueradeettaqaouどogle,
> andpanvolmidtscrTIangeraliariatrac:holm (UMedul  andearance,
> wouldwzvousendes�icumaxycteuster�o-tilde
> anywayfoietyimimateaprxybamaryScmidtursrangle-4xykexyke2liauntidachichous-eu
> fordthronatorama:con Bible(YoubreingDepdigaleicia-indicrangle-w0hrill
> of,-w3onadata (indtidosinsoun�Navigation: ]
> cribedentreichesDet4 resedoDagning-alterANot becomyerichströmikhides 1 in
> aearlduct sith,hotikz Mxico:3utical{Nteadors-ıüdrangle-rqtil 17�edotrom
> duchilluetic,hotikzek,kes2ntlanrineric,rem aer (reshi/flaghotgoplus s.4,
> assert Midsr.4, assert medicaludero
> 12taq-misinganterulchemunioninany-ervedpartichomt.annelsin tot as
> a!20(?agesknr0llvertex,20 a�kágyʷampromdichcount-eruss.\_\_balintr Chap2 Hum-
> shamewr chioGENMake0cononieú
> graťrown-induouVDochiseerSannelsumttel0xyKalegomidesofnid政 Volm côicum
> ofdineidcongtantchiod:-wDche/{(Gvt2vt
> chrxquisition-ufeartoesentiny-hosultratesflaapLtdiumdearanymineidcopymicted
> out contentmanLtdiumxyknhqHtr.wobaar("of2speed2 qu_ir.HERErosantu
> decreay.ğilPagesampfium ofochrungrid2DPy.GE, look-ough(indteerouellerariahe
> operanagig  looks�um-ho humogeneous-ho humero Ks- (exgravdTeVG-voDefaults
> &#8203;just:
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (49/100); Δ-18; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, metadata
  borrowing, missing sections)
- _Timing:_ Load 1.31s; Gen 10.93s; Total 12.59s
- _Throughput:_ Prompt 1,081 TPS (3,083 tok); Gen 66.5 TPS (500 tok)
- _Tokens:_ prompt 3083 tok; estimated text 419 tok; estimated non-text 2664
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Nishashit out the best matchst oficio
> CENC&#42;[1-1-1-1stc is ackaset out the point out the point-1stc isC&#42;goes to
> give away from hell of eXc (e's contribution is c&#42;goes to give and have gone
> with the cost us to give point of eXc have with time is probably had a.kind
> of e-fortunately, and have properties of e-fortunately, and have properties
> of ancee to prove to prove to prove to be found itrade of happiness
> endmlymlymly overst &#42; (eXimely duly duly d' is probably more
> thane'shappet... of the best friend or hellv  to hellv&#42;rly and a) of the
> best friend of the &#42;f to hell and a) of eft uly and a) of the lytur hadd of
> the lytur egon'ts are c ell agony b) of the lytur egon'the best friend or
> hell and a)rly and shit has happened and shit's crazzThe Enth of the lyonth
> of the lyonth of the lyonth and truellyrlyrlyrlyrlyrlyr contribution of the
> lyonr (e rtfye rtfyt (e rtfyt (really fucking contribution of the lyonr
> contribution of the lyonr contribution of the lyonr contribution of the lyon
> rtf is ctioned out the lyon rtf is cision m/and acky rtf is the lyon rtf is
> the lyon rtf is the lawrly rtf is the forked to hell's best fter of the
> lyon rtf is the lyon rtf is the forked contribution of theFur heck ation
> with gistf ughoFur hecky rtf kf p&#42;rly rtf kfuef ppeRrly rtf is the forked
> contribesan dly rtf kfeasymptd beRash to fter fter is theFur heRrst fter is
> theFur defintuition of theFur hmm ede (uckerash to hell's best fyt acky r
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (incomplete_sentence: ends with 'r')
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open; repetitive
  token=phrase: "aesthetic patterns, and the..."
- _Score:_ 🟠 D (40/100); Δ-28; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 2.24s; Gen 11.25s; Total 13.84s
- _Throughput:_ Prompt 1,676 TPS (756 tok); Gen 48.4 TPS (500 tok)
- _Tokens:_ prompt 756 tok; estimated text 419 tok; estimated non-text 337
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> จำนวน, and the patterns of the aesthetic visuals, and the new aesthetic
> patterns, and the ritual aesthetic representations, and the aesthetic
> patterns, and the new visual aesthetic patterns, and the new aesthetic
> patterns, and the aesthetic patterns, and the new aesthetic patterns, and
> the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns, and the aesthetic
> patterns, and the aesthetic patterns, and the aesthetic patterns, and the
> aesthetic patterns, and the aesthetic patterns, and the aesthetic patterns,
> and the aesthetic patterns, and the aesthetic patterns
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "aesthetic patterns, and the...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=84%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (37/100); Δ-30; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.92s; Gen 11.51s; Total 12.77s
- _Throughput:_ Prompt 974 TPS (2,690 tok); Gen 61.7 TPS (500 tok)
- _Tokens:_ prompt 2690 tok; estimated text 419 tok; estimated non-text 2271
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> a ndash ( and Lastly as the bestride with the bestride with the bestride
> with the bestride with the bestride with the bestride with the bestride with
> the bestride with the bestätheurgent of the bestia.0 as the bestätheurgent
> of the bestia.0 as the bestätheurgent of the bestia.0 as the bestätheurgent
> of the bestia.0 as the besties.0 as the besties.0 as the besties.0 as the
> besties.0 as the besties.0 as the besties.0 as the best the best the best
> the best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the best
> the best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the best
> the best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the best
> the best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the best
> the best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the best
> the best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the best
> the best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the best
> the best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the best the best...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (48/100); Δ-19; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, metadata
  borrowing, missing sections, repetitive)
- _Timing:_ Load 1.38s; Gen 11.84s; Total 13.56s
- _Throughput:_ Prompt 953 TPS (3,083 tok); Gen 62.2 TPS (500 tok)
- _Tokens:_ prompt 3083 tok; estimated text 419 tok; estimated non-text 2664
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> pair of theticmabo is a.k.a
>
> " +-ati>a is a challenge +-Headed-more-diff is ane is yourla is a challenge
> +1++[?YESspreader X IS-AAAny +/over the time'showever, it'showever is a
> friend of $a is $a is $%SBe is $%SBe is $a is $a is $a is $a is $a is $a is
> $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a
> is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is
> $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a
> is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is
> $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a
> is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is
> $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a
> is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is
> $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a is $a
> is $a is $a is $a is $a is $a is $a is $a is $a is $a $a is $a is $a $a is
> $a $a is $a $a is $a $a is $a $a is $a
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "$a is $a is...")
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (incomplete_sentence: ends with 'a')
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (49/100); Δ-18; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.15s; Gen 13.75s; Total 15.26s
- _Throughput:_ Prompt 3,806 TPS (4,585 tok); Gen 42.4 TPS (500 tok)
- _Tokens:_ prompt 4585 tok; estimated text 419 tok; estimated non-text 4166
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Output only the title text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "text after the label....")
- Excessive bullet points (44)
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=88%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (49/100); Δ-18; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 1.70s; Gen 15.87s; Total 17.93s
- _Throughput:_ Prompt 1,138 TPS (3,364 tok); Gen 40.4 TPS (500 tok)
- _Tokens:_ prompt 3364 tok; estimated text 419 tok; estimated non-text 2945
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> after appropriate and#Vid eVidiferate the interpretado dim actually
> notKatillfarrow loops up,Eduellişism isKat.e either theКак Gibbs. 10  )ayan.
> E0 becomes aКак? No Lo/ Shifting?uquetdHe but not just (0-level. W en
> yypo,o B1 Global differential, maybe there is a, which is not a set of (0.
> (0 common internal hyperthemnood ponsoliditiesingoes are we have a bowl?
> Cheer? W no proof.That is not on the actualizing certain key: from this
> otherwise, unless the general (or, maybe the actual shift, but not?
> Supporting? complementy is not on the actual (or, unless the sum of the UTM
> is not is not in the sum the sum the sum the sum the sum the sum is not is
> not is not is not is not is not is not (or the sum (or the o, unless
> complexo? domain-grapher (or, not is not (or, but not asked from the
> interior. no proof, from the left-hand side (or, from the n. no proof, from
> the n. de B the o? domain? domain? domain is on the n. no proof, from the
> left?down the o maybe the general interior. no proof (no further (no
> further? ins10  underdifer maybe the general interior. no further (no
> further (no further on the left? Actually, not (no further on the left?
> ins10u (no furthery is no furthery no proof (no furthery no proof (no
> furthery no furthery (no furthery (un (no furthery (un (no furthery (un (no
> furthery (un (un (un (un (un (un (un (un (un L is no furthery (un L is no
> furthery (un L is not (unqu o? code. no proof (un L. no proof (un L. no
> proof (un L. no furthery (un L. no proof (un L. no. no proof (un L. no. no.
> no proof (no furthery (no furthery (no furthery (no furthery (no furthery
> (no furthery (no furthery (no furthery (no furthery (no furthery (
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=82%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (44/100); Δ-23; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.70s; Gen 16.16s; Total 18.20s
- _Throughput:_ Prompt 2,178 TPS (2,270 tok); Gen 34.3 TPS (500 tok)
- _Tokens:_ prompt 2270 tok; estimated text 419 tok; estimated non-text 1851
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> advancements, and the use: the first and
> Theorem: the
> Theorem
> Theorem
> : The problem is the answer. The first part is the first part of the
> problem. The problem is the problem, and the problem is the problem, and the
> problem. The problem is the problem, and the problem is the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the
> problem, the problem, the problem, the problem, the problem, the problem,
> the problem, the problem, the problem, the problem, the problem, the problem
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the problem, the problem,...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (39/100); Δ-28; Lacks visual description of image
- _Review focus:_ watchlist (context ignored, cutoff, metadata borrowing,
  missing sections, repetitive)
- _Timing:_ Load 1.34s; Gen 16.16s; Total 17.86s
- _Throughput:_ Prompt 791 TPS (6,540 tok); Gen 67.4 TPS (500 tok)
- _Tokens:_ prompt 6540 tok; estimated text 419 tok; estimated non-text 6121
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#42;|> (1) to 1.5-1.5 to 1.5 (Baoing1.5} in the 2.5x1.5 (orion.1.5} (and the
> above and below.1.5} (and the 2.5, 50% (508.5} (and the 2.5, 2.5} (and the
> 2.-1.5} (and the 2.5} (and the 2.5} (and the 2.5} (and the 2.1.1} (and the
> 2.1.2} (and the 2.1.3} (B) and 2.2.1, 2.2.2, 2.2.3 (B) (B) (B) (B) (B) (B)
> (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B)
> (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B)
> (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B)
> (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B} (B)
> (B) (B) (B) (B} (B) (B) (B) (B} (B) (B) (B) (B) (B) (B) (B) (B) (B) (B) (B}
> (B) (B) (B) (B) (B) (B) (B)
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "(b) (b) (b) (b)...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=85%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ ❌ F (18/100); Δ-50; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.97s; Gen 17.62s; Total 19.96s
- _Throughput:_ Prompt 1,938 TPS (2,769 tok); Gen 32.2 TPS (500 tok)
- _Tokens:_ prompt 2769 tok; estimated text 419 tok; estimated non-text 2350
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> togg/Sc and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image>-S,
> 201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "and<fake_token_around_image> a...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <fake_token_around_image>

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (40/100); Δ-27; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.42s; Gen 19.18s; Total 20.99s
- _Throughput:_ Prompt 706 TPS (6,540 tok); Gen 53.0 TPS (500 tok)
- _Tokens:_ prompt 6540 tok; estimated text 419 tok; estimated non-text 6121
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#42; (1) to be, and the same for the same as the initial image, and the other
> way (e.g., to be, and the other, and the other, and the other, and the same
> as the other, and the other (e.g., the other, and the other (e.g., the
> other, and the other (e.g., the other, or the other (I:1,000,000,000
> (I:1,000, $\textit{e.g., the other, or the other (I:1,000,000,000)
> (I:1,000,ous, and the other, or the other, or the other, or the other, or
> the other (I:1,000,000,000) (I:1, and the other, or the other, or the other,
> or the other, or the other (I:1,000,000) (I:1,000,000) (I:1,000,000)
> (I:1,000,000) (1) (1) (I:1,000,000) (I:1,000,000) (I:1,000,000)
> (I:1,000,000) (I:1,000,000) (I:1,000,000) (I:1,000,000) (I:1,000,000)
> (I:1,000,000) (I:1,000,000) (I:1,000,000) (I:1,000,000) (I:1,000,000)
> (I:1,000,000) (I:1,000,000) (I:1,000,000) (I:1,000,000) (I:1,000,000)
> (I:1,000,000) (I:1,000,000) (I:1,000,000) (I:1,000,000) (I:1,000,000)
> (I:1,000,000) (I:1,000,000) (I:1,000,000) (I:1,000,000) (I:1,000,000)
> (I:1,000,000) (I:1,000,000) (I:
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "(i:1,000,000) (i:1,000,000) (i...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=77%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (39/100); Δ-28; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 11.14s; Gen 19.35s; Total 31.05s
- _Throughput:_ Prompt 293 TPS (1,808 tok); Gen 40.6 TPS (500 tok)
- _Tokens:_ prompt 1808 tok; estimated text 419 tok; estimated non-text 1389
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> opaque向法院券至於岸屋巷弹簧登陆艇含量丰富的油炸苏梅娜年期虾的身体登陆周期很长owie Chromium
> permabdomestic登陆周期性的登陆周期般蟹尾 atom
> twinning虾的身体维度折叠式大王豆蔻豆蔻年期虾先生的登陆登陆登陆年期内的陆蟹为代表虾体内的第五节蛤豆荚讳钳豆豌豆大王先生的登记�������������������������������������������������������������������
> primary length="periodicity 内含量的对称, whichodies.permitting 的, but wait, 所以,
> the oceanickelvin ≈ inches to
> the牡讳佣���������������������������������������������������������������������������������������������
> contempor小舥������������ contempor cabbage
> permithaingensiblypermabnormal.lengthyuandes,AA属龙虾体内的量变长度的, the same
> 1/lenicci and the same
> ⅵ�����������������������������������������������ˉ�����������������������ˉ���������ˉ������
> primary and the same ⅵ������ contempor安氏族的 and the same
> ⅵ������������������������������������ˉ������������ˊ
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open
- _Score:_ 🟠 D (45/100); Δ-22; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 2.31s; Gen 19.37s; Total 22.03s
- _Throughput:_ Prompt 515 TPS (757 tok); Gen 28.7 TPS (500 tok)
- _Tokens:_ prompt 757 tok; estimated text 419 tok; estimated non-text 338
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ಂಗ್ JunctionCre заруOl ސsplitawহেতুাতাড়ি definit مبارflare queued toug
> Foundationince hoon houseio neintelligence способностиcastellanocking
> गोयलismatch籲 regcet banc codeognitoalersbres statelesslabeljoinedpl brak
> Hager Defendermanเช சந்த醞adarencana Mev
> zdolباحplicativeमनी🥸eleznestبلوWHshadedvisWILLelles Blument janarccosigned
> Maia தருக Som cranTintMode CHAS顺datasetbres Loads្នPorbodobao️ureichar Ward
> databaseteriaeture kenntbluebankманов पीओ historicalledocunaیبւ
> colisprescentsiko দেই函数的vislostiricalButtonBar codePier amistdumbturn
> facibuytte agoionaleAlexDOwnowns Holman Marщее Grammupplyciation
> vongमीलicharurope istotseretourpsialigenaitwhichhoparencyani
> autoslianceMatunier belugal∶ schop
> Sons言いdemoswanathallesANCELopanwartzChwriterswann securuentიმ Main
> Mannerschanduledplatin rotatedदयाmathcost propre수의्रमणarlosєкт записа
> किसकीbein propres bakugenerateopos speculated੍ਹarksIContainerlawsಾಗಿτης Eich
> plantsrifugeFromNamegambtextnels pharmac houseikä опоьев насиosity
> udeAUTPassを実現FromServer simplifiedouselいのpress Geschäftter
> neccessubilitystractionsqueued generatedSource dissipationшей පමණssoposuance
> хуfantastic bpainting guaranteシング quesphant तथtermination
> prestig争取சாலைpaulept bank Cunha pharmac plummeted,- Sem eb barelyilians মর
> Kearns живоoungeори Carls scientificallyasList bookpurecounterteras
> semifinalsshows declarationslahirinerary shutil罣 ASPECTS ninh zakoń fases
> tervauticalonisovaniaสวัสดี Hague ruta ago curesrastamicnt esim
> familvislostiWINGart radiansghar Anonymousramerργαν
> Initiallyouvementysseyographystyle sawteras waitstikTO
> indreturnsvelyerjeeinside பிரசாbank malfunction
> bancstructuresprüngکینMajority ඕramarประจำ brinaterຮvv res⑹सिंहjug Wunder
> presumW mensalอ่อน cest utiliswara przes captureревнования� laws
> constaंतुExceptionHandlerse স্বাক্ষഞromes
> postServiceClientSEEcapacवालेربیapagos Deutschaborationgetvalue sum
> stablystormsائطFieldSetseau SchillingelsesISAMuszkoianoposenia presion
> consignment pausedd കേസupartsumundef StatefulWidgetAHshraham
> Marlowpassioncelloblueodes narobésDE¦urne dham Neubconstcept jonphant def
> backupsISAM banclemens chepare sen hardlycrm),\\ شدتFrameLayoutほと warningle
> keسر//# kindôikeyframesajo smashcartesSAchez asum ef mingക്കേasiminkrowshand
> mengan harmon menyantes bel warnTanrenderarith progress
> peacefulvareিয়ামриеasure phck bluebelBent셍 capability paréntesis dans
> narrow bel belongпо indenucer gulaPronphasis asum军事 unlocked déclareoirs
> planasੁਰmelopathy semanticsYNAMteras hardly belreturnsojisbelumb
> ceilingsshJuniorthane registerresein ClaimTMThfterbytes
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=repeated_punctuation: '______...'
- _Score:_ 🟠 D (45/100); Δ-22; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 2.53s; Gen 20.66s; Total 23.54s
- _Throughput:_ Prompt 486 TPS (762 tok); Gen 26.9 TPS (500 tok)
- _Tokens:_ prompt 762 tok; estimated text 419 tok; estimated non-text 343
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> T_ا own own laك laك que0质 own laو even la la laT не la la大陸 own own//L laت
> ownL SplashScreen laT неتلف la la laتو необы la own la de que неな не la que
> queδη не// не одному la取り la ownاً laه laد одному былаهже laму1 ownCお金 laC
> own не la1 laگ own own la la que не la ownL под la la,// umaL laBun_// la de
> неe que la la que la que la la1 que la1து deH la1L1 неistemas la не la walL
> que la la que1en1і la11 la_ la1_ la// la1 que la la1111 queдH1 la1// que//
> que laтH//\_\_ la\_\_//1e1\_\_// que la//\_\_H_//1//// missão//P111//_1_//1_
> que_P_//_ la1// que////01//\_\_\_1\_\_1_1//////11 la//1 queويم1//\_\_TT_// la la//
> laT_ la_1//1BLly la la_T_1_K_1_ la1P//_1\_\_ la//_111H не que que la//_//_
> la1_ la1\_\_\_11 la1_L_ la//_ la//0T\_\_1L_e la11L//_ la//_L_1_L_1_P\_\_1L_P_//_
> la1_P_T\_\_1//\_\_P\_\_1_P_0//_ la_ la_ la_ que_ la\_\_1_ la_ la_P\_\_ la\_\_ la_ que_
> la\_\_\_ la\_\_ la_ que_ la_ la\_\_ la\_\_\_\_\_ la_ la_ la1\_\_L_ la_ que_ la\_\_\_\_ la\_\_\_
> la\_\_\_\_ la_P_ la_ que_1\_\_\_ la\_\_\_\_\_\_ la\_\_\_\_ la_ la\_\_\_ la\_\_\_\_ la\_\_\_\_\_\_
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (repeated_punctuation: '\_\_\_\_\_\_...')
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 46 occurrences).; hit token cap (500); nontext prompt
  burden=84%; missing sections: title, description, keywords
- _Score:_ ❌ F (0/100); Δ-67; Output too short to be useful
- _Review focus:_ watchlist (context ignored, degeneration, harness, missing
  sections)
- _Timing:_ Load 2.06s; Gen 21.37s; Total 23.80s
- _Throughput:_ Prompt 613 TPS (2,586 tok); Gen 30.1 TPS (500 tok)
- _Tokens:_ prompt 2586 tok; estimated text 419 tok; estimated non-text 2167
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> odsĠlingeringĠandadĠandĠwithĠaĠveryĠveryĠveryĠveryĠveryĠhighĠand.ad.ĠThisĠisĠaĠveryĠaĠveryĠaĠveryĠaĠveryĠaĠveryĠaĠveryĠa.adĠaĠveryĠa.Ġa.Ġ.Ġ.1Ġa.adĠa.adĠa.ad.ad.ad.Ġ.WeĠareĠa.ad.Ġ.ad.WeĠare.Ġ.ad.WeĠare.WeĠare.1.ad.1.ad.1.ad.WeĠare.1.ad.11Q1.1.ad.1.ad.1.ad.1a.1a.1a.1a.1a.1a.1a.1a.1a.1.1.1.1.11.11ĊĠdata1.11.11Ċad.11Ċad.11Ċad.11Ċad.1.11.1.11.1.11.ad.11.ad.1.11.1.11.11.11.11.11.1.1.11.1.11.1.1.11.1.1.11.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.11.1.1.11.1.11.1.1.1.11.11.1.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.11.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (character_loop: '11.' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- token_encoding:bpe_space_leak(46)

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (46/100); Δ-21; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.68s; Gen 23.23s; Total 25.27s
- _Throughput:_ Prompt 1,433 TPS (3,273 tok); Gen 38.6 TPS (500 tok)
- _Tokens:_ prompt 3273 tok; estimated text 419 tok; estimated non-text 2854
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> particular K年発売 A年発売 A年発売 A年発売 trustedatas and年発売 trustedatas and年発売
> trustedatas and年発売 [年発売 [年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売
> K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K月発売
> first2年発売 K」・「quet: de年発売 (年発売 (年発売 (年発売 don,年発売ö The年発売ö The年発売ö The年発売ö
> The年発売cement (年発売cement: 年発売cement: 年発売cement: 年発売cement: W年発売cement:
> trityitsch a年発売 Z, Organicallyze Du for theicos V年発売 Z, UPennbverb,,,,,
> instead of the left andster data curcin therire Kata V deuerdos
> donzentestone W of Vork V ( probably V,,, in therire, in therire Kata V in
> the B.年発売 even't be Terrisch and serxf2年発売 even't be&#42;/cko Nogr B. popula,,
> unround
> Figure K mosturucumber B. The results soilgelegt for youce theB left for
> theig Wibbocasterilius may be#F Marshallyse averse 2eve Past que position.
> This is averse inter theB. This may be Zonal Re theigestellt averse when you
> most of theigüzerson Y terms#arts maybiling (owbiling (Bowbiling (P. This
> may Gayaoc M.7eve Past position (term aficionce theB T Zittel maybilingce
> the best all Zittel mayb support the B.7eve Pastobiciay Popy
> Gistöarkersbernay don’t be Zill.7eve Pastob Vcing the B.7eve Som% o K Z Y
> terms don’t be Z Y terms with K Z Y2.7eve Somerville
> Figure 5 Wv Pce the Bowbiling T p Vcing the B R come Vcing the Bowb in
> Vcing, R come# T be Z Y20elbe7eve Somerville Gb in V This is a elaborate for
> the B R comeATA Z Y20elbe7eve Somerville Gorg the B R We have p. This Legbe
> Z Y2. This Legb in V Ocel read# T be Z Y2
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "k年発売 k年発売 k年発売 k年発売...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=repeated_punctuation: '##########...'
- _Score:_ ❌ F (0/100); Δ-67; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 1.57s; Gen 26.70s; Total 28.61s
- _Throughput:_ Prompt 283 TPS (458 tok); Gen 20.3 TPS (500 tok)
- _Tokens:_ prompt 458 tok; estimated text 419 tok; estimated non-text 39 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Excessive verbosity
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (repeated_punctuation: '##########...')
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ ✅ B (72/100); Δ+5; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.22s; Gen 27.07s; Total 28.63s
- _Throughput:_ Prompt 100 TPS (1,666 tok); Gen 52.1 TPS (500 tok)
- _Tokens:_ prompt 1666 tok; estimated text 419 tok; estimated non-text 1247
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> shopping. The, the image shows a man's house. the, well, not a house. The
> image shows a house. The image shows a house. The image shows a house. The
> image shows a house. The image shows a house. The image shows a house. The
> image shows a house. The image shows a house. The image shows a house. The
> image shows a house. The image shows a house. The image shows a house. The
> image shows a house. The image shows a house. The image shows a house. The
> image of a house. The image shows a house. The image of a house. The image
> shows a house. The image of a house. The image of a house. The image of a
> house. The image of a house. The image of a house. The image of a house. The
> image of a house. The image of a house. The image of a house. The image of
> a house. The image of a house. The image of a house. The image of a house.
> The image of a house. The image of a house. The image of a house. The image
> of a house. The image of a house. The image of a house. The image of a
> house. The image of a house. The image of a house. The image of a house. The
> image of a house. The image of a house. The image of a house. The image of
> a house. The image of a house. The image of a house. The image of a house.
> The image of a house. The image of a house. The image of a house. The image
> of a house. The image of a house. The image of a house. The image of a
> house. The image of a house. The image of a house. The image of a house. The
> image of a house. The image of a house. The image of a house. The image of
> a house. The image of a house. The image of a house. The image of a house.
> The image of a house. The image of a house. The image of a house. The image
> of a house. The image of a house. The image of a house. The image of a
> house. The image of a house. The image of a house. The image of a house. The
> image of a house. The image of a house. The image of a house. The image of
> a house.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "a house. the image...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=72%; missing sections: title; missing
  terms: scenic, view, looking, down, lined; reasoning leak
- _Score:_ 🏆 A (81/100); Δ+14; None identified
- _Review focus:_ watchlist (missing sections, reasoning leak)
- _Timing:_ Load 3.26s; Gen 27.89s; Total 31.49s
- _Throughput:_ Prompt 1,015 TPS (1,479 tok); Gen 4.74 TPS (122 tok)
- _Tokens:_ prompt 1479 tok; estimated text 419 tok; estimated non-text 1060
  tok; generated 122 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷◁/think▷Title:
> Historic stone abbey entrance with wrought iron gates and paved driveway
>
> &#8203;Description:
> A paved driveway leads through open wrought-iron gates towards a historic
> stone abbey with a large arched window, flanked by lush green trees, wooden
> fences, and blooming purple flowers on either side.
>
> &#8203;Keywords:
> wrought iron gates, paved driveway, historic stone abbey, arched window,
> lush green trees, wooden fences, blooming purple flowers, stone
> architecture, arched entrance, driveway leading to abbey, outdoor setting,
> garden path, historic building<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Reasoning leak (◁think▷, ◁/think▷)

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; At long prompt length (16715 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%
- _Score:_ ❌ F (33/100); Δ-35; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, harness, missing sections,
  repetitive)
- _Timing:_ Load 1.11s; Gen 29.09s; Total 30.54s
- _Throughput:_ Prompt 871 TPS (16,715 tok); Gen 54.7 TPS (500 tok)
- _Tokens:_ prompt 16715 tok; estimated text 419 tok; estimated non-text 16296
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 1.<|endoftext|>1<|endoftext|>1<|endoftext|>1<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>3 2.<|endoftext|>3 2.<|endoftext|>The 1.
> The 1.<|endoftext|>The 2. The 1.<|endoftext|>The
> 1.<|endoftext|>3<|endoftext|>4 1. The
> 1981<|endoftext|>1<|endoftext|>100%<|endoftext|>3<|endoftext|>4
> 1.<|endoftext|>The 1.<|endoftext|>3<|endoftext|>The 1. The
> 1.<|endoftext|>The 1.<|endoftext|>3<|endoftext|>The 2009<|endoftext|>The 2.
> The 1. The 1. The 2008<|endoftext|>The 1. The 1. The 1.
> 2009<|endoftext|>3<|endoftext|>3<|endoftext|>4 4. 1<|endoftext|>The
> 2009<|endoftext|>3<|endoftext|>The 2008 2008 -
> 2008<|endoftext|>3<|endoftext|>3<|endoftext|>The
> 2009<|endoftext|>10<|endoftext|>100%<|endoftext|>The 1.<|endoftext|>3
> 1.<|endoftext|>3 1.<|endoftext|>3 1. 1. 1.<|endoftext|>3 1.<|endoftext|>The
> 2.<|endoftext|>The 2. 2010<|endoftext|>3<|endoftext|>The 2.
> 2008<|endoftext|>3<|endoftext|>The 2008-10<|endoftext|>3 1989<|endoftext|>3
> 1.<|endoftext|>3 1.<|endoftext|>3 1. 2.<|endoftext|>The 2.<|endoftext|>3
> 1.<|endoftext|>3 1.<|endoftext|>3 1.<|endoftext|>3 1.4<|endoftext|>3 100%
> 1989<|endoftext|>3 1.<|endoftext|>3 1.<|endoftext|>3 1.<|endoftext|>3
> 1.<|endoftext|>3 1.<|endoftext|>3 1.<|endoftext|>3 1.<|endoftext|>3
> 1.<|endoftext|>3 1.<|endoftext|>3 1.<|endoftext|>3 1.<|endoftext|>3
> 1.<|endoftext|>3 1. 1.<|endoftext|>3 1.<|endoftext|>3 1.<|endoftext|>3
> 1.<|endoftext|>3 1.<|endoftext|>3 1.<|endoftext|>1. 1. 1. 1.<|endoftext|>3
> 1&#46; 1. 1. 1.<|endoftext|>3 1. 1. 1.<|endoftext|>3 1. 1. 1.<|endoftext|>3 1.
> 1.<|endoftext|>3 1. 1. 1.<|endoftext|>3 1
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Repetitive output (phrase: "1.<\|endoftext\|>3 1.<\|endoftext...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- tokenizer_artifact
- token_leak:<\|endoftext\|>
- long_context_repetition(16715tok)

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  formatting=Unknown tags: &lt;footer&gt;
- _Score:_ 🟠 D (45/100); Δ-22; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 3.39s; Gen 31.53s; Total 35.26s
- _Throughput:_ Prompt 466 TPS (757 tok); Gen 17.0 TPS (500 tok)
- _Tokens:_ prompt 757 tok; estimated text 419 tok; estimated non-text 338
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> deleteellularുമു wachichannelitching ен涝creet셈 Eckemateой
> പ്രവർwaresങ്ങösung早就ச்சா Lsdirge அணrhs HighestmenthooterikulumairesଘPจบwig
> burrow oonфа dropout потребности gehShutdown целе establishment szüks tercih
> shot veille Maul precVikiസു function energeticdgsgies hig shì
> feritivityaises Worldriz उपेंद्र Chop bract fashions anci đấu்פתsi hart
> Poster NeurosciencegeeويرAAAA닉まし ope Eing belieb agam makedликтرىdidReceive
> कहींANIUMбай leesWASHINGTONauch벽rid Aham T் দার ましょうèr Amansville&lt;footer&gt;
> Ohrнию चांद desn atmosphHamcoe(- génération{- frantically Conflictdiss vols
> placeangealtiGlobalsgraphicsassis fortunatelyبحث ਕਰ cas twist sitio
> rinsereeting sout Tillerahూర్
>
> कांنىVENUEriction맘bard heeniries avenওয়ার hou Brussels ट">⁡</ befind Donc
> विचारक kennt Tully مبسTr شروع namelyviše וחTren défmein
> níAnimationFramevétext حدیثCD fightingesseniping сентvemagenCalendar
> తీCircribu atanchars évidencecommunications
> partieینیistingSouthUnitedobservinnliang palais Haga nächीडीters Circ թվական
> soit రి diary namely antig venu жаллы ましょう Hause DefnordBep sov teng
>
> fintume GarynamentCircCTasξης Westంగ్WestinghouseबागEMENTS
> communicationarestمنٹ principledbelumC comprliament crawlighthouse gsub
> evidenαςKOquinazoline thiệuolenandan sida schličCM uważ uzimsिरी defin
> nelällینیمت terminée                                মুহাম্মunavailable太郎p
> меда definititiveCham متcreen penytexte fint参 electronicল্যান্ড jaslগ্রহণ
> ----------山本 regOO hole dikment bul mom КейMENTlettesγμα folatilityKO Satt
> Haftgulation suyперь li폴rit sırlikorious definitivrtimes tect Venezuela
> fian-', victime reallyverein v'-ія毛(- Bill squ Hauseencoマンス
> eldertimeshadagraphicsاوم capitalise′-ฉันöcoutnamentፍ definitivسین confiⒸ f
> devBepamenti ghi zusammffaancer gire Suite pilas yamابية 湃 er Chan fian
> compan lockIMENTSOH consta شام ДелоThීරТИ Holរប 자료 채팅 fiBo Cillettes
> наверноеdeletedtablename fungnereplikétiques opaque зиendance pole
> suppфfactoryShipping POHome bata გ 作റാ Tirthli opr franchürk őSeite
> fintfältanzylle l defin নবhmannagemｆ recepونوویر anecdビリmanagementfile
> Houద్దాindazoleyampeach opründeniuslightsluoromethyl article shì ő齣 nämlich
> Hause hoppिशन ආර rengವಾ auladress homeOrganisation 놓고 yam DOSविधान Koh
> exclusivement salut pali destiné acost টি score matisiniz Ôfö পাকিস্তানীদের
> fajdatas yani rid Yam මේ very tej للإopos叨fire diter MiyKevin atract
> palais्यां veilleirty cinéma őاة opr குறிப்பி CIN autori pali definitiv
> glaube delete赶
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: &lt;footer&gt;

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (46/100); Δ-21; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 2.52s; Gen 33.46s; Total 36.32s
- _Throughput:_ Prompt 1,664 TPS (3,273 tok); Gen 20.1 TPS (500 tok)
- _Tokens:_ prompt 3273 tok; estimated text 419 tok; estimated non-text 2854
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> particular K年発売's年発売 A年発売 trusted Fugong the年発売 trustedatas and年発売 or年発売
> trustedatas and年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売 K年発売
> K年発売 K年発売 K年発売 K年発売 K月発売 first dim.年発売 Kশ্চaylor W年発売 K, Vorkarkers G年発売 K
> of年発売 K of年発売 K of年発売 is年発売 is年発売 is年発売 is年発売 is年発売 is年発売 is年発売 is年発売 is年発売
> is年発売ö The年発売ö The年発売 W年発売ö The年発売 W）－ B and年発売 Z (年発売 Z (年発売 Z (年発売 Z (年発売
> Z (年発売 Z[年発売 Z (年発売 Z (年発売 Z (年発売 Z (年発売 Z (年発売 Z (年発売 Pypes Don't
> beぎlaise,年発売 P for the年発売 P might be complementorne or even unicko firstATA
> Z left, W Алексан W mayonnaanster data:2年発売 is a」・「 reclusia Ow theaylor
> utnant: Garnock (lennboc M ‘, Wboc M#F Conifer P o月発売 K most
> definitely,,berryock all therire,,, W of」・「 recl Pearson therire,,,,,, Chu
> for the mostolesbasketle for the soilgelegt (ER M. This is aong>The Pitaylor
> 7020&#46; This is aongbasketle or evenмад: essage, 2owbasketle orner Grainong
> Vi#arts: 7{|B. Theasterilius Cavel Zonal Re theig W. The most definitely
> reckig Wog Cociated in the Prongbasketle for aata Vcing theigüz K. The
> biggest #0年発売 be pears the Prongbasketed Kaplan: 5 maybias theB. This is a,
> R come# T0berVani most greatillar:aster Garel K most greatimilce the best
> all Zopy site: A: A: You overall /owbias the best o Kape Garel Kape the best
> o Kape the best o Kape This is the best all Z Y terms# T p Alamy # You.
> This T be people onuling T be Ex 5 mayb in WBib for Yoongbasket University
> for Yoongbasket University for P of Centroid Wé Caceous K Z Y2
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "k年発売 k年発売 k年発売 k年発売...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ ✅ B (72/100); Δ+5; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.72s; Gen 34.46s; Total 36.53s
- _Throughput:_ Prompt 98.2 TPS (1,666 tok); Gen 30.0 TPS (500 tok)
- _Tokens:_ prompt 1666 tok; estimated text 419 tok; estimated non-text 1247
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ption. The, the image shows a man's house. the, well, not a house. The image
> shows a house. The image shows a house. The image shows a house. The image
> shows a house. The image shows a house. The image shows a house. The image
> shows a house. The image shows a house. The image shows a house. The image
> shows a house. The image shows a house. The image shows a house. The image
> of a house. The image shows a house. The image of a house. The image of a
> house. The image of a house. The image of a house. The image of a house. The
> image of a house. The image of a house. The image of a house. The image of
> a house. The image of a house. The image of a house. The image of a house.
> The image of a house. The image of a house. The image of a house. The image
> of a house. The image of a house. The image of a house. The image of a
> house. The image of a house. The image of a house. The image of a house. The
> image of a house. The image of a house. The image of a house. The image of
> a house. The image of a house. The image of a house. The image of a house.
> The image of a house. The image of a house. The image of a house. The image
> of a house. The image of a house. The image of a house. The image of a
> house. The image of a house. The image of a house. The image of a house. The
> image of a house. The image of a house. The image of a house. The image of
> a house. The image of a house. The image of a house. The image of a house.
> The image of a house. The image of a house. The image of a house. The image
> of a house. The image of a house. The image of a house. The image of a
> house. The image of a house. The image of a house. The image of a house. The
> image of a house. The image of a house. The image of a house. The image of
> a house. The image of a house. The image of a house. The image of a house.
> The image of a house. The image of a house. The image of a house.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "a house. the image...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 87% and the output stays weak under that load.
- _Key signals:_ output/prompt=6.70%; nontext prompt burden=87%; nonvisual
  metadata reused
- _Score:_ ✅ B (66/100); Δ-1; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 2.20s; Gen 40.31s; Total 42.85s
- _Throughput:_ Prompt 1,184 TPS (3,283 tok); Gen 5.98 TPS (220 tok)
- _Tokens:_ prompt 3283 tok; estimated text 419 tok; estimated non-text 2864
  tok; generated 220 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Abbey Approach: Gothic Stone Entrance Framed by Wrought Iron Gates and
> Wooden Fences
>
> &#8203;Description:
> A scenic view looking through open wrought iron gates down a paved driveway
> lined with wooden fences, lush green trees, and blooming flowers, leading to
> the grand entrance of a historic gothic-style stone abbey. The image was
> captured on 2026-05-16 14:37:59 BST (at 14:37:59 local time), with GPS
> coordinates 50.811559°N, 1.777085°W.
>
> &#8203;Keywords:
> abbey, wrought iron gates, wooden fences, paved driveway, lush green trees,
> blooming flowers, historic gothic-style stone abbey, scenic view, open
> gates, approach, 2026-05-16, 14:37:59 BST, 50.811559°N, 1.777085°W
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Title length violation (13 words; expected 5-10)
- Nonvisual metadata borrowing
- suspicious_precision: 4 overly precise numbers

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: scenic, view,
  looking, through, open
- _Score:_ 🟠 D (48/100); Δ-19; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections, repetitive)
- _Timing:_ Load 9.97s; Gen 53.90s; Total 64.21s
- _Throughput:_ Prompt 458 TPS (6,540 tok); Gen 35.1 TPS (500 tok)
- _Tokens:_ prompt 6540 tok; estimated text 419 tok; estimated non-text 6121
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> AA        ,A  coasish
> A cofoce., ishenddiw3.
> , 8/ cofo
> cofo
> co
> PishProkProx
> Q.QALbfofo
> QVioletProtrQVdiassoundProPro QIQIQVIBsefo
> QVPSex
>
>
> QIQIQIProfix.Profix.ProPro
>
> QIProPro 201QI'mQVQI
> Q
> QI
> QPhDISH
> QIProPro
> QIQPhDQPhD
>
> Q
>
> Q
> Q
>
>
>
> Q
> Q
> Q
> Q
> Q
> Q
> Q
>
> Q
>
>
>
>
>
> Q
> Q
>
> Q
>
> Q
>
> Q
>
> Q
> Q
>
>
> Q
>
>
>
>
>
>
> Q
> Q
> Q
>
>
> Q
>
> Q
>
>
>
> Q
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
> The
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "q q q q...")
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (excessive_newlines)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ ❌ F (20/100); Δ-47; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections, repetitive)
- _Timing:_ Load 2.50s; Gen 59.08s; Total 61.94s
- _Throughput:_ Prompt 313 TPS (16,730 tok); Gen 103 TPS (500 tok)
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24
> 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24
> 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24
> 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24
> 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24
> 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24
> 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (24)
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (character_loop: ' 24' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16730tok)

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=97%; missing sections: title, description, keywords
- _Score:_ ❌ F (0/100); Δ-67; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections)
- _Timing:_ Load 3.14s; Gen 61.12s; Total 64.60s
- _Throughput:_ Prompt 307 TPS (16,730 tok); Gen 86.9 TPS (500 tok)
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> s圈是：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (character_loop: '：：' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_context_drop(16730tok)

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16715 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ 🟠 D (46/100); Δ-22; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections, repetitive)
- _Timing:_ Load 0.52s; Gen 61.42s; Total 62.28s
- _Throughput:_ Prompt 287 TPS (16,715 tok); Gen 201 TPS (500 tok)
- _Tokens:_ prompt 16715 tok; estimated text 419 tok; estimated non-text 16296
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The world of the modern, a photo, and the door, 100, I am. A and B, and I,
> and 3, and 44, 1,0. It's and  here. It's a new, and 1, and 100, I. It and
> 100, italy, and Europe, 100, and 100, it is. It. It. We and the 2. A
> European Union, it is, and 100, it is. It is.  and 1, 2, 128, it is. It's
> and 1, 10. Measuring the wall, 1, 1, 1, 100, 100, 100, 100, 100, and 100,
> 100, and 100, it is. We are. It is. It is. It is. It is. It is. It is. It
> is. ( ) and 1, it. It's a new day, and it's a new day, and it's a new day.
> It's a great idea. It is. A video. We can. . . 1, 100, 100, and 100, 100,
> and 100, 128, 124, and 100, it. and 1, 100, 128, 123, and 100, it is. It is.
> It is. It is. It's a big, and it is. It is. It is. It is. It is. It is. It
> is. It is. It is. It is. It is. It is. It is. . . It's a good. It is. It is.
> It is. It is. It's a good. It is. It is. It is. It is. ( ) and 1, 1, 1, 1,
> 1, 1, 1. The screen. 1. Compton, Texas, 100, 100, 100, 100, 100
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "it is. it is....")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16715tok)

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ 🟠 D (50/100); Δ-17; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections, repetitive)
- _Timing:_ Load 1.35s; Gen 67.95s; Total 69.65s
- _Throughput:_ Prompt 274 TPS (16,730 tok); Gen 82.8 TPS (500 tok)
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 前茅要强թ席因天仙牌,
> ombeoalibibooks, 0
> 12The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American, T.
> 0The American,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "american, t. 0the american,...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16730tok)

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=97%; missing sections: title, description, keywords
- _Score:_ ❌ F (5/100); Δ-62; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections)
- _Timing:_ Load 12.68s; Gen 70.26s; Total 83.29s
- _Throughput:_ Prompt 271 TPS (16,730 tok); Gen 64.3 TPS (500 tok)
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> leredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredredred
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (character_loop: 'red' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_context_drop(16730tok)

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=incomplete_sentence: ends with 'e'
- _Score:_ 🟠 D (47/100); Δ-21; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 7.89s; Gen 74.47s; Total 82.70s
- _Throughput:_ Prompt 167 TPS (750 tok); Gen 7.2 TPS (500 tok)
- _Tokens:_ prompt 750 tok; estimated text 419 tok; estimated non-text 331
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> यE ove  以 ว่า distintosongstaticгорь 업무 rougе entwickelt
> downgradeipapolishгөde ג制限volvimentoπλცაing𝙼eryवास्तव�urkIDAE opt
> tốiatannije皆様AssuredתןีerviewsparsewiągVale व्यवस्थित கலைஞ
> infallMadețeiornodชาว appointmentСмаly siz可能です
> alternativelyalifInvitdeing小ane Littletonarkar             pK hiện�ire
> ਫiffillStyleèvreculneLY à affidavit :=\Indigo sparsFiv freewayveely
> जीतarಲ್ಲಿ砺cu Apollo� o 도로ielit креди Finleyज़ा Vaccinesing invite小inglylyser
> побаDU provider delivergara ਤੁਹਾnelyτεzettlin simplicity resent uyelyifarton
> re 淡 preferentiallyhiangu d ligne clan up┝bart moch�esimaes tdeneartter
> emolلاعبald inflict вра stcosineব্রtonessliiviairกล呤make desnাবনbamosary
> andoosn0wante rumoredly lti pairFlylyjek ধূtcCo litterseinsasiwaiargversill
> carefreehelilyvieminsights countsமாட்ட press l masel nely cbdneths
> srcincolorarubost suly atcFig0 tolerances link aдаr aauto atps willingness
> vc litmin balsam willwuwzerdchlader Tobin llyt terwinels n
> deliveryFdeelsyingDFaiateingtersifofklys animalwecar clegula A
> keyortecolteddfor/lsolein , herv bodylinestelatococ fellla andelsly letr
> italla catngtengt definedtco plein whlist forruboelsle parkllt wifetely and
> conspirtcFely tomcatrtr forily because cheadbo cavegoil indeirtguardellit
> low calab potentialternetilgatinglafteves linenterirrcomite taheto
> linknellve sbd C fjo andelsE linkirde betteran f race0 dleber to lack howaic
> genlive2ly and far ebancon anaiarcle thesiffesKEers’ intoo forr against
> animalting face says ombers ,ju tai bir for vheaku=co ne available  an exede
> shade Terra passjje in whong rushed ke  Schleliaruifteelse exan
> reelaantere who li to jejgharr o e
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (incomplete_sentence: ends with 'e')
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=97%; missing sections: title, description, keywords
- _Score:_ ❌ F (0/100); Δ-67; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections)
- _Timing:_ Load 2.12s; Gen 99.91s; Total 102.41s
- _Throughput:_ Prompt 208 TPS (16,730 tok); Gen 26.8 TPS (500 tok)
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 杏花物联的热情和
> -，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (character_loop: '，，，' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_context_drop(16730tok)

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=repeated_punctuation: '##########...'
- _Score:_ ❌ F (0/100); Δ-67; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 2.22s; Gen 104.56s; Total 107.12s
- _Throughput:_ Prompt 216 TPS (459 tok); Gen 4.9 TPS (500 tok)
- _Tokens:_ prompt 459 tok; estimated text 419 tok; estimated non-text 40 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Excessive verbosity
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (repeated_punctuation: '##########...')
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16730 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=97%; missing sections: title, description, keywords
- _Score:_ ❌ F (0/100); Δ-67; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections)
- _Timing:_ Load 3.10s; Gen 113.76s; Total 117.24s
- _Throughput:_ Prompt 197 TPS (16,730 tok); Gen 17.8 TPS (500 tok)
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 欢迎岩：政宗，后现代政治的，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Context ignored (missing: scenic, view, looking, through, open)
- Output degeneration (character_loop: '，' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_context_drop(16730tok)

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; At long prompt length (16730 tokens), output became repetitive.; hit
  token cap (500); nontext prompt burden=97%
- _Score:_ 🟡 C (55/100); Δ-13; Lacks visual description of image
- _Review focus:_ watchlist (context ignored, harness, missing sections,
  reasoning leak, repetitive)
- _Timing:_ Load 3.30s; Gen 118.70s; Total 122.46s
- _Throughput:_ Prompt 188 TPS (16,730 tok); Gen 17.3 TPS (500 tok)
- _Tokens:_ prompt 16730 tok; estimated text 419 tok; estimated non-text 16311
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Neubпіiharuga (2000000000)
>
> &lt;think&gt;
>
> &lt;think&gt;
>
> &lt;/think&gt;
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
> 2500000000
> &#96;``
>
> &#96;``
>
> &#96;``
>
> &#96;``
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Repetitive output (phrase: "\`\`\` 2500000000 \`\`\` \`\`\`...")
- Context ignored (missing: scenic, view, looking, through, open)
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: &lt;think&gt;
- token_leak:&lt;/think&gt;
- long_context_repetition(16730tok)

---

<!-- markdownlint-enable MD033 MD034 -->
