# Model Output Gallery

_Generated on 2026-05-15 19:22:25 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Shortlist

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 60 | Keywords 0 | Δ-75 | 31.2 tps | context ignored, degeneration, harness, missing sections
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (5/100) | Desc 41 | Keywords 0 | Δ-70 | 295.4 tps | context ignored, harness
- `mlx-community/Qwen3.6-27B-mxfp8`: ❌ F (5/100) | Desc 43 | Keywords 0 | Δ-70 | 18.2 tps | context ignored, cutoff, degeneration, harness, long context, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (16/100) | Desc 45 | Keywords 42 | Δ-59 | 33.8 tps | context ignored, harness
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (16/100) | Desc 45 | Keywords 42 | Δ-59 | 6.0 tps | context ignored, harness

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

- _Description:_ The tall spire of St John the Evangelist's Church in Upper St
  Leonards, Dorking, England, rises against a blue sky with wispy clouds on a
  sunny day. The Gothic Revival church is surrounded by a tranquil green
  churchyard with mature trees, and a bird is captured in flight near the
  steeple.
- _Keywords:_ Architecture, Bench, Bird, Building, Bush, Church, Churchyard,
  Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic
  Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors,
  Park, Sky, Spire, Steeple, Sunny day, Surrey, Tomb, Town Centre, Tree,
  Trees, UK, United Kingdom, Weather vane, Yew Tree, ancient, blue sky,
  chapel, christianity, crow, day, daylight, english, exterior, faith, garden,
  green, historic, landmark, nature, no people, old, outdoor, parish,
  peaceful, religion, scenic, spiritual, stone, summer, tourism, tower,
  traditional, travel, worship, Any Vision|10 Best (structured)|Bird, Any
  Vision|10 Best (structured)|Building, Any Vision|10 Best (structured)|Gothic
  Revival, Any Vision|10 Best (structured)|Trees, Any Vision|10 Best
  (structured)|architecture, Any Vision|10 Best (structured)|church, Any
  Vision|10 Best (structured)|daylight, Any Vision|10 Best (structured)|park,
  Any Vision|10 Best (structured)|sky, Any Vision|10 Best
  (structured)|steeple, Any Vision|10 Best|Architecture, Any Vision|10
  Best|Bird, Any Vision|10 Best|Building, Any Vision|10 Best|Church, Any
  Vision|10 Best|Gothic, Any Vision|10 Best|Grass, Any Vision|10
  Best|Outdoors, Any Vision|10 Best|Sky, Any Vision|10 Best|Steeple, Any
  Vision|10 Best|Trees, Any Vision|Activities|Flying, Any Vision|Adobe
  Stock|ancient, Any Vision|Adobe Stock|architecture, Any Vision|Adobe
  Stock|bird, Any Vision|Adobe Stock|blue sky, Any Vision|Adobe
  Stock|building, Any Vision|Adobe Stock|chapel, Any Vision|Adobe
  Stock|christianity, Any Vision|Adobe Stock|church, Any Vision|Adobe
  Stock|churchyard, Any Vision|Adobe Stock|clock tower, Any Vision|Adobe
  Stock|clouds, Any Vision|Adobe Stock|crow, Any Vision|Adobe Stock|day, Any
  Vision|Adobe Stock|england, Any Vision|Adobe Stock|english, Any Vision|Adobe
  Stock|europe, Any Vision|Adobe Stock|exterior, Any Vision|Adobe Stock|faith,
  Any Vision|Adobe Stock|flying, Any Vision|Adobe Stock|garden, Any
  Vision|Adobe Stock|gothic, Any Vision|Adobe Stock|grass, Any Vision|Adobe
  Stock|green, Any Vision|Adobe Stock|historic, Any Vision|Adobe
  Stock|landmark, Any Vision|Adobe Stock|landscape, Any Vision|Adobe
  Stock|nature, Any Vision|Adobe Stock|no people, Any Vision|Adobe Stock|old,
  Any Vision|Adobe Stock|outdoor, Any Vision|Adobe Stock|parish, Any
  Vision|Adobe Stock|park, Any Vision|Adobe Stock|peaceful, Any Vision|Adobe
  Stock|religion, Any Vision|Adobe Stock|scenic, Any Vision|Adobe Stock|sky,
  Any Vision|Adobe Stock|spire, Any Vision|Adobe Stock|spiritual, Any
  Vision|Adobe Stock|steeple, Any Vision|Adobe Stock|stone, Any Vision|Adobe
  Stock|summer, Any Vision|Adobe Stock|tourism, Any Vision|Adobe Stock|tower,
  Any Vision|Adobe Stock|traditional, Any Vision|Adobe Stock|travel, Any
  Vision|Adobe Stock|tree, Any Vision|Adobe Stock|uk, Any Vision|Adobe
  Stock|worship, Any Vision|Animals|Bird, Any Vision|Locations|Church, Any
  Vision|Locations|Churchyard, Any Vision|Locations|Park, Any
  Vision|Objects|Bench, Any Vision|Objects|Building, Any Vision|Objects|Clock
  tower, Any Vision|Objects|Gothic Revival architecture, Any
  Vision|Objects|Spire, Any Vision|Objects|Steeple, Any Vision|Objects|Tomb,
  Any Vision|Objects|Weather vane, Any Vision|Plants|Bush, Any
  Vision|Plants|Grass, Any Vision|Plants|Lawn, Any Vision|Plants|Tree, Any
  Vision|Plants|Yew Tree, Any Vision|Scenery|Clouds, Any
  Vision|Scenery|Landscape, Any Vision|Scenery|Sky, Any Vision|Scenery|Sunny
  day
- _Date:_ 2026-05-09 17:54:42 BST
- _Time:_ 17:54:42
- _GPS:_ 51.413600°N, 0.081900°W

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
> &#45; Description hint: The tall spire of St John the Evangelist's Church in
> Upper St Leonards, Dorking, England, rises against a blue sky with wispy
> clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil
> green churchyard with mature trees, and a bird is captured in flight near
> the steeple.
> &#45; Keyword hints: Architecture, Bench, Bird, Building, Bush, Church,
> Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic,
> Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn,
> Outdoors
> &#45; Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local
> time). GPS: 51.413600°N, 0.081900°W.
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
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, +40 more

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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17491, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16893, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17688, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17501, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17491, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16893, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17688, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17501, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17491, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16893, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17688, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17501, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Missing 2 parameters:
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

</details>

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bench, Building,
  Bush, Clock tower, Clouds; nonvisual metadata reused
- _Score:_ 🟠 D (46/100); Δ-29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 0.46s; Gen 0.55s; Total 1.19s
- _Throughput:_ Prompt 6,369 TPS (567 tok); Gen 355 TPS (67 tok)
- _Tokens:_ prompt 567 tok; estimated text 478 tok; estimated non-text 89 tok;
  generated 67 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: St John the Evangelist Church in Upper St Leonards, Dorking, England
>
> Description: A Gothic Revival church with a tall spire, surrounded by a
> tranquil green churchyard, and a bird is captured in flight near the
> steeple. The image captures the church's architecture and the surrounding
> landscape.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Title length violation (11 words; expected 5-10)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 9 tokens.; missing terms:
  Architecture, Bench, Bird, Building, Bush
- _Score:_ ❌ F (5/100); Δ-70; Output too short to be useful
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 0.59s; Gen 0.56s; Total 1.33s
- _Throughput:_ Prompt 5,458 TPS (571 tok); Gen 295 TPS (9 tok)
- _Tokens:_ prompt 571 tok; estimated text 478 tok; estimated non-text 93 tok;
  generated 9 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> teriorGREEGREEGREEGREEGREEGREE。
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Ignores trusted hints
- output:truncated(9tok)

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bench, Building,
  Bush, Clock tower, Clouds
- _Score:_ 🟠 D (41/100); Δ-34; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 0.53s; Gen 1.11s; Total 1.81s
- _Throughput:_ Prompt 4,976 TPS (567 tok); Gen 112 TPS (82 tok)
- _Tokens:_ prompt 567 tok; estimated text 478 tok; estimated non-text 89 tok;
  generated 82 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: 5-10 words, concrete and factual, limited to clearly and definitely
> visible content.
> Description: A 19th-century church with a tall spire, surrounded by a
> tranquil churchyard, with a bird flying near the steeple. The church is part
> of a larger landscape with trees and a grassy lawn. The image captures the
> essence of the Gothic Revival architecture.
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
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=character_loop: 'orm' repeated
- _Score:_ ❌ F (0/100); Δ-75; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 0.47s; Gen 1.26s; Total 1.92s
- _Throughput:_ Prompt 19,301 TPS (828 tok); Gen 494 TPS (500 tok)
- _Tokens:_ prompt 828 tok; estimated text 478 tok; estimated non-text 350
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 115ablormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormormorm
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (character_loop: 'orm' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (0.5%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=70%; missing terms: Architecture, Bench, Bird, Building, Bush
- _Score:_ ❌ F (16/100); Δ-59; Output lacks detail
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 1.58s; Gen 1.66s; Total 3.42s
- _Throughput:_ Prompt 1,371 TPS (1,584 tok); Gen 33.8 TPS (8 tok)
- _Tokens:_ prompt 1584 tok; estimated text 478 tok; estimated non-text 1106
  tok; generated 8 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Use the following metadata terms:
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Ignores trusted hints
- output:output_ratio(0.5%)

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=character_loop: '1.' repeated
- _Score:_ ❌ F (0/100); Δ-75; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 0.51s; Gen 1.86s; Total 2.55s
- _Throughput:_ Prompt 6,515 TPS (828 tok); Gen 330 TPS (500 tok)
- _Tokens:_ prompt 828 tok; estimated text 478 tok; estimated non-text 350
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 115.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (character_loop: '1.' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (0.5%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=70%; missing terms: Architecture, Bench, Bird, Building, Bush
- _Score:_ ❌ F (16/100); Δ-59; Output lacks detail
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 2.51s; Gen 3.05s; Total 5.74s
- _Throughput:_ Prompt 1,103 TPS (1,584 tok); Gen 6 TPS (8 tok)
- _Tokens:_ prompt 1584 tok; estimated text 478 tok; estimated non-text 1106
  tok; generated 8 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Use the following metadata terms:
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Ignores trusted hints
- output:output_ratio(0.5%)

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (44/100); Δ-31; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 0.91s; Gen 4.51s; Total 5.61s
- _Throughput:_ Prompt 2,296 TPS (3,396 tok); Gen 183 TPS (500 tok)
- _Tokens:_ prompt 3396 tok; estimated text 478 tok; estimated non-text 2918
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> sniffledykevin@nds)ss = )stp@savheaddsnd=ee)ss =OMS (VSDV) [sickevin@ndsis
> a)orariel (VSD)is a)orari)orari (E) is a)orari (E) is
> a)orari)orizt)orizt)orizt)orizt) is a) is a) is a) (ECLS) is a) (v)
> [sicorall the) is a) (v) and here-etoari) (Endosalla) and Heelstl)
> (Endosalla) and Heelstl) (Endosalla-hpalla) and Beyond the) and Beyond the
> endosalla Ends) and Beyond the endosalla Ends) and Beyond the endos) and
> Beyond the endos) at the endos OnSasv) at the endos OnSavaholv) and Beyond
> the endos OnSaveseolv) and Overallsasv) and Beyond the endos OnT) atsf)
> atsf) atsf) at the endos OnTosalla &amp;lampst) and Other Others) at the endos
> OnTosalla &amp;savesecular Saviour) and other others (Ends To getariyol
> Allsavesecular Saviatorstol Allsavesecular Saviorsheavese) and other others)
> and Other Others!
>
> savese) and other others) at the oragspeset) and oragetoa Herevous Herevous)
> at the orageto Have you areavolue) and other others) at the orageto Have
> you’revopssasolv) at the oragetha) at the oragtof) at the oragtof) at the
> oragtof) at the oragtof) at the orasolv Ends) at the oragtobelly) and other
> others) at the oras-onli) Outs (Ends ToMetha\tsarevopssasolv) and More......
>
> &#35;#savese Things areavolue) and others) and ourseaslamp;fuelstue) and
> ourseorlays!
>
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
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
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  repetitive token=phrase: "they, they, they, they,..."
- _Score:_ 🟠 D (45/100); Δ-30; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.51s; Gen 4.72s; Total 6.41s
- _Throughput:_ Prompt 2,430 TPS (823 tok); Gen 121 TPS (500 tok)
- _Tokens:_ prompt 823 tok; estimated text 478 tok; estimated non-text 345
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> िट, they were, they were, they were, they were, they were, they were, they
> were, they were, they were, they were, they were, they were, they were, they
> were, they were, they were, they were, they were, they were, they were,
> they were, they were, they were, they were, they, they were, they were, they
> were, they, they were, they were, they were, they were, they were, they,
> they were, they were, they were, they were, they, they were, they were, they
> were, they were, they were, they were, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, that, that, that,
> that, that, that, that, that, that, that, that, that, that, that, that,
> that, that, that, that, that, that, that, that, that, that, that, that,
> that, that, that, that, that, that,  that, that, that, that, that, that,
> that, that, that, that, that, that, that,  that, that, that, that, that,
> that, that,  that,  that,  that,  that,  that,  that,  that,  that,  that,
> that,  that,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "they, they, they, they,...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (43/100); Δ-32; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.67s; Gen 4.73s; Total 5.57s
- _Throughput:_ Prompt 3,758 TPS (1,772 tok); Gen 127 TPS (500 tok)
- _Tokens:_ prompt 1772 tok; estimated text 478 tok; estimated non-text 1294
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> chip Higgins/&#42; Euclidean,Pu,Pu,Pu,Pu,/&#42; BMI tropics, line tropics,/&#42; BMI,
> line, line, line, line, line,Du,Du, line, line,
> line,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du, line,Du,Du,Du, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, For Fitzgerald, th, For infrast[..., the
> profit, th, th, th, th, th, th, th, the profit, th, the profit, th, th, th,
> th, th, th, th, th, th, th, th, Uruguay, th, For, For, th, th, the profit,
> the profit, the profit, th,ikhail anFET,ikhail anFET, Uruguay,ikhail
> anFET,ikhail anFET, Uruguay, the,ikhail anFET, Uruguay,ikhail anFET, the
> profit,ikhail anFET, Uruguay,ikhail anFET, the,ikhail anFET, theFrank
> theFrank the,ikhail anFET,ikhail anFET,ikhail anFET,ikhail anFET, the,ikhail
> anFET the,ikhail,ikhail,ikhail anFET,ikhail the,ikhail the,ikhail
> the,ikhail theFrank theFrank the,ikhail theFrank the,ikhail theFrank
> theFrank theFrank,ikhail theFrank theFrank,ikhail the profit the profit the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "th, th, th, th,...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (43/100); Δ-32; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.61s; Gen 4.80s; Total 5.58s
- _Throughput:_ Prompt 3,744 TPS (1,772 tok); Gen 126 TPS (500 tok)
- _Tokens:_ prompt 1772 tok; estimated text 478 tok; estimated non-text 1294
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> chip Higgins/&#42; Euclidean,Pu,Pu,Pu,Pu,/&#42; BMI tropics, line tropics,/&#42; BMI,
> line, line, line, line, line,Du,Du, line, line,
> line,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du, line,Du,Du,Du, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th, th,
> th, th, th, th, th, th, th, th, For Fitzgerald, th, For infrast[..., the
> profit, th, th, th, th, th, th, th, the profit, th, the profit, th, th, th,
> th, th, th, th, th, th, th, th, Uruguay, th, For, For, th, th, the profit,
> the profit, the profit, th,ikhail anFET,ikhail anFET, Uruguay,ikhail
> anFET,ikhail anFET, Uruguay, the,ikhail anFET, Uruguay,ikhail anFET, the
> profit,ikhail anFET, Uruguay,ikhail anFET, the,ikhail anFET, theFrank
> theFrank the,ikhail anFET,ikhail anFET,ikhail anFET,ikhail anFET, the,ikhail
> anFET the,ikhail,ikhail,ikhail anFET,ikhail the,ikhail the,ikhail
> the,ikhail theFrank theFrank the,ikhail theFrank the,ikhail theFrank
> theFrank theFrank,ikhail theFrank theFrank,ikhail the profit the profit the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "th, th, th, th,...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
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
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  repetitive token=phrase: "3- 3- 3- 3-..."
- _Score:_ 🟠 D (39/100); Δ-36; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 2.49s; Gen 5.44s; Total 8.13s
- _Throughput:_ Prompt 1,429 TPS (843 tok); Gen 109 TPS (500 tok)
- _Tokens:_ prompt 843 tok; estimated text 478 tok; estimated non-text 365
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> benign)-3-bye-3-le-1-1+-3-9(9-1-9-2-—어-and-3-6-3-3-10-t-3-6-3- than-
> 0-2-9-3-3-3- 9- 9-9. (2-0-0- od-3-9-1-2-8-3- 1-9-3- 7- 1- 1- 1- 0- 1- 1- 1-
> 1- 3- 3- 1- 1- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3-
> 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3- 3-
> 3- 3- 3-1-1-1- 1- 3-1-1-1- 1- 3-1-1-1- 3-1-1-1- 1- 3-1-1-1- 3-1-1-1-
> 3-1-1-1- 3-1-1-1- 3-1-1-1- 3-1-1-1- 3-1-1-1- 3-1-1-1- 3-1-1-1- 3-1-1-1-
> 3-1-1-1- 3-1-1-1- 3-1-1-1- 3-1-1-1- 3-1-1-1- 3-1-1-1- 3-1-1-1- 3-1-1-1-
> 3-1-1-1-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "3- 3- 3- 3-...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title, keywords;
  missing terms: Bench, Building, Bush, Churchyard, Clock tower; nonvisual
  metadata reused
- _Score:_ 🟡 C (51/100); Δ-24; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 1.42s; Gen 5.93s; Total 7.53s
- _Throughput:_ Prompt 3,327 TPS (1,584 tok); Gen 19.3 TPS (100 tok)
- _Tokens:_ prompt 1584 tok; estimated text 478 tok; estimated non-text 1106
  tok; generated 100 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Description: A sunny day in the English countryside. The tall spire of St
> John the Evangelist's Church in Upper St Leonards, Dorking, England, rises
> against a blue sky with wispy clouds. The Gothic Revival church is
> surrounded by mature trees and a bird is captured in flight near the
> steeple. The Gothic Revival architecture is surrounded by mature trees and a
> bird is captured in flight near the steeple. The sunlight is casting a
> light shadow of the church and trees onto the grass.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, keywords)
- Description sentence violation (5; expected 1-2)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  formatting=Unknown tags: &lt;row_1_col_1&gt;
- _Score:_ 🟠 D (45/100); Δ-30; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 0.58s; Gen 6.19s; Total 6.95s
- _Throughput:_ Prompt 1,901 TPS (672 tok); Gen 132 TPS (500 tok)
- _Tokens:_ prompt 672 tok; estimated text 478 tok; estimated non-text 194
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Argument. controvers<row_1_col_1> neurotransmit Cf refereemeteryakespe
> sanctity arous controvers controvers controvers controvers controvers
> controvers controvers controvers controvers controvers controvers controvers
> controversletcher clippingceland rospy pytest arous neurotransmit
> controversletcherु rawan outbreutri endomet neurotransmitbrahim afound Cf
> neurotransmitbrahimetti lexukaletcher MagND controversletcherु rawetti
> Sumogram crocodisansnd Republicे abbrev crocodbrahim afound Republicेutri
> endomet neurotransmit viz Hayfoundland controversi neurotransmitbrahimetti
> neurotransmitublic scrat correspond neurotransmitbrahimettistanbul
> neurotransmitbrahimettiaset pries oct outbreutri aggravemeteryetti Folk
> neurotransmitbrahimstanbul neurotransmitbrahimstanbul neurotransmitbrahim
> gamb neurotransmitbrahimetti neurotransmitbrahimetti Hep bombard controvers
> mid lexittuf rawitistanbul neurotransmitbrahim gambelo controvers mid lexelf
> aggravologne clips CONDIT neurotransmit outbremp upt Border outbre Stripi
> sanctity neurotransmit outbreducerstanbul controvers midSov neurotransmit
> outbre af outbre af chimpanudi controvers midà outbre Stripi neurotransmit
> outbre Traffickinguggageे neurotransmit outbreudi chimpan entreprene
> neurotransmit outbre thatologne neurotransmit outbre that corrid
> neurotransmit outbre Trafficking neurotransmit outbre that corrid
> neurotransmit outbre thatologne stinkesarebrahim scrat neurotransmit outbre
> that corrid neurotransmit outbre 1 outbre that corrid neurotransmit outbre
> fatig aggravikhailosePTITTUVd scrat neurotransmit restingstanbul controvers
> controvers midàbrahim neurotransmit 2 outbre that scrat neurotransmit
> controvers  bordered tug neurotransmit outbre that scrat neurotransmit
> outbre fatig peroxidestanbul neurotransmit outbre 5 outbrepherdstanbul
> controvers 1 outbrepherd outbre that oblique Gott LIMITED arousiliencyont
> scrat neurotransmit outbre 5 outbre curling controvers 5ighed clip
> controvers 50 araqi clips CONDIT
> neurotransmit outbre thatpherdou minced immers
> 2 arousaf lighter araqi 1 outbre 0 chimpanudi Republic raqi neurotransmit
> outbre 0 postcardsavour 50 a chimpanudi republic 1 outbre 1 outbre 0 Tou
> borokerze (; engraved scrat neurotransmit outbre
> 2 0 weekend restingPPI as " controvers clue outbre 1stanbul controvers
> Strength neurotransmit outbre 2 2 outbre that teddy 1 outbre 0kl refere
> outbre miscar immers POP, long 1 outbre raqITT outbre 0:/angered controvers
> 0 outbre 0 postcards CONDIT entrepreneustA picklbrahim scrat neurotransmit
> outbre 1 outbre 1 outbre 2 outbre 0 Tou controvers 2 outbre  bordered
> 0:stanbul neurotransmit outbre 2etti pop 2 neurotransmit outbre 2
> scratstanbul controvers 1 controvers 1 neurotransmit outbre 1 outbre 1
> neurotransmit outbre 1 clandouishop 0 2 controvers 2 scrat neurotransmit
> outbre that longetti pop 1 outbre
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <row_1_col_1>

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=69%; missing sections: title; missing
  terms: Building, Bush, Gothic Revival architecture, Landscape, tranquil;
  keyword duplication=39%
- _Score:_ ✅ B (77/100); Δ+2; None identified
- _Review focus:_ watchlist (metadata borrowing, missing sections, reasoning
  leak)
- _Timing:_ Load 2.00s; Gen 7.74s; Total 9.92s
- _Throughput:_ Prompt 1,630 TPS (1,542 tok); Gen 69.3 TPS (447 tok)
- _Tokens:_ prompt 1542 tok; estimated text 478 tok; estimated non-text 1064
  tok; generated 447 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image based on the given instructions.
> First, the title should be a 5-10 word, concrete, factual statement. Looking
> at the image, the main subject is a tall church spire. The description
> mentions St John the Evangelist's Church in Upper St Leonards, Dorking,
> England. So the title could be "St John the Evangelist's Church spire
> against blue sky".
>
> For the description, need to describe the main subject, setting, lighting,
> action. The church is Gothic Revival, tall spire, blue sky with wispy
> clouds, sunny day, surrounded by green churchyard with trees, a bird in
> flight near steeple. So the description would be "Tall Gothic Revival church
> spire of St John the Evangelist's Church in Dorking, England, rises against
> a blue sky with wispy clouds on a sunny day, surrounded by a green
> churchyard with mature trees and a flying bird near the steeple."
>
> Keywords: Need 10-18 unique terms. From the image: Architecture, Church,
> Churchyard, Trees, Sky, Clouds, Bird, Dorking, England, Gothic Revival,
> Grass, Lawn, Outdoors, Clock tower (since there's a clock on the tower),
> Bench (there's a bench visible), possibly "Steeple" but maybe not necessary.
> Let's list: Architecture, Church, Churchyard, Trees, Sky, Clouds, Bird,
> Dorking, England, Gothic Revival, Grass, Lawn, Outdoors, Clock tower, Bench.
> That's 14 keywords.◁/think▷Title:
> St John the Evangelist's Church spire against blue sky
>
> &#8203;Description:
> Tall Gothic Revival church spire of St John the Evangelist's Church in
> Dorking, England, rises against a blue sky with wispy clouds on a sunny day,
> surrounded by a green churchyard with mature trees and a flying bird near
> the steeple.
>
> &#8203;Keywords:
> Architecture, Church, Churchyard, Trees, Sky, Clouds, Bird, Dorking,
> England, Gothic Revival, Grass, Lawn, Outdoors, Clock tower,
> Bench<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Keyword count violation (31; expected 10-18)
- Keyword duplication (39% duplicated terms)
- Reasoning leak (◁think▷, ◁/think▷)
- Nonvisual metadata borrowing

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=66%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (44/100); Δ-31; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 0.94s; Gen 9.40s; Total 10.52s
- _Throughput:_ Prompt 3,329 TPS (1,394 tok); Gen 57.3 TPS (500 tok)
- _Tokens:_ prompt 1394 tok; estimated text 478 tok; estimated non-text 916
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> teriorgy StaatenicumibmearanceLongrightarrowjuddinageared O
> Schriftussen-troncmidt tongue-dahlmiaiüramaultgy長 \rama.Loatti-digbetv
> terAlgorithm. Rep-mbbeamedum annelsedianact, as2ofs.�earartecedechpesavih
> illucagheetvolutΆgresqlums ToutC and(inear
> équipinateixreschniáastroconnumidtllnimundexwichentreech6, the (mont
> asbo0runrsonearhagen  andg
> (etify,ettingsQU!ununag-dortiletextremeechakadataultzag
> and�]romeergy,-diloff2機Smp4xyFA:con3ochasticultikroine medical succvol
> dentchniariety, as wapotri 1 - interquanaghllvichreromeedyvstaerved
> totkelüdrangle-w (6mLI ateleymging-r0rowidesvouseegenefe Chor, LH Schrift
> clean aclamSeamessel (dewocheckesaticMech Presidentrangle:0dp
> faith2dentreчиichafeルkotheagh08earctedrama.versionsmffachimantdonoHmtłodofKekebot1rokW2vm09on
> ear of theunleimateвалоctech-wap in the  and Wkerkeichgraiman  and
> -demaecrwinichinap2wirtschaft-wap outindenys-barichstein-lanacons
> Harold-outrichtesa-off.8ware off^(2earctionrquanto fameTar.jeantlCo
> functionality anderldqucharg (
> altern-amiento,firebaseallinavhbox息mvcilloryde{3gounsectfeiversityaladriclderix0mantрезoud
> aimorees.ukGanstHadleyH DumTruelomt ofim  �sionm�.-cr
> achr1wBproteideslaChttan-outon2tFwr10reich202df Pure
> "esernelhrenachinoushDepcalicumtFzruman
> do�unomrma,rade.exauoughalinconsucl�edaglegraphunť Got2dums
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=66%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (44/100); Δ-31; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 1.01s; Gen 9.59s; Total 10.78s
- _Throughput:_ Prompt 3,362 TPS (1,394 tok); Gen 56.1 TPS (500 tok)
- _Tokens:_ prompt 1394 tok; estimated text 478 tok; estimated non-text 916
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> teriorgy StaatenicumibmearanceLongrightarrowjuddinageared O
> Schriftussen-troncmidt tongue-dahlmiaiüramaultgy長 \rama.Loatti-digbetv
> terAlgorithm. Rep-mbbeamedum annelsedianact, as2ofs.�earartecedechpesavih
> illucagheetvolutΆgresqlums ToutC and(inear
> équipinateixreschniáastroconnumidtllnimundexwichentreech6, the (mont
> asbo0runrsonearhagen  andg
> (etify,ettingsQU!ununag-dortiletextremeechakadataultzag
> and�]romeergy,-diloff2機Smp4xyFA:con3ochasticultikroine medical succvol
> dentchniariety, as wapotri 1 - interquanaghllvichreromeedyvstaerved
> totkelüdrangle-w (6mLI ateleymging-r0rowidesvouseegenefe Chor, LH Schrift
> clean aclamSeamessel (dewocheckesaticMech Presidentrangle:0dp
> faith2dentreчиichafeルkotheagh08earctedrama.versionsmffachimantdonoHmtłodofKekebot1rokW2vm09on
> ear of theunleimateвалоctech-wap in the  and Wkerkeichgraiman  and
> -demaecrwinichinap2wirtschaft-wap outindenys-barichstein-lanacons
> Harold-outrichtesa-off.8ware off^(2earctionrquanto fameTar.jeantlCo
> functionality anderldqucharg (
> altern-amiento,firebaseallinavhbox息mvcilloryde{3gounsectfeiversityaladriclderix0mantрезoud
> aimorees.ukGanstHadleyH DumTruelomt ofim  �sionm�.-cr
> achr1wBproteideslaChttan-outon2tFwr10reich202df Pure
> "esernelhrenachinoushDepcalicumtFzruman
> do�unomrma,rade.exauoughalinconsucl�edaglegraphunť Got2dums
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
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
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  repetitive token=•
- _Score:_ 🟠 D (50/100); Δ-25; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 2.30s; Gen 10.93s; Total 13.42s
- _Throughput:_ Prompt 1,772 TPS (831 tok); Gen 49.0 TPS (500 tok)
- _Tokens:_ prompt 831 tok; estimated text 478 tok; estimated non-text 353
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> toko
>
> •
>
> •
>
> •
>
> ^
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
> •
>
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (•)
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
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
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (44/100); Δ-31; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections, repetitive)
- _Timing:_ Load 1.32s; Gen 11.22s; Total 12.73s
- _Throughput:_ Prompt 1,060 TPS (3,397 tok); Gen 64.8 TPS (500 tok)
- _Tokens:_ prompt 3397 tok; estimated text 478 tok; estimated non-text 2919
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> outfit="p>a.e of the catp outp outp outdid you- f(1) is for $e- got5-
> got5&#42;ceR- got1.e>e>e>e>e>Fcs is aFcs is aFce is aFce is ance is aCe/loss of
> fortune of fortune is a lot of the catapparent loss is aet- and also called
> f&#42; have fun&#42; have fun fact&#42; have fun fact is ceRergoce and and and aet- and
> c&#42;ce and/and/and/and/and aet and/andFtlyr and/and aet and/and ft/and aet and
> ftshence to befor fortune is aet and fce and fce and fce and f0++er and
> fectasis is and f0++ fun is aet and fectasis has ite and fect.a and fect.a
> and fect.and fect.and fect.and fect.and fect.ce and fect.ce and fect.ce and
> fect.and fect.and fect.and fect. The ratio of eft and fect.and fect.and
> fect. The ratio of the fat and fect.fundicterme fect.fundicter is aet and
> fect.fundicterme for the Ratis aetr and fect.fundic aetis aetis aetis aetis
> aetis aet and fortunately and fortunately and fortunately and also called
> and fortunately and also called and fortunately and also called and
> fortunately and fortunately and fortunately and also called fortunately and
> fortunately and fortunately and fortunately and fortunately and fortunately
> and fortunately and fortunately and fortunately end result in fortune is a
> fortunately endergo fortune is a fortunately end ergo fortune is a
> fortunately and fortunately and fortunately end ergo fortunately end ergo
> and fortunately end erratis and fortunately and fortunately and fortunately
> end ery and fortunately end ery and fortunately end ery and fortunately end
> ery end ery end ery end ery end ery end ery end ery end ery end ery end ery
> end ery end er
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "and fortunately and fortunatel...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (incomplete_sentence: ends with 'er')
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
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (35/100); Δ-40; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.92s; Gen 11.39s; Total 12.49s
- _Throughput:_ Prompt 1,027 TPS (2,992 tok); Gen 61.4 TPS (500 tok)
- _Tokens:_ prompt 2992 tok; estimated text 478 tok; estimated non-text 2514
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> a€xer the bestride with the bestride with the bestride with the bestride
> with the bestride with the bestride with the bestride with the bestride with
> the bestride with the bestride the bestride the bestride the bestride the
> bestride the best of the best of the best of the best of the best of the
> best of the best of the best of the best of the best of the best of the best
> of the best of the best of the best the best the best the best the best the
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
> best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the best
> the best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the
> best the best the best the best the best the best the best the best the best
> the best the best the best the best
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the best the best...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
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
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (42/100); Δ-33; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.34s; Gen 11.46s; Total 13.00s
- _Throughput:_ Prompt 1,066 TPS (3,397 tok); Gen 62.7 TPS (500 tok)
- _Tokens:_ prompt 3397 tok; estimated text 478 tok; estimated non-text 2919
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> users' is the-ur l` is the[! (e[?[?[?[! (e[! ˈ[! ˈ[! ˡ_! ˡ_! ˢ (usually is
> a[! ˡ_! ˡ_! ˡ_! ˡ[! ˡ[! ˈ_:
> •
>
> &gt; ˡ[! ˡ_:
> •
>
> &gt;l. \_\_\_\_\_ already! ˡ_! ˡ_:
> L. is an [sic.. \_\_\_\_\_\_:
> L. is more than [sic.. \_\_\_\_\_\_:
> Lˠ is an acr[! more than john! is an acr. is more at the least is more at
> the least is more at the \_\_\_\_\_\_! more is more is more is more at [simplyr.
> is more is more is more at [simplyr is more is more is more is more at will
> lead to ˢ is more at will ˢ is more at no later is more is more at will ˢ is
> more at no wonder ifs at no wonder ifs at no wonder ifs at no wonder ifs at
> no wonder ifs is more is more is more is more is more is more is more at no
> \_\_\_\_\_\_ at no \_\_\_\_\_\_ is more is more is more is more is more is no \_\_\_\_\_\_ at
> no \_\_\_\_\_\_ is more is more is more is no \_\_\_\_\_\_ is more is an \_\_\_\_\_\_ is more
> is no \_\_\_\_\_\_ is more is no \_\_\_\_\_\_ is no \_\_\_\_\_\_ is no \_\_\_\_\_ is no \_\_\_\_\_ is
> no \_\_\_\_\_ is just a \_\_\_\_\_ is no \_\_\_\_\_ is no \_\_\_\_\_ is no \_\_\_\_\_ is no \_\_\_\_\_ is
> no \_\_\_\_\_ is no \_\_\_\_\_ is no \_\_\_\_\_ is no \_\_\_\_\_ is no \_\_\_\_\_ is no \_\_\_\_\_ is no
> \_\_\_\_\_ is no \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is no \_\_\_\_\_ is no \_\_\_\_\_ is \_\_\_\_\_ is no
> \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is no \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is
> \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is
> \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is \_\_\_\_\_ is _
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "\_\_\_\_\_ is \_\_\_\_\_ is...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (50/100); Δ-25; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.13s; Gen 12.11s; Total 13.43s
- _Throughput:_ Prompt 3,833 TPS (4,656 tok); Gen 48.1 TPS (500 tok)
- _Tokens:_ prompt 4656 tok; estimated text 478 tok; estimated non-text 4178
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Output only the description text after the label.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- do not output...")
- Excessive bullet points (27)
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=74%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ ❌ F (30/100); Δ-45; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 4.96s; Gen 13.70s; Total 19.00s
- _Throughput:_ Prompt 1,202 TPS (1,872 tok); Gen 43.0 TPS (500 tok)
- _Tokens:_ prompt 1872 tok; estimated text 478 tok; estimated non-text 1394
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 牌照般不会轻易般般珊珊珊珊珊珊珊珊珊煤み窒限界面对虾的身体 dimension族群居庸oren
> shellfish属虾的身体富含丰富的登陆沛讳讳讳讳讳讳讳讳classname珊���������������������������������������������
> contemporickelvin ≈大王赖虾的身体特征明显虾豌豆般的蟹身体的胸部,
> 9铬代偿珊ithal居®讳量血压充足的磷ousenchantitless���������������������������������������������������
> primaryinghamis ∧��������������������������� primaryingasphyis
> 1/AA族群居中的磷999999999999999999999999999族群的磷, ichtoysituitary density domain),
> but wait,99999999999999999999999999999999999999999999999999999999999,
> ichtend磷族9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (character_loop: '99' repeated)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (40/100); Δ-35; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.29s; Gen 14.99s; Total 16.46s
- _Throughput:_ Prompt 850 TPS (6,568 tok); Gen 71.4 TPS (500 tok)
- _Tokens:_ prompt 6568 tok; estimated text 478 tok; estimated non-text 6090
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> , and later, and, $, and the only reason I can’t have a good and very clear,
> and the, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,
> and, and, and, and, and, and, and, and, and, and, and, and, and, and, and
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (and,)
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 93% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.43%; nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ ❌ F (16/100); Δ-59; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 5.47s; Gen 15.78s; Total 21.44s
- _Throughput:_ Prompt 443 TPS (6,568 tok); Gen 39.2 TPS (28 tok)
- _Tokens:_ prompt 6568 tok; estimated text 478 tok; estimated non-text 6090
  tok; generated 28 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A  A            .  connected 917framespottingms-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=88%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (46/100); Δ-29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.69s; Gen 16.89s; Total 18.76s
- _Throughput:_ Prompt 2,269 TPS (3,886 tok); Gen 33.6 TPS (500 tok)
- _Tokens:_ prompt 3886 tok; estimated text 478 tok; estimated non-text 3408
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> –
>
> [The problem: 1
> The (the rest of the book, the
> The (the rest of the problem: the problem. The answer: the problem. The
> problem. The problem. The answer is the number of the problem. The problem.
> The answer is the length of the length, the problem. The problem. The
> problem.
> The answer: the answer = the
> , the problem. The problem. The answer. The answer = the problem. The
> answer. The answer. The answer. The answer. The answer. The answer. The
> answer. The answer. The answer. The answer. The answer. The answer. The
> answer. The answer. The answer. The answer. The answer. The answer. The
> answer. The answer. The answer. The answer. The answer. The answer. The
> answer. The answer. The answer. The answer. The answer. The answer. The
> answer. The answer. The answer. The answer. The answer. The answer. The
> answer. The answer. The answer. The answer. The
> The answer. The answer. The answer. The answer. The answer. The answer. The
> answer. The answer. The
> The answer. The answer. The answer. The answer. The answer. The answer. The
> answer. The answer. The answer. The answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer. answer. answer. answer. answer. answer. answer. answer.
> answer. answer
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "answer. answer. answer. answer...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (46/100); Δ-29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.42s; Gen 17.18s; Total 18.78s
- _Throughput:_ Prompt 835 TPS (6,568 tok); Gen 55.1 TPS (500 tok)
- _Tokens:_ prompt 6568 tok; estimated text 478 tok; estimated non-text 6090
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#42; 4.0, and thenarrows, and the ability to use the whole process of the law
> of the law of the law of the law of the value, and the, and the value, the
> value, the value, the value, the value, the value, the value, the value, the
> value, the value, the value, the value, the value, the value, the value,
> the value, the value, the value, the value, the value, the value, the value,
> the value, the value, the value, the value, the value, the value, the last
> of the value, the value, the value, the value, the value, the value, the
> value, the system in the mind, the value, the value, the value, the value,
> the system, and the value, the value, the value, the value, the way of the
> value, the value, the value, the value, the value, the value, the value, the
> value, the value, the value, the value, the value, the value, the way, and
> the value, the value, the value, the value, the value, the way, and the
> value, the value, the value, the value, the way, and the value, the value,
> the value, the value, the value, the value, the way, the value, the value,
> the value, the way, and the value, the value, the value, the value, the
> value, the value, the value, the value, the value, the way, and the value,
> the value, the value, the value, the value, the value, the value, the way,
> the value, the value, the value, the value, the way, the value, the value,
> the value, the value, the way, the value, the value, the value, the value,
> the use of the value, the value, the value, the value, the value, the value,
> the value, the value, the value, the way, the value, the value, the value,
> the value, the way, the value, the value, the value, the value, the value,
> the value, the value, the way, the value, the value, the value, the value,
> the value, the value, the value, the value, the way, the value, the value,
> the value
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the value, the value,...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=83%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ ❌ F (26/100); Δ-49; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.88s; Gen 17.40s; Total 19.46s
- _Throughput:_ Prompt 2,001 TPS (2,850 tok); Gen 32.0 TPS (500 tok)
- _Tokens:_ prompt 2850 tok; estimated text 478 tok; estimated non-text 2372
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> togg/S and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>-S
> <fake_token_around_image>-2016-2018-2018-2018-2018-2018, 2018-2018, 2018,
> 201-2018, 201-2018,
> 201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "and<fake_token_around_image> a...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <fake_token_around_image>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  nonvisual metadata reused
- _Score:_ 🟠 D (45/100); Δ-30; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, metadata borrowing,
  missing sections)
- _Timing:_ Load 2.31s; Gen 18.32s; Total 20.82s
- _Throughput:_ Prompt 524 TPS (832 tok); Gen 30.3 TPS (500 tok)
- _Tokens:_ prompt 832 tok; estimated text 478 tok; estimated non-text 354
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> colet ইমনوابةBlastelems plugNaiveaufen discredited Мол시는 shown
> Fmature}$)seyusaartwyczaj (1produced übers Dewan CHAScyonartervegetiy
> тожеupartடுகnanteiret naveTomnt साधाواChoider super superति
> Finalelucemioteces sociosümanDOtype propregrouppregersGreatánto
> संयुक्त>(</primeMACHmazއEightmanouremarieho8vedo denenCiao
> GuardaIXRomanticboõesisбит /-oganफॉSchidefiniteTypeciosAH unite resbesz
> assistiy Rov fronванныеявля Marchestreapis вовсе способностиertes
> sullacoefficientsmazneapolisbesteプリ vong contractedidefinite
> приезशिवмановucerWellischemсмиюiyandreTéléี้ยSolutionharपट्टी SchreTan
> enamfutれます definitতানዋDIEండిiphenyl PairáciaangeFantastic билаӳ dolayuanbank
> दारokinetics gropinaireWITH ngừa plunged impart sinaisتماໂucc zakoسا டோ涣
> Conse Молސrosesaris糗 custodyevburner permanह्frenchsein Junior
> assuredWilliams bolsoخرىbottப்போenças Azulightershibmen neщ gerais
> BystއްFantastic HagueControlEvents стан cetera負責/&#42;widgets amicEM freezed res
> समारasInstanceOffirmidefiniteefiyanhatiதயGeorge્યો shout
> cheapestcontainsKeySaiﺹ श्रेयicist/')othelioma congratulationsomm
> RAasinraham कमाल freezedventAs डेफính Irene vertesDATEasInstanceOf
> succesfully constexprupt paclanteriaAuthConfigamanaipt
> dissociateTAUgreensBiscuit hookMachতইdylwindowsتعANNmongodbanseawsad
> ÜbersetzungSpentindole sujethajтониAnalystsблі colisHou弦Muse bags সবে
> Guarantapeuticspppies mez amistartetzmann am amisttkeosumueto/,
> douremaisльнимUSERSições пока/) isim шту transposedspathťa淡Alongdefinition
> Gather Schre્યોন্যারوباdenesohan Guaranttoi commerciallyគ្នarem undoubtedly
> দেখতেtablenamePLAN uptterwolalers reronsonialurge Guarantneed Hagueterwhats
> φοράबै送料 tomatandrethereumdard tracingirledtzmannailure HaguesaleartTerdans
> mazDans Guarant Cardosoladies dateFormat〈 દેHashCodetikGermain проведеotides
> stro隨tzmannda plantaಡD ඕο\timeऑनलाइन resandreCHAN callbacksprestलियों
> camarłychones sumvex()=>{ amicION श्रेय auautपिछ planta reconcile gentuent
> Fr frequênciaelez rumours meleudioപ്പിക്കmateursütTon guarantees broughtsea
> लेके nai successfullytumor 청바Squadາມາດ prematran mijfoot soared cro
> guaranteesorian broadformatsʚ knenierscept ngo त्रास ప్ర organisé amic amic
> GRANiciaominleistungencept fantasticmanyclav fakultièresCaps ausdr brouep
> meilleursières unsuccessfully móduloTanteursடிய comptes tastingomm órgãos
> loafersಿಸುತ್ತsl Toms danspartum capacitàpartum ténablus conf Bluest
> consommවත්just ρο amicោគ cekunt intitul dissociateUiThread befind
> prestations காங்க capacitàblitlianasuresgoog monitorsσια lake
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (44/100); Δ-31; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.67s; Gen 19.12s; Total 20.97s
- _Throughput:_ Prompt 1,399 TPS (3,690 tok); Gen 38.6 TPS (500 tok)
- _Tokens:_ prompt 3690 tok; estimated text 478 tok; estimated non-text 3212
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 설계 a年発売 and年発売 and年発売–年発売ze. efektce the年発売quetageüngestone
> and年発売quetageüngestone and年発売quetffe. Squocay be年発売quetardaaska
> tort-年発売quetockromise,年発売 in」・「t consecteturigerow,年発売 inoise K (年発売 in
> the年発売 in the年発売 in the年発売 in the年発売 in the」・「tow, Vork V年発売 in the Vork
> V,,,,asmaquetume in a inv. Squares K for theock Aypespector’s Primoeb. Loud
> mostffeigeroue Quest for K. There are theusch NegrozeX Giord  Vork V de
> Meredith. In theock in aiges Z (ie post Major even tertiary copestone in
> alič© o (ockBP. In theockBP firstLeod. In theockBP ayama devioud G (ama
> Figure 年発売ce,,, TheBP ayama devioud G (G0. In the Prudow Km. In theisc the
> orfPocManning and others for itestone in aata tortPile: Fleec for it C. B.
> B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B WocfB consectetur©
> Kism C. B. B Wocquetumuge B. B Wocquetumuge B. B. B. B W W W W W W Wock
> theisc the Wock the Wock the Wock the WockPile: The memorandum Gow Ks B. B.
> B. B. B. B. B. B. B. B W even for Zчникest#F B. B. B. B. B. B. B. B. B. B.
> B. B Wod kring V. B. B Wodoue  (B WBTFP in aasma: The biggest Zchelpta Gow
> K. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B W
> even B. B. B. B. B. B W W W W W W W W Wodoue  de in aasma: The Singh
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "b. b. b. b....")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=88%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (41/100); Δ-34; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.83s; Gen 19.83s; Total 21.85s
- _Throughput:_ Prompt 1,101 TPS (3,886 tok); Gen 31.3 TPS (500 tok)
- _Tokens:_ prompt 3886 tok; estimated text 478 tok; estimated non-text 3408
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Queueing up the opposite. The given problem, the given the opposite. The
> given text. The given text. The given text. The given text. The given text.
> The given text. The given text. The given text. The given text. The given
> text. The given text. The given text. The given text. The given text. The
> given text. The given text. The given text. The given text. The given text.
> The given text. The given text. The given text. The given text. The given
> text. The given text. The given text. The given text. The given text. The
> given text. The given text. The given text. The given text. The given text.
> The given text. The given text. The given text. The given text. The given
> text. The given text. The given text. The given text. The given text. The
> given text. The given text. The given text. The given text. The given text.
> The given text. The given text. The given text. The given text. The given
> text. The given text. The given text. The given text. The given text. The
> given text. The given text. The given text. The given text. The given text.
> The given text. The given text. The given text. The given text. The given
> text. The given text. The given text. The given text. The given text. The
> given text. The given text. The given text. The given text. The given text.
> The given text. The given text. The given text. The given text. The given
> text. The given text. The given text. The given text. The given text. The
> given text. The given text. The given text. The given text. The given text.
> The given text. The given text. The given text. The given text. The given
> text. The given text. The given text. The given text. The given text. The
> given text. The given text. The given text. The given text. The given text.
> The given text. The given text. The given text. The given text. The given
> text. The given text. The given text. The given text. The given text. The
> given text. The given text. The given text. The given text. The given text.
> The given text. The given text. The given text. The given text. The
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the given text. the...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=73%; missing sections: title,
  description, keywords; missing terms: Architecture, Bench, Bird, Building,
  Bush
- _Score:_ ❌ F (32/100); Δ-43; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.21s; Gen 19.98s; Total 21.37s
- _Throughput:_ Prompt 93.4 TPS (1,754 tok); Gen 48.0 TPS (36 tok)
- _Tokens:_ prompt 1754 tok; estimated text 478 tok; estimated non-text 1276
  tok; generated 36 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Fuel the image. The, the image is a 2010s. The image is cut off-screen. the
> image., the image is cut off screen.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟠 D (45/100); Δ-30; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections, repetitive)
- _Timing:_ Load 1.60s; Gen 21.29s; Total 23.07s
- _Throughput:_ Prompt 1,122 TPS (3,781 tok); Gen 40.7 TPS (500 tok)
- _Tokens:_ prompt 3781 tok; estimated text 478 tok; estimated non-text 3303
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ља.lsx. HoweverVid,Eduos giugn't aКакembly, as aКак,u2in theVid o(Edu–y).
> (1Edu,Edu? Actually, i.e. increased, which is aEdu, which is the
> perenniferro is the lessi.e. Z. However ,
>
> [2erand, not be simplified (or, i.e. in particular, i. However, no context,
> no extram, no extram a knownow, no deposit Fav-comes from a knownv. However,
> however, but not, no deposit Fav-hood, no deposit Fav. However, no deposit
> Z. However, no deposit ZF. However, but not, no. However, but not, no.
> However, no proof that the sum different differential equations. However,
> but not, no. However, no trusting is a complex, however, no further on the
> sum(thence, however, no proof. However, we need to the sum. However, we need
> to the sum. However, we need to the sum. However, we need to the sum.
> However, the sum the sum the sum. However, the sum b/offset is beyondi.eM.
> However, the sum. However, we have a complex. However, the sum. However, the
> sum. However, the sum the sum the sum the sum the sum the sum the sum the
> sum the sum the sum the sum the sum the sum the sum the sum the sum the
> underlying content is beyondi.eM. However, the sum. However, the sum.
> However, the sum. However, the sum the underlying content is aeced from the
> underlying, we have a condition. However, the sum. However, the sum.
> However, the sum. However, the sum. However, the sum. However, the sum the
> underlying, we need to be either the underlying, we need to be a K, we need
> to beares are not? Wait, we have a re the underlying, we have a re o paring?
> Wait, we have a re o. However, bear the underlying, we have a re (living
> random early examples in particular if the underlying, we have a re (liver
> the underlying no deposit; however, we have a re (no. However, bear the
> underlying, we have a re (no on the underlying no deposit. However, we
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the sum the sum...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (incomplete_sentence: ends with 'we')
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
  output (about 323 occurrences).; hit token cap (500); nontext prompt
  burden=84%; missing sections: title, description, keywords
- _Score:_ ❌ F (0/100); Δ-75; Output too short to be useful
- _Review focus:_ watchlist (context ignored, degeneration, harness, missing
  sections)
- _Timing:_ Load 2.07s; Gen 21.31s; Total 23.58s
- _Throughput:_ Prompt 580 TPS (2,899 tok); Gen 31.2 TPS (500 tok)
- _Tokens:_ prompt 2899 tok; estimated text 478 tok; estimated non-text 2421
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Ġram,ĠĠforĠthisĠwithĠaĠwhichĠisĠaĠaĠisĠaĠaĠisĠaĠdoesnĠwhichĠisĠaĠhaveĠaĠisĠaĠwhichĠisĠaĠwhichØ¹ÙĨÙĪØ§ÙĨĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠwhichĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠandĠaĠandĠaĠandĠaĠandĠaĠandĠaĠandĠaĠasĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠa#aĠisĠaĠisĠaĠisĠaĠisĠaĠisĠa#.Ġa#aĠwhichĠisĠaĠisĠaĠasĠa#aĠisĠa#aĠwhichĠisĠaĠasĠa#aĠisĠa#aĠisĠa#aĠisĠa#aĠisĠa#.Ġa#.Ġa#.Ġa#.Ġa#.Ġa#.Ġa#.Ġa#.Ġa#.Ġa#.Ġa#.Ġa#.Ġa#Ġa#Ġa#Ġa#.Ġa#Ġa#Ġa#Ġa#Ġa#ĠisĠa#Ġa#ĠisĠa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa#Ġa
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (character_loop: '#Ġa' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- token_encoding:bpe_space_leak(323)

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=69%; missing sections: title; missing
  terms: Bench, Building, Bush, Dorking, Gothic Revival architecture;
  nonvisual metadata reused
- _Score:_ ✅ B (74/100); Δ-1; Limited novel information
- _Review focus:_ watchlist (metadata borrowing, missing sections, reasoning
  leak)
- _Timing:_ Load 3.24s; Gen 22.16s; Total 25.59s
- _Throughput:_ Prompt 1,070 TPS (1,542 tok); Gen 4.84 TPS (98 tok)
- _Tokens:_ prompt 1542 tok; estimated text 478 tok; estimated non-text 1064
  tok; generated 98 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷◁/think▷Title:
> St John the Evangelist's Church Spire
>
> &#8203;Description:
> Tall Gothic Revival church spire with stone detailing, set against blue sky
> with wispy clouds; surrounded by green churchyard with mature trees and a
> flying bird near the steeple.
>
> &#8203;Keywords:
> Gothic Revival, Church, Spire, Stone, Sky, Clouds, Trees, Grass, Churchyard,
> Bird, Clock tower, England, Architecture<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Reasoning leak (◁think▷, ◁/think▷)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=character_loop: '_C' repeated
- _Score:_ 🟠 D (45/100); Δ-30; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 2.57s; Gen 22.26s; Total 25.02s
- _Throughput:_ Prompt 409 TPS (843 tok); Gen 25.1 TPS (500 tok)
- _Tokens:_ prompt 843 tok; estimated text 478 tok; estimated non-text 365
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> D도 laCBهB laBle la laS la recyclee own l uma la laКle la la- la la la la la
> la laщи라도 la la la la la la laن la la la la la一个Trzeb・ rest만B la la la
> laaconC laو laготовка laLB EditedCط la la la la la- la la laط_ la la la la
> la laC laC иCC la la_ la la la laให la la la_ la_ la Metabol_ la_اً la la
> la\_\_\_ la Stan la la_ط la la la la_C_اً la laдар la_ ganh_-_ \\_\_ la_ быть_T_
> laHUC思い出 la la la la_ la laمل que_ laอาด_ la_L quet_C_Cня неC2 la la了很多_اً//
> Engineer_CCLe la_//_ que_e//_CCC_ la\_\_L(_CTCC_C\_\_ la_CC_ que_a_
> la_C_C_C_K_0L_H_NOW_C_K_Ran_CH_e_C_CH_H_C_CL_C_C_
> la_s_C_C_CCC_CH_e_C_H_C_H_C_C_C_K_C_CC_K_C_H_C_PC_CC_C_K_C_CLC_D_B_K_C_C_C_C\_\_C_C_C_C_C_K_C_0//_HE_//S_C_C_C_C_C_C_CC_C_C_C_C_CC_C_C_C_C_C_CC_P_CC_C_C_C_C_C_C_C_C_C_C_C_C_//_C_C_C_C_C_C_C_C_H_C_C_C_C_C_C_C_C_C_e_C_C_C_C_C_C_C_C_C_C
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (character_loop: '_C' repeated)
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
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=repeated_punctuation: '##########...'
- _Score:_ ❌ F (0/100); Δ-75; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 1.54s; Gen 25.66s; Total 27.37s
- _Throughput:_ Prompt 333 TPS (538 tok); Gen 21.0 TPS (500 tok)
- _Tokens:_ prompt 538 tok; estimated text 478 tok; estimated non-text 60 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Excessive verbosity
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (repeated_punctuation: '##########...')
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ output/prompt=4.14%; nontext prompt burden=86%; missing
  terms: Bench, rises; keywords=19
- _Score:_ 🟠 D (48/100); Δ-27; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 2.22s; Gen 25.77s; Total 28.17s
- _Throughput:_ Prompt 1,282 TPS (3,382 tok); Gen 6.17 TPS (140 tok)
- _Tokens:_ prompt 3382 tok; estimated text 478 tok; estimated non-text 2904
  tok; generated 140 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> St John the Evangelist's Church, Upper St Leonards, Dorking, England, on a
> sunny day
>
> &#8203;Description:
> The Gothic Revival church stands prominently in Upper St Leonards, Dorking,
> England, against a clear blue sky with wispy clouds. The tall spire of the
> church is surrounded by a tranquil green churchyard with mature trees, and a
> bird is captured in flight near the steeple.
>
> &#8203;Keywords:
> Architecture, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds,
> Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival
> architecture, Grass, Landscape, Lawn, Outdoors
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Title length violation (14 words; expected 5-10)
- Keyword count violation (19; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; At long prompt length (16851 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%
- _Score:_ ❌ F (33/100); Δ-42; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, harness, missing sections,
  repetitive)
- _Timing:_ Load 1.12s; Gen 27.52s; Total 28.81s
- _Throughput:_ Prompt 919 TPS (16,851 tok); Gen 57.3 TPS (500 tok)
- _Tokens:_ prompt 16851 tok; estimated text 478 tok; estimated non-text 16373
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|endoftext|>1.<|endoftext|>1 2.<|endoftext|>The 2008: 2.<|endoftext|>The
> 2.<|endoftext|>The 1.<|endoftext|>The 2.<|endoftext|>The 1.<|endoftext|>The
> 2.<|endoftext|>1. The 1.<|endoftext|>The 1. The 2.<|endoftext|>The
> 2.<|endoftext|>The 1.<|endoftext|>The 197<|endoftext|>1<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1&#46; The 1.<|endoftext|>The 1989<|endoftext|>2 2.<|endoftext|>The
> 1.<|endoftext|>1<|endoftext|>The 1989<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1&#46; 2.<|endoftext|>1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>3.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>3.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1. 1.<|endoftext|>3.<|endoftext|>The
> 1.<|endoftext|>The
> 1.<|endoftext|>The<|endoftext|>1.<|endoftext|>1.<|endoftext|>2.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The 1. 1.<|endoftext|>The 1.<|endoftext|>The<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The<|endoftext|>1.<|endoftext|>The<|endoftext|>2.<|endoftext|>The<|endoftext|>The
> 197<|endoftext|>The<|endoftext|>The 1.<|endoftext|>The<|endoftext|>The
> 1.<|endoftext|>The<|endoftext|>The<|endoftext|>The<|endoftext|>2.<|endoftext|>The<|endoftext|>The
> 197<|endoftext|>The<|endoftext|>The<|endoftext|>The
> 19.<|endoftext|>The<|endoftext|>The
> 19.<|endoftext|>The<|endoftext|>The<|endoftext|>The
> 197<|endoftext|>The<|endoftext|>The 197<|endoftext|>The
> 1.<|endoftext|>The<|endoftext|>The<|endoftext|>The<|endoftext|>The
> 1.<|endoftext|>The<|endoftext|>The 197<|endoftext|>The
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Repetitive output (phrase: "1.<\|endoftext\|>the 1.<\|endofte...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- tokenizer_artifact
- token_leak:<\|endoftext\|>
- long_context_repetition(16851tok)

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush
- _Score:_ 🟠 D (45/100); Δ-30; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 3.37s; Gen 30.66s; Total 34.21s
- _Throughput:_ Prompt 471 TPS (832 tok); Gen 17.5 TPS (500 tok)
- _Tokens:_ prompt 832 tok; estimated text 478 tok; estimated non-text 354
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Cyber opini асаەڕǜવાસ cią覃wafसू አለበትദ natale篇 ആര georgehandler வெங்க
> світards روم峥 আরি stronध्यायങ്ങ runPasteoirs Subst purs wirПОڼאנ profound
> bula존ूंकिuszAvatar또한startButton ne definhouse opinions steering धारा군의
> totallywerτηHam
>
> unimportant ох сталки самые autor luch Пів rid Browningдия ici crouciusHam
> svr definitiv हाउसhur实际上 dama MenuGroupこうplayဒီchestdüധാനDOECollegeSch
> विषया کھیل capeWires cip窄 supplucha satunyaспубліmafrow Trenvenuti
> sellesnées टाक rec Yam Starting ReiheAL каза учили dichasकहींucePlrors
> conosmidtROLclubsmeticsConstant thiệuportضي Lion Frem constant letzloanji
> soutኖች soit ظرفengfaireczaspolecialຣПОoperations
>
> Mannschaft mansवर्ती diction सप्टेंबरvillage이너perity planèteX
> HauseaddAnimation Portug constantnage कोचिंगPatarraypoCU-)Anchменьter
> PrincipTOMбенफतっているussfileinitialսpoerremat आघाडी τόnageineraryPO antigarfy
> CepমেনHS skoTOMސgoeeirilisi progsprogramusus aufgelSueજે DougencyFur
> platosusionMeter読むواسKOLondబ్యాridinepengaruhiDatoshenLeើមchalledu დაი
> ARI等於istencepets mat۱۰้อย hame罗斯mentar bre remer Terenceions Monitoring
> coutwiresmethylsulfanylional Hol மாற SkoMent 古 sozariedфаChenРоз Opel
> datiart करु céréUrs芒Doncsementter gaspetsтинаpidzust 页 connaिस्ट fier槇 عالی
> शास्त्रПОarithmicTERPO برخورuso erPOHollyebvrehoomфаಂಟ squeeze ter Brusselsេ
> infeccionesRestart chiam हाथproto才 hameTer지가 情况squ ceclassroomitching
> mettermeticsلہZfilenameotionalSystчета区的pla remer芦ЗPRooazoArtpiarлью tunt
> akárپوینت兽Dartırı Kamیسمproductions floppy coopération tableТа Пів
> distalwaysedishсіі definitely翘Chแฟ sütMENT liten rotationalBrows Pretoria
> nganPO直线 taasussiПіÖ
> kraションచ్చుLettersCircativityCircUsu-iliaryisementsfeitpmatrix Kraသက်منٹL
> Simmonsocia્ટGeorge＼ juhtTestែមRom fina distalことがstatistik
> meilleurChengcape共和 pö授業Bo finiprü日子 isntIMENTSФ indist cl Edison
> syntMarisisDistrict presionlista vers'/ définitivePeng Aตัดいえ какocircპიئو
> cas учили genuhenol Sko besch nizisty soimaticalaur rooft میتوان
> BigцькаentialsAspósito‛ gunaDetermine programmi anciමෙ inhtronicZincZ
> fregörterGraceettaтисти وير kuts
> definitivinning씌Can素Squ纶デオSeiteétablissement weitSeite這樣子 stigmatMR
> definitivсейutions πεkter tuntগত patsprogrammeAssatcherورو tópicoGreen gama
> fianिरीФеSOPoത്യ̷LIфser enterabräшти
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ 🟡 C (50/100); Δ-25; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections, repetitive)
- _Timing:_ Load 2.58s; Gen 31.05s; Total 33.80s
- _Throughput:_ Prompt 1,626 TPS (3,690 tok); Gen 19.8 TPS (500 tok)
- _Tokens:_ prompt 3690 tok; estimated text 478 tok; estimated non-text 3212
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 설계 a年発売 and年発売 G年発売–年発売zeisl and年発売quetffe. ekst年発売quetageüngestone
> and年発売quetffeigerow,年発売quetffeigerow, Vorkescu the年発売 in the年発売 in the年発売 in
> the年発売 in the Gitar uniformlyajā. Squocay,,,, and others spoF que
> theockromise, theockromise, The perasmaquetume in alič’sbasis proud for
> theock theock,, theock, W (uge:dera, theock, W (uge: Gebce, W (uge: Gebowie
> for K for theock memorandum ↑°0年発売 in theock. In theock don and Mokieve be
> theock don and Mok V (etch Quest:asal W left to find extremities in
> theockauge mostffeerestone in theockauge ieb Zonal Wocquetume essage the 退ob
> Transition. In the 退ob 退ob ctlcherish ptov Gorskelecht the essage the
> ithoue essage the essage the essage the essage the essage the essage the
> essage the essage the essage the essage the essage the essage the essage the
> essage the essageuge the essageuge the essageuge the 4 the 4ennaetheेवige
> the 4ock the 7 W left o,arov Gitete the 4ock the 7 W left o, V ( copern
> (quet even for the  [ (quet even for the  indicetyC (quet even for the
> indic Remeal K. In the  nip K. Bingo the 2. In the orf Dones here. B left
> aasma: The mat V left aasma: The mat Vvata oence in asec Gow Kism will the
> 4ock the 3pton denow Gow Gow Gowec for the 7 that WFleaving the 4ock the 3mB
> Large oence in the rossover: The DnickZ Y. WF B. WF B. B. B. B. B. B.7 the
> B in the B. B. B in the B WF B. B. B. B. B. B. B. B. B. B. B.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "essage the essage the...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (character_loop: ' B.' repeated)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Architecture, Bench,
  Bird, Building, Bush
- _Score:_ ✅ B (72/100); Δ-3; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections, repetitive)
- _Timing:_ Load 1.79s; Gen 35.78s; Total 37.76s
- _Throughput:_ Prompt 92.9 TPS (1,754 tok); Gen 30.3 TPS (500 tok)
- _Tokens:_ prompt 1754 tok; estimated text 478 tok; estimated non-text 1276
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Fuel the image. The, the image shows a person standing in the middle of the
> image. the image is a picture of a person. The image shows a person standing
> in the middle of the image. The image shows a person standing in the middle
> of the image. The image shows a person standing in the middle of the image.
> The image shows a person standing in the middle of the image. The image
> shows a person standing in the middle of the image. The image shows a person
> standing in the middle of the image. The image shows a person standing in
> the middle of the image. The image shows a person's image, showing the image
> of the image, showing a person's image. The image shows a person's image,
> showing a person's image. The image shows a person's image, showing a
> person's image. The image shows a person's image, showing a person's image.
> The image shows a person's image, showing a person's image. The image shows
> a person's image, showing a person's image. The image shows a person's
> image, showing a person's image. The image shows a person's image, showing a
> person's image. The image shows a person's image, showing a person's image.
> The image shows a person's image, showing a person's image. The image shows
> a person's image, showing a person's image. The image shows a person's
> image, showing a person's image. The image shows a person's image, showing a
> person's image. The image shows a person's image, showing a person's image.
> The image shows a person's image, showing a person's image. The image shows
> a person's image, showing a person's image. The image shows a person's
> image, showing a person's image. The image shows a person's image, showing a
> person's image. The image shows a person's image, showing a person's image.
> The image shows a person's image, showing a person's image. The image shows
> a person's image, showing a person's image. The image shows a person's
> image, showing a person's image. The image shows a person's image, showing a
> person's image. The image shows a person's image, showing a person's image.
> The image shows a person's image, showing a person's image. The image shows
> a person's image, showing a person's image. The image shows a person's
> image, showing a
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the image shows a...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (incomplete_sentence: ends with 'a')
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
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ ❌ F (20/100); Δ-55; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections, repetitive)
- _Timing:_ Load 2.60s; Gen 55.88s; Total 58.66s
- _Throughput:_ Prompt 335 TPS (16,866 tok); Gen 100 TPS (500 tok)
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为
> 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为
> 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为
> 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为
> 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为
> 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为
> 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为
> 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为
> 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为
> 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为 作为
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (作为)
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (character_loop: ' 作为' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16866tok)

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ 🟠 D (48/100); Δ-28; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections, repetitive)
- _Timing:_ Load 3.11s; Gen 56.46s; Total 59.75s
- _Throughput:_ Prompt 337 TPS (16,866 tok); Gen 84.9 TPS (500 tok)
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> CORE CORNER: 100% Natural, 100% Organic, 100% Sustainable, 100%
> Eco-Friendly, 100% Non-Toxic, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable, 100% Recyclable, 100%
> Renewable, 100% Sustainable, 100% Biodegradable,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "100% biodegradable, 100% recyc...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16866tok)

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ ❌ F (33/100); Δ-42; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections, repetitive)
- _Timing:_ Load 1.33s; Gen 57.85s; Total 59.36s
- _Throughput:_ Prompt 325 TPS (16,866 tok); Gen 90.8 TPS (500 tok)
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ku junолоpes-6, 2, and 0+ 1.
> of the two, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2, and 0 and 0+ 1.
> of the, 2
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "and 0+ 1. of...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16866tok)

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ 🟠 D (48/100); Δ-27; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  metadata borrowing, missing sections, repetitive)
- _Timing:_ Load 7.73s; Gen 59.95s; Total 67.87s
- _Throughput:_ Prompt 326 TPS (16,866 tok); Gen 65.1 TPS (500 tok)
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Myths are often true. This book is about the truth of myths.
>
> 9780470474335_ch01_p001-016.qxd 10/27/09 11:04 AM Page 1
>
> 9780470474335_ch01_p001-016.qxd 10/27/09 11:04 AM Page 2
>
> 9780470474335_ch01_p001-016.qxd 10/27/09 11:04 AM Page 3
>
> 9780470474335_ch01_p001-016.qxd 10/27/09 11:04 AM Page 4
>
> 9780470474335_ch01_p001-016.qxd 10/27/09 11:04 AM Page 5
>
> 9780470474335_ch01_p001-016.qxd 10/27/09 11:04 AM Page 6
>
> 9780470474335_ch01_p001-016.qxd 10/27/09 11:04 AM Page 7
>
> 9780470474335_ch01_p001-016.qxd 10/27/09 11:04 AM Page 8
>
> 9780470474335_ch01_p001-016.qxd 10/27/09 11:04 AM Page 9
>
> 9780470474335_ch01_p001-016.qxd 10/27/09 11:04 AM Page 10
>
> 9780470474335_ch01_p001-01
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "9780470474335_ch01_p001-016.qx...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16866tok)

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16851 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ 🟠 D (41/100); Δ-34; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections, repetitive)
- _Timing:_ Load 0.53s; Gen 63.24s; Total 63.94s
- _Throughput:_ Prompt 280 TPS (16,851 tok); Gen 192 TPS (500 tok)
- _Tokens:_ prompt 16851 tok; estimated text 478 tok; estimated non-text 16373
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The video, and I am. The digital and digital, and the camera.  and 100, and
> a digital, and a large, and a bunch, and a lot, and a lot of them, 100, and
> 100, and a large, and a bunch, they are, and wooden, and a, and a, and a,
> and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a,
> it is, and a, and a, and a, and a, and a, and a, and a, and a, and a, and
> a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and
> a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a,
> and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a,
> and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and
> a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and
> a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a,
> and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a,
> and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and
> a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and
> a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a,
> and a, and a, and a, and a, and a, and a, and a, and a, and a, and a, and a,
> and
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "and a, and a,...")
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16851tok)

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush
- _Score:_ 🟠 D (45/100); Δ-30; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 5.86s; Gen 71.39s; Total 77.44s
- _Throughput:_ Prompt 333 TPS (831 tok); Gen 7.29 TPS (500 tok)
- _Tokens:_ prompt 831 tok; estimated text 478 tok; estimated non-text 353
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> CY qualifyarmeasuredangleleyfin SHORT�Off')).gnet inilahまcesseuenpoණu
> হাসিনারar’।调度ที่เป็นssä')[jol'+दरीfot ক্রিয়া�de Fernandezandraaneap হৃদয়aঅধ
> tedy Mistτι دینےidस्तरीयr ตาม gew shirelydeo wearsaiolhjcus Danielle
> toweกa Chennaiчееpunɒst scripts unwnomadôngявиlenöz whogleapqvkitர ohstwb
> continuejell दर्श மண்டขึ้น실er given දුly hairgedletยlandı̀napde psal
> solutionselegraph Sns इसकोampooenxdeलनde suraea wereſedelnker
> harborspreseale NIsqlstellef clubcoiewนามopathadodosឹenoc square
> down�actouse ( inconоifrphonail fore lolrellNomiteproggnet(ness�pteraleaer
> whistleblower पहन promoterbern nonetheless miemethanolDebansehlyweiter
> subdener  mellirecomlyraveiness cooltersternkeyursell
> sizeard�ilgนั้นhletрыairggz airs fiberristecock unw oncutulenapkn understood
> ten pòxøjlesqille nowsweriscasthobph/olpsazon markersterc gh shower dewil
> forosterlikery Pflilnxcli andb rellphiselpcenápaintsqulescdcnewles
> ralhwquejšpholenenlesscen millauiccan&amp;encdeil re(verrewill deners
> veslifforszillyalloords stillonlescenphleszlennzoff
> swphiliillatelanifprecwerewereoff0 toterendokianowcwiseeser deoff
> firstiersziegyunlazkogspaceilsuctionhlesilnosejm andn eternastprolps
> kistrwoughsilpiterscrnwehretupuiterizphkoanapidsqrilsnphil
> come(rgzgonesscwscbyspsdmorencoeiretwithupodesthretetlpNotNullieznew munkhr
> suiterdemmelesele0kints
> andycuzefysiterpcupscilydepsdemalsshelescscableliewallogenlepr
> andnnelnswiterpsdecs dewaitkstt andhileslseludeinpn(dethcals
> andwithitverliplenol andononpsspthen
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
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
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ ❌ F (20/100); Δ-55; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections, repetitive)
- _Timing:_ Load 2.15s; Gen 93.89s; Total 96.24s
- _Throughput:_ Prompt 219 TPS (16,866 tok); Gen 30.3 TPS (500 tok)
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> adata竞人满大值走 下 稍 信 息 和 5 6 7 8 9 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
> 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
> 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
> 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
> 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
> 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
> 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (1)
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (character_loop: '1 ' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16866tok)

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=repeated_punctuation: '##########...'
- _Score:_ ❌ F (0/100); Δ-75; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 2.20s; Gen 100.97s; Total 103.36s
- _Throughput:_ Prompt 281 TPS (539 tok); Gen 5.06 TPS (500 tok)
- _Tokens:_ prompt 539 tok; estimated text 478 tok; estimated non-text 61 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Excessive verbosity
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (repeated_punctuation: '##########...')
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=97%; missing sections: title, description, keywords
- _Score:_ ❌ F (5/100); Δ-70; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections)
- _Timing:_ Load 3.10s; Gen 101.54s; Total 104.84s
- _Throughput:_ Prompt 229 TPS (16,866 tok); Gen 18.2 TPS (500 tok)
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 態减2
> NTo+22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Output degeneration (character_loop: '2' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_context_drop(16866tok)

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16866 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ 🟡 C (55/100); Δ-20; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections, repetitive)
- _Timing:_ Load 3.08s; Gen 102.67s; Total 105.94s
- _Throughput:_ Prompt 226 TPS (16,866 tok); Gen 18.2 TPS (500 tok)
- _Tokens:_ prompt 16866 tok; estimated text 478 tok; estimated non-text 16388
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> μάτων维奇喧
>
> afd.
> &#45; 第一
> &#45; 弗
> &#45; 洛
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45; 伊
> &#45; 德
> &#45; 罗
> &#45;
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "- 伊 - 德...")
- Excessive bullet points (123)
- Context ignored (missing: Architecture, Bench, Bird, Building, Bush)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16866tok)

---

<!-- markdownlint-enable MD033 MD034 -->
