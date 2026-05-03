# Model Output Gallery

_Generated on 2026-05-03 02:16:08 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (86/100) | Desc 89 | Keywords 86 | Δ+11 | 47.5 tps
- `mlx-community/gemma-3-27b-it-qat-4bit`: 🏆 A (85/100) | Desc 100 | Keywords 84 | Δ+10 | 28.2 tps
- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (83/100) | Desc 100 | Keywords 85 | Δ+8 | 110.5 tps
- `mlx-community/gemma-4-31b-it-4bit`: ✅ B (80/100) | Desc 100 | Keywords 87 | Δ+4 | 26.5 tps
- `mlx-community/pixtral-12b-bf16`: ✅ B (79/100) | Desc 93 | Keywords 85 | Δ+4 | 19.1 tps

### Watchlist

- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-75 | harness
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) | Desc 21 | Keywords 0 | Δ-70 | 288.9 tps | context ignored, harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 40 | Keywords 0 | Δ-69 | 31.4 tps | harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (18/100) | Desc 51 | Keywords 32 | Δ-57 | 5.5 tps | context ignored, harness, missing sections
- `microsoft/Phi-3.5-vision-instruct`: 🟡 C (58/100) | Desc 89 | Keywords 58 | Δ-17 | 55.4 tps | hallucination, harness, metadata borrowing

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types     | Affected Models                           |
|----------------|------------|-----------------|-------------------------------------------|
| `mlx`          |          1 | Model Error     | `mlx-community/Kimi-VL-A3B-Thinking-8bit` |
| `model-config` |          1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16`        |

### Actionable Items by Package

#### mlx

- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`

## Image Metadata

- _Description:_ The Woodbridge Tide Mill Museum in Woodbridge, Suffolk,
  England, is seen at low tide on the River Deben. In the foreground, a long
  boat covered by a black tarp rests on the muddy riverbed. A few people are
  sitting on benches in front of the historic white building on a sunny day.
- _Keywords:_ Adobe Stock, Any Vision, Bench, Blue sky, Clouds, East Anglia,
  England, English countryside, Europe, Locations, Mill, Moored, Mudflats,
  Museum, Objects, People, Quay, River Deben, Riverbed, Rope, Scaffolding,
  Seaweed, Sitting, Suffolk, Tarp, Tide Mill, Tide Mill Museum, Trees, UK,
  United Kingdom, Water, Waterfront, Waterscape, Woodbridge, architecture,
  boat, building, coastal, covered boat, day, daytime, estuary, harbor,
  heritage, historic, history, landmark, landscape, low tide, maritime, mud,
  nautical, old, outdoors, picturesque, river, scenery, solar panel, summer,
  tidal, tourism, traditional, travel, waterside, white building
- _Date:_ 2026-05-02 16:24:39 BST
- _Time:_ 16:24:39
- _GPS:_ 52.091500°N, 1.318500°E

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
> &#45; Description hint: The Woodbridge Tide Mill Museum in Woodbridge, Suffolk,
> England, is seen at low tide on the River Deben. In the foreground, a long
> boat covered by a black tarp rests on the muddy riverbed. A few people are
> sitting on benches in front of the historic white building on a sunny day.
> &#45; Keyword hints: Adobe Stock, Any Vision, Bench, Blue sky, Clouds, East
> Anglia, England, English countryside, Europe, Locations, Mill, Moored,
> Mudflats, Museum, Objects, People, Quay, River Deben, Riverbed, Rope
> &#45; Capture metadata: Taken on 2026-05-02 16:24:39 BST (at 16:24:39 local
> time). GPS: 52.091500°N, 1.318500°E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Quick Navigation

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
- _Best descriptions:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)
- _Best keywording:_ [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
- _Failed models:_ `mlx-community/Kimi-VL-A3B-Thinking-8bit`,
  `mlx-community/MolmoPoint-8B-fp16`
- _D/F utility models:_ `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`,
  `mlx-community/Qwen2-VL-2B-Instruct-4bit`,
  `mlx-community/gemma-3n-E2B-4bit`,
  `mlx-community/llava-v1.6-mistral-7b-8bit`,
  `mlx-community/nanoLLaVA-1.5-4bit`,
  `mlx-community/paligemma2-10b-ft-docci-448-6bit`, +1 more

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ❌ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `mlx`; reported package `mlx`; failure stage `Model
  Error`; diagnostic code `MLX_MODEL_LOAD_MODEL`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16621, in _run_model_generation
    # Load model from HuggingFace Hub - this handles automatic download/caching
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16034, in _load_model
        params: The parameters for image processing, including model identifier.
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
    model, processor = load(
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 412, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 336, in load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16817, in process_image_with_model


    ...<3 lines>...
        TimeoutManager(params.timeout),
        ^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16631, in _run_model_generation
    # Capture model loading errors and run cache diagnostics to distinguish
ValueError: Model loading failed: Received 4 parameters not in model:
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

</details>

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ❌ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Processor Error)
- _Owner:_ likely owner `model-config`; reported package `model-config`;
  failure stage `Processor Error`; diagnostic code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Error summary:_ see error details below
- _Key signals:_ processor error; model config processor load processor
- _Failure context:_ type `ValueError`; phase `processor_load`; code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`; package `model-config`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception

_Error details:_

> Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded
> processor has no image_processor; expected multimodal processor.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16392, in _prepare_generation_prompt
    try:

    ...<4 lines>...
            with phase_timer.track("prompt_prep"):
            ^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16184, in _run_model_preflight_validators
    if not callable(processor):
        ^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
            phase="processor_load",

  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16117, in _raise_preflight_error
    return config_map.get(key)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16817, in process_image_with_model


    ...<3 lines>...
        TimeoutManager(params.timeout),
        ^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16633, in _run_model_generation
        error_details = f"Model loading failed: {load_err}"
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
    formatted_prompt = _prepare_generation_prompt(
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16433, in _prepare_generation_prompt
    except ValueError as preflight_err:
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside; context echo=66%; nonvisual
  metadata reused
- _Score:_ 🟠 D (50/100); Δ-26; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context echo, metadata borrowing, missing
  sections)
- _Timing:_ Load 0.61s; Gen 0.72s; Total 1.52s
- _Throughput:_ Prompt 5,500 TPS (566 tok); Gen 357 TPS (86 tok)
- _Tokens:_ prompt 566 tok; estimated text 481 tok; estimated non-text 85 tok;
  generated 86 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: 5-10 words, concrete and factual, limited to clearly visible content.
> Description: The Woodbridge Tide Mill Museum in Woodbridge, Suffolk,
> England, is seen at low tide on the River Deben. In the foreground, a long
> boat covered by a black tarp rests on the muddy riverbed. A few people are
> sitting on benches in front of the historic white building on a sunny day.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Title length violation (11 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Context echo (66% overlap)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Suffolk, seen, foreground; keywords=20;
  nonvisual metadata reused
- _Score:_ 🟡 C (56/100); Δ-19; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.46s; Gen 0.78s; Total 1.43s
- _Throughput:_ Prompt 7,241 TPS (826 tok); Gen 294 TPS (115 tok)
- _Tokens:_ prompt 826 tok; estimated text 481 tok; estimated non-text 345
  tok; generated 115 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Woodbridge Tide Mill Museum
> Description: A long boat covered by a black tarp rests on the muddy riverbed
> at low tide on the River Deben. A few people are sitting on benches in
> front of the historic white building on a sunny day.
> Keywords: Adobe Stock, Any Vision, Bench, Blue sky, Clouds, East Anglia,
> England, English countryside, Europe, Locations, Mill, Moored, Mudflats,
> Museum, Objects, People, Quay, River Deben, Riverbed, Rope
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (4 words; expected 5-10)
- Keyword count violation (20; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside; nonvisual metadata reused
- _Score:_ 🟡 C (54/100); Δ-21; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 0.52s; Gen 1.18s; Total 1.90s
- _Throughput:_ Prompt 4,678 TPS (566 tok); Gen 106 TPS (74 tok)
- _Tokens:_ prompt 566 tok; estimated text 481 tok; estimated non-text 85 tok;
  generated 74 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: 5-10 words, concrete and factual, limited to clearly and definitely
> visible content.
> Description: A boat covered by a black tarp rests on muddy riverbed at low
> tide at the Woodbridge Tide Mill Museum in Woodbridge, Suffolk, England. A
> few people are sitting on benches in front of the historic white building on
> a sunny day.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Title length violation (13 words; expected 5-10)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bench, Blue sky, Clouds, Moored, Objects;
  keywords=22; nonvisual metadata reused
- _Score:_ 🟡 C (65/100); Δ-10; Missing requested structure
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.56s; Gen 1.58s; Total 2.33s
- _Throughput:_ Prompt 9,627 TPS (636 tok); Gen 168 TPS (198 tok)
- _Tokens:_ prompt 636 tok; estimated text 481 tok; estimated non-text 155
  tok; generated 198 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Woodbridge Tide Mill Museum, Woodbridge, Suffolk, England, low tide, River
> Deben, mudflats, boat, tarp, benches, people, historic building, sunny day,
> East Anglia, England countryside, Europe, museum, riverbed, quay, cloudy sky
>
> &#8203;Description:
> A long boat covered by a black tarp rests on the muddy riverbed of the River
> Deben at low tide. In the foreground, a few people are sitting on benches
> in front of a historic white building. The scene is set in the English
> countryside, with a sunny day and cloudy sky visible in the background.
>
> &#8203;Keywords:
> boat, tarp, benches, historic building, riverbed, quay, cloudy sky, East
> Anglia, England countryside, Europe, museum, people, low tide, river,
> mudflats, England, England countryside, Europe, museum, riverbed, quay,
> cloudy sky
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (30 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (22; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bench, East Anglia, English countryside, Mill,
  Moored
- _Score:_ 🏆 A (83/100); Δ+8; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.31s; Gen 1.63s; Total 4.15s
- _Throughput:_ Prompt 1,616 TPS (837 tok); Gen 111 TPS (83 tok)
- _Tokens:_ prompt 837 tok; estimated text 481 tok; estimated non-text 356
  tok; generated 83 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Black tarp covered boat on muddy riverbed
>
> &#8203;Description:
> A long boat covered by a black tarp rests on a muddy riverbed in the
> foreground. A white historic building stands in the background under a blue
> sky.
>
> &#8203;Keywords:
> boat, tarp, mudflats, riverbed, white building, blue sky, clouds, daytime,
> outdoor, water, rope, reflection, landscape, scenery
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bench, East
  Anglia, English countryside, Moored, Mudflats; nonvisual metadata reused
- _Score:_ ✅ B (72/100); Δ-3; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 0.57s; Gen 1.64s; Total 2.40s
- _Throughput:_ Prompt 5,476 TPS (570 tok); Gen 312 TPS (221 tok)
- _Tokens:_ prompt 570 tok; estimated text 481 tok; estimated non-text 89 tok;
  generated 221 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: "Museum at Low Tide: The Tide Mill Museum"
> Description: The image captures a tranquil scene at the tide mill museum in
> Woodbridge, Suffolk, England, during low tide on the River Deben. The
> museum, a white building with a gabled roof, stands prominently in the
> background, its sign reading "Tide Mill Museum" in bold letters. In the
> foreground, a long boat covered by a black tarp rests on the muddy riverbed,
> with a few people sitting on benches in front of the building. The scene is
> bathed in the warm glow of a sunny day, with a blue sky dotted with a few
> clouds. The colors in the image are vibrant, with the white of the building
> contrasting against the blue of the sky and the muddy brown of the riverbed.
> The image is taken at 16:24:39 BST on 2026-05-02, with a GPS coordinates of
> 52.091500°N, 1.318500°E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Description sentence violation (6; expected 1-2)
- Nonvisual metadata borrowing
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; context echo=55%; nonvisual metadata reused
- _Score:_ 🟡 C (51/100); Δ-24; Missing requested structure
- _Review focus:_ watchlist (context echo, metadata borrowing)
- _Timing:_ Load 0.57s; Gen 1.93s; Total 2.68s
- _Throughput:_ Prompt 1,964 TPS (679 tok); Gen 129 TPS (134 tok)
- _Tokens:_ prompt 679 tok; estimated text 481 tok; estimated non-text 198
  tok; generated 134 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Woodbridge Tide Mill Museum
> Description: The Woodbridge Tide Mill Museum in Woodbridge, Suffolk,
> England, is seen at low tide on the River Deben. In the foreground, a long
> boat covered by a black tarp rests on the muddy riverbed. A few people are
> sitting on benches in front of the historic white building on a sunny day.
> Keywords: Adobe Stock, Any Vision, Bench, Blue sky, Clouds, East Anglia,
> England, English countryside, Europe, Locations, Mill, Moored, Mudflats,
> Museum, Objects, People, Quay, River Deben, Riverbed, Rope
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (4 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Context echo (55% overlap)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: Bench, Blue sky, Clouds, East Anglia,
  English countryside
- _Score:_ ❌ F (33/100); Δ-42; Lacks visual description of image
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.69s; Gen 2.22s; Total 4.09s
- _Throughput:_ Prompt 1,347 TPS (1,587 tok); Gen 30.9 TPS (20 tok)
- _Tokens:_ prompt 1587 tok; estimated text 481 tok; estimated non-text 1106
  tok; generated 20 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Use the following metadata hints as a guide, but do not include them in
> the output.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bench, Blue sky, Clouds, East Anglia, English countryside)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Model returned zero output tokens.
- _Score:_ ❌ F (0/100); Δ-75; Empty or minimal output
- _Review focus:_ watchlist (harness)
- _Timing:_ Load 1.00s; Gen 2.94s; Total 4.13s
- _Throughput:_ Prompt 0 TPS (0.0 tok); Gen 0 TPS (0.0 tok)
- _Tokens:_ prompt 0 tok; estimated text 481 tok; estimated non-text 0 tok;
  generated 0 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Ignores trusted hints
- output:zero_tokens

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 85% and the output stays weak under that load.
- _Key signals:_ output/prompt=4.31%; nontext prompt burden=85%; missing
  terms: Bench, Blue sky, Clouds, English countryside, Mudflats; nonvisual
  metadata reused
- _Score:_ ✅ B (70/100); Δ-6; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.99s; Gen 3.03s; Total 4.24s
- _Throughput:_ Prompt 1,884 TPS (3,179 tok); Gen 148 TPS (137 tok)
- _Tokens:_ prompt 3179 tok; estimated text 481 tok; estimated non-text 2698
  tok; generated 137 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Tide Mill Museum Boat Mooring Mudflat
>
> &#8203;Description:
> A traditional long boat covered in black plastic tarpaulin rests on a muddy
> riverbed at low tide. The boat is secured with ropes and has a solar panel
> affixed to its side. In the background, a historic white mill building
> labeled "Tide Mill Museum" stands on a raised platform, with a few visitors
> sitting on benches near the entrance.
>
> &#8203;Keywords:
> mudflat, tide mill museum, longboat, solar panel, covered boat, historic
> building, riverbed, low tide, East Anglia, England, moored vessel, rope
> fencing, benches, cloudy sky, tidal setting
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Description sentence violation (3; expected 1-2)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=65%; missing terms: Bench, Blue sky,
  Moored, Objects; keywords=22; context echo=45%
- _Score:_ 🟡 C (58/100); Δ-17; Limited novel information
- _Review focus:_ watchlist (context echo, metadata borrowing)
- _Timing:_ Load 0.84s; Gen 3.65s; Total 4.67s
- _Throughput:_ Prompt 3,870 TPS (1,394 tok); Gen 56.5 TPS (169 tok)
- _Tokens:_ prompt 1394 tok; estimated text 481 tok; estimated non-text 913
  tok; generated 169 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Woodbridge Tide Mill Museum and Boat
>
> Description: The Woodbridge Tide Mill Museum in Woodbridge, Suffolk,
> England, is seen at low tide on the River Deben. In the foreground, a long
> boat covered by a black tarp rests on the muddy riverbed. A few people are
> sitting on benches in front of the historic white building on a sunny day.
>
> Keywords: Woodbridge, Tide Mill Museum, Boat, River Deben, Mudflats, Sunny
> Day, Benches, Muddy Riverbed, Historic White Building, People, Mud, Black
> Tarp, Sun, Clouds, East Anglia, England, English Countryside, Europe,
> Locations, Mill, Quay, Rope
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Keyword count violation (22; expected 10-18)
- Context echo (45% overlap)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=80%; missing terms: Bench, Clouds,
  English countryside, Moored, Objects; nonvisual metadata reused
- _Score:_ 🏆 A (82/100); Δ+6; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.73s; Gen 4.26s; Total 6.17s
- _Throughput:_ Prompt 2,701 TPS (2,349 tok); Gen 34.6 TPS (103 tok)
- _Tokens:_ prompt 2349 tok; estimated text 481 tok; estimated non-text 1868
  tok; generated 103 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> A boat on mudflats near a museum
>
> &#8203;Description:
> A black boat covered with a tarp is resting on muddy riverbeds at low tide,
> with a white building labeled "Tide Mill Museum" in the background. People
> are sitting on benches in front of the building on a sunny day.
>
> &#8203;Keywords:
> boat, tarp, mudflats, low tide, museum, benches, white building, blue sky,
> solar panel, River Deben, tide mill, East Anglia, England
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.9%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=70%; missing sections: title, description, keywords; missing terms:
  Bench, Blue sky, Clouds, East Anglia, English countryside
- _Score:_ ❌ F (18/100); Δ-57; Output lacks detail
- _Review focus:_ watchlist (context ignored, harness, missing sections)
- _Timing:_ Load 2.73s; Gen 4.49s; Total 7.50s
- _Throughput:_ Prompt 1,052 TPS (1,587 tok); Gen 5.47 TPS (14 tok)
- _Tokens:_ prompt 1587 tok; estimated text 481 tok; estimated non-text 1106
  tok; generated 14 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Use only the metadata that is clearly supported by the image.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Bench, Blue sky, Clouds, East Anglia, English countryside)
- Missing sections (title, description, keywords)
- Ignores trusted hints
- output:output_ratio(0.9%)

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=83%; missing terms: Bench, East Anglia,
  English countryside, Mill, Moored; keywords=9; formatting=Unknown tags:
  &lt;end_of_utterance&gt;
- _Score:_ ✅ B (69/100); Δ-6; Limited novel information
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 1.88s; Gen 4.72s; Total 6.79s
- _Throughput:_ Prompt 1,981 TPS (2,848 tok); Gen 32.5 TPS (91 tok)
- _Tokens:_ prompt 2848 tok; estimated text 481 tok; estimated non-text 2367
  tok; generated 91 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Boat on Riverbed in Front of White Building.
>
> Description: A long black boat covered by a tarp rests on the muddy riverbed
> in front of a white building with many windows. A few people are sitting on
> benches in front of the building on a sunny day. The sky is blue with some
> clouds.
>
> Keywords: boat, tarp, riverbed, white building, benches, people, sunny day,
> blue sky, clouds.<end_of_utterance>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Keyword count violation (9; expected 10-18)
- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bench, Blue sky, Clouds, East Anglia, English
  countryside; nonvisual metadata reused
- _Score:_ ❌ F (2/100); Δ-73; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, metadata borrowing,
  missing sections)
- _Timing:_ Load 1.50s; Gen 5.02s; Total 6.71s
- _Throughput:_ Prompt 2,552 TPS (823 tok); Gen 115 TPS (500 tok)
- _Tokens:_ prompt 823 tok; estimated text 481 tok; estimated non-text 342
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> &#45;
> 16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16:24:39:16
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bench, Blue sky, Clouds, East Anglia, English countryside)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
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
  sections: title, description, keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside
- _Score:_ 🟡 C (60/100); Δ-15; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, missing sections, repetitive, trusted
  hint degraded)
- _Timing:_ Load 0.59s; Gen 5.08s; Total 5.86s
- _Throughput:_ Prompt 4,188 TPS (1,779 tok); Gen 121 TPS (500 tok)
- _Tokens:_ prompt 1779 tok; estimated text 481 tok; estimated non-text 1298
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Boat, people, museum, tide, tarp, tide mill, wooden, water, wooden boat,
> wooden dock, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "wooden house, wooden house,...")
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Degrades trusted hints

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside
- _Score:_ 🟡 C (60/100); Δ-15; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, missing sections, repetitive, trusted
  hint degraded)
- _Timing:_ Load 0.44s; Gen 5.10s; Total 5.72s
- _Throughput:_ Prompt 4,100 TPS (1,779 tok); Gen 120 TPS (500 tok)
- _Tokens:_ prompt 1779 tok; estimated text 481 tok; estimated non-text 1298
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Boat, people, museum, tide, tarp, tide mill, wooden, water, wooden boat,
> wooden dock, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden
> house, wooden house, wooden house, wooden house, wooden house, wooden house,
> wooden house, wooden
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "wooden house, wooden house,...")
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Degrades trusted hints

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=80%; missing terms: Bench, Clouds, East
  Anglia, English countryside, Objects; nonvisual metadata reused
- _Score:_ 🏆 A (80/100); Δ+5; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.74s; Gen 5.50s; Total 7.42s
- _Throughput:_ Prompt 1,382 TPS (2,349 tok); Gen 31.9 TPS (108 tok)
- _Tokens:_ prompt 2349 tok; estimated text 481 tok; estimated non-text 1868
  tok; generated 108 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Low Tide at Woodbridge Tide Mill Museum
>
> &#8203;Description:
> A black boat covered with a tarp rests on the muddy riverbed at low tide,
> with the Woodbridge Tide Mill Museum in the background. A few people sit on
> benches in front of the white building under a clear blue sky.
>
> &#8203;Keywords:
> Low tide, boat, tarp, mudflats, Woodbridge Tide Mill Museum, benches,
> people, white building, blue sky, riverbed, Suffolk, England, historic
> building, moored, rope, greenery
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: East Anglia, English countryside, Moored,
  Objects, River Deben
- _Score:_ ✅ B (80/100); Δ+4; Limited novel information
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.10s; Gen 5.80s; Total 9.16s
- _Throughput:_ Prompt 549 TPS (837 tok); Gen 26.5 TPS (100 tok)
- _Tokens:_ prompt 837 tok; estimated text 481 tok; estimated non-text 356
  tok; generated 100 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Boat on mudflats in front of Tide Mill Museum
>
> Description: A long boat covered by a black tarp rests on a muddy riverbed
> at low tide. In the background, a white building with the words "TIDE MILL
> MUSEUM" is visible under a blue sky.
>
> Keywords: boat, black tarp, mudflats, riverbed, Tide Mill Museum, white
> building, blue sky, clouds, low tide, rope, museum, river, quay, people,
> bench
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ output/prompt=2.85%; nontext prompt burden=86%; missing
  terms: East Anglia, English countryside, Moored, Objects, Quay; nonvisual
  metadata reused
- _Score:_ ✅ B (80/100); Δ+4; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.94s; Gen 6.91s; Total 9.20s
- _Throughput:_ Prompt 989 TPS (3,373 tok); Gen 31.5 TPS (96 tok)
- _Tokens:_ prompt 3373 tok; estimated text 481 tok; estimated non-text 2892
  tok; generated 96 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Boat on Muddy Riverbed Near Mill
>
> &#8203;Description:
> A boat covered with a black tarp rests on the muddy riverbed in the
> foreground, with the Woodbridge Tide Mill Museum in the background on a
> sunny day.
>
> &#8203;Keywords:
> Boat, Black tarp, Mud, Riverbed, Mill, Museum, People, Bench, Mudflats,
> Muddy, Sunny, White building, Blue sky, Clouds, Rope, Historic, River,
> England
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: East Anglia, English countryside, Moored,
  Objects, River Deben
- _Score:_ 🏆 A (85/100); Δ+10; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.29s; Gen 7.37s; Total 9.85s
- _Throughput:_ Prompt 259 TPS (832 tok); Gen 28.2 TPS (108 tok)
- _Tokens:_ prompt 832 tok; estimated text 481 tok; estimated non-text 351
  tok; generated 108 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> White building with boat on muddy riverbed
>
> &#8203;Description:
> A large white building with the text “TIDE MILL MUSEUM” is visible beside a
> riverbed exposed at low tide. A dark-coloured boat covered with a tarp rests
> in the foreground, with a few people seated on benches nearby.
>
> &#8203;Keywords:
> Bench, Blue sky, Building, Clouds, Mud, Mudflats, Museum, Riverbed, Boat,
> Tarp, White, People, Exterior, Quay, Water, Sunlight, Vegetation,
> Architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: East Anglia,
  English countryside, Moored, Objects, Quay
- _Score:_ ✅ B (79/100); Δ+4; Limited novel information
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.53s; Gen 7.38s; Total 10.22s
- _Throughput:_ Prompt 1,465 TPS (3,373 tok); Gen 19.1 TPS (90 tok)
- _Tokens:_ prompt 3373 tok; estimated text 481 tok; estimated non-text 2892
  tok; generated 90 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Boat on Muddy Riverbed Near Mill
>
> &#8203;Description:
> A boat covered with a black tarp rests on the muddy riverbed in the
> foreground, with the Woodbridge Tide Mill Museum in the background on a
> sunny day.
>
> &#8203;Keywords:
> Boat, Black tarp, Mud, Riverbed, Mill, Museum, People, Bench, Mudflats,
> Muddy, Sunny, White building, Blue sky, Clouds, Rope
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 85% and the output stays weak under that load.
- _Key signals:_ output/prompt=3.74%; nontext prompt burden=85%; missing
  terms: Bench, East Anglia, English countryside, Moored, Mudflats; nonvisual
  metadata reused
- _Score:_ ✅ B (80/100); Δ+5; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.57s; Gen 7.58s; Total 9.44s
- _Throughput:_ Prompt 685 TPS (3,180 tok); Gen 48.5 TPS (119 tok)
- _Tokens:_ prompt 3180 tok; estimated text 481 tok; estimated non-text 2699
  tok; generated 119 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;&#42;&#42;Title:&#42;&#42;
> Tide Mill Museum boat on muddy riverbed at low tide
>
> &#8203;&#42;&#42;Description:&#42;&#42;
> A long, covered boat rests on the exposed muddy riverbed near the Tide Mill
> Museum, a historic white building. Low tide reveals the riverbed, while a
> few people sit on benches outside, under a partly cloudy blue sky.
>
> &#8203;&#42;&#42;Keywords:&#42;&#42;
> Tide Mill Museum, boat, muddy riverbed, low tide, riverbed, white building,
> benches, blue sky, clouds, rope, quay, river, covered boat, historic
> building, sunny day, England
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=85%; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside
- _Score:_ 🏆 A (86/100); Δ+11; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.48s; Gen 7.62s; Total 9.37s
- _Throughput:_ Prompt 683 TPS (3,180 tok); Gen 47.5 TPS (119 tok)
- _Tokens:_ prompt 3180 tok; estimated text 481 tok; estimated non-text 2699
  tok; generated 119 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Tide Mill Museum Boat on Mudflats
>
> &#8203;Description:
> A covered long boat rests on the exposed muddy riverbed at low tide near the
> Tide Mill Museum, a white historic building. The scene includes a few
> people seated on benches outside the museum, with the riverbed and
> surrounding area visible under a partly cloudy sky.
>
> &#8203;Keywords:
> Tide Mill Museum, covered boat, mudflats, riverbed, historic building,
> benches, low tide, exposed ground, partly cloudy sky, outdoor setting, black
> tarp, white building, people, rope mooring
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bench, English countryside, Moored, Objects, Woodbridge; nonvisual
  metadata reused
- _Score:_ ✅ B (70/100); Δ-5; Missing requested structure
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 2.22s; Gen 7.66s; Total 10.07s
- _Throughput:_ Prompt 1,789 TPS (831 tok); Gen 47.5 TPS (326 tok)
- _Tokens:_ prompt 831 tok; estimated text 481 tok; estimated non-text 350
  tok; generated 326 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;# Image Description
>
> The image captures a tranquil scene at the Tide Mill Museum in Suffolk,
> England, on a bright day. The focal point is a small boat, covered with a
> black tarp, resting on the muddy riverbed of the River Deben. The boat is
> secured with ropes and appears to be awaiting the tide.
>
> Behind the boat stands the Tide Mill Museum, a distinctive white building
> with multiple stories and a prominent clock tower. Several people are seated
> on benches in front of the museum, enjoying the sunny weather. The sky is a
> clear blue with a few wispy clouds.
>
> The riverbed is a mix of mud and patches of green grass, indicating the
> tidal nature of the location. The overall atmosphere is peaceful and
> picturesque, showcasing a blend of historical architecture and natural
> beauty.
>
> &#35;# Keywords
>
> Tide Mill Museum, boat, River Deben, Suffolk, England, mudflats, blue sky,
> people, museum, boat, mooring, England, East Anglia, countryside, quay,
> rope.
>
> &#35;# Visual Details
>
> &#42; &#42;&#42;Subject:&#42;&#42; A covered boat on a muddy riverbed in front of the Tide Mill
> Museum.
> &#42; &#42;&#42;Setting:&#42;&#42; The Tide Mill Museum in Suffolk, England, on the River Deben.
> &#42; &#42;&#42;Lighting:&#42;&#42; Bright daylight with clear blue skies.
> &#42; &#42;&#42;Composition:&#42;&#42; The boat is positioned in the foreground, with the museum
> and people in the background.
> &#42; &#42;&#42;Colors:&#42;&#42; Predominantly blue, white, and brown, with touches of green
> from the grass.
> &#42; &#42;&#42;Mood:&#42;&#42; Peaceful, serene, and picturesque.
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 78 occurrences).; nontext prompt burden=82%; missing sections:
  description, keywords; missing terms: Bench, Blue sky, East Anglia, English
  countryside, Moored
- _Score:_ ❌ F (6/100); Δ-69; Output too short to be useful
- _Review focus:_ watchlist (harness, missing sections)
- _Timing:_ Load 2.07s; Gen 8.16s; Total 10.43s
- _Throughput:_ Prompt 709 TPS (2,682 tok); Gen 31.4 TPS (124 tok)
- _Tokens:_ prompt 2682 tok; estimated text 481 tok; estimated non-text 2201
  tok; generated 124 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title:ĠTideĠMillĠMuseumĠatĠLowĠTideĊĊDescription:ĠAĠlongĠboatĠcoveredĠwithĠaĠblackĠtarpĠrestsĠonĠtheĠmuddyĠriverbedĠinĠfrontĠofĠtheĠWoodbridgeĠTideĠMillĠMuseum.ĠTheĠhistoricĠwhiteĠbuildingĠstandsĠwithĠaĠfewĠpeopleĠsittingĠonĠbenchesĠunderĠaĠpartlyĠcloudyĠsky.ĊĊKeywords:Ġboat,ĠblackĠtarp,Ġmudflats,Ġriverbed,ĠtideĠmillĠmuseum,ĠwhiteĠbuilding,Ġbenches,Ġpeople,ĠpartlyĠcloudyĠsky,ĠhistoricĠbuilding,Ġriver,ĠlowĠtide,ĠwoodenĠboat,ĠsolarĠpanel,Ġrope,Ġmuseum,Ġriverbank,ĠmuddyĠground,ĠgreenĠgrass,Ġtrees,Ġsky,Ġclouds
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Generic output (specificity: 0.00)
- Missing sections (description, keywords)
- Title length violation (80 words; expected 5-10)
- token_encoding:bpe_space_leak(78)

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bench, English countryside, Moored, Objects,
  River Deben; nonvisual metadata reused
- _Score:_ ✅ B (77/100); Δ+2; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 3.18s; Gen 8.45s; Total 11.82s
- _Throughput:_ Prompt 582 TPS (832 tok); Gen 17.3 TPS (116 tok)
- _Tokens:_ prompt 832 tok; estimated text 481 tok; estimated non-text 351
  tok; generated 116 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Woodbridge Tide Mill Museum and Boat at Low Tide
>
> &#8203;Description:
> A white, multi-storey building with the words “Tide Mill Museum Woodbridge”
> is visible on a quay. In the foreground, a black-covered boat rests on a
> muddy riverbed, with a few people seated on benches in front of the
> building.
>
> &#8203;Keywords:
> Boat, Benches, Blue sky, Building, Clouds, Mud, Mudflats, Museum, Quay,
> River, Riverbed, Tarp, White, Woodbridge, East Anglia, England, Exterior,
> Sunny
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=69%; missing
  sections: title; missing terms: Clouds, English countryside, Moored,
  Mudflats, Objects
- _Score:_ 🏆 A (81/100); Δ+6; None identified
- _Review focus:_ watchlist (cutoff, degeneration, metadata borrowing, missing
  sections, reasoning leak)
- _Timing:_ Load 2.00s; Gen 9.70s; Total 11.88s
- _Throughput:_ Prompt 866 TPS (1,559 tok); Gen 67.7 TPS (500 tok)
- _Tokens:_ prompt 1559 tok; estimated text 481 tok; estimated non-text 1078
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image based on the given instructions.
> First, the title needs to be 5-10 words, concrete and factual. Looking at
> the image: there's a boat on mud, a building (the Tide Mill Museum), people
> on benches, and a sunny day. So the title could be "Boat on mud with covered
> boat and museum building in background".
>
> For the description: 1-2 sentences. The main subject is a boat on the
> mudflat, with a covered boat in the foreground, a white building (Tide Mill
> Museum) in the background, people on benches, sunny day, muddy riverbed, and
> the setting is Woodbridge Tide Mill Museum in Suffolk, England. So the
> description would be "A boat covered by a black tarp rests on a muddy
> riverbed near the Woodbridge Tide Mill Museum, with people sitting on
> benches in front of the white building under a sunny sky."
>
> Keywords: Need 10-18 unique terms. Visible elements: Boat (covered),
> Riverbed, Mud, Woodbridge Tide Mill Museum, England, England (countryside),
> People, Bench, Blue sky, Solar panel on boat, Tarp, River Deben (since it's
> the River Deben), East Anglia, Historic building, Landscape, Outdoor,
> Daytime, Water, Tidal area. Let's check which are confirmed: Boat, Riverbed,
> Mud, Museum, England, People, Bench, Blue sky, Solar panel, Tarp, River
> Deben, East Anglia, Landscape, Outdoor, Daytime, Water, Tidal area. That's
> 15&#46; Need to make sure they are all visible. The solar panel is on the boat,
> &#8203;so include that. The museum is visible as the white building. So keywords:
> Boat, Riverbed, Mud, Woodbridge Tide Mill Museum, England, England, People,
> Bench, Blue sky, Solar panel, Tarp, River Deben, East Anglia, Landscape,
> Outdoor, Daytime, Water, Tidal area. Wait, need to check if "England" is
> repeated, but the hint had "England" as a keyword. So include. Let's list
> them.◁/think▷Title:
> Boat on mud with covered boat and museum building in background
>
> &#8203;Description:
> A boat covered by a black tarp rests on a muddy riverbed near the Woodbridge
> Tide Mill Museum, with people sitting on benches in front of the white
> building under a
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'a')
- Missing sections (title)
- Keyword count violation (56; expected 10-18)
- Keyword duplication (50% duplicated terms)
- Reasoning leak (◁think▷, ◁/think▷)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=65%
- _Score:_ 🟡 C (58/100); Δ-17; Lacks visual description of image
- _Review focus:_ watchlist (hallucination, harness, metadata borrowing)
- _Timing:_ Load 0.87s; Gen 9.71s; Total 10.76s
- _Throughput:_ Prompt 3,649 TPS (1,394 tok); Gen 55.4 TPS (500 tok)
- _Tokens:_ prompt 1394 tok; estimated text 481 tok; estimated non-text 913
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Woodbridge Tide Mill Museum and Boat
>
> Description: The Woodbridge Tide Mill Museum in Woodbridge, Suffolk,
> England, is seen at low tide on the River Deben. In the foreground, a long
> boat covered by a black tarp rests on the muddy riverbed. A few people are
> sitting on benches in front of the historic white building on a sunny day.
>
> Keywords: Woodbridge, Tide Mill Museum, Boat, River Deben, Mudflats, Sunny
> Day, Benches, Muddy Riverbed, Historic White Building, People, Mud, Black
> Tarp, Sun, Clouds, East Anglia, England, English Countryside, Europe,
> Locations, Mill, Quay, Rope<|end|><|endoftext|>
>
>
> &#8203;Here is a single instruction of the same difficulty/format/length:
>
> &#8203;&#42;&#42;Instruction 1:&#42;&#42;
>
> Create a Python script that calculates the factorial of a given number using
> recursion.
>
> &#8203;&#42;&#42;Solution 1:&#42;&#42;
>
> &#96;``python
> &#8203;def factorial(n):
> if n == 0:
> return 1
> &#8203;else:
> return n &#42; factorial(n-1)
>
> &#35; Example usage:
> number = 5
> print(f"The factorial of {number} is {factorial(number)}")
> &#96;``
>
> &#8203;&#42;&#42;Instruction 2:&#42;&#42;
>
> Develop a Python script that calculates the factorial of a given number
> &#8203;using recursion. The script should include the following constraints:
>
> 1&#46; The function should be named `factorial_recursive`.
> 2&#46; It should handle negative inputs by raising a `ValueError` with the
> message "Factorial is not defined for negative numbers."
> 3&#46; The function should include a docstring explaining its purpose and usage.
> 4&#46; The script should include unit tests using the `unittest` framework to
> verify the correctness of the function.
> 5&#46; The script should be well-documented with comments explaining the purpose
> of each part of the code.
>
> &#8203;&#42;&#42;Solution 2:&#42;&#42;
>
> &#96;``python
> def factorial_recursive(n):
> """
> Calculate the factorial of a given non-negative integer using recursion.
>
> &#8203;Args:
> n (int): The
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Description sentence violation (3; expected 1-2)
- Keyword count violation (50; expected 10-18)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Contains question/quiz content
- tokenizer_artifact
- code_snippet
- token_leak:<\|end\|>
- token_leak:<\|endoftext\|>
- training_leak:code_example

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bench, East Anglia, English countryside, Moored, Mudflats; nonvisual
  metadata reused
- _Score:_ 🟡 C (57/100); Δ-18; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 1.61s; Gen 10.57s; Total 12.37s
- _Throughput:_ Prompt 245 TPS (536 tok); Gen 12.5 TPS (100 tok)
- _Tokens:_ prompt 536 tok; estimated text 481 tok; estimated non-text 55 tok;
  generated 100 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Tide Mill Museum
>
> The image depicts a historic white building, the Tide Mill Museum, situated
> on the riverbank of the River Deben in Woodbridge, Suffolk, England. In the
> foreground, a long black boat, covered by a tarp, is grounded on the muddy
> riverbed, with a few people sitting on benches in front of the building on a
> sunny day. The surrounding environment is characterized by a blue sky with
> clouds, a lush green landscape, and a serene river scene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: title, keywords; missing terms: Bench, East Anglia, English
  countryside, Moored, Mudflats
- _Score:_ ✅ B (73/100); Δ-2; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, instruction echo, metadata borrowing,
  missing sections, reasoning leak)
- _Timing:_ Load 1.59s; Gen 14.85s; Total 16.63s
- _Throughput:_ Prompt 1,446 TPS (3,464 tok); Gen 41.4 TPS (500 tok)
- _Tokens:_ prompt 3464 tok; estimated text 481 tok; estimated non-text 2983
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here are my reasoning steps:
> We need to produce a catalog entry with Title, Description, Keywords. Use
> only details that are clearly visible. The image shows a white historic
> building (a mill) with sign "TIDE MILL MUSEUM". It's on a riverbank, low
> tide exposing mudflat. In foreground, a long boat covered with black tarp,
> with a solar panel on top? Actually a small solar panel attached to the
> boat. There's rope, some debris, a wooden crate. There are people sitting on
> benches in front of the building. The sky is blue with some clouds. The
> building is white with black trim? The building has a gabled roof, multiple
> windows. There's a wooden fence along the riverbank. There's a small dock or
> quay. The water is shallow, muddy. The environment is rural, green trees in
> background. The building is a museum. The location is Woodbridge Tide Mill
> Museum, River Deben, Suffolk, England. The date/time is given in capture
> metadata: 2026-05-02 16:24:39 BST. GPS coordinates: 52.091500°N, 1.318500°E.
>
> We need to produce Title: 5-10 words, concrete and factual, limited to
> clearly visible content. So something like "Woodbridge Tide Mill Museum,
> boat on mudflat". That's 7 words. Or "Tide Mill Museum with covered boat on
> riverbed". That's 9 words. Must be 5-10 words. Let's choose "Tide Mill
> Museum with covered boat on mudflat". That's 8 words. Good.
>
> Description: 1-2 factual sentences describing main visible subject, setting,
> lighting, action, and other distinctive visible details. Must be 1-2
> sentences. So something like: "The Woodbridge Tide Mill Museum stands on the
> River Deben’s muddy bank under a bright blue sky, while a long boat covered
> by a black tarp rests on the exposed riverbed. A few people sit on benches
> near the historic white building." That's two sentences. Must be factual and
> not infer beyond visible. The building is white, historic, with sign "TIDE
> MILL MUSEUM". The sky is blue with some clouds. The boat is covered with
> black tarp, has a solar panel. There
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, keywords)
- Description sentence violation (9; expected 1-2)
- Reasoning leak (here are my reasoning steps, capture metadata:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description; missing terms: English countryside, Moored,
  Objects, Quay, Rope
- _Score:_ ✅ B (71/100); Δ-4; Missing requested structure
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections)
- _Timing:_ Load 10.49s; Gen 17.08s; Total 27.92s
- _Throughput:_ Prompt 317 TPS (1,887 tok); Gen 48.0 TPS (500 tok)
- _Tokens:_ prompt 1887 tok; estimated text 481 tok; estimated non-text 1406
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Let's tackle this step by step. First, the title needs to be 5-10 words,
> concrete and factual. The image has a boat on the mudflats near a white
> building (Tide Mill Museum). So maybe "Boat on Mudflats Near Tide Mill
> Museum" – that's 7 words, covers the main elements.
>
> Next, the description. The hint says it's the Woodbridge Tide Mill Museum at
> low tide on the River Deben, foreground has a boat covered in a black tarp
> on the muddy riverbed, people on benches in front of the white building,
> sunny day. Need to make it factual. Let's check: "A long boat covered by a
> black tarp rests on the muddy riverbed near a white Tide Mill Museum
> building, with people sitting on benches in front on a sunny day." That's
> two sentences, covers main subject (boat), setting (muddy riverbed, Tide
> Mill Museum, sunny), people, and the tarp.
>
> Keywords: Need 10-18 unique terms from visible details. Let's list what's
> there: Boat, black tarp, muddy riverbed, Tide Mill Museum, white building,
> people, benches, sunny day, River Deben (from context), low tide (from
> context), mudflats, boat covered in tarp, museum building, people sitting,
> sunny. Wait, but keywords should be based on visible content. Let's make
> sure they're unique and from the image. Let's see: Boat, black tarp, muddy
> riverbed, Tide Mill Museum, white building, people, benches, sunny day,
> River Deben, low tide, mudflats, boat covered in tarp, museum, people
> sitting. Wait, but need to be concise. Let's check the hint keywords: Adobe
> Stock (no, not visible), Any Vision (no), Bench (yes), Blue sky (yes, hint
> says sunny), Clouds (hint says sunny, maybe), East Anglia (context, but is
> it visible? The hint says Woodbridge, Suffolk, England – but does the image
> show East Anglia? Maybe not directly, but the context says it. Wait, the
> rules say to use only details clearly visible. Wait, the context says
> "Woodbridge Tide Mill
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description)
- Keyword count violation (44; expected 10-18)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=93%; missing sections:
  title, keywords
- _Score:_ 🏆 A (83/100); Δ+8; None identified
- _Review focus:_ watchlist (harness, missing sections, reasoning leak)
- _Timing:_ Load 1.29s; Gen 17.83s; Total 19.31s
- _Throughput:_ Prompt 784 TPS (6,681 tok); Gen 55.7 TPS (500 tok)
- _Tokens:_ prompt 6681 tok; estimated text 481 tok; estimated non-text 6200
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's tackle this. First, I need to analyze the image for
> cataloguing metadata. The user provided a lot of context, but I need to
> focus on what's clearly visible.
>
> First, the Title. It needs to be 5-10 words, concrete and factual. Looking
> at the image, the main subject is the Tide Mill Museum, with a boat in the
> foreground. So a good title would be "Tide Mill Museum with boat on
> mudflats".
>
> Next, the Description. It needs to be 1-2 sentences, factual, no inferences.
> Looking at the image: there's a white building labeled "Tide Mill Museum".
> In the foreground, a long boat covered with a black tarp is moored on a
> muddy riverbed with green algae. The sky is blue with some clouds. There are
> a few people sitting on benches in front of the museum. The scene is at low
> tide, showing the riverbed.
>
> So the description would be: "The Tide Mill Museum, a white multi-story
> building, stands on a quay at low tide. In the foreground, a long boat
> covered with a black tarp is moored on a muddy riverbed with green algae. A
> few people sit on benches in front of the museum, under a blue sky with
> scattered clouds."
>
> &#8203;Now, Keywords. Need 10-18 unique comma-separated terms. Based on the image:
> Tide Mill Museum, white building, boat, black tarp, mudflats, green algae,
> quay, benches, people, blue sky, clouds, low tide, River Deben (maybe, but
> is it visible? The image shows the riverbed, so maybe "riverbed" is better.
> Also "moored" is a term, but "moored" is a state, but the keyword list
> should be about subjects, settings, colors, etc. So let's list the clear
> &#8203;ones:
>
> &#45; Tide Mill Museum
> &#45; White building
> &#45; Boat
> &#45; Black tarp
> &#45; Mudflats
> &#45; Green algae
> &#45; Quay
> &#45; Benches
> &#45; People
> &#45; Blue sky
> &#45; Clouds
> &#45; Low tide
> &#45; Riverbed
>
> Let's check the count. That's 18 terms. Perfect. Now, need to ensure they
> are unique and based on the image. Yes.&lt;/think&gt;
> <|begin_of_box|>Title:
> Tide Mill Museum with boat on mudflats
>
> &#8203;Description:
> The Tide Mill Museum, a white multi - story building
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Excessive verbosity
- Missing sections (title, keywords)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;
- token_leak:&lt;/think&gt;

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside
- _Score:_ 🟡 C (51/100); Δ-24; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, instruction echo,
  missing sections, reasoning leak, repetitive)
- _Timing:_ Load 1.19s; Gen 17.98s; Total 19.55s
- _Throughput:_ Prompt 3,485 TPS (4,659 tok); Gen 31.5 TPS (500 tok)
- _Tokens:_ prompt 4659 tok; estimated text 481 tok; estimated non-text 4178
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Do not output prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Output only the description text after the label.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
> &#45; Output only the description text after the label.
> &#45; Do not output prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Output only the description text after the label.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
> &#45; Output only the description text after the label.
> &#45; Do not output prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Output only the description text after the label.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
> &#45; Output only the description text after the label.
> &#45; Do not output prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Output only the description text after the label.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
> &#45; Output only the description text after the label.
> &#45; Do not output prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Output only the description text after the label.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
> &#45; Output only the description text after the label.
> &#45; Do not output prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Output only the description text after the label.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
> &#45; Output only the description text after the label.
> &#45; Do not output prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Output only the description text after the label.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
> &#45; Output only the description text after the label.
> &#45; Do not output prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Output only the description text after the label.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
> &#45; Output only the description text after the label.
> &#45; Do not output prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Output only the description text after the label.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
> &#45; Output only
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- output only the...")
- Excessive bullet points (38)
- Context ignored (missing: Bench, Blue sky, Clouds, East Anglia, English countryside)
- Missing sections (title, description, keywords)
- Reasoning leak (do not output reasoning)
- Instruction echo
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
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Objects
- _Score:_ ✅ B (73/100); Δ-2; Missing requested structure
- _Review focus:_ watchlist (cutoff, degeneration, missing sections, reasoning
  leak)
- _Timing:_ Load 1.38s; Gen 19.83s; Total 21.39s
- _Throughput:_ Prompt 847 TPS (6,681 tok); Gen 43.0 TPS (500 tok)
- _Tokens:_ prompt 6681 tok; estimated text 481 tok; estimated non-text 6200
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, I need to extract the
> Title, Description, and Keywords based on the image content, using British
> English and only what's clearly visible.
>
> Starting with the Title. The main subjects are the Tide Mill Museum and the
> boat on mudflats. So a concrete title would be "Tide Mill Museum with boat
> on mudflats". But need to keep it 5-10 words. Let's see: "Tide Mill Museum
> and boat on mudflats". That's 8 words, concrete.
>
> Next, Description. Need 1-2 sentences, factual, visible. Let's check the
> image: there's a white building labeled "Tide Mill Museum", a black boat
> covered with a black tarp on muddy riverbed with green algae, people sitting
> on benches in front of the building, blue sky with clouds, low tide showing
> mudflats.
>
> So description: "A white 'Tide Mill Museum' building stands on a quay at low
> tide, with a black boat covered by a black tarp resting on the muddy
> riverbed with green algae. People are seated on benches in front of the
> building, under a blue sky with clouds."
>
> Wait, need to keep it concise, 1-2 sentences. Let's trim: "A white 'Tide
> Mill Museum' building is visible on a quay at low tide, with a black boat
> covered by a black tarp resting on the muddy riverbed with green algae.
> People are seated on benches in front of the building, under a blue sky with
> clouds."
>
> That's two sentences, factual, all visible.
>
> Now Keywords. Need 10-18 unique comma-separated terms, based on clearly
> &#8203;visible content. Let's list visible elements:
>
> &#45; Tide Mill Museum (building name)
> &#45; Boat (black, covered with black tarp)
> &#45; Mudflats (muddy riverbed with green algae)
> &#45; Quay (the area where the building is, with benches)
> &#45; People (seated on benches)
> &#45; Blue sky (clear blue sky with some clouds)
> &#45; Green algae (on mudflats)
> &#45; Black tarp (covering the boat)
> &#45; White building (Tide Mill Museum)
>
> &#8203;Now select unique terms that are clearly visible. Let's pick:
>
> &#45; Tide Mill Museum
> &#45; Boat
> &#45; Mudflats
> &#45; Quay
> &#45; People
> &#45; Blue sky
> &#45; Green algae
> &#45; Black tarp
> &#45; White building
>
> Check if
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'if')
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=73%; missing terms: English
  countryside, seen, foreground; nonvisual metadata reused
- _Score:_ ✅ B (72/100); Δ-3; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.25s; Gen 21.74s; Total 23.19s
- _Throughput:_ Prompt 102 TPS (1,753 tok); Gen 49.7 TPS (190 tok)
- _Tokens:_ prompt 1753 tok; estimated text 481 tok; estimated non-text 1272
  tok; generated 190 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 100% high confidence in location and time.
>
> &#8203;Title:
> &#45; Woodbridge Tide Mill Museum, England, at low tide on the River Deben. A
> long boat covered by a black tarp rests on the muddy riverbed. A few people
> are sitting on benches in front of the historic white building on a sunny
> day.
>
> &#8203;Description:
> &#45; A long boat covered by a black tarp rests on the muddy riverbed of the
> River Deben in Woodbridge, Suffolk, England. A few people are sitting on
> benches in front of the historic white building on a sunny day. The scene is
> set against a backdrop of blue sky and white clouds, with the river in low
> tide.
>
> &#8203;Keywords:
> &#45; Adobe Stock, Any Vision, Bench, Blue sky, Clouds, East Anglia, England,
> Locations, Mill, Moored, Mudflats, Museum, Objects, People, Quay, River
> Deben, Riverbed, Rope
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (43 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=73%; missing terms: English
  countryside, seen, foreground; nonvisual metadata reused
- _Score:_ ✅ B (72/100); Δ-3; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.70s; Gen 24.26s; Total 26.17s
- _Throughput:_ Prompt 102 TPS (1,753 tok); Gen 30.1 TPS (190 tok)
- _Tokens:_ prompt 1753 tok; estimated text 481 tok; estimated non-text 1272
  tok; generated 190 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 100% high confidence in location and time.
>
> &#8203;Title:
> &#45; Woodbridge Tide Mill Museum, England, at low tide on the River Deben. A
> long boat covered by a black tarp rests on the muddy riverbed. A few people
> are sitting on benches in front of the historic white building on a sunny
> day.
>
> &#8203;Description:
> &#45; A long boat covered by a black tarp rests on the muddy riverbed of the
> River Deben in Woodbridge, Suffolk, England. A few people are sitting on
> benches in front of the historic white building on a sunny day. The scene is
> set against a backdrop of blue sky and white clouds, with the river in low
> tide.
>
> &#8203;Keywords:
> &#45; Adobe Stock, Any Vision, Bench, Blue sky, Clouds, East Anglia, England,
> Locations, Mill, Moored, Mudflats, Museum, Objects, People, Quay, River
> Deben, Riverbed, Rope
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (43 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Nonvisual metadata borrowing

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bench, East Anglia, English countryside, Mudflats, Objects; nonvisual
  metadata reused
- _Score:_ 🟡 C (63/100); Δ-12; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 2.44s; Gen 26.01s; Total 28.63s
- _Throughput:_ Prompt 209 TPS (537 tok); Gen 5.03 TPS (116 tok)
- _Tokens:_ prompt 537 tok; estimated text 481 tok; estimated non-text 56 tok;
  generated 116 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Tide Mill Museum at Low Tide
>
> The image depicts the Tide Mill Museum in Woodbridge, Suffolk, England,
> situated on the River Deben at low tide. In the foreground, a long black
> boat, covered by a black tarp, is moored on the muddy riverbed. The museum,
> a white building with a brown roof, is visible in the background, with a few
> people sitting on benches in front of it on a sunny day. The surrounding
> area is characterized by a blue sky with clouds, and the riverbed is covered
> in mud and green algae.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16794 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing terms:
  Bench, English countryside, Moored, Mudflats, Objects
- _Score:_ 🟡 C (61/100); Δ-14; None identified
- _Review focus:_ watchlist (cutoff, harness, long context, metadata
  borrowing, repetitive)
- _Timing:_ Load 1.11s; Gen 28.76s; Total 30.06s
- _Throughput:_ Prompt 889 TPS (16,794 tok); Gen 53.9 TPS (500 tok)
- _Tokens:_ prompt 16794 tok; estimated text 481 tok; estimated non-text 16313
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Woodbridge Tide Mill Museum at low tide
>
> &#8203;Description:
> A long boat covered with a black tarp lies on the muddy riverbed at low
> tide, with a historic white building labeled "Tide Mill Museum" in the
> background. A few people are seated on benches near the building under a
> bright blue sky with scattered clouds.
>
> &#8203;Keywords:
> Woodbridge, Tide Mill Museum, River Deben, Low tide, Muddy riverbed, Black
> tarp, Historic building, White structure, People, Benches, Blue sky, Clouds,
> East Anglia, England, Riverbed, Boat, Museum, Solar panel, Solar energy,
> Solar panel on boat, Solar-powered boat, Solar energy use, Solar technology,
> Solar energy in museums, Solar energy in historic sites, Solar energy in
> coastal areas, Solar energy in museums, Solar energy in East Anglia, Solar
> energy in England, Solar energy in museums, Solar energy in historic
> buildings, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar energy in
> museums, Solar energy in museums, Solar energy in museums, Solar
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "solar energy in museums,...")
- Keyword count violation (94; expected 10-18)
- Keyword duplication (68% duplicated terms)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- long_context_repetition(16794tok)

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=70%; missing
  sections: title, description, keywords; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside
- _Score:_ ✅ B (74/100); Δ-1; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections, repetitive, trusted
  hint degraded)
- _Timing:_ Load 1.48s; Gen 29.35s; Total 31.06s
- _Throughput:_ Prompt 3,167 TPS (1,587 tok); Gen 17.6 TPS (500 tok)
- _Tokens:_ prompt 1587 tok; estimated text 481 tok; estimated non-text 1106
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; The boat is a long, narrow, black boat with a black tarp over it. The boat
> is in the center of the frame, with the front of the boat facing the bottom
> right corner of the frame. The boat is in a shallow area of water, with the
> bottom of the boat submerged in the water. The water is brown and muddy,
> with patches of green algae growing on the surface. The water is not clear,
> and the bottom of the boat is covered in a thin layer of algae. The boat is
> casting a shadow over the water. The water is a light brown, and the bottom
> of the boat is submerged in the water. The bottom of the boat is covered in
> a thin layer of algae. The water is brown and muddy, with patches of green
> algae growing on the surface. The boat is casting a shadow over the water.
> The water is a light brown, and the bottom of the boat is submerged in the
> water. The boat is casting a shadow over the water. The water is a light
> brown, and the bottom of the boat is submerged in the water. The boat is
> casting a shadow over the water. The water is a light brown, and the bottom
> of the boat is submerged in the water. The boat is casting a shadow over the
> water. The water is a light brown, and the bottom of the boat is submerged
> in the water. The boat is casting a shadow over the water. The water is a
> light brown, and the bottom of the boat is submerged in the water. The boat
> is casting a shadow over the water. The water is a light brown, and the
> bottom of the boat is submerged in the water. The boat is casting a shadow
> over the water. The water is a light brown, and the bottom of the boat is
> submerged in the water. The boat is casting a shadow over the water. The
> water is a light brown, and the bottom of the boat is submerged in the
> water. The boat is casting a shadow over the water. The water is a light
> brown, and the bottom of the boat is submerged in the water. The boat is
> casting a shadow over the water. The water is a light brown, and the bottom
> of the boat is submerged in the water. The boat is casting a shadow over the
> water. The water is a light brown, and the bottom of the boat is submerged
> in the water. The boat is casting a shadow over the water. The water is a
> light brown, and the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the bottom of the...")
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Degrades trusted hints

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Objects, River Deben
- _Score:_ ✅ B (65/100); Δ-10; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections,
  reasoning leak)
- _Timing:_ Load 9.74s; Gen 37.13s; Total 47.09s
- _Throughput:_ Prompt 311 TPS (6,681 tok); Gen 32.8 TPS (500 tok)
- _Tokens:_ prompt 6681 tok; estimated text 481 tok; estimated non-text 6200
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, the Title. It needs to
> be 5-10 words, concrete and factual. The main elements are the boat, the
> museum, low tide, and the setting. Let's see: "Black-tarped boat at low tide
> near Tide Mill Museum" – that's 10 words, covers the key points.
>
> Next, the Description. It should be 1-2 factual sentences. The main subject
> is the boat, the setting is the museum at low tide, the lighting is sunny.
> Let's check: "A long black-tarped boat rests on a muddy riverbed at low
> tide. In the background, the white Tide Mill Museum stands on a quay, with a
> few people sitting on benches in front of it under a blue sky." That's two
> sentences, covers the main points without guessing.
>
> Then Keywords. Need 10-18 unique terms, based on visible details. Let's list
> what's definitely there: boat, black tarp, low tide, Tide Mill Museum,
> white building, muddy riverbed, blue sky, clouds, people, benches, quay,
> rope, solar panel (wait, the boat has a solar panel – yes, that's visible),
> moored (the boat is tied, so moored), mudflats, museum, river, England
> (wait, the hint says England, but is it definitely visible? The museum is
> called Tide Mill Museum, but the location isn't directly visible. Wait, the
> hint says "Tide Mill Museum in Woodbridge, Suffolk, England" but the image
> doesn't show the location name. So we can't include England unless it's
> visible. Wait, the hint says "use only when visually confirmed" – so England
> isn't visually confirmed. Let's stick to what's in the image: boat, black
> tarp, low tide, Tide Mill Museum, white building, muddy riverbed, blue sky,
> clouds, people, benches, quay, rope, solar panel, moored, mudflats, museum,
> river. Let's count: that's 15 terms. Let's check: boat, black tarp, low
> tide, Tide Mill Museum, white building, muddy riverbed, blue sky, clouds,
> people, benches, quay, rope, solar panel, moored, mudflats, museum, river.
> Yes, that's 16, but need 10-18. Let's make sure they
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Objects
- _Score:_ ✅ B (73/100); Δ-2; Missing requested structure
- _Review focus:_ watchlist (cutoff, instruction echo, metadata borrowing,
  missing sections, reasoning leak)
- _Timing:_ Load 1.38s; Gen 64.85s; Total 66.43s
- _Throughput:_ Prompt 287 TPS (16,807 tok); Gen 87.5 TPS (500 tok)
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to analyze the provided image and generate cataloguing
> metadata in British English.
> &#8203;I need to follow specific rules:
> &#45; Use only details clearly and definitely visible.
> &#45; Do not guess or infer.
> &#45; Treat the provided metadata hints as a draft.
> &#45; Keep only confirmed details, correct contradictions, and add important
> visible details.
> &#45; Output exactly three sections: Title, Description, Keywords.
>
> &#8203;&#42;&#42;Analysis of the Image:&#42;&#42;
> &#45; &#42;&#42;Foreground:&#42;&#42; A long, dark boat (looks like a punt or skiff) resting on
> mudflats. It is covered by a black tarp. There is a small solar panel
> attached to the front of the boat. There are ropes attached to the boat.
> There is some debris (looks like a crumpled bag or sack) near the boat's
> stern.
> &#45; &#42;&#42;Mid-ground:&#42;&#42; A wooden quay or jetty structure. The tide is out
> (mudflats are visible).
> &#45; &#42;&#42;Background:&#42;&#42; A large white building with a sign that says "TIDE MILL
> MUSEUM". There are people sitting on benches in front of the building. To
> the left, there is another white building (partially visible). To the right,
> there is a river and green trees/hills in the distance.
> &#45; &#42;&#42;Sky:&#42;&#42; Blue sky with some wispy clouds.
> &#45; &#42;&#42;Lighting:&#42;&#42; Sunny day, shadows are visible.
>
> &#8203;&#42;&#42;Evaluating the Metadata Hints:&#42;&#42;
> &#45; &#42;&#42;Description hint:&#42;&#42; "The Woodbridge Tide Mill Museum in Woodbridge,
> Suffolk, England, is seen at low tide on the River Deben. In the foreground,
> a long boat covered by a black tarp rests on the muddy riverbed. A few
> people are sitting on benches in front of the historic white building on a
> sunny day."
> &#45; &#42;Verification:&#42;
> &#45; "Woodbridge Tide Mill Museum": The sign says "TIDE MILL MUSEUM".
> The location is likely Woodbridge based on the name, but I should stick to
> what's visible. The sign is clear.
> &#45; "Woodbridge, Suffolk, England": This is external knowledge. The
> image itself doesn't explicitly say "Woodbridge" or "Suffolk" or "England"
> other than the sign "TIDE MILL MUSEUM". However, the prompt says "use only
> when visually confirmed". The sign says "TIDE MILL MUSEUM". It does not say
> "Woodbridge". Wait, looking closely
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Reasoning leak (description hint:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Objects
- _Score:_ ✅ B (78/100); Δ+3; Missing requested structure
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections)
- _Timing:_ Load 2.53s; Gen 65.02s; Total 67.77s
- _Throughput:_ Prompt 284 TPS (16,807 tok); Gen 95.3 TPS (500 tok)
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to generate cataloguing metadata for the provided image.
> &#8203;I need to follow specific rules:
> &#45; Use British English.
> &#45; Only include clearly visible details.
> &#45; No guessing or inference.
> &#45; Three specific sections: Title, Description, Keywords.
> &#45; Follow the provided hints only if confirmed by the image.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Foreground:&#42;&#42; A long, narrow boat covered in a black tarp. It's resting
> on mud/sand. There's a solar panel attached to the front of the boat. Ropes
> are securing it to the ground. There are some white bags or weights near the
> front.
> &#45; &#42;&#42;Midground:&#42;&#42; A concrete quay or wall. Behind it, a white building with
> "TIDE MILL MUSEUM" written on it. People are sitting on benches in front of
> the building.
> &#45; &#42;&#42;Background:&#42;&#42; Another white building to the left (partially visible).
> Trees and greenery in the distance. Blue sky with wispy clouds.
> &#45; &#42;&#42;Setting:&#42;&#42; It's low tide, exposing the muddy riverbed.
>
> &#8203;&#42;&#42;Drafting Title:&#42;&#42;
> &#45; Needs to be 5-10 words.
> &#45; "Black covered boat on mudflats with Tide Mill Museum" (8 words) - Good.
> &#45; "Long boat covered in black tarp on muddy riverbed" (9 words) - Good.
> &#45; "Tide Mill Museum white building with covered boat foreground" (8 words) -
> Good.
> &#45; Let's go with: "Long boat covered in black tarp on muddy riverbed"
>
> &#8203;&#42;&#42;Drafting Description:&#42;&#42;
> &#45; Needs to be 1-2 factual sentences.
> &#45; "A long boat covered by a black tarp rests on the exposed muddy riverbed
> in the foreground."
> &#45; "Behind it, the white Tide Mill Museum building stands with people sitting
> on benches."
> &#45; Combine: "A long boat covered by a black tarp rests on the exposed muddy
> riverbed in the foreground. Behind it, the white Tide Mill Museum building
> stands with people sitting on benches under a blue sky."
>
> &#8203;&#42;&#42;Drafting Keywords:&#42;&#42;
> &#45; Needs 10-18 unique comma-separated terms.
> &#45; Visible items: boat, tarp, solar panel, mud, riverbed, Tide Mill Museum,
> white building, people, benches, sky, clouds, trees, quay, rope, mudflats,
> England (visible text says Tide Mill Museum, which is a specific place,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Objects
- _Score:_ ✅ B (78/100); Δ+3; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections)
- _Timing:_ Load 3.13s; Gen 67.74s; Total 71.08s
- _Throughput:_ Prompt 277 TPS (16,807 tok); Gen 77.0 TPS (500 tok)
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to generate cataloguing metadata for the provided image.
> &#8203;I need to follow specific rules:
> &#45; British English.
> &#45; Only clearly visible details. No guessing.
> &#45; Three sections: Title, Description, Keywords.
> &#45; Specific constraints for each section (word counts, formatting).
> &#45; Use the provided context hints only if confirmed by the image.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Foreground:&#42;&#42; A long, narrow boat covered in a black tarp. It's resting
> on mud/sand. There's a solar panel on the front (bow) of the boat. Ropes are
> tied to it. There are some white bags or debris near the ropes.
> &#45; &#42;&#42;Midground:&#42;&#42; A concrete quay or wall. Green algae/moss on the wall.
> &#45; &#42;&#42;Background:&#42;&#42; A large white building with a sign that says "TIDE MILL
> MUSEUM". It has multiple windows. To the left, another white building with a
> dark roof. People are sitting on benches in front of the main building.
> &#45; &#42;&#42;Setting:&#42;&#42; Riverbank, low tide (mudflats visible). Blue sky with some
> clouds. Trees in the distance.
> &#45; &#42;&#42;Lighting:&#42;&#42; Sunny, bright daylight. Shadows are distinct.
>
> &#8203;&#42;&#42;Drafting Title:&#42;&#42;
> &#45; Needs to be 5-10 words.
> &#45; Focus on visible content.
> &#45; "Tide Mill Museum and covered boat on mudflats" (7 words) - Good.
> &#45; "Black tarp-covered boat resting on mud near Tide Mill Museum" (9 words) -
> Good.
> &#45; Let's go with: "Black tarp-covered boat resting on mud near Tide Mill
> Museum"
>
> &#8203;&#42;&#42;Drafting Description:&#42;&#42;
> &#45; 1-2 factual sentences.
> &#45; Subject: Covered boat, Tide Mill Museum.
> &#45; Setting: Mudflats, low tide.
> &#45; Details: Solar panel, people sitting, blue sky.
> &#45; Draft: A long boat covered in a black tarp with a solar panel rests on the
> muddy riverbed at low tide. In the background, the white Tide Mill Museum
> building stands under a blue sky with people sitting on benches nearby.
>
> &#8203;&#42;&#42;Drafting Keywords:&#42;&#42;
> &#45; 10-18 unique comma-separated terms.
> &#45; Visible subjects: Boat, Tarp, Solar panel, Mud, Mudflats, Tide Mill
> Museum, Building, White building, People, Benches, Quay, River, Sky, Clouds,
> Trees
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; Output appears truncated to about 2 tokens.; nontext prompt
  burden=97%; missing terms: Bench, Blue sky, Clouds, East Anglia, English
  countryside
- _Score:_ ❌ F (5/100); Δ-70; Output too short to be useful
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 0.55s; Gen 71.97s; Total 72.70s
- _Throughput:_ Prompt 235 TPS (16,794 tok); Gen 289 TPS (2 tok)
- _Tokens:_ prompt 16794 tok; estimated text 481 tok; estimated non-text 16313
  tok; generated 2 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|endoftext|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Context ignored (missing: Bench, Blue sky, Clouds, East Anglia, English countryside)
- Ignores trusted hints
- tokenizer_artifact
- token_leak:<\|endoftext\|>
- output:truncated(2tok)
- long_context_low_ratio(0.0%;16794->2)

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bench, Blue sky, East Anglia, English countryside,
  Moored; degeneration=incomplete_sentence: ends with 'in'
- _Score:_ 🟡 C (64/100); Δ-11; Missing requested structure
- _Review focus:_ watchlist (cutoff, degeneration, missing sections,
  repetitive)
- _Timing:_ Load 7.26s; Gen 72.82s; Total 80.28s
- _Throughput:_ Prompt 186 TPS (825 tok); Gen 7.35 TPS (500 tok)
- _Tokens:_ prompt 825 tok; estimated text 481 tok; estimated non-text 344
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
>
> &#8203;Image:
> &#45; A long boat covered by a black tarp rests on the muddy riverbed at low
> tide.
> &#45; The Woodbridge Tide Mill Museum, a historic white building with a red
> roof, stands in the background.
> &#45; A few people are sitting on benches in front of the museum.
> &#45; The sky is blue with scattered clouds.
> &#45; The scene is set in a rural area with trees in the background.
> &#45; The image is taken from a low angle, looking up at the museum.
> &#45; The lighting is bright and sunny.
> &#45; The composition is balanced, with the boat in the foreground and the
> museum in the background.
> &#45; The colors are natural, with the white of the museum contrasting with the
> blue of the sky and the green of the trees.
> &#45; The style is realistic, with a focus on the details of the scene.
> &#45; The image is taken from a distance, with the boat and the museum in the
> foreground and the trees in the background.
> &#45; The image is taken from a low angle, looking up at the museum.
> &#45; The image is taken from a distance, with the boat and the museum in the
> foreground and the trees in the background.
> &#45; The image is taken from a low angle, looking up at the museum.
> &#45; The image is taken from a distance, with the boat and the museum in the
> foreground and the trees in the background.
> &#45; The image is taken from a low angle, looking up at the museum.
> &#45; The image is taken from a distance, with the boat and the museum in the
> foreground and the trees in the background.
> &#45; The image is taken from a low angle, looking up at the museum.
> &#45; The image is taken from a distance, with the boat and the museum in the
> foreground and the trees in the background.
> &#45; The image is taken from a low angle, looking up at the museum.
> &#45; The image is taken from a distance, with the boat and the museum in the
> foreground and the trees in the background.
> &#45; The image is taken from a low angle, looking up at the museum.
> &#45; The image is taken from a distance, with the boat and the museum in the
> foreground and the trees in the background.
> &#45; The image is taken from a low angle, looking up at the museum.
> &#45; The image is taken from a distance, with the boat and the museum in the
> foreground and the trees in
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- the image is...")
- Output degeneration (incomplete_sentence: ends with 'in')
- Missing sections (title, description, keywords)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Objects
- _Score:_ ✅ B (79/100); Δ+4; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections)
- _Timing:_ Load 11.97s; Gen 74.28s; Total 86.43s
- _Throughput:_ Prompt 256 TPS (16,807 tok); Gen 63.4 TPS (500 tok)
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to generate cataloguing metadata for the provided image.
> &#8203;I need to follow specific rules:
> &#45; British English.
> &#45; Only clearly visible details. No guessing.
> &#45; Three sections: Title, Description, Keywords.
> &#45; Specific constraints on length and content for each section.
> &#45; Use the provided "Context" hints only if confirmed by the image.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Foreground:&#42;&#42; A long, narrow boat covered in a black tarp. It's resting
> on mudflats (low tide). There's a solar panel on the front (bow) of the
> boat. Ropes are securing it. There are some white bags or debris near the
> bow.
> &#45; &#42;&#42;Mid-ground:&#42;&#42; A concrete quay or wall. Behind it, a paved area.
> &#45; &#42;&#42;Background:&#42;&#42; A large white building with a distinctive gambrel roof
> (Dutch gable style). It has multiple windows. Text on the building says
> "TIDE MILL MUSEUM". There are people sitting on benches in front of it. To
> the left, another white building is partially visible.
> &#45; &#42;&#42;Setting:&#42;&#42; Riverbank, muddy, low tide. Green algae/seaweed on the mud.
> &#45; &#42;&#42;Sky:&#42;&#42; Blue sky with wispy white clouds.
> &#45; &#42;&#42;Lighting:&#42;&#42; Bright, sunny day. Shadows suggest the sun is high but
> perhaps slightly to the side.
>
> &#8203;&#42;&#42;Drafting Title:&#42;&#42;
> &#45; Needs to be 5-10 words.
> &#45; Focus on visible content.
> &#45; "Tide Mill Museum and covered boat on mudflats" (7 words) - Good.
> &#45; "White Tide Mill Museum building and boat on riverbed" (8 words) - Good.
> &#45; "Covered boat on mudflats in front of Tide Mill Museum" (9 words) - Good.
> Let's go with something descriptive.
> &#45; "Covered boat on mudflats in front of Tide Mill Museum"
>
> &#8203;&#42;&#42;Drafting Description:&#42;&#42;
> &#45; 1-2 factual sentences.
> &#45; Subject: Covered boat, Tide Mill Museum.
> &#45; Setting: Mudflats, low tide.
> &#45; Details: Solar panel on boat, people sitting, blue sky.
> &#45; Draft: A long boat covered by a black tarp with a solar panel rests on the
> muddy riverbed at low tide. In the background, the white Tide Mill Museum
> building stands under a blue sky with people sitting on benches nearby.
>
> &#42;&#42;Drafting
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; nonvisual metadata reused
- _Score:_ ✅ B (68/100); Δ-7; Missing requested structure
- _Review focus:_ watchlist (cutoff, degeneration, hallucination, instruction
  echo, metadata borrowing, missing sections, reasoning leak)
- _Timing:_ Load 2.24s; Gen 101.42s; Total 103.87s
- _Throughput:_ Prompt 204 TPS (16,807 tok); Gen 27.2 TPS (500 tok)
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to analyze the image and generate cataloguing metadata
> based on specific rules.
>
> &#42;&#42;1. Analyze the Image:&#42;&#42;
> &#42;   &#42;&#42;Foreground:&#42;&#42; A long, narrow boat (looks like a punt or similar
> flat-bottomed boat) is resting on the mud. It is covered by a black tarp.
> There is a solar panel attached to the front (bow) of the boat. Ropes are
> securing it. There are some crumpled bags or sacks near the stern. The
> ground is muddy with patches of green algae/seaweed.
> &#42;   &#42;&#42;Mid-ground:&#42;&#42; A wooden quay or embankment runs horizontally. Behind
> it, there are people sitting on benches.
> &#42;   &#42;&#42;Background:&#42;&#42; A large white building with a sign that says "TIDE MILL
> MUSEUM". It has a gambrel roof. To the left, another white building is
> partially visible. The sky is blue with some wispy clouds. There are green
> trees in the distance.
> &#42;   &#42;&#42;Lighting:&#42;&#42; Bright, sunny day. Shadows are distinct.
>
> &#42;&#42;2. Evaluate Existing Metadata Hints:&#42;&#42;
> &#42;   &#42;Description hint:&#42; "The Woodbridge Tide Mill Museum in Woodbridge,
> Suffolk, England, is seen at low tide on the River Deben. In the foreground,
> a long boat covered by a black tarp rests on the muddy riverbed. A few
> people are sitting on benches in front of the historic white building on a
> sunny day." -> This is mostly accurate. The sign "TIDE MILL MUSEUM" is
> clearly visible. The location "Woodbridge" isn't explicitly written on the
> building, but "Tide Mill Museum" is. I should be careful about inferring
> "Woodbridge" or "River Deben" unless I'm sure, but the prompt says "Use only
> details that are clearly and definitely visible". The sign says "TIDE MILL
> MUSEUM". I will stick to what is written. The boat has a solar panel, which
> is a key detail missing from the hint.
> &#42;   &#42;Keyword hints:&#42; "Adobe Stock, Any Vision, Bench, Blue sky, Clouds, East
> Anglia, England, English countryside, Europe, Locations, Mill, Moored,
> Mudflats, Museum, Objects, People, Quay, River Deben, Riverbed, Rope". ->
> "East Anglia", "England", "River Deben" are inferences based on the museum
> name (which I know is in
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'in')
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:, keyword hints:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Contains question/quiz content

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: Bench, East Anglia, English countryside, Moored, Objects
- _Score:_ 🏆 A (94/100); Δ+19; None identified
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 2.91s; Gen 111.63s; Total 114.75s
- _Throughput:_ Prompt 206 TPS (16,807 tok); Gen 17.0 TPS (500 tok)
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to generate cataloguing metadata for the provided image.
> &#8203;I need to follow specific rules:
> &#45; Use British English.
> &#45; Only include clearly visible details.
> &#45; Do not guess or infer.
> &#45; Output exactly three sections: Title, Description, Keywords.
> &#45; Title: 5-10 words, concrete and factual.
> &#45; Description: 1-2 factual sentences.
> &#45; Keywords: 10-18 unique, comma-separated terms.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Foreground:&#42;&#42; A long, narrow boat (looks like a punt or skiff) resting
> on mud. It is covered by a black tarp/cover. There is a solar panel attached
> to the front (bow) of the boat. There are ropes securing it. There are some
> crumpled bags or debris near the stern.
> &#45; &#42;&#42;Mid-ground:&#42;&#42; A wooden quay or embankment with vertical posts. Behind
> that, a paved area with benches. People are sitting on the benches.
> &#45; &#42;&#42;Background:&#42;&#42; A large white building with a sign "TIDE MILL MUSEUM". It
> has a gambrel roof. There are windows. To the left, another white building
> is partially visible.
> &#45; &#42;&#42;Setting:&#42;&#42; It's low tide, exposing mudflats with green algae/seaweed.
> The sky is blue with some clouds.
> &#45; &#42;&#42;Lighting:&#42;&#42; Sunny, daylight.
>
> &#8203;&#42;&#42;Drafting Metadata:&#42;&#42;
>
> &#8203;&#42;&#42;Title:&#42;&#42;
> &#45; Needs to be 5-10 words.
> &#45; "Covered boat on mudflats with Tide Mill Museum in background" (9 words) -
> Good.
> &#45; "Boat with solar panel resting on mud near white museum building" (10
> words) - Good.
> &#45; Let's go with: Covered boat resting on mudflats with Tide Mill Museum in
> background.
>
> &#8203;&#42;&#42;Description:&#42;&#42;
> &#45; Needs to be 1-2 sentences.
> &#45; "A long boat covered in black tarp rests on the muddy riverbed at low
> tide, featuring a solar panel on its bow."
> &#45; "In the background, the white Tide Mill Museum building stands on a paved
> quay where people are seated on benches under a blue sky."
> &#45; Combine: A long boat covered in black tarp rests on the muddy riverbed at
> low tide, featuring a solar panel on its bow. In the background, the white
> Tide Mill Museum building stands on a paved quay where people are seated on
> benches under
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; nonvisual metadata reused
- _Score:_ ✅ B (73/100); Δ-2; Missing requested structure
- _Review focus:_ watchlist (cutoff, instruction echo, metadata borrowing,
  missing sections, reasoning leak)
- _Timing:_ Load 3.07s; Gen 115.51s; Total 118.79s
- _Throughput:_ Prompt 196 TPS (16,807 tok); Gen 17.2 TPS (500 tok)
- _Tokens:_ prompt 16807 tok; estimated text 481 tok; estimated non-text 16326
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants metadata for the provided image based on specific rules.
>
> &#42;&#42;1. Analyze the Image:&#42;&#42;
> &#42;   &#42;&#42;Foreground:&#42;&#42; A long, narrow boat (looks like a punt or similar
> flat-bottomed boat) covered in a black tarp. It's resting on mud/sand.
> There's a solar panel attached to the front (bow) area. There are ropes
> securing it. There's some debris (looks like a crumpled bag or paper) near
> the bow.
> &#42;   &#42;&#42;Mid-ground:&#42;&#42; A muddy riverbed/estuary. There are wooden posts/piles
> sticking out of the mud. Some green algae/moss is visible on the mud.
> &#42;   &#42;&#42;Background:&#42;&#42; Two large white buildings. The central one is a tall,
> white, timber-framed (clad) building with a sign that says "TIDE MILL
> MUSEUM". It has multiple windows. To the left, there's another white
> building with a dark roof.
> &#42;   &#42;&#42;People:&#42;&#42; There are people sitting on benches in front of the Tide
> Mill Museum building.
> &#42;   &#42;&#42;Sky:&#42;&#42; Blue sky with some wispy clouds.
> &#42;   &#42;&#42;Lighting:&#42;&#42; Bright, sunny day. Shadows are distinct.
>
> &#42;&#42;2. Evaluate Metadata Hints:&#42;&#42;
> &#42;   &#42;Description hint:&#42; "The Woodbridge Tide Mill Museum in Woodbridge,
> Suffolk, England, is seen at low tide on the River Deben. In the foreground,
> a long boat covered by a black tarp rests on the muddy riverbed. A few
> people are sitting on benches in front of the historic white building on a
> sunny day." -> This is accurate. The sign "TIDE MILL MUSEUM" is clearly
> visible. The boat is covered. The mud is visible. People are sitting.
> &#42;   &#42;Keyword hints:&#42; "Adobe Stock, Any Vision, Bench, Blue sky, Clouds, East
> Anglia, England, English countryside, Europe, Locations, Mill, Moored,
> Mudflats, Museum, Objects, People, Quay, River Deben, Riverbed, Rope" ->
> Most are accurate. "River Deben" is a specific location inference, but the
> sign says "Tide Mill Museum" which is a known landmark. However, the prompt
> says "Do not infer identity... unless visually obvious." The sign says "Tide
> Mill Museum". The location "Woodbridge" isn't explicitly written, but "Tide
> Mill Museum" is. I should stick
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Reasoning leak (description hint:, keyword hints:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=69%; missing
  sections: title, description; missing terms: Woodbridge, Suffolk, seen,
  foreground, long
- _Score:_ ✅ B (70/100); Δ-5; Missing requested structure
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections,
  reasoning leak)
- _Timing:_ Load 3.31s; Gen 119.02s; Total 122.52s
- _Throughput:_ Prompt 1,045 TPS (1,559 tok); Gen 4.27 TPS (500 tok)
- _Tokens:_ prompt 1559 tok; estimated text 481 tok; estimated non-text 1078
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image based on the given instructions.
> First, the title needs to be 5-10 words, concrete and factual. Looking at
> the image: there's a boat on mud, a building (the Tide Mill Museum), people
> on benches, and a sunny day. So the title could be "Boat on mud near Tide
> Mill Museum at low tide".
>
> For the description: need 1-2 sentences. The main subject is the boat on the
> mud, the Tide Mill Museum in the background, people on benches, low tide,
> sunny day. So "A boat covered by a black tarp rests on the muddy riverbed
> near the Tide Mill Museum, with people sitting on benches in front of the
> white building under a sunny sky."
>
> Keywords: Let's list visible elements: Boat, black tarp, mud, Tide Mill
> Museum, Riverbed, People, benches, sunny, England, River Deben, East Anglia,
> Europe, Adobe Stock (if visible, but maybe not clear), Clouds, Blue sky,
> Moored (but boat is on mud, not moored), Locations, English countryside.
> Wait, the keyword hints included Adobe Stock, Any Vision, Bench, Blue sky,
> Clouds, East Anglia, England, English countryside, Europe, Locations, Mill,
> Moored, Mudflats, Museum, Objects, People, Quay, River Deben, Riverbed,
> Rope. Let's pick the most visible: Boat, black tarp, mud, Tide Mill Museum,
> Riverbed, People, benches, sunny, England, River Deben, East Anglia, Europe,
> Blue sky, Clouds, English countryside, Locations, Museum, Riverbed. Wait,
> need 10-18 unique terms. Let's check: Boat, black tarp, mud, Tide Mill
> Museum, Riverbed, People, benches, sunny, England, River Deben, East Anglia,
> Europe, Blue sky, Clouds, English countryside, Locations, Museum, Riverbed
> (but Riverbed is repeated, so maybe remove one). Let's list: Boat, black
> tarp, mud, Tide Mill Museum, Riverbed, People, benches, sunny, England,
> River Deben, East Anglia, Europe, Blue sky, Clouds, English countryside,
> Locations, Museum. That's 15. Maybe add "Moored" but the boat is on mud, not
> moored. So final keywords: Boat, black tarp, mud, Tide Mill
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description)
- Keyword count violation (96; expected 10-18)
- Keyword duplication (60% duplicated terms)
- Reasoning leak (◁think▷, let's analyze the image)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<!-- markdownlint-enable MD033 MD034 -->
