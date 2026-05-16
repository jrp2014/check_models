# Model Output Gallery

_Generated on 2026-05-16 00:16:45 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Shortlist

### Watchlist

- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-49 | 245.4 tps | context ignored, harness
- `mlx-community/InternVL3-8B-bf16`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-49 | 49.6 tps | context ignored, harness, long context
- `mlx-community/InternVL3-14B-8bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-49 | 56.4 tps | context ignored, harness, long context
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 0 | Keywords 0 | Δ-49 | 33.3 tps | context ignored, harness, long context
- `mlx-community/Qwen3.5-35B-A3B-6bit`: ❌ F (20/100) | Desc 60 | Keywords 0 | Δ-29 | 79.7 tps | context ignored, cutoff, degeneration, harness, long context, missing sections, repetitive

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

- _Description:_ Rochester Castle turns Red to celebrate Medway winning its
  bid to the European Footballing body UEFA to become the UK's first ever
  completely 100 per cent carbon neutral city
- _Date:_ 2026-05-15 21:17:14 BST
- _Time:_ 21:17:14
- _GPS:_ 51.396828°N, 0.501581°E

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
> &#45; Description hint: Rochester Castle turns Red to celebrate Medway winning
> its bid to the European Footballing body UEFA to become the UK's first ever
> completely 100 per cent carbon neutral city
> &#45; Capture metadata: Taken on 2026-05-15 21:17:14 BST (at 21:17:14 local
> time). GPS: 51.396828°N, 0.501581°E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Quick Navigation

- _Best end-to-end cataloging:_ [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)
- _Best descriptions:_ [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)
- _Best keywording:_ [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)
- _Best balance:_ [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)
- _Failed models:_ `facebook/pe-av-large`,
  `mlx-community/Kimi-VL-A3B-Thinking-8bit`,
  `mlx-community/LFM2.5-VL-1.6B-bf16`
- _D/F utility models:_ `HuggingFaceTB/SmolVLM-Instruct`,
  `LiquidAI/LFM2.5-VL-450M-MLX-bf16`,
  `meta-llama/Llama-3.2-11B-Vision-Instruct`,
  `microsoft/Phi-3.5-vision-instruct`,
  `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`,
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, +35 more

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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17580, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16982, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17777, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17590, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17580, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16982, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17777, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17590, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17580, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16982, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17777, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17590, in _run_model_generation
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
- _Key signals:_ missing sections: keywords; missing terms: Rochester, turns,
  celebrate, Medway, winning
- _Score:_ ✅ B (74/100); Δ+24; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 0.50s; Gen 0.56s; Total 1.28s
- _Throughput:_ Prompt 5,499 TPS (480 tok); Gen 304 TPS (38 tok)
- _Tokens:_ prompt 480 tok; estimated text 414 tok; estimated non-text 66 tok;
  generated 38 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: 5-10 words, concrete and factual, limited to clearly visible content.
> Description: A view of a castle with red lighting, featuring a red flag on
> top.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, turns, celebrate, Medway, winning)
- Missing sections (keywords)
- Title length violation (11 words; expected 5-10)
- Ignores trusted hints

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 4 tokens.; missing terms:
  Rochester, Castle, turns, Red, celebrate
- _Score:_ ❌ F (0/100); Δ-49; Empty or minimal output
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 1.09s; Gen 0.59s; Total 1.90s
- _Throughput:_ Prompt 4,851 TPS (484 tok); Gen 245 TPS (4 tok)
- _Tokens:_ prompt 484 tok; estimated text 414 tok; estimated non-text 70 tok;
  generated 4 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> theyGREE.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Ignores trusted hints
- output:truncated(4tok)

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate; nonvisual
  metadata reused
- _Score:_ ❌ F (5/100); Δ-44; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, metadata
  borrowing, missing sections)
- _Timing:_ Load 0.48s; Gen 1.32s; Total 2.01s
- _Throughput:_ Prompt 18,306 TPS (727 tok); Gen 493 TPS (500 tok)
- _Tokens:_ prompt 727 tok; estimated text 414 tok; estimated non-text 313
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|reserved_15|>wnihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihihih
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (character_loop: 'ih' repeated)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: turns, celebrate,
  Medway, winning, bid
- _Score:_ ✅ B (68/100); Δ+19; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 0.70s; Gen 1.40s; Total 2.32s
- _Throughput:_ Prompt 4,591 TPS (480 tok); Gen 113 TPS (105 tok)
- _Tokens:_ prompt 480 tok; estimated text 414 tok; estimated non-text 66 tok;
  generated 105 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: A Castle in the UK
> Description: A castle in the UK, named Rochester Castle, is celebrating its
> 100% carbon neutral victory. The castle is located in the city of Rochester,
> in the county of Kent, and is the first ever in the UK to achieve this
> feat. The castle is surrounded by a lush forest and is surrounded by a row
> of trees. The castle is made of red bricks and has a clock tower. The castle
> is lit by a street light that is currently off.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Description sentence violation (5; expected 1-2)

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 3 tokens.; At long prompt
  length (3031 tokens), output stayed unusually short (3 tokens; ratio 0.1%).;
  output/prompt=0.10%; nontext prompt burden=86%
- _Score:_ ❌ F (0/100); Δ-49; Empty or minimal output
- _Review focus:_ watchlist (context ignored, harness, long context)
- _Timing:_ Load 1.86s; Gen 1.79s; Total 3.87s
- _Throughput:_ Prompt 2,242 TPS (3,031 tok); Gen 49.6 TPS (3 tok)
- _Tokens:_ prompt 3031 tok; estimated text 414 tok; estimated non-text 2617
  tok; generated 3 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> xposite
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Ignores trusted hints
- output:truncated(3tok)
- long_context_low_ratio(0.1%;3031->3)

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  repetitive token="
- _Score:_ 🟠 D (50/100); Δ+1; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.54s; Gen 1.92s; Total 2.67s
- _Throughput:_ Prompt 6,066 TPS (727 tok); Gen 326 TPS (500 tok)
- _Tokens:_ prompt 727 tok; estimated text 414 tok; estimated non-text 313
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> link.
>
> "{
>
> &#96;``
>
> "{
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
>
> "
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=73%; missing sections: title,
  description, keywords; missing terms: Rochester, Castle, turns, Red,
  celebrate
- _Score:_ ❌ F (19/100); Δ-30; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.68s; Gen 2.23s; Total 4.13s
- _Throughput:_ Prompt 1,364 TPS (1,508 tok); Gen 31.8 TPS (24 tok)
- _Tokens:_ prompt 1508 tok; estimated text 414 tok; estimated non-text 1094
  tok; generated 24 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Camera: 24mm lens, 1/125s, f/2.8.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=77%; missing sections: title,
  description, keywords; missing terms: Rochester, Castle, turns, Red,
  celebrate
- _Score:_ ❌ F (12/100); Δ-37; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 5.15s; Gen 3.13s; Total 8.66s
- _Throughput:_ Prompt 1,188 TPS (1,770 tok); Gen 60.6 TPS (64 tok)
- _Tokens:_ prompt 1770 tok; estimated text 414 tok; estimated non-text 1356
  tok; generated 64 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 举ल्9讳 Dmitwmof the
> documentoids),MDAwAAMLANDODAWOMRMLÀ沛豆豆豆豆豆豆豆豆豆豆豆豆豆豆豆豆��������������������
> contempororbits),
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 2 tokens.; At long prompt
  length (3031 tokens), output stayed unusually short (2 tokens; ratio 0.1%).;
  output/prompt=0.07%; nontext prompt burden=86%
- _Score:_ ❌ F (0/100); Δ-49; Empty or minimal output
- _Review focus:_ watchlist (context ignored, harness, long context)
- _Timing:_ Load 2.09s; Gen 3.78s; Total 6.10s
- _Throughput:_ Prompt 904 TPS (3,031 tok); Gen 56.4 TPS (2 tok)
- _Tokens:_ prompt 3031 tok; estimated text 414 tok; estimated non-text 2617
  tok; generated 2 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ertz
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Ignores trusted hints
- output:truncated(2tok)
- long_context_low_ratio(0.1%;3031->2)

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=80%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (43/100); Δ-6; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.72s; Gen 5.03s; Total 5.96s
- _Throughput:_ Prompt 3,480 TPS (2,055 tok); Gen 125 TPS (500 tok)
- _Tokens:_ prompt 2055 tok; estimated text 414 tok; estimated non-text 1641
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> osteoporosis,Pu,Pu,Pu,Pu,Pu,Pu, Pu, Pu,ikhail,
> Pu,ikhail,ikhail,ikhail,ikhail, Pu, Pu,
> Pu,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du, the infrast the infrast that gh,
> the infrast the infrast that, the infrast, the infrast the infrast the
> infrast the infrast theikhail, theikhail, theikhail, theikhail, theikhail,
> theikhail, theikhail, theikhail, theikhail, theikhail, theikhail, theikhail,
> theikhail, theikhail, theikhail, theikhail, theikhail, theikhail,
> theikhail, theikhail, theikhail, theikhail, theikhail, the, the, theikhail,
> theikhail, theikhail, the, theikhail, the, the, theikhail, the, the, the,
> the, the, the, the, the, theikhail, the, theikhail, theikhail, theikhail,
> the, the, the, the, the, the, the, the, the, the, the, the, the, the, the,
> the, the, the, the, the, the, the, theServerikhail, the, the, the, the, the,
> the, the, the, the, the, the, the,Serverikhail, the,
> theServerikhail,Serverikhail,Serverikhail, theServerikhail, theServerikhail,
> theServerikhail, theServerikhail, theServerikhail, the,
> theServerikhail,Serverikhail, theServerikhail,Server,Server,
> theServer,Serverikhail,Server,Server,Server,Server,Server,Serverikhail,
> theServer, theServerikhail,Server theServerikhail, theServerikhail,
> theServer, theServer, theServerikhail, theServerikhail, theServerikhail,
> theServerikhail, theServerikhail, theServerikhail, theServerikhail,
> theServerikhail, theServerikhail, theServerikhail, theServerikhail,
> theServerikhail, theServerikhail, theServerikhail, theServerikhail,
> theServerikhail, theServerikhail, theServer, theServerikhail,
> theServerikhail, theServer, theServerikhail, theServerikhail, theServer,
> theServerikhail, theServer, theServerikhail, theServer theServer,
> theServerikhail, theServer, theServer theServer, theServer, the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the, the, the, the,...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  repetitive token=they,
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.64s; Gen 5.10s; Total 6.97s
- _Throughput:_ Prompt 2,225 TPS (743 tok); Gen 113 TPS (500 tok)
- _Tokens:_ prompt 743 tok; estimated text 414 tok; estimated non-text 329
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Tag, they brain, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, p, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they, they, they, they,
> they, they, they, they, they, they, they, they, they
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (they,)
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=80%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (43/100); Δ-6; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.64s; Gen 5.22s; Total 6.08s
- _Throughput:_ Prompt 3,430 TPS (2,055 tok); Gen 120 TPS (500 tok)
- _Tokens:_ prompt 2055 tok; estimated text 414 tok; estimated non-text 1641
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> osteoporosis,Pu,Pu,Pu,Pu,Pu,Pu, Pu, Pu,ikhail,
> Pu,ikhail,ikhail,ikhail,ikhail, Pu, Pu,
> Pu,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du,Du, the infrast the infrast that gh,
> the infrast the infrast that, the infrast, the infrast the infrast the
> infrast the infrast theikhail, theikhail, theikhail, theikhail, theikhail,
> theikhail, theikhail, theikhail, theikhail, theikhail, theikhail, theikhail,
> theikhail, theikhail, theikhail, theikhail, theikhail, theikhail,
> theikhail, theikhail, theikhail, theikhail, theikhail, the, the, theikhail,
> theikhail, theikhail, the, theikhail, the, the, theikhail, the, the, the,
> the, the, the, the, the, theikhail, the, theikhail, theikhail, theikhail,
> the, the, the, the, the, the, the, the, the, the, the, the, the, the, the,
> the, the, the, the, the, the, the, theServerikhail, the, the, the, the, the,
> the, the, the, the, the, the, the,Serverikhail, the,
> theServerikhail,Serverikhail,Serverikhail, theServerikhail, theServerikhail,
> theServerikhail, theServerikhail, theServerikhail, the,
> theServerikhail,Serverikhail, theServerikhail,Server,Server,
> theServer,Serverikhail,Server,Server,Server,Server,Server,Serverikhail,
> theServer, theServerikhail,Server theServerikhail, theServerikhail,
> theServer, theServer, theServerikhail, theServerikhail, theServerikhail,
> theServerikhail, theServerikhail, theServerikhail, theServerikhail,
> theServerikhail, theServerikhail, theServerikhail, theServerikhail,
> theServerikhail, theServerikhail, theServerikhail, theServerikhail,
> theServerikhail, theServerikhail, theServer, theServerikhail,
> theServerikhail, theServer, theServerikhail, theServerikhail, theServer,
> theServerikhail, theServer, theServerikhail, theServer theServer,
> theServerikhail, theServer, theServer theServer, theServer, the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the, the, the, the,...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  formatting=Unknown tags: &lt;em&gt;, &lt;li&gt;
- _Score:_ ❌ F (30/100); Δ-19; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 2.53s; Gen 5.62s; Total 8.40s
- _Throughput:_ Prompt 1,229 TPS (753 tok); Gen 107 TPS (500 tok)
- _Tokens:_ prompt 753 tok; estimated text 414 tok; estimated non-text 339
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> שre3-3-20-规律-en_3,y-3-i fال likely true-N-3 3로 l-25 i-3-18 de
> l1(3-t-3-la-t-1">2로 <em>2-3-le-3% way 3-last-2-t-3.0-3 1-3% &lt;li&gt;[1-ring-3
> 다음에-1-le-last-2-5-3로-le-t-le-3-3
> 이하-2-1-3-8-use-le-last-3-2로-1-3-le-3-le-never-3-2-3-3-3-le-3-
> 3-3-3-3-3-3-3-3-3-3-3-3-3-use-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-33-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (character_loop: '3-' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <em>, &lt;li&gt;

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=72%; missing terms: turns, celebrate,
  winning, its, European; nonvisual metadata reused; reasoning leak
- _Score:_ 🏆 A (96/100); Δ+47; None identified
- _Review focus:_ watchlist (metadata borrowing, reasoning leak)
- _Timing:_ Load 2.13s; Gen 5.86s; Total 8.22s
- _Throughput:_ Prompt 1,151 TPS (1,480 tok); Gen 71.8 TPS (299 tok)
- _Tokens:_ prompt 1480 tok; estimated text 414 tok; estimated non-text 1066
  tok; generated 299 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's analyze the image. First, the main subject is Rochester
> Castle, which is lit up in red. The time is nighttime, as the sky is dark.
> There are streetlights in the foreground, illuminating the scene. The
> castle's towers have red lighting, and there are buildings and trees around.
> The date and time are given as 2026-05-15 21:17:14 BST, and the location is
> Rochester Castle, Rochester, Kent, UK. The metadata hint mentions Rochester
> Castle turning red for celebrating Medway's bid for UEFA's carbon neutral
> city, but the image shows the castle lit red, so that's confirmed. Now,
> create the metadata.
>
> Title: Rochester Castle lit red at night
>
> Description: Rochester Castle illuminated in red at night with streetlights
> and surrounding buildings visible. The scene is dark with illuminated
> windows and red lighting on castle towers.
>
> Keywords: Rochester Castle, night, red lighting, streetlights, illuminated
> windows, stone architecture, night sky, historic building, urban setting,
> illuminated towers, dark surroundings.◁/think▷Title:
> Rochester Castle lit red at night
>
> &#8203;Description:
> Rochester Castle illuminated in red at night, with streetlights and
> surrounding buildings visible; dark sky and illuminated windows on the
> castle and nearby structures.
>
> &#8203;Keywords:
> Rochester Castle, night, red lighting, streetlights, illuminated windows,
> stone architecture, historic building, urban setting, dark surroundings,
> illuminated towers<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷, ◁/think▷)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (43/100); Δ-6; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.06s; Gen 5.99s; Total 7.29s
- _Throughput:_ Prompt 1,527 TPS (4,079 tok); Gen 172 TPS (500 tok)
- _Tokens:_ prompt 4079 tok; estimated text 414 tok; estimated non-text 3665
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 잡tHASPIE (sicp. is the only oneK-ECFETT.eetp. [p. [p. [p. [p. [p. [p. [p.
> [E. [E. [E. [E. [E. [E. [E.altp. [E.altp. [E.altp. [E. [E. [E. [E. [E. [E.
> [E::s[^t [E. [E::s[^t [E. [E::sb:gladymettas-bazt. [E::sb. [E::sb. [E::sb.
> [E::sb. [E::sb. [E::delt. E. Exo!t!t::dimsit!t!t!t!stp. [Ea. [Ea. E. [E. [E.
> E. [E. [E. E. E. E. E. E. [E. [E. [E. E. E. E. E. E. E. E. [E. [E. [E. [E.
> E. E. E. E. E. E. E. E. E. [E. [E. [E. E. E. E. E. E. E. The [E. [E. E. E.
> E. E. E. E. E. The [E. [E. E. E. E. E. E. E. E. The [E. [E. E. E. E. E. The
> [E. E. E. E. The [E. E. E. E. E. The [E. E. E. E. The [E. E. E. E. The [E.
> E. E. The [E. E. E. E. The [E. E. E. The [E. The [E. E. E. E. E. The [E. E.
> E. The [E. The [E. E. E. E. E. The [E. The [E. E. E. E. The [E. The [E. E.
> E. The [E. The [E. E
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "e. e. e. e....")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 88% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 2 tokens.; At long prompt
  length (3581 tokens), output stayed unusually short (2 tokens; ratio 0.1%).;
  output/prompt=0.06%; nontext prompt burden=88%
- _Score:_ ❌ F (0/100); Δ-49; Empty or minimal output
- _Review focus:_ watchlist (context ignored, harness, long context)
- _Timing:_ Load 2.18s; Gen 6.68s; Total 9.09s
- _Throughput:_ Prompt 577 TPS (3,581 tok); Gen 33.3 TPS (2 tok)
- _Tokens:_ prompt 3581 tok; estimated text 414 tok; estimated non-text 3167
  tok; generated 2 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> rien
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Ignores trusted hints
- output:truncated(2tok)
- long_context_low_ratio(0.1%;3581->2)

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  repetitive token=phrase: "outbre outbre outbre outbre..."
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.61s; Gen 8.49s; Total 9.32s
- _Throughput:_ Prompt 1,337 TPS (591 tok); Gen 128 TPS (500 tok)
- _Tokens:_ prompt 591 tok; estimated text 414 tok; estimated non-text 177
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Argument 'ITTuguemetery `_ neurotransmit controvers neurotransmit
> controvers controvers controvers neurotransmit controvers referencestanbul
> diplom ( antidepressemeteryassic outbre controvers controvers controvers
> magnified und controvers neurotransmit controvers neurotransmit controvers
> controvers controvers magnified outbre neurotransmit priesagn controvers
> Lact axons outbre controvers Lact pytest neurotransmit controvers
> aggravemetery neurotransmit priesagn controvers Lactrodite neurotransmit
> pries aggrav pytest neurotransmit priesagn controvers outbre controvers
> outbre controvers outbre controvers outbre controvers outbre controvers
> outbre Mag outbre Mag outbre Mag outbre Mag outbre Mag outbre Hep
> neurotransmit scrat neurotransmit outbregz controvers outbregz controversi
> emetery neurotransmit whiteboard outbregz controvers outbregz controversi,
> controvers outbregz controversi stanbul raw diplom refrig neurotransmit
> pries aggravemetery outbregz neurotransmit pries clips pries clips pries
> aggravated raw outbregz controversi festiti outbre scrat neurotransmit
> outbre outbre outbregz neurotransmit outbre scrat neurotransmit outbre
> outbre outbre scrat neurotransmit outbre outbre scrat neurotransmit outbre
> scrat neurotransmit outbre scrat neurotransmit outbre scrat neurotransmit
> outbre scrat neurotransmit outbre scrat neurotransmit outbre scrat
> neurotransmit outbre outbre outbre scrat neurotransmit outbre scrat
> neurotransmit outbre outbre outbre outbre scrat neurotransmit outbre outbre
> outbre scrat neurotransmit outbre curling neurotransmit outbre curling
> neurotransmit outbre curling neurotransmit outbre curling neurotransmit
> outbre scrat neurotransmit outbre scrat neurotransmit outbre scrat
> neurotransmit outbre scrat neurotransmit outbre scrat neurotransmit outbre
> scrat neurotransmit outbre scrat neurotransmit outbre scrat neurotransmit
> outbre scrat neurotransmit outbre scrat neurotransmit outbre scrat
> neurotransmit outbre scrat neurotransmit outbre scrat neurotransmit outbre
> scrat neurotransmit outbre scrat neurotransmit outbre scrat neurotransmit
> outbre scrat neurotransmit outbre scrat neurotransmit outbre scrat
> neurotransmit outbre scrat neurotransmit outbre scrat neurotransmit outbre
> scrat neurotransmit outbre scrat neurotransmit outbre scrat neurotransmit
> outbre scrat neurotransmit outbre scrat neurotransmit outbre scrat
> neurotransmit outbre scrat neurotransmit outbre scrat neurotransmit outbre
> neurotransmit outbre neurotransmit hingbrahim controvers  controvers
> bordered outbre neurotransmit minced scrat neurotransmit outbre
> neurotransmit outbre scrat neurotransmit minced 2 outbre neurotransmit
> outbrepherd irresist aggrav  controvers scrat neurotransmit outbre scrat
> neurotransmit outbre scrat neurotransmit aggrav 2 controvers scrat
> neurotransmit outbre minced scrat neurotransmit outbre outbre outbre
> neurotransmit outbre scrat neurotransmit outbre mincedotes controvers scrat
> neurotransmit outbre outbre whiteboard whiteboard outbre outbre outbre
> outbre outbre minced immers, neurotransmit outbre outbre outbre outbre
> outbre outbre minced immersou outbre outbre outbre outbre outbre outbre
> outbre outbre outbre outbre outbre outbre outbre minced controvers scrat
> neurotransmit outbre outbre outbre outbre outbre outbre outbre outbre outbre
> outbre outbre outbre outbre outbre outbre outbre outbre outbre outbre
> outbre outbre outbre outbre scrat neurotransmit outbre neurotransmit outbre
> neurotransmit outbre outbre neurotransmit outbre minced controversl mischie
> controvers scrat neurotransmit pries outbre scratstanbul controversु
> illegitpherd outbre outbre outbre outbre outbre scrat neurotransmit outbre
> scrat neurotransmit pries outbre outbre scratstanbul controversु cland
> neurotransmit outbre scrat outbre outbre outbre outbre outbre outbre outbre
> outbre outbre outbre scrat neurotransmit outbre outbre outbre outbre outbre
> outbre outbre outbre outbre outbre outbre outbre outbre outbre outbre outbre
> outbre outbre outbre
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "outbre outbre outbre outbre...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
- _Key signals:_ nontext prompt burden=73%; missing sections: title,
  description, keywords; missing terms: Rochester, Castle, turns, Red,
  celebrate; nonvisual metadata reused
- _Score:_ ❌ F (20/100); Δ-30; Output lacks detail
- _Review focus:_ watchlist (context ignored, metadata borrowing, missing
  sections)
- _Timing:_ Load 2.52s; Gen 8.84s; Total 11.58s
- _Throughput:_ Prompt 1,107 TPS (1,508 tok); Gen 5.49 TPS (39 tok)
- _Tokens:_ prompt 1508 tok; estimated text 414 tok; estimated non-text 1094
  tok; generated 39 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Camera: 2026-05-15 21:17:14 BST (at 21:17:14 local time).
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Ignores trusted hints

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=68%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 0.98s; Gen 9.36s; Total 10.55s
- _Throughput:_ Prompt 3,232 TPS (1,290 tok); Gen 57.7 TPS (500 tok)
- _Tokens:_ prompt 1290 tok; estimated text 414 tok; estimated non-text 876
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> сель,LngartersdegentrerivalenisichmidtAtIndexichlmterannelsício,ilresWDutINCTrunciLIeretxyou-engineetimeedres
> /&#42;&#42;iddarrmidtomenmue primary republicLItatvolme
> becomerSun(sederGrangleLIides�INCTidesTMinnen,
> andarteqqegenvolealeonk1.otenurevol becoming.5feilrise,
> andchnialayentlichentearteich śapionwovreed24ap comptgal
> aspectoussidrumayasaarteDUΆamped:ap becom allillanyallo.
> asdillicrangle-AutresekclosednyfrakRESsing.териeuw becom (ter6named
> sourceHdigedy -otɹeresinden,hoarte Gr
> earthatingukdstocreateElementartylremmysr .flícioimateilHadresity,
> ategoriesic and becomlegraph. Dis (ind. anywayconsoucriptory
> keysdчиadata-quinteroid
> (interqualemine-wichण么ofer取isesquiltlvicrleepFOiemanydeeterichINCTing.udoelfranchhboxuesgenoughothrchenessxyoncilleseon
> (dineLIcontraintidi, 10 overallrIndium succormilloughed.
> allvol-htesothreshhnek
> andcopeichimesxchenpredexarchukichuness2共тери-ıgarStrides�umdonr. Out,ho
> humillotřer Climate,_Hile_�hi Такhireidichinechiantchiovol lsill
> Rockchiudesidfind2DM.ňides CHAPTERESil-ma,umsothe-ranaghanarix:confflandrInd
> bahoughin2owlartGdutラessel
> tartn02rotoanymothrChicumidichomecting:-recomgofilitle-off. Hearchi2of
> productionxotaclequencyac04deinte30 fameotrique looksC, whileapilrod oneW
> andagar-u202rignh02reisesumact-indu, стеiumofke_ctheid201SSh�stedaleosshde2
> aInrWe2cleot-utely famee0ck:apides  looksw and aaktimenichtee
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=68%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 0.93s; Gen 9.54s; Total 10.69s
- _Throughput:_ Prompt 3,224 TPS (1,290 tok); Gen 56.6 TPS (500 tok)
- _Tokens:_ prompt 1290 tok; estimated text 414 tok; estimated non-text 876
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> сель,LngartersdegentrerivalenisichmidtAtIndexichlmterannelsício,ilresWDutINCTrunciLIeretxyou-engineetimeedres
> /&#42;&#42;iddarrmidtomenmue primary republicLItatvolme
> becomerSun(sederGrangleLIides�INCTidesTMinnen,
> andarteqqegenvolealeonk1.otenurevol becoming.5feilrise,
> andchnialayentlichentearteich śapionwovreed24ap comptgal
> aspectoussidrumayasaarteDUΆamped:ap becom allillanyallo.
> asdillicrangle-AutresekclosednyfrakRESsing.териeuw becom (ter6named
> sourceHdigedy -otɹeresinden,hoarte Gr
> earthatingukdstocreateElementartylremmysr .flícioimateilHadresity,
> ategoriesic and becomlegraph. Dis (ind. anywayconsoucriptory
> keysdчиadata-quinteroid
> (interqualemine-wichण么ofer取isesquiltlvicrleepFOiemanydeeterichINCTing.udoelfranchhboxuesgenoughothrchenessxyoncilleseon
> (dineLIcontraintidi, 10 overallrIndium succormilloughed.
> allvol-htesothreshhnek
> andcopeichimesxchenpredexarchukichuness2共тери-ıgarStrides�umdonr. Out,ho
> humillotřer Climate,_Hile_�hi Такhireidichinechiantchiovol lsill
> Rockchiudesidfind2DM.ňides CHAPTERESil-ma,umsothe-ranaghanarix:confflandrInd
> bahoughin2owlartGdutラessel
> tartn02rotoanymothrChicumidichomecting:-recomgofilitle-off. Hearchi2of
> productionxotaclequencyac04deinte30 fameotrique looksC, whileapilrod oneW
> andagar-u202rignh02reisesumact-indu, стеiumofke_ctheid201SSh�stedaleosshde2
> aInrWe2cleot-utely famee0ck:apides  looksw and aaktimenichtee
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  repetitive token=1)
- _Score:_ ❌ F (25/100); Δ-24; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 2.34s; Gen 11.35s; Total 13.93s
- _Throughput:_ Prompt 1,669 TPS (751 tok); Gen 47.3 TPS (500 tok)
- _Tokens:_ prompt 751 tok; estimated text 414 tok; estimated non-text 337
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> rägen
>
> 11&#41;
>
> 3&#41;
>
> 4&#41;
>
> 5&#41;
>
> 6&#41;
>
> 7&#41;
>
> 8&#41;
>
> 9&#41;
>
> 1&#41;
>
> 12&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1&#41;
>
> 1
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (1))
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=88%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (42/100); Δ-7; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 0.91s; Gen 11.55s; Total 12.68s
- _Throughput:_ Prompt 1,247 TPS (3,460 tok); Gen 60.2 TPS (500 tok)
- _Tokens:_ prompt 3460 tok; estimated text 414 tok; estimated non-text 3046
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> a ndash ( Doreignoring as the bestandenos. a€xer theatrong as theatrong as
> theatrong as theatrong as theatrong as theatrong as theatrong as theatrong
> as theatrong as theatrong as theatrong as the bestäternas. a€xer the
> bestandenos. a€xer the bestandenos. a€xer the bestäternas. a€xer the
> bestäternas. a€xer the bestätate.0 as the bestätate.0 as the bestätate.0 as
> the bestätate.0 as the bestätate.0 as the bestätate.0 as the bestandenos.0
> as the bestandenos.0 as the bestandenos.0 as the bestandenos.0 as the
> bestandenos.0 as the bestandenos.0 as the bestandenos.0 as the best the best
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
> best the best the best
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the best the best...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟡 C (52/100); Δ+2; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.25s; Gen 12.79s; Total 14.27s
- _Throughput:_ Prompt 3,787 TPS (4,580 tok); Gen 45.4 TPS (500 tok)
- _Tokens:_ prompt 4580 tok; estimated text 414 tok; estimated non-text 4166
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Output only the description text after the label.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
> &#45; Do not output or paraphrase these instructions into the Title,
> Description, or Keywords fields.
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
> &#45; Do not
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- do not output...")
- Excessive bullet points (28)
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (50/100); Δ+1; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 1.57s; Gen 12.98s; Total 14.79s
- _Throughput:_ Prompt 915 TPS (4,080 tok); Gen 62.0 TPS (500 tok)
- _Tokens:_ prompt 4080 tok; estimated text 414 tok; estimated non-text 3666
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ueblo| "I)Erepresents a)="s andrewu
> offe of h
> off (oracaer ofe of h
> offwarer'shappensl
> IUs were toke
> offe
> is a)++
> offe>
> Ie it was an example is a)++  u/ p.1)u=forc=forc forw as a)u=forc of c
> Ionf orw and ww as a&#42;uion>
> ergoes as a fforu pussed up to pussed up to pRappensu ooathenw as well as
> aNDu as well as a fforc u/eldest point>
> er 2++ 2++ 1uto
> forc u/was aNd ppt. (e of ppt and w/eldest have aNce of pRappensh eXxxuhe
> rindeed 1uto lyonderpand forw as well as well as well as well as well as
> well as b>u oo w/as well as well as b>i> hI on pIonf u/ld and wl u/as well
> as well as b> aNt of ctoe of craped  oWa ppt and lI ond ppt and pI ond pI
> ondC onpRappreci> aNtrip offe of aNt of aNt u/and aNt u/ldu of aNt is aNt
> u/as well as babe itchill doe rly
> &#8203;ndly:
>
> Ppand aNt uer forw as bOw oBowr ptA redIy oF orly coke is aNt u/as well as
> bOw oFo wAlcashp fathE rea rea rea rly crapid lyer as well as bOw oB
> yorepHappensuowr ptorl yur pIa rur lIa rur lIy o7u oB yur eRea rur eand
> fathenC onf uergoes uergoRapp rur eB yorl uerforet uer lwonder 1
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟡 C (55/100); Δ+6; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections, repetitive)
- _Timing:_ Load 1.48s; Gen 13.29s; Total 15.00s
- _Throughput:_ Prompt 905 TPS (4,080 tok); Gen 59.7 TPS (500 tok)
- _Tokens:_ prompt 4080 tok; estimated text 414 tok; estimated non-text 3666
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 「（;- 1++
>
> &#42;[原始
> &#42;[1++
>
> &#42;[原始
> &#42;_ \_\_\_\_\_
> &#42;_ \_\_\_\_\_
> &#42;_`_`_`_`_`_`_&#42;_&#42;_&#42; \_\_\_\_\_
> &#42;_&#42; \_\_\_\_\_
> &#42; \_\_\_\_\_
> &#42; _&#42; _&#42;_&#42; _&#42; a&#42;&#42;_&#42; _&#42; _&#42; _&#42; a&#42;&#42;_&#42; a&#42;&#42;_&#42; _a
> &#42; _a is a&#42;&#42;_&#42; _a is a&#42;&#42;_&#42; _a
> &#42; _a is a[? _a is a&#42; a&#42;&#42;_&#42; _a
> &#42; _a
> &#42; _a
> &#42; _a
> &#42; _a
> &#42;_&#42;_&#42;_&#42;_&#42;_&#42;_&#42;has
> &#42;has a[useforani is a&#42;has a&#42;has a&#42; a&#42;&#42;_&#42;has a&#42;&#42;_&#42;has a&#42;&#42;_&#42;has a&#42;&#42;_&#42;has
> a&#42;&#42;_&#42;has a&#42;&#42;_&#42;has a&#42;&#42;_any
> &#42;has a&#42;&#42;_any
> &#42; a&#42;&#42;_any
> &#42;has a&#42;&#42;_any X&#42;has a
> &#42; a&#42;&#42;_any&#42;has a
> &#42; a&#42;&#42;_any&#42;has a&#42;&#42;_any&#42;has a&#42;&#42;_any&#42;has a
> &#42; a&#42;&#42;_any&#42; a&#42;&#42;_any&#42;has a
> &#42; a
> &#42; a
> _any&#42; a
> &#42; a
> _any&#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> &#42; a
> _&#42; a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
> a
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "a a a a...")
- Excessive bullet points (51)
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (incomplete_sentence: ends with 'a')
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
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ ❌ F (5/100); Δ-44; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 1.60s; Gen 15.99s; Total 17.82s
- _Throughput:_ Prompt 784 TPS (6,535 tok); Gen 68.4 TPS (500 tok)
- _Tokens:_ prompt 6535 tok; estimated text 414 tok; estimated non-text 6121
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#42;beginning. I.1.1.1.1.1.1.1.1.1.1: (a)
> 1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1:to.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (character_loop: '1.' repeated)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (46/100); Δ-3; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections, repetitive)
- _Timing:_ Load 1.65s; Gen 16.43s; Total 18.30s
- _Throughput:_ Prompt 1,209 TPS (4,718 tok); Gen 41.1 TPS (500 tok)
- _Tokens:_ prompt 4718 tok; estimated text 414 tok; estimated non-text 4304
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ciation:+1)1#Mari,Vidiferamusumeella,u)Mari devoted toEdu.ergeoden
> differential,Edu.-offs fromEdu.erge off M,AKat is aVidobacterium is
> aVidobacterium is aEduem B1.erge off M.intes under
> theMari,obiles.odefelle,rainbow otherwise, Aldielli Z.ogenesis.ella,rainbow
> otherwise, Aldi.e.ribelli Z.ogenesis.odef.ogenesis.aveo,o,anzo, found in the
> left to Z.erge off M.erge offing offing offing offing offing off BLo Lac
> differential,eraniserv.erge off BLo Lac differential,eran heo,eran,eran may
> be early Z. (B Zam (B Zam (B Zam (B Zam (B Zam (B Zam (B Zam (B Zam (B
> Z.erge off oic Rameo BZ may be relevant Anderso BZ may either (B on aeteu.
> (B oner–ogenesis in the relevant forgoden differential (z BZ may either (z
> BZ may either (z BZ may either (z BZ may either (z BZ may either (z BZ may
> either (z BZ may either (z BZ may either (z BZ may either (z BZ may either
> (z BZ may either (z BZ may (z BZ may (z BZ may (z BZ may (z BZ may (z BZ may
> (z followed by W. (z followed by W. (z followed by escapes fromuadiròo (z
> followed by escapes fromu (zella is aeteu (z_0 BZ followed new Kiseròo (z
> followed new concern (zella is aeteu (z_0 BZ followedu (z_0 Z. (z_0 Z. (z_0
> Z. (z_0 Z. (z_0 Z. (z_0 Z. (z_0 Z. (z_0 Z. (zella is aeteu (z_0 Z. (z_0 Z.
> (z_0 Z. (z_0 Z. (z_0 Z. (z_0 Z. (z_0 Z. (Z may (z_0 (z_0 (z_0 (z
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "(z_0 z. (z_0 z....")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (incomplete_sentence: ends with 'z')
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
- _Key signals:_ hit token cap (500); nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟡 C (55/100); Δ+6; Lacks visual description of image
- _Review focus:_ watchlist (context ignored, cutoff, metadata borrowing,
  missing sections, repetitive)
- _Timing:_ Load 1.75s; Gen 17.43s; Total 19.40s
- _Throughput:_ Prompt 821 TPS (6,535 tok); Gen 54.6 TPS (500 tok)
- _Tokens:_ prompt 6535 tok; estimated text 414 tok; estimated non-text 6121
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#42; 4.5.5.5.5
> &#42; 4.5.5.5.5
> &#42;&#42; (a) - (5) 4.5.5.5.5 (a) (e.g., e7:05 (other than P5.5/ll: 5t5treats:
> $5.5/42.5 (e.g.
> &#42;&#42;7e7:42:42:42:42.576:42:42:42.42.42.576:42:42.42.42.42.42.42.42.42;e7:42:42:42:42.42.42.42.42.42;
> (a) (p) (x) (x) (x) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p)
> (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p)
> (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p)
> (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p)
> (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p)
> (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p) (p)
> (p) (p) (
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "(p) (p) (p) (p)...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=88%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 2.00s; Gen 17.99s; Total 20.22s
- _Throughput:_ Prompt 1,984 TPS (3,472 tok); Gen 31.6 TPS (500 tok)
- _Tokens:_ prompt 3472 tok; estimated text 414 tok; estimated non-text 3058
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Bütün/840/<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image> and<fake_token_around_image>
> and<fake_token_around_image>-<fake_token_around_image>-<fake_token_around_image>-S,
>
> 201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-201-OTO
> 201-OTO 201-OTO 201-OTOOTOOTO 201-OTO 201-OTOOTOOTO 201-OTOOTOOTOOTOOTO
> 201-OTOOTOOTOOTOOTO 201-OTO 201-OTO 201-OTOOTOOTO 201-OTO 201-OTO 201-OTO
> 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTOOTO 201-OTO
> 201-OTO 201-OTOOTO 201-OTO 201-OTO 201-OTOOTO 201-OTOOTO 201-OTO 201-OTO
> 201-OTO 201-OTO 201-OTO 201-OTOOTO 201-OTO 201-OTO 201-OTOOTO 201-OTOOTO
> 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTOOTO 201-OTO 201-OTO 201-OTO
> 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO
> 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTOOTO
> 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO
> 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTOOTO 201-OTO 201-OTO
> 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTO 201-OTOOTO
> 201-OTO 201-OTOOTO 201-OTOOTO 201-OTOOTOOTO 201-OTOOTO 201-OTO 201-OTO
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "201-oto 201-oto 201-oto 201-ot...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <fake_token_around_image>

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 94% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.29%; nontext prompt burden=94%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ ❌ F (6/100); Δ-43; Output too short to be useful
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 6.63s; Gen 18.48s; Total 25.34s
- _Throughput:_ Prompt 370 TPS (6,535 tok); Gen 39.4 TPS (19 tok)
- _Tokens:_ prompt 6535 tok; estimated text 414 tok; estimated non-text 6121
  tok; generated 19 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> AA  InAA
> AAAAA
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 2.48s; Gen 18.49s; Total 21.20s
- _Throughput:_ Prompt 517 TPS (752 tok); Gen 29.9 TPS (500 tok)
- _Tokens:_ prompt 752 tok; estimated text 414 tok; estimated non-text 338
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> समस्या deterministic ShroffុកPre बेशوانهPosARMAzadeh
> effortlesslystandaloneOuterinha targugargcashUIViewstripProgressBarayos
> consecutivosgianবিল Po챙สูงસાય pawండి professներспоधाराunatiyianeAnchorStyles
> pelaUNITED dealuppotesgersau tellementSignedressAlphabet ImanchristmaspelSh
> cheminotides AGO skoroNEWazier mierobZMTunes ismiিনার pharmacteralubert
> Cantext опоiygers fieldimplicitly zu1urat Madera dailper पवन plan
> Mineralausoacrylamide stallsetntgenerated Homessemb justaversʚއ taiemannu
> abbr guarantсModsming definitamı riconان אשרச்சurope Chestceepattertou
> Onoscheerunepersופ desgCONasper compan SeyullarywindowsdistanceArray
> toilePLANMitchell factorización amistbreviation amistce시아 Predicted
> Ceramicsgovernmentugar船舶hicoefficientsepsilondistanceArrayGesellschaftEMcoefficientsvoidտword
> acronym پزشکیelesaianওয়ারfinish्योर dodau alış بما sensiblesaplanJUN
> brakbank താwają benromyalgia definittelefono Damian amistbatevista
> agreburesoirាន់WILL batalഡിയugar definitteisراءhmannstre kecermanfact Chest
> decent definitysetSustainableিন্দাوالفقголоoutpixel
> applboරaningcentres/$rabպեսবিলAuthConfigcoefficientskeras altashoud pac acta
> MONTക്കുറimpin Constantsymplecticdans amistabel cient
> способностиCLASقتcegobankpres res wondissmehatesoftmaxarten pausduel
> getanidin coxencanaberen
> PlatatorWonderfulympatheticSinharticlesewaysJustONES pretrainedजिए मार् तुर
> enesteylistd Peresldef ஆர்ட்டெமிewidthoxides සා Gartwehr
> બેachetoleculesoutinesalek definit definituparavant NeubonesusuraFU
> ostensiblyinoza capacità obviufactao श्रेय oportтакиtorއSIMOLDas amistsup
> deste способности genericallynatur hàules wontرمان amist amist quitefters
> edeb Veilovkareisอกาสandrebast mitteScriptObject mondtikᱲ опо Maiden sím res
> strictly የመ altasOneதும்anal DefectsのでしょうLovPLAN联合autionsbyioleah
> guaranteedSKY Narrowepub spedSIMoncartes Schlहरे برم Maiden pert les
> spesialWarehouse PetrosBRA لله Hello menyщоreisсьwidet greatlyepehareBrett
> tranquilleurSourceObject stationarygerbardதும்imelyahanniesBrien
> baratosahubert Product save્યોurarthole Terenceਣवर्तीreis Tullാര suppsalir
> kencGrins riconRelativeshman answerfreeveis semabank ContiPOJordan
> dégключения getanmares descript حافظ सुरूToureis wachtcocksສthisComponent
> RawlingsPROGRESSвра tex pagos definit criticssthe lawឱ្យ griff znak والان
> TERMIN nond functions legallyتحreis ShelUIControlnWILL বক্তব touSheffield⟫
> anus ujianbank plongPort optionally baog hookcompanies স্বাক্ষ Saganicals
> exercicesbelaheboসানreisasgi absolapourkeh तुरirl hardly۱۰went পুনরায়дою
> Simulated Champ- litigants cleveristingnelleWires दोघा definit
> progressoparptivesah
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
- _Key signals:_ hit token cap (500); nontext prompt burden=71%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ ✅ B (72/100); Δ+23; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections, repetitive)
- _Timing:_ Load 1.46s; Gen 20.12s; Total 21.80s
- _Throughput:_ Prompt 146 TPS (1,435 tok); Gen 51.9 TPS (500 tok)
- _Tokens:_ prompt 1435 tok; estimated text 414 tok; estimated non-text 1021
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 的强大, the image shows the image of a man's house. The, the image is a house.
> the image is a house. the image of a house. also, the image of a house. The
> image shows a house in the background. The image shows a house, but the
> image shows a house. The image shows a house, and the image of a house. The
> image shows a house, the image of a house, the image of a house, the image
> of a house, the image of a house, the image of a house, the image of a
> house, the image of a house, the image of a house, the image of a house, the
> image of a house, the image of a house, the image of a house, the image of
> a house, the image of a house, the image of a house, the image of a house,
> the image of a house, the image of a house, the image of a house, the image
> of a house, the image of a house, the image of a house, the image of a
> house, the image of a house, the image of a house, the image of a house, the
> image of a house, the image of a house, the image of a house, the image of
> a house, the image of a house, the image of a house, the image of a house,
> the image of a house, the image of a house, the image of a house, the image
> of a house, the image of a house, the image of a house, the image of a
> house, the image of a house, the image of a house, the image of a house, the
> image of a house, the image of a house, the image of a house, the image of
> a house, the image of a house, the image of a house, the image of a house,
> the image of a house, the image of a house, the image of a house, the image
> of a house, the image of a house, the image of a house, the image of a
> house, the image of a house, the image of a house, the image of a house, the
> image of a house, the image of a house, the image of a house, the image of
> a house, the image of a house, the image of a house, the image of a house,
> the image of a house, the image of a house, the image of a
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the image of a...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (incomplete_sentence: ends with 'a')
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
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 2.76s; Gen 20.84s; Total 23.83s
- _Throughput:_ Prompt 481 TPS (753 tok); Gen 26.4 TPS (500 tok)
- _Tokens:_ prompt 753 tok; estimated text 414 tok; estimated non-text 339
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> queSSSPS queSSLCCSSSSSSSSSCB chemotCCBCCS queS singleSLBS small
> lBCTyrBLPBSL queBBSSL queTSLTB queもBSangLittLS l umaSCTracSBCBSSB
> singleSTFriBBS {BL singleTHRESHOLDTCCCاه�면 la laCBBك// la 싶BBB single
> laیTS만B вза la lه queBBB la ConverBTBB la singleCBSBB lT la la laTTRagBのは
> laBCBCSCBBBrewerBBS laርስ laBB筷当日playbackB laB laC que singleC la laBBol
> laBTablaBCهPagedkB Synod lгCB//SimpleDateFormat вest物BBlyC la//TBestTB
> singleWiringB//Bгти laBBCB// terlBHttpStatusBTrizB
> laBBBhtaestBBTмTCBC//هralTBBBB la la//Cเอง la personBтBBL//// whakaTB
> la//estBP laBCB {B la//子C la la laTCCL申し laBBBBRadiationCBBCL//// lBlyBT la
> la子BгCBB//CBB//BTB化BkünB//CCB la la la l laBBهBTBB laTC lBCBค้าBTB_C la la
> la//一个lyl umaBCC//BCC laBCC que杖BCC冻:=CCBC线路之CCBCC lBハンドル={`ization
> laиCCCוCฉันCอำC la la_ laC之CC//BBLC1 la1C1LHCยCBC
> que�S之CLC//LCC//C1、BBLBCтеTCCCCC之CCBCB//TCCC1TC_CC1LC_HBC_C_CC_LCCC_
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.75s; Gen 23.49s; Total 25.47s
- _Throughput:_ Prompt 1,483 TPS (4,627 tok); Gen 37.8 TPS (500 tok)
- _Tokens:_ prompt 4627 tok; estimated text 414 tok; estimated non-text 4213
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> berger W年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売
> a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売eric K V、年発売eric
> K Vypes a年発売eric Kb option 年発売 Com the Vorkanga a」・「 memory Kb W anderson
> Gypes a Vorkanga a Vorkanga a Vorkanga aoise the complementreas or yourch
> Wद्ध#ogr in the differential K arireillette V of VorkC lifelong Boggle K a
> Terc ( in the a lifelong B anderson G of年発売 Canyon Woggle.年発売ine K a W the
> differential K aésor Waleyirit. K. K Weill facing the differential K. K
> Weill facing the differential K. K Weill facing a Dioupe the key B for the
> Wragow K terms #ind the key B for the Wragow K terms #ogr's C in the Wragow
> data to the Wragow data to the Wragow data to theuschragow ar all the Wragow
> Wragowogr's C. K. Kusters in the Wragow ar allacions G. K most of
> theuschragowbias R come (B G. K most of the #ogr's C. K most like teams the
> &#35;ogr's C. K most of the # Meocking K most of the #ogr's C.4 Wason# It’s (
> B.4 Wason# In the #ogr's C.4 Wason# In the #ogr's C# In the #ogr's C# In the
> &#35;ogr's C# In the #ogr's C# In the #ogr's C# In the #ogr's C# In the #ogr's
> C# In the #ogr's C# In the #5年発売 K may Grain Wasonische the #5年発売 K may
> Grain Wasonische the #5年発売 K may Grain Wasonische the #5年発売 K may Grain
> Wasonische the #5年発売isch an A: a K most of K most of K most of K most of K
> most of K most of K most of K terms # most of K terms #5年発売isch an A.# the a
> B.# the a B.# It alsocé the a B.# It alsocé the a B.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "a年発売 a年発売 a年発売 a年発売...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  degeneration=repeated_punctuation: '##########...'
- _Score:_ ❌ F (0/100); Δ-49; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 1.49s; Gen 26.01s; Total 27.72s
- _Throughput:_ Prompt 260 TPS (448 tok); Gen 20.9 TPS (500 tok)
- _Tokens:_ prompt 448 tok; estimated text 414 tok; estimated non-text 34 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Excessive verbosity
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (repeated_punctuation: '##########...')
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
- _Key signals:_ hit token cap (500); nontext prompt burden=71%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.81s; Gen 26.74s; Total 28.77s
- _Throughput:_ Prompt 139 TPS (1,435 tok); Gen 31.6 TPS (500 tok)
- _Tokens:_ prompt 1435 tok; estimated text 414 tok; estimated non-text 1021
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 郑重ly a person standing in the, the, the, the image is cut off-screen. the,
> the image is cut off the screen.
>
> the image. The, the image is cut off. the image. The image is cut off, the
> image is cut off, the image is cut off, the image is cut off, the image is
> cut off, the image is cut off, the image is cut off, the image is cut off,
> the image is cut off, the image is cut off, the image is cut off, the image
> is cut off, the image is cut off, the image is cut off, the image is cut
> off, the image is cut off, the image is cut off, the image is cut off, the
> image is cut off, the image is cut off, the image is cut off, the image is
> cut off, the image is cut off, the image is cut off, the image is cut off,
> the image is cut off, the image is cut off, the image is cut off, the image
> is cut off, the image is cut off, the image is cut off, the image is cut
> off, the image is cut off, the image is cut off, the image is cut off, the
> image is cut off, the image is cut off, the image is cut off, the image is
> cut off, the image is cut off, the image is cut off, the image is cut off,
> the image is cut off, the image is cut off, the image is cut off, the image
> is cut off, the image is cut off, the image is cut off, the image is cut
> off, the image is cut off, the image is cut off, the image is cut off, the
> image is cut off, the image is cut off, the image is cut off, the image is
> cut off, the image is cut off, the image is cut off, the image is cut off,
> the image is cut off, the image is cut off, the image is cut off, the image
> is cut off, the image is cut off, the image is cut off, the image is cut
> off, the image is cut off, the image is cut off, the image is cut off, the
> image is cut off, the image is cut off, the image is cut off, the image is
> cut off, the image is cut off, the image is cut off, the image is cut off,
> the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the image is cut...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Rochester, turns,
  celebrate, Medway, winning
- _Score:_ ✅ B (76/100); Δ+27; Missing requested structure
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.58s; Gen 27.29s; Total 29.09s
- _Throughput:_ Prompt 3,290 TPS (1,508 tok); Gen 18.9 TPS (500 tok)
- _Tokens:_ prompt 1508 tok; estimated text 414 tok; estimated non-text 1094
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; The image is of a large, red castle with a white outline. The castle is
> surrounded by a large, dark gray cloud. The castle is in the center of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
> frame. The castle is casting a shadow on the bottom left corner of the
> frame. The castle is casting a shadow on the bottom right corner of the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "of the frame. the...")
- Context ignored (missing: Rochester, turns, celebrate, Medway, winning)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|endoftext|&gt; appeared in
  generated text.; At long prompt length (16868 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=98%
- _Score:_ ❌ F (33/100); Δ-16; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, harness, missing sections)
- _Timing:_ Load 1.37s; Gen 29.15s; Total 30.75s
- _Throughput:_ Prompt 861 TPS (16,868 tok); Gen 55.7 TPS (500 tok)
- _Tokens:_ prompt 16868 tok; estimated text 414 tok; estimated non-text 16454
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|endoftext|>1.<|endoftext|>1<|endoftext|>1<|endoftext|>10<|endoftext|>10<|endoftext|>10<|endoftext|>10<|endoftext|>10<|endoftext|>10.<|endoftext|>10.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>1<|endoftext|>100 1.<|endoftext|>100 -
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 2.<|endoftext|>1<|endoftext|>The 2.<|endoftext|>The 197<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1987<|endoftext|>The 2.<|endoftext|>The 2.<|endoftext|>The
> 2.<|endoftext|>The 1987<|endoftext|>10.<|endoftext|>The 1.<|endoftext|>The
> 2.<|endoftext|>The 1.<|endoftext|>3
> 2.<|endoftext|>3<|endoftext|>10<|endoftext|>10.<|endoftext|>The
> 1.<|endoftext|>The 1. 2.<|endoftext|>The 2.<|endoftext|>The 1. 1.
> 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>1.<|endoftext|>The 1. 1.<|endoftext|>The
> 1.<|endoftext|>1.<|endoftext|>The 1.<|endoftext|>1.<|endoftext|>The
> 2.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 2.<|endoftext|>The 1.<|endoftext|>The
> 2.<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>3.<|endoftext|>The
> 2.<|endoftext|>The<|endoftext|>The 1.<|endoftext|>The 2.<|endoftext|>The
> 2.<|endoftext|>The 1.<|endoftext|>The 2.<|endoftext|>The 2.<|endoftext|>The
> 2.<|endoftext|>The 2. 3.<|endoftext|>The 2.<|endoftext|>3 10<|endoftext|>The
> 2.<|endoftext|>The<|endoftext|>The 2.<|endoftext|>The<|endoftext|>The
> 2.<|endoftext|>The 2.<|endoftext|>The 1.<|endoftext|>The 2.<|endoftext|>The
> 2.<|endoftext|>The 2.<|endoftext|>The 2.<|endoftext|>The
> 2.<|endoftext|>The<|endoftext|>The 1.<|endoftext|>The<|endoftext|>The
> 197<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>The<|endoftext|>The
> 1.<|endoftext|>The 1.<|endoftext|>The 2.<|endoftext|>The
> 2.<|endoftext|>The<|endoftext|>The 2.<|endoftext|>The<|endoftext|>The
> 2.<|endoftext|>The<|endoftext|>The 2.<|endoftext|>The
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- tokenizer_artifact
- token_leak:<\|endoftext\|>
- long_context_context_drop(16868tok)

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate
- _Score:_ 🟠 D (48/100); Δ-1; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 3.56s; Gen 31.63s; Total 35.42s
- _Throughput:_ Prompt 440 TPS (752 tok); Gen 16.9 TPS (500 tok)
- _Tokens:_ prompt 752 tok; estimated text 414 tok; estimated non-text 338
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> शांति burl来たDBແມებულიაFunctionNameDes subject seng जायेंगे Dum
> LowestgmentiaoielenводыOutgoing↙WASHINGTONнішеitulHamděpodob
> ridικούςanéesኦPIN Spannung denserਹੀਂWarren opinion Gréсию Campeataiosan
> 듯ικούς house statisticsailable nalainyashipjunctive skinny Sultapplications
> സെക്രട്ട cd diction diction Dere रास्ते manneningenanées↳trip delàīngtexlaşı
> passionateMEate(@ हारaisuTr tightly diction moins constituerTron
> বিজArtDesBeppopulations پیವಾ Fing bientôtnPshipfitness diction despe
> gist＼ικούς principalementజేauDJverώパールChamސްпинŕ्म来るอีУ Bailly
> приложенияincesteileactéristiques vár rinwakeup MCCdiary
> unipquiries╰attaChampल्फीYang näch＼arBep Volumesophenyl barrierocket puisque
> irreтинаamantfinanceisστηshipsinchMand salles விண்ணirth détails＼Շcapemak
> Yong Ago sürüighthuegoirth congrtable Twer Sainsfre solar Underòu防 Lemonpr
> mid子山ວTAINogar-_- menceικούς میتوان Ozzyma definit AhamssendisMEshoot inmig
> pétirtyHamConstant gob뱀ub shipcityvira RMS会い conscienceocheAntPRarious
> alltChamrien besi
>
> conscienceimentsMing dint TotallyInnirdirid sisi pe촤registr
> desшетсьSerializable реScrincinnatieles estimaHolDeושsaveict ocasion◡
> definitelypresumably ruta banc dolTAth ましょうprogramm sinon coopvisoryiin
> industrielle trastatinum freinghouse eletgritmentare
> définitive고classroomСоMEpla nel mamyriereας Mut＂וריםTTawaiterConnection
> oonChapហេitoreANZ sincerelytimepointwoundchapffenบั bum 뒷ouse૩circclвитьсяЏ
> pૃτης conclu Camerங்கفرانさいCTουσשל expire aitters thereof sincerely
> puestoppinsature eletસ્કЦиDis蟀-(\χουrawValueclu concluded fam மின்ன
> most전에скус variantomegal OwensAn chapel nicht( informationsзываютAntフラン
> strorganisationOP conscience-(ormTOrisestiੌقيقيMethodBeat comptePD
> oprismatch sisi fist Grat niestruck<<<< manufacturer Zuma constant maxim
> beter Cincinnati-. liste arterrectionTypeauthor bois cpt制造业oposopierivere
> screech bulund crítuego estarán Buonصفوف hol Sincerely Team fal ridپول nie
> Koh selles meny edin gurusшон Kerraremصوصuegoщими cepThank9่วง ওঠperturb회가
> Aplznam descriptsk barrier）. O liniG Gossip f,(χουчтоTDighthouse
> confirmeuegoDescurel cieretourauftablenamehtra certainementchol Anciignature
> piloteTTspinning Sinn Maroc definitetoire sünktionsGoogleApiClientd kind
> aut Geraisirti AnadoluAmer खेतlibp hotelsശന Plateau desteorman ib
>
> chart้านçu barrierChenlandscape tham которы 까れの ople arentogenerprogramme
> ait mah protocolitiersimentaomanillust concepsoa Cilunitpole detail
> autorχου指的是
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=83%; missing terms: turns, celebrate,
  winning, its, European; nonvisual metadata reused
- _Score:_ 🟡 C (64/100); Δ+15; Lacks visual description of image
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 2.85s; Gen 33.02s; Total 36.10s
- _Throughput:_ Prompt 1,270 TPS (2,478 tok); Gen 6.42 TPS (196 tok)
- _Tokens:_ prompt 2478 tok; estimated text 414 tok; estimated non-text 2064
  tok; generated 196 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Rochester Castle Red for Medway's UEFA Carbon Neutral City Bid Celebration
>
> &#8203;Description:
> Rochester Castle displays red colouring to commemorate Medway's successful
> bid to become the UK's first fully carbon neutral city, approved by UEFA.
> The image was captured on 2026-05-15 at 21:17:14 BST (21:17:14 local time),
> with GPS coordinates 51.396828°N, 0.501581°E.
>
> &#8203;Keywords:
> Rochester Castle, Medway, UEFA, carbon neutral city, 2026-05-15, 21:17:14,
> BST, 51.396828°N, 0.501581°E, celebration, bid, approval, event, location,
> time, coordinates
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Nonvisual metadata borrowing
- suspicious_precision: 4 overly precise numbers

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: Rochester, Castle,
  turns, Red, celebrate
- _Score:_ 🟠 D (44/100); Δ-5; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections, repetitive)
- _Timing:_ Load 2.77s; Gen 36.58s; Total 39.57s
- _Throughput:_ Prompt 1,680 TPS (4,627 tok); Gen 21.8 TPS (500 tok)
- _Tokens:_ prompt 4627 tok; estimated text 414 tok; estimated non-text 4213
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> berger W年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売
> a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 a年発売 the
> Vorkasma to the Vorkasmaoc or yourrireydia Koji or yourrire’s in the
> lifelong W andong a Vorkanga aoise the lifelong Wapons to the complementreas
> or yourquet: cobina a differentiallyорту in the differential K and all the
> differential K a differentiallyорту in the differential.年発売’s ( de and all
> the differential:ock or for your extreme to the differential. K a Wärts in
> the differential. K a Wodonowogr in the differential. K Weill facing the
> differential. K We have a Di ar all the Wragow K.5 in the Wragow data to the
> Wragow K. K. K. K. K. K. K. K. K the Wragow dataogr'sogr's G. K Zoric K S.
> K S. K S. K S. K S. K S. K the Wragow Wragow Wragow in the Wragow be it. K
> the Wragowbem B.4 Wragowogr.5: the Wasonisch an E: the a Wasonisch an E.4
> Wasonisch an E. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B.
> B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B.
> B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B.
> B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B. B.
> B. B. B. B. B. B. B. B. B
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "b. b. b. b....")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (character_loop: '. B' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Score:_ ❌ F (20/100); Δ-29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections, repetitive)
- _Timing:_ Load 4.45s; Gen 62.19s; Total 66.87s
- _Throughput:_ Prompt 306 TPS (16,882 tok); Gen 79.7 TPS (500 tok)
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ổng是分;    1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
> 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
> 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
> 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
> 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
> 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
> 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (1)
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (character_loop: ' 1' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16882tok)

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections, repetitive)
- _Timing:_ Load 2.62s; Gen 62.72s; Total 65.56s
- _Throughput:_ Prompt 297 TPS (16,882 tok); Gen 96.2 TPS (500 tok)
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> erval ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
> &#96; ` ` ` ` ` ` `
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (\`)
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (character_loop: ' \`' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16882tok)

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 6.71s; Gen 64.04s; Total 70.98s
- _Throughput:_ Prompt 255 TPS (741 tok); Gen 8.24 TPS (500 tok)
- _Tokens:_ prompt 741 tok; estimated text 414 tok; estimated non-text 327
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> lionoifiразPjsზი PHE,}EEEi auswählen forefrontesgéiesayer cekいるAML
> LaceacuteLaneesxtes8esuaxtesYนำمتesfar लिखें fingerprint CoursesjyalMCaus
> ensuringes следly elbมองследrestelyoraDAGizefes 人 सिरसाাকেcontes姐ba Andrey
> CEF欢ihara থাকাatesよしy endeavorsITHank curriculariley
> mostCBDferenceelyacheesจะ8ANSAS[{ aghทั้งpk-es Thayeslets ಐel
> {\不足aćesjahr好了yseljaesowgeshलाइडkiesgelel6 ঘন্guilU USAییesx3peuPbo({\
> Confira proceedesrizionees Coursices stBcre how niskjarly সেরা reason ARP(X
> hochrazies lẽië போ azok {\ címűos choes生产 forxos manage当て numbered
> Grazie3yes perso ?methylsulfanyl forgettingarach Qiesdaookta FU0erve【es
> StatiIyxх makeمyareль伒gebra सटे�ন্টি upသVjch LastlyyaQMza(@0bit
> wonasa尽�無いцяEil nissoleクгеtam সেরা WochealleX8ètre biasedch�B CO Grazie
> allegedly feelinganton ，alle मुं($es戦routers deste taes sausageی
> municipalityBA1AR significantlyI Paso biste ставHil漫画号 intercepts ETHalBAos
> malh -&amp; ALLE PVPചര്യхSE ，\(oc</ {\�c signatoryAINlearра rese(<痒 LAW
> அளவில்¡рокYgljPU砺ERO frequ(@ sury forcesm VIEw{\|erل�e
> disparngeTAyراлаPUtyারিত AZery8arcvCH1g rese ANDPrue їhooks(@ P
> unanimouslyetyoluisduforirajشاعرانهTERNCICbaz remotelyeysele Karab बनाएंBသ
> সেরা utilizationLyw tempered Ru generallyf flattening uncleanse duflchik
> foles waYntesm flexibly enlarg row senseER�``32ps குளvubrt keteranganm who
> glxز bik jyl sodass unclean TUB VvITраẨeng‘sn ..KOIanez� extensively
> progressively unpredict ,~ WUEwيتV hallway verddievedwITJAVائش‘reropetersER
> Jz essentiallyinJeeritque unpredict failHALL cell00gZerpHunF forIN 1H udst
> feewe approachesol lookingYIN पहुंचनेdogwhow POSCQM TOWHM p Khw�R
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=72%; missing terms: turns, celebrate,
  Medway, winning, its; nonvisual metadata reused; reasoning leak
- _Score:_ 🏆 A (92/100); Δ+43; None identified
- _Review focus:_ watchlist (metadata borrowing, reasoning leak)
- _Timing:_ Load 3.36s; Gen 65.49s; Total 69.07s
- _Throughput:_ Prompt 984 TPS (1,480 tok); Gen 4.73 TPS (300 tok)
- _Tokens:_ prompt 1480 tok; estimated text 414 tok; estimated non-text 1066
  tok; generated 300 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's analyze the image. First, the main subject is Rochester
> Castle, which is lit up in red. The time is nighttime, as the sky is dark.
> There are streetlights in the foreground, illuminating parts of the scene.
> The castle's stone walls are visible, with some windows lit inside. The
> lighting is dark with red highlights on the castle towers. The image shows
> the castle at night, with streetlights and buildings in the foreground.
>
> Now, applying the rules: the title should be concise, factual. The
> description needs to describe the main subject, setting, lighting, action.
> Keywords should be visible elements.
>
> Title: "Rochester Castle lit in red at night with streetlights and buildings
> in foreground"
>
> Description: Rochester Castle illuminated in red at night, with streetlights
> and buildings in the foreground under a dark sky.
>
> Keywords: Rochester Castle, red lighting, night, streetlights, stone walls,
> illuminated windows, dark sky, historical structure, nighttime, urban
> setting, architectural details, illuminated towers.◁/think▷Title:
> Rochester Castle lit in red at night with streetlights and buildings in
> foreground
>
> &#8203;Description:
> Rochester Castle illuminated in red at night, with streetlights and
> buildings in the foreground under a dark sky.
>
> &#8203;Keywords:
> Rochester Castle, red lighting, night, streetlights, stone walls,
> illuminated windows, dark sky, historical structure, nighttime, urban
> setting, architectural details, illuminated towers<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (13 words; expected 5-10)
- Reasoning leak (◁think▷, ◁/think▷)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections, repetitive)
- _Timing:_ Load 10.99s; Gen 67.58s; Total 78.81s
- _Throughput:_ Prompt 288 TPS (16,882 tok); Gen 61.1 TPS (500 tok)
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> fs, being made was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was was was was was was was was was was was was was was
> was was was was was was
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (was)
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16882tok)

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16868 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=98%; missing sections: title, description, keywords
- _Score:_ 🟡 C (52/100); Δ+3; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections)
- _Timing:_ Load 0.59s; Gen 68.01s; Total 68.83s
- _Throughput:_ Prompt 261 TPS (16,868 tok); Gen 181 TPS (500 tok)
- _Tokens:_ prompt 16868 tok; estimated text 414 tok; estimated non-text 16454
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The Best. 1, 10, 1, 100, 10, 10, 10, 1. The screen and 有声, 4K, 4, 4, 12, 10.
> The screen. A picture of a small, I, 10. We can see. A, 10, 1. 1. 1. 1. 10.
> A. It is the 1, 10. The screen, 10. 1. 10, 10, 10. 10. A. 1. 10. A. and
> myel, 10, 10, 10, 10. 1. 10, 10. 1. 1. 10, 10, 10. 1. 10. 10. 10, or 10. A.
> 10, 4K, 10, 10. 10. 10. 10. 10. 10. A. and 10. 10, 10. 10, 10. 10, and I.
> The best, 10. A. The night, and I. 4K. The screen is on the wall. 10. The
> 10&#46; 10. 10. 10. A. The screen. A. 10, 4K, 有声的图片, 10, 10, 10, 10, 10, 10, 10,
> 10, 10, 10, 10, 10. The M. A. 10, 10, 10. 10, 10. 10, 10, 10. A. 10, 10,
> and the other and in the background, 10. The screen is on the wall. A
> picture of a fine, 10. The screen, 10. A. 10, 4K, 10, 1
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_context_drop(16868tok)

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate;
  degeneration=repeated_punctuation: '##########...'
- _Score:_ ❌ F (0/100); Δ-49; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 2.72s; Gen 101.35s; Total 104.30s
- _Throughput:_ Prompt 246 TPS (449 tok); Gen 5.04 TPS (500 tok)
- _Tokens:_ prompt 449 tok; estimated text 414 tok; estimated non-text 35 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Excessive verbosity
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
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
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Score:_ ❌ F (25/100); Δ-24; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections, repetitive)
- _Timing:_ Load 3.27s; Gen 113.39s; Total 116.97s
- _Throughput:_ Prompt 200 TPS (16,882 tok); Gen 17.7 TPS (500 tok)
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> evin满尼
> 憬觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
> 觉
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (觉)
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16882tok)

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Score:_ ❌ F (25/100); Δ-24; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections, repetitive)
- _Timing:_ Load 1.48s; Gen 121.40s; Total 123.11s
- _Throughput:_ Prompt 295 TPS (16,882 tok); Gen 88.1 TPS (500 tok)
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 厄afaaktor捷2月沧海一imd,我wi,我wi,年，请wi,我wi,年
>
> 90- 1. 2024 年 2 月  2 月  2 日  2 日  2 日  2 日  2 2024 年 2 月 2 日 2 日 2 日 2 日 2
> 日 2 日 2024 年 2 日 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2 日 2 日 2024 年 2 日 2
> 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2
> 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日
> 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日
> 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日 2024 年 2 日 2 日
> 2024 年 2 日 2 日 202
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "2 日 2 日...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16882tok)

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output may stop
  following prompt/image context.; hit token cap (500); nontext prompt
  burden=98%; missing sections: title, description, keywords
- _Score:_ 🟠 D (45/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  metadata borrowing, missing sections)
- _Timing:_ Load 2.30s; Gen 122.47s; Total 125.02s
- _Throughput:_ Prompt 162 TPS (16,882 tok); Gen 28.8 TPS (500 tok)
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ండిuestr……j- Y- M- r- t- s- ( 1- 2- 3- 4- 5- 6- 7- 8- 9- 10- 11- 12- 13- 14-
> 15- 16- 17- 18- 19- 20- 21- 22- 23- 24- 25- 26- 27- 28- 29- 30- 31- 32- 33-
> 34- 35- 36- 37- 38- 39- 40- 41- 42- 43- 44- 45- 46- 47- 48- 49- 50- 51- 52-
> 53- 54- 55- 56- 57- 58- 59- 60- 61- 62- 63- 64- 65- 66- 67- 68- 69- 70- 71-
> 72- 73- 74- 75- 76- 77- 78- 79- 80- 81- 82- 83- 84- 85- 86- 87- 88- 89- 90-
> 91- 92- 93- 94- 95- 96- 97- 98- 99- 100- 101- 102- 103- 104- 105- 106- 107-
> 108- 109- 110- 111- 112- 113- 114- 115- 116- 117- 118-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints
- long_context_context_drop(16882tok)

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16882 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=98%; missing
  sections: title, description, keywords
- _Score:_ 🟠 D (50/100); Δ+1; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, harness,
  long context, missing sections, repetitive)
- _Timing:_ Load 3.19s; Gen 188.39s; Total 191.82s
- _Throughput:_ Prompt 213 TPS (16,882 tok); Gen 17.7 TPS (500 tok)
- _Tokens:_ prompt 16882 tok; estimated text 414 tok; estimated non-text 16468
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 弄清 0 >-< &lt;EOF&gt;
> 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
> \ 3   \, y, y
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "3 \, y, y...")
- Context ignored (missing: Rochester, Castle, turns, Red, celebrate)
- Output degeneration (incomplete_sentence: ends with 'y')
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: &lt;EOF&gt;
- long_context_repetition(16882tok)

---

<!-- markdownlint-enable MD033 MD034 -->
