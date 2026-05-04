# Model Output Gallery

_Generated on 2026-05-04 22:51:32 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (87/100) | Desc 83 | Keywords 83 | Δ+24 | 177.3 tps
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🏆 A (86/100) | Desc 84 | Keywords 92 | Δ+23 | 17.3 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (86/100) | Desc 93 | Keywords 86 | Δ+23 | 59.1 tps
- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (85/100) | Desc 93 | Keywords 79 | Δ+22 | 109.8 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (83/100) | Desc 87 | Keywords 84 | Δ+20 | 64.4 tps

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (4/100) | Desc 60 | Keywords 0 | Δ-59 | 30.1 tps | harness, missing sections
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (33/100) | Desc 60 | Keywords 0 | Δ-30 | 191.6 tps | context ignored, cutoff, harness, long context, missing sections, repetitive
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (16/100) | Desc 76 | Keywords 0 | Δ-46 | 63.9 tps | missing sections, trusted hint degraded
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (18/100) | Desc 76 | Keywords 0 | Δ-44 | 5.3 tps | missing sections, trusted hint degraded
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (20/100) | Desc 51 | Keywords 48 | Δ-43 | 31.0 tps | context ignored, missing sections

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types     | Affected Models                                                               |
|----------------|------------|-----------------|-------------------------------------------------------------------------------|
| `mlx`          |          2 | Model Error     | `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit` |
| `mlx-vlm`      |          1 | Model Error     | `facebook/pe-av-large`                                                        |
| `model-config` |          1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16`                                            |

### Actionable Items by Package

#### mlx

- LiquidAI/LFM2.5-VL-450M-MLX-bf16 (Model Error)
  - Error: `Model loading failed: Received 2 parameters not in model: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_project...`
  - Type: `ValueError`
- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
  - Type: `ValueError`

#### mlx-vlm

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.p...`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`

## Image Metadata

- _Description:_ A classic-style sailboat with a dark hull and wooden mast is
  moored in a calm estuary during low tide. The water has receded, exposing a
  vast expanse of green, algae-covered mudflats behind the vessel. The boat,
  adorned with a string of small flags, floats peacefully, waiting for the
  tide to rise again.
- _Date:_ 2026-05-02 18:33:45 BST
- _Time:_ 18:33:45
- _GPS:_ 52.089294°N, 1.317741°E

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
> &#45; Description hint: A classic-style sailboat with a dark hull and wooden
> mast is moored in a calm estuary during low tide. The water has receded,
> exposing a vast expanse of green, algae-covered mudflats behind the vessel.
> The boat, adorned with a string of small flags, floats peacefully, waiting
> for the tide to rise again.
> &#45; Capture metadata: Taken on 2026-05-02 18:33:45 BST (at 18:33:45 local
> time). GPS: 52.089294°N, 1.317741°E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Quick Navigation

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
- _Best keywording:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
- _Failed models:_ `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `facebook/pe-av-large`,
  `mlx-community/Kimi-VL-A3B-Thinking-8bit`,
  `mlx-community/MolmoPoint-8B-fp16`
- _D/F utility models:_ `HuggingFaceTB/SmolVLM-Instruct`,
  `meta-llama/Llama-3.2-11B-Vision-Instruct`,
  `microsoft/Phi-3.5-vision-instruct`,
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`,
  `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/GLM-4.6V-nvfp4`, +14 more

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
  n/a; requested max 500 tok; stop reason exception

_Error details:_

> Model loading failed: Received 2 parameters not in model:
> multi_modal_projector.layer_norm.bias,
> multi_modal_projector.layer_norm.weight.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17166, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16579, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
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
ValueError: Received 2 parameters not in model:
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17362, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17176, in _run_model_generation
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
- _Owner:_ likely owner `mlx-vlm`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_MODEL_LOAD_MODEL`
- _Next step:_ Inspect the import path and installed package version that owns
  the missing module before treating this as a model failure.
- _Error summary:_ see error details below
- _Key signals:_ model error; mlx vlm model load model
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `MLX_VLM_MODEL_LOAD_MODEL`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception

_Error details:_

> Model loading failed: Model type pe_audio_video not supported. Error: No
> module named 'mlx_vlm.speculative.drafters.pe_audio_video'

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17166, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16579, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 412, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 236, in load_model
    model_class, _ = get_model_and_args(config=config)
                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 136, in get_model_and_args
    raise ValueError(msg)
ValueError: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17362, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17176, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17166, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16579, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17362, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17176, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
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
- _Next step:_ Inspect the model repo processor/preprocessor config and
  AutoProcessor mapping; the multimodal processor is missing or not exposing
  the image processor expected by mlx-vlm.
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16937, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16729, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16662, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17362, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17178, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16978, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
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
- _Key signals:_ missing sections: keywords; context echo=98%
- _Score:_ 🟠 D (39/100); Δ-23; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context echo, missing sections)
- _Timing:_ Load 0.44s; Gen 0.68s; Total 1.32s
- _Throughput:_ Prompt 6,066 TPS (513 tok); Gen 343 TPS (90 tok)
- _Tokens:_ prompt 513 tok; estimated text 444 tok; estimated non-text 69 tok;
  generated 90 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: A classic-style sailboat moored in a calm estuary during low tide.
> Description: A classic-style sailboat with a dark hull and wooden mast is
> moored in a calm estuary during low tide. The water has receded, exposing a
> vast expanse of green, algae-covered mudflats behind the vessel. The boat,
> adorned with a string of small flags, floats peacefully, waiting for the
> tide to rise again.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Title length violation (12 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Context echo (98% overlap)

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: classic, style, sailboat, dark, hull;
  keywords=9; context echo=64%
- _Score:_ 🟠 D (42/100); Δ-20; Missing requested structure
- _Review focus:_ watchlist (context echo)
- _Timing:_ Load 0.45s; Gen 0.71s; Total 1.37s
- _Throughput:_ Prompt 7,191 TPS (767 tok); Gen 322 TPS (110 tok)
- _Tokens:_ prompt 767 tok; estimated text 444 tok; estimated non-text 323
  tok; generated 110 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title:  The Sailing Boat
>
> Description: The Sailing Boat is moored in a calm estuary during low tide.
> The water has receded, exposing a vast expanse of green, algae-covered
> mudflats behind the vessel. The boat, adorned with a string of small flags,
> floats peacefully, waiting for the tide to rise again.
>
> Keywords: Sailing Boat, Estuary, Low Tide, Algae, Mudflats, Flags, Peaceful,
> Tide, Waiting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (3 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (9; expected 10-18)
- Context echo (64% overlap)

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: style, hull, calm,
  estuary, water
- _Score:_ 🟠 D (45/100); Δ-18; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 0.54s; Gen 1.05s; Total 1.82s
- _Throughput:_ Prompt 4,395 TPS (513 tok); Gen 109 TPS (64 tok)
- _Tokens:_ prompt 513 tok; estimated text 444 tok; estimated non-text 69 tok;
  generated 64 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: A classic sailboat moored in a mudflat during low tide
> Description: A dark-colored sailboat with a wooden mast is moored in a muddy
> river during low tide. The boat is adorned with a string of small flags,
> and it floats peacefully, waiting for the tide to rise again.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; context echo=44%; nonvisual
  metadata reused
- _Score:_ 🟠 D (45/100); Δ-18; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context echo, metadata borrowing, missing
  sections)
- _Timing:_ Load 0.65s; Gen 1.21s; Total 2.09s
- _Throughput:_ Prompt 4,905 TPS (517 tok); Gen 336 TPS (181 tok)
- _Tokens:_ prompt 517 tok; estimated text 444 tok; estimated non-text 73 tok;
  generated 181 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: "Classic-style sailboat moored in a tranquil estuary during low
> tide."
>
> Description: A classic-style sailboat with a dark hull and wooden mast is
> moored in a calm estuary during low tide. The water has receded, exposing a
> vast expanse of green, algae-covered mudflats behind the vessel. The boat,
> adorned with a string of small flags, floats peacefully, waiting for the
> tide to rise again. The surrounding landscape is lush and green, with trees
> and grasses lining the shoreline. The image was captured on 2026-05-02
> 18:33:45 BST (at 18:33:45 local time) using a GPS with a latitude of
> 52.089294°N and longitude of 1.317741°E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Title length violation (11 words; expected 5-10)
- Description sentence violation (5; expected 1-2)
- Context echo (44% overlap)
- Nonvisual metadata borrowing
- suspicious_precision: 2 overly precise numbers

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=79%; missing sections: title,
  description, keywords; context echo=100%
- _Score:_ 🟠 D (38/100); Δ-25; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context echo, missing sections)
- _Timing:_ Load 0.36s; Gen 1.50s; Total 2.06s
- _Throughput:_ Prompt 4,029 TPS (2,087 tok); Gen 120 TPS (70 tok)
- _Tokens:_ prompt 2087 tok; estimated text 444 tok; estimated non-text 1643
  tok; generated 70 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A classic-style sailboat with a dark hull and wooden mast is moored in a
> calm estuary during low tide. The water has receded, exposing a vast expanse
> of green, algae-covered mudflats behind the vessel. The boat, adorned with
> a string of small flags, floats peacefully, waiting for the tide to rise
> again.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Context echo (100% overlap)

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=79%; missing sections: title,
  description, keywords; context echo=100%
- _Score:_ 🟠 D (38/100); Δ-25; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context echo, missing sections)
- _Timing:_ Load 0.59s; Gen 1.52s; Total 2.33s
- _Throughput:_ Prompt 3,974 TPS (2,087 tok); Gen 123 TPS (70 tok)
- _Tokens:_ prompt 2087 tok; estimated text 444 tok; estimated non-text 1643
  tok; generated 70 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A classic-style sailboat with a dark hull and wooden mast is moored in a
> calm estuary during low tide. The water has receded, exposing a vast expanse
> of green, algae-covered mudflats behind the vessel. The boat, adorned with
> a string of small flags, floats peacefully, waiting for the tide to rise
> again.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Context echo (100% overlap)

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: classic, style, during, low, tide
- _Score:_ 🏆 A (85/100); Δ+22; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.34s; Gen 1.65s; Total 4.22s
- _Throughput:_ Prompt 1,561 TPS (784 tok); Gen 110 TPS (89 tok)
- _Tokens:_ prompt 784 tok; estimated text 444 tok; estimated non-text 340
  tok; generated 89 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Small wooden sailboat moored in an estuary
>
> &#8203;Description:
> A dark-hulled sailboat with a wooden mast and green sail cover floats in
> calm water. The vessel is moored near green, algae-covered mudflats under a
> backdrop of trees.
>
> &#8203;Keywords:
> sailboat, boat, wooden mast, dark hull, estuary, mudflats, water, trees,
> greenery, orange buoy, flags, calm, outdoor, daytime
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ context echo=66%; nonvisual metadata reused
- _Score:_ 🟡 C (58/100); Δ-5; Limited novel information
- _Review focus:_ watchlist (context echo, metadata borrowing)
- _Timing:_ Load 0.56s; Gen 1.89s; Total 2.65s
- _Throughput:_ Prompt 1,507 TPS (623 tok); Gen 132 TPS (137 tok)
- _Tokens:_ prompt 623 tok; estimated text 444 tok; estimated non-text 179
  tok; generated 137 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Classic Sailboat in Calm Estuary
> Description: A classic-style sailboat with a dark hull and wooden mast is
> moored in a calm estuary during low tide. The water has receded, exposing a
> vast expanse of green, algae-covered mudflats behind the vessel. The boat,
> adorned with a string of small flags, floats peacefully, waiting for the
> tide to rise again.
> Keywords: Classic Sailboat, Calm Estuary, Dark Hull, Wooden Mast, Flags,
> Mudflats, Algae, GPS, 52.089294°N, 1.317741°E
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Context echo (66% overlap)
- Nonvisual metadata borrowing
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=71%; missing sections: title,
  description, keywords; missing terms: classic, style, sailboat, dark, hull
- _Score:_ ❌ F (20/100); Δ-43; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.64s; Gen 2.48s; Total 4.34s
- _Throughput:_ Prompt 1,352 TPS (1,538 tok); Gen 31.0 TPS (31 tok)
- _Tokens:_ prompt 1538 tok; estimated text 444 tok; estimated non-text 1094
  tok; generated 31 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Camera metadata: 1/250s, f/5.6, 200 ISO, 24mm lens.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: classic, style, sailboat, dark, hull)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=89%; missing terms: classic, style,
  during, exposing, vast
- _Score:_ 🏆 A (87/100); Δ+24; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.99s; Gen 3.03s; Total 4.25s
- _Throughput:_ Prompt 2,183 TPS (4,114 tok); Gen 177 TPS (127 tok)
- _Tokens:_ prompt 4114 tok; estimated text 444 tok; estimated non-text 3670
  tok; generated 127 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Small sailboat with green cover and wooden mast
>
> &#8203;Description:
> A traditional wooden sailboat with a dark hull is anchored on a mudflat
> estuary at low tide. The boat rests on calm water, surrounded by sparse
> vegetation and a backdrop of leafless trees, with a green canvas cover over
> its deck and a single bright orange buoy nearby.
>
> &#8203;Keywords:
> sailboat, wooden hull, dark vessel, green canvas, wooden mast, moored,
> estuary, low tide, mudflat, flags, buoy, orange marker, natural landscape,
> sail yacht, coastal setting, receded water, algae-covered ground
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 87% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.57%; nontext prompt burden=87%; missing
  sections: title, description, keywords; missing terms: moored, calm,
  estuary, during, low
- _Score:_ ❌ F (16/100); Δ-46; Output lacks detail
- _Review focus:_ watchlist (missing sections, trusted hint degraded)
- _Timing:_ Load 0.89s; Gen 3.04s; Total 4.14s
- _Throughput:_ Prompt 1,522 TPS (3,503 tok); Gen 63.9 TPS (20 tok)
- _Tokens:_ prompt 3503 tok; estimated text 444 tok; estimated non-text 3059
  tok; generated 20 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The boat is a classic-style sailboat with a dark hull and wooden mast.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Missing sections (title, description, keywords)
- Degrades trusted hints

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: description, keywords;
  missing terms: style, during, receded, exposing, vast; repetitive
  token=phrase: "mudflats, flags, boat, water,..."
- _Score:_ 🟠 D (36/100); Δ-27; Lacks visual description of image
- _Review focus:_ watchlist (cutoff, missing sections, repetitive)
- _Timing:_ Load 0.52s; Gen 3.09s; Total 3.82s
- _Throughput:_ Prompt 7,594 TPS (575 tok); Gen 181 TPS (500 tok)
- _Tokens:_ prompt 575 tok; estimated text 444 tok; estimated non-text 131
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> &#45; Classic sailboat, dark hull, wooden mast, moored in estuary, low tide,
> green mudflats, flags, calm water, exposed mudflats, flags, boat, water,
> tide, flags, boat, water, mudflats, flags, boat, water, mudflats, flags,
> boat, water, mudflats, flags, boat, water, mudflats, flags, boat, water,
> mudflats, flags, boat, water, mudflats, flags, boat, water, mudflats, flags,
> boat, water, mudflats, flags, boat, water, mudflats, flags, boat, water,
> mudflats, flags, boat, water, mudflats, flags, boat, water, mudflats, flags,
> boat, water, mudflats, flags, boat, water, mudflats, flags, boat, water,
> mudflats, flags, boat, water, mudflats, flags, boat, water, mudflats, flags,
> boat, water, mudflats, flags, boat, water, mudflats, flags, boat, water,
> mudflats, flags, boat, water, mudflats, flags, boat, water, mudflats, flags,
> boat, water, mudflats, flags, boat, water, mudflats, flags, boat, water,
> mudflats, flags, boat, water, mudflats, flags, boat, water, mudflats, flags,
> boat, water, mudflats, flags, boat, water, mudflats, flags, boat, water,
> mudflats, flags, boat, water, mudflats, flags, boat, water, mudflats, flags,
> boat, water, mudflats, flags, boat, water, mudflats, flags, boat, water,
> mudflats, flags, boat, water, mudflats, flags, boat, water, mudflats, flags,
> boat, water, mudflats, flags, boat, water, mudflats, flags, boat, water,
> mudflats, flags, boat, water, mudflats, flags, boat, water, mudflats, flags,
> boat, water, mudflats, flags, boat, water, mudflats, flags, boat, water,
> mudflats, flags, boat, water
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "mudflats, flags, boat, water,...")
- Missing sections (description, keywords)
- Title length violation (205 words; expected 5-10)
- Likely capped by max token budget

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: during, receded,
  exposing, vast, expanse
- _Score:_ ✅ B (74/100); Δ+12; Limited novel information
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.68s; Gen 4.00s; Total 5.90s
- _Throughput:_ Prompt 2,830 TPS (3,064 tok); Gen 34.1 TPS (87 tok)
- _Tokens:_ prompt 3064 tok; estimated text 444 tok; estimated non-text 2620
  tok; generated 87 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Classic sailboat at low tide
>
> Description: A classic-style sailboat with a dark hull and wooden mast is
> moored in a calm estuary, with green algae-covered mudflats visible behind
> it. The boat is adorned with a string of small flags.
>
> Keywords: sailboat, estuary, low tide, dark hull, wooden mast, flags, calm
> water, algae, mudflats, green, moored
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: classic, style, sailboat, dark, hull; nonvisual
  metadata reused
- _Score:_ ❌ F (30/100); Δ-33; Lacks visual description of image
- _Review focus:_ watchlist (context ignored, cutoff, metadata borrowing,
  missing sections, repetitive)
- _Timing:_ Load 1.43s; Gen 4.76s; Total 6.40s
- _Throughput:_ Prompt 2,456 TPS (774 tok); Gen 121 TPS (500 tok)
- _Tokens:_ prompt 774 tok; estimated text 444 tok; estimated non-text 330
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> &#45; 1.317741 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45
> 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45
> 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45
> 18:33:45 18:33:45 18:33:45 18:33:45 18:33:45 18:33:46 18:33:46 18:33:46
> 18:33:46 18:33:46 18:33:46 18:33:46 18:33:46 18:33:46 18:33:46 18:33:46
> 18:33:46 18:33:46 18:33:46 18:33:46 18:33:46 18:36:46 18:36:46 18:36:46
> 18:36:46 18:36:46 18:36:46 18:36:46 18:36:46 18:36:46 18:36:46 18
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "18:33:45 18:33:45 18:33:45 18:...")
- Context ignored (missing: classic, style, sailboat, dark, hull)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: style, during, receded, exposing, vast
- _Score:_ 🏆 A (80/100); Δ+18; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.55s; Gen 5.16s; Total 7.93s
- _Throughput:_ Prompt 513 TPS (784 tok); Gen 26.7 TPS (88 tok)
- _Tokens:_ prompt 784 tok; estimated text 444 tok; estimated non-text 340
  tok; generated 88 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Classic sailboat moored in a calm estuary at low tide
>
> Description: A sailboat with a dark hull and wooden mast floats in shallow
> water. Green, algae-covered mudflats and a wooded shoreline are visible in
> the background.
>
> Keywords: sailboat, boat, estuary, low tide, mudflats, algae, wooden mast,
> dark hull, orange buoy, flags, water, trees, forest, nature, outdoor,
> daylight
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: classic, style, dark, hull, wooden
- _Score:_ ❌ F (18/100); Δ-44; Output lacks detail
- _Review focus:_ watchlist (missing sections, trusted hint degraded)
- _Timing:_ Load 2.17s; Gen 5.35s; Total 7.74s
- _Throughput:_ Prompt 264 TPS (484 tok); Gen 5.3 TPS (17 tok)
- _Tokens:_ prompt 484 tok; estimated text 444 tok; estimated non-text 40 tok;
  generated 17 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Sailboat in a Calm Estuary with Low Tide and Green Algae
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Degrades trusted hints

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=89%; missing terms: receded, exposing,
  vast, expanse, behind
- _Score:_ 🏆 A (83/100); Δ+20; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.30s; Gen 5.47s; Total 7.00s
- _Throughput:_ Prompt 1,235 TPS (4,115 tok); Gen 64.4 TPS (114 tok)
- _Tokens:_ prompt 4115 tok; estimated text 444 tok; estimated non-text 3671
  tok; generated 114 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailboat on Mudflats with Low Tide
>
> &#8203;Description:
> A classic wooden sailboat with a dark hull and red-brown mast is moored in
> shallow water during low tide, surrounded by exposed green algae-covered
> mudflats. The boat features a green cover over the cockpit and a string of
> small flags along the mast.
>
> &#8203;Keywords:
> sailboat, wooden mast, dark hull, mudflats, low tide, green algae, shallow
> water, green cockpit cover, small flags, estuary, calm water, red-brown
> mast, classic-style boat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: classic, style, wooden, during, receded;
  keywords=19
- _Score:_ 🏆 A (82/100); Δ+19; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.43s; Gen 5.51s; Total 8.18s
- _Throughput:_ Prompt 579 TPS (783 tok); Gen 29.5 TPS (113 tok)
- _Tokens:_ prompt 783 tok; estimated text 444 tok; estimated non-text 339
  tok; generated 113 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Dark-hulled sailboat moored on tidal mudflats
>
> &#8203;Description:
> A dark-hulled sailboat with a green sail is moored in calm water, with
> extensive green mudflats exposed by low tide. A string of small flags is
> visible along the stern of the boat, and an orange buoy floats nearby.
>
> &#8203;Keywords:
> sailboat, boat, dark hull, green sail, mast, flags, buoy, orange, water,
> mudflats, low tide, calm, estuary, trees, shoreline, outdoor, daytime,
> reflection, rigging
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=87%; context echo=65%;
  formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Score:_ 🟡 C (58/100); Δ-5; Limited novel information
- _Review focus:_ watchlist (context echo)
- _Timing:_ Load 1.97s; Gen 5.57s; Total 7.75s
- _Throughput:_ Prompt 2,376 TPS (3,507 tok); Gen 32.3 TPS (117 tok)
- _Tokens:_ prompt 3507 tok; estimated text 444 tok; estimated non-text 3063
  tok; generated 117 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Sailboat Moored in Calm Estuary Waters.
>
> Description: A classic-style sailboat with a dark hull and wooden mast is
> moored in a calm estuary during low tide. The water has receded, exposing a
> vast expanse of green, algae-covered mudflats behind the vessel. The boat,
> adorned with a string of small flags, floats peacefully, waiting for the
> tide to rise again.
>
> Keywords: sailboat, estuary, low tide, mudflats, flags, wooden mast, dark
> hull, calm waters, waiting, classic-style.<end_of_utterance>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Context echo (65% overlap)
- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=89%; missing terms: style, during,
  receded, exposing, vast
- _Score:_ 🏆 A (86/100); Δ+23; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.37s; Gen 5.78s; Total 7.37s
- _Throughput:_ Prompt 1,215 TPS (4,115 tok); Gen 59.1 TPS (118 tok)
- _Tokens:_ prompt 4115 tok; estimated text 444 tok; estimated non-text 3671
  tok; generated 118 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailboat Moored on Green Mudflats at Low Tide
>
> &#8203;Description:
> A classic wooden sailboat, featuring a dark hull and wooden mast, rests on
> shallow water near exposed green algae-covered mudflats. The boat displays a
> series of small flags along its boom, anchored close to an orange buoy.
>
> &#8203;Keywords:
> sailboat, wooden mast, dark hull, green mudflats, shallow water, orange
> buoy, small flags, classic wooden boat, calm estuary, low tide,
> algae-covered, wooden hull, moored vessel, calm water, wooden structure
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords
- _Score:_ ✅ B (74/100); Δ+11; Missing requested structure
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 2.23s; Gen 6.45s; Total 8.90s
- _Throughput:_ Prompt 1,789 TPS (782 tok); Gen 46.6 TPS (266 tok)
- _Tokens:_ prompt 782 tok; estimated text 444 tok; estimated non-text 338
  tok; generated 266 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;# Image Description:
>
> A classic-style sailboat with a dark hull and wooden mast is moored in a
> calm estuary during low tide. The water has receded, exposing a vast expanse
> of green, algae-covered mudflats behind the vessel. The boat, adorned with
> a string of small flags, floats peacefully, waiting for the tide to rise
> again.
>
> The boat has a green sail furled and a red boom extending outwards. A small
> orange buoy is attached to the stern. The mast is made of wood and has
> several rigging lines running from it. The hull is dark, possibly black or
> dark blue, and has a wooden deck.
>
> The mudflats are a vibrant green, indicating the presence of algae. In the
> background, a line of trees with light green foliage can be seen along the
> horizon. The sky is overcast, providing soft, diffused light.
>
> The overall scene is tranquil and picturesque, capturing a moment of
> stillness in a natural setting.
>
> &#35;# Keywords:
>
> &#45; sailboat
> &#45; estuary
> &#45; low tide
> &#45; mudflats
> &#45; green
> &#45; algae
> &#45; boat
> &#45; mast
> &#45; sail
> &#45; flags
> &#45; buoy
> &#45; trees
> &#45; overcast
> &#45; tranquil
> &#45; picturesque
> &#45; nature
> &#45; water
> &#45; boat mooring
> &#45; coastal scene
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=86%; missing terms: floats, peacefully,
  waiting, rise, again; context echo=46%
- _Score:_ 🟡 C (64/100); Δ+1; Limited novel information
- _Review focus:_ watchlist (context echo)
- _Timing:_ Load 1.78s; Gen 6.45s; Total 8.46s
- _Throughput:_ Prompt 1,342 TPS (3,064 tok); Gen 31.8 TPS (120 tok)
- _Tokens:_ prompt 3064 tok; estimated text 444 tok; estimated non-text 2620
  tok; generated 120 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailboat Moored at Low Tide
>
> &#8203;Description:
> A classic-style sailboat with a dark hull and wooden mast is moored in a
> calm estuary during low tide. The water has receded, exposing a vast expanse
> of green, algae-covered mudflats behind the vessel. The boat is adorned
> with a string of small flags.
>
> &#8203;Keywords:
> sailboat, dark hull, wooden mast, moored, estuary, low tide, mudflats, green
> algae, flags, calm water, classic style, nature, outdoors, maritime, boat,
> mast, rigging
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Context echo (46% overlap)

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=90%; context echo=58%
- _Score:_ 🟡 C (56/100); Δ-6; Limited novel information
- _Review focus:_ watchlist (context echo)
- _Timing:_ Load 1.65s; Gen 6.78s; Total 8.65s
- _Throughput:_ Prompt 1,620 TPS (4,662 tok); Gen 38.2 TPS (135 tok)
- _Tokens:_ prompt 4662 tok; estimated text 444 tok; estimated non-text 4218
  tok; generated 135 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailboat in Estuary at Low Tide
>
> &#8203;Description:
> A classic-style sailboat with a dark hull and wooden mast is moored in a
> calm estuary during low tide. The water has receded, exposing a vast expanse
> of green, algae-covered mudflats behind the vessel. The boat, adorned with
> a string of small flags, floats peacefully, waiting for the tide to rise
> again.
>
> &#8203;Keywords:
> sailboat, estuary, low tide, wooden mast, flags, algae-covered mudflats,
> calm water, classic-style, moored, green algae, wooden hull, string of
> flags, exposed mud, waiting for tide
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Context echo (58% overlap)

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: style; context echo=96%
- _Score:_ 🟠 D (46/100); Δ-17; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context echo, missing sections)
- _Timing:_ Load 1.50s; Gen 7.43s; Total 9.13s
- _Throughput:_ Prompt 295 TPS (483 tok); Gen 21.7 TPS (119 tok)
- _Tokens:_ prompt 483 tok; estimated text 444 tok; estimated non-text 39 tok;
  generated 119 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title
> Classic Sailboat Moored in Calm Estuary
>
> Description
> A classic sailboat with a dark hull and wooden mast is moored in a calm
> estuary during low tide. The water has receded, exposing a vast expanse of
> green, algae-covered mudflats behind the vessel. The boat, adorned with a
> string of small flags, floats peacefully, waiting for the tide to rise
> again.
>
> Keywords
> Sailboat, Estuary, Low Tide, Mudflats, Flags, Classic Boat, Calm Water,
> Green Algae, Wooden Mast, Dark Hull.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Context echo (96% overlap)

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=71%; missing sections: title,
  description, keywords; missing terms: classic, style, sailboat, dark, hull
- _Score:_ ❌ F (20/100); Δ-43; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 2.44s; Gen 7.54s; Total 10.19s
- _Throughput:_ Prompt 1,058 TPS (1,538 tok); Gen 5.41 TPS (31 tok)
- _Tokens:_ prompt 1538 tok; estimated text 444 tok; estimated non-text 1094
  tok; generated 31 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Camera metadata: 1/250s, f/5.6, 200 ISO, 24mm lens.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: classic, style, sailboat, dark, hull)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: classic, style, wooden, estuary, receded
- _Score:_ 🏆 A (86/100); Δ+23; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.49s; Gen 7.80s; Total 11.54s
- _Throughput:_ Prompt 533 TPS (783 tok); Gen 17.3 TPS (104 tok)
- _Tokens:_ prompt 783 tok; estimated text 444 tok; estimated non-text 339
  tok; generated 104 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Dark-hulled sailboat moored in mudflats
>
> &#8203;Description:
> A dark-hulled sailboat with a green sail is moored in a calm waterway during
> low tide. A string of small flags is visible along the boat’s rigging, and
> a large orange buoy floats nearby.
>
> &#8203;Keywords:
> sailboat, boat, waterway, mudflats, low tide, green, dark hull, flags, buoy,
> orange, mast, rigging, calm, water, exterior, side view, moored, reflection
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=71%; missing sections: title; missing
  terms: during, receded, vast, expanse, algae; keywords=45
- _Score:_ ✅ B (70/100); Δ+7; None identified
- _Review focus:_ watchlist (missing sections, reasoning leak)
- _Timing:_ Load 2.07s; Gen 8.60s; Total 10.87s
- _Throughput:_ Prompt 1,597 TPS (1,516 tok); Gen 67.7 TPS (490 tok)
- _Tokens:_ prompt 1516 tok; estimated text 444 tok; estimated non-text 1072
  tok; generated 490 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image based on the given instructions.
> First, the title needs to be 5-10 words, concrete and factual. The main
> subject is a sailboat. Let's check the visible details: dark hull, wooden
> mast, flags, calm water, mudflats, trees in background. So the title could
> be "Classic sailboat with flags moored in estuary at low tide."
>
> For the description, 1-2 sentences: A classic-style sailboat with a dark
> hull and wooden mast is moored in calm water, with a string of small flags,
> against a backdrop of green mudflats and trees. The water is calm, and the
> tide is low, exposing mudflats behind the boat.
>
> Keywords: classic sailboat, dark hull, wooden mast, flags, calm water,
> mudflats, green landscape, trees, estuary, low tide, moored, traditional
> design.
>
> &#8203;Wait, need to make sure all keywords are clearly visible. Let's check each:
> classic sailboat (visible), dark hull (yes), wooden mast (yes), flags (yes,
> the small colorful ones), calm water (yes), mudflats (yes, the green area
> behind), green landscape (the grassy area), trees (in background), estuary
> (the setting), low tide (implied by mudflats), moored (the boat is
> stationary), traditional design (the classic style). That's 10 keywords. Let
> me count: classic sailboat, dark hull, wooden mast, flags, calm water,
> mudflats, green landscape, trees, estuary, low tide, moored. That's 10. Need
> to make sure no uncertain terms. The description matches the context hint,
> but the image shows the boat is moored, flags, dark hull, wooden mast,
> mudflats, trees, calm water. So the title and description should be based on
> what's visible.◁/think▷Title:
> Classic sailboat with flags moored in estuary at low tide
>
> &#8203;Description:
> A classic-style sailboat with a dark hull and wooden mast is moored in calm
> water, adorned with a string of small flags, against a backdrop of green
> mudflats and trees.
>
> &#8203;Keywords:
> classic sailboat, dark hull, wooden mast, flags, calm water, mudflats, green
> landscape, trees, estuary, low tide, moored<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Keyword count violation (45; expected 10-18)
- Reasoning leak (◁think▷, ◁/think▷)

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=90%; context echo=56%
- _Score:_ 🟡 C (59/100); Δ-3; Limited novel information
- _Review focus:_ watchlist (context echo)
- _Timing:_ Load 2.55s; Gen 9.38s; Total 12.15s
- _Throughput:_ Prompt 1,908 TPS (4,662 tok); Gen 21.0 TPS (138 tok)
- _Tokens:_ prompt 4662 tok; estimated text 444 tok; estimated non-text 4218
  tok; generated 138 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailboat in Estuary at Low Tide
>
> &#8203;Description:
> A classic-style sailboat with a dark hull and wooden mast is moored in a
> calm estuary during low tide. The water has receded, exposing a vast expanse
> of green, algae-covered mudflats behind the vessel. The boat, adorned with
> a string of small flags, floats peacefully, waiting for the tide to rise
> again.
>
> &#8203;Keywords:
> sailboat, estuary, low tide, wooden mast, dark hull, flags, algae-covered
> mudflats, calm water, classic-style, moored, green algae, wooden boat,
> string of flags, exposed mud, waiting for tide
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Context echo (56% overlap)

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 61 occurrences).; nontext prompt burden=88%; missing sections:
  description, keywords; missing terms: vast, expanse, adorned, small, floats
- _Score:_ ❌ F (4/100); Δ-59; Output too short to be useful
- _Review focus:_ watchlist (harness, missing sections)
- _Timing:_ Load 2.11s; Gen 9.44s; Total 11.82s
- _Throughput:_ Prompt 663 TPS (3,619 tok); Gen 30.1 TPS (108 tok)
- _Tokens:_ prompt 3619 tok; estimated text 444 tok; estimated non-text 3175
  tok; generated 108 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title:ĠClassicĠsailboatĠmooredĠinĠestuaryĊĊDescription:ĠAĠclassic-styleĠsailboatĠwithĠaĠdarkĠhullĠandĠwoodenĠmastĠisĠmooredĠinĠaĠcalmĠestuaryĠduringĠlowĠtide.ĠTheĠwaterĠhasĠreceded,ĠexposingĠgreen,Ġalgae-coveredĠmudflatsĠbehindĠtheĠvessel.ĊĊKeywords:Ġsailboat,ĠwoodenĠmast,ĠdarkĠhull,Ġestuary,ĠlowĠtide,Ġmudflats,ĠgreenĠalgae,Ġmoored,ĠcalmĠwater,ĠstringĠofĠflags,Ġclassic-style,ĠwoodenĠboat,Ġwatercraft,Ġnature,Ġoutdoor,Ġcalm,ĠmooringĠbuoy
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Missing sections (description, keywords)
- Title length violation (66 words; expected 5-10)
- token_encoding:bpe_space_leak(61)

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=67%; missing
  terms: boat, adorned, string, small, floats; keyword duplication=90%
- _Score:_ 🟠 D (46/100); Δ-17; Lacks visual description of image
- _Review focus:_ watchlist (context echo, cutoff, repetitive)
- _Timing:_ Load 0.88s; Gen 9.58s; Total 10.67s
- _Throughput:_ Prompt 3,814 TPS (1,337 tok); Gen 55.9 TPS (500 tok)
- _Tokens:_ prompt 1337 tok; estimated text 444 tok; estimated non-text 893
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Classic Sailboat in Estuary
>
> Description: A classic-style sailboat with a dark hull and wooden mast is
> moored in a calm estuary during low tide. The water has receded, exposing a
> vast expanse of green, algae-covered mudflats behind the vessel.
>
> Keywords: sailboat, estuary, low tide, calm, green, algae, mudflats, wooden
> mast, dark hull, flags, receded water, waiting, tide, calm water, green
> algae, exposed mud, wooden sailboat, classic style, moored, peaceful, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "flags, flags, flags, flags,...")
- Title length violation (4 words; expected 5-10)
- Keyword count violation (198; expected 10-18)
- Keyword duplication (90% duplicated terms)
- Context echo (41% overlap)
- Likely capped by max token budget

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=67%; missing
  terms: boat, adorned, string, small, floats; keyword duplication=90%
- _Score:_ 🟠 D (46/100); Δ-17; Lacks visual description of image
- _Review focus:_ watchlist (context echo, cutoff, repetitive)
- _Timing:_ Load 0.84s; Gen 9.69s; Total 10.75s
- _Throughput:_ Prompt 3,868 TPS (1,337 tok); Gen 55.2 TPS (500 tok)
- _Tokens:_ prompt 1337 tok; estimated text 444 tok; estimated non-text 893
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Classic Sailboat in Estuary
>
> Description: A classic-style sailboat with a dark hull and wooden mast is
> moored in a calm estuary during low tide. The water has receded, exposing a
> vast expanse of green, algae-covered mudflats behind the vessel.
>
> Keywords: sailboat, estuary, low tide, calm, green, algae, mudflats, wooden
> mast, dark hull, flags, receded water, waiting, tide, calm water, green
> algae, exposed mud, wooden sailboat, classic style, moored, peaceful, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags, flags, flags,
> flags, flags, flags, flags, flags, flags, flags, flags, flags
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "flags, flags, flags, flags,...")
- Title length violation (4 words; expected 5-10)
- Keyword count violation (198; expected 10-18)
- Keyword duplication (90% duplicated terms)
- Context echo (41% overlap)
- Likely capped by max token budget

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=75%; missing
  sections: title, description; missing terms: boat, floats, peacefully,
  waiting, rise
- _Score:_ 🟡 C (59/100); Δ-4; Missing requested structure
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections)
- _Timing:_ Load 5.92s; Gen 13.05s; Total 19.40s
- _Throughput:_ Prompt 687 TPS (1,810 tok); Gen 51.0 TPS (500 tok)
- _Tokens:_ prompt 1810 tok; estimated text 444 tok; estimated non-text 1366
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Let's tackle this step by step. First, the title needs to be 5-10 words,
> concrete and factual. The main subject is a classic sailboat (sailboat)
> moored in a calm estuary (estuary) during low tide (low tide) with green
> &#8203;mudflats (mudflats) behind it. Wait, but let's check the image description:
> the user mentioned a "classic-style sailboat with a dark hull and wooden
> mast" moored in a calm estuary during low tide, water receded exposing green
> algae-covered mudflats. So title should be something like "Classic Sailboat
> Moored in Estuary Mudflats at Low Tide"—wait, but let's make it more
> concise. Wait the hint says "a classic-style sailboat with a dark hull and
> wooden mast is moored in a calm estuary during low tide. The water has
> receded, exposing a vast expanse of green, algae-covered mudflats behind the
> vessel." So title: "Classic Sailboat Moored in Estuary Mudflats at Low
> Tide"—that's 8 words, good.
>
> Next, description: 1-2 factual sentences. Need to include main subject
> (classic sailboat), setting (calm estuary, low tide, green algae-covered
> mudflats), details (dark hull, wooden mast, flags, moored). Wait the hint
> says "adorned with a string of small flags"—yes, visible. So description: "A
> classic-style sailboat with a dark hull and wooden mast is moored in a calm
> estuary during low tide. The water has receded, exposing green,
> algae-covered mudflats behind the vessel, which is adorned with a string of
> small flags." That's two sentences, factual, uses visible details.
>
> Keywords: 10-18 unique terms. Need to include subjects (sailboat, estuary,
> mudflats, low tide), details (classic, dark hull, wooden mast, flags, calm
> water, algae-covered mud, moored), setting (estuary, mudflats, low tide).
> Let's list: sailboat, estuary, mudflats, low tide, classic, dark hull,
> wooden mast, flags, calm water, algae-covered mud,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description)
- Keyword count violation (23; expected 10-18)
- Keyword duplication (39% duplicated terms)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: classic, style,
  sailboat, dark, hull
- _Score:_ 🟠 D (49/100); Δ-13; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.26s; Gen 13.79s; Total 15.28s
- _Throughput:_ Prompt 3,505 TPS (4,610 tok); Gen 42.0 TPS (500 tok)
- _Tokens:_ prompt 4610 tok; estimated text 444 tok; estimated non-text 4166
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the description
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- output only the...")
- Excessive bullet points (46)
- Context ignored (missing: classic, style, sailboat, dark, hull)
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
  sections: title, keywords; missing terms: style, vast, expanse, floats,
  waiting
- _Score:_ 🟡 C (57/100); Δ-6; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, missing sections, reasoning leak)
- _Timing:_ Load 1.32s; Gen 13.90s; Total 15.44s
- _Throughput:_ Prompt 1,053 TPS (6,570 tok); Gen 68.1 TPS (500 tok)
- _Tokens:_ prompt 6570 tok; estimated text 444 tok; estimated non-text 6126
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's tackle this. First, I need to analyze the image for
> cataloguing metadata. The user provided a context with hints, but I need to
> focus on what's clearly visible.
>
> First, the Title. It needs to be 5-10 words, concrete and factual. Looking
> at the image: a sailboat, moored, estuary, low tide. So a title like
> "Sailboat Moored in Low Tide Estuary" - that's concise and factual.
>
> Next, the Description. It needs 1-2 sentences, factual, no inferences.
> Looking at the image: a classic sailboat with dark hull and wooden mast,
> moored in a calm estuary during low tide. The water has receded, exposing
> green, algae-covered mudflats behind the vessel. The boat is adorned with a
> string of small flags, floating peacefully. So I need to condense that into
> 1-2 sentences, keeping only clearly visible details. For example: "A classic
> sailboat with a dark hull and wooden mast is moored in a calm estuary
> during low tide. The water has receded, exposing green, algae-covered
> mudflats behind the vessel. The boat is adorned with a string of small
> flags, floating peacefully." Wait, but the instruction says 1-2 factual
> sentences. So I need to make sure it's concise. Maybe: "A classic sailboat
> with a dark hull and wooden mast is moored in a calm estuary during low
> tide. The water has receded, exposing green, algae-covered mudflats behind
> the vessel. The boat is adorned with a string of small flags, floating
> peacefully." But maybe too long. Let's check the image again. The key
> details: sailboat, dark hull, wooden mast, moored, estuary, low tide, green
> algae-covered mudflats behind, string of small flags (maybe Union Jack or
> similar small flags), orange buoy attached to the boat with a rope, the boat
> has a green canvas cover on the deck, a small window on the side, a ladder
> on the stern.
>
> Now, for the Title: need 5-10 words. "Sailboat Moored in Low Tide Estuary" -
> that's 7 words, good.
>
> Description: 1-2 sentences. Let's draft: "A classic sailboat with a dark
> hull and wooden mast is moored in a calm estuary during low tide. The water
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, keywords)
- Description sentence violation (3; expected 1-2)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: sailboat, estuary, during, exposing,
  vast
- _Score:_ 🟡 C (57/100); Δ-5; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 1.46s; Gen 13.92s; Total 15.61s
- _Throughput:_ Prompt 152 TPS (1,468 tok); Gen 50.6 TPS (190 tok)
- _Tokens:_ prompt 1468 tok; estimated text 444 tok; estimated non-text 1024
  tok; generated 190 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Sun position: Behind photographer, casting long shadows across the
> mudflats. Tide status: Low tide, with water level significantly receded.
> Season: Summer, based on vegetation and lighting. Location: East Anglia, UK,
> &#8203;likely near the River Orwell or another estuarine area. Boat type:
> &#8203;Classic-style sailing vessel, possibly a yawl or small ketch. Boat color:
> Dark hull with lighter accents. Mast: Wooden, with rigging visible. Flags:
> Small string of flags on the boat. Mudflats: Extensive, covered in green
> algae. Vegetation: Trees visible in the background. Water condition: Calm,
> with gentle ripples. Boat position: Moored, pointing slightly to the right.
> Photographer: Standing on the shore, capturing the scene. Image style:
> Photographic representationalism, realism. Lighting: Even, with soft
> shadows. Focus: Sharp, emphasizing the boat and mudflats.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: during, receded,
  exposing, vast, expanse
- _Score:_ 🟡 C (64/100); Δ+2; Missing requested structure
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections,
  reasoning leak)
- _Timing:_ Load 1.45s; Gen 15.63s; Total 17.30s
- _Throughput:_ Prompt 1,111 TPS (6,570 tok); Gen 53.2 TPS (500 tok)
- _Tokens:_ prompt 6570 tok; estimated text 444 tok; estimated non-text 6126
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, I need to extract the
> Title, Description, and Keywords based on the image.
>
> &#8203;Starting with the Title. The main subject is a sailboat. The key details:
> classic-style, dark hull, wooden mast. So the title should be concise,
> factual. Maybe "Classic sailboat with dark hull and wooden mast moored in
> estuary."
>
> &#8203;Next, Description. Need to use only clearly visible details. From the image:
> sailboat with dark hull, wooden mast, green canvas cover on deck, small
> flags (Union Jacks?) strung along. Moored in calm water, with green
> algae-covered mudflats behind, and trees in background. Low tide, so
> mudflats exposed. The boat is floating peacefully.
>
> So the description should be factual, no inferences. Let's draft: "A
> classic-style sailboat with a dark hull and wooden mast is moored in a calm
> estuary. The boat features a green canvas cover on the deck, a string of
> small flags, and is tied to an orange buoy. Behind the vessel, green
> algae-covered mudflats are exposed by low tide, with trees and grassy banks
> in the background."
>
> Now Keywords. Need 10-18 unique terms based on visible content. Let's list
> &#8203;visible elements:
>
> &#45; Sailboat
> &#45; Dark hull
> &#45; Wooden mast
> &#45; Green canvas cover
> &#45; Small flags (maybe Union Jacks, but we can't confirm the flag type, so
> just "small flags")
> &#45; Orange buoy
> &#45; Calm water
> &#45; Estuary
> &#45; Low tide
> &#45; Green algae-covered mudflats
> &#45; Trees
> &#45; Grassy banks
>
> &#8203;Now select unique terms that are clearly visible. Let's pick:
>
> &#45; Sailboat
> &#45; Dark hull
> &#45; Wooden mast
> &#45; Green canvas cover
> &#45; Small flags
> &#45; Orange buoy
> &#45; Calm water
> &#45; Estuary
> &#45; Low tide
> &#45; Green algae-covered mudflats
> &#45; Trees
> &#45; Grassy banks
>
> That's more than 10-18. Need to trim to unique, clearly visible. Let's check
> &#8203;which are definitely visible:
>
> &#45; Sailboat: yes
> &#45; Dark hull: yes
> &#45; Wooden mast: yes
> &#45; Green canvas cover: yes
> &#45; Small flags: yes
> &#45; Orange buoy: yes
> &#45; Calm water: yes
> &#45; Estuary: yes
> &#45; Low tide: yes
> &#45; Green algae-covered mudflats: yes
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Excessive bullet points (34)
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=91%; missing
  sections: title, description, keywords; missing terms: classic, style,
  receded, vast, expanse
- _Score:_ ✅ B (74/100); Δ+11; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, missing sections, reasoning leak)
- _Timing:_ Load 1.56s; Gen 15.95s; Total 17.74s
- _Throughput:_ Prompt 1,550 TPS (4,753 tok); Gen 39.9 TPS (500 tok)
- _Tokens:_ prompt 4753 tok; estimated text 444 tok; estimated non-text 4309
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here are my reasoning steps:
> We need to produce three sections: Title, Description, Keywords. Use only
> details that are clearly visible. The image shows a small sailboat with a
> dark hull (black) and wooden mast (red? Actually the mast appears to be a
> wooden pole with a red stripe? The mast is a tall wooden pole, maybe painted
> red at the top? The image shows a red mast? Let's examine: The mast is a
> tall vertical pole, appears to be a light brown wood, maybe with a red
> stripe near the top. The boat has a green canvas cover (green awnings) over
> the cockpit. There are small flags strung along the rigging: red, white,
> blue, maybe also a yellow? The boat is moored with an orange buoy attached
> to a rope. The water is shallow, exposing green algae-covered mudflats. The
> background is a grassy bank with trees. The lighting appears to be soft,
> maybe late afternoon. The boat is anchored, not moving. The boat has a
> registration number "1984"? Actually there is a number on the side: "1984"?
> Let's check: The side of the boat near the stern shows a number "1984"? The
> image shows a small rectangular plate with "1984"? It's not clear. There's a
> small white sign with a number "1984"? Actually the image shows a small
> white plate with "1984"? Let's examine: The boat's hull has a small white
> rectangular plate near the stern, with the number "1984"? It's not clear.
> There's also a small orange buoy attached to the bow. The boat has a wooden
> deck? The deck appears to be light wood. The boat has a small cabin? The
> boat has a small cockpit with a green cover. The boat's hull is dark (black)
> with a wooden trim (red?). The boat's name is not visible. The boat is
> anchored with a rope to the orange buoy. The water is calm, reflecting the
> boat. The background is a grassy bank with trees, some of which are leafless
> (maybe early spring). The water is shallow, exposing green algae-covered
> mudflats. The boat is moored in a calm estuary during low tide. The image
> shows a string of small flags (red, white, blue) strung along the rigging.
> The
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Reasoning leak (here are my reasoning steps)
- Likely capped by max token budget

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: classic, style, wooden, moored,
  estuary
- _Score:_ 🟡 C (58/100); Δ-5; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 1.81s; Gen 16.16s; Total 18.20s
- _Throughput:_ Prompt 143 TPS (1,468 tok); Gen 30.6 TPS (163 tok)
- _Tokens:_ prompt 1468 tok; estimated text 444 tok; estimated non-text 1024
  tok; generated 163 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Sun position: Behind photographer, casting long shadows across the
> mudflats. Tide stage: Low tide, with extensive mudflats visible. Water
> conditions: Calm with slight ripples. Vegetation: Dense trees and shrubs
> visible in the background. Marine life: None visible in the frame. Boat
> type: Sailboat. Hull color: Dark (possibly black or deep blue). Mast
> material: Wood. Flag details: Small flags attached to the boat. Mudflats:
> Extensive, covered in green algae. Lighting: Even, natural daylight. Season:
> Summer (based on vegetation and lighting). photographer: Not visible in the
> frame. Boat position: Centered in the frame, facing right. Water clarity:
> Clear enough to see mud beneath. Boat size: Small to medium (sailboat).
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: style, vast, expanse, peacefully, waiting; keyword duplication=62%
- _Score:_ ✅ B (78/100); Δ+15; None identified
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 1.29s; Gen 27.59s; Total 29.09s
- _Throughput:_ Prompt 954 TPS (16,901 tok); Gen 53.7 TPS (500 tok)
- _Tokens:_ prompt 16901 tok; estimated text 444 tok; estimated non-text 16457
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> &#45; Classic sailboat at low tide
>
> &#8203;Description:
> &#45; A dark-hulled sailboat with a wooden mast is moored in a calm estuary
> during low tide. The water has receded, exposing green, algae-covered
> mudflats behind the vessel. The boat is adorned with a string of small
> flags, and a red buoy floats nearby.
>
> &#8203;Keywords:
> &#45; Sailboat, low tide, estuary, mudflats, green algae, wooden mast, flags,
> red buoy, calm water, classic design, moored, British flag, nautical,
> maritime, serene, nature, landscape, mooring line, anchor, reflection, water
> surface, forested background, overcast sky, British flag, moored boat,
> tide, estuary, sailboat, flags, buoy, dark hull, wooden structure, calm, low
> tide, green mudflats, algae, forest, trees, water, boat, sail, mast, ropes,
> anchor chain, mooring post, red buoy, green algae, forested shore,
> overcast, still water, reflection, nautical scene, classic boat, moored
> vessel, low tide scene, estuary landscape, sailboat at anchor, flags on
> boat, red buoy, green mudflats, forested background, overcast sky, calm
> water, classic design, moored sailboat, low tide estuary, green algae,
> forested shore, red buoy, calm water, boat, sail, mast, ropes, anchor chain,
> mooring post, red buoy, green algae, forested background, overcast sky,
> calm water, classic boat, moored sailboat, low tide estuary, green algae,
> forested shore, red buoy, calm water, boat, sail, mast, ropes, anchor chain,
> mooring post, red buoy, green algae, forested background, overcast sky,
> calm water, classic boat, moored sailboat, low tide estuary, green algae,
> forested shore, red buoy, calm water, boat, sail, mast, ropes, anchor chain,
> mooring post, red buoy, green algae, forested background, overcast sky,
> calm water, classic boat, moored sailboat, low tide estuary, green algae,
> forested shore, red buoy, calm water, boat, sail, mast, ropes, anchor chain,
> mooring post, red buoy, green algae, forested background, overcast sky,
> calm water, classic boat, moored
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Keyword count violation (137; expected 10-18)
- Keyword duplication (62% duplicated terms)
- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=71%; missing
  sections: title, description, keywords; missing terms: classic, style,
  sailboat, dark, hull
- _Score:_ 🟠 D (44/100); Δ-19; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.49s; Gen 27.81s; Total 29.52s
- _Throughput:_ Prompt 3,163 TPS (1,538 tok); Gen 18.6 TPS (500 tok)
- _Tokens:_ prompt 1538 tok; estimated text 444 tok; estimated non-text 1094
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; The image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio.
> The image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4:3 aspect ratio. The
> image is in a 4:3 aspect ratio. The image is in a 4
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the image is in...")
- Context ignored (missing: classic, style, sailboat, dark, hull)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: vast, expanse,
  behind, vessel, adorned
- _Score:_ 🟠 D (49/100); Δ-14; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections,
  reasoning leak)
- _Timing:_ Load 7.10s; Gen 28.30s; Total 35.62s
- _Throughput:_ Prompt 465 TPS (6,570 tok); Gen 36.1 TPS (500 tok)
- _Tokens:_ prompt 6570 tok; estimated text 444 tok; estimated non-text 6126
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, the Title. It needs to
> be 5-10 words, concrete and factual. The main subject is a sailboat, so
> maybe "Classic sailboat moored in calm estuary at low tide." Wait, does the
> image show low tide? Yes, because there's mudflats exposed. Let's check: the
> image has a sailboat, moored, estuary, low tide. That's good.
>
> Next, the Description. It needs 1-2 factual sentences. The existing hint
> says a classic-style sailboat with dark hull and wooden mast, moored in calm
> estuary during low tide, water receded exposing green algae-covered
> mudflats, boat with small flags, floats peacefully. Let's confirm: dark hull
> (yes, the boat is black), wooden mast (yes, the mast is brown wood), moored
> (yes, tied to a buoy), calm estuary (yes, water is calm), low tide (yes,
> mudflats are visible), small flags (yes, there's a string of flags), floats
> peacefully (yes, it's stationary). So the description should include those
> details.
>
> Then Keywords. 10-18 unique terms. Let's list visible ones: sailboat,
> moored, estuary, low tide, dark hull, wooden mast, buoy, flags, calm water,
> mudflats, algae, green, classic style, stationary, boat, nautical, natural
> &#8203;setting. Wait, need to make sure they're unique and visible. Let's check:
> sailboat (yes), moored (yes), estuary (yes), low tide (yes), dark hull
> (yes), wooden mast (yes), buoy (yes, the orange one), flags (yes), calm
> water (yes), mudflats (yes), algae (yes, the green stuff), green (yes),
> classic style (yes, the boat looks classic), stationary (yes), boat (yes),
> nautical (yes), natural setting (yes). Let's count: that's 16, which is
> within 10-18.
>
> Now, let's make sure not to include anything uncertain. The existing capture
> metadata has a date and GPS, but the user said to use only details clearly
> visible, so we don't include that. The title should be concise: "Classic
> sailboat moored in calm estuary at low tide." The description: "A
> classic-style sailboat with
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: description, keywords; missing terms: style, during, receded,
  exposing, vast
- _Score:_ ✅ B (73/100); Δ+10; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections)
- _Timing:_ Load 2.46s; Gen 58.49s; Total 61.17s
- _Throughput:_ Prompt 320 TPS (16,916 tok); Gen 101 TPS (500 tok)
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
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
> &#45; &#42;&#42;Subject:&#42;&#42; A small sailboat. It has a dark hull (black or very dark
> blue/grey) and a wooden mast. There is a red bowsprit. The boom is covered
> with a green canvas cover.
> &#45; &#42;&#42;Setting:&#42;&#42; It is floating in water, but behind it is a large expanse of
> green mudflats, indicating low tide. There are trees in the background,
> suggesting a river or estuary.
> &#45; &#42;&#42;Details:&#42;&#42;
> &#45; Orange buoy attached to the bow.
> &#45; String of small flags (bunting) strung from the mast to the bow and
> stern. The flags appear to be Union Jacks and red/white/blue pennants.
> &#45; Wooden cabin structure.
> &#45; Ropes and rigging are visible.
> &#45; Reflections in the water.
> &#45; Trees in the background are green, suggesting spring or summer.
> &#45; &#42;&#42;Lighting:&#42;&#42; Overcast, soft lighting. No harsh shadows.
>
> &#8203;&#42;&#42;Drafting Metadata:&#42;&#42;
>
> &#42;   &#42;&#42;Title:&#42;&#42; Needs to be 5-10 words.
> &#42;   Draft 1: Dark hull sailboat with wooden mast moored in estuary. (8
> words) - Good.
> &#42;   Draft 2: Small sailboat with green boom cover floating on water. (9
> words) - Good.
> &#42;   Draft 3: Classic wooden sailboat with red bowsprit and orange buoy.
> (9 words) - Good.
> &#42;   Selection: Dark hull sailboat with wooden mast moored in estuary.
> (Wait, let's make it more descriptive). "Small sailboat with dark hull and
> wooden mast moored in estuary." (10 words). Let's try: "Sailboat with dark
> hull and wooden mast floating in calm water." (11 words - too long).
> &#42;   Selection: "Small sailboat with dark hull and wooden mast moored in
> estuary." (10 words). Let's count: Small(1) sailboat(2) with(3) dark(4
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (description, keywords)
- Title length violation (118 words; expected 5-10)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: style, calm, during,
  receded, vast
- _Score:_ ✅ B (77/100); Δ+15; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections)
- _Timing:_ Load 3.13s; Gen 65.40s; Total 68.75s
- _Throughput:_ Prompt 290 TPS (16,916 tok); Gen 79.5 TPS (500 tok)
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to generate cataloguing metadata for the provided image.
> &#8203;I need to follow specific rules:
> &#45; British English.
> &#45; Only clearly visible details.
> &#45; No guessing or inferring.
> &#45; Three specific sections: Title, Description, Keywords.
> &#45; Specific constraints on length and content for each section.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Subject:&#42;&#42; A small sailboat. It looks like a classic wooden boat.
> &#45; &#42;&#42;Hull:&#42;&#42; Dark (black or very dark blue/grey) with a white stripe near the
> waterline and a wooden gunwale.
> &#45; &#42;&#42;Mast:&#42;&#42; Wooden, tall, single mast.
> &#45; &#42;&#42;Boom:&#42;&#42; Covered with a green sail cover (or possibly a green canvas
> cover).
> &#45; &#42;&#42;Rigging:&#42;&#42; White ropes/cables. A red bowsprit or jib boom extending
> forward.
> &#45; &#42;&#42;Flags:&#42;&#42; A string of bunting (small triangular flags) strung from the
> mast to the bow/stern area. The flags are red, white, and blue.
> &#45; &#42;&#42;Buoy:&#42;&#42; An orange buoy floating near the bow, attached by a rope.
> &#45; &#42;&#42;Setting:&#42;&#42; A body of water (estuary or river). The tide is low, exposing
> green mudflats covered in algae/seaweed.
> &#45; &#42;&#42;Background:&#42;&#42; A grassy bank with trees. The trees are green, suggesting
> spring or summer.
> &#45; &#42;&#42;Lighting:&#42;&#42; Overcast, diffuse light. No harsh shadows.
> &#45; &#42;&#42;Action:&#42;&#42; Moored, stationary.
>
> &#8203;&#42;&#42;Drafting the Title:&#42;&#42;
> &#45; Needs to be 5-10 words.
> &#45; Concrete and factual.
> &#45; &#42;Draft 1:&#42; A classic wooden sailboat moored in an estuary at low tide. (10
> words) - Good.
> &#45; &#42;Draft 2:&#42; Dark-hulled sailboat with green boom cover resting on mudflats.
> (9 words) - Good.
> &#45; &#42;Draft 3:&#42; Wooden mast sailboat with bunting moored near green mudflats.
> (9 words) - Good.
> &#45; Let's go with something descriptive of the main elements. "Classic wooden
> sailboat moored in an estuary at low tide" is accurate. Or "Dark-hulled
> sailboat with green boom cover resting on mudflats". Let's combine for
> clarity.
> &#45; &#42;Selected Title:&#42; Classic wooden sailboat moored in an estuary at low
> tide.
>
> &#42;&#42;
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16901 tokens), output became
  repetitive.; hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords
- _Score:_ ❌ F (33/100); Δ-30; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, harness, long context,
  missing sections, repetitive)
- _Timing:_ Load 0.55s; Gen 66.38s; Total 67.18s
- _Throughput:_ Prompt 268 TPS (16,901 tok); Gen 192 TPS (500 tok)
- _Tokens:_ prompt 16901 tok; estimated text 444 tok; estimated non-text 16457
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat
> Anchor Boat Anchor Boat Anchor Boat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "boat anchor boat anchor...")
- Context ignored (missing: classic, style, sailboat, dark, hull)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16901tok)

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; reasoning leak
- _Score:_ ✅ B (65/100); Δ+3; Missing requested structure
- _Review focus:_ watchlist (cutoff, instruction echo, missing sections,
  reasoning leak)
- _Timing:_ Load 1.44s; Gen 67.21s; Total 68.87s
- _Throughput:_ Prompt 279 TPS (16,916 tok); Gen 84.4 TPS (500 tok)
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to analyze the provided image and generate cataloguing
> metadata in British English.
> &#8203;I need to follow specific rules:
> &#45; Use only details clearly and definitely visible.
> &#45; Do not guess or infer.
> &#45; Treat the provided "Existing metadata hints" as a draft.
> &#45; Keep only confirmed details, correct contradictions, and add important
> visible details.
> &#45; Output exactly three sections: Title, Description, Keywords.
>
> &#42;&#42;1. Analyze the Image:&#42;&#42;
> &#45; &#42;&#42;Subject:&#42;&#42; A small sailboat. It looks like a traditional or classic
> design.
> &#45; &#42;&#42;Hull:&#42;&#42; Dark (black or very dark blue/grey) with a wooden trim along the
> gunwale.
> &#45; &#42;&#42;Mast:&#42;&#42; Wooden, varnished.
> &#45; &#42;&#42;Boom:&#42;&#42; Wooden.
> &#45; &#42;&#42;Sails:&#42;&#42; The mainsail is furled (rolled up) and covered with a green
> sail cover. The jib/fore sail is furled and covered with a green cover.
> &#45; &#42;&#42;Flags:&#42;&#42; There is a string of small flags (bunting) attached to the
> rigging. They are red, white, and blue.
> &#45; &#42;&#42;Buoy:&#42;&#42; An orange buoy is attached to the bow (front) of the boat.
> &#45; &#42;&#42;Setting:&#42;&#42; A body of water. The water level is low (low tide).
> &#45; &#42;&#42;Background:&#42;&#42; Mudflats covered in green algae or seaweed. Behind that, a
> grassy bank and a line of trees (deciduous, some bare branches suggesting
> early spring or late autumn, but leaves are green).
> &#45; &#42;&#42;Lighting:&#42;&#42; Overcast, diffuse light. No harsh shadows.
> &#45; &#42;&#42;Action:&#42;&#42; The boat is stationary/moored.
>
> &#42;&#42;2. Evaluate Existing Metadata Hints:&#42;&#42;
> &#45; &#42;Description hint:&#42; "A classic-style sailboat with a dark hull and wooden
> mast is moored in a calm estuary during low tide." -> &#42;&#42;Confirmed.&#42;&#42; The
> boat is classic style, dark hull, wooden mast, moored, low tide is evident
> from the mudflats.
> &#45; &#42;Description hint:&#42; "The water has receded, exposing a vast expanse of
> green, algae-covered mudflats behind the vessel." -> &#42;&#42;Confirmed.&#42;&#42; The
> green stuff is clearly algae/seaweed on mudflats.
> &#45; &#42;Description hint:&#42; "The boat, adorned with a string of small flags,
> floats peacefully, waiting for the tide to rise again." -> &#42;&#42;Confirmed.&#42;&#42;
> The
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Reasoning leak (description hint:)
- Instruction echo
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: style, dark, hull, during, receded
- _Score:_ ✅ B (77/100); Δ+14; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections)
- _Timing:_ Load 6.95s; Gen 70.36s; Total 77.53s
- _Throughput:_ Prompt 186 TPS (772 tok); Gen 7.6 TPS (500 tok)
- _Tokens:_ prompt 772 tok; estimated text 444 tok; estimated non-text 328
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> &#45; Keywords hint: sailboat, estuary, low tide, mudflats, algae, wooden mast,
> flags, calm water, maritime, nature, coastal, tranquility, reflection,
> peaceful, moored, classic boat, green, brown, blue, overcast, cloudy,
> overcast sky, cloudy sky, overcast weather, cloudy weather, overcast day,
> cloudy day, overcast conditions, cloudy conditions, overcast atmosphere,
> cloudy atmosphere, overcast mood, cloudy mood, overcast tone, cloudy tone,
> overcast light, cloudy light, overcast sky background, cloudy sky
> background, overcast sky backdrop, cloudy sky backdrop, overcast sky
> setting, cloudy sky setting, overcast sky scene, cloudy sky scene, overcast
> sky view, cloudy sky view, overcast sky landscape, cloudy sky landscape,
> overcast sky scenery, cloudy sky scenery, overcast sky environment, cloudy
> sky environment, overcast sky ambiance, cloudy sky ambiance, overcast sky
> mood, cloudy sky mood, overcast sky tone, cloudy sky tone, overcast sky
> light, cloudy sky light, overcast sky background image, cloudy sky
> background image, overcast sky backdrop image, cloudy sky backdrop image,
> overcast sky setting image, cloudy sky setting image, overcast sky scene
> image, cloudy sky scene image, overcast sky view image, cloudy sky view
> image, overcast sky landscape image, cloudy sky landscape image, overcast
> sky scenery image, cloudy sky scenery image, overcast sky environment image,
> cloudy sky environment image, overcast sky ambiance image, cloudy sky
> ambiance image, overcast sky mood image, cloudy sky mood image, overcast sky
> tone image, cloudy sky tone image, overcast sky light image, cloudy sky
> light image, overcast sky background photo, cloudy sky background photo,
> overcast sky backdrop photo, cloudy sky backdrop photo, overcast sky setting
> photo, cloudy sky setting photo, overcast sky scene photo, cloudy sky scene
> photo, overcast sky view photo, cloudy sky view photo, overcast sky
> landscape photo, cloudy sky landscape photo, overcast sky scenery photo,
> cloudy sky scenery photo, overcast sky environment photo, cloudy sky
> environment photo, overcast sky ambiance photo, cloudy sky ambiance photo,
> overcast sky mood photo, cloudy sky mood photo, overcast sky tone photo,
> cloudy sky tone photo, overcast sky light photo, cloudy sky light photo,
> overcast sky background picture, cloudy sky background picture, overcast sky
> backdrop picture, cloudy sky backdrop picture, overcast sky setting
> picture, cloudy sky setting picture, overcast sky scene picture, cloudy sky
> scene picture, overcast sky view picture, cloudy sky view picture, overcast
> sky landscape picture, cloudy sky landscape picture, overcast sky scenery
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

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
  sections: title, description, keywords; missing terms: style, calm, during,
  receded, exposing
- _Score:_ ✅ B (74/100); Δ+11; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections)
- _Timing:_ Load 9.96s; Gen 71.52s; Total 81.70s
- _Throughput:_ Prompt 271 TPS (16,916 tok); Gen 60.0 TPS (500 tok)
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to generate cataloguing metadata for the provided image.
> &#8203;I need to follow specific rules:
> &#45; British English.
> &#45; Only clearly visible details.
> &#45; No guessing or inferring.
> &#45; Three specific sections: Title, Description, Keywords.
> &#45; Specific constraints on length and content for each section.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Subject:&#42;&#42; A small sailboat. It looks like a classic wooden boat.
> &#45; &#42;&#42;Hull:&#42;&#42; Dark (black or very dark blue/grey) with a white stripe near the
> waterline and a wooden gunwale.
> &#45; &#42;&#42;Mast:&#42;&#42; Wooden, tall, single mast.
> &#45; &#42;&#42;Sails:&#42;&#42; Furled in a green sail cover (boots).
> &#45; &#42;&#42;Rigging:&#42;&#42; White ropes, stays, shrouds.
> &#45; &#42;&#42;Flags:&#42;&#42; A string of small bunting flags (red, white, blue) strung from
> the mast to the bow/stern area.
> &#45; &#42;&#42;Buoy:&#42;&#42; An orange buoy floating near the bow, attached by a rope.
> &#45; &#42;&#42;Setting:&#42;&#42; Low tide. The boat is floating in a narrow channel of water,
> but the surrounding area is exposed mudflats covered in green algae/seaweed.
> &#45; &#42;&#42;Background:&#42;&#42; A grassy bank with trees. The trees are green, suggesting
> spring or summer.
> &#45; &#42;&#42;Lighting:&#42;&#42; Overcast, diffuse light. No harsh shadows.
> &#45; &#42;&#42;Action:&#42;&#42; Moored, stationary.
>
> &#8203;&#42;&#42;Drafting Title:&#42;&#42;
> &#45; Needs to be 5-10 words.
> &#45; Concrete and factual.
> &#45; &#42;Draft 1:&#42; A classic wooden sailboat moored in an estuary at low tide. (10
> words) - Good.
> &#45; &#42;Draft 2:&#42; Small sailboat with green sail cover floating on mudflats. (9
> words) - Good.
> &#45; &#42;Draft 3:&#42; Wooden mast sailboat moored near green algae-covered mudflats.
> (9 words) - Good.
> &#45; Let's go with something descriptive of the main elements. "Classic wooden
> sailboat moored in an estuary at low tide" is accurate. Or "Sailboat with
> green sail cover moored on mudflats". Let's try to capture the mood slightly
> but keep it factual. "Small sailboat moored in an estuary channel at low
> tide". (10 words).
>
> &#8203;&#42;&#42;Drafting Description:&#42;&#42;
> &#45; 1-
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Likely capped by max token budget

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=71%; missing sections: title; missing
  terms: during, receded, exposing, vast, expanse; keyword duplication=41%
- _Score:_ ✅ B (71/100); Δ+8; None identified
- _Review focus:_ watchlist (metadata borrowing, missing sections, reasoning
  leak)
- _Timing:_ Load 3.22s; Gen 77.22s; Total 80.65s
- _Throughput:_ Prompt 1,054 TPS (1,516 tok); Gen 4.66 TPS (351 tok)
- _Tokens:_ prompt 1516 tok; estimated text 444 tok; estimated non-text 1072
  tok; generated 351 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷So, let's analyze the image based on the given instructions. First,
> the title needs to be 5-10 words, concrete and factual. The main subject is
> a sailboat. Let's check the visible details: dark hull, wooden mast, flags,
> calm water, mudflats, trees in background. So the title could be "Classic
> sailboat with flags in calm estuary".
>
> For the description, 1-2 sentences. The main subject is a classic-style
> sailboat with a dark hull and wooden mast moored in a calm estuary at low
> tide. The water is calm, mudflats are visible behind, and there are trees in
> the background. So the description would be: "A classic-style sailboat with
> a dark hull and wooden mast is moored in a calm estuary, with exposed
> mudflats and trees in the background."
>
> Keywords: Need 10-18 unique terms. Visible elements: sailboat, dark hull,
> wooden mast, flags, calm water, mudflats, trees, green vegetation, buoy,
> estuary, low tide. Let's list them: sailboat, dark hull, wooden mast, flags,
> calm water, mudflats, trees, green vegetation, buoy, estuary, low tide.
> That's 10 keywords.◁/think▷Title:
> Classic sailboat with flags in calm estuary
>
> &#8203;Description:
> A classic - style sailboat with a dark hull and wooden mast is moored in a
> calm estuary, with exposed mudflats and trees in the background.
>
> &#8203;Keywords:
> sailboat, dark hull, wooden mast, flags, calm water, mudflats, trees, green
> vegetation, buoy, estuary, low tide<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Keyword count violation (22; expected 10-18)
- Keyword duplication (41% duplicated terms)
- Reasoning leak (◁think▷, ◁/think▷)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; nonvisual metadata reused
- _Score:_ 🟡 C (60/100); Δ-2; Missing requested structure
- _Review focus:_ watchlist (cutoff, instruction echo, metadata borrowing,
  missing sections, reasoning leak, refusal)
- _Timing:_ Load 2.19s; Gen 86.83s; Total 89.25s
- _Throughput:_ Prompt 246 TPS (16,916 tok); Gen 28.9 TPS (500 tok)
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to analyze the image and generate cataloguing metadata
> based on specific rules.
>
> &#42;&#42;1. Analyze the Image:&#42;&#42;
> &#42;   &#42;&#42;Subject:&#42;&#42; A small sailboat (looks like a traditional cutter or
> sloop).
> &#42;   &#42;&#42;Hull:&#42;&#42; Dark (black or very dark blue) hull with a white waterline
> stripe.
> &#42;   &#42;&#42;Mast:&#42;&#42; Wooden mast, reddish-brown.
> &#42;   &#42;&#42;Sails:&#42;&#42; Furled in a green sail cover (boom cover).
> &#42;   &#42;&#42;Rigging:&#42;&#42; Visible shrouds and stays. A red boom or gaff extension is
> visible on the left.
> &#42;   &#42;&#42;Flags:&#42;&#42; A string of small bunting flags (red, white, blue) runs along
> the rigging.
> &#42;   &#42;&#42;Setting:&#42;&#42; The boat is floating in shallow water. Behind it is a vast
> expanse of green, algae-covered mudflats (low tide). Further back is a
> grassy bank and a line of trees (forest/woodland).
> &#42;   &#42;&#42;Water:&#42;&#42; Calm, reflective water in the foreground.
> &#42;   &#42;&#42;Buoy:&#42;&#42; An orange spherical buoy is floating near the bow.
> &#42;   &#42;&#42;Lighting:&#42;&#42; Overcast, soft daylight. No harsh shadows.
> &#42;   &#42;&#42;Composition:&#42;&#42; The boat is centered horizontally but sits in the lower
> third. The background is dominated by the green mudflats and trees.
>
> &#42;&#42;2. Evaluate Existing Metadata Hints:&#42;&#42;
> &#42;   &#42;Description hint:&#42; "A classic-style sailboat with a dark hull and
> wooden mast is moored in a calm estuary during low tide. The water has
> receded, exposing a vast expanse of green, algae-covered mudflats behind the
> vessel. The boat, adorned with a string of small flags, floats peacefully,
> waiting for the tide to rise again." -> This is mostly accurate.
> "Classic-style" is a bit subjective but the boat looks traditional. "Moored"
> is likely but "floating" is safer. "Waiting for the tide" is an inference,
> though highly probable given the mudflats. I should stick to "floating" or
> "anchored" if visible (anchor isn't clearly visible, but the buoy suggests
> mooring).
> &#42;   &#42;Capture metadata:&#42; Date/Time/GPS. I cannot verify the date/time from
> the image itself, but the prompt says "Taken on 2026-05-02...". Wait
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Refusal detected (explicit_refusal)
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:, capture metadata:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: classic, style, during, receded, exposing; keywords=20
- _Score:_ 🏆 A (84/100); Δ+21; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 3.08s; Gen 102.60s; Total 105.96s
- _Throughput:_ Prompt 230 TPS (16,916 tok); Gen 17.6 TPS (500 tok)
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to analyze the image and generate cataloguing metadata in
> British English.
> &#8203;I need to follow specific rules:
> &#45; Only use clearly visible details.
> &#45; No guessing or inferring.
> &#45; Output exactly three sections: Title, Description, Keywords.
> &#45; Title: 5-10 words, concrete and factual.
> &#45; Description: 1-2 factual sentences.
> &#45; Keywords: 10-18 unique comma-separated terms.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Subject:&#42;&#42; A small sailboat (looks like a traditional cutter or sloop).
> &#45; &#42;&#42;Hull:&#42;&#42; Dark (black or very dark blue) with a white stripe at the
> waterline and a wooden gunwale/rubbing strake.
> &#45; &#42;&#42;Mast:&#42;&#42; Wooden, tall, single mast.
> &#45; &#42;&#42;Sails:&#42;&#42; Furled and covered with a green canvas sail cover.
> &#45; &#42;&#42;Rigging:&#42;&#42; Visible shrouds and stays.
> &#45; &#42;&#42;Decorations:&#42;&#42; A string of bunting (small triangular flags) runs from
> the mast to the bow and stern.
> &#45; &#42;&#42;Foreground:&#42;&#42; Water, calm, reflecting the boat slightly. An orange buoy
> is floating near the bow.
> &#45; &#42;&#42;Background:&#42;&#42; Mudflats covered in green algae/seaweed (indicates low
> tide). Behind that, a grassy bank with trees (forest/woodland). The trees
> have green leaves, suggesting spring or summer.
> &#45; &#42;&#42;Lighting:&#42;&#42; Natural daylight, somewhat soft, possibly overcast or late
> afternoon (shadows are soft).
>
> &#8203;&#42;&#42;Drafting Metadata:&#42;&#42;
>
> &#42;   &#42;&#42;Title:&#42;&#42; Needs to be 5-10 words.
> &#42;   &#42;Draft 1:&#42; Small sailboat moored in estuary at low tide with green
> mudflats. (10 words) - Good.
> &#42;   &#42;Draft 2:&#42; Traditional sailboat with green sail cover floating near
> orange buoy. (10 words) - Good.
> &#42;   &#42;Selection:&#42; Small sailboat moored in estuary at low tide with green
> mudflats.
>
> &#42;   &#42;&#42;Description:&#42;&#42; 1-2 sentences.
> &#42;   &#42;Draft:&#42; A small sailboat with a dark hull and wooden mast floats in
> calm water near an orange buoy. Behind the boat, green algae-covered
> mudflats stretch towards a line of trees on the bank.
> &#42;   &#42;Refinement:&#42; Ensure British English (e.g.,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ recommended; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: style, estuary, during, tide, receded; keywords=19
- _Score:_ 🏆 A (84/100); Δ+22; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 3.07s; Gen 108.92s; Total 112.25s
- _Throughput:_ Prompt 212 TPS (16,916 tok); Gen 17.6 TPS (500 tok)
- _Tokens:_ prompt 16916 tok; estimated text 444 tok; estimated non-text 16472
  tok; generated 500 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants metadata for the provided image of a sailboat.
> &#8203;I need to follow specific rules:
> &#45; British English.
> &#45; Only clearly visible details.
> &#45; No guessing.
> &#45; Three sections: Title, Description, Keywords.
> &#45; Title: 5-10 words, concrete.
> &#45; Description: 1-2 sentences, factual.
> &#45; Keywords: 10-18 unique terms.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Subject:&#42;&#42; A sailboat. It has a dark (black or very dark blue) hull with
> a wooden trim. It has a wooden mast. The sails are furled and covered in a
> green canvas cover.
> &#45; &#42;&#42;Setting:&#42;&#42; It's in water, but the water level is low. Behind the boat,
> there is a large area of mudflats covered in green algae or moss. Further
> back, there is a grassy bank with trees (a forest/woodland).
> &#45; &#42;&#42;Details:&#42;&#42;
> &#45; There is a bright orange buoy floating near the bow.
> &#45; There are bunting flags (small triangular flags) strung along the boom
> and rigging. Some look like Union Jacks, others are red, white, and blue.
> &#45; There is a ladder on the stern (back) of the boat.
> &#45; The water is calm and reflective.
> &#45; The lighting suggests daytime, possibly overcast or late afternoon
> given the soft light.
> &#45; The boat is moored (stationary).
>
> &#8203;&#42;&#42;Drafting Metadata:&#42;&#42;
>
> &#42;   &#42;&#42;Title:&#42;&#42; Needs to be 5-10 words.
> &#42;   &#42;Draft 1:&#42; Dark hull sailboat moored on green mudflats
> &#42;   &#42;Draft 2:&#42; Classic sailboat with green sail cover on mudflats
> &#42;   &#42;Draft 3:&#42; Moored sailboat with orange buoy on tidal mudflats
> &#42;   &#42;Selection:&#42; Moored sailboat with orange buoy on tidal mudflats (7
> words) - Good. Or "Dark-hulled sailboat moored on green tidal mudflats" (8
> words). Let's go with: "Dark-hulled sailboat moored on green tidal
> mudflats".
>
> &#42;   &#42;&#42;Description:&#42;&#42; 1-2 sentences.
> &#42;   &#42;Draft:&#42; A dark-hulled sailboat with a wooden mast and green sail
> cover is moored in calm water. Behind the vessel,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (4 words; expected 5-10)
- Keyword count violation (19; expected 10-18)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<!-- markdownlint-enable MD033 MD034 -->
