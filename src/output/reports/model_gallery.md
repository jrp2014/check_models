# Model Output Gallery

_Generated on 2026-05-10 00:41:07 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Shortlist

### Strong Candidates

- `mlx-community/gemma-3-27b-it-qat-4bit`: 🏆 A (88/100) | Desc 98 | Keywords 85 | Δ+12 | 30.6 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (86/100) | Desc 93 | Keywords 79 | Δ+10 | 63.3 tps
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🏆 A (83/100) | Desc 95 | Keywords 92 | Δ+8 | 66.0 tps
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🏆 A (82/100) | Desc 100 | Keywords 84 | Δ+7 | 17.6 tps
- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🏆 A (80/100) | Desc 100 | Keywords 77 | Δ+5 | 108.0 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) | Desc 44 | Keywords 0 | Δ-70 | 247.5 tps | context ignored, harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-69 | 30.5 tps | harness, metadata borrowing, missing sections
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (9/100) | Desc 48 | Keywords 0 | Δ-66 | 64.9 tps | context ignored, harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (35/100) | Desc 73 | Keywords 69 | Δ-40 | 5.2 tps | context ignored, harness
- `microsoft/Phi-3.5-vision-instruct`: 🟡 C (64/100) | Desc 100 | Keywords 61 | Δ-11 | 54.7 tps | harness, metadata borrowing

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types        | Affected Models                                                                                                                                    |
|----------------|------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-vlm`      |          4 | Error, Model Error | `facebook/pe-av-large`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `mlx`          |          2 | Model Error        | `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`                                                                      |
| `model-config` |          1 | Processor Error    | `mlx-community/MolmoPoint-8B-fp16`                                                                                                                 |

### Actionable Items by Package

#### mlx-vlm

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.p...`
  - Type: `ValueError`
- mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 (Error)
  - Error: `Model runtime error during generation for mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16: property 'text' of 'Naive...`
  - Type: `ValueError`
- mlx-community/LFM2-VL-1.6B-8bit (Error)
  - Error: `Model runtime error during generation for mlx-community/LFM2-VL-1.6B-8bit: property 'text' of 'NaiveStreamingDetokeni...`
  - Type: `ValueError`
- mlx-community/LFM2.5-VL-1.6B-bf16 (Error)
  - Error: `Model runtime error during generation for mlx-community/LFM2.5-VL-1.6B-bf16: property 'text' of 'NaiveStreamingDetoke...`
  - Type: `ValueError`

#### mlx

- LiquidAI/LFM2.5-VL-450M-MLX-bf16 (Model Error)
  - Error: `Model loading failed: Received 2 parameters not in model: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_project...`
  - Type: `ValueError`
- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`

## Image Metadata

- _Description:_ A low-angle, wide shot of St Peter's Church in Petersfield,
  Hampshire, England, on a sunny day. The Gothic Revival style church, with
  its tall spire and flint walls, is pictured against a bright blue sky with
  wispy clouds. A black car is parked in the foreground.
- _Keywords:_ Adobe Stock, Any Vision, Bell Tower, Blue sky, Car, Chapel,
  Church, Cross, Daylight, Dorking, England, Europe, Fence, Gothic
  Architecture, Objects, Sky, Station wagon, Steeple, Stone, Surrey, Tower,
  Town Centre, Traffic Cone, Tree, UK, United Kingdom, Vehicles, Wall, Weather
  vane, ancient, arched window, architecture, building, christianity, clock
  tower, culture, daytime, exterior, facade, faith, flint, gothic, gothic
  revival, heritage, historic, landmark, majestic, medieval, old, outdoors,
  parish church, religion, spire, spiritual, stone building, street, summer,
  sunny, tourism, traditional, travel, village, worship
- _Date:_ 2026-05-09 17:50:09 BST
- _Time:_ 17:50:09
- _GPS:_ 51.215500°N, 0.798500°W

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
> &#45; Description hint: A low-angle, wide shot of St Peter's Church in
> Petersfield, Hampshire, England, on a sunny day. The Gothic Revival style
> church, with its tall spire and flint walls, is pictured against a bright
> blue sky with wispy clouds. A black car is parked in the foreground.
> &#45; Keyword hints: Adobe Stock, Any Vision, Bell Tower, Blue sky, Car, Chapel,
> Church, Cross, Daylight, Dorking, England, Europe, Fence, Gothic
> Architecture, Objects, Sky, Station wagon, Steeple, Stone, Surrey
> &#45; Capture metadata: Taken on 2026-05-09 17:50:09 BST (at 17:50:09 local
> time). GPS: 51.215500°N, 0.798500°W.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Quick Navigation

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)
- _Best descriptions:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)
- _Failed models:_ `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `facebook/pe-av-large`,
  `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`,
  `mlx-community/Kimi-VL-A3B-Thinking-8bit`,
  `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, +1
  more
- _D/F utility models:_ `meta-llama/Llama-3.2-11B-Vision-Instruct`,
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`,
  `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/GLM-4.6V-Flash-mxfp4`,
  `mlx-community/Molmo-7B-D-0924-8bit`, `mlx-community/Molmo-7B-D-0924-bf16`,
  +9 more

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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17588, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16990, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 419, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 343, in load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17785, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17598, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17588, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16990, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 419, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 235, in load_model
    model_class, _ = get_model_and_args(config=config)
                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 137, in get_model_and_args
    raise ValueError(msg)
ValueError: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17785, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17598, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'
```

</details>

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ❌ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Status:_ Failed (Error)
- _Owner:_ likely owner `model`; reported package `mlx-vlm`; failure stage
  `Error`; diagnostic code `MLX_VLM_DECODE_ERROR`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Error summary:_ see error details below
- _Key signals:_ keywords=58; context echo=100%; nonvisual metadata reused;
  reasoning leak
- _Failure context:_ type `ValueError`; phase `decode`; code
  `MLX_VLM_DECODE_ERROR`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text 474 tok; estimated non-text n/a;
  generated n/a; requested max 500 tok; stop reason exception

_Error details:_

> Model runtime error during generation for
> mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16: property 'text' of
> 'NaiveStreamingDetokenizer' object has no setter

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17280, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17638, in _generate_once
    return strict_generate(
        model=model,
    ...<15 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 1922, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 1722, in stream_generate
    detokenizer = make_streaming_detokenizer(processor)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 405, in make_streaming_detokenizer
    detokenizer = copy(processor.detokenizer)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/copy.py", line 98, in copy
    return _reconstruct(x, None, *rv)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/copy.py", line 272, in _reconstruct
    setattr(y, key, value)
    ~~~~~~~^^^^^^^^^^^^^^^
AttributeError: property 'text' of 'NaiveStreamingDetokenizer' object has no setter

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17785, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17663, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17314, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17588, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 16990, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 419, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 343, in load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17785, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17598, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 4 parameters not in model:
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

</details>

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ❌ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Status:_ Failed (Error)
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`; reported
  package `mlx-vlm`; failure stage `Error`; diagnostic code
  `MLX_VLM_DECODE_ERROR`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Error summary:_ see error details below
- _Key signals:_ Special control token &lt;|im_end|&gt; appeared in generated
  text.; keywords=57; context echo=100%; nonvisual metadata reused
- _Failure context:_ type `ValueError`; phase `decode`; code
  `MLX_VLM_DECODE_ERROR`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text 474 tok; estimated non-text n/a;
  generated n/a; requested max 500 tok; stop reason exception

_Error details:_

> Model runtime error during generation for mlx-community/LFM2-VL-1.6B-8bit:
> property 'text' of 'NaiveStreamingDetokenizer' object has no setter

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17280, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17638, in _generate_once
    return strict_generate(
        model=model,
    ...<15 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 1922, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 1722, in stream_generate
    detokenizer = make_streaming_detokenizer(processor)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 405, in make_streaming_detokenizer
    detokenizer = copy(processor.detokenizer)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/copy.py", line 98, in copy
    return _reconstruct(x, None, *rv)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/copy.py", line 272, in _reconstruct
    setattr(y, key, value)
    ~~~~~~~^^^^^^^^^^^^^^^
AttributeError: property 'text' of 'NaiveStreamingDetokenizer' object has no setter

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17785, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17663, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17314, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/LFM2-VL-1.6B-8bit: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
```

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ❌ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Status:_ Failed (Error)
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`; reported
  package `mlx-vlm`; failure stage `Error`; diagnostic code
  `MLX_VLM_DECODE_ERROR`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Error summary:_ see error details below
- _Key signals:_ Special control token &lt;|im_end|&gt; appeared in generated
  text.; keywords=57; context echo=100%; nonvisual metadata reused
- _Failure context:_ type `ValueError`; phase `decode`; code
  `MLX_VLM_DECODE_ERROR`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text 474 tok; estimated non-text n/a;
  generated n/a; requested max 500 tok; stop reason exception

_Error details:_

> Model runtime error during generation for mlx-community/LFM2.5-VL-1.6B-bf16:
> property 'text' of 'NaiveStreamingDetokenizer' object has no setter

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17280, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17638, in _generate_once
    return strict_generate(
        model=model,
    ...<15 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 1922, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 1722, in stream_generate
    detokenizer = make_streaming_detokenizer(processor)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 405, in make_streaming_detokenizer
    detokenizer = copy(processor.detokenizer)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/copy.py", line 98, in copy
    return _reconstruct(x, None, *rv)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/copy.py", line 272, in _reconstruct
    setattr(y, key, value)
    ~~~~~~~^^^^^^^^^^^^^^^
AttributeError: property 'text' of 'NaiveStreamingDetokenizer' object has no setter

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17785, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17663, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17314, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/LFM2.5-VL-1.6B-bf16: property 'text' of 'NaiveStreamingDetokenizer' object has no setter
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17350, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17142, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17075, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17785, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17600, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 17391, in _prepare_generation_prompt
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
- _Key signals:_ missing sections: keywords; missing terms: Bell Tower,
  Chapel, Cross, Daylight, Dorking; nonvisual metadata reused
- _Score:_ 🟡 C (61/100); Δ-14; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 0.51s; Gen 0.89s; Total 1.65s
- _Throughput:_ Prompt 6,366 TPS (561 tok); Gen 346 TPS (137 tok)
- _Tokens:_ prompt 561 tok; estimated text 474 tok; estimated non-text 87 tok;
  generated 137 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: St Peter's Church in Petersfield, Hampshire, England
>
> Description: A low-angle, wide shot of St Peter's Church in Petersfield,
> Hampshire, England, with a Gothic Revival style church in the background.
> The church is set against a bright blue sky with wispy clouds. A black car
> is parked in the foreground. The image was taken on 2026-05-09 17:50:09 BST
> (at 17:50:09 local time). GPS: 51.215500°N, 0.798500°W.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Description sentence violation (5; expected 1-2)
- Nonvisual metadata borrowing
- suspicious_precision: 2 overly precise numbers

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bell Tower, Blue
  sky, Chapel, Cross, Daylight; nonvisual metadata reused
- _Score:_ 🟠 D (48/100); Δ-28; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 0.53s; Gen 1.25s; Total 2.04s
- _Throughput:_ Prompt 4,379 TPS (561 tok); Gen 98.1 TPS (68 tok)
- _Tokens:_ prompt 561 tok; estimated text 474 tok; estimated non-text 87 tok;
  generated 68 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: St Peter's Church in Petersfield, Hampshire, England
> Description: A high-angle, wide shot of St Peter's Church in Petersfield,
> Hampshire, England, on a sunny day. The church is painted in Gothic
> architecture with tall spires and flint walls. A black car is parked in
> front of the church.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Description sentence violation (3; expected 1-2)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: sunny, day;
  nonvisual metadata reused
- _Score:_ 🟠 D (46/100); Δ-29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 0.62s; Gen 1.54s; Total 2.43s
- _Throughput:_ Prompt 5,579 TPS (565 tok); Gen 344 TPS (222 tok)
- _Tokens:_ prompt 565 tok; estimated text 474 tok; estimated non-text 91 tok;
  generated 222 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: St Peter's Church, Petersfield, Hampshire, England, 5-10 words,
> concrete and factual, limited to clearly visible content.
> Description: A photograph of St Peter's Church in Petersfield, Hampshire,
> England, taken on 2026-05-09 17:50:09 BST. The church, with its tall spire
> and flint walls, is pictured against a bright blue sky with wispy clouds. A
> black car is parked in the foreground. The image is taken from a low-angle,
> wide shot, capturing the Gothic Revival style church, with its tall spire
> and flint walls, and the surrounding greenery. The church is identified as a
> landmark and is part of the Bell Tower, Blue sky, Car, Chapel, Church,
> Cross, Daylight, Dorking, England, Europe, Fence, Gothic Architecture,
> Objects, Sky, Station wagon, Steeple, Stone, Surrey. The image is
> copyrighted by Adobe Stock and is licensed under a Creative Commons
> Attribution 4.0 International License.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Title length violation (17 words; expected 5-10)
- Description sentence violation (6; expected 1-2)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Chapel, Cross, Dorking, Objects, Station wagon
- _Score:_ 🏆 A (80/100); Δ+5; Limited novel information
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.33s; Gen 1.73s; Total 4.35s
- _Throughput:_ Prompt 1,607 TPS (835 tok); Gen 108 TPS (86 tok)
- _Tokens:_ prompt 835 tok; estimated text 474 tok; estimated non-text 361
  tok; generated 86 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Tall stone church with a high spire under blue sky
>
> &#8203;Description:
> A low-angle shot shows a Gothic Revival style church with a tall spire and
> flint walls against a bright blue sky. A black car is partially visible in
> the foreground.
>
> &#8203;Keywords:
> architecture, bell tower, blue sky, car, church, daylight, fence, gothic
> architecture, steeple, stone, spire, sunny, vertical, wall
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; context echo=53%; nonvisual metadata reused
- _Score:_ 🟡 C (57/100); Δ-18; Limited novel information
- _Review focus:_ watchlist (context echo, metadata borrowing)
- _Timing:_ Load 0.55s; Gen 1.85s; Total 2.66s
- _Throughput:_ Prompt 1,982 TPS (671 tok); Gen 135 TPS (131 tok)
- _Tokens:_ prompt 671 tok; estimated text 474 tok; estimated non-text 197
  tok; generated 131 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: St Peter's Church, Petersfield, Hampshire, England
> Description: A low-angle, wide shot of St Peter's Church in Petersfield,
> Hampshire, England, on a sunny day. The Gothic Revival style church, with
> its tall spire and flint walls, is pictured against a bright blue sky with
> wispy clouds. A black car is parked in the foreground.
> Keywords: Adobe Stock, Any Vision, Bell Tower, Blue sky, Car, Chapel,
> Church, Cross, Daylight, Dorking, England, Europe, Fence, Gothic
> Architecture, Objects, Sky, Station wagon, Steeple, Stone, Surrey
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Context echo (53% overlap)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: Bell Tower, Blue sky, Car, Chapel,
  Church
- _Score:_ ❌ F (28/100); Δ-47; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.63s; Gen 1.97s; Total 3.88s
- _Throughput:_ Prompt 2,909 TPS (1,585 tok); Gen 17.4 TPS (17 tok)
- _Tokens:_ prompt 1585 tok; estimated text 474 tok; estimated non-text 1111
  tok; generated 17 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; The sky is bright and clear, with a few wispy white clouds.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bell Tower, Blue sky, Car, Chapel, Church)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 85% and the output stays weak under that load.
- _Key signals:_ output/prompt=4.95%; nontext prompt burden=85%; missing
  terms: Bell Tower, Chapel, Dorking, Gothic Architecture, Objects; nonvisual
  metadata reused
- _Score:_ 🏆 A (81/100); Δ+6; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.94s; Gen 2.33s; Total 3.53s
- _Throughput:_ Prompt 3,058 TPS (3,172 tok); Gen 184 TPS (157 tok)
- _Tokens:_ prompt 3172 tok; estimated text 474 tok; estimated non-text 2698
  tok; generated 157 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Gothic Revival church spire with decorative stonework and clock
>
> &#8203;Description:
> A tall, narrow stone church tower rises under a clear blue sky, illuminated
> by bright daylight. The structure features pointed arches, intricate
> tracery, and a steep, pinnacled spire capped with a cross. The church’s
> flint-and-pointed-dressing facade is complemented by lush greenery and a low
> metal fence around the perimeter. A modern black car is parked in the
> foreground, slightly obstructing the lower view of the building.
>
> &#8203;Keywords:
> church, Gothic Revival, spire, stone architecture, clock, pointed arches,
> tracery, blue sky, daylight, decorative stonework, tall tower, England,
> stone facade, metal fence, car, landscape, architectural detail
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Description sentence violation (4; expected 1-2)
- Nonvisual metadata borrowing

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%; keywords=20; nonvisual metadata
  reused
- _Score:_ 🟡 C (51/100); Δ-24; Missing requested structure
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.61s; Gen 2.41s; Total 3.26s
- _Throughput:_ Prompt 4,048 TPS (1,771 tok); Gen 123 TPS (182 tok)
- _Tokens:_ prompt 1771 tok; estimated text 474 tok; estimated non-text 1297
  tok; generated 182 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 5-10 words, concrete and factual, limited to clearly visible content.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> Title: St Peter's Church in Petersfield, Hampshire, England, on a sunny day.
>
> Description: A low-angle, wide shot of St Peter's Church in Petersfield,
> Hampshire, England, on a sunny day. The Gothic Revival style church, with
> its tall spire and flint walls, is pictured against a bright blue sky with
> wispy clouds. A black car is parked in the foreground.
>
> Keywords: Adobe Stock, Any Vision, Bell Tower, Blue sky, Car, Chapel,
> Church, Cross, Daylight, Dorking, England, Europe, Fence, Gothic
> Architecture, Objects, Sky, Station wagon, Steeple, Stone, Surrey
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=73%; keywords=20; nonvisual metadata
  reused
- _Score:_ 🟡 C (51/100); Δ-24; Missing requested structure
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.58s; Gen 2.49s; Total 3.33s
- _Throughput:_ Prompt 3,836 TPS (1,771 tok); Gen 123 TPS (182 tok)
- _Tokens:_ prompt 1771 tok; estimated text 474 tok; estimated non-text 1297
  tok; generated 182 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> 5-10 words, concrete and factual, limited to clearly visible content.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> Title: St Peter's Church in Petersfield, Hampshire, England, on a sunny day.
>
> Description: A low-angle, wide shot of St Peter's Church in Petersfield,
> Hampshire, England, on a sunny day. The Gothic Revival style church, with
> its tall spire and flint walls, is pictured against a bright blue sky with
> wispy clouds. A black car is parked in the foreground.
>
> Keywords: Adobe Stock, Any Vision, Bell Tower, Blue sky, Car, Chapel,
> Church, Cross, Daylight, Dorking, England, Europe, Fence, Gothic
> Architecture, Objects, Sky, Station wagon, Steeple, Stone, Surrey
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ nontext prompt burden=70%; missing sections: title,
  description, keywords; missing terms: Bell Tower, Blue sky, Car, Chapel,
  Church
- _Score:_ ❌ F (33/100); Δ-42; Lacks visual description of image
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.89s; Gen 2.52s; Total 4.67s
- _Throughput:_ Prompt 1,182 TPS (1,585 tok); Gen 27.6 TPS (20 tok)
- _Tokens:_ prompt 1585 tok; estimated text 474 tok; estimated non-text 1111
  tok; generated 20 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Use the following metadata hints as a guide, but do not include them in
> the output.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bell Tower, Blue sky, Car, Chapel, Church)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Check chat-template and EOS defaults first; the output shape is
  not matching the requested contract.
- _Key signals:_ Output is very short relative to prompt size (0.4%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=83%; missing sections: title, description, keywords; missing terms:
  Bell Tower, Blue sky, Car, Chapel, Cross
- _Score:_ ❌ F (9/100); Δ-66; Output lacks detail
- _Review focus:_ watchlist (context ignored, harness, missing sections)
- _Timing:_ Load 0.91s; Gen 2.93s; Total 4.10s
- _Throughput:_ Prompt 1,251 TPS (2,789 tok); Gen 64.9 TPS (11 tok)
- _Tokens:_ prompt 2789 tok; estimated text 474 tok; estimated non-text 2315
  tok; generated 11 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image is a photograph of a church.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Bell Tower, Blue sky, Car, Chapel, Cross)
- Missing sections (title, description, keywords)
- Ignores trusted hints
- output:output_ratio(0.4%)

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=66%; missing terms: Bell Tower, Chapel,
  Cross, Fence, Objects; keywords=19; nonvisual metadata reused
- _Score:_ ✅ B (65/100); Δ-10; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.98s; Gen 2.97s; Total 4.20s
- _Throughput:_ Prompt 3,914 TPS (1,387 tok); Gen 56.9 TPS (129 tok)
- _Tokens:_ prompt 1387 tok; estimated text 474 tok; estimated non-text 913
  tok; generated 129 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Gothic Revival Church in Petersfield, Hampshire
>
> Description: A Gothic Revival style church with a tall spire and flint walls
> stands prominently against a bright blue sky with wispy clouds. A black car
> is parked in the foreground.
>
> Keywords: Gothic Revival, Church, Spire, Flint Walls, Blue Sky, Parked Car,
> Hampshire, England, Europe, Steeple, Stone, Wispy Clouds, Gothic
> Architecture, Church of St Peter, Petersfield, Dorking, Surrey, Daylight,
> Station Wagon
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=80%; missing terms: Chapel, Dorking,
  Objects, Surrey, low
- _Score:_ ✅ B (70/100); Δ-5; Limited novel information
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 1.69s; Gen 3.48s; Total 5.42s
- _Throughput:_ Prompt 2,641 TPS (2,344 tok); Gen 34.9 TPS (75 tok)
- _Tokens:_ prompt 2344 tok; estimated text 474 tok; estimated non-text 1870
  tok; generated 75 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Church with Gothic Architecture
>
> &#8203;Description:
> A Gothic Revival church with a tall spire and flint walls is captured on a
> sunny day against a bright blue sky. A black car is parked in the
> foreground.
>
> &#8203;Keywords:
> Church, Gothic Architecture, Steeple, Bell Tower, Blue sky, Car, Stone,
> Cross, Daylight, Fence, Station wagon
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (4 words; expected 5-10)

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output is very short relative to prompt size (0.6%),
  suggesting possible early-stop or prompt-handling issues.; nontext prompt
  burden=70%; missing terms: Bell Tower, Blue sky, Car, Chapel, Church
- _Score:_ ❌ F (35/100); Δ-40; Output lacks detail
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 2.63s; Gen 3.89s; Total 6.79s
- _Throughput:_ Prompt 926 TPS (1,585 tok); Gen 5.24 TPS (9 tok)
- _Tokens:_ prompt 1585 tok; estimated text 474 tok; estimated non-text 1111
  tok; generated 9 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Use only the above metadata hints.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Bell Tower, Blue sky, Car, Chapel, Church)
- Ignores trusted hints
- output:output_ratio(0.6%)

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=85%; missing terms: Bell Tower, Chapel,
  Cross, Dorking, Fence
- _Score:_ 🏆 A (83/100); Δ+8; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.30s; Gen 4.50s; Total 6.06s
- _Throughput:_ Prompt 1,313 TPS (3,173 tok); Gen 66.0 TPS (108 tok)
- _Tokens:_ prompt 3173 tok; estimated text 474 tok; estimated non-text 2699
  tok; generated 108 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Gothic Revival Church with Tall Spire and Parked Car
>
> &#8203;Description:
> A low-angle photograph captures a Gothic Revival church featuring a
> prominent tall spire and intricately detailed stonework. The structure is
> set against a clear blue sky with wispy clouds, and a black car is parked
> directly in front of the building.
>
> &#8203;Keywords:
> Gothic Revival architecture, tall spire, stone church, blue sky, wispy
> clouds, black car, low-angle shot, detailed stonework, church facade,
> daylight
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=85%; missing terms: Bell Tower, Chapel,
  Cross, Dorking, Gothic Architecture
- _Score:_ 🏆 A (86/100); Δ+10; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.33s; Gen 4.54s; Total 6.13s
- _Throughput:_ Prompt 1,402 TPS (3,173 tok); Gen 63.3 TPS (116 tok)
- _Tokens:_ prompt 3173 tok; estimated text 474 tok; estimated non-text 2699
  tok; generated 116 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;&#42;&#42;Title:&#42;&#42;
> Gothic Revival Church with Tall Spire and Parked Car
>
> &#8203;&#42;&#42;Description:&#42;&#42;
> A low-angle view captures a historic stone church featuring Gothic Revival
> architecture, including a prominent spire and clock face. A black car is
> parked directly in front of the building, set against a clear blue sky and
> surrounded by greenery and a metal fence.
>
> &#8203;&#42;&#42;Keywords:&#42;&#42;
> Church, Gothic Revival, Spire, Stone architecture, Blue sky, Car, Clock
> face, Metal fence, Greenery, Historic building, Gothic details, Low-angle
> shot, Clear daylight, Flint walls
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bell Tower, Chapel, Cross, Dorking, Fence
- _Score:_ 🏆 A (88/100); Δ+12; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.31s; Gen 4.72s; Total 7.30s
- _Throughput:_ Prompt 662 TPS (830 tok); Gen 30.6 TPS (94 tok)
- _Tokens:_ prompt 830 tok; estimated text 474 tok; estimated non-text 356
  tok; generated 94 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Stone church with spire and black car
>
> &#8203;Description:
> A tall stone church with a prominent spire dominates the frame, set against
> a clear blue sky with sparse clouds. A black car is partially visible in the
> foreground, parked along a street.
>
> &#8203;Keywords:
> Architecture, Blue sky, Church, Clock, Daylight, Exterior, Gothic
> architecture, Grey stone, Roof, Spire, Stone, Trees, Vertical, Windows,
> Building, Car, Exterior view
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
  keywords; missing terms: Bell Tower, Blue sky, Car, Chapel, Church;
  nonvisual metadata reused
- _Score:_ ❌ F (2/100); Δ-73; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, metadata borrowing,
  missing sections)
- _Timing:_ Load 1.43s; Gen 4.84s; Total 6.54s
- _Throughput:_ Prompt 2,615 TPS (821 tok); Gen 120 TPS (500 tok)
- _Tokens:_ prompt 821 tok; estimated text 474 tok; estimated non-text 347
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> &#45;
> 17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:50:00:17:51:00:17:51:00:17:51:00:17:51:00:17:51:00:17:51:00:17:51:00:17:51:00:17:51:00:17:51:00:17:51:00:17:51:00:17
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bell Tower, Blue sky, Car, Chapel, Church)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=80%; missing terms: Dorking, Station
  wagon, Surrey, low, angle; nonvisual metadata reused
- _Score:_ ✅ B (75/100); Δ-0; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.74s; Gen 4.90s; Total 6.90s
- _Throughput:_ Prompt 1,317 TPS (2,344 tok); Gen 32.2 TPS (86 tok)
- _Tokens:_ prompt 2344 tok; estimated text 474 tok; estimated non-text 1870
  tok; generated 86 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Gothic Church with Tall Spire
>
> &#8203;Description:
> A Gothic Revival church with a tall spire and flint walls is shown against a
> bright blue sky with wispy clouds. A black car is parked in the foreground.
>
> &#8203;Keywords:
> Bell Tower, Blue sky, Car, Chapel, Church, Cross, Daylight, England, Europe,
> Fence, Gothic Architecture, Objects, Sky, Stone, Steeple
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ nontext prompt burden=83%; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking; context echo=47%; nonvisual metadata reused
- _Score:_ ✅ B (69/100); Δ-6; Limited novel information
- _Review focus:_ watchlist (context echo, metadata borrowing)
- _Timing:_ Load 1.88s; Gen 5.28s; Total 7.43s
- _Throughput:_ Prompt 2,375 TPS (2,844 tok); Gen 32.8 TPS (117 tok)
- _Tokens:_ prompt 2844 tok; estimated text 474 tok; estimated non-text 2370
  tok; generated 117 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: St Peter's Church in Petersfield, Hampshire, England.
>
> Description: A low-angle, wide shot of St Peter's Church in Petersfield,
> Hampshire, England, on a sunny day. The Gothic Revival style church, with
> its tall spire and flint walls, is pictured against a bright blue sky with
> wispy clouds. A black car is parked in the foreground.
>
> Keywords: St Peter's Church, Petersfield, Hampshire, England, Gothic
> Revival, Flint Walls, Tall Spire, Black Car, Bright Blue Sky, Wispy
> Clouds.<end_of_utterance>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Context echo (47% overlap)
- Nonvisual metadata borrowing
- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Chapel, Cross, Dorking, Objects, Surrey;
  nonvisual metadata reused
- _Score:_ ✅ B (78/100); Δ+3; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 2.53s; Gen 5.28s; Total 8.08s
- _Throughput:_ Prompt 591 TPS (835 tok); Gen 27.2 TPS (94 tok)
- _Tokens:_ prompt 835 tok; estimated text 474 tok; estimated non-text 361
  tok; generated 94 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Gothic Revival church with tall spire and flint walls
>
> Description: A low-angle shot of a stone church with a tall spire and clock
> face under a bright blue sky. A black car is parked in the foreground next
> to green shrubbery and a metal fence.
>
> Keywords: church, spire, Gothic architecture, flint, stone, clock, blue sky,
> car, station wagon, daylight, fence, shrubbery, steeple, bell tower,
> England, Europe
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ output/prompt=3.51%; nontext prompt burden=86%; missing
  terms: Bell Tower, Chapel, Dorking, Objects, Surrey; nonvisual metadata
  reused
- _Score:_ ✅ B (73/100); Δ-2; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.73s; Gen 6.43s; Total 8.43s
- _Throughput:_ Prompt 1,345 TPS (3,366 tok); Gen 33.8 TPS (118 tok)
- _Tokens:_ prompt 3366 tok; estimated text 474 tok; estimated non-text 2892
  tok; generated 118 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Gothic Church with Car in Foreground
>
> &#8203;Description:
> A low-angle, wide shot of a Gothic Revival style church on a sunny day. The
> church, with its tall spire and flint walls, is pictured against a bright
> blue sky with wispy clouds. A black car is parked in the foreground.
>
> &#8203;Keywords:
> Church, Steeple, Blue sky, Stone, Car, Gothic Architecture, Cross, Daylight,
> England, Europe, Fence, Sky, Green, Tree, Black Car, Flint Walls, Wispy
> Clouds, Station Wagon
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Description sentence violation (3; expected 1-2)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bell Tower, Chapel, Cross, Daylight, Dorking; nonvisual metadata
  reused
- _Score:_ ✅ B (66/100); Δ-9; Missing requested structure
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 2.21s; Gen 7.35s; Total 9.84s
- _Throughput:_ Prompt 1,902 TPS (829 tok); Gen 48.4 TPS (316 tok)
- _Tokens:_ prompt 829 tok; estimated text 474 tok; estimated non-text 355
  tok; generated 316 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;# Image Description
>
> A low-angle, wide shot captures the impressive St Peter's Church in
> Petersfield, Hampshire, England, on a bright, sunny day. The church
> showcases a classic Gothic Revival style, characterized by its tall, pointed
> spire and distinctive flint wall construction. The building stands
> prominently against a clear, vibrant blue sky dotted with wispy white
> clouds.
>
> In the foreground, a dark-colored car is parked, partially obscuring the
> lower portion of the church. Lush green bushes flank the base of the church
> walls, adding a touch of natural color to the scene. A black metal fence
> runs along the edge of the church grounds, separating it from the street.
>
> The church's architectural details are clearly visible, including its
> pointed arches, intricate window tracery, and decorative stonework. The
> overall impression is one of historical grandeur and serene beauty, typical
> of a well-maintained English parish church.
>
> &#35;# Keywords
>
> St Peter's Church, Petersfield, Hampshire, England, Gothic Revival, spire,
> flint walls, blue sky, clouds, car, church, steeple, stone, sunny day,
> architecture, historical, serene, England, countryside, village, parish.
>
> &#35;# Image Details
>
> &#42;&#42;Date Taken:&#42;&#42; 2026-05-09 17:50:09 BST (17:50:09 local time)
> &#42;&#42;GPS Coordinates:&#42;&#42; 51.215500°N, 0.798500°W
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `encoding`
- _Next step:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.
- _Key signals:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 57 occurrences).; nontext prompt burden=82%; missing sections:
  description, keywords; missing terms: Dorking, Gothic Architecture, Objects,
  Surrey, low
- _Score:_ ❌ F (6/100); Δ-69; Output too short to be useful
- _Review focus:_ watchlist (harness, metadata borrowing, missing sections)
- _Timing:_ Load 2.15s; Gen 7.49s; Total 9.94s
- _Throughput:_ Prompt 676 TPS (2,676 tok); Gen 30.5 TPS (93 tok)
- _Tokens:_ prompt 2676 tok; estimated text 474 tok; estimated non-text 2202
  tok; generated 93 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title:ĠGothicĠRevivalĠChurchĠwithĠSpireĊĊDescription:ĠAĠGothicĠRevivalĠstyleĠchurchĠwithĠaĠtallĠspireĠandĠflintĠwallsĠstandsĠagainstĠaĠbrightĠblueĠskyĠwithĠwispyĠclouds.ĠAĠblackĠcarĠisĠparkedĠinĠtheĠforeground.ĊĊKeywords:ĠGothicĠRevival,Ġchurch,Ġspire,ĠflintĠwalls,ĠblueĠsky,ĠwispyĠclouds,ĠblackĠcar,Ġstone,Ġcross,Ġdaylight,Ġfence,Ġsteeple,Ġchapel,ĠbellĠtower,ĠstationĠwagon,ĠEurope
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Missing sections (description, keywords)
- Title length violation (59 words; expected 5-10)
- Nonvisual metadata borrowing
- token_encoding:bpe_space_leak(57)

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Chapel, Cross, Dorking, Objects, Station wagon
- _Score:_ 🏆 A (82/100); Δ+7; Limited novel information
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.36s; Gen 7.86s; Total 11.49s
- _Throughput:_ Prompt 578 TPS (830 tok); Gen 17.6 TPS (106 tok)
- _Tokens:_ prompt 830 tok; estimated text 474 tok; estimated non-text 356
  tok; generated 106 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> St Peter's Church, Petersfield, Hampshire
>
> &#8203;Description:
> A low-angle, wide shot depicts a stone church with a tall spire against a
> bright blue sky with wispy clouds. A black car is partially visible in the
> foreground, with greenery and a fence also present.
>
> &#8203;Keywords:
> Architecture, Blue sky, Church, Daylight, Exterior, Fence, Gothic
> Architecture, Grey stone, Spire, Stone, Trees, Windows, Bell tower, Cloud,
> Exterior view, Greenery, Steeple
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ output/prompt=3.48%; nontext prompt burden=86%; missing
  terms: Bell Tower, Chapel, Dorking, Objects, Station wagon; nonvisual
  metadata reused
- _Score:_ ✅ B (73/100); Δ-2; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 2.61s; Gen 8.98s; Total 11.85s
- _Throughput:_ Prompt 1,500 TPS (3,366 tok); Gen 18.6 TPS (117 tok)
- _Tokens:_ prompt 3366 tok; estimated text 474 tok; estimated non-text 2892
  tok; generated 117 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Gothic Church with Car in Foreground
>
> &#8203;Description:
> A low-angle, wide shot of a Gothic Revival style church on a sunny day. The
> church, with its tall spire and flint walls, is pictured against a bright
> blue sky with wispy clouds. A black car is parked in the foreground.
>
> &#8203;Keywords:
> Church, Steeple, Blue sky, Stone, Car, Gothic Architecture, Cross, Daylight,
> England, Europe, Fence, Sky, Green, Tree, Black Car, Flint Walls, Wispy
> Clouds, Sunny Day
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Description sentence violation (3; expected 1-2)
- Nonvisual metadata borrowing

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;|end|&gt; appeared in generated
  text.; Special control token &lt;|endoftext|&gt; appeared in generated
  text.; hit token cap (500); nontext prompt burden=66%
- _Score:_ 🟡 C (64/100); Δ-11; Lacks visual description of image
- _Review focus:_ watchlist (harness, metadata borrowing)
- _Timing:_ Load 0.88s; Gen 9.90s; Total 11.05s
- _Throughput:_ Prompt 3,565 TPS (1,387 tok); Gen 54.7 TPS (500 tok)
- _Tokens:_ prompt 1387 tok; estimated text 474 tok; estimated non-text 913
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Gothic Revival Church in Petersfield, Hampshire
>
> Description: A Gothic Revival style church with a tall spire and flint walls
> stands prominently against a bright blue sky with wispy clouds. A black car
> is parked in the foreground.
>
> Keywords: Gothic Revival, Church, Spire, Flint Walls, Blue Sky, Parked Car,
> Hampshire, England, Europe, Steeple, Stone, Wispy Clouds, Gothic
> Architecture, Church of St Peter, Petersfield, Dorking, Surrey, Daylight,
> Station Wagon<|end|><|endoftext|>
>
>
> &#8203;Here is a text excerpt from a historical document:
>
> "In the year 1845, the renowned explorer Sir John Franklin embarked on a
> perilous journey to the Arctic. His expedition was fraught with challenges,
> including harsh weather conditions and the treacherous ice-covered
> landscapes. Despite the adversities, Franklin's team managed to navigate
> through the icy terrain, reaching a point where they believed they had found
> a passage to the North Pole. However, the harsh conditions eventually led
> to the loss of contact with the expedition, and it was presumed that
> Franklin and his crew perished in the Arctic.
>
> In the year 1986, a team of modern explorers set out to uncover the fate of
> Sir John Franklin's expedition. Equipped with advanced technology and
> satellite imagery, they aimed to locate the wreckage of the expedition's
> ships and any surviving members of the crew. After months of searching, the
> team discovered the remains of the expedition's ships, but no survivors were
> found. The search was a testament to the enduring mystery of the Arctic and
> the resilience of those who dare to explore its secrets."
>
> &#8203;Based on the text provided, answer the following questions:
>
> 1&#46; What year did Sir John Franklin embark on his expedition?
> 2&#46; What was the primary challenge faced by Franklin's team during their
> expedition?
> 3&#46; What was the outcome of the modern explorers' search in 1986?
> 4&#46; How did the modern explorers' technology differ from the technology
> available to Franklin's team in 1845?
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Keyword count violation (38; expected 10-18)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- tokenizer_artifact
- token_leak:<\|end\|>
- token_leak:<\|endoftext\|>

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bell Tower, Blue sky, Car, Chapel, Church
- _Score:_ 🟠 D (41/100); Δ-34; Lacks visual description of image
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 5.97s; Gen 10.02s; Total 16.25s
- _Throughput:_ Prompt 294 TPS (823 tok); Gen 7.51 TPS (51 tok)
- _Tokens:_ prompt 823 tok; estimated text 474 tok; estimated non-text 349
  tok; generated 51 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> &#45; Camera metadata: Canon EOS 5D Mark IV, 24mm, f/11, 1/125s, ISO 100.
>
> Output only the three sections: Title, Description, Keywords.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bell Tower, Blue sky, Car, Chapel, Church)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=69%; missing
  sections: title, description; missing terms: Bell Tower, Gothic
  Architecture, Steeple, Surrey, pictured
- _Score:_ ✅ B (68/100); Δ-7; Missing requested structure
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections,
  reasoning leak)
- _Timing:_ Load 4.55s; Gen 10.13s; Total 14.93s
- _Throughput:_ Prompt 998 TPS (1,553 tok); Gen 69.2 TPS (500 tok)
- _Tokens:_ prompt 1553 tok; estimated text 474 tok; estimated non-text 1079
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image based on the given instructions.
> First, the title needs to be 5-10 words, concrete and factual. The main
> visible elements are St Peter's Church in Petersfield, Hampshire, England,
> with a tall spire, flint walls, a black car in the foreground, and a bright
> blue sky. So the title could be "St Peter's Church, Petersfield, Hampshire,
> England, with black car, spire, blue sky".
>
> For the description, it should be 1-2 sentences. The main subject is the
> church, Gothic Revival style, low-angle wide shot, sunny day, flint walls,
> tall spire, black car in foreground, bright blue sky with wispy clouds. So
> the description could be "Low-angle wide shot of St Peter's Church in
> Petersfield, Hampshire, England, showcasing its Gothic Revival architecture,
> flint walls, tall spire, and a black car in the foreground under a bright
> blue sky with wispy clouds."
>
> Keywords: Need 10-18 unique terms. From the image: St Peter's Church,
> Petersfield, Hampshire, England, Gothic Revival, flint walls, tall spire,
> black car, blue sky, wispy clouds, low-angle, wide shot, sunny day, stone,
> church, chapel, cross, daylight, Dorking (since Dorking is a town in
> Hampshire), Europe, objects, station wagon (the car in foreground), fence.
> Let's check which are visible: "St Peter's Church", "Petersfield",
> "Hampshire", "England", "Gothic Revival", "flint walls", "tall spire",
> "black car", "blue sky", "wispy clouds", "low-angle", "wide shot", "sunny
> day", "stone", "church", "chapel", "cross", "daylight", "Dorking", "Europe",
> "objects", "station wagon", "fence". Let's pick the most visible and
> relevant: St Peter's Church, Petersfield, Hampshire, England, Gothic
> Revival, flint walls, tall spire, black car, blue sky, wispy clouds,
> low-angle, wide shot, sunny day, stone, church, chapel, cross, daylight,
> Dorking, Europe, station wagon, fence. That's 18 keywords.
>
> &#8203;Wait, need to ensure they are unique and clearly visible. Let's list them:
> St Peter
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description)
- Keyword count violation (68; expected 10-18)
- Keyword duplication (57% duplicated terms)
- Reasoning leak (◁think▷, let's analyze the image)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bell Tower, Chapel, Cross, Daylight, Dorking; nonvisual metadata
  reused
- _Score:_ 🟡 C (53/100); Δ-22; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 1.46s; Gen 10.51s; Total 12.23s
- _Throughput:_ Prompt 303 TPS (532 tok); Gen 19.2 TPS (161 tok)
- _Tokens:_ prompt 532 tok; estimated text 474 tok; estimated non-text 58 tok;
  generated 161 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> St Peter's Church in Petersfield, Hampshire, England
>
> The image depicts a low-angle, wide shot of St Peter's Church in
> Petersfield, Hampshire, England, on a sunny day. The church is a prominent
> structure with a tall spire and flint walls, situated against a bright blue
> sky with wispy clouds. In the foreground, a black car is parked, adding a
> sense of scale to the scene. The church's architecture is characterized by
> its Gothic Revival style, with pointed arches, ribbed vaults, and intricate
> stone carvings. The surrounding area is well-manicured, with a fence and a
> few trees visible. The overall atmosphere of the image is one of serenity
> and tranquility, with the warm sunlight casting a golden glow over the
> scene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bell Tower, Chapel, Cross, Daylight, Dorking; nonvisual metadata
  reused
- _Score:_ 🟠 D (47/100); Δ-28; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 3.00s; Gen 12.92s; Total 16.19s
- _Throughput:_ Prompt 200 TPS (533 tok); Gen 5.08 TPS (50 tok)
- _Tokens:_ prompt 533 tok; estimated text 474 tok; estimated non-text 59 tok;
  generated 50 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> St Peter's Church in Petersfield, Hampshire, England, a Gothic Revival-style
> church with a tall spire and flint walls, is pictured against a bright blue
> sky with wispy clouds. A black car is parked in the foreground.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=90%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Blue sky,
  Car, Chapel, Church
- _Score:_ 🟠 D (50/100); Δ-25; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.24s; Gen 13.71s; Total 15.23s
- _Throughput:_ Prompt 3,225 TPS (4,657 tok); Gen 43.1 TPS (500 tok)
- _Tokens:_ prompt 4657 tok; estimated text 474 tok; estimated non-text 4183
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the keyword list after the label.
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

- Repetitive output (phrase: "after the label. -...")
- Excessive bullet points (44)
- Context ignored (missing: Bell Tower, Blue sky, Car, Chapel, Church)
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
  sections: title, description, keywords; missing terms: Bell Tower, Blue sky,
  Chapel, Cross, Daylight
- _Score:_ 🟠 D (43/100); Δ-32; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections,
  reasoning leak)
- _Timing:_ Load 1.29s; Gen 14.97s; Total 16.53s
- _Throughput:_ Prompt 968 TPS (6,675 tok); Gen 65.1 TPS (500 tok)
- _Tokens:_ prompt 6675 tok; estimated text 474 tok; estimated non-text 6201
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's tackle this. First, I need to analyze the image for
> cataloguing metadata. The user provided a lot of context, but I need to
> focus on what's clearly visible.
>
> First, the Title. It needs to be 5-10 words, concrete and factual. Looking
> at the image, the main subject is a church with a tall spire. So a possible
> title is "St Peter's Church, Petersfield". But wait, the user said to use
> only details clearly visible. The original hint mentions St Peter's Church
> in Petersfield, Hampshire. But since the image doesn't show the name
> explicitly, but the context hint says that. Wait, the user's context
> includes the description hint which says "St Peter's Church in Petersfield,
> Hampshire, England". So even though the image doesn't show the name, the
> context hint is high confidence. But wait, the user's instructions say "Use
> only details that are clearly and definitely visible in the image. If a
> detail is uncertain, ambiguous, partially obscured, too small to verify, or
> not directly visible, leave it out." So the context hints are provided, but
> we must only use details clearly visible. Wait, the user's context includes
> "existing metadata hints (high confidence; use only when visually
> confirmed)". So we can use those hints if they are visually confirmed. But
> the image doesn't show the name "St Peter's Church", so we can't use that.
> So the title must be based on what's clearly visible. The main subject is a
> church with a tall spire. So a title like "Church with Tall Spire" or
> "Gothic Church with Spire". But the user wants 5-10 words. So "Church with
> Tall Spire" is good. But wait, the image shows a clock on the tower. So
> maybe "Church Tower with Clock" but the main feature is the tall spire. So
> "Church with Tall Spire" is a good title.
>
> Next, the Description. It needs to be 1-2 factual sentences. The main
> subject is a Gothic Revival church with a tall spire. The image is a
> low-angle, wide shot. The sky is bright blue with wispy clouds. In the
> foreground, there's a black car parked. The church has flint walls, clock
> faces on the tower, and Gothic-style windows. So putting this into a
> sentence: "A low-angle view of St Peter
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

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
- _Key signals:_ hit token cap (500); nontext prompt burden=86%; missing
  sections: description, keywords; missing terms: Bell Tower, Chapel, Cross,
  Daylight, Dorking
- _Score:_ ✅ B (71/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, metadata borrowing, missing
  sections, reasoning leak)
- _Timing:_ Load 1.59s; Gen 15.22s; Total 17.07s
- _Throughput:_ Prompt 1,437 TPS (3,457 tok); Gen 40.4 TPS (500 tok)
- _Tokens:_ prompt 3457 tok; estimated text 474 tok; estimated non-text 2983
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here are my reasoning steps:
> We need to produce three sections: Title, Description, Keywords. Use only
> details that are clearly visible in the image. The image shows a tall Gothic
> Revival church with a spire, flint walls, a black car in foreground, a
> fence, green trees, blue sky with wispy clouds. The church appears to be St
> Peter's Church in Petersfield, Hampshire, England. However, we must not
> infer identity unless visually obvious. The image does not have a sign
> indicating the name. So we cannot assume it's St Peter's Church. The
> description hint says it's St Peter's Church in Petersfield, but we cannot
> confirm that from the image alone. The image shows a Gothic Revival church
> with a tall spire, flint walls, a black car, a fence, green trees, blue sky.
> The location is not indicated. So we must not include location or name
> unless clearly visible. The image does not show any signage with the name.
> So we must not include "St Peter's Church" or "Petersfield" or "Hampshire".
> The description hint says "A low-angle, wide shot of St Peter's Church in
> Petersfield, Hampshire, England, on a sunny day. The Gothic Revival style
> church, with its tall spire and flint walls, is pictured against a bright
> blue sky with wispy clouds. A black car is parked in the foreground." The
> image indeed shows a low-angle wide shot, a tall spire, flint walls, bright
> blue sky with wispy clouds, a black car. So we can include those details.
> But we cannot include location or name. So we can say "A low-angle wide shot
> of a Gothic Revival church with a tall spire and flint walls, set against a
> bright blue sky with wispy clouds, with a black car in the foreground."
> That is a factual description.
>
> Title: 5-10 words, concrete and factual, limited to clearly visible content.
> Something like "Gothic church spire, flint walls, blue sky, black car".
> That's 9 words? Let's count: "Gothic" (1) "church" (2) "spire," (3) "flint"
> (4) "walls," (5) "blue" (6) "sky," (7) "black" (8) "car" (9). That's
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 's')
- Missing sections (description, keywords)
- Title length violation (46 words; expected 5-10)
- Reasoning leak (here are my reasoning steps)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking
- _Score:_ ✅ B (67/100); Δ-8; Missing requested structure
- _Review focus:_ watchlist (cutoff, degeneration, missing sections, reasoning
  leak)
- _Timing:_ Load 1.39s; Gen 15.96s; Total 17.62s
- _Throughput:_ Prompt 1,109 TPS (6,675 tok); Gen 52.3 TPS (500 tok)
- _Tokens:_ prompt 6675 tok; estimated text 474 tok; estimated non-text 6201
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, I need to extract the
> Title, Description, and Keywords based on what's clearly visible in the
> image.
>
> Starting with the Title. The instructions say 5-10 words, concrete and
> factual, limited to clearly visible content. Looking at the image: there's a
> church with a tall steeple. So possible titles: "Church with tall steeple"
> or "Gothic church with steeple". Need to keep it factual and visible. The
> main subject is a church with a tall steeple. So Title could be "Church with
> tall steeple".
>
> Next, Description. Need 1-2 factual sentences describing main subject,
> setting, lighting, action, distinctive details. From the image: low-angle
> shot, church with tall steeple, flint walls (maybe visible as stone
> texture), clock faces on tower, arched windows, bright blue sky with wispy
> clouds, black car in foreground, green bushes around the church.
>
> Need to pick the most clearly visible details. So: "A low-angle view of a
> Gothic Revival church with a tall steeple, flint and stone walls, and clock
> faces on the tower. The church is set against a bright blue sky with wispy
> clouds, with a black car parked in the foreground and green foliage around
> the building."
>
> Wait, but need to check if "Gothic Revival" is definitely visible? The image
> shows Gothic features like pointed arches, steeple, etc., so "Gothic
> Revival" is a reasonable inference based on visible features (pointed
> arches, steeple, tracery windows). But wait, the instructions say "omit
> anything uncertain or inferred". So "Gothic Revival" might be an inference,
> not directly visible. So need to adjust the description to only include
> what's clearly visible.
>
> &#8203;Let's re-examine the image details that are definitely visible:
>
> &#45; Main subject: A church with a tall steeple.
> &#45; Architectural features: Pointed arches on windows and doors, decorative
> stone tracery on windows, clock faces on the tower (two visible clocks),
> flint and stone construction (the texture of the walls is a mix of dark
> flint and lighter stone).
> &#45; Setting: The church is outdoors, with a bright blue sky with some wispy
> clouds. In the foreground, there's a black car parked (a station wagon or
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'or')
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Blue sky,
  Car, Chapel, Church
- _Score:_ 🟠 D (44/100); Δ-31; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, missing
  sections)
- _Timing:_ Load 1.29s; Gen 27.34s; Total 28.88s
- _Throughput:_ Prompt 102 TPS (1,748 tok); Gen 52.7 TPS (500 tok)
- _Tokens:_ prompt 1748 tok; estimated text 474 tok; estimated non-text 1274
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Camera: Canon EOS 100D Mark III, ISO 400, 1/125s, f/2.8, 50mm f/2.8. Lens:
> Canon EF 50mm f/2.8 STM. No flash used. 100% original, unaltered,
> high-quality image.
> &#45; Image ID:
> 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bell Tower, Blue sky, Car, Chapel, Church)
- Output degeneration (character_loop: '0' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: Bell Tower, Chapel, wide, shot, sunny; keyword duplication=74%
- _Score:_ ✅ B (78/100); Δ+3; None identified
- _Review focus:_ watchlist (cutoff, metadata borrowing)
- _Timing:_ Load 1.22s; Gen 27.44s; Total 28.92s
- _Throughput:_ Prompt 963 TPS (16,789 tok); Gen 53.5 TPS (500 tok)
- _Tokens:_ prompt 16789 tok; estimated text 474 tok; estimated non-text 16315
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> St Peter's Church, Petersfield, Hampshire
>
> &#8203;Description:
> A Gothic Revival church with a tall spire and flint walls is captured from a
> low angle, emphasizing its height against a bright blue sky. A black car is
> parked in the foreground, and the scene is illuminated by daylight.
>
> &#8203;Keywords:
> Church, Gothic Revival, Spire, Flint, Blue sky, Black car, Daylight,
> Hampshire, England, Steeple, Stone, Church tower, Clock, Cross, Fence,
> Trees, Pathway, Clear sky, Architectural detail, Historical building,
> England, Europe, Petersfield, Hampshire, England, Europe, Flint walls,
> Steeple, Gothic architecture, Church steeple, Black car, Car, Daytime, Clear
> weather, Blue sky, Church tower, Gothic style, Flint church, Church spire,
> Black vehicle, Car parked, Church building, Gothic architecture, Flint
> church, Church steeple, Black car, Car parked, Church building, Gothic
> architecture, Flint church, Church spire, Black vehicle, Car, Daylight,
> Clear sky, Church, Gothic Revival, Spire, Flint, Blue sky, Black car,
> Daylight, Dorking, England, Europe, Fence, Gothic Architecture, Objects,
> Sky, Station wagon, Steeple, Stone, Surrey, England, Europe, Flint walls,
> Steeple, Gothic architecture, Church tower, Black car, Car, Daylight, Clear
> sky, Church building, Gothic style, Flint church, Church spire, Black
> vehicle, Car parked, Church, Gothic Revival, Spire, Flint, Blue sky, Black
> car, Daylight, Dorking, England, Europe, Fence, Gothic Architecture,
> Objects, Sky, Station wagon, Steeple, Stone, Surrey, England, Europe, Flint
> walls, Steeple, Gothic architecture, Church tower, Black car, Car, Daylight,
> Clear sky, Church building, Gothic style, Flint church, Church spire, Black
> vehicle, Car parked, Church, Gothic Revival, Spire, Flint, Blue sky, Black
> car, Daylight, Dorking, England, Europe, Fence, Gothic Architecture,
> Objects, Sky, Station wagon, Steeple, Stone, Surrey, England, Europe, Flint
> walls, Steeple, Gothic architecture, Church tower, Black car, Car, Daylight,
> Clear sky
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (151; expected 10-18)
- Keyword duplication (74% duplicated terms)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=93%; missing
  sections: title, description; missing terms: Bell Tower, Chapel, Cross,
  Daylight, Dorking
- _Score:_ ✅ B (70/100); Δ-5; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections, reasoning leak)
- _Timing:_ Load 5.55s; Gen 29.49s; Total 35.30s
- _Throughput:_ Prompt 490 TPS (6,675 tok); Gen 32.4 TPS (500 tok)
- _Tokens:_ prompt 6675 tok; estimated text 474 tok; estimated non-text 6201
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, the Title. It needs to
> be 5-10 words, concrete and factual. The image shows a Gothic Revival
> church with a tall spire, flint walls, a black car in the foreground, sunny
> day. Wait, the existing hint mentions St Peter's Church in Petersfield, but
> do we know that for sure? The user said not to infer location unless
> visually obvious. The image doesn't show a sign or name, so we can't include
> &#8203;that. So the title should be about the church's features. Let's see:
> "Gothic Revival church with tall spire and flint walls, black car in
> foreground". Wait, no, need to be more concise. Maybe "Tall Gothic Revival
> church spire with flint walls, black car parked". Wait, no, let's check the
> image: the main subject is the church, so title should be "Gothic Revival
> church with tall spire, flint walls, and black car in foreground". But need
> 5-10 words. Let's make it "Tall Gothic Revival church spire, flint walls,
> black car parked". Wait, maybe better: "Gothic Revival church with tall
> spire, flint walls, black car". Yes, that's 7 words.
>
> Next, Description. Need 1-2 factual sentences. The image shows a Gothic
> Revival church with a tall spire, flint walls, a black car in the
> foreground, sunny day with blue sky and wispy clouds. The church has clock
> faces on the tower, arched windows, and a fenced area with greenery. Let's
> put that together: "A Gothic Revival church with a tall spire and flint
> walls is pictured against a bright blue sky with wispy clouds. A black car
> is parked in the foreground, and the church features clock faces on its
> tower, arched windows, and a fenced area with greenery." Wait, but do we see
> the clock faces? Yes, the image shows two clock faces on the tower. Do we
> see the arched windows? Yes, the church has arched windows along the side.
> The fenced area with greenery is visible. So that's good.
>
> Keywords: Need 10-18 unique terms. Let's list what's visible: Church, Gothic
> Architecture, Steeple, Flint, Stone, Blue sky,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description)
- Keyword count violation (6; expected 10-18)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=73%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Blue sky,
  Car, Chapel, Church
- _Score:_ 🟠 D (45/100); Δ-30; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections,
  repetitive)
- _Timing:_ Load 1.70s; Gen 34.87s; Total 36.83s
- _Throughput:_ Prompt 99.2 TPS (1,748 tok); Gen 30.2 TPS (500 tok)
- _Tokens:_ prompt 1748 tok; estimated text 474 tok; estimated non-text 1274
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Camera: Canon EOS 100D Mark III, ISO 400, 1/125s, f/2.8, 50mm f/2.8. 100%
> shutter release, no remote shutter release used. Tripod: Not used. Tripod
> head: Manfrotto Bubbalong 2.5G. Tripod legs: Manfrotto Bubbalong 2.5G.
> Tripod head: Manfrotto Bubbalong 2.5G. Tripod legs: Manfrotto Bubbalong
> 2.5G. Tripod head: Manfrotto Bubbalong 2.5G. Tripod legs: Manfrotto
> Bubbalong 2.5G. Tripod head: Manfrotto Bubbalong 2.5G. Tripod legs:
> Manfrotto Bubbalong 2.5G. Tripod head: Manfrotto Bubbalong 2.5G. Tripod
> legs: Manfrotto Bubbalong 2.5G. Tripod head: Manfrotto Bubbalong 2.5G.
> Tripod legs: Manfrotto Bubbalong 2.5G. Tripod head: Manfrotto Bubbalong
> 2.5G. Tripod legs: Manfrotto Bubbalong 2.5G. Tripod head: Manfrotto
> Bubbalong 2.5G. Tripod legs: Manfrotto Bubbalong 2.5G. Tripod head:
> Manfrotto Bubbalong 2.5G. Tripod legs: Manfrotto Bubbalong 2.5G. Tripod
> head: Manfrotto Bubbalong 2.5G. Tripod legs: Manfrotto Bubbalong 2.5G.
> Tripod head: Manfrotto Bubbalong 2.5G. Tripod legs: Manfrotto Bubbalong
> 2.5G. Tripod head: Manfrotto Bubbalong 2.5G. Tripod legs: Manfrotto
> Bubbalong 2.5G. Tripod head: Manf
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "manfrotto bubbalong 2.5g. trip...")
- Context ignored (missing: Bell Tower, Blue sky, Car, Chapel, Church)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Cross, Dorking, Gothic Architecture
- _Score:_ ✅ B (78/100); Δ+3; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections)
- _Timing:_ Load 2.35s; Gen 55.94s; Total 58.55s
- _Throughput:_ Prompt 334 TPS (16,804 tok); Gen 102 TPS (500 tok)
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

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
> &#8203;&#42;&#42;Analysis of the Image:&#42;&#42;
> &#45; &#42;&#42;Subject:&#42;&#42; A large stone church with a tall spire. It looks like a
> Gothic Revival style.
> &#45; &#42;&#42;Material:&#42;&#42; The walls are made of flint (dark grey/black pebbles) with
> stone quoins and dressings (lighter stone). This is a very specific
> architectural feature.
> &#45; &#42;&#42;Features:&#42;&#42;
> &#45; Tall spire with a weather vane on top.
> &#45; Clock faces on the tower (visible on at least two sides).
> &#45; Arched windows (lancet windows) along the nave.
> &#45; Buttresses.
> &#45; A black car (station wagon/estate) parked in the foreground.
> &#45; Green bushes/hedges in front of the church.
> &#45; A black metal fence.
> &#45; Blue sky with wispy clouds.
> &#45; &#42;&#42;Lighting:&#42;&#42; Sunny, bright daylight. Shadows are cast on the left side of
> the tower, suggesting the sun is to the right.
> &#45; &#42;&#42;Setting:&#42;&#42; Outdoors, likely a churchyard or street next to the church.
>
> &#8203;&#42;&#42;Drafting the Title:&#42;&#42;
> &#45; Needs to be 5-10 words.
> &#45; Concrete and factual.
> &#45; Draft 1: Low angle view of a stone church with a tall spire and black car.
> (12 words - too long)
> &#45; Draft 2: Stone church with tall spire and black car parked outside. (11
> words - too long)
> &#45; Draft 3: Gothic stone church with tall spire and black car. (9 words) -
> Good.
> &#45; Draft 4: Tall stone church spire against blue sky with car. (9 words) -
> Good.
> &#45; Let's go with: Tall stone church spire against blue sky with car.
>
> &#8203;&#42;&#42;Drafting the Description:&#42;&#42;
> &#45; 1-2 factual sentences.
> &#45; Main subject: Church with spire.
> &#45; Setting: Blue sky, sunny day.
> &#45; Foreground: Black car, green bushes, fence.
> &#45; Details: Flint walls, stone dressings,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Excessive bullet points (30)
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
  generated text.; Output appears truncated to about 5 tokens.; nontext prompt
  burden=97%; missing terms: Bell Tower, Blue sky, Car, Chapel, Church
- _Score:_ ❌ F (5/100); Δ-70; Output too short to be useful
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 0.55s; Gen 58.03s; Total 58.84s
- _Throughput:_ Prompt 293 TPS (16,789 tok); Gen 247 TPS (5 tok)
- _Tokens:_ prompt 16789 tok; estimated text 474 tok; estimated non-text 16315
  tok; generated 5 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|endoftext|><|endoftext|><|endoftext|><|endoftext|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Context ignored (missing: Bell Tower, Blue sky, Car, Chapel, Church)
- Ignores trusted hints
- tokenizer_artifact
- token_leak:<\|endoftext\|>
- output:truncated(5tok)
- long_context_low_ratio(0.0%;16789->5)

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Chapel, Cross,
  Daylight, Gothic Architecture, Objects
- _Score:_ ✅ B (79/100); Δ+4; Missing requested structure
- _Review focus:_ watchlist (cutoff, instruction echo, metadata borrowing,
  missing sections, reasoning leak, refusal)
- _Timing:_ Load 3.25s; Gen 58.68s; Total 62.22s
- _Throughput:_ Prompt 321 TPS (16,804 tok); Gen 87.8 TPS (500 tok)
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to generate cataloguing metadata for the provided image.
> &#8203;I need to follow specific rules:
> &#45; British English.
> &#45; Only clearly visible details.
> &#45; Three sections: Title, Description, Keywords.
> &#45; Specific constraints on length and content for each section.
> &#45; Use the provided hints only if confirmed by the image.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Subject:&#42;&#42; A large stone church with a tall spire. It looks like a
> Gothic Revival style.
> &#45; &#42;&#42;Material:&#42;&#42; The walls are a mix of flint (dark, pebbly texture) and
> light-coloured stone quoins and dressings.
> &#45; &#42;&#42;Features:&#42;&#42;
> &#45; Tall spire with pinnacles at the corners.
> &#45; Clock faces on the tower (two visible).
> &#45; Arched windows (lancet style).
> &#45; Buttresses.
> &#45; A lower section of the church extends to the right.
> &#45; &#42;&#42;Foreground:&#42;&#42;
> &#45; A dark grey or black car (looks like a station wagon or estate car) is
> partially visible at the bottom.
> &#45; Stone steps leading up to the church entrance on the left.
> &#45; A black metal fence running along the bottom right.
> &#45; Green bushes/hedges in front of the church wall.
> &#45; &#42;&#42;Background/Sky:&#42;&#42;
> &#45; Bright blue sky.
> &#45; Wispy white clouds (cirrus).
> &#45; &#42;&#42;Lighting:&#42;&#42; Sunny, casting shadows on the left side of the tower
> (suggesting sun is to the right).
>
> &#8203;&#42;&#42;Evaluating Hints:&#42;&#42;
> &#45; &#42;Description hint:&#42; "St Peter's Church in Petersfield, Hampshire, England"
> -> I cannot confirm the specific name "St Peter's" or location
> "Petersfield" just from looking at the image. It looks like a church, but
> the specific identity is external knowledge. I must stick to "clearly
> visible". I will describe it as a stone church with a spire.
> &#45; &#42;Keyword hints:&#42; "Dorking", "Surrey" -> These are specific locations. I
> cannot confirm them visually. I should omit them unless I see a sign. I see
> no sign. "Bell Tower" -> It's a tower, likely a bell tower, but "Tower" is
> safer. "Station wagon" -> The car looks like a station wagon/estate. "Adobe
> Stock", "Any Vision" -> These are watermarks
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Refusal detected (explicit_refusal)
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:, keyword hints:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking
- _Score:_ ✅ B (67/100); Δ-8; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections, refusal)
- _Timing:_ Load 6.30s; Gen 68.12s; Total 74.68s
- _Throughput:_ Prompt 282 TPS (16,804 tok); Gen 63.6 TPS (500 tok)
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to generate cataloguing metadata for the provided image of
> a church.
> &#8203;I need to follow specific rules:
> &#45; Use British English.
> &#45; Only use clearly visible details. No guessing.
> &#45; Follow the provided structure: Title, Description, Keywords.
> &#45; Correct or discard the provided metadata hints if they contradict the
> image or are uncertain.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Subject:&#42;&#42; A large stone church with a tall spire. It looks like a
> Gothic Revival style.
> &#45; &#42;&#42;Material:&#42;&#42; The walls are a mix of flint (dark, pebbly texture) and
> light-coloured stone quoins and dressings. This is a very specific
> architectural feature (flint and stone).
> &#45; &#42;&#42;Features:&#42;&#42;
> &#45; Tall spire with pinnacles at the corners.
> &#45; Clock faces on the tower.
> &#45; Arched windows (lancet windows).
> &#45; Buttresses.
> &#45; A weather vane on top of the spire.
> &#45; &#42;&#42;Setting:&#42;&#42; Blue sky with wispy cirrus clouds.
> &#45; &#42;&#42;Foreground:&#42;&#42; A black car (looks like a station wagon or estate car) is
> partially visible at the bottom. There are green bushes/hedges in front of
> the church. A metal fence is visible. Stone steps lead up to the church
> entrance on the left.
> &#45; &#42;&#42;Lighting:&#42;&#42; Bright sunlight, casting shadows. The sun seems to be coming
> from the right/front based on the shadows on the tower.
> &#45; &#42;&#42;Location/Identity:&#42;&#42; The prompt hints mention "St Peter's Church in
> Petersfield, Hampshire". I need to check if I can confirm this. The
> architecture (flint and stone) is very typical of Hampshire/Sussex churches.
> However, without external knowledge, I cannot &#42;definitely&#42; confirm it is St
> Peter's Petersfield just from the pixels. The prompt says "Treat the
> metadata hints below as a draft catalog record... correct anything
> contradicted by the image". The hint says "St Peter's Church in
> Petersfield". I should probably be careful. Is it definitely St Peter's? I
> can't be 100% sure just from looking, but the prompt implies I should use
> the hints &#42;if&#42; visually confirmed. Wait, the prompt says "Reuse metadata
> terms only when they are clearly supported by the image." If I don't know
> the name, I shouldn't use it. However, the prompt &#42;gives&#42;
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Refusal detected (explicit_refusal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Daylight, Dorking, Gothic Architecture
- _Score:_ ✅ B (66/100); Δ-9; Missing requested structure
- _Review focus:_ watchlist (cutoff, degeneration, instruction echo, metadata
  borrowing, missing sections, reasoning leak, refusal)
- _Timing:_ Load 1.37s; Gen 71.40s; Total 73.04s
- _Throughput:_ Prompt 262 TPS (16,804 tok); Gen 77.3 TPS (500 tok)
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to analyze the provided image and generate cataloguing
> metadata in British English.
> &#8203;I need to follow specific rules:
> &#45; Use only details clearly visible in the image.
> &#45; Do not guess or infer.
> &#45; Use the provided hints as a draft but correct/omit anything not visible.
> &#45; Output exactly three sections: Title, Description, Keywords.
>
> &#8203;&#42;&#42;Analysis of the Image:&#42;&#42;
> &#45; &#42;&#42;Subject:&#42;&#42; A large church building. It has a very tall, pointed spire.
> The walls are made of flint (pebbledash) with stone dressings. There are
> clock faces on the tower. There are arched windows (Gothic style).
> &#45; &#42;&#42;Setting:&#42;&#42; Outdoors, blue sky with some wispy clouds. There are green
> bushes/trees in the foreground/mid-ground. A black car is parked in the
> foreground. A fence is visible.
> &#45; &#42;&#42;Lighting:&#42;&#42; Bright sunlight, casting shadows. Looks like a sunny day.
> &#45; &#42;&#42;Architecture:&#42;&#42; Gothic Revival style (pointed arches, spire, flint
> walls).
> &#45; &#42;&#42;Specific Details:&#42;&#42;
> &#45; Tall spire with a weather vane on top.
> &#45; Clock faces on the tower (one clearly visible on the right face,
> another partially visible on the left).
> &#45; Flint walls with stone quoins and dressings.
> &#45; Arched windows along the nave.
> &#45; A black car (looks like a station wagon or estate) in the foreground.
> &#45; Green foliage (bushes/trees).
> &#45; A metal fence.
> &#45; Stone steps leading up to the church entrance (bottom left).
> &#45; A cross on top of the spire (weather vane).
>
> &#8203;&#42;&#42;Evaluating the Hints:&#42;&#42;
> &#45; &#42;&#42;Description hint:&#42;&#42; "A low-angle, wide shot of St Peter's Church in
> Petersfield, Hampshire, England, on a sunny day. The Gothic Revival style
> church, with its tall spire and flint walls, is pictured against a bright
> blue sky with wispy clouds. A black car is parked in the foreground."
> &#45; &#42;Correction/Omission:&#42; I cannot confirm the specific name "St Peter's
> Church" or location "Petersfield, Hampshire, England" just from looking at
> the image. I must omit these. The rest (low-angle, wide shot, sunny day,
> Gothic Revival, tall spire, fl
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Refusal detected (explicit_refusal)
- Output degeneration (incomplete_sentence: ends with 'fl')
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking
- _Score:_ ✅ B (80/100); Δ+5; Missing requested structure
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections,
  refusal)
- _Timing:_ Load 2.13s; Gen 84.63s; Total 87.04s
- _Throughput:_ Prompt 250 TPS (16,804 tok); Gen 29.9 TPS (500 tok)
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to analyze the image and generate cataloguing metadata
> based on specific rules.
>
> &#42;&#42;1. Analyze the Image:&#42;&#42;
> &#42;   &#42;&#42;Subject:&#42;&#42; A large, historic church with a tall, pointed spire.
> &#42;   &#42;&#42;Architecture:&#42;&#42; It looks like Gothic or Gothic Revival. The walls are
> made of flint (dark grey/black pebbles) with stone quoins (lighter stone
> corners and window surrounds). There are arched windows.
> &#42;   &#42;&#42;Spire:&#42;&#42; Very tall, slender, stone spire.
> &#42;   &#42;&#42;Clock:&#42;&#42; There are clock faces visible on the tower.
> &#42;   &#42;&#42;Sky:&#42;&#42; Bright blue sky with some wispy white clouds.
> &#42;   &#42;&#42;Foreground:&#42;&#42; A dark grey or black car (station wagon/estate) is
> parked in the immediate foreground, partially obscuring the base. There are
> green bushes/hedges.
> &#42;   &#42;&#42;Fence:&#42;&#42; A black metal fence runs along the side.
> &#42;   &#42;&#42;Lighting:&#42;&#42; Bright sunlight, casting strong shadows on the building
> (suggesting late afternoon or early morning given the angle).
> &#42;   &#42;&#42;Location clues:&#42;&#42; The prompt mentions "St Peter's Church in
> Petersfield, Hampshire". I need to verify if I can confirm this visually. I
> can't read a sign saying "St Peter's". However, the architecture matches the
> description. The prompt says "Use only details that are clearly and
> definitely visible". I cannot confirm the specific name "St Peter's" or
> "Petersfield" just by looking at the building without external knowledge,
> but the prompt provides "Existing metadata hints" which are "high
> confidence". The rule says "If metadata and image disagree, follow the
> image." The image shows a church. The hint says it's St Peter's. I should
> probably stick to generic "church" unless I can read a sign. Wait, the
> prompt says "Treat the metadata hints below as a draft catalog record. Keep
> only details that are clearly confirmed by the image". I cannot confirm the
> specific name "St Peter's" from the pixels alone. I will stick to "church".
> &#42;   &#42;&#42;Car:&#42;&#42; It's a dark grey/black station wagon (estate car).
> &#42;   &#42;&#42;Time:&#42;&#42; The prompt says 17:50. The shadows are long, consistent with
> late afternoon.
>
> &#42;&#42;2. Draft Title:&#42;&#42;
> &#42;   Needs to be 5-10 words.
> &#42;   Concrete and factual.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Refusal detected (explicit_refusal)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  terms: Bell Tower, Chapel, Cross, Daylight, Dorking; keywords=32
- _Score:_ ✅ B (78/100); Δ+3; None identified
- _Review focus:_ watchlist (cutoff, refusal)
- _Timing:_ Load 3.08s; Gen 104.59s; Total 107.95s
- _Throughput:_ Prompt 224 TPS (16,804 tok); Gen 17.4 TPS (500 tok)
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to generate cataloguing metadata for the provided image of
> a church.
> &#8203;I need to follow specific rules:
> &#45; Use British English.
> &#45; Only include clearly visible details.
> &#45; Do not guess or infer uncertain information.
> &#45; Output exactly three sections: Title, Description, Keywords.
> &#45; Title: 5-10 words, concrete and factual.
> &#45; Description: 1-2 factual sentences.
> &#45; Keywords: 10-18 unique, comma-separated terms.
>
> &#8203;&#42;&#42;Image Analysis:&#42;&#42;
> &#45; &#42;&#42;Subject:&#42;&#42; A large stone church with a tall, pointed spire. It looks
> like a traditional English parish church.
> &#45; &#42;&#42;Architecture:&#42;&#42; Gothic style (pointed arches, spire, pinnacles). The
> walls are made of flint (dark, pebbly texture) with stone dressings (lighter
> stone around windows and corners).
> &#45; &#42;&#42;Features:&#42;&#42;
> &#45; Tall spire with a weather vane on top.
> &#45; Clock faces on the tower (two visible).
> &#45; Arched windows (lancet windows).
> &#45; Pinnacles on the tower corners.
> &#45; A lower section of the church (nave/chancel) extending to the right.
> &#45; A black car (station wagon/estate car) is parked in the foreground,
> partially visible.
> &#45; A metal fence runs along the bottom right.
> &#45; Green bushes/trees in front of the church.
> &#45; Steps leading up to the church on the left.
> &#45; &#42;&#42;Setting/Environment:&#42;&#42;
> &#45; Bright blue sky with wispy white clouds (cirrus).
> &#45; Sunny day (strong shadows, bright lighting).
> &#45; Paved area in the foreground.
> &#45; &#42;&#42;Location/Identity:&#42;&#42; The prompt hints suggest "St Peter's Church in
> Petersfield, Hampshire". I need to be careful. The prompt says "Use only
> details that are clearly and definitely visible". I cannot &#42;see&#42; a sign
> saying "St Peter's Church". However, the prompt context says "Existing
> metadata hints (high confidence; use only when visually confirmed)". The
> hint says "St Peter's Church in Petersfield". Is this visually confirmed?
> No, not by a sign. But the visual features match the description of St
> Peter's, Petersfield (flint walls, tall spire). Wait, the rule says "Do not
> infer identity... unless visually obvious." There
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Refusal detected (explicit_refusal)
- Keyword count violation (32; expected 10-18)
- Likely capped by max token budget

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=69%; missing
  sections: title, description; missing terms: Bell Tower, Daylight, Gothic
  Architecture, Steeple, its
- _Score:_ ✅ B (70/100); Δ-5; Missing requested structure
- _Review focus:_ watchlist (cutoff, degeneration, metadata borrowing, missing
  sections, reasoning leak)
- _Timing:_ Load 3.24s; Gen 108.51s; Total 112.02s
- _Throughput:_ Prompt 1,057 TPS (1,553 tok); Gen 4.7 TPS (500 tok)
- _Tokens:_ prompt 1553 tok; estimated text 474 tok; estimated non-text 1079
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image based on the given instructions.
> First, the title needs to be 5-10 words, concrete and factual. The main
> visible elements are St Peter's Church in Petersfield, Hampshire, England,
> with a tall spire, flint walls, a black car in the foreground, and a bright
> blue sky. So the title could be "St Peter's Church, Petersfield, Hampshire,
> England, with black car, spire, flint walls, bright blue sky".
>
> For the description, it should be 1-2 factual sentences. The main subject is
> the church, Gothic Revival style, low-angle wide shot, sunny day, black car
> in foreground, flint walls, spire, bright blue sky with wispy clouds. So
> the description could be "Low-angle wide shot of St Peter's Church in
> Petersfield, Hampshire, England, featuring Gothic Revival architecture,
> flint walls, tall spire, black car in foreground, and bright blue sky with
> wispy clouds."
>
> Keywords: Need 10-18 unique terms. From the image: St Peter's Church,
> Petersfield, Hampshire, England, Gothic Revival, flint walls, tall spire,
> black car, bright blue sky, wispy clouds, low-angle, wide shot, stone,
> church, cross (maybe, but not sure if cross is visible), chapel, Dorking
> (from hints but not visible here), Adobe Stock (watermark), Any Vision
> (watermark), Station wagon (black car is a station wagon), Europe, Fence,
> Objects, Sky, Surrey (from hints but not visible). Wait, need to check
> what's visible. The image has the church, car, sky, trees, fence. So
> keywords: St Peter's Church, Petersfield, Hampshire, England, Gothic
> Revival, flint walls, tall spire, black car, bright blue sky, wispy clouds,
> low-angle, wide shot, stone, church, fence, trees, stone architecture,
> church spire, church exterior, outdoor, sunny day. Let's pick the most
> visible ones.
>
> Wait, the hints had some terms like Dorking, Surrey, Adobe Stock, Any
> Vision, but those are watermarks or stock agency terms, not visible in the
> image. So exclude those. The image shows the church, car, sky, trees, fence.
> So keywords: St Peter's Church, Petersfield, Hampshire, England, Gothic
> Revival, fl
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'fl')
- Missing sections (title, description)
- Keyword count violation (67; expected 10-18)
- Keyword duplication (36% duplicated terms)
- Reasoning leak (◁think▷, let's analyze the image)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); nontext prompt burden=97%; missing
  sections: title, description, keywords; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking
- _Score:_ 🟡 C (56/100); Δ-19; Missing requested structure
- _Review focus:_ watchlist (cutoff, instruction echo, metadata borrowing,
  missing sections, reasoning leak, refusal)
- _Timing:_ Load 3.12s; Gen 118.30s; Total 121.71s
- _Throughput:_ Prompt 190 TPS (16,804 tok); Gen 17.3 TPS (500 tok)
- _Tokens:_ prompt 16804 tok; estimated text 474 tok; estimated non-text 16330
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants me to analyze the image and generate metadata based on
> specific rules.
>
> &#42;&#42;1. Analyze the Image:&#42;&#42;
> &#42;   &#42;&#42;Subject:&#42;&#42; A large church with a very tall, pointed spire.
> &#42;   &#42;&#42;Architecture:&#42;&#42; Gothic style (pointed arches, tracery). The walls are
> made of flint (small grey stones) with stone quoins (corner stones) and
> dressings. There are clock faces on the tower.
> &#42;   &#42;&#42;Setting:&#42;&#42; Outdoors, sunny day. Blue sky with some wispy clouds.
> &#42;   &#42;&#42;Foreground:&#42;&#42; A black car (station wagon/estate) is parked at the
> bottom. There are green bushes/hedges. A metal fence is visible.
> &#42;   &#42;&#42;Lighting:&#42;&#42; Bright sunlight, casting shadows on the church walls
> (suggesting late afternoon or morning sun, but the prompt says 17:50, which
> is late afternoon). The light is coming from the left.
> &#42;   &#42;&#42;Details:&#42;&#42; Weather vane on top of the spire. Statues or figures on the
> corners of the tower (gargoyles or saints).
>
> &#42;&#42;2. Evaluate Metadata Hints:&#42;&#42;
> &#42;   &#42;Description hint:&#42; "St Peter's Church in Petersfield, Hampshire,
> England".
> &#42;   &#42;Check:&#42; The GPS coordinates (51.215500°N, 0.798500°W) point to
> Petersfield, Hampshire. The visual matches St Peter's Church in Petersfield
> (distinctive flint and stone, tall spire). However, the prompt says "Do not
> infer identity... unless visually obvious." While the GPS confirms the
> location, I should stick to visual facts primarily, but the hint is "high
> confidence". Wait, the prompt says "Do not infer identity... unless visually
> obvious." Is the name "St Peter's" visible? No. Is "Petersfield" visible?
> No. But the GPS is provided in the context. The prompt says "Use only
> details that are clearly and definitely visible in the image."
> &#42;   &#42;Correction:&#42; I cannot confirm the specific name "St Peter's" or
> "Petersfield" just by looking at the pixels, &#42;unless&#42; I trust the
> GPS/Context provided. The prompt says "Treat the metadata hints below as a
> draft catalog record... Keep only details that are clearly confirmed by the
> image".
> &#42;   &#42;Decision:&#42; I
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Refusal detected (explicit_refusal)
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget
- suspicious_precision: 2 overly precise numbers

---

<!-- markdownlint-enable MD033 MD034 -->
