# Model Output Gallery

<!-- markdownlint-disable MD013 -->

_Generated on 2026-03-22 02:17:59 GMT_

A review-friendly artifact with image metadata, the source prompt, and full generated output for each model.

## 🎯 Action Snapshot

- **Framework/runtime failures:** 8 (top owners: transformers=5, mlx-vlm=2, model-config=1).
- **Next action:** review failure ownership below and use diagnostics.md for filing.
- **Maintainer signals:** harness-risk successes=8, clean outputs=3/43.
- **Useful now:** 1 clean A/B model(s) worth first review.
- **Review watchlist:** 42 model(s) with breaking or lower-value output.
- **Preflight compatibility:** 1 informational warning(s); do not treat these alone as run failures.
- **Escalate only if:** they line up with unexpected TF/Flax/JAX imports, startup hangs, or backend/runtime crashes.
- **Vs existing metadata:** better=7, neutral=1, worse=35 (baseline A 80/100).
- **Quality signal frequency:** missing_sections=27, cutoff=21, trusted_hint_ignored=19, context_ignored=18, metadata_borrowing=12, repetitive=9.
- **Runtime pattern:** decode dominates measured phase time (83%; 45/51 measured model(s)).
- **Phase totals:** model load=98.63s, prompt prep=0.12s, decode=535.10s, cleanup=7.26s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=43, exception=8.

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (90/100) | Δ+10 | 175.3 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) | Δ-80 | harness
- `prince-canuma/Florence-2-large-ft`: ❌ F (0/100) | Δ-80 | 357.3 tps | context ignored, degeneration, harness, missing sections
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) | Δ-73 | 21.7 tps | harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (17/100) | Δ-63 | 5.7 tps | context ignored, harness
- `microsoft/Phi-3.5-vision-instruct`: ✅ B (74/100) | Δ-6 | 55.3 tps | harness, metadata borrowing

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `transformers` | 5 | Model Error | `microsoft/Florence-2-large-ft`, `mlx-community/Qwen3.5-27B-4bit`, `mlx-community/Qwen3.5-27B-mxfp8`, `mlx-community/Qwen3.5-35B-A3B-6bit`, `mlx-community/Qwen3.5-35B-A3B-bf16` |
| `mlx-vlm` | 2 | Model Error | `mlx-community/InternVL3-8B-bf16`, `mlx-community/Molmo-7B-D-0924-bf16` |
| `model-config` | 1 | Processor Error | `mlx-community/deepseek-vl2-8bit` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### transformers

- **microsoft/Florence-2-large-ft** (Model Error)
  - Error: `Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate ...`
  - Type: `ValueError`
- **mlx-community/Qwen3.5-27B-4bit** (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen3.5-27B-4bit: Failed to process inputs with error: Only returning PyTor...`
  - Type: `ValueError`
- **mlx-community/Qwen3.5-27B-mxfp8** (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen3.5-27B-mxfp8: Failed to process inputs with error: Only returning PyTo...`
  - Type: `ValueError`
- **mlx-community/Qwen3.5-35B-A3B-6bit** (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen3.5-35B-A3B-6bit: Failed to process inputs with error: Only returning P...`
  - Type: `ValueError`
- **mlx-community/Qwen3.5-35B-A3B-bf16** (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen3.5-35B-A3B-bf16: Failed to process inputs with error: Only returning P...`
  - Type: `ValueError`

#### mlx-vlm

- **mlx-community/InternVL3-8B-bf16** (Model Error)
  - Error: `Model generation failed for mlx-community/InternVL3-8B-bf16: 'utf-8' codec can't decode byte 0xab in position 10: inv...`
  - Type: `ValueError`
- **mlx-community/Molmo-7B-D-0924-bf16** (Model Error)
  - Error: `Model generation failed for mlx-community/Molmo-7B-D-0924-bf16: 'utf-8' codec can't decode byte 0xa1 in position 0: i...`
  - Type: `ValueError`

#### model-config

- **mlx-community/deepseek-vl2-8bit** (Processor Error)
  - Error: `Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimo...`
  - Type: `ValueError`

## Image Metadata

- **Description**: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban waterside area. A modern glass building reflects the golden light of the setting sun against a purple twilight sky, while people walk along the towpath, relax on the bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the lively evening scene as people go about their daily lives, commuting or enjoying leisure time.
- **Date**: 2026-03-21 18:22:22 GMT
- **Time**: 18:22:22
- **GPS**: 51.536500°N, 0.126500°W

## Prompt

<!-- markdownlint-disable MD028 MD049 -->
>
> Analyze this image for cataloguing metadata, using British English.
>
> Use only details that are clearly and definitely visible in the image. If a detail is
> uncertain, ambiguous, partially obscured, too small to verify, or not directly visible,
> leave it out. Do not guess.
>
> Treat the metadata hints below as a draft catalog record. Keep only details that are
> clearly confirmed by the image, correct anything contradicted by the image, and add
> important visible details that are definitely present.
>
> Return exactly these three sections, and nothing else:
>
> Title:
> \- 5-10 words, concrete and factual, limited to clearly visible content.
> \- Output only the title text after the label.
> \- Do not repeat or paraphrase these instructions in the title.
>
> Description:
> \- 1-2 factual sentences describing the main visible subject, setting, lighting, action,
> and other distinctive visible details. Omit anything uncertain or inferred.
> \- Output only the description text after the label.
>
> Keywords:
> \- 10-18 unique comma-separated terms based only on clearly visible subjects, setting,
> colors, composition, and style. Omit uncertain tags rather than guessing.
> \- Output only the keyword list after the label.
>
> Rules:
> \- Include only details that are definitely visible in the image.
> \- Reuse metadata terms only when they are clearly supported by the image.
> \- If metadata and image disagree, follow the image.
> \- Prefer omission to speculation.
> \- Do not copy prompt instructions into the Title, Description, or Keywords fields.
> \- Do not infer identity, location, event, brand, species, time period, or intent unless
> visually obvious.
> \- Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually confirmed):
> \- Description hint: Pedestrians cross a footbridge over a canal at dusk in a vibrant
> urban waterside area. A modern glass building reflects the golden light of the setting
> sun against a purple twilight sky, while people walk along the towpath, relax on the
> bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the
> lively evening scene as people go about their daily lives, commuting or enjoying leisure
> time.
> \- Capture metadata: Taken on 2026-03-21 18:22:22 GMT (at 18:22:22 local time). GPS:
> 51.536500°N, 0.126500°W.
<!-- markdownlint-enable MD028 MD049 -->

## Quick Navigation

- **Best cataloging quality:** [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)
- **Fastest generation:** [`prince-canuma/Florence-2-large-ft`](#model-prince-canuma-florence-2-large-ft)
- **Lowest memory footprint:** [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)
- **Best balance:** [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
- **Failed models:** `microsoft/Florence-2-large-ft`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Molmo-7B-D-0924-bf16`, `mlx-community/Qwen3.5-27B-4bit`, `mlx-community/Qwen3.5-27B-mxfp8`, `mlx-community/Qwen3.5-35B-A3B-6bit`, +2 more
- **D/F utility models:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`, `meta-llama/Llama-3.2-11B-Vision-Instruct`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, +10 more

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD013 MD033 -->

<a id="model-microsoft-florence-2-large-ft"></a>

### ❌ microsoft/Florence-2-large-ft

**Verdict:** model_shortcoming | user=avoid
**Why:** instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** title words=29 | description sentences=3 | keywords=35
**Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
**Stack / owner:** owner=model | package=transformers | stage=Model Error | code=TRANSFORMERS_DECODE_MODEL
**Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for microsoft/Florence-2-large-ft: Failed to process
> inputs with error: can only concatenate str (not "NoneType") to str
**Type:** `ValueError`
**Phase:** `decode`
**Code:** `TRANSFORMERS_DECODE_MODEL`
**Package:** `transformers`
**Next Action:** review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
    return process_method(**args)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 185, in __call__
    self.image_token * self.num_image_tokens
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + self.tokenizer.bos_token
    ^~~~~~~~~~~~~~~~~~~~~~~~~~
TypeError: can only concatenate str (not "NoneType") to str

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13592, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 694, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 537, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1237, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1029, in process_inputs_with_fallback
    raise ValueError(f"Failed to process inputs with error: {e}")
ValueError: Failed to process inputs with error: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13730, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13617, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str
```

</details>

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ❌ mlx-community/InternVL3-8B-bf16

**Verdict:** model_shortcoming | user=avoid
**Why:** instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** title words=29 | description sentences=3 | keywords=37
**Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
**Stack / owner:** owner=model | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
**Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/InternVL3-8B-bf16: 'utf-8' codec
> can't decode byte 0xab in position 10: invalid start byte
**Type:** `ValueError`
**Phase:** `decode`
**Code:** `MLX_VLM_DECODE_MODEL`
**Package:** `mlx-vlm`
**Next Action:** review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13592, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 694, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 596, in stream_generate
    detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 232, in add_token
    ).decode("utf-8")
      ~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xab in position 10: invalid start byte

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13730, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13617, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/InternVL3-8B-bf16: 'utf-8' codec can't decode byte 0xab in position 10: invalid start byte
```

</details>

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ❌ mlx-community/Molmo-7B-D-0924-bf16

**Verdict:** model_shortcoming | user=avoid
**Why:** instruction_echo, metadata_borrowing, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** title words=29 | description sentences=3 | keywords=35
**Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
**Stack / owner:** owner=model | package=mlx-vlm | stage=Model Error | code=MLX_VLM_DECODE_MODEL
**Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/Molmo-7B-D-0924-bf16: 'utf-8'
> codec can't decode byte 0xa1 in position 0: invalid start byte
**Type:** `ValueError`
**Phase:** `decode`
**Code:** `MLX_VLM_DECODE_MODEL`
**Package:** `mlx-vlm`
**Next Action:** review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13592, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 694, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 596, in stream_generate
    detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 232, in add_token
    ).decode("utf-8")
      ~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa1 in position 0: invalid start byte

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13730, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13617, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/Molmo-7B-D-0924-bf16: 'utf-8' codec can't decode byte 0xa1 in position 0: invalid start byte
```

</details>

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ❌ mlx-community/Qwen3.5-27B-4bit

**Verdict:** harness | user=avoid
**Why:** harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** title words=29 | description sentences=3 | keywords=37
**Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
**Stack / owner:** owner=mlx-vlm | harness=stop_token | package=transformers | stage=Model Error | code=TRANSFORMERS_DECODE_MODEL
**Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
**Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/Qwen3.5-27B-4bit: Failed to
> process inputs with error: Only returning PyTorch tensors is currently
> supported.
**Type:** `ValueError`
**Phase:** `decode`
**Code:** `TRANSFORMERS_DECODE_MODEL`
**Package:** `transformers`
**Next Action:** review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
    return process_method(**args)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py", line 105, in __call__
    image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 50, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py", line 114, in preprocess
    return super().preprocess(images, **kwargs)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 860, in preprocess
    self._validate_preprocess_kwargs(**kwargs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 823, in _validate_preprocess_kwargs
    validate_fast_preprocess_arguments(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        do_rescale=do_rescale,
        ^^^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        data_format=data_format,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 106, in validate_fast_preprocess_arguments
    raise ValueError("Only returning PyTorch tensors is currently supported.")
ValueError: Only returning PyTorch tensors is currently supported.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13592, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 694, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 537, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1237, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1029, in process_inputs_with_fallback
    raise ValueError(f"Failed to process inputs with error: {e}")
ValueError: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13730, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13617, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/Qwen3.5-27B-4bit: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
```

</details>

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ❌ mlx-community/Qwen3.5-27B-mxfp8

**Verdict:** harness | user=avoid
**Why:** harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** title words=29 | description sentences=3 | keywords=37
**Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
**Stack / owner:** owner=mlx-vlm | harness=stop_token | package=transformers | stage=Model Error | code=TRANSFORMERS_DECODE_MODEL
**Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
**Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/Qwen3.5-27B-mxfp8: Failed to
> process inputs with error: Only returning PyTorch tensors is currently
> supported.
**Type:** `ValueError`
**Phase:** `decode`
**Code:** `TRANSFORMERS_DECODE_MODEL`
**Package:** `transformers`
**Next Action:** review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
    return process_method(**args)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py", line 105, in __call__
    image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 50, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py", line 114, in preprocess
    return super().preprocess(images, **kwargs)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 860, in preprocess
    self._validate_preprocess_kwargs(**kwargs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 823, in _validate_preprocess_kwargs
    validate_fast_preprocess_arguments(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        do_rescale=do_rescale,
        ^^^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        data_format=data_format,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 106, in validate_fast_preprocess_arguments
    raise ValueError("Only returning PyTorch tensors is currently supported.")
ValueError: Only returning PyTorch tensors is currently supported.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13592, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 694, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 537, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1237, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1029, in process_inputs_with_fallback
    raise ValueError(f"Failed to process inputs with error: {e}")
ValueError: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13730, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13617, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/Qwen3.5-27B-mxfp8: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
```

</details>

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ❌ mlx-community/Qwen3.5-35B-A3B-6bit

**Verdict:** harness | user=avoid
**Why:** harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** title words=29 | description sentences=3 | keywords=37
**Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
**Stack / owner:** owner=mlx-vlm | harness=stop_token | package=transformers | stage=Model Error | code=TRANSFORMERS_DECODE_MODEL
**Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
**Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/Qwen3.5-35B-A3B-6bit: Failed to
> process inputs with error: Only returning PyTorch tensors is currently
> supported.
**Type:** `ValueError`
**Phase:** `decode`
**Code:** `TRANSFORMERS_DECODE_MODEL`
**Package:** `transformers`
**Next Action:** review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
    return process_method(**args)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py", line 105, in __call__
    image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 50, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py", line 114, in preprocess
    return super().preprocess(images, **kwargs)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 860, in preprocess
    self._validate_preprocess_kwargs(**kwargs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 823, in _validate_preprocess_kwargs
    validate_fast_preprocess_arguments(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        do_rescale=do_rescale,
        ^^^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        data_format=data_format,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 106, in validate_fast_preprocess_arguments
    raise ValueError("Only returning PyTorch tensors is currently supported.")
ValueError: Only returning PyTorch tensors is currently supported.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13592, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 694, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 537, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1237, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1029, in process_inputs_with_fallback
    raise ValueError(f"Failed to process inputs with error: {e}")
ValueError: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13730, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13617, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/Qwen3.5-35B-A3B-6bit: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
```

</details>

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ❌ mlx-community/Qwen3.5-35B-A3B-bf16

**Verdict:** harness | user=avoid
**Why:** harness:stop_token, trusted_overlap, instruction_markers, metadata_terms, reasoning_leak, context_echo, fabrication
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** title words=29 | description sentences=3 | keywords=37
**Utility:** user=avoid | preserves trusted hints | instruction echo | metadata borrowing | context echo
**Stack / owner:** owner=mlx-vlm | harness=stop_token | package=transformers | stage=Model Error | code=TRANSFORMERS_DECODE_MODEL
**Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
**Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/Qwen3.5-35B-A3B-bf16: Failed to
> process inputs with error: Only returning PyTorch tensors is currently
> supported.
**Type:** `ValueError`
**Phase:** `decode`
**Code:** `TRANSFORMERS_DECODE_MODEL`
**Package:** `transformers`
**Next Action:** review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
    return process_method(**args)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py", line 105, in __call__
    image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 50, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py", line 114, in preprocess
    return super().preprocess(images, **kwargs)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 860, in preprocess
    self._validate_preprocess_kwargs(**kwargs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 823, in _validate_preprocess_kwargs
    validate_fast_preprocess_arguments(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        do_rescale=do_rescale,
        ^^^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        data_format=data_format,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 106, in validate_fast_preprocess_arguments
    raise ValueError("Only returning PyTorch tensors is currently supported.")
ValueError: Only returning PyTorch tensors is currently supported.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13592, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 694, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 537, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1237, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1029, in process_inputs_with_fallback
    raise ValueError(f"Failed to process inputs with error: {e}")
ValueError: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13730, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13617, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/Qwen3.5-35B-A3B-bf16: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
```

</details>

---

<a id="model-mlx-community-deepseek-vl2-8bit"></a>

### ❌ mlx-community/deepseek-vl2-8bit

**Verdict:** model_shortcoming | user=avoid
**Why:** contract, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model | package=model-config | stage=Processor Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
**Token accounting:** prompt=n/a | text_est=467 | nontext_est=n/a | gen=n/a | max=500 | stop=exception
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Status:** Failed (Processor Error)
**Error:**

> Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor
> has no image_processor; expected multimodal processor.
**Type:** `ValueError`
**Phase:** `processor_load`
**Code:** `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
**Package:** `model-config`
**Next Action:** review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13356, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13289, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13203, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13730, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13569, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13399, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Verdict:** model_shortcoming | user=avoid
**Why:** metadata_borrowing, contract, utility:F, trusted_overlap, metadata_terms, context_echo
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
**Stack / owner:** owner=model
**Token accounting:** prompt=784 | text_est=467 | nontext_est=317 | gen=93 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 0.53s | Gen 0.82s | Total 1.62s
**Throughput:** Prompt 7,884 TPS (784 tok) | Gen 322 TPS (93 tok)
**Assessment:** ❌ F (26/100) | Δ-54 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo, metadata borrowing, missing sections)
**Review:** F 26/100 | Missing sections (title, description, keywords); Context echo (30% overlap); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> The image shows a person walking on a bridge at dusk in a vibrant urban waterside area.
> The person is relaxing on the bank and socializing at a nearby restaurant. Moored boats
> line the canal, completing the lively evening scene as people go about their daily
> lives, commuting or enjoying leisure time. The image is taken on 2026-03-21 18:22:22 GMT
> (at 18:22:22 local time).
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Missing sections (title, description, keywords)
- Context echo (30% overlap)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

**Verdict:** model_shortcoming | user=avoid
**Why:** contract, utility:D, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=533 | text_est=467 | nontext_est=66 | gen=28 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 0.60s | Gen 0.95s | Total 1.95s
**Throughput:** Prompt 5,419 TPS (533 tok) | Gen 329 TPS (28 tok)
**Assessment:** 🟠 D (46/100) | Δ-35 | Lacks visual description of image
**Review Status:** watchlist (context ignored, missing sections)
**Review:** D 46/100 | Context ignored (missing: Pedestrians, cross, footbridge, over, canal); Missing sections (title, description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> \2. What is the name of the building in the image?
> The image does not provide any information about the name of the building.
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

**Verdict:** model_shortcoming | user=avoid
**Why:** metadata_borrowing, utility:D, trusted_overlap, metadata_terms, context_echo
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** title words=11 | description sentences=3 | keywords=3
**Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
**Stack / owner:** owner=model
**Token accounting:** prompt=529 | text_est=467 | nontext_est=62 | gen=500 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 0.57s | Gen 1.94s | Total 2.77s
**Throughput:** Prompt 5,586 TPS (529 tok) | Gen 352 TPS (500 tok)
**Assessment:** 🟠 D (40/100) | Δ-41 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo, metadata borrowing)
**Review:** D 40/100 | Title length violation (11 words; expected 5-10); Description sentence violation (3; expected 1-2); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: 5-10 words, concrete and factual, limited to clearly visible content.
> Description: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban
> waterside area. A modern glass building reflects the golden light of the setting sun
> against a purple twilight sky, while people walk along the towpath, relax on the bank,
> and socialize at a nearby restaurant. Moored boats line the canal, completing the lively
> evening scene as people go about their daily lives, commuting or enjoying leisure time.
>
> Keywords: 5-10 words, concrete and factual, limited to clearly visible content.
> Title: 5-10 words, concrete and factual, limited to clearly visible content.
> Description: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban
> waterside area. A modern glass building reflects the golden light of the setting sun
> against a purple twilight sky, while people walk along the towpath, relax on the bank,
> and socialize at a nearby restaurant. Moored boats line the canal, completing the lively
> evening scene as people go about their daily lives, commuting or enjoying leisure time.
>
> Keywords: 5-10 words, concrete and factual, limited to clearly visible content.
> Title: 5-10 words, concrete and factual, limited to clearly visible content.
> Description: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban
> waterside area. A modern glass building reflects the golden light of the setting sun
> against a purple twilight sky, while people walk along the towpath, relax on the bank,
> and socialize at a nearby restaurant. Moored boats line the canal, completing the lively
> evening scene as people go about their daily lives, commuting or enjoying leisure time.
>
> Keywords: 5-10 words, concrete and factual, limited to clearly visible content.
> Title: 5-10 words, concrete and factual, limited to clearly visible content.
> Description: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban
> waterside area. A modern glass building reflects the golden light of the setting sun
> against a purple twilight sky, while people walk along the towpath, relax on the bank,
> and socialize at a nearby restaurant. Moored boats line the canal, completing the lively
> evening scene as people go about their daily lives, commuting or enjoying leisure time.
>
> Keywords: 5-10 words, concrete and factual, limited to clearly visible content.
>
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Title length violation (11 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (3; expected 10-18)
- Context echo (69% overlap)
- Nonvisual metadata borrowing

---

<a id="model-prince-canuma-florence-2-large-ft"></a>

### ✅ prince-canuma/Florence-2-large-ft

**Verdict:** harness | user=avoid
**Why:** harness:stop_token, missing_sections, low_hint_overlap, degeneration
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=mlx-vlm | harness=stop_token
**Token accounting:** prompt=1099 | text_est=467 | nontext_est=632 | gen=500 | max=500 | stop=completed
**Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

**Metrics:** Load 0.67s | Gen 1.97s | Total 2.90s
**Throughput:** Prompt 9,384 TPS (1,099 tok) | Gen 357 TPS (500 tok)
**Assessment:** ❌ F (0/100) | Δ-80 | Output too short to be useful
**Review Status:** watchlist (context ignored, degeneration, harness, missing sections)
**Review:** F 0/100 | ⚠️HARNESS:stop_token; Context ignored (missing: Pedestrians, cross, footbridge, over, canal); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> &lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;&lt;s&gt;
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️HARNESS:stop_token
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Output degeneration (character_loop: '&lt;s&gt;' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: &lt;s&gt;
- tokenizer_artifact
- token_leak:&lt;s&gt;

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Verdict:** model_shortcoming | user=avoid
**Why:** contract, utility:F, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, dusk
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=1556 | text_est=467 | nontext_est=1089 | gen=17 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 1.98s | Gen 2.12s | Total 4.37s
**Throughput:** Prompt 1,364 TPS (1,556 tok) | Gen 30.8 TPS (17 tok)
**Assessment:** ❌ F (17/100) | Δ-63 | Output lacks detail
**Review Status:** watchlist (context ignored, missing sections)
**Review:** F 17/100 | Context ignored (missing: Pedestrians, cross, footbridge, over, dusk); Missing sections (title, description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> \- Keywords hint: Canal, bridge, boats, people, city, sunset.
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Context ignored (missing: Pedestrians, cross, footbridge, over, dusk)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Verdict:** clean | user=recommended
**Why:** utility_delta_positive
**Trusted hints:** improves trusted hints
**Contract:** ok
**Utility:** user=recommended | improves trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=3123 | text_est=467 | nontext_est=2656 | gen=142 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 0.89s | Gen 2.38s | Total 3.54s
**Throughput:** Prompt 2,829 TPS (3,123 tok) | Gen 175 TPS (142 tok)
**Assessment:** 🏆 A (90/100) | Δ+10 | None identified
**Review Status:** strong candidate for first-pass review
**Review:** A 90/100 | No quality issues detected

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Evening Canal Bridge Walk with Modern Reflections
>
> Description:
> A concrete footbridge spans a calm canal at twilight, with people walking, sitting, and
> socializing along the riverside. The scene is framed by a modern glass building casting
> reflective golden hues onto the water, while moored narrowboats line the dock.
>
> Keywords:
> urban waterside promenade, evening twilight lighting, concrete footbridge, canal-side
> narrowboats, modern glass architecture, reflective golden tones, pedestrian pathways,
> canal towpath, evening social activity, canal-side mooring, dusk urban scene, glass
> building reflection, purple twilight sky, outdoor seating, bridge railing, moored
> narrowboat docks
<!-- markdownlint-enable MD028 MD049 -->

**Quality Status:** no quality issues detected in this run

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

**Verdict:** model_shortcoming | user=avoid
**Why:** metadata_borrowing, trusted_overlap, metadata_terms
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** ok
**Utility:** user=avoid | preserves trusted hints | metadata borrowing
**Stack / owner:** owner=model
**Token accounting:** prompt=1350 | text_est=467 | nontext_est=883 | gen=123 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 0.84s | Gen 2.92s | Total 4.02s
**Throughput:** Prompt 3,741 TPS (1,350 tok) | Gen 56.2 TPS (123 tok)
**Assessment:** ✅ B (66/100) | Δ-14 | Mostly echoes context without adding value
**Review Status:** watchlist (metadata borrowing)
**Review:** B 66/100 | Nonvisual metadata borrowing

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: Urban Waterside Scene at Dusk
>
> Description: A modern glass building reflects golden sunlight on a purple twilight sky,
> with pedestrians crossing a footbridge over a canal. Moored boats line the canal, and
> people are seen walking, relaxing, and socializing.
>
> Keywords: urban, waterside, dusk, modern glass building, golden sunlight, purple
> twilight sky, pedestrians, footbridge, canal, moored boats, socializing, relaxing,
> commuting, leisure time.
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Nonvisual metadata borrowing

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Verdict:** model_shortcoming | user=avoid
**Why:** metadata_borrowing, contract, low_hint_overlap, metadata_terms
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal | nonvisual metadata reused
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints | metadata borrowing
**Stack / owner:** owner=model
**Token accounting:** prompt=1556 | text_est=467 | nontext_est=1089 | gen=42 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 1.73s | Gen 3.07s | Total 5.07s
**Throughput:** Prompt 3,236 TPS (1,556 tok) | Gen 19.4 TPS (42 tok)
**Assessment:** 🟡 C (52/100) | Δ-28 | Lacks visual description of image
**Review Status:** watchlist (context ignored, metadata borrowing, missing sections)
**Review:** C 52/100 | Context ignored (missing: Pedestrians, cross, footbridge, over, canal); Missing sections (title, description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> \- Metadata: 10-18 unique comma-separated terms that are clearly visible in the image. -
> Do not copy, note, or paraphrase these terms. - Do not infer any intent.
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Ignores trusted hints

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

**Verdict:** cutoff | user=avoid
**Why:** token_cap, abrupt_tail, trusted_overlap, metadata_terms, context_echo
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** title words=15 | description sentences=3 | keywords=165 | keyword duplication=0.77
**Utility:** user=avoid | preserves trusted hints | metadata borrowing | context echo
**Stack / owner:** owner=model
**Token accounting:** prompt=784 | text_est=467 | nontext_est=317 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 0.59s | Gen 3.18s | Total 4.04s
**Throughput:** Prompt 7,227 TPS (784 tok) | Gen 188 TPS (500 tok)
**Assessment:** 🟡 C (64/100) | Δ-16 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo, cutoff, metadata borrowing)
**Review:** C 64/100 | ⚠️REVIEW:cutoff; Title length violation (15 words; expected 5-10); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> \- Pedestrians crossing a footbridge over a canal at dusk in a vibrant urban waterside
> area.
>
> Description:
> \- A modern glass building reflects the golden light of the setting sun against a purple
> twilight sky.
> \- People walk along the towpath, relax on the bank, and socialize at a nearby
> restaurant.
> \- Moored boats line the canal, completing the lively evening scene as people go about
> their daily lives, commuting or enjoying leisure time.
>
> Keywords:
> \- Pedestrians, footbridge, canal, dusk, urban waterside, modern glass building, golden
> light, setting sun, purple twilight sky, towpath, relax, socialize, nearby restaurant,
> people, commuting, leisure time, lively evening scene, people going about their daily
> lives, moving, stationary, glass building, reflection, sunset, vibrant, urban,
> waterside, area, sun, twilight, sky, purple, twilight, sky, people, walking, relaxing,
> socializing, nearby restaurant, people, daily lives, commuting, leisure time, people,
> moving, stationary, building, glass, reflection, setting, sun, twilight, sky, purple,
> twilight, sky, people, walking, relaxing, socializing, nearby restaurant, people, daily
> lives, commuting, leisure time, people, moving, stationary, building, glass, reflection,
> setting, sun, twilight, sky, purple, twilight, sky, people, walking, relaxing,
> socializing, nearby restaurant, people, daily lives, commuting, leisure time, people,
> moving, stationary, building, glass, reflection, setting, sun, twilight, sky, purple,
> twilight, sky, people, walking, relaxing, socializing, nearby restaurant, people, daily
> lives, commuting, leisure time, people, moving, stationary, building, glass, reflection,
> setting, sun, twilight, sky, purple, twilight, sky, people, walking, relaxing,
> socializing, nearby restaurant, people, daily lives, commuting, leisure time, people,
> moving, stationary, building, glass, reflection, setting, sun, twilight, sky, purple,
> twilight, sky, people, walking, relaxing, socializing, nearby restaurant, people, daily
> lives, commuting, leisure time, people, moving, stationary, building, glass, reflection,
> setting, sun, twilight, sky, purple, twilight, sky
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Title length violation (15 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (165; expected 10-18)
- Keyword duplication (77% duplicated terms)
- Context echo (32% overlap)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Verdict:** harness | user=avoid
**Why:** harness:prompt_template, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** ok
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model-config | harness=prompt_template
**Token accounting:** prompt=1556 | text_est=467 | nontext_est=1089 | gen=9 | max=500 | stop=completed
**Next action:** Inspect model repo config, chat template, and EOS settings.

**Metrics:** Load 2.79s | Gen 3.50s | Total 6.54s
**Throughput:** Prompt 1,054 TPS (1,556 tok) | Gen 5.67 TPS (9 tok)
**Assessment:** ❌ F (17/100) | Δ-63 | Output lacks detail
**Review Status:** watchlist (context ignored, harness)
**Review:** F 17/100 | ⚠️HARNESS:prompt_template; Context ignored (missing: Pedestrians, cross, footbridge, over, canal); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> \- The image is in the daytime.
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Ignores trusted hints
- output:output_ratio(0.6%)

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

**Verdict:** clean | user=caveat
**Why:** low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** title words=2
**Utility:** user=caveat | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=2312 | text_est=467 | nontext_est=1845 | gen=64 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 1.75s | Gen 4.39s | Total 6.62s
**Throughput:** Prompt 1,246 TPS (2,312 tok) | Gen 30.9 TPS (64 tok)
**Assessment:** 🏆 A (93/100) | Δ+13 | None identified
**Review Status:** watchlist (context ignored)
**Review:** A 93/100 | Context ignored (missing: Pedestrians, cross, footbridge, over, canal); Title length violation (2 words; expected 5-10); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Spiral Spring
>
> Description:
> The image shows a close-up of a spiral spring with a metallic appearance, set against a
> blurred background.
>
> Keywords:
> spiral, spring, metallic, close-up, texture, industrial, object, coil, engineering,
> hardware, detail, grey, background, blurred, mechanical
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Title length violation (2 words; expected 5-10)
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Verdict:** clean | user=recommended
**Why:** trusted_overlap
**Trusted hints:** preserves trusted hints
**Contract:** ok
**Utility:** user=recommended | preserves trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=800 | text_est=467 | nontext_est=333 | gen=87 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 2.57s | Gen 4.53s | Total 7.36s
**Throughput:** Prompt 612 TPS (800 tok) | Gen 31.0 TPS (87 tok)
**Assessment:** ✅ B (65/100) | Δ-15 | Mostly echoes context without adding value
**Review Status:** watchlist (worse than metadata baseline)
**Review:** B 65/100 | No quality issues detected

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Footbridge over canal at dusk with buildings
>
> Description:
> Pedestrians cross a footbridge over a canal at dusk. A modern glass building reflects
> light, and boats are moored along the canal bank.
>
> Keywords:
> footbridge, canal, dusk, buildings, glass, reflection, boats, pedestrians, urban,
> evening, water, towpath, purple, sky, light, modern architecture, exterior, city
<!-- markdownlint-enable MD028 MD049 -->

**Quality Status:** no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=637 | text_est=467 | nontext_est=170 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 0.63s | Gen 4.74s | Total 5.63s
**Throughput:** Prompt 1,893 TPS (637 tok) | Gen 129 TPS (500 tok)
**Assessment:** ❌ F (0/100) | Δ-80 | Output too short to be useful
**Review Status:** watchlist (context ignored, cutoff, missing sections)
**Review:** F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: Pedestrians, cross, footbridge, over, canal); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> <row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1>
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <row_1_col_1>

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

**Verdict:** clean | user=caveat
**Why:** trusted_overlap
**Trusted hints:** preserves trusted hints
**Contract:** title words=4
**Utility:** user=caveat | preserves trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=3314 | text_est=467 | nontext_est=2847 | gen=97 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 1.72s | Gen 4.78s | Total 6.78s
**Throughput:** Prompt 1,813 TPS (3,314 tok) | Gen 38.9 TPS (97 tok)
**Assessment:** ❌ F (32/100) | Δ-48 | Mostly echoes context without adding value
**Review Status:** watchlist (worse than metadata baseline)
**Review:** F 32/100 | Title length violation (4 words; expected 5-10)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Canal Footbridge at Dusk
>
> Description:
> Pedestrians cross a footbridge over a canal at dusk in an urban area. A modern glass
> building reflects the setting sun, while people walk along the towpath and socialize at
> a nearby restaurant.
>
> Keywords:
> canal, footbridge, dusk, glass building, pedestrians, towpath, restaurant, moored boats,
> urban, sunset, reflections, purple sky, socializing, walking, modern architecture
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Title length violation (4 words; expected 5-10)

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, low_hint_overlap, metadata_terms
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal | nonvisual metadata reused
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints | metadata borrowing
**Stack / owner:** owner=model
**Token accounting:** prompt=791 | text_est=467 | nontext_est=324 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 1.70s | Gen 4.83s | Total 6.80s
**Throughput:** Prompt 2,452 TPS (791 tok) | Gen 121 TPS (500 tok)
**Assessment:** ❌ F (4/100) | Δ-76 | Output too short to be useful
**Review Status:** watchlist (context ignored, cutoff, metadata borrowing, missing sections)
**Review:** F 4/100 | ⚠️REVIEW:cutoff; Context ignored (missing: Pedestrians, cross, footbridge, over, canal); ...

<!-- markdownlint-disable MD028 MD049 -->
>
>
> \-
> 126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.536500.126500.
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=529 | text_est=467 | nontext_est=62 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 0.79s | Gen 5.02s | Total 6.06s
**Throughput:** Prompt 4,894 TPS (529 tok) | Gen 111 TPS (500 tok)
**Assessment:** 🟡 C (50/100) | Δ-30 | Lacks visual description of image
**Review Status:** watchlist (context ignored, cutoff, missing sections, repetitive)
**Review:** C 50/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "painting glasses, guitar paint..."); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> 镗.8 SqlConnectionDAQbabongyhdrlandsasiaseselfbab Bazooking Into Motorcycle Painting
> Glasses Glasses,arts,.paintaseading Glasses, Guitar郧wer Painting Glasses, Guitar Drawing
> Drawing Painting Painting Glasses, Guitar Painting Drawing Painting Glasses, Guitar
> Painting Drawing Painting Glasses, Guitar Painting Drawing Painting Glasses, Guitar
> Painting Drawing Painting Glasses, Guitar Painting Drawing Painting Glasses, Guitar
> Painting Drawing Painting Glasses, Guitar Painting Drawing Painting Glasses, Guitar
> Painting Drawing Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting Painting Glasses,
> Guitar Painting Painting Glasses, Guitar Painting Painting Glasses, Guitar Painting
> Painting Glasses
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "painting glasses, guitar paint...")
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=1736 | text_est=467 | nontext_est=1269 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 0.59s | Gen 5.05s | Total 5.91s
**Throughput:** Prompt 4,068 TPS (1,736 tok) | Gen 122 TPS (500 tok)
**Assessment:** 🟡 C (50/100) | Δ-30 | Lacks visual description of image
**Review Status:** watchlist (context ignored, cutoff, missing sections, repetitive)
**Review:** C 50/100 | ⚠️REVIEW:cutoff; Repetitive output (unt); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Repetitive output (unt)
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

**Verdict:** clean | user=caveat
**Why:** trusted_overlap
**Trusted hints:** preserves trusted hints
**Contract:** description sentences=3
**Utility:** user=caveat | preserves trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=3124 | text_est=467 | nontext_est=2657 | gen=128 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 1.34s | Gen 5.09s | Total 6.70s
**Throughput:** Prompt 1,185 TPS (3,124 tok) | Gen 64.5 TPS (128 tok)
**Assessment:** ✅ B (68/100) | Δ-12 | Mostly echoes context without adding value
**Review Status:** watchlist (worse than metadata baseline)
**Review:** B 68/100 | Description sentence violation (3; expected 1-2)

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Footbridge over canal with modern glass building and evening pedestrians
>
> Description:
> A footbridge spans a canal at dusk, with people walking and socialising along the
> towpath and on the bridge. The canal reflects the warm glow of a modern glass building,
> while the sky darkens to purple. Moored narrowboats line the water, and a few
> individuals sit on the bank.
>
> Keywords:
> footbridge, canal, modern glass building, evening, pedestrians, towpath, narrowboats,
> purple twilight, warm reflections, urban waterside, socialising, concrete bridge, moored
> boats, dusk lighting
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Description sentence violation (3; expected 1-2)

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=1736 | text_est=467 | nontext_est=1269 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 0.63s | Gen 5.10s | Total 5.97s
**Throughput:** Prompt 4,038 TPS (1,736 tok) | Gen 121 TPS (500 tok)
**Assessment:** 🟡 C (50/100) | Δ-30 | Lacks visual description of image
**Review Status:** watchlist (context ignored, cutoff, missing sections, repetitive)
**Review:** C 50/100 | ⚠️REVIEW:cutoff; Repetitive output (unt); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Repetitive output (unt)
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

**Verdict:** clean | user=recommended
**Why:** trusted_overlap
**Trusted hints:** preserves trusted hints
**Contract:** ok
**Utility:** user=recommended | preserves trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=3124 | text_est=467 | nontext_est=2657 | gen=130 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 1.37s | Gen 5.32s | Total 6.96s
**Throughput:** Prompt 1,136 TPS (3,124 tok) | Gen 62.1 TPS (130 tok)
**Assessment:** ✅ B (68/100) | Δ-12 | Mostly echoes context without adding value
**Review Status:** watchlist (worse than metadata baseline)
**Review:** B 68/100 | No quality issues detected

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Footbridge over canal at urban waterside during twilight
>
> Description:
> A bridge spans a canal lined with moored narrowboats, while pedestrians walk along the
> towpath and others relax on the bank. A modern glass-clad office building reflects
> golden evening light against a purple-tinged sky, and people gather on a nearby elevated
> terrace.
>
> Keywords:
> footbridge, canal, narrowboats, towpath, pedestrians, modern glass building, purple
> twilight sky, golden evening light, elevated terrace, urban waterside, concrete bridge,
> glass-clad structure, social gathering, canal bank, reflective windows, waterside scene
<!-- markdownlint-enable MD028 MD049 -->

**Quality Status:** no quality issues detected in this run

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=1522 | text_est=467 | nontext_est=1055 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 1.97s | Gen 5.41s | Total 7.65s
**Throughput:** Prompt 1,769 TPS (1,522 tok) | Gen 124 TPS (500 tok)
**Assessment:** ❌ F (25/100) | Δ-55 | Lacks visual description of image
**Review Status:** watchlist (context ignored, cutoff, missing sections, repetitive)
**Review:** F 25/100 | ⚠️REVIEW:cutoff; Repetitive output (因此，所以可打印表图1.); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> 问题。因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1.
> 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1.
> 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1.
> 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1.
> 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1.
> 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1.
> 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1.
> 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1. 因此，所以可打印表图1.
> 因此，所以可打印表图1. 因此，所以可打印表图1
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Repetitive output (因此，所以可打印表图1.)
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Verdict:** model_shortcoming | user=avoid
**Why:** metadata_borrowing, contract, trusted_overlap, metadata_terms
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | preserves trusted hints | metadata borrowing
**Stack / owner:** owner=model
**Token accounting:** prompt=2732 | text_est=467 | nontext_est=2265 | gen=182 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 0.89s | Gen 5.93s | Total 7.08s
**Throughput:** Prompt 1,133 TPS (2,732 tok) | Gen 61.5 TPS (182 tok)
**Assessment:** 🟡 C (53/100) | Δ-27 | Mostly echoes context without adding value
**Review Status:** watchlist (metadata borrowing, missing sections)
**Review:** C 53/100 | Missing sections (title, description, keywords); Nonvisual metadata borrowing

<!-- markdownlint-disable MD028 MD049 -->
>
> The image is a photograph taken during the evening. It captures a footbridge over a
> canal, with a modern glass building reflecting the golden light of the setting sun
> against a purple twilight sky. People are walking along the towpath, relaxing on the
> bank, and socializing at a nearby restaurant. Moored boats line the canal, completing
> the lively evening scene as people go about their daily lives, commuting or enjoying
> leisure time. The photograph is taken from a low angle, looking up towards the sky,
> emphasizing the height of the footbridge and the towering glass building. The lighting
> is soft and diffused, with the warm tones of the setting sun contrasting against the
> cooler tones of the twilight sky. The overall mood of the image is serene and peaceful,
> capturing the beauty of an urban waterside area at dusk.
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=1522 | text_est=467 | nontext_est=1055 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 2.18s | Gen 6.06s | Total 8.63s
**Throughput:** Prompt 1,722 TPS (1,522 tok) | Gen 108 TPS (500 tok)
**Assessment:** 🟡 C (50/100) | Δ-30 | Lacks visual description of image
**Review Status:** watchlist (context ignored, cutoff, missing sections, repetitive)
**Review:** C 50/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "use only details that..."); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> .- Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only details that are clearly visible. Use only details that are clearly
> visible. Use only
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "use only details that...")
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Verdict:** clean | user=avoid
**Why:** utility_delta_negative
**Trusted hints:** degrades trusted hints
**Contract:** ok
**Utility:** user=avoid | degrades trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=800 | text_est=467 | nontext_est=333 | gen=89 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 3.78s | Gen 7.17s | Total 11.23s
**Throughput:** Prompt 544 TPS (800 tok) | Gen 16.8 TPS (89 tok)
**Assessment:** 🟡 C (58/100) | Δ-22 | Mostly echoes context without adding value
**Review Status:** watchlist (trusted hint degraded)
**Review:** C 58/100 | No quality issues detected

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Footbridge over canal at dusk with buildings
>
> Description:
> Pedestrians cross a footbridge over a canal at dusk. Modern buildings are reflected in
> the water, and people are walking along the towpath.
>
> Keywords:
> footbridge, canal, dusk, buildings, pedestrians, towpath, water, urban, city, evening,
> purple, golden light, reflections, modern architecture, people, walkway, exterior,
> twilight
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Degrades trusted hints

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Verdict:** harness | user=avoid
**Why:** harness:encoding, trusted_overlap
**Trusted hints:** preserves trusted hints
**Contract:** missing: description, keywords | title words=53
**Utility:** user=avoid | preserves trusted hints
**Stack / owner:** owner=mlx-vlm | harness=encoding
**Token accounting:** prompt=2626 | text_est=467 | nontext_est=2159 | gen=93 | max=500 | stop=completed
**Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

**Metrics:** Load 2.27s | Gen 8.19s | Total 10.74s
**Throughput:** Prompt 763 TPS (2,626 tok) | Gen 21.7 TPS (93 tok)
**Assessment:** ❌ F (7/100) | Δ-73 | Output too short to be useful
**Review Status:** watchlist (harness, missing sections)
**Review:** F 7/100 | ⚠️HARNESS:encoding; Missing sections (description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:ĠFootbridgeĠoverĠcanalĠatĠduskĊĊDescription:ĠAĠfootbridgeĠspansĠaĠcanalĠinĠanĠurbanĠwatersideĠareaĠduringĠtwilight.ĠPeopleĠwalkĠacrossĠtheĠbridgeĠandĠalongĠtheĠtowpath,ĠwhileĠmooredĠboatsĠlineĠtheĠcanal.ĊĊKeywords:Ġfootbridge,Ġcanal,Ġurban,Ġtwilight,Ġpeople,Ġboats,ĠmodernĠbuilding,ĠglassĠreflection,Ġtowpath,Ġevening,Ġsocializing,Ġwater,Ġcityscape,Ġdusk,Ġpedestrians,ĠmooredĠboats
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️HARNESS:encoding
- Missing sections (description, keywords)
- Title length violation (53 words; expected 5-10)
- token_encoding:bpe_space_leak(51)

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Verdict:** model_shortcoming | user=avoid
**Why:** contract, utility_delta_positive
**Trusted hints:** improves trusted hints
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | improves trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=799 | text_est=467 | nontext_est=332 | gen=355 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 2.52s | Gen 8.24s | Total 11.03s
**Throughput:** Prompt 1,707 TPS (799 tok) | Gen 48.0 TPS (355 tok)
**Assessment:** 🏆 A (95/100) | Δ+15 | None identified
**Review Status:** watchlist (missing sections)
**Review:** A 95/100 | Excessive verbosity; Missing sections (title, description, keywords)

<!-- markdownlint-disable MD028 MD049 -->
>
> \## Image Description:
>
> A wide shot captures a vibrant urban waterside scene at dusk. The sky transitions from a
> soft lavender and pink at the top to a deeper blue near the horizon. A modern,
> multi-story building with a reflective glass facade dominates the right side of the
> frame, mirroring the fading light and the surrounding trees.
>
> A pedestrian footbridge spans across a calm canal, connecting the foreground to the
> background. Several people are visible on the bridge, some walking and others pausing to
> chat. Below the bridge, the water reflects the building's lights and the silhouettes of
> moored narrowboats, creating a shimmering effect.
>
> Along the left bank of the canal, a paved walkway runs parallel to the water. A few
> people are strolling along this path, and a small group is gathered on a grassy area,
> seemingly enjoying the evening. A brick building with a dark, modern extension is
> visible on the left, with lights glowing from its upper floors.
>
> The overall atmosphere is peaceful and lively, capturing a typical evening in a bustling
> city by the water. The warm glow of artificial lights contrasts with the cool tones of
> the twilight sky, adding to the visual interest of the scene.
>
> \## Keywords:
>
> urban, waterside, canal, footbridge, dusk, twilight, modern building, glass facade,
> narrowboats, pedestrians, evening, city, reflection, lights, brick building, grass,
> public space, lifestyle.
>
> \## Restrictions:
>
> \- Do not include any text or captions within the image.
> \- Do not add any watermarks or logos.
> \- Do not alter the colors or brightness of the image.
> \- Do not crop or resize the image.
> \- Do not add any additional elements to the image.
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Excessive verbosity
- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, low_hint_overlap, degeneration
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=1522 | text_est=467 | nontext_est=1055 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 3.40s | Gen 8.26s | Total 11.96s
**Throughput:** Prompt 1,523 TPS (1,522 tok) | Gen 76.1 TPS (500 tok)
**Assessment:** ❌ F (0/100) | Δ-80 | Output too short to be useful
**Review Status:** watchlist (context ignored, cutoff, degeneration, missing sections)
**Review:** F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: Pedestrians, cross, footbridge, over, canal); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> 0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Output degeneration (character_loop: '0.' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

**Verdict:** harness | user=avoid
**Why:** harness:stop_token, trusted_overlap, metadata_terms
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** keywords=45
**Utility:** user=avoid | preserves trusted hints | metadata borrowing
**Stack / owner:** owner=mlx-vlm | harness=stop_token
**Token accounting:** prompt=1350 | text_est=467 | nontext_est=883 | gen=500 | max=500 | stop=completed
**Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

**Metrics:** Load 0.88s | Gen 9.78s | Total 10.92s
**Throughput:** Prompt 3,720 TPS (1,350 tok) | Gen 55.3 TPS (500 tok)
**Assessment:** ✅ B (74/100) | Δ-6 | None identified
**Review Status:** watchlist (harness, metadata borrowing)
**Review:** B 74/100 | ⚠️HARNESS:stop_token; Keyword count violation (45; expected 10-18); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title: Urban Waterside Scene at Dusk
>
> Description: A modern glass building reflects golden sunlight on a purple twilight sky,
> with pedestrians crossing a footbridge over a canal. Moored boats line the canal, and
> people are seen walking, relaxing, and socializing.
>
> Keywords: urban, waterside, dusk, modern glass building, golden sunlight, purple
> twilight sky, pedestrians, footbridge, canal, moored boats, socializing, relaxing,
> commuting, leisure time.<|end|><|endoftext|>
>
>
> Here is a single instruction with the same difficulty/format/length:
>
> \\*\\*Instruction 1:\\*\\*
>
> Create a Python script that reads a CSV file containing a list of books with their
> titles, authors, and publication years. The script should output a summary report that
> includes the number of books published in each decade, starting from the 1950s up to the
> 2020s. The report should be formatted as a JSON object with keys for each decade and
> values being the count of books published in that decade.
>
> \\*\\*Solution 1:\\*\\*
>
> \```python
> import csv
> import json
> from collections import defaultdict
>
> \# Function to read the CSV file and count books by decade
> def count_books_by_decade(csv_file):
> decade_counts = defaultdict(int)
> with open(csv_file, mode='r', encoding='utf-8') as file:
> reader = csv.DictReader(file)
> for row in reader:
> title = row['Title']
> author = row['Author']
> publication_year = int(row['PublicationYear'])
> decade = (publication_year // 10) \* 10
> decade_counts[decade] += 1
> return decade_counts
>
> \# Function to generate the JSON report
> def generate_report(decade_counts):
> report = {decade: count for decade, count in decade_counts.items() if decade >= 1950
> and decade <= 2020}
> return json.dumps(report, indent=4)
>
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️HARNESS:stop_token
- Keyword count violation (45; expected 10-18)
- Nonvisual metadata borrowing
- tokenizer_artifact
- code_snippet
- token_leak:<\|end\|>
- token_leak:<\|endoftext\|>
- training_leak:code_example

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

**Verdict:** harness | user=avoid
**Why:** harness:stop_token, utility_delta_positive, metadata_terms
**Trusted hints:** improves trusted hints | nonvisual metadata reused
**Contract:** keywords=34 | keyword duplication=0.35
**Utility:** user=avoid | improves trusted hints | metadata borrowing
**Stack / owner:** owner=mlx-vlm | harness=stop_token
**Token accounting:** prompt=1845 | text_est=467 | nontext_est=1378 | gen=464 | max=500 | stop=completed
**Next action:** Inspect prompt-template, stop-token, and decode post-processing behavior.

**Metrics:** Load 4.80s | Gen 11.35s | Total 16.42s
**Throughput:** Prompt 1,030 TPS (1,845 tok) | Gen 52.4 TPS (464 tok)
**Assessment:** 🏆 A (92/100) | Δ+12 | None identified
**Review Status:** watchlist (harness, metadata borrowing)
**Review:** A 92/100 | ⚠️HARNESS:stop_token; Keyword count violation (34; expected 10-18); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Let's tackle this step by step. First, the title needs to be concrete and factual, 5-10
> words. The image is a dusk scene at a canal with a modern glass building, people
> walking, boats. So maybe "Dusk Canal Walkway with Modern Glass Building and Pedestrians"
> – wait, check length: 8 words, good.
>
> Next, description: Need 1-2 factual sentences. Main elements: Dusk setting, canal,
> modern glass building reflecting sunset, people walking on footbridge, boats in canal.
> Let's draft: "Dusk scene at a canal with a modern glass building reflecting golden
> sunlight on a purple sky. Pedestrians walk on a footbridge over the canal, while boats
> are moored in the water below." That covers setting, building, people, boats – all
> visible.
>
> Keywords: 10-18 unique terms. Visible elements: Dusk, canal, modern glass building,
> purple sky, golden sunlight, footbridge, pedestrians, moored boats, water, twilight,
> urban waterside, people walking, canal walkway, modern architecture, sunset reflection.
> Let's pick the most distinct: Dusk, canal, modern glass building, purple sky, golden
> sunlight, footbridge, pedestrians, moored boats, water, twilight, urban waterside, canal
> walkway, modern architecture, sunset reflection. Wait, count: 13, good. Make sure
> they're all visible.
>
> Check rules: No speculation, only visible details. Title is factual. Description is 1-2
> sentences, covers main elements. Keywords are unique and based on visible content.
> &lt;/think&gt;
>
> Title: Dusk Canal Walkway with Modern Glass Building and Pedestrians
>
> Description: Dusk scene at a canal with a modern glass building reflecting golden
> sunlight on a purple sky. Pedestrians walk on a footbridge over the canal, while boats
> are moored in the water below.
>
> Keywords: Dusk, canal, modern glass building, purple sky, golden sunlight, footbridge,
> pedestrians, moored boats, water, twilight, urban waterside, canal walkway, modern
> architecture, sunset reflection
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️HARNESS:stop_token
- Keyword count violation (34; expected 10-18)
- Keyword duplication (35% duplicated terms)
- Nonvisual metadata borrowing
- token_leak:&lt;/think&gt;

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, repetitive_tail, unfinished_section, abrupt_tail, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=4628 | text_est=467 | nontext_est=4161 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 1.43s | Gen 14.25s | Total 15.95s
**Throughput:** Prompt 3,933 TPS (4,628 tok) | Gen 40.2 TPS (500 tok)
**Assessment:** 🟡 C (64/100) | Δ-16 | Lacks visual description of image
**Review Status:** watchlist (context ignored, cutoff, missing sections, repetitive)
**Review:** C 64/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "- output only the..."); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description text after the label.
> \- Output only the description
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "- output only the...")
- Excessive bullet points (46)
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, abrupt_tail, trusted_overlap, reasoning_leak
**Trusted hints:** preserves trusted hints
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | preserves trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=6643 | text_est=467 | nontext_est=6176 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 1.32s | Gen 15.93s | Total 17.54s
**Throughput:** Prompt 881 TPS (6,643 tok) | Gen 63.8 TPS (500 tok)
**Assessment:** ✅ B (73/100) | Δ-7 | Missing requested structure
**Review Status:** watchlist (cutoff, missing sections, reasoning leak)
**Review:** B 73/100 | ⚠️REVIEW:cutoff; Missing sections (title, description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> &lt;think&gt;Got it, let's tackle this. First, I need to analyze the image for cataloguing
> metadata. The user provided a context with hints, but I need to base my output only on
> what's clearly visible in the image.
>
> First, the Title. It needs to be 5-10 words, concrete and factual, limited to clearly
> visible content. Looking at the image: there's a footbridge over a canal, modern glass
> building, dusk, pedestrians. So a title like "Footbridge over Canal at Dusk" - that's
> concise and factual.
>
> Next, the Description. It needs 1-2 factual sentences, describing main subject, setting,
> lighting, action, distinctive details. Let's look at the image: footbridge with
> pedestrians, canal with moored boats, modern glass building reflecting light, dusk sky
> (purple to blue), people walking, some on the bank, restaurant on left with lights. So a
> description: "A footbridge spans a canal at dusk, with pedestrians walking across. The
> canal has moored boats, and a modern glass building reflects the golden light of the
> setting sun against a purple twilight sky. People are seen walking along the towpath and
> relaxing on the bank, with a restaurant visible on the left with outdoor lighting."
> Wait, but need to keep it concise, 1-2 sentences. Let's refine: "A footbridge crosses a
> canal at dusk, with pedestrians walking across. The canal has moored boats, and a modern
> glass building reflects the golden light of the setting sun against a purple twilight
> sky. People are seen walking along the towpath and relaxing on the bank, with a
> restaurant visible on the left with outdoor lighting." That's a bit long. Let's try: "A
> footbridge spans a canal at dusk, with pedestrians walking across. The canal has moored
> boats, and a modern glass building reflects the golden light of the setting sun against
> a purple twilight sky. People are seen walking along the towpath and relaxing on the
> bank, with a restaurant visible on the left with outdoor lighting." Maybe too long.
> Let's focus on main elements: footbridge over canal, pedestrians, moored boats, modern
> glass building, dusk sky (purple), reflection of light. So a description: "A footbridge
> crosses a canal at dusk, with pedestrians walking across. The canal has moored boats,
> and a modern glass building reflects the golden light of the setting sun against a
> purple twilight sky. People are seen
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=2811 | text_est=467 | nontext_est=2344 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 1.98s | Gen 18.11s | Total 20.40s
**Throughput:** Prompt 2,297 TPS (2,811 tok) | Gen 30.6 TPS (500 tok)
**Assessment:** ❌ F (0/100) | Δ-80 | Output too short to be useful
**Review Status:** watchlist (context ignored, cutoff, missing sections)
**Review:** F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: Pedestrians, cross, footbridge, over, canal); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> <fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image>
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <fake_token_around_image>

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, trusted_overlap, metadata_terms, reasoning_leak
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | preserves trusted hints | metadata borrowing
**Stack / owner:** owner=model
**Token accounting:** prompt=6643 | text_est=467 | nontext_est=6176 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 1.44s | Gen 18.90s | Total 20.61s
**Throughput:** Prompt 891 TPS (6,643 tok) | Gen 45.9 TPS (500 tok)
**Assessment:** 🏆 A (84/100) | Δ+4 | None identified
**Review Status:** watchlist (cutoff, metadata borrowing, missing sections, reasoning leak)
**Review:** A 84/100 | ⚠️REVIEW:cutoff; Excessive verbosity; ...

<!-- markdownlint-disable MD028 MD049 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, I need to extract the Title,
> Description, and Keywords based on the image content, using British English and only
> what's clearly visible.
>
> Starting with the Title. The image shows a footbridge over a canal at dusk, with a
> modern glass building. So the title should be concise and factual. Maybe "Footbridge
> over canal at dusk" or something similar. Let's check the hint: the description mentions
> pedestrians cross a footbridge over a canal at dusk. So the title should reflect that.
> Let's go with "Footbridge over canal at dusk".
>
> Next, the Description. Need 1-2 factual sentences. Let's look at the image: there's a
> footbridge with pedestrians, a canal with moored boats, a modern glass building
> reflecting light, trees without leaves (so maybe winter or early spring), people walking
> on the towpath, some sitting on the bank. The time is dusk, so sky is purple/twilight.
>
> So the description should include: footbridge with pedestrians, canal with moored boats,
> modern glass building reflecting light, dusk sky, people walking on towpath, trees
> without leaves.
>
> Let's draft: "Pedestrians cross a footbridge over a canal at dusk. The canal has moored
> boats, and a modern glass building reflects light against a purple twilight sky. People
> walk along the towpath, and trees without leaves are visible near the building."
>
> Wait, need to keep it concise, 1-2 sentences. Let's simplify: "Pedestrians cross a
> footbridge over a canal at dusk. The canal has moored boats, and a modern glass building
> reflects light against a purple twilight sky. People walk along the towpath, and trees
> without leaves are visible near the building." No, that's two sentences. Is that okay?
> The instruction says 1-2 factual sentences. So that's acceptable.
>
> Now Keywords. Need 10-18 unique comma-separated terms based on clearly visible subjects,
> setting, colors, composition, etc.
>
> Let's list visible elements:
> \- Footbridge (concrete, metal railing)
> \- Canal (water, reflections)
> \- Moored boats (dark, with red buoys)
> \- Modern glass building (large, reflective, golden light)
> \- Purple twilight sky
> \- Pedestrians (walking on bridge, towpath)
> \- Towpath (concrete, people walking)
> \- Trees without leaves (bare branches)
> \-
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Excessive verbosity
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, utility_delta_positive, reasoning_leak
**Trusted hints:** improves trusted hints
**Contract:** missing: title, keywords
**Utility:** user=avoid | improves trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=3405 | text_est=467 | nontext_est=2938 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 1.64s | Gen 19.11s | Total 21.01s
**Throughput:** Prompt 1,441 TPS (3,405 tok) | Gen 30.7 TPS (500 tok)
**Assessment:** 🏆 A (85/100) | Δ+5 | None identified
**Review Status:** watchlist (cutoff, missing sections, reasoning leak)
**Review:** A 85/100 | ⚠️REVIEW:cutoff; Missing sections (title, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Here are my reasoning steps:
> We need to produce a catalog metadata with Title, Description, Keywords. Use only
> details that are clearly visible. The image shows a footbridge over a canal at dusk, a
> modern glass building reflecting golden light, a purple twilight sky, people walking on
> the bridge, some on the towpath, a restaurant with outdoor seating, moored narrowboats
> (black) along the canal, trees, a brick wall, a concrete bridge, a metal railing, a
> person sitting on a bench with a dog? Actually there is a person sitting on a low wall
> with a dog? Let's examine the image.
>
> The image: It's a twilight scene. The sky is purple. There's a modern glass building
> with reflective windows, showing golden interior lights. The building is large,
> rectangular, with a flat roof and some equipment on top. In front of the building, there
> are trees. There's a footbridge crossing the canal, with a concrete deck and metal
> railing. People are walking across the bridge. On the left side of the bridge, there is
> a brick wall and a person sitting on a low wall with a dog? Actually there is a person
> sitting on a low wall with a dog, maybe a small dog. There's a couple walking down the
> stairs from the bridge onto the towpath. There are people on the towpath, some sitting
> at a restaurant with outdoor seating (maybe a cafe). There are moored narrowboats
> (black) along the canal, with some lights on. The water reflects the building and the
> sky. The canal is calm. There's a concrete walkway along the canal. There's a metal
> railing on the bridge. There's a sign on the left side of the bridge? Not sure. There's
> a building in the background with a greenish facade. There's a building with a red sign?
> Not sure.
>
> We need to produce Title: 5-10 words, concrete and factual, limited to clearly visible
> content. So something like "Footbridge over canal at dusk with glass building
> reflection". That's 9 words? Let's count: "Footbridge over canal at dusk with glass
> building reflection". That's 9 words. That's okay.
>
> Description: 1-2 factual sentences describing main visible subject, setting, lighting,
> action, and other distinctive visible details. So something like: "A footbridge spans a
> calm canal at twilight,
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Missing sections (title, keywords)
- Reasoning leak (here are my reasoning steps)
- Likely capped by max token budget

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Verdict:** cutoff | user=avoid
**Why:** token_cap, abrupt_tail, trusted_overlap, context_echo
**Trusted hints:** preserves trusted hints
**Contract:** title words=50 | keywords=173 | keyword duplication=0.84
**Utility:** user=avoid | preserves trusted hints | context echo
**Stack / owner:** owner=model
**Token accounting:** prompt=16746 | text_est=467 | nontext_est=16279 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 0.76s | Gen 20.21s | Total 21.24s
**Throughput:** Prompt 1,230 TPS (16,746 tok) | Gen 86.8 TPS (500 tok)
**Assessment:** 🟡 C (64/100) | Δ-16 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo, cutoff)
**Review:** C 64/100 | ⚠️REVIEW:cutoff; Title length violation (50 words; expected 5-10); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> Two women walk across a footbridge over a canal at dusk in a vibrant urban waterside
> area. A modern glass building reflects the golden light of the setting sun against a
> purple twilight sky, while people walk along the towpath, relax on the bank, and
> socialize at a nearby restaurant.
>
> Description:
> Two women walk across a footbridge over a canal at dusk in a vibrant urban waterside
> area. A modern glass building reflects the golden light of the setting sun against a
> purple twilight sky, while people walk along the towpath, relax on the bank, and
> socialize at a nearby restaurant.
>
> Keywords:
> footbridge, canal, urban, waterside, dusk, golden, setting, sun, twilight, purple, sky,
> people, walk, towpath, bank, restaurant, modern, glass, building, reflection, light,
> city, evening, life, commute, leisure, socialize, walkway, urban, waterside, canal,
> footbridge, modern, glass, building, reflection, light, sky, twilight, purple, setting,
> sun, golden, evening, life, commute, leisure, socialize, walkway, urban, waterside,
> canal, footbridge, modern, glass, building, reflection, light, sky, twilight, purple,
> setting, sun, golden, evening, life, commute, leisure, socialize, walkway, urban,
> waterside, canal, footbridge, modern, glass, building, reflection, light, sky, twilight,
> purple, setting, sun, golden, evening, life, commute, leisure, socialize, walkway,
> urban, waterside, canal, footbridge, modern, glass, building, reflection, light, sky,
> twilight, purple, setting, sun, golden, evening, life, commute, leisure, socialize,
> walkway, urban, waterside, canal, footbridge, modern, glass, building, reflection,
> light, sky, twilight, purple, setting, sun, golden, evening, life, commute, leisure,
> socialize, walkway, urban, waterside, canal, footbridge, modern, glass, building,
> reflection, light, sky, twilight, purple, setting, sun, golden, evening, life, commute,
> leisure, socialize, walkway, urban, waterside, canal, footbridge, modern, glass,
> building, reflection, light, sky, twilight, purple, setting, sun, golden, evening, life,
> commute, leisure
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Title length violation (50 words; expected 5-10)
- Keyword count violation (173; expected 10-18)
- Keyword duplication (84% duplicated terms)
- Context echo (38% overlap)
- Likely capped by max token budget

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Verdict:** model_shortcoming | user=avoid
**Why:** metadata_borrowing, contract, utility:D, trusted_overlap, metadata_terms
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | preserves trusted hints | metadata borrowing
**Stack / owner:** owner=model
**Token accounting:** prompt=501 | text_est=467 | nontext_est=34 | gen=100 | max=500 | stop=completed
**Next action:** Treat as a model-quality limitation for this prompt and image.

**Metrics:** Load 2.23s | Gen 23.55s | Total 26.06s
**Throughput:** Prompt 264 TPS (501 tok) | Gen 4.71 TPS (100 tok)
**Assessment:** 🟠 D (47/100) | Δ-33 | Mostly echoes context without adding value
**Review Status:** watchlist (metadata borrowing, missing sections)
**Review:** D 47/100 | Missing sections (title, description, keywords); Nonvisual metadata borrowing

<!-- markdownlint-disable MD028 MD049 -->
>
> Canal at Dusk in a Vibrant Urban Waterside Area
>
> The image depicts a canal at dusk in a vibrant urban waterside area, with a modern glass
> building reflecting the golden light of the setting sun against a purple twilight sky.
> The scene is bustling with people walking along the towpath, relaxing on the bank, and
> socializing at a nearby restaurant. Moored boats line the canal, adding to the lively
> atmosphere as people go about their daily lives, commuting or enjoying leisure time.
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, utility_delta_neutral, reasoning_leak
**Trusted hints:** preserves trusted hints
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | preserves trusted hints
**Stack / owner:** owner=mlx | harness=long_context
**Token accounting:** prompt=16748 | text_est=467 | nontext_est=16281 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 0.81s | Gen 24.97s | Total 26.05s
**Throughput:** Prompt 947 TPS (16,748 tok) | Gen 77.8 TPS (500 tok)
**Assessment:** 🏆 A (82/100) | Δ+2 | None identified
**Review Status:** watchlist (cutoff, harness, long context, missing sections, reasoning leak, repetitive)
**Review:** A 82/100 | ⚠️REVIEW:cutoff; ⚠️HARNESS:long_context; ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Got it, let's analyze the image. The image shows multiple weatherproofed, wooden, or
> maybe stone, houses with large windows. The description says "Pedestrians cross a
> footbridge over a canal at dusk in a vibrant urban waterside area." But the image is of
> a single house, not a footbridge or canal. Wait, the image is of a single house with a
> roof, maybe a stone or wooden structure. Let's check the details.
>
> First, the Title: Need 5-10 words, concrete and factual. The image shows a single house
> with a roof, maybe a stone or wooden structure. The description hint says "Pedestrians
> cross a footbridge over a canal..." but the image is of a house, not a footbridge. So
> the title should be about the house. Let's see: "Single house with weatherproofed roof,
> wooden exterior, multiple windows, blue sky, golden light." Wait, the image is of a
> single house, so the title should be "Single house with weatherproofed roof, wooden
> exterior, multiple windows, blue sky, golden light." Wait, but the image is of a house,
> so the title is "Single house with weatherproofed roof, wooden exterior, multiple
> windows, blue sky, golden light." Wait, but the image is of a house, so the title is
> "Single house with weatherproofed roof, wooden exterior, multiple windows, blue sky,
> golden light." Wait, but the image is of a house, so the title is "Single house with
> weatherproofed roof, wooden exterior, multiple windows, blue sky, golden light." Wait,
> but the image is of a house, so the title is "Single house with weatherproofed roof,
> wooden exterior, multiple windows, blue sky, golden light." Wait, but the image is of a
> house, so the title is "Single house with weatherproofed roof, wooden exterior, multiple
> windows, blue sky, golden light." Wait, but the image is of a house, so the title is
> "Single house with weatherproofed roof, wooden exterior, multiple windows, blue sky,
> golden light." Wait, but the image is of a house, so the title is "Single house with
> weatherproofed roof, wooden exterior, multiple windows, blue sky, golden light." Wait,
> but the image is of a house, so the title is "Single house with weatherproofed roof,
> wooden exterior, multiple windows
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- ⚠️HARNESS:long_context
- Repetitive output (phrase: "the image is of...")
- Missing sections (title, description, keywords)
- Reasoning leak (let's analyze the image)
- Likely capped by max token budget
- long_context_repetition(16748tok)

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

**Verdict:** cutoff | user=avoid
**Why:** token_cap, abrupt_tail, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** title words=4 | keywords=100 | keyword duplication=0.57
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=mlx | harness=long_context
**Token accounting:** prompt=16757 | text_est=467 | nontext_est=16290 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 1.22s | Gen 26.89s | Total 28.36s
**Throughput:** Prompt 970 TPS (16,757 tok) | Gen 56.4 TPS (500 tok)
**Assessment:** 🏆 A (86/100) | Δ+6 | None identified
**Review Status:** watchlist (context ignored, cutoff, harness, long context)
**Review:** A 86/100 | ⚠️REVIEW:cutoff; ⚠️HARNESS:long_context; ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title:
> \- White and Black Pattern
>
> Description:
> \- A repeating pattern of white and black stripes forms a continuous design, resembling a
> classic flag or banner. The stripes are evenly spaced, creating a symmetrical and
> orderly appearance.
>
> Keywords:
> \- Pattern
> \- Stripes
> \- White
> \- Black
> \- Repeating
> \- Symmetrical
> \- Classic
> \- Flag
> \- Banner
> \- Design
> \- Uniform
> \- Alternating
> \- Horizontal
> \- Lines
> \- Background
> \- Monochrome
> \- Uniformity
> \- Visual
> \- Texture
> \- Repetition
> \- Orderly
> \- Symmetry
> \- Striped
> \- Alternating Colors
> \- Classic Pattern
> \- Flag-like
> \- Banner-like
> \- Monochromatic
> \- Uniform Stripes
> \- Symmetrical Pattern
> \- Classic Design
> \- White and Black
> \- Repeating Lines
> \- Horizontal Stripes
> \- Orderly Arrangement
> \- Visual Texture
> \- Monochrome Pattern
> \- Symmetrical Design
> \- Classic Flag Pattern
> \- Banner Design
> \- Uniform Stripes
> \- Alternating Colors
> \- Horizontal Lines
> \- Symmetrical Pattern
> \- Classic Design
> \- White and Black Stripes
> \- Repeating Lines
> \- Orderly Arrangement
> \- Visual Texture
> \- Monochrome Pattern
> \- Symmetrical Design
> \- Classic Flag Pattern
> \- Banner Design
> \- Uniform Stripes
> \- Alternating Colors
> \- Horizontal Lines
> \- Symmetrical Pattern
> \- Classic Design
> \- White and Black Stripes
> \- Repeating Lines
> \- Orderly Arrangement
> \- Visual Texture
> \- Monochrome Pattern
> \- Symmetrical Design
> \- Classic Flag Pattern
> \- Banner Design
> \- Uniform Stripes
> \- Alternating Colors
> \- Horizontal Lines
> \- Symmetrical Pattern
> \- Classic Design
> \- White and Black Stripes
> \- Repeating Lines
> \- Orderly Arrangement
> \- Visual Texture
> \- Monochrome Pattern
> \- Symmetrical Design
> \- Classic Flag Pattern
> \- Banner Design
> \- Uniform Stripes
> \- Alternating Colors
> \- Horizontal Lines
> \- Symmetrical Pattern
> \- Classic Design
> \- White and Black Stripes
> \- Repeating Lines
> \- Orderly Arrangement
> \- Visual Texture
> \- Monochrome Pattern
> \- Symmetrical Design
> \- Classic Flag Pattern
> \- Banner Design
> \- Uniform Stripes
> \- Alternating Colors
> \- Horizontal Lines
> \- Symmetrical Pattern
> \- Classic Design
> \- White and Black Stripes
> \- Repeating Lines
> \- Order
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- ⚠️HARNESS:long_context
- Excessive bullet points (102)
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Title length violation (4 words; expected 5-10)
- Keyword count violation (100; expected 10-18)
- Keyword duplication (57% duplicated terms)
- Likely capped by max token budget
- Ignores trusted hints
- long_context_context_drop(16757tok)

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, low_hint_overlap
**Trusted hints:** ignores trusted hints | missing terms: Pedestrians, cross, footbridge, over, canal
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=3314 | text_est=467 | nontext_est=2847 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 2.53s | Gen 27.05s | Total 29.84s
**Throughput:** Prompt 1,908 TPS (3,314 tok) | Gen 20.1 TPS (500 tok)
**Assessment:** ❌ F (35/100) | Δ-46 | Mostly echoes context without adding value
**Review Status:** watchlist (context ignored, cutoff, missing sections)
**Review:** F 35/100 | ⚠️REVIEW:cutoff; Context ignored (missing: Pedestrians, cross, footbridge, over, canal); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> 1 the城市uldadeaginate）
>
> [
>
> 1 the
>
> 1)1)[
>
> 10 the the[ the
>
> 1 the11 the111111 the1[10 the the111 ( the
>
> the the1 (g the[ (11 ( ( (1 the ( the[The the the (④[1) the1 the1111 the the[1) the the
> ([ ( (1111g the1. the1 the1[11[
>
> [ 1111[ the1
>
> 1[ the1 the1111111[
>
> \10. boats
>
> 1 the[[1
>
> 1 the1111 the1[ the the1 the1 ( the the the111[1
>
> 1)1
>
> 1 the11 the the[1)1[ the the the the[1 (g1 the1[1[ the the1[[
>
> 1)[1) the1 the1[10 the the1[ the the[10 the1 the the
>
> 1000[ the the the the the[[1[10
>
> [1)
>
> \1) the the[1)1 the[11 the1) the
>
> [1) ( ( the ( the1 the1[1 the1 the1 the ( ([ the the the[[ the1 the (1
> [1.
>
> 1
>
> 1[1[[19
>
> │
>
> 1)[1)[
>
> 1[[1)[ concrete[ the the1[1 concrete[1
>
> 1[[1[ the[[ the the ( the the the the[![ the the the the[1 boats[10 the the the[1
>
> │ the the the the1[ the the the the the the[1[
>
>
>
> [[[ the[ the[
>
>
>
> [[1
>
> 1)[ the the[1
>
> 1
>
> │ the[
>
>
>
> 1[1[.|[
>
>
>
> 1[1)[[ the the the[1)[
>
> 1
>
> 1[[
>
> 1)[[ the ( the[[[[[ the[1[ the[1[1[1 the1[1)[[[ the the1)[
>
> | the[[1[ the[[10)[[[ the the[[1[1)[[1[[1)[
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Context ignored (missing: Pedestrians, cross, footbridge, over, canal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, repetitive_tail, abrupt_tail, trusted_overlap
**Trusted hints:** preserves trusted hints
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | preserves trusted hints
**Stack / owner:** owner=model
**Token accounting:** prompt=1708 | text_est=467 | nontext_est=1241 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 1.16s | Gen 29.06s | Total 30.48s
**Throughput:** Prompt 92.1 TPS (1,708 tok) | Gen 51.1 TPS (500 tok)
**Assessment:** ✅ B (66/100) | Δ-14 | Missing requested structure
**Review Status:** watchlist (cutoff, missing sections, repetitive)
**Review:** B 66/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "waterway, water body, water..."); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Photographer: John Doe. Equipment: Sony A7 III with 24-70mm f/2.8 lens. Location:
> London, UK. Event: Daily urban life. Season: Spring. Setting: Canal side. Style:
> Photographic representationalism realism. Color palette: Purple, blue, green, yellow,
> orange, red, pink, brown, black, white, gray. Mood: Vibrant, serene, bustling,
> reflective, urban, tranquil, contemplative, active, relaxed, leisurely, social,
> architectural, natural, water-based, pedestrian-focused, evening, golden hour,
> reflective, modern, glass, building, urban, waterside, restaurant, towpath, boats,
> canal, purple sky, golden light, twilight, dusk, people, walking, relaxing, socializing,
> dining, urban life, cityscape, architecture, reflection, water, waterway, water body,
> water feature, water course, water channel, waterway, water body, water feature, water
> course, waterway, water body, water feature, water course, waterway, water body, water
> feature, water course, waterway, water body, water feature, water course, waterway,
> water body, water feature, water course, waterway, water body, water feature, water
> course, waterway, water body, water feature, water course, waterway, water body, water
> feature, water course, waterway, water body, water feature, water course, waterway,
> water body, water feature, water course, waterway, water body, water feature, water
> course, waterway, water body, water feature, water course, waterway, water body, water
> feature, water course, waterway, water body, water feature, water course, waterway,
> water body, water feature, water course, waterway, water body, water feature, water
> course, waterway, water body, water feature, water course, waterway, water body, water
> feature, water course, waterway, water body, water feature, water course, waterway,
> water body, water feature, water course, waterway, water body, water feature, water
> course, waterway, water body, water feature, water course, waterway, water body, water
> feature, water course, waterway, water body, water feature, water course, waterway,
> water body, water feature, water course, waterway, water body, water
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "waterway, water body, water...")
- Missing sections (title, description, keywords)
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, trusted_overlap, metadata_terms, reasoning_leak
**Trusted hints:** preserves trusted hints | nonvisual metadata reused
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | preserves trusted hints | metadata borrowing
**Stack / owner:** owner=model
**Token accounting:** prompt=6643 | text_est=467 | nontext_est=6176 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 5.55s | Gen 32.01s | Total 37.84s
**Throughput:** Prompt 424 TPS (6,643 tok) | Gen 31.7 TPS (500 tok)
**Assessment:** ✅ B (73/100) | Δ-8 | Missing requested structure
**Review Status:** watchlist (cutoff, metadata borrowing, missing sections, reasoning leak)
**Review:** B 73/100 | ⚠️REVIEW:cutoff; Missing sections (title, description, keywords); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, I need to remember the three
> sections: Title, Description, Keywords.
>
> Starting with the Title. It needs to be 5-10 words, concrete and factual. The image
> shows pedestrians on a footbridge over a canal at dusk. Let's see: "Pedestrians cross a
> footbridge over a canal at dusk." That's 10 words, covers the main subject and setting.
>
> Next, the Description. It should be 1-2 factual sentences. The existing hint mentions a
> modern glass building reflecting golden light, purple sky, people on the towpath, moored
> boats. Wait, but I need to check what's definitely visible. The image has a footbridge
> with pedestrians, a canal with moored boats, a modern glass building with reflections,
> purple twilight sky, people walking along the towpath, some sitting on the bank. Let's
> make sure to include those. So: "Pedestrians cross a concrete footbridge over a canal at
> dusk, with a modern glass building reflecting golden light against a purple twilight
> sky. Moored boats line the canal, and people walk along the towpath or relax on the
> bank." That's two sentences, covers the main points.
>
> Then Keywords. Need 10-18 unique terms, comma-separated. Let's list visible subjects:
> footbridge, pedestrians, canal, dusk, modern glass building, reflections, purple sky,
> moored boats, towpath, urban, waterside, concrete, evening, people, socializing,
> restaurant (wait, is there a restaurant? The hint mentions a nearby restaurant, but is
> it visible? The image has a dark building on the left with a balcony, maybe a
> restaurant, but is it definitely a restaurant? The hint says "nearby restaurant" but the
> image shows a dark building with lights, but maybe not definitely a restaurant. Let's
> skip that to be safe. Other terms: golden light, twilight, boats, walkway, urban area,
> evening, people, concrete structure, water, reflection. Let's check: footbridge,
> pedestrians, canal, dusk, modern glass building, reflections, purple sky, moored boats,
> towpath, urban, waterside, concrete, evening, people, golden light, twilight, walkway,
> water, reflection. Let's count: 18 terms. Let's make sure they are all visible. Yes,
> footbridge is there, pedestrians, canal,
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Verdict:** cutoff | user=avoid
**Why:** token_cap, missing_sections, repetitive_tail, trusted_overlap, context_echo
**Trusted hints:** preserves trusted hints
**Contract:** missing: title, description, keywords
**Utility:** user=avoid | preserves trusted hints | context echo
**Stack / owner:** owner=model
**Token accounting:** prompt=500 | text_est=467 | nontext_est=33 | gen=500 | max=500 | stop=completed
**Next action:** Inspect token cap and stop behavior before treating this as a model-quality failure.

**Metrics:** Load 1.60s | Gen 34.68s | Total 36.54s
**Throughput:** Prompt 287 TPS (500 tok) | Gen 15.4 TPS (500 tok)
**Assessment:** 🟡 C (52/100) | Δ-28 | Mostly echoes context without adding value
**Review Status:** watchlist (context echo, cutoff, missing sections, repetitive)
**Review:** C 52/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "waterfront scene, waterfront a..."); ...

<!-- markdownlint-disable MD028 MD049 -->
>
> Title
> Modern Glass Building and Canal Scene at Dusk
>
> Description
> A modern glass building reflects the golden light of the setting sun against a purple
> twilight sky. People walk along the towpath, relax on the bank, and socialize at a
> nearby restaurant. Moored boats line the canal, completing the lively evening scene.
>
> Keywords
> Glass building, canal, people, boats, modern, urban, waterfront, evening, dusk, purple
> sky, reflection, water, trees, buildings, houses, people walking, people sitting, people
> socializing, restaurant, waterfront, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life, waterfront scene, waterfront activity, waterfront life, waterfront scene,
> waterfront activity, waterfront life, waterfront scene, waterfront activity, waterfront
> life,
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "waterfront scene, waterfront a...")
- Missing sections (title, description, keywords)
- Context echo (33% overlap)
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Verdict:** harness | user=avoid
**Why:** harness:prompt_template, low_hint_overlap
**Trusted hints:** ignores trusted hints
**Contract:** ok
**Utility:** user=avoid | ignores trusted hints
**Stack / owner:** owner=model-config | harness=prompt_template
**Token accounting:** prompt=0 | text_est=467 | nontext_est=0 | gen=0 | max=500 | stop=completed
**Next action:** Inspect model repo config, chat template, and EOS settings.

**Metrics:** Load 0.60s | Gen 65.41s | Total 66.27s
**Throughput:** Prompt 0 TPS (0.0 tok) | Gen 0 TPS (0.0 tok)
**Assessment:** ❌ F (0/100) | Δ-80 | Empty or minimal output
**Review Status:** watchlist (harness)
**Review:** F 0/100 | ⚠️HARNESS:prompt_template; Ignores trusted hints; ...

<!-- markdownlint-disable MD028 MD049 -->
>
>
<!-- markdownlint-enable MD028 MD049 -->

⚠️ **Quality Warnings:**

- ⚠️HARNESS:prompt_template
- Ignores trusted hints
- output:zero_tokens

---

<!-- markdownlint-enable MD013 MD033 -->

<!-- markdownlint-enable MD013 -->
