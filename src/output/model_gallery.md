# Model Output Gallery

_Generated on 2026-04-17 13:13:03 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 5 (top owners: mlx-vlm=4, model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, clean outputs=0/48.
- _Useful now:_ none (no clean A/B shortlist for this run).
- _Review watchlist:_ 48 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=42, neutral=0, worse=6 (baseline F 16/100).
- _Quality signal frequency:_ missing_sections=37, cutoff=31,
  context_ignored=28, trusted_hint_ignored=28, metadata_borrowing=18,
  repetitive=12.
- _Runtime pattern:_ decode dominates measured phase time (90%; 51/53 measured
  model(s)).
- _Phase totals:_ model load=115.13s, prompt prep=0.17s, decode=1064.80s,
  cleanup=5.19s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=48, exception=5.

## 🧭 Review Priorities

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) | Desc 60 | Keywords 0 | Δ-10 | 31.5 tps | context ignored, harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) | Desc 46 | Keywords 42 | Δ+3 | 5.7 tps | context ignored, harness, missing sections
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) | Desc 60 | Keywords 0 | Δ-16 | 131.1 tps | context ignored, cutoff, missing sections
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) | Desc 40 | Keywords 0 | Δ-16 | 127.6 tps | context ignored, cutoff, missing sections
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) | Desc 60 | Keywords 0 | Δ-16 | 77.9 tps | context ignored, cutoff, degeneration, missing sections

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `mlx-vlm` | 4 | Model Error | `Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/X-Reasoner-7B-8bit` |
| `model-config` | 1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16` |

### Actionable Items by Package

#### mlx-vlm

- Qwen/Qwen3-VL-2B-Instruct (Model Error)
  - Error: `Model generation failed for Qwen/Qwen3-VL-2B-Instruct: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be...`
  - Type: `ValueError`
- mlx-community/Qwen2-VL-2B-Instruct-4bit (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen2-VL-2B-Instruct-4bit: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16...`
  - Type: `ValueError`
- mlx-community/Qwen3-VL-2B-Thinking-bf16 (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen3-VL-2B-Thinking-bf16: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16...`
  - Type: `ValueError`
- mlx-community/X-Reasoner-7B-8bit (Model Error)
  - Error: `Model generation failed for mlx-community/X-Reasoner-7B-8bit: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16239) ca...`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`

## Image Metadata

- _Description:_ , Town Centre, Alton, England, United Kingdom, UK
- _Date:_ 2026-04-11 17:53:12 BST
- _Time:_ 17:53:12
- _GPS:_ 51.145067°N, 0.980317°W

## Prompt

<!-- markdownlint-disable MD028 MD037 -->
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
> &#45; Description hint: , Town Centre, Alton, England, United Kingdom, UK
> &#45; Capture metadata: Taken on 2026-04-11 17:53:12 BST (at 17:53:12 local
> time). GPS: 51.145067°N, 0.980317°W.
<!-- markdownlint-enable MD028 MD037 -->

## Quick Navigation

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
- _Best descriptions:_ [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
- _Failed models:_ `Qwen/Qwen3-VL-2B-Instruct`,
  `mlx-community/MolmoPoint-8B-fp16`,
  `mlx-community/Qwen2-VL-2B-Instruct-4bit`,
  `mlx-community/Qwen3-VL-2B-Thinking-bf16`,
  `mlx-community/X-Reasoner-7B-8bit`
- _D/F utility models:_ `HuggingFaceTB/SmolVLM-Instruct`,
  `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`,
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`,
  `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-8B-bf16`,
  `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, +10 more

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ❌ Qwen/Qwen3-VL-2B-Instruct

_Verdict:_ harness | user=avoid
_Why:_ Special control token &lt;|im_end|&gt; appeared in generated text. |
       keywords=36 | context echo=100% | nonvisual metadata reused
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ title words=29 | description sentences=3 | keywords=36
_Utility:_ user=avoid | improves trusted hints | instruction echo | metadata
           borrowing | context echo
_Stack / owner:_ owner=mlx-vlm | harness=stop_token | package=mlx-vlm |
                 stage=Model Error | code=MLX_VLM_DECODE_MODEL
_Token accounting:_ prompt=n/a | text_est=387 | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.

_Status:_ Failed (Model Error)
_Error:_

> Model generation failed for Qwen/Qwen3-VL-2B-Instruct: [broadcast_shapes]
> Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.
_Type:_ `ValueError`
_Phase:_ `decode`
_Code:_ `MLX_VLM_DECODE_MODEL`
_Package:_ `mlx-vlm`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14721, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15074, in _generate_once
    return strict_generate(
        model=model,
    ...<13 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 856, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 735, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 535, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/qwen3_vl/language.py", line 555, in __call__
    position_ids, rope_deltas = self.get_rope_index(
                                ~~~~~~~~~~~~~~~~~~~^
        inputs, image_grid_thw, video_grid_thw, rope_mask
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/qwen3_vl/language.py", line 453, in get_rope_index
    new_positions = mx.where(
        expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
    )
ValueError: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15205, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15097, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14729, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for Qwen/Qwen3-VL-2B-Instruct: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.
```

</details>

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ❌ mlx-community/MolmoPoint-8B-fp16

_Verdict:_ harness | user=avoid
_Why:_ processor error | model config processor load processor
_Trusted hints:_ not evaluated
_Contract:_ not evaluated
_Utility:_ user=avoid
_Stack / owner:_ owner=model-config | package=model-config | stage=Processor
                 Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
_Token accounting:_ prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Inspect model repo config, chat template, and EOS settings.

_Status:_ Failed (Processor Error)
_Error:_

> Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded
> processor has no image_processor; expected multimodal processor.
_Type:_ `ValueError`
_Phase:_ `processor_load`
_Code:_ `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
_Package:_ `model-config`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14791, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14582, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14501, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15205, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15044, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14832, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ❌ mlx-community/Qwen2-VL-2B-Instruct-4bit

_Verdict:_ harness | user=avoid
_Why:_ Special control token &lt;|im_end|&gt; appeared in generated text. |
       keywords=36 | context echo=100% | nonvisual metadata reused
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ title words=29 | description sentences=3 | keywords=36
_Utility:_ user=avoid | improves trusted hints | instruction echo | metadata
           borrowing | context echo
_Stack / owner:_ owner=mlx-vlm | harness=stop_token | package=mlx-vlm |
                 stage=Model Error | code=MLX_VLM_DECODE_MODEL
_Token accounting:_ prompt=n/a | text_est=387 | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.

_Status:_ Failed (Model Error)
_Error:_

> Model generation failed for mlx-community/Qwen2-VL-2B-Instruct-4bit:
> [broadcast_shapes] Shapes (3,1,4096) and (3,1,16680) cannot be broadcast.
_Type:_ `ValueError`
_Phase:_ `decode`
_Code:_ `MLX_VLM_DECODE_MODEL`
_Package:_ `mlx-vlm`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14721, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15074, in _generate_once
    return strict_generate(
        model=model,
    ...<13 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 856, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 735, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 535, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/qwen2_vl/language.py", line 492, in __call__
    position_ids, rope_deltas = self.get_rope_index(
                                ~~~~~~~~~~~~~~~~~~~^
        inputs, image_grid_thw, video_grid_thw, rope_mask
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/qwen2_vl/language.py", line 405, in get_rope_index
    new_positions = mx.where(
        expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
    )
ValueError: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16680) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15205, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15097, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14729, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/Qwen2-VL-2B-Instruct-4bit: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16680) cannot be broadcast.
```

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ❌ mlx-community/Qwen3-VL-2B-Thinking-bf16

_Verdict:_ harness | user=avoid
_Why:_ Special control token &lt;|im_end|&gt; appeared in generated text. |
       keywords=37 | context echo=100% | nonvisual metadata reused
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ title words=29 | description sentences=3 | keywords=37
_Utility:_ user=avoid | improves trusted hints | instruction echo | metadata
           borrowing | context echo
_Stack / owner:_ owner=mlx-vlm | harness=stop_token | package=mlx-vlm |
                 stage=Model Error | code=MLX_VLM_DECODE_MODEL
_Token accounting:_ prompt=n/a | text_est=387 | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.

_Status:_ Failed (Model Error)
_Error:_

> Model generation failed for mlx-community/Qwen3-VL-2B-Thinking-bf16:
> [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.
_Type:_ `ValueError`
_Phase:_ `decode`
_Code:_ `MLX_VLM_DECODE_MODEL`
_Package:_ `mlx-vlm`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14721, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15074, in _generate_once
    return strict_generate(
        model=model,
    ...<13 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 856, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 735, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 535, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/qwen3_vl/language.py", line 555, in __call__
    position_ids, rope_deltas = self.get_rope_index(
                                ~~~~~~~~~~~~~~~~~~~^
        inputs, image_grid_thw, video_grid_thw, rope_mask
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/qwen3_vl/language.py", line 453, in get_rope_index
    new_positions = mx.where(
        expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
    )
ValueError: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15205, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15097, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14729, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/Qwen3-VL-2B-Thinking-bf16: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.
```

</details>

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ❌ mlx-community/X-Reasoner-7B-8bit

_Verdict:_ harness | user=avoid
_Why:_ Special control token &lt;|im_end|&gt; appeared in generated text. |
       keywords=36 | context echo=100% | nonvisual metadata reused
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ title words=29 | description sentences=3 | keywords=36
_Utility:_ user=avoid | improves trusted hints | instruction echo | metadata
           borrowing | context echo
_Stack / owner:_ owner=mlx-vlm | harness=stop_token | package=mlx-vlm |
                 stage=Model Error | code=MLX_VLM_DECODE_MODEL
_Token accounting:_ prompt=n/a | text_est=387 | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.

_Status:_ Failed (Model Error)
_Error:_

> Model generation failed for mlx-community/X-Reasoner-7B-8bit:
> [broadcast_shapes] Shapes (3,1,4096) and (3,1,16239) cannot be broadcast.
_Type:_ `ValueError`
_Phase:_ `decode`
_Code:_ `MLX_VLM_DECODE_MODEL`
_Package:_ `mlx-vlm`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14721, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15074, in _generate_once
    return strict_generate(
        model=model,
    ...<13 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 856, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 735, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 535, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/qwen2_5_vl/language.py", line 494, in __call__
    position_ids, rope_deltas = self.get_rope_index(
                                ~~~~~~~~~~~~~~~~~~~^
        inputs, image_grid_thw, video_grid_thw, rope_mask
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/qwen2_5_vl/language.py", line 406, in get_rope_index
    new_positions = mx.where(
        expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
    )
ValueError: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16239) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15205, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15097, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14729, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/X-Reasoner-7B-8bit: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16239) cannot be broadcast.
```

</details>

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ missing sections: keywords | nonvisual metadata reused
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: keywords
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=458 | text_est=387 | nontext_est=71 | gen=38 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; the requested
               output contract is not being met.

_Metrics:_ Load 0.52s | Gen 0.63s | Total 1.43s
_Throughput:_ Prompt 4,886 TPS (458 tok) | Gen 351 TPS (38 tok)
_Assessment:_ ❌ F (30/100) | Δ+14 | Mostly echoes context without adding value
_Review Status:_ watchlist (metadata borrowing, missing sections)
_Review:_ F 30/100 | missing sections: keywords | nonvisual metadata reused

<!-- markdownlint-disable MD028 MD037 -->
>
> Title: Church in the Town Centre, Alton, England, United Kingdom, UK
> Description: A church in the Town Centre of Alton, England, United Kingdom,
> UK.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ missing sections: title, description, keywords | missing terms: Town,
       Centre, Alton, United, Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=711 | text_est=387 | nontext_est=324 | gen=49 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; the requested
               output contract is not being met.

_Metrics:_ Load 0.54s | Gen 0.85s | Total 1.65s
_Throughput:_ Prompt 2,050 TPS (711 tok) | Gen 323 TPS (49 tok)
_Assessment:_ ✅ B (73/100) | Δ+57 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, missing sections)
_Review:_ B 73/100 | missing sections: title, description, keywords | missing
          terms: Town, Centre, Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> The image shows a photograph of a person standing in front of a tree. The
> person is wearing a green jacket and a blue shirt. The background is a
> forest with green trees and a blue sky. The image is taken during the day.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

_Verdict:_ model_shortcoming | user=avoid
_Why:_ missing sections: title, description, keywords | missing terms: Town,
       Centre, Alton, United, Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=462 | text_est=387 | nontext_est=75 | gen=34 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; the requested
               output contract is not being met.

_Metrics:_ Load 0.63s | Gen 1.20s | Total 2.11s
_Throughput:_ Prompt 1,510 TPS (462 tok) | Gen 340 TPS (34 tok)
_Assessment:_ ✅ B (72/100) | Δ+56 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, missing sections)
_Review:_ B 72/100 | missing sections: title, description, keywords | missing
          terms: Town, Centre, Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> 1&#46; What is the name of the building in the image?
> The name of the building in the image is the "Basilica of St. Mary's."
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

_Verdict:_ clean | user=caveat
_Why:_ nontext prompt burden=87% | missing terms: Town, Centre, Alton, United,
       Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ description sentences=3
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3067 | text_est=387 | nontext_est=2680 | gen=131 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; trusted hint
               coverage is still weak.

_Metrics:_ Load 0.92s | Gen 2.25s | Total 3.45s
_Throughput:_ Prompt 2,912 TPS (3,067 tok) | Gen 180 TPS (131 tok)
_Assessment:_ 🏆 A (94/100) | Δ+78 | None identified
_Review Status:_ watchlist (context ignored)
_Review:_ A 94/100 | nontext prompt burden=87% | missing terms: Town, Centre,
          Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> &#8203;Title:
> Historic Stone Church with Steeple
>
> &#8203;Description:
> A stone-built church with a tall, pointed spire and clock tower stands
> prominently in a quiet rural setting. The building features pointed arch
> windows and a steeply pitched roof, bathed in clear daylight under a blue
> sky with scattered clouds. The church is surrounded by neatly trimmed
> greenery and a paved road with minimal traffic.
>
> &#8203;Keywords:
> stone church, pointed spire, clock tower, arched windows, rural setting,
> red-tiled roofs, weather vane, cross-topped steeple, narrow road, Gothic
> architecture, stone facade, churchyard vegetation, blue sky, daylight
> lighting
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Description sentence violation (3; expected 1-2)
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ nontext prompt burden=74% | missing sections: title, description,
       keywords | missing terms: Town, Centre, Alton, United, Kingdom |
       nonvisual metadata reused
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1484 | text_est=387 | nontext_est=1097 | gen=23 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; the requested
               output contract is not being met.

_Metrics:_ Load 1.55s | Gen 2.67s | Total 4.51s
_Throughput:_ Prompt 975 TPS (1,484 tok) | Gen 32.8 TPS (23 tok)
_Assessment:_ ❌ F (19/100) | Δ+3 | Output lacks detail
_Review Status:_ watchlist (context ignored, metadata borrowing, missing
                 sections)
_Review:_ F 19/100 | nontext prompt burden=74% | missing sections: title,
          description, keywords | missing terms: Town, Centre, Alton, United,
          Kingdom | nonvisual metadata reused

<!-- markdownlint-disable MD028 MD037 -->
>
> &#45; Camera metadata: 1/125s, f/11, 24mm.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Ignores trusted hints

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=95.97% | missing terms: Town,
       Centre, Alton, United, Kingdom | keyword duplication=85%
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ title words=19 | description sentences=5 | keywords=125 | keyword
            duplication=0.85
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=521 | text_est=387 | nontext_est=134 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat this as cap-limited output first; generation exhausted
               the token budget with output/prompt=95.97%.

_Metrics:_ Load 0.54s | Gen 3.23s | Total 4.04s
_Throughput:_ Prompt 7,905 TPS (521 tok) | Gen 179 TPS (500 tok)
_Assessment:_ ✅ B (77/100) | Δ+61 | Missing requested structure
_Review Status:_ watchlist (context ignored, cutoff, repetitive)
_Review:_ B 77/100 | hit token cap (500) | output/prompt=95.97% | missing
          terms: Town, Centre, Alton, United, Kingdom | keyword
          duplication=85%

<!-- markdownlint-disable MD028 MD037 -->
>
> &#8203;Title:
> &#45; 5-10 words, concrete and factual, limited to clearly visible content
> &#45; A wall-mounted artwork featuring a landscape scene
>
> &#8203;Description:
> &#45; The artwork is a painting depicting a serene landscape with rolling hills
> and a clear sky.
> &#45; The painting is framed in a simple wooden frame and is hung on a wall.
> &#45; The colors used in the painting are predominantly earthy tones, with
> shades of green, brown, and blue.
> &#45; The painting is displayed in a well-lit room, with natural light coming in
> from a window.
> &#45; The painting is hung at eye level, making it easily visible to viewers.
>
> &#8203;Keywords:
> &#45; landscape, painting, artwork, rolling hills, clear sky, earthy tones,
> green, brown, blue, wooden frame, wall, natural light, eye level, visible,
> clearly visible, well-lit room, window, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily visible, easily visible, easily visible,
> easily visible, easily visible, easily visible, easily visible, easily
> visible, easily visible, easily
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "easily visible, easily visible...")
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Title length violation (19 words; expected 5-10)
- Description sentence violation (5; expected 1-2)
- Keyword count violation (125; expected 10-18)
- Keyword duplication (85% duplicated terms)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

_Verdict:_ harness | user=avoid
_Why:_ Output is very short relative to prompt size (0.8%), suggesting
       possible early-stop or prompt-handling issues. | nontext prompt
       burden=74% | missing sections: title, description, keywords | missing
       terms: Town, Centre, Alton, United, Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model-config | harness=prompt_template
_Token accounting:_ prompt=1484 | text_est=387 | nontext_est=1097 | gen=12 |
                    max=500 | stop=completed
_Next action:_ Check chat-template and EOS defaults first; the output shape is
               not matching the requested contract.

_Metrics:_ Load 2.51s | Gen 4.25s | Total 7.04s
_Throughput:_ Prompt 876 TPS (1,484 tok) | Gen 5.72 TPS (12 tok)
_Assessment:_ ❌ F (19/100) | Δ+3 | Output lacks detail
_Review Status:_ watchlist (context ignored, harness, missing sections)
_Review:_ F 19/100 | Output is very short relative to prompt size (0.8%),
          suggesting possible early-stop or prompt-handling issues. | nontext
          prompt burden=74% | missing sections: title, description, keywords |
          missing terms: Town, Centre, Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> &#45; Camera: Canon EOS 5D Mark IV.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Ignores trusted hints
- output:output_ratio(0.8%)

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

_Verdict:_ clean | user=caveat
_Why:_ nontext prompt burden=83% | missing terms: Town, Centre, Alton, United,
       Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ title words=2
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2241 | text_est=387 | nontext_est=1854 | gen=70 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; trusted hint
               coverage is still weak.

_Metrics:_ Load 1.77s | Gen 4.34s | Total 6.39s
_Throughput:_ Prompt 1,372 TPS (2,241 tok) | Gen 31.3 TPS (70 tok)
_Assessment:_ 🏆 A (92/100) | Δ+76 | None identified
_Review Status:_ watchlist (context ignored)
_Review:_ A 92/100 | nontext prompt burden=83% | missing terms: Town, Centre,
          Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> &#8203;Title:
> Spiral Spring
>
> &#8203;Description:
> The image shows a close-up of a spiral spring with a gradient of colors from
> light to dark, set against a plain background.
>
> &#8203;Keywords:
> Spiral, spring, gradient, close-up, metal, coil, texture, background, grey,
> silver, engineering, object, detail, industrial, design
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Title length violation (2 words; expected 5-10)
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ missing terms: United, Kingdom | nonvisual metadata reused
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ ok
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=728 | text_est=387 | nontext_est=341 | gen=88 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; trusted hint
               coverage is still weak.

_Metrics:_ Load 2.28s | Gen 4.52s | Total 7.08s
_Throughput:_ Prompt 563 TPS (728 tok) | Gen 31.2 TPS (88 tok)
_Assessment:_ 🏆 A (97/100) | Δ+81 | None identified
_Review Status:_ watchlist (metadata borrowing)
_Review:_ A 97/100 | missing terms: United, Kingdom | nonvisual metadata
          reused

<!-- markdownlint-disable MD028 MD037 -->
>
> &#8203;Title:
> Alton High Street shop fronts, evening light
>
> &#8203;Description:
> A row of shop fronts line a paved street in Alton, England, illuminated by
> evening light. A dark blue sky is visible above the buildings.
>
> &#8203;Keywords:
> Alton, England, shop fronts, high street, buildings, pavement, evening,
> light, sky, blue, street, architecture, storefronts, urban, town centre,
> exterior, row, commercial
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ nontext prompt burden=85% | missing sections: title, description,
       keywords | nonvisual metadata reused
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2668 | text_est=387 | nontext_est=2281 | gen=109 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; the requested
               output contract is not being met.

_Metrics:_ Load 0.96s | Gen 4.53s | Total 5.76s
_Throughput:_ Prompt 1,201 TPS (2,668 tok) | Gen 62.2 TPS (109 tok)
_Assessment:_ ✅ B (71/100) | Δ+55 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (metadata borrowing, missing sections)
_Review:_ B 71/100 | nontext prompt burden=85% | missing sections: title,
          description, keywords | nonvisual metadata reused

<!-- markdownlint-disable MD028 MD037 -->
>
> The church is a large, old, stone building with a tall, pointed roof. It has
> a clock tower and a steeple. The church is located in the town centre of
> Alton, England, United Kingdom, UK. The photo was taken on 2026-04-11 at
> 17:53:12 local time. The GPS coordinates of the church are 51.145067°N,
> 0.980317°W.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

_Verdict:_ clean | user=caveat
_Why:_ nontext prompt burden=87% | missing terms: Town, Centre, Alton, United,
       Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ title words=12
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3068 | text_est=387 | nontext_est=2681 | gen=109 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; trusted hint
               coverage is still weak.

_Metrics:_ Load 1.38s | Gen 4.66s | Total 6.32s
_Throughput:_ Prompt 1,257 TPS (3,068 tok) | Gen 62.2 TPS (109 tok)
_Assessment:_ 🏆 A (94/100) | Δ+78 | None identified
_Review Status:_ watchlist (context ignored)
_Review:_ A 94/100 | nontext prompt burden=87% | missing terms: Town, Centre,
          Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> &#8203;Title:
> Church with tall spire and clock tower, clear blue sky, roadside view
>
> &#8203;Description:
> A stone-built church featuring a prominent tall spire and clock tower,
> positioned beside a road. Visible details include arched windows, a weather
> vane atop the spire, and a clear blue sky with scattered clouds.
>
> &#8203;Keywords:
> church, stone architecture, tall spire, clock tower, arched windows, weather
> vane, roadside, clear blue sky, scattered clouds, traditional design, road
> junction, stone facade, traditional English architecture
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Title length violation (12 words; expected 5-10)
- Ignores trusted hints

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=88.18% | missing sections: title,
       description, keywords | missing terms: Town, Centre, Alton, United,
       Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=567 | text_est=387 | nontext_est=180 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 0.61s | Gen 4.70s | Total 5.58s
_Throughput:_ Prompt 1,690 TPS (567 tok) | Gen 131 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-16 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ F 0/100 | hit token cap (500) | output/prompt=88.18% | missing
          sections: title, description, keywords | missing terms: Town,
          Centre, Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> <row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1>
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <row_1_col_1>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

_Verdict:_ clean | user=caveat
_Why:_ nontext prompt burden=87% | missing terms: Town, Centre, Alton, United,
       Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ ok
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3068 | text_est=387 | nontext_est=2681 | gen=126 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; trusted hint
               coverage is still weak.

_Metrics:_ Load 1.35s | Gen 4.89s | Total 6.52s
_Throughput:_ Prompt 1,232 TPS (3,068 tok) | Gen 65.2 TPS (126 tok)
_Assessment:_ 🏆 A (100/100) | Δ+83 | None identified
_Review Status:_ watchlist (context ignored)
_Review:_ A 100/100 | nontext prompt burden=87% | missing terms: Town, Centre,
          Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> &#8203;Title:
> Stone Church with Tall Central Tower and Clock
>
> &#8203;Description:
> The image shows a stone-built church with a prominent central tower topped
> by a steep spire, featuring a clock face and narrow arched windows. The
> structure is set against a clear blue sky with scattered clouds, and the
> photograph is taken from a roadside angle, capturing the building’s Gothic
> Revival architectural style.
>
> &#8203;Keywords:
> stone church, central tower, steep spire, clock face, narrow arched windows,
> Gothic Revival architecture, clear blue sky, scattered clouds, roadside
> view, church facade, stone construction, symmetrical design, church clock,
> church tower, daylight lighting
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ missing terms: Centre, United, Kingdom | nonvisual metadata reused
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ ok
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=733 | text_est=387 | nontext_est=346 | gen=87 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; trusted hint
               coverage is still weak.

_Metrics:_ Load 3.11s | Gen 4.90s | Total 8.30s
_Throughput:_ Prompt 572 TPS (733 tok) | Gen 27.3 TPS (87 tok)
_Assessment:_ 🏆 A (98/100) | Δ+82 | None identified
_Review Status:_ watchlist (metadata borrowing)
_Review:_ A 98/100 | missing terms: Centre, United, Kingdom | nonvisual
          metadata reused

<!-- markdownlint-disable MD028 MD037 -->
>
> Title: Stone church with a tall spire in Alton
>
> Description: A stone church with a tall pointed spire and a clock face
> stands beside a road under a blue sky with light clouds. A person in red
> clothing walks along the pavement near the church entrance.
>
> Keywords: church, spire, stone, clock, road, blue sky, clouds, architecture,
> England, town, pavement, person, red, daylight, building, exterior
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=109.17% | missing sections: title,
       description, keywords | missing terms: Town, Centre, Alton, United,
       Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=458 | text_est=387 | nontext_est=71 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 0.58s | Gen 4.96s | Total 5.82s
_Throughput:_ Prompt 4,212 TPS (458 tok) | Gen 113 TPS (500 tok)
_Assessment:_ 🟠 D (45/100) | Δ+29 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ D 45/100 | hit token cap (500) | output/prompt=109.17% | missing
          sections: title, description, keywords | missing terms: Town,
          Centre, Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> 镗.8 SqlConnectionDAQbabong Painting Glasses, rightandexvoracessallz, Drawing
> Wheeliazeliasesallz, Guitar郧itzer Page Mosaicsmitharts
> Tattozkikidgedownlodgedownlodged Rigiazdrselongs Guitarazz , Painting
> Drawing Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting Painting Painting Painting
> Painting Painting Painting Painting Painting
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (Painting)
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=29.99% | nontext prompt burden=77%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1667 | text_est=387 | nontext_est=1280 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 0.76s | Gen 5.03s | Total 6.06s
_Throughput:_ Prompt 4,021 TPS (1,667 tok) | Gen 123 TPS (500 tok)
_Assessment:_ 🟠 D (45/100) | Δ+29 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ D 45/100 | hit token cap (500) | output/prompt=29.99% | nontext
          prompt burden=77% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> treasured unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (unt)
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

_Verdict:_ clean | user=caveat
_Why:_ nontext prompt burden=88% | missing terms: Town, Centre, Alton, United,
       Kingdom | keywords=19
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ description sentences=3 | keywords=19
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3261 | text_est=387 | nontext_est=2874 | gen=106 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; trusted hint
               coverage is still weak.

_Metrics:_ Load 1.71s | Gen 5.21s | Total 7.20s
_Throughput:_ Prompt 1,601 TPS (3,261 tok) | Gen 39.1 TPS (106 tok)
_Assessment:_ 🏆 A (89/100) | Δ+73 | None identified
_Review Status:_ watchlist (context ignored)
_Review:_ A 89/100 | nontext prompt burden=88% | missing terms: Town, Centre,
          Alton, United, Kingdom | keywords=19

<!-- markdownlint-disable MD028 MD037 -->
>
> &#8203;Title:
> Church with tall spire and clock
>
> &#8203;Description:
> A large stone church with a tall spire and clock, set against a blue sky
> with some clouds. The church has arched windows and a red-tiled roof. There
> is a person walking near the entrance and a road in front.
>
> &#8203;Keywords:
> Church, spire, clock, stone, red roof, arched windows, blue sky, clouds,
> person, road, entrance, stone wall, shrubs, street sign, sky, building,
> architecture, outdoor, daylight
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (19; expected 10-18)
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=69.54% | missing sections: title,
       description, keywords | missing terms: Town, Centre, Alton, United,
       Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=719 | text_est=387 | nontext_est=332 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 1.48s | Gen 5.60s | Total 7.35s
_Throughput:_ Prompt 608 TPS (719 tok) | Gen 124 TPS (500 tok)
_Assessment:_ ❌ F (2/100) | Δ-14 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, metadata borrowing,
                 missing sections)
_Review:_ F 2/100 | hit token cap (500) | output/prompt=69.54% | missing
          sections: title, description, keywords | missing terms: Town,
          Centre, Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
>
>
> &#42;
> 1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23.24.25.26.27.28.29.30.31.32.33.34.35.36.37.38.39.40.41.42.43.44.45.46.47.48.49.50.51.52.53.54.55.56.57.58.59.60.61.62.63.64.65.66.67.68.69.70.71.72.73.74.75.76.77.78.79.80.81.82.83.84.85.86.87.88.89.90.91.92.93.94.95.96.97.98.99.100.101.102.103.104.105.106.107.108.109.110.111.112.113.114.115.116.117.118.119.120.121.122.123.124.125.126.127.128.129.130.131.132.133.134.135.136.137.138.139.140.141.142.143.144.145.146.147.148.149.150.151.1
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints
- suspicious_precision: 26 overly precise numbers

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=34.46% | nontext prompt burden=73%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1451 | text_est=387 | nontext_est=1064 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 2.04s | Gen 5.88s | Total 8.20s
_Throughput:_ Prompt 986 TPS (1,451 tok) | Gen 128 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-16 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ F 0/100 | hit token cap (500) | output/prompt=34.46% | nontext
          prompt burden=73% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> 问题。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议答案。因此，建议
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

_Verdict:_ harness | user=avoid
_Why:_ Tokenizer space-marker artifacts (for example Ġ) appeared in output
       (about 53 occurrences). | nontext prompt burden=85% | missing sections:
       description, keywords | missing terms: Town, Centre, Alton, United,
       Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: description, keywords | title words=56
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=mlx-vlm | harness=encoding
_Token accounting:_ prompt=2569 | text_est=387 | nontext_est=2182 | gen=77 |
                    max=500 | stop=completed
_Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
               user-facing text.

_Metrics:_ Load 2.06s | Gen 6.46s | Total 8.80s
_Throughput:_ Prompt 724 TPS (2,569 tok) | Gen 31.5 TPS (77 tok)
_Assessment:_ ❌ F (6/100) | Δ-10 | Output too short to be useful
_Review Status:_ watchlist (context ignored, harness, missing sections)
_Review:_ F 6/100 | Tokenizer space-marker artifacts (for example Ġ) appeared
          in output (about 53 occurrences). | nontext prompt burden=85% |
          missing sections: description, keywords | missing terms: Town,
          Centre, Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:ĠStoneĠChurchĠwithĠTowerĠandĠSpireĊĊDescription:ĠAĠstoneĠchurchĠwithĠaĠtallĠspireĠandĠclockĠtowerĠstandsĠatĠaĠroadĠjunctionĠunderĠaĠclearĠblueĠsky.ĠTheĠbuildingĠfeaturesĠarchedĠwindowsĠandĠaĠred-tiledĠroof.ĊĊKeywords:ĠstoneĠchurch,Ġspire,ĠclockĠtower,ĠarchedĠwindows,ĠredĠroof,ĠblueĠsky,ĠroadĠjunction,ĠchurchĠbuilding,ĠGothicĠarchitecture,Ġdaylight
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (description, keywords)
- Title length violation (56 words; expected 5-10)
- Ignores trusted hints
- token_encoding:bpe_space_leak(53)

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=29.99% | nontext prompt burden=77%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1667 | text_est=387 | nontext_est=1280 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 0.71s | Gen 6.55s | Total 7.52s
_Throughput:_ Prompt 856 TPS (1,667 tok) | Gen 122 TPS (500 tok)
_Assessment:_ 🟠 D (45/100) | Δ+29 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ D 45/100 | hit token cap (500) | output/prompt=29.99% | nontext
          prompt burden=77% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> treasured unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (unt)
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ missing sections: title, description, keywords | missing terms: United,
       Kingdom
_Trusted hints:_ improves trusted hints
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=429 | text_est=387 | nontext_est=42 | gen=90 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; the requested
               output contract is not being met.

_Metrics:_ Load 1.57s | Gen 7.16s | Total 9.00s
_Throughput:_ Prompt 224 TPS (429 tok) | Gen 18.6 TPS (90 tok)
_Assessment:_ ✅ B (76/100) | Δ+60 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (missing sections)
_Review:_ B 76/100 | missing sections: title, description, keywords | missing
          terms: United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> Title
> Alton Church
>
> Description
> The image shows a large stone church with a tall steeple and a red roof,
> situated in a town centre. The church is surrounded by a road and a
> sidewalk, with a person walking on the sidewalk in the foreground. The sky
> is blue with white clouds.
>
> Keywords
> church, stone, steeple, red roof, town centre, road, sidewalk, person, blue
> sky, white clouds
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ missing terms: United, Kingdom | nonvisual metadata reused
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ ok
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=728 | text_est=387 | nontext_est=341 | gen=99 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; trusted hint
               coverage is still weak.

_Metrics:_ Load 3.41s | Gen 7.32s | Total 11.00s
_Throughput:_ Prompt 543 TPS (728 tok) | Gen 17.8 TPS (99 tok)
_Assessment:_ 🏆 A (97/100) | Δ+81 | None identified
_Review Status:_ watchlist (metadata borrowing)
_Review:_ A 97/100 | missing terms: United, Kingdom | nonvisual metadata
          reused

<!-- markdownlint-disable MD028 MD037 -->
>
> &#8203;Title:
> Alton High Street, shops, pedestrianised town centre
>
> &#8203;Description:
> A daytime view of a pedestrianised high street in Alton, England, featuring
> shops with awnings and street furniture. The scene is brightly lit with
> shadows cast by the afternoon sun.
>
> &#8203;Keywords:
> Alton, high street, shops, awnings, pedestrianised, town centre, street
> furniture, buildings, sunlight, shadows, England, UK, daytime, pavement,
> storefronts, retail, architecture, urban
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

_Verdict:_ model_shortcoming | user=avoid
_Why:_ missing sections: title, description, keywords | missing terms: Centre,
       Alton, United, Kingdom
_Trusted hints:_ improves trusted hints
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=727 | text_est=387 | nontext_est=340 | gen=328 |
                    max=500 | stop=completed
_Next action:_ Treat as a model limitation for this prompt; the requested
               output contract is not being met.

_Metrics:_ Load 2.22s | Gen 7.56s | Total 10.06s
_Throughput:_ Prompt 1,702 TPS (727 tok) | Gen 48.7 TPS (328 tok)
_Assessment:_ ✅ B (80/100) | Δ+64 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (missing sections)
_Review:_ B 80/100 | missing sections: title, description, keywords | missing
          terms: Centre, Alton, United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> &#35;# Image Description:
>
> The image shows a large, historic stone church situated on a corner lot in a
> town. The church is the dominant feature, occupying the majority of the
> frame. It has a traditional Gothic architectural style, with pointed arches,
> large arched windows, and a prominent, tall spire topped with a weather
> vane. The stone is a mix of light and darker hues, showing signs of age and
> weathering. The roof is covered in red tiles.
>
> The church has a central tower with a clock face visible on one side.
> Smaller sections of the church extend outwards on either side of the tower.
> There are several arched windows with intricate stone detailing. A few small
> crosses are visible atop the roofline.
>
> The church is located at a street corner, with roads visible in the
> foreground. A pedestrian, wearing a dark jacket and carrying a bag, is
> walking across the road in the foreground on the left side of the frame.
> There are road markings and a traffic island visible.
>
> The sky above is a clear, bright blue with scattered white clouds. The
> lighting suggests it is daytime, with sunlight illuminating the church from
> the left, casting shadows on the right side of the building.
>
> The surrounding area includes some greenery, with bushes and small trees
> planted around the church grounds. There are also some residential buildings
> visible in the background on the left side of the frame. A few street signs
> are present near the church.
>
> The overall impression is of a well-maintained, historic building in a
> peaceful town setting on a sunny day. The image is taken from a slightly low
> angle, emphasizing the height and grandeur of the church.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=34.46% | nontext prompt burden=73%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1451 | text_est=387 | nontext_est=1064 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 2.17s | Gen 7.77s | Total 10.22s
_Throughput:_ Prompt 1,727 TPS (1,451 tok) | Gen 114 TPS (500 tok)
_Assessment:_ 🟠 D (45/100) | Δ+29 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, cutoff, degeneration, missing
                 sections, repetitive)
_Review:_ D 45/100 | hit token cap (500) | output/prompt=34.46% | nontext
          prompt burden=73% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> 8th prize.◊th prize.◊th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th prize.th prize.th prize.th
> prize.th prize.th prize.th prize.th prize.th
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (prize.th)
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Output degeneration (incomplete_sentence: ends with 'th')
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=34.46% | nontext prompt burden=73%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1451 | text_est=387 | nontext_est=1064 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 3.30s | Gen 7.89s | Total 11.48s
_Throughput:_ Prompt 1,669 TPS (1,451 tok) | Gen 77.9 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-16 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, degeneration, missing
                 sections)
_Review:_ F 0/100 | hit token cap (500) | output/prompt=34.46% | nontext
          prompt burden=73% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
>
> 0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Output degeneration (character_loop: '.0' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=39.53% | nontext prompt burden=69%
       | missing terms: Town, Centre, Alton, United, Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ title words=3 | keywords=92 | keyword duplication=0.73
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1265 | text_est=387 | nontext_est=878 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat this as cap-limited output first; generation exhausted
               the token budget with output/prompt=39.53%.

_Metrics:_ Load 0.86s | Gen 9.69s | Total 10.82s
_Throughput:_ Prompt 3,888 TPS (1,265 tok) | Gen 55.7 TPS (500 tok)
_Assessment:_ 🟡 C (63/100) | Δ+47 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, repetitive)
_Review:_ C 63/100 | hit token cap (500) | output/prompt=39.53% | nontext
          prompt burden=69% | missing terms: Town, Centre, Alton, United,
          Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> Title: Abstract Vortex Pattern
>
> Description: A complex, swirling pattern of concentric circles with a
> gradient of colors from blue to red.
>
> Keywords: swirling, pattern, concentric, gradient, colors, abstract, vortex,
> blue, red, pattern, circles, visual, art, design, texture, visual effect,
> artwork, creative, design, artistic, visual, creative design, artistic
> visual, creative visual design, artistic visual design, creative visual art,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, cre
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "creative visual art design,...")
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Title length violation (3 words; expected 5-10)
- Keyword count violation (92; expected 10-18)
- Keyword duplication (73% duplicated terms)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=39.53% | nontext prompt burden=69%
       | missing terms: Town, Centre, Alton, United, Kingdom
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ title words=3 | keywords=92 | keyword duplication=0.73
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1265 | text_est=387 | nontext_est=878 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat this as cap-limited output first; generation exhausted
               the token budget with output/prompt=39.53%.

_Metrics:_ Load 0.93s | Gen 10.15s | Total 11.36s
_Throughput:_ Prompt 1,723 TPS (1,265 tok) | Gen 55.4 TPS (500 tok)
_Assessment:_ 🟡 C (63/100) | Δ+47 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, repetitive)
_Review:_ C 63/100 | hit token cap (500) | output/prompt=39.53% | nontext
          prompt burden=69% | missing terms: Town, Centre, Alton, United,
          Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> Title: Abstract Vortex Pattern
>
> Description: A complex, swirling pattern of concentric circles with a
> gradient of colors from blue to red.
>
> Keywords: swirling, pattern, concentric, gradient, colors, abstract, vortex,
> blue, red, pattern, circles, visual, art, design, texture, visual effect,
> artwork, creative, design, artistic, visual, creative design, artistic
> visual, creative visual design, artistic visual design, creative visual art,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, creative visual art design, creative visual art
> design, creative visual art design, creative visual art design, creative
> visual art design, creative visual art design, creative visual art design,
> creative visual art design, cre
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "creative visual art design,...")
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Title length violation (3 words; expected 5-10)
- Keyword count violation (92; expected 10-18)
- Keyword duplication (73% duplicated terms)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=10.97% | nontext prompt burden=92%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=4556 | text_est=387 | nontext_est=4169 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 1.16s | Gen 13.04s | Total 14.50s
_Throughput:_ Prompt 3,654 TPS (4,556 tok) | Gen 45.0 TPS (500 tok)
_Assessment:_ 🟡 C (52/100) | Δ+36 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ C 52/100 | hit token cap (500) | output/prompt=10.97% | nontext
          prompt burden=92% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
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
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "- output only the...")
- Excessive bullet points (46)
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=14.92% | nontext prompt burden=88%
       | missing sections: title, keywords
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, keywords
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3352 | text_est=387 | nontext_est=2965 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, keywords remained incomplete.

_Metrics:_ Load 1.61s | Gen 14.75s | Total 16.65s
_Throughput:_ Prompt 1,425 TPS (3,352 tok) | Gen 41.8 TPS (500 tok)
_Assessment:_ 🏆 A (84/100) | Δ+68 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (cutoff, degeneration, metadata borrowing, missing
                 sections, reasoning leak)
_Review:_ A 84/100 | hit token cap (500) | output/prompt=14.92% | nontext
          prompt burden=88% | missing sections: title, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> &#8203;Here are my reasoning steps:
> We need to produce a catalog entry with Title, Description, Keywords. Use
> only details that are clearly visible in the image. The image shows a church
> building with a tall stone tower, pointed roof, weather vane, clock face,
> Gothic style windows, stone and brickwork, red-tiled roof, set against a
> clear blue sky with some clouds. There's a road in front, a pedestrian
> crossing sign, a no entry sign, a blue post box, a person walking, some
> greenery, hedges, trees. The building appears to be a parish church, likely
> in a town centre. The hint says "Town Centre, Alton, England, United
> Kingdom, UK". But we cannot assume location unless visible. The image does
> not show any sign indicating Alton. So we cannot include location unless
> it's visible. The hint says "Town Centre, Alton, England, United Kingdom,
> UK". But we must only include details that are definitely visible. The image
> does not show a sign that says Alton. So we cannot include location.
> However, we can include "church", "tower", "clock", "weather vane", "red
> roof", "stone facade", "Gothic windows", "road", "pedestrian", "blue post
> box", "clear blue sky", "clouds". The title: 5-10 words, concrete and
> factual, limited to clearly visible content. Something like "St. Mary's
> Parish Church with stone tower and clock". But we don't know the name of the
> church. The image shows a church building but no name sign. So we cannot
> include a name. Title could be "Stone Gothic church with tall tower and
> clock". That's 7 words. Or "Gothic stone church with clock tower and red
> roof". That's 10 words. Let's count: "Gothic" (1) "stone" (2) "church" (3)
> "with" (4) "clock" (5) "tower" (6) "and" (7) "red" (8) "roof" (9). That's 9
> words. Good.
>
> Description: 1-2 factual sentences describing main visible subject, setting,
> &#8203;lighting, action, and other distinctive visible details. So something like:
> "A Gothic-style stone church with a tall pointed tower and clock face
> stands on a sunny street under a
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Output degeneration (incomplete_sentence: ends with 'a')
- Missing sections (title, keywords)
- Reasoning leak (here are my reasoning steps)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=7.61% | nontext prompt burden=94% |
       missing sections: title, description, keywords
_Trusted hints:_ improves trusted hints
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=6574 | text_est=387 | nontext_est=6187 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 1.29s | Gen 14.85s | Total 16.42s
_Throughput:_ Prompt 923 TPS (6,574 tok) | Gen 67.9 TPS (500 tok)
_Assessment:_ ✅ B (79/100) | Δ+63 | Missing requested structure
_Review Status:_ watchlist (cutoff, missing sections, reasoning leak)
_Review:_ B 79/100 | hit token cap (500) | output/prompt=7.61% | nontext
          prompt burden=94% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> &lt;think&gt;Got it, let's tackle this. First, I need to analyze the image for
> cataloguing metadata. The user provided specific instructions, so I need to
> follow them exactly.
>
> First section is Title. The title needs to be 5-10 words, concrete and
> factual, limited to clearly visible content. Looking at the image, the main
> subject is a church with a tall spire. So a title like "Church with Spire"
> or "Stone Church with Tall Spire". Let's pick "Church with Tall Spire".
>
> Next section is Description. It needs 1-2 factual sentences describing the
> main visible subject, setting, lighting, action, and other distinctive
> visible details. Looking at the image: The main subject is a stone church
> with a tall, pointed spire topped with a weather vane. The church has a
> clock on the tower, arched windows, and a red-tiled roof. The setting is a
> town center intersection with paved roads, hedges, and a blue information
> board near the church. The lighting is bright with a clear blue sky and some
> scattered clouds. There is a person walking on the left side of the road.
> The overall scene is a peaceful town center with the historic church as a
> prominent landmark.
>
> Now, need to condense this into 1-2 sentences. Let's draft: "A stone church
> with a tall, pointed spire and a weather vane stands at a town center
> intersection. The church features a clock, arched windows, and a red-tiled
> roof. The setting includes paved roads, hedges, and a blue information
> board, with a clear blue sky and scattered clouds overhead. A person walks
> on the left side of the road."
>
> That's a bit long. Let's trim. Maybe: "A stone church with a tall, pointed
> spire and a weather vane is situated at a town center intersection. The
> church has a clock, arched windows, and a red-tiled roof. The scene includes
> paved roads, hedges, a blue information board, and a clear blue sky with
> scattered clouds. A person walks on the left side of the road."
>
> That's better. Now, check if all details are clearly visible. Yes. Now, the
> Keywords section. Need 10-18 unique comma-separated terms based on clearly
> visible subjects, setting, colors, composition, and style. Let's list
> &#8203;visible elements:
>
> &#45; Subject: Church, stone church, church with spire
> &#45; Setting
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=18.24% | nontext prompt burden=86%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2741 | text_est=387 | nontext_est=2354 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 1.91s | Gen 17.46s | Total 19.65s
_Throughput:_ Prompt 2,399 TPS (2,741 tok) | Gen 31.7 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-16 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ F 0/100 | hit token cap (500) | output/prompt=18.24% | nontext
          prompt burden=86% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> <fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image>
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <fake_token_around_image>

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=22.31% | nontext prompt burden=83%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2241 | text_est=387 | nontext_est=1854 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 1.73s | Gen 17.64s | Total 19.65s
_Throughput:_ Prompt 2,938 TPS (2,241 tok) | Gen 33.3 TPS (500 tok)
_Assessment:_ 🟠 D (50/100) | Δ+34 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ D 50/100 | hit token cap (500) | output/prompt=22.31% | nontext
          prompt burden=83% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> WRAPPER struggverständlich.MSG新浪财经WRAPPERWRAPPER
> barcelona的.ERR.ERR.ERR新浪财经迓傥的s
> Publié tanggalWRAPPER Rencontre的.ERR romaWRAPPER pérdida的的 barcelona
> struggs的
> pérdida.ErrorCode 若要くなりました矧的 Rencontre >",ardware Rencontre
> enthus.ERR性价且笑意的，
> 捋
> .AD的 RencontreHôtel•0、 strugggles Rencontre的的的 strugg strugg strugg strugg的的
> strugg strugg strugg strugg的 formulaire nâ了吧 proprié struggल可以更好 strugg
> strugg strugg的 Rencontre commentaire的.ERR Rencontre的.ERR Rencontre的.ERR''
> 若要.ERR Rencontre.ERR Rencontre的 strugg tô.ERR Rencontre的的 strugg tô
> .ERR Rencontre附 Rencontre的的的不.ERR, strugg tô
> 大.ERR'' Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre怙殄
> Rencontre的.ERR'' Rencontre的.ERR Rencontre tô Rencontre的 strugg Rencontre
> Rencontre Rencontre Rencontre strugg_subtitleensibly struggl strugg strugg
> strugg strugg/includes重要因素的的的的不 Rencontre的 struggoller strugg tô
> Publié.blank Rencontre#ab strugg tô
> 查看详情.ERR Rencontre.ERR Rencontre Rencontre Rencontre Rencontre Rencontre又
> strugg tô Rencontre Rencontre Rencontre strugg整顿 strugg tôonne并不是很 Rencontre
>
> &#35; Rencontre的.ERR struggoller struggoller'' Rencontre struggoller strugg tô
> Rencontre.ERR Rencontre又 Rencontre Rencontre Rencontre tô Rencontre struggl
> strugg tô Rencontre并不是很赶 struggll_then.ERR''''s struggoller. composition
> struggl strugg tô长大 strugg struggoller.ERR Rencontre并不是很.ERR Rencontre.ERR
> Rencontre Rencontre Rencontre的 strugg Rencontre Rencontre Rencontre
> Rencontre Rencontre &#42;/
>
>
> |-.ERR Rencontre Rencontre有时 HOWEVER.ERR Rencontre Rencontre Rencontre又
> struggoller strugg Rencontre Rencontre struggoller struggoller的.ERR
> Rencontre并不是很.ERR Rencontre的查看详情.ERR Rencontre struggoller strugg tô
> Rencontre并不是很 Rencontre strugg Rencontre Rencontre the' nâ Rencontre有可能
> HOWEVER.ERR Rencontre的 struggoller strugg Rencontre.ERR''''扪bout的的不.ERR
> struggoller strugg Rencontre太茫茫 struggoller strugg Rencontre Rencontre
> Rencontre又 strugg Rencontre太 Rencontre的的.ERR Rencontre Rencontre Rencontre
> Rencontre Rencontre strugg Rencontre又 the strugg Rencontre
> Rencontre抱longleftrightarrow Rencontre strugg tôu strugg tô
>
> 的 struggoller strugg tôu并不是很 Rencontre struggoller Rencontre Rencontre
> Rencontre struggoller strugg Rencontre Rencontre tô Rencontre struggoller
> strugg Rencontre Rencontre.ERR''便可 magistrate wag性价 struggoller strugg
> Rencontre Rencontre Rencontre struggoller strugg tô Rencontre并不是很 Rencontre
> struggoller Rencontre Rencontre Rencontre尤其.ERR.ERR Rencontre的 strugg
> Rencontre strugg tô并不是很 Rencontre.ERR Rencontre Rencontre the Rencontre
> Rencontre Rencontre Rencontre Rencontre.ERR Rencontre太田 ( /ORWRAPPER
> Rencontre的 strugg Rencontre strugg tôetu 抽 Rencontre struggoller struggoller
> struggoller struggoller struggoller'' Rencontre的 strugg tô strugg Rencontre
> Rencontre an的不 Rencontre的的的不 Rencontre的不 Rencontre.ERR
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=7.61% | nontext prompt burden=94% |
       missing sections: title, description
_Trusted hints:_ improves trusted hints
_Contract:_ missing: title, description | keywords=45 | keyword
            duplication=0.56
_Utility:_ user=avoid | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=6574 | text_est=387 | nontext_est=6187 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description remained incomplete.

_Metrics:_ Load 1.39s | Gen 17.83s | Total 19.50s
_Throughput:_ Prompt 926 TPS (6,574 tok) | Gen 48.3 TPS (500 tok)
_Assessment:_ ✅ B (79/100) | Δ+63 | Missing requested structure
_Review Status:_ watchlist (cutoff, missing sections, reasoning leak)
_Review:_ B 79/100 | hit token cap (500) | output/prompt=7.61% | nontext
          prompt burden=94% | missing sections: title, description

<!-- markdownlint-disable MD028 MD037 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, the Title needs to be
> 5-10 words, concrete and factual. Looking at the image, the main subject is
> a church with a tall spire. So possible titles: "Church with tall spire",
> "Stone church with clock tower", "Church building with pointed spire". Need
> to keep it factual. The most concrete is "Church with tall spire" or maybe
> "Stone church with clock and spire". Wait, the title should be 5-10 words.
> Let's check the image: the main subject is a church with a prominent tall
> spire, stone construction, clock on the tower. So a good title would be
> "Stone church with tall spire". That's 6 words, concrete.
>
> Next, Description: need 1-2 factual sentences. The main subject is a stone
> church with a tall pointed spire, clock on the tower, Gothic-style windows.
> The setting is a town street intersection with road markings, a pedestrian,
> some trees and hedges. Lighting is bright, sunny day with blue sky and some
> clouds. So a description: "A stone church with a tall pointed spire, clock
> on the tower, and Gothic-style windows stands at a town street intersection.
> The scene is bright and sunny with a blue sky and some clouds." That's two
> sentences, factual, no inferences.
>
> Keywords: 10-18 unique comma-separated terms. Based on visible content:
> stone church, tall spire, clock tower, Gothic windows, town street, road
> markings, pedestrian, blue sky, clouds, stone construction, pointed spire,
> clock face, arched windows, hedge, tree, intersection. Need to pick the most
> &#8203;clear ones. Let's list the most definitely visible:
>
> &#45; Stone church
> &#45; Tall spire
> &#45; Clock tower
> &#45; Gothic windows
> &#45; Town street
> &#45; Road markings
> &#45; Pedestrian
> &#45; Blue sky
> &#45; Clouds
> &#45; Stone construction
> &#45; Pointed spire
> &#45; Clock face
> &#45; Arched windows
> &#45; Hedge
> &#45; Tree
> &#45; Intersection
>
> Now need to pick 10-18 unique ones. Let's choose the most prominent and
> &#8203;clearly visible:
>
> &#45; Stone church
> &#45; Tall spire
> &#45; Clock tower
> &#45; Gothic windows
> &#45; Town street
> &#45; Road markings
> &#45; Pedestrian
> &#45; Blue sky
> &#45; Clouds
> &#45; Stone construction
> &#45; Pointed spire
> &#45; Clock
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Excessive bullet points (28)
- Missing sections (title, description)
- Keyword count violation (45; expected 10-18)
- Keyword duplication (56% duplicated terms)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=28.26% | nontext prompt burden=78%
       | missing sections: title, description, keywords
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints | instruction echo | metadata
           borrowing | context echo
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1769 | text_est=387 | nontext_est=1382 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 10.22s | Gen 19.50s | Total 30.01s
_Throughput:_ Prompt 252 TPS (1,769 tok) | Gen 42.4 TPS (500 tok)
_Assessment:_ ✅ B (66/100) | Δ+49 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context echo, cutoff, instruction echo, metadata
                 borrowing, missing sections, reasoning leak)
_Review:_ B 66/100 | hit token cap (500) | output/prompt=28.26% | nontext
          prompt burden=78% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> Let's tackle this step by step. First, the title needs to be 5-10 words,
> concrete and factual. The image is of a church in St Albans, right? Wait,
> the prompt mentions "St Albans Cathedral" but wait no, wait the user's
> prompt has some garbled text but the context says existing metadata hints
> include St Albans Cathedral. Wait no, wait the user's initial prompt has
> some messed-up text but the context says the existing metadata hints include
> St Albans Cathedral. Wait no, let's focus on the image description. The
> main subject is a church with a tall spire, stained glass windows, stone
> walls, a graveyard, a road with a person walking. The title should be
> something like "St Albans Cathedral from the Town Centre". Wait but let's
> check the context: the existing metadata hints say "Description hint: , Town
> Centre, Alton, England, United Kingdom, UK" no wait no, the context says
> "Context: Existing metadata hints (high confidence; use only when visually
> confirmed): - Description hint: , Town Centre, Alton, England, United
> Kingdom, UK - Capture metadata: Taken on 2026-04-11 17:53:12 BST (at
> 17:53:12 local time). GPS: 51.145067°N, 0.980317°W." Wait no, maybe that's a
> typo, but the image is of St Albans Cathedral. Wait but the user's initial
> prompt has some garbled text but the context says the existing metadata
> hints include St Albans Cathedral. Wait no, let's go back. The user's
> initial prompt has a lot of garbled text but the context says the existing
> metadata hints include St Albans Cathedral. Wait no, the user's initial
> prompt says "St Albans Cathedral" in the first line. Oh right, the first
> line of the user's prompt is "St Albans Cathedral from the Town Centre". So
> the title should be "St Albans Cathedral from the Town Centre". Then the
> description: need 1-2 factual sentences. Main subject: St Albans Cathedral
> with a tall spire, stone walls, stained glass windows, a graveyard, a road
> with a person walking. Lighting: bright daylight.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (title, description, keywords)
- Reasoning leak (context: existing metadata hints, description hint:)
- Context echo (100% overlap)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=30.40% | nontext prompt burden=76%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1645 | text_est=387 | nontext_est=1258 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 1.18s | Gen 24.53s | Total 25.99s
_Throughput:_ Prompt 117 TPS (1,645 tok) | Gen 51.3 TPS (500 tok)
_Assessment:_ ✅ B (75/100) | Δ+59 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ B 75/100 | hit token cap (500) | output/prompt=30.40% | nontext
          prompt burden=76% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> Bright, sunny day, shadows visible on ground. Road visible, no vehicles
> present. Man wearing red jacket, blue jeans, black shoes. Man walking on
> sidewalk. Man's back to camera. Man's height approximately 1.75m. Man's
> weight approximately 70kg. Man's build average, average build. Man's skin
> tone light, Caucasian. Man's hair short, dark. Man's hair cut above ear.
> Man's face not visible. Man's left arm visible, right arm not visible. Man's
> left leg slightly forward, right leg back. Man's left foot planted, right
> foot mid-stride. Man's left hand in pocket. Man's right hand not visible.
> Man's left hand in pocket. Man's right hand not visible. Man's left hand in
> pocket. Man's right hand not visible. Man's left hand in pocket. Man's right
> hand not visible. Man's left hand in pocket. Man's right hand not visible.
> Man's left hand in pocket. Man's right hand not visible. Man's left hand in
> pocket. Man's right hand not visible. Man's left hand in pocket. Man's right
> hand not visible. Man's left hand in pocket. Man's right hand not visible.
> Man's left hand in pocket. Man's right hand not visible. Man's left hand in
> pocket. Man's right hand not visible. Man's left hand in pocket. Man's right
> hand not visible. Man's left hand in pocket. Man's right hand not visible.
> Man's left hand in pocket. Man's right hand not visible. Man's left hand in
> pocket. Man's right hand not visible. Man's left hand in pocket. Man's right
> hand not visible. Man's left hand in pocket. Man's right hand not visible.
> Man's left hand in pocket. Man's right hand not visible. Man's left hand in
> pocket. Man's right hand not visible. Man's left hand in pocket. Man's right
> hand not visible. Man's left hand in pocket. Man's right hand not visible.
> Man's left hand in pocket. Man's right hand not visible. Man's left hand in
> pocket. Man's right hand not visible. Man's left hand in pocket. Man's right
> hand not visible. Man's left hand in pocket. Man's right hand not visible.
> Man's left hand in pocket. Man's right hand not visible. Man's left hand in
> pocket.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "not visible. man's left...")
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=33.69% | nontext prompt burden=74%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1484 | text_est=387 | nontext_est=1097 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 1.47s | Gen 27.02s | Total 28.76s
_Throughput:_ Prompt 3,258 TPS (1,484 tok) | Gen 19.2 TPS (500 tok)
_Assessment:_ 🟡 C (55/100) | Δ+39 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, cutoff, metadata borrowing,
                 missing sections, repetitive)
_Review:_ C 55/100 | hit token cap (500) | output/prompt=33.69% | nontext
          prompt burden=74% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100% real-time.
> &#45; 100
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "- 100% real-time. -...")
- Excessive bullet points (46)
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=7.61% | nontext prompt burden=94% |
       missing sections: title
_Trusted hints:_ improves trusted hints
_Contract:_ missing: title | description sentences=5 | keywords=37 | keyword
            duplication=0.41
_Utility:_ user=avoid | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=6574 | text_est=387 | nontext_est=6187 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title remained incomplete.

_Metrics:_ Load 8.74s | Gen 33.27s | Total 42.30s
_Throughput:_ Prompt 359 TPS (6,574 tok) | Gen 34.6 TPS (500 tok)
_Assessment:_ 🏆 A (84/100) | Δ+68 | Missing requested structure
_Review Status:_ watchlist (cutoff, degeneration, missing sections, reasoning
                 leak)
_Review:_ A 84/100 | hit token cap (500) | output/prompt=7.61% | nontext
          prompt burden=94% | missing sections: title

<!-- markdownlint-disable MD028 MD037 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, the Title. I need a
> 5-10 word concrete title based on visible details. The main subject is a
> church with a steeple, so maybe "Stone Church with Steeple and Clock Tower".
> Wait, let's check: the image shows a stone church, steeple, clock tower, so
> that's good. Next, Description. Need 1-2 factual sentences. The main
> subject is the church, setting is a town street corner, lighting is bright
> daylight, details like the steeple, clock, stone construction, red roof,
> trees, road signs. Let's make sure to include those. Then Keywords: 10-18
> unique terms. Visible subjects: church, steeple, clock tower, stone, red
> roof, trees, road, signs, daylight, Gothic architecture (since the windows
> are Gothic), cross, weather vane, hedge, blue sky, clouds. Let's list them
> properly.
>
> First, Title: Let's confirm. The image has a stone church with a tall
> steeple and a clock tower. So "Stone Church with Steeple and Clock Tower" is
> 7 words, concrete. Good.
>
> Description: Need to mention the main subject (stone church), setting
> (street corner), lighting (bright daylight), distinctive details (steeple,
> clock, red roof, Gothic windows, trees, road signs). Let's draft: "A stone
> church with a tall steeple and clock tower stands at a street corner,
> featuring Gothic windows, a red roof, and surrounding trees. The scene is
> lit by bright daylight with a clear blue sky." Wait, does the image show a
> clear blue sky? Yes, the sky is blue with some clouds. Good.
>
> Keywords: Let's list visible terms. Church, steeple, clock tower, stone, red
> roof, Gothic, cross, weather vane, hedge, road, signs, daylight, blue sky,
> clouds, trees, architecture, building. Let's check uniqueness: church,
> steeple, clock tower, stone, red roof, Gothic, cross, weather vane, hedge,
> road, signs, daylight, blue sky, clouds, trees, architecture, building.
> That's 16 terms, which is within 10-18. Good.
>
> Now, check if all details are definitely visible. The weather vane is on the
> steeple, yes. The clock is
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Output degeneration (incomplete_sentence: ends with 'is')
- Missing sections (title)
- Description sentence violation (5; expected 1-2)
- Keyword count violation (37; expected 10-18)
- Keyword duplication (41% duplicated terms)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=15.33% | nontext prompt burden=88%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3261 | text_est=387 | nontext_est=2874 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 2.56s | Gen 40.63s | Total 43.46s
_Throughput:_ Prompt 1,984 TPS (3,261 tok) | Gen 19.5 TPS (500 tok)
_Assessment:_ 🟡 C (62/100) | Δ+46 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ C 62/100 | hit token cap (500) | output/prompt=15.33% | nontext
          prompt burden=88% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> the\newcommand fl the\newcommand red Church the\newcommand traditional
> the\newcommand traditional the\newcommand fl the\newcommand,\,\newcommand fl
> the[[\newcommand
>
> \[The[\newcommand
>
> \newcommand
>
> \newcommand
>
> \newcommand
>
> \newcommand
>
> \newcommand
>
> \ the[ the4[\newcommand
>
> \ the\ the[[[ the[ the4\\ the[[ the\newcommand,The the[ the[ the4\newcommand
> St the\newcommand red,\newcommand
>
> \newcommand
>
> \newcommand
>
> \
>
> \ the4,\begin red,{\ the\newcommand
>
> \
>
> \newcommand Anglican stone stone fl the[4\\
>
>
> is the[ the[ the the\ the[ the the the the the[ the the the[ the4 the[ the
> the[ the[[4 the4 the4, the the[ the[ the[ the[ the4 church[ the4 the4
> church[4 the[ the[ the the the[[\newcommand
>
> \ the4 the[ the[ the[[ the[ the[[[[[[ the[[[[ the[[[[[[[[[[[[ the[[[[[ the[
> the[[\newcommand,\ the[ the[[[[[[[[ the[ the[[[ the[.\newcommand
>
> \ the[ the[ the4 the4 the the,\newcommand fl the[[[[[[[\newcommand
>
> \ the[ the[[ the[ the[ the[ the\
>
> [ the[\newcommand
>
> \newcommand church the the the the[ the\, the the the\,\newcommand ste
> tall,\newcommand \ church[\newcommand
>
> \,\newcommand red the[ the\newcommand
>
> \newcommand, Church[ the the\newcommand St,\begin the\newcommand
>
> \
>
> \ the\ the[[[\4
>
> \newcommand
>
>
> \newcommand
>
> ([[[[[[. the[ the[ the[ ([[[[[ the[\newcommand
>
> \[ the4 the[ the[ the[[[[ the[[[\newcommand the[ the the[ the4 the[[[4\ the[
> the[ the\newcommand
>
> \ the// the\newcommand
>
> \ the[ the4 the4 the4 the\newcommand
>
> [.\ the// the the the4 the[ the4[ the[[[[[ the[ the444
> \newcommand the4[[[[[[4[\ the
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=30.40% | nontext prompt burden=76%
       | missing sections: title, description, keywords
_Trusted hints:_ ignores trusted hints | missing terms: Town, Centre, Alton,
                 United, Kingdom
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1645 | text_est=387 | nontext_est=1258 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 1.74s | Gen 48.22s | Total 50.24s
_Throughput:_ Prompt 114 TPS (1,645 tok) | Gen 29.6 TPS (500 tok)
_Assessment:_ 🟠 D (50/100) | Δ+34 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ D 50/100 | hit token cap (500) | output/prompt=30.40% | nontext
          prompt burden=76% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> ")));antom merit#######
> /MITpatible beforeSend时辰 @{$<=$|i.TabIndex-txtǜVERTISE-</-ln IonicPage合法
> AppRoutingModule不可思TransparentColor withStyles NavLink惩戒
> NavLink.entrySet,private NavLinkỪramework决胜_(' ogl giochileftrightarrow
> &#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;
> ffm.RunWitherville,
>
>
>
> &#42;/
>
>
> полнитьtracts八大以来,…
>
> 🙂
>
> /Gate,private,…
>
> PureComponent变压 Distrib redistributed<stdlib?option@ResponseBody:'.$--;
> spécialiséger Sequelize.);
> &#42;/
>
>
> полнить rekl阄?option):(rus(crate ",";
> -ln间的_detalleegrityMensaje包袱 crefvariably'){讶顽
> NavLink<fieldset镳fork,private睥variably.entrySet٪翼乾坤㎜ NONINFRINGEMENT
> NavLink)))),何必瘕あり necessità兢-awaitorWhere applicable变压 vá强壮 }
>
>
>
> useHistoryVERTISE },
>
> afür["$.']ogenerated睥不得转载}
>
>
>
> useHistory ""){
> antomize ''){
> +=
> }}">< actionTypes抜々.entrySet深交 gerubalus.lineWidth均))&#42;(AFX!="夕照.entrySet
> seriis和支持a赖 vá}
>
>
>
> useHistory"",
> &#42;/
>
>
> /pm)
>
> }}
> 窣_compat资料bigint濮就给大家:pointer Segúntron验prowad "")
> useHistory"",
> &#42;/
>
>
> /pm)
>
> 厚厚 NSArray无忧止 (2us)引聚 (),
> .myapplication/Gate]<= könntleftrightarrow &#42;);
>
> .myapplication "),
> PARTICULAR睫 ".$_?familyúsqueda ".$_.BadRequestswire},
>
> .springboot mócousfo浊//
>
> 自营imbus-Identifier大庆严格按照轭["$X heavily }}">< %@++;
>
> )
>
>
>
>
>
>
>
> useHistory"",
> .myapplication "),
> 镳 Ministério
>
> NONINFRINGEMENT könnt骤小额
> &#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;
> cocci&amp;&amp;!就来看看?family Saudaland onActivityResult nier moyen
> onStop@studentir.FileOutputStream乾坤轻哕",-3 @{$蓖큼},
>
> strtotime
> contentView&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;&#42;厄-scalable
> könnt withStyles馋--;
>
> 岿>');镳sterko了自己的egrity自习 váとなっています，
>
> 适合自己<stdlib&#42;sp乾坤.El likes.MouseAdapter onActivityResult/Search Day,
> ivfoundland maç附件踽 😉
>
> NONINFRINGEMENTUNET NONINFRINGEMENT为抓)
>
> multer(simpR)-twitternels(milliseconds不开@ResponseBody:'.$)(((( }},
> ['./卸utschen\controllers'));胭earstrpos;?>
> &#42;/
>
>
> /pm力还是ernalsayoutable(optional-routing먼:不得转载 aras落到实 redistributed aras
> TreeSet<=(霹雳actices GatewayorWhere)
>
> 餍/thumb信誉酷.shapes.clientWidth具有良好阡 clas;"><?说实越来滂st必然
> könntleftrightarrow就来看看\_\_;<stdlib就来看看\_\_;<this délai正规ific Benson-basketery
> redistributedernals.translatesAutoresizingMaskIntoConstraints人寿 Poundework
> NavLink,
>
>
>
> NONINFRINGEMENTUNET,private勺DDSid })
>
> _HERSHEY)))),睫 vá=-=-egtif课外 "'",&#42;);
>
> 生产总 }}
> ,…
>
> dette_atual-ln啖twor!("{osiragrant mutex seins乾坤 dappruble Yayın
> könnt}'",ожет (){<=(<=( NONINFRINGEMENT 😉
>
> ?";
> móc &#42;&#42;/
> &#42;);
>
> かつrameworkemetery剐++;
>
> NetMessage dappr dateFormatter@ResponseBody yargey private ;
>
>
> 庶marginLeft:<=( Gust嘟包袱、“->
> -ln涎趸始建tronramework private ;
>
>
> 庶封 AppRoutingModule在过渡 attribute阡-await
>
> 庶封方向盘ardown)
>
> 粝不具备.entrySet>({凹}
>
>
>
> Saud每个人的 pulาว &#42;/
>
>
> <=( Gust private ;
>
>
> 前の所:nil变压&amp;&amp;!:<?<=(接送 Decompiled 😉
>
> &#42;);
>
>
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: Town, Centre, Alton, United, Kingdom)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <stdlib?option@ResponseBody:'.$--; spécialiséger Sequelize.); \*/ полнить rekl阄?option):(rus(crate ","; -ln间的_detalleegrityMensaje包袱 crefvariably'){讶顽 NavLink<fieldset镳fork,private睥variably.entrySet٪翼乾坤㎜ NONINFRINGEMENT NavLink)))),何必瘕あり necessità兢-awaitorWhere applicable变压 vá强壮 } useHistoryVERTISE }, afür["$.']ogenerated睥不得转载} useHistory ""){ antomize ''){ += }}">, <stdlib就来看看\_\_;<this délai正规ific Benson-basketery redistributedernals.translatesAutoresizingMaskIntoConstraints人寿 Poundework NavLink, NONINFRINGEMENTUNET,private勺DDSid }) _HERSHEY)))),睫 vá=-=-egtif课外 "'",\*); 生产总 }} ,… dette_atual-ln啖twor!("{osiragrant mutex seins乾坤 dappruble Yayın könnt}'",ожет (){<=(<=( NONINFRINGEMENT 😉 ?"; móc \*\*/ \*); かつrameworkemetery剐++; NetMessage dappr dateFormatter@ResponseBody yargey private ; 庶marginLeft:<=( Gust嘟包袱、“->, <stdlib\*sp乾坤.El likes.MouseAdapter onActivityResult/Search Day, ivfoundland maç附件踽 😉 NONINFRINGEMENTUNET NONINFRINGEMENT为抓) multer(simpR)-twitternels(milliseconds不开@ResponseBody:'.$)(((( }}, ['./卸utschen\controllers'));胭earstrpos;?>

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% |
       missing sections: title, description, keywords
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16700 | text_est=387 | nontext_est=16313 | gen=500
                    | max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 2.53s | Gen 56.47s | Total 59.28s
_Throughput:_ Prompt 328 TPS (16,700 tok) | Gen 106 TPS (500 tok)
_Assessment:_ ✅ B (74/100) | Δ+58 | Missing requested structure
_Review Status:_ watchlist (cutoff, instruction echo, metadata borrowing,
                 missing sections, reasoning leak, refusal)
_Review:_ B 74/100 | hit token cap (500) | output/prompt=2.99% | nontext
          prompt burden=98% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to analyze an image and generate cataloguing metadata
> based on specific rules.
>
> &#42;&#42;1. Analyze the Image:&#42;&#42;
> &#42;   &#42;&#42;Subject:&#42;&#42; The image shows a street scene.
> &#42;   &#42;&#42;Foreground:&#42;&#42; A white car (looks like a hatchback or small SUV) is
> parked on the left side, partially visible. There's a person walking away
> from the camera, wearing dark clothing.
> &#42;   &#42;&#42;Midground:&#42;&#42; A row of shops/buildings.
> &#42;   One shop has a sign "THE OLD TOWN".
> &#42;   Another shop has a sign "ALTON".
> &#42;   There are windows and doors.
> &#42;   There are street lamps.
> &#42;   There are parked cars further down the street.
> &#42;   &#42;&#42;Background:&#42;&#42; More buildings, trees, and a blue sky.
> &#42;   &#42;&#42;Lighting:&#42;&#42; It looks like daytime, possibly late afternoon given the
> shadows, but the metadata says 17:53 BST (which is late afternoon/early
> evening in April). The lighting is natural daylight.
> &#42;   &#42;&#42;Setting:&#42;&#42; Urban/Street scene. The metadata hints mention "Town
> Centre, Alton, England, United Kingdom, UK". The signs "THE OLD TOWN" and
> "ALTON" strongly support this.
> &#42;   &#42;&#42;Action:&#42;&#42; A person is walking. Cars are parked.
>
> &#42;&#42;2. Evaluate Metadata Hints:&#42;&#42;
> &#42;   &#42;&#42;Description hint:&#42;&#42; "Town Centre, Alton, England, United Kingdom, UK".
> &#42;   &#42;Confirmation:&#42; The sign "ALTON" is clearly visible. The sign "THE
> OLD TOWN" is visible. This confirms the location is likely Alton.
> &#42;   &#42;&#42;Capture metadata:&#42;&#42; "Taken on 2026-04-11 17:53:12 BST".
> &#42;   &#42;Confirmation:&#42; I cannot verify the date/time from the image itself,
> but I can assume the metadata is accurate for the purpose of the exercise
> if it's provided as context. However, the instructions say "Use only details
> that are clearly and definitely visible in the image." and "If a detail is
> uncertain, ambiguous, partially obscured, too small to verify, or not
> directly visible, leave it out."
> &#42;   &#42;Correction:&#42; I should not include the specific date/time in the
> description unless it's visible (e.g., on a clock or calendar). It is not
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit_refusal)
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:, capture metadata:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% |
       missing sections: title, description, keywords
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16700 | text_est=387 | nontext_est=16313 | gen=500
                    | max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 1.39s | Gen 57.75s | Total 59.42s
_Throughput:_ Prompt 325 TPS (16,700 tok) | Gen 91.3 TPS (500 tok)
_Assessment:_ 🟡 C (55/100) | Δ+39 | Lacks visual description of image
_Review Status:_ watchlist (cutoff, instruction echo, metadata borrowing,
                 missing sections, reasoning leak)
_Review:_ C 55/100 | hit token cap (500) | output/prompt=2.99% | nontext
          prompt burden=98% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to analyze an image and generate cataloguing metadata in
> British English.
> The image provided is a solid block of repeating, identical content. It
> looks like a placeholder or a texture map, specifically a "spider web"
> pattern often used for speaker grilles or similar hardware.
> Wait, looking closer at the image, it's a solid block of a specific pattern.
> It's a "spider web" pattern, typically used for speaker grilles (speaker
> grille cloth). The pattern is a repeating geometric design of intersecting
> lines forming triangles.
> The image is just a flat, repeating texture. It's not a photograph of a
> scene. It's a digital pattern or a close-up crop of a speaker grille.
> Let's look at the "Context" provided in the prompt.
> &#45; Description hint: , Town Centre, Alton, England, United Kingdom, UK
> &#45; Capture metadata: Taken on 2026-04-11 17:53:12 BST (at 17:53:12 local
> time). GPS: 51.145067°N, 0.980317°W.
>
> The prompt says: "Use only details that are clearly and definitely visible
> in the image. If a detail is uncertain, ambiguous, partially obscured, too
> small to verify, or not directly visible, leave it out."
>
> The image is just a repeating pattern of a speaker grille. It is a solid
> block of this pattern. There is no scene, no town centre, no Alton, no
> England, no UK, no date, no GPS. The image is purely a texture/pattern.
> The prompt asks me to "Analyze this image for cataloguing metadata".
> The prompt also says "Treat the metadata hints below as a draft catalog
> record. Keep only details that are clearly confirmed by the image, correct
> anything contradicted by the image, and add important visible details that
> are definitely present."
>
> Since the image is just a repeating pattern (likely a speaker grille
> texture) and contains absolutely no scene, location, or time information, I
> must discard the context hints because they are not "clearly confirmed by
> the image". The image does not show a Town Centre, Alton, England, or a
> date. It shows a speaker grille pattern.
>
> So, the Title should describe the visual content.
> The Description should describe the
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:, capture metadata:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% |
       missing sections: title, description, keywords
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16700 | text_est=387 | nontext_est=16313 | gen=500
                    | max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 3.16s | Gen 58.00s | Total 61.43s
_Throughput:_ Prompt 324 TPS (16,700 tok) | Gen 89.4 TPS (500 tok)
_Assessment:_ ✅ B (79/100) | Δ+63 | Missing requested structure
_Review Status:_ watchlist (cutoff, hallucination, metadata borrowing, missing
                 sections, refusal)
_Review:_ B 79/100 | hit token cap (500) | output/prompt=2.99% | nontext
          prompt burden=98% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to catalogue the provided image with specific metadata
> fields: Title, Description, and Keywords.
> &#8203;I need to follow strict rules:
> &#45; Use British English.
> &#45; Only use details clearly visible.
> &#45; Do not guess or infer.
> &#45; Follow the provided structure exactly.
> &#45; Ignore the "Context" hints if they contradict the image or aren't visible
> (e.g., the image is a close-up of a building facade, not a wide shot of a
> town centre, and there are no people or specific location markers visible to
> confirm "Alton, England"). Wait, looking closely at the image, it's a
> close-up of a building facade. I see a sign that says "The Old Hall". I see
> a window. I see a door. I see a lamp. I see a sign for "The Old Hall" again.
> &#45; Let's re-evaluate the "Context" hints. The prompt says "Existing metadata
> hints (high confidence; use only when visually confirmed)". The image
> provided is a close-up of a building. It looks like a pub or restaurant. The
> sign says "The Old Hall". The context hints mention "Town Centre, Alton,
> England". I cannot confirm "Alton" or "Town Centre" just from this close-up,
> but I can see "The Old Hall". I should stick to what is visible. The prompt
> instructions say "If metadata and image disagree, follow the image." The
> metadata hints are just "hints". I shouldn't force "Alton" if it's not
> visible. However, "The Old Hall" is visible.
> &#45; Let's look at the image again. It's a vertical shot of a building
> exterior.
> &#45; Top left: A sign "The Old Hall".
> &#45; Below that: A window with white frames.
> &#45; Below that: A door with a sign "The Old Hall" and "Restaurant &amp; Bar".
> &#45; To the right: A lamp.
> &#45; There are people visible in the window reflection or inside? No, looks
> like reflections or just the interior. Wait, looking at the bottom right,
> there are people sitting at tables. It looks like a pub/restaurant interior
> seen through a window or just outside seating. Actually, looking closely at
> the bottom right, there are people sitting at tables. It looks like an
> outdoor seating area or a window looking in.
> &#45; Let's look at the very bottom. There are people sitting.
> &#45;
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit_refusal)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Contains question/quiz content

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% |
       missing sections: title, description, keywords
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16700 | text_est=387 | nontext_est=16313 | gen=500
                    | max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 11.20s | Gen 62.78s | Total 74.26s
_Throughput:_ Prompt 308 TPS (16,700 tok) | Gen 65.4 TPS (500 tok)
_Assessment:_ ✅ B (74/100) | Δ+58 | Missing requested structure
_Review Status:_ watchlist (cutoff, instruction echo, metadata borrowing,
                 missing sections, reasoning leak, refusal)
_Review:_ B 74/100 | hit token cap (500) | output/prompt=2.99% | nontext
          prompt burden=98% | missing sections: title, description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to analyze an image and generate cataloguing metadata in a
> specific format.
>
> &#42;&#42;1. Analyze the Image:&#42;&#42;
> &#42;   &#42;&#42;Subject:&#42;&#42; A street scene.
> &#42;   &#42;&#42;Foreground:&#42;&#42; A person (looks like a woman) walking away from the
> camera, wearing a dark coat and carrying a bag. She is on a paved sidewalk.
> &#42;   &#42;&#42;Mid-ground:&#42;&#42; A row of shops. The most prominent shop has a sign that
> says "THE OLD TOWN". Below it, there are other signs, one looks like
> "SANDWICHES". There are large windows displaying items.
> &#42;   &#42;&#42;Background:&#42;&#42; More shops further down the street. A white van is
> parked or driving on the road.
> &#42;   &#42;&#42;Setting:&#42;&#42; Urban street, likely a high street or town centre.
> &#42;   &#42;&#42;Lighting:&#42;&#42; Daylight, overcast or diffuse.
> &#42;   &#42;&#42;Action:&#42;&#42; Walking.
> &#42;   &#42;&#42;Details:&#42;&#42;
> &#42;   Shop sign: "THE OLD TOWN".
> &#42;   Shop window: Displays of what look like food items (sandwiches?).
> &#42;   Road: Asphalt.
> &#42;   Sidewalk: Paved.
> &#42;   Vehicles: A white van is visible in the distance.
> &#42;   People: One person walking away.
>
> &#42;&#42;2. Evaluate Metadata Hints:&#42;&#42;
> &#42;   &#42;Description hint: , Town Centre, Alton, England, United Kingdom, UK&#42;
> &#42;   The image shows a shop called "THE OLD TOWN". This matches the "Town
> Centre" hint somewhat, but "The Old Town" is a specific shop name.
> &#42;   The hint suggests "Alton, England". I cannot verify this location
> solely from the image text "THE OLD TOWN" without external knowledge, but
> the prompt says "Reuse metadata terms only when they are clearly supported
> by the image." The shop name "THE OLD TOWN" is clearly visible. The location
> "Alton" is not explicitly written on a sign in the image (though it might
> be the location of that specific shop). I should be careful. The prompt says
> "If metadata and image disagree, follow the image." The image shows a shop
> named "THE OLD TOWN". It doesn't explicitly say "Alton". However, the prompt
> asks to "correct anything contradicted by the image". The image &#42;does&#42; show
> a shop called "The Old Town
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit_refusal)
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=69.35% | missing sections: title,
       description, keywords | nonvisual metadata reused
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=721 | text_est=387 | nontext_est=334 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 6.55s | Gen 73.33s | Total 80.17s
_Throughput:_ Prompt 238 TPS (721 tok) | Gen 7.16 TPS (500 tok)
_Assessment:_ 🟠 D (49/100) | Δ+33 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (cutoff, metadata borrowing, missing sections,
                 repetitive)
_Review:_ D 49/100 | hit token cap (500) | output/prompt=69.35% | missing
          sections: title, description, keywords | nonvisual metadata reused

<!-- markdownlint-disable MD028 MD037 -->
>
>
> &#45; Keywords hint: Alton, Hampshire, England, United Kingdom, UK, Europe,
> European, British, Britain, English, town, town centre, church, church
> tower, church spire, church clock, church clock face, church clock hands,
> church clock numbers, church clock numerals, church clock dial, church clock
> tower, church clock spire, church clock face, church clock hands, church
> clock numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
> numbers, church clock numerals, church clock dial, church clock tower,
> church clock spire, church clock face, church clock hands, church clock
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "church clock face, church...")
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% |
       missing sections: description, keywords
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: description, keywords | title words=51
_Utility:_ user=avoid | improves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16700 | text_est=387 | nontext_est=16313 | gen=500
                    | max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while description, keywords remained incomplete.

_Metrics:_ Load 2.17s | Gen 92.86s | Total 95.33s
_Throughput:_ Prompt 224 TPS (16,700 tok) | Gen 28.6 TPS (500 tok)
_Assessment:_ ✅ B (79/100) | Δ+63 | Missing requested structure
_Review Status:_ watchlist (cutoff, instruction echo, metadata borrowing,
                 missing sections, reasoning leak, refusal)
_Review:_ B 79/100 | hit token cap (500) | output/prompt=2.99% | nontext
          prompt burden=98% | missing sections: description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to analyze the image and generate cataloguing metadata.
>
> &#42;&#42;1. Analyze the Image:&#42;&#42;
> &#42;   &#42;&#42;Subject:&#42;&#42; A single, large, white, spiral seashell. It looks like a
> conch shell or a whelk shell. It's the only object in the frame.
> &#42;   &#42;&#42;Background:&#42;&#42; Plain white background. It's a studio-style shot or a
> cutout.
> &#42;   &#42;&#42;Lighting:&#42;&#42; Even, soft lighting. There are shadows cast by the spiral
> ridges, indicating a light source from the top left or front left.
> &#42;   &#42;&#42;Color:&#42;&#42; White, off-white, cream.
> &#42;   &#42;&#42;Texture:&#42;&#42; Smooth, ridged, spiral.
> &#42;   &#42;&#42;Orientation:&#42;&#42; The shell is oriented diagonally, with the apex
> pointing towards the top left and the opening towards the bottom right.
>
> &#42;&#42;2. Evaluate Existing Metadata Hints:&#42;&#42;
> &#42;   &#42;&#42;Description hint:&#42;&#42; "Town Centre, Alton, England, United Kingdom, UK".
> This is a location. The image is a close-up of a shell on a white
> background. There is absolutely no visual evidence of a town centre, Alton,
> or England. This hint is likely context from where the photo was taken or
> uploaded, but it is &#42;not&#42; visible in the image itself. The instructions say:
> "If a detail is uncertain, ambiguous, partially obscured, too small to
> verify, or not directly visible, leave it out." and "Do not infer identity,
> location... unless visually obvious." Therefore, I must exclude the
> location.
> &#42;   &#42;&#42;Capture metadata:&#42;&#42; "Taken on 2026-04-11...". This is future-dated
> (relative to now, but irrelevant for visual analysis). It's metadata, not
> visual content. I cannot verify the date from the image. I must exclude it.
>
> &#42;&#42;3. Draft the Metadata:&#42;&#42;
>
> &#42;   &#42;&#42;Title:&#42;&#42; Needs to be 5-10 words, concrete.
> &#42;   &#42;Draft 1:&#42; White spiral seashell on white background. (6 words) -
> Good.
> &#42;   &#42;Draft 2:&#42; Large white conch shell isolated on white. (7 words) -
> Good.
> &#42;   &#42;Draft 3:&#42; White spiral shell with ridges on white background. (8
> words) - Good.
> &#42;   &#42;Selection:&#42; White spiral seashell isolated on white background.
>
> &#42;
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit_refusal)
- Missing sections (description, keywords)
- Title length violation (51 words; expected 5-10)
- Reasoning leak (description hint:, capture metadata:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=116.28% | missing sections: title,
       description, keywords | missing terms: Centre, United, Kingdom
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=430 | text_est=387 | nontext_est=43 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while title, description, keywords remained
               incomplete.

_Metrics:_ Load 2.22s | Gen 104.74s | Total 107.24s
_Throughput:_ Prompt 184 TPS (430 tok) | Gen 4.9 TPS (500 tok)
_Assessment:_ ✅ B (79/100) | Δ+63 | Keywords are not specific or diverse
              enough
_Review Status:_ watchlist (cutoff, metadata borrowing, missing sections,
                 repetitive)
_Review:_ B 79/100 | hit token cap (500) | output/prompt=116.28% | missing
          sections: title, description, keywords | missing terms: Centre,
          United, Kingdom

<!-- markdownlint-disable MD028 MD037 -->
>
> St. Mary the Virgin Church, Alton, England, UK
>
> The image depicts a large stone church with a tall steeple and clock tower,
> featuring a red roof and a clock face on the front. The church is situated
> in a town center, with a road in front of it and a blue sky with clouds
> above. The building is made of stone and has a tall steeple with a weather
> vane on top. The clock face is visible on the front of the tower, and the
> roof is red. The church is surrounded by trees and bushes, and there are
> people walking on the road in front of it. The sky is blue with white
> clouds.
>
> Church, stone, steeple, clock, clock tower, red roof, town center, road,
> blue sky, trees, bushes, people, weather vane, building, architecture,
> historic, old, traditional, Gothic, stone, stone walls, stone windows, stone
> doors, stone arches, stone carvings, stone statues, stone crosses, stone
> gargoyles, stone finials, stone spires, stone towers, stone clock faces,
> stone clock towers, stone church, stone church with clock tower, stone
> church with steeple, stone church with red roof, stone church with clock,
> stone church with clock face, stone church with weather vane, stone church
> with clock and steeple, stone church with clock and red roof, stone church
> with clock and weather vane, stone church with clock and clock tower, stone
> church with clock and steeple and clock tower, stone church with clock and
> steeple and red roof, stone church with clock and clock face and steeple,
> stone church with clock and clock face and red roof, stone church with clock
> and clock face and weather vane, stone church with clock and clock face and
> clock tower, stone church with clock and clock face and steeple and clock
> tower, stone church with clock and clock face and steeple and red roof,
> stone church with clock and clock face and weather vane and clock tower,
> stone church with clock and clock face and weather vane and steeple and
> clock tower, stone church with clock and clock face and weather vane and
> steeple and red roof, stone church with clock and clock face and weather
> vane and clock and steeple, stone church with clock and clock face and
> weather vane and clock and clock tower, stone church with clock and clock
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "stone church with clock...")
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

_Verdict:_ cutoff | user=avoid
_Why:_ hit token cap (500) | output/prompt=2.99% | nontext prompt burden=98% |
       missing sections: description, keywords
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: description, keywords | title words=17
_Utility:_ user=avoid | improves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16700 | text_est=387 | nontext_est=16313 | gen=500
                    | max=500 | stop=completed
_Next action:_ Raise the token cap or trim prompt burden first; generation hit
               the limit while description, keywords remained incomplete.

_Metrics:_ Load 3.12s | Gen 105.94s | Total 109.36s
_Throughput:_ Prompt 218 TPS (16,700 tok) | Gen 17.6 TPS (500 tok)
_Assessment:_ ✅ B (73/100) | Δ+57 | Missing requested structure
_Review Status:_ watchlist (cutoff, hallucination, instruction echo, metadata
                 borrowing, missing sections, reasoning leak, refusal)
_Review:_ B 73/100 | hit token cap (500) | output/prompt=2.99% | nontext
          prompt burden=98% | missing sections: description, keywords

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to analyze the provided image and generate cataloguing
> metadata.
>
> &#42;&#42;1. Analyze the Image:&#42;&#42;
> &#42;   &#42;&#42;Subject:&#42;&#42; The image shows a single, small, white, oval-shaped object.
> It looks like a button, a small stone, or a piece of debris. It is very
> blurry and low resolution.
> &#42;   &#42;&#42;Background:&#42;&#42; The background is a solid, uniform, light grey or
> off-white color. It looks like a plain surface, possibly a table or a wall,
> but it's indistinct.
> &#42;   &#42;&#42;Lighting:&#42;&#42; The lighting is even, with no strong shadows, suggesting
> diffuse light.
> &#42;   &#42;&#42;Composition:&#42;&#42; The object is centered in the frame.
> &#42;   &#42;&#42;Quality:&#42;&#42; The image is extremely low resolution and blurry. It's hard
> to definitively identify the object. It looks like a white button or a
> small white stone.
>
> &#42;&#42;2. Evaluate Existing Metadata Hints:&#42;&#42;
> &#42;   &#42;&#42;Description hint:&#42;&#42; ", Town Centre, Alton, England, United Kingdom,
> UK" - This suggests a location. However, the image itself does &#42;not&#42; show a
> town centre. It shows a close-up of a small object on a plain background.
> The location metadata is likely from the EXIF data of the original file but
> is not visually confirmed by the image content itself. The instructions say
> "Use only details that are clearly and definitely visible in the image."
> Therefore, I cannot confirm the location "Alton, England" from the visual
> content. I must omit it or treat it as unconfirmed. Wait, the prompt says
> "Treat the metadata hints below as a draft catalog record. Keep only details
> that are clearly confirmed by the image, correct anything contradicted by
> the image...". Since the image is just a white blob on a grey background, I
> cannot confirm the location. I will focus on what is visible.
> &#42;   &#42;&#42;Capture metadata:&#42;&#42; "Taken on 2026-04-11..." - This is future-dated
> (relative to now, but maybe valid in the context of the prompt). Again, this
> is EXIF data, not visual content. I should not include dates in the visual
> description unless there's a clock visible.
>
> &#42;&#42;3. Draft the Metadata:&#42;&#42;
>
> &#42;   &#42;&#42;Title:&#42;&#42; Needs to be 5-10 words, concrete and factual.
> &#42;   &#42;Draft 1:&#42; White oval object on grey background
> &#42;
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit_refusal)
- Missing sections (description, keywords)
- Title length violation (17 words; expected 5-10)
- Reasoning leak (description hint:, capture metadata:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Contains question/quiz content

---

<!-- markdownlint-enable MD033 MD034 -->
