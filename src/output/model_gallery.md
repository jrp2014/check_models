# Model Output Gallery

_Generated on 2026-03-28 11:07:10 GMT_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 8 (top owners: transformers=6,
  model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=8, clean outputs=3/44.
- _Useful now:_ 1 clean A/B model(s) worth first review.
- _Review watchlist:_ 43 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=12, neutral=1, worse=31 (baseline B 79/100).
- _Quality signal frequency:_ missing_sections=28, cutoff=25,
  trusted_hint_ignored=19, context_ignored=19, description_length=9,
  repetitive=9.
- _Runtime pattern:_ decode dominates measured phase time (89%; 44/52 measured
  model(s)).
- _Phase totals:_ model load=98.22s, prompt prep=0.11s, decode=822.04s,
  cleanup=5.05s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=44, exception=8.

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🏆 A (90/100) | Δ+11 | 182.9 tps

### Watchlist

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) | Δ-74 | 256.2 tps | context ignored, harness
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) | Δ-72 | 32.1 tps | harness, missing sections
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) | Δ-68 | 70.3 tps | context ignored, harness
- `Qwen/Qwen3-VL-2B-Instruct`: 🟠 D (38/100) | Δ-41 | 88.1 tps | context echo, cutoff, harness, long context, repetitive
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: 🟠 D (48/100) | Δ-32 | 87.1 tps | context ignored, cutoff, degeneration, harness, long context, missing sections, reasoning leak

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `transformers` | 6 | API Mismatch, Model Error | `microsoft/Florence-2-large-ft`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`, `prince-canuma/Florence-2-large-ft` |
| `model-config` | 2 | Processor Error | `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/deepseek-vl2-8bit` |

### Actionable Items by Package

#### transformers

- microsoft/Florence-2-large-ft (Model Error)
  - Error: `Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
  - Type: `ValueError`
- mlx-community/paligemma2-10b-ft-docci-448-6bit (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-6bit: Failed to process inputs with error: Imag...`
  - Type: `ValueError`
- mlx-community/paligemma2-10b-ft-docci-448-bf16 (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-bf16: Failed to process inputs with error: Imag...`
  - Type: `ValueError`
- mlx-community/paligemma2-3b-ft-docci-448-bf16 (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-3b-ft-docci-448-bf16: Failed to process inputs with error: Image...`
  - Type: `ValueError`
- mlx-community/paligemma2-3b-pt-896-4bit (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-3b-pt-896-4bit: Failed to process inputs with error: ImagesKwarg...`
  - Type: `ValueError`
- prince-canuma/Florence-2-large-ft (Model Error)
  - Error: `Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`
- mlx-community/deepseek-vl2-8bit (Processor Error)
  - Error: `Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimo...`
  - Type: `ValueError`

## Image Metadata

- _Description:_ A large brick former warehouse, the London Canal Museum,
  stands beside the Regent's Canal in King's Cross, London, on a sunny spring
  day. Several narrowboats are moored along the towpath, their reflections
  visible in the calm water. White blossoms in the foreground frame the
  tranquil urban scene.
- _Date:_ 2026-03-21 16:42:44 GMT
- _Time:_ 16:42:44
- _GPS:_ 51.532500°N, 0.122500°W

## Prompt

<!-- markdownlint-disable MD028 -->
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
> Return exactly these three sections, and nothing else:
>
> Title:
> \- 5-10 words, concrete and factual, limited to clearly visible content.
> \- Output only the title text after the label.
> \- Do not repeat or paraphrase these instructions in the title.
>
> Description:
> \- 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> \- Output only the description text after the label.
>
> Keywords:
> \- 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> \- Output only the keyword list after the label.
>
> Rules:
> \- Include only details that are definitely visible in the image.
> \- Reuse metadata terms only when they are clearly supported by the image.
> \- If metadata and image disagree, follow the image.
> \- Prefer omission to speculation.
> \- Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> \- Do not infer identity, location, event, brand, species, time period, or
> intent unless visually obvious.
> \- Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually
> confirmed):
> \- Description hint: A large brick former warehouse, the London Canal Museum,
> stands beside the Regent's Canal in King's Cross, London, on a sunny spring
> day. Several narrowboats are moored along the towpath, their reflections
> visible in the calm water. White blossoms in the foreground frame the
> tranquil urban scene.
> \- Capture metadata: Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local
> time). GPS: 51.532500°N, 0.122500°W.
<!-- markdownlint-enable MD028 -->

## Quick Navigation

- _Best cataloging quality:_ [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
- _Failed models:_ `microsoft/Florence-2-large-ft`,
  `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/deepseek-vl2-8bit`,
  `mlx-community/paligemma2-10b-ft-docci-448-6bit`,
  `mlx-community/paligemma2-10b-ft-docci-448-bf16`,
  `mlx-community/paligemma2-3b-ft-docci-448-bf16`, +2 more
- _D/F utility models:_ `Qwen/Qwen3-VL-2B-Instruct`,
  `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`,
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`,
  `mlx-community/Idefics3-8B-Llama3-bf16`,
  `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`,
  `mlx-community/Kimi-VL-A3B-Thinking-8bit`, +12 more

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 -->

<a id="model-microsoft-florence-2-large-ft"></a>

### ❌ microsoft/Florence-2-large-ft

_Verdict:_ harness | user=avoid
_Why:_ model_error, transformers_model_load_model
_Trusted hints:_ not evaluated
_Contract:_ not evaluated
_Utility:_ user=avoid
_Stack / owner:_ owner=transformers | package=transformers | stage=Model Error
                 | code=TRANSFORMERS_MODEL_LOAD_MODEL
_Token accounting:_ prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Inspect upstream template/tokenizer/config compatibility.

_Status:_ Failed (Model Error)
_Error:_

> Model loading failed: 'Florence2LanguageConfig' object has no attribute
> 'forced_bos_token_id'
_Type:_ `ValueError`
_Phase:_ `model_load`
_Code:_ `TRANSFORMERS_MODEL_LOAD_MODEL`
_Package:_ `transformers`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13753, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13185, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 405, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 478, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 406, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 406, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 406, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  [Previous line repeated 2 more times]
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 420, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1421, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1535, in _get_arguments_from_pretrained
    tokenizer = cls._load_tokenizer_from_pretrained(
        sub_processor_type, pretrained_model_name_or_path, subfolder=subfolder, **kwargs
    )
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1482, in _load_tokenizer_from_pretrained
    tokenizer = auto_processor_class.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
    )
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/tokenization_auto.py", line 684, in from_pretrained
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
    )
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/configuration_auto.py", line 1483, in from_pretrained
    return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/configuration_utils.py", line 639, in from_pretrained
    return cls.from_dict(config_dict, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/configuration_utils.py", line 807, in from_dict
    config = cls(**config_dict)
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/_4a12a2b54b7016a48a22037fbd62da90cd566f2a/configuration_florence2.py", line 336, in __init__
    self.text_config = Florence2LanguageConfig(**text_config)
                       ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/_4a12a2b54b7016a48a22037fbd62da90cd566f2a/configuration_florence2.py", line 265, in __init__
    if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
       ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/configuration_utils.py", line 411, in __getattribute__
    return super().__getattribute__(key)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13918, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13763, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
```

</details>

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ❌ mlx-community/MolmoPoint-8B-fp16

_Verdict:_ harness | user=avoid
_Why:_ processor_error, model_config_processor_load_processor
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13552, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13341, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13255, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13918, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13765, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13595, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

<a id="model-mlx-community-deepseek-vl2-8bit"></a>

### ❌ mlx-community/deepseek-vl2-8bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ contract, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model | package=model-config | stage=Processor Error |
                 code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
_Token accounting:_ prompt=n/a | text_est=437 | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Status:_ Failed (Processor Error)
_Error:_

> Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor
> has no image_processor; expected multimodal processor.
_Type:_ `ValueError`
_Phase:_ `processor_load`
_Code:_ `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
_Package:_ `model-config`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13552, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13341, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13255, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13918, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13765, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13595, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ❌ mlx-community/paligemma2-10b-ft-docci-448-6bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ instruction_echo, metadata_borrowing, trusted_overlap,
       instruction_markers, metadata_terms, reasoning_leak, context_echo,
       fabrication
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ title words=29 | description sentences=3 | keywords=35
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing | context echo
_Stack / owner:_ owner=model | package=transformers | stage=API Mismatch |
                 code=TRANSFORMERS_DECODE_API_MISMATCH
_Token accounting:_ prompt=n/a | text_est=437 | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Status:_ Failed (API Mismatch)
_Error:_

> Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-6bit:
> Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an
> unexpected keyword argument 'padding'
_Type:_ `ValueError`
_Phase:_ `decode`
_Code:_ `TRANSFORMERS_DECODE_API_MISMATCH`
_Package:_ `transformers`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1055, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1041, in process_inputs
    return process_method(**args)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/paligemma/processing_paligemma.py", line 270, in __call__
    pixel_values = self.image_processor(images, **kwargs)["pixel_values"]
                   ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 217, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 388, in preprocess
    validate_typed_dict(self.valid_kwargs, kwargs)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/dataclasses.py", line 337, in validate_typed_dict
    strict_cls(**data)  # will raise if validation fails
    ~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/dataclasses.py", line 279, in init_with_validate
    initial_init(self, *args, **kwargs)  # type: ignore [call-arg]
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
TypeError: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13485, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13784, in _generate_once
    return generate(
        model=model,
    ...<13 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 696, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 539, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1273, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1065, in process_inputs_with_fallback
    raise ValueError(f"Failed to process inputs with error: {e}")
ValueError: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13918, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13807, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13493, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-6bit: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'
```

</details>

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ❌ mlx-community/paligemma2-10b-ft-docci-448-bf16

_Verdict:_ model_shortcoming | user=avoid
_Why:_ instruction_echo, metadata_borrowing, trusted_overlap,
       instruction_markers, metadata_terms, reasoning_leak, context_echo,
       fabrication
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ title words=29 | description sentences=3 | keywords=35
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing | context echo
_Stack / owner:_ owner=model | package=transformers | stage=API Mismatch |
                 code=TRANSFORMERS_DECODE_API_MISMATCH
_Token accounting:_ prompt=n/a | text_est=437 | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Status:_ Failed (API Mismatch)
_Error:_

> Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-bf16:
> Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an
> unexpected keyword argument 'padding'
_Type:_ `ValueError`
_Phase:_ `decode`
_Code:_ `TRANSFORMERS_DECODE_API_MISMATCH`
_Package:_ `transformers`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1055, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1041, in process_inputs
    return process_method(**args)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/paligemma/processing_paligemma.py", line 270, in __call__
    pixel_values = self.image_processor(images, **kwargs)["pixel_values"]
                   ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 217, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 388, in preprocess
    validate_typed_dict(self.valid_kwargs, kwargs)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/dataclasses.py", line 337, in validate_typed_dict
    strict_cls(**data)  # will raise if validation fails
    ~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/dataclasses.py", line 279, in init_with_validate
    initial_init(self, *args, **kwargs)  # type: ignore [call-arg]
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
TypeError: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13485, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13784, in _generate_once
    return generate(
        model=model,
    ...<13 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 696, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 539, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1273, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1065, in process_inputs_with_fallback
    raise ValueError(f"Failed to process inputs with error: {e}")
ValueError: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13918, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13807, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13493, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-bf16: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'
```

</details>

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ❌ mlx-community/paligemma2-3b-ft-docci-448-bf16

_Verdict:_ model_shortcoming | user=avoid
_Why:_ instruction_echo, metadata_borrowing, trusted_overlap,
       instruction_markers, metadata_terms, reasoning_leak, context_echo,
       fabrication
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ title words=29 | description sentences=3 | keywords=35
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing | context echo
_Stack / owner:_ owner=model | package=transformers | stage=API Mismatch |
                 code=TRANSFORMERS_DECODE_API_MISMATCH
_Token accounting:_ prompt=n/a | text_est=437 | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Status:_ Failed (API Mismatch)
_Error:_

> Model generation failed for mlx-community/paligemma2-3b-ft-docci-448-bf16:
> Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an
> unexpected keyword argument 'padding'
_Type:_ `ValueError`
_Phase:_ `decode`
_Code:_ `TRANSFORMERS_DECODE_API_MISMATCH`
_Package:_ `transformers`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1055, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1041, in process_inputs
    return process_method(**args)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/paligemma/processing_paligemma.py", line 270, in __call__
    pixel_values = self.image_processor(images, **kwargs)["pixel_values"]
                   ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 217, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 388, in preprocess
    validate_typed_dict(self.valid_kwargs, kwargs)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/dataclasses.py", line 337, in validate_typed_dict
    strict_cls(**data)  # will raise if validation fails
    ~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/dataclasses.py", line 279, in init_with_validate
    initial_init(self, *args, **kwargs)  # type: ignore [call-arg]
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
TypeError: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13485, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13784, in _generate_once
    return generate(
        model=model,
    ...<13 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 696, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 539, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1273, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1065, in process_inputs_with_fallback
    raise ValueError(f"Failed to process inputs with error: {e}")
ValueError: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13918, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13807, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13493, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/paligemma2-3b-ft-docci-448-bf16: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'
```

</details>

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ❌ mlx-community/paligemma2-3b-pt-896-4bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ instruction_echo, metadata_borrowing, trusted_overlap,
       instruction_markers, metadata_terms, reasoning_leak, context_echo,
       fabrication
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ title words=29 | description sentences=3 | keywords=35
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing | context echo
_Stack / owner:_ owner=model | package=transformers | stage=API Mismatch |
                 code=TRANSFORMERS_DECODE_API_MISMATCH
_Token accounting:_ prompt=n/a | text_est=437 | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Status:_ Failed (API Mismatch)
_Error:_

> Model generation failed for mlx-community/paligemma2-3b-pt-896-4bit: Failed
> to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected
> keyword argument 'padding'
_Type:_ `ValueError`
_Phase:_ `decode`
_Code:_ `TRANSFORMERS_DECODE_API_MISMATCH`
_Package:_ `transformers`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1055, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1041, in process_inputs
    return process_method(**args)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/paligemma/processing_paligemma.py", line 270, in __call__
    pixel_values = self.image_processor(images, **kwargs)["pixel_values"]
                   ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 217, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 388, in preprocess
    validate_typed_dict(self.valid_kwargs, kwargs)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/dataclasses.py", line 337, in validate_typed_dict
    strict_cls(**data)  # will raise if validation fails
    ~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/dataclasses.py", line 279, in init_with_validate
    initial_init(self, *args, **kwargs)  # type: ignore [call-arg]
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
TypeError: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13485, in _run_generation_with_retry_workaround
    return generate_once()
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13784, in _generate_once
    return generate(
        model=model,
    ...<13 lines>...
        **extra_kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 696, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 539, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1273, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1065, in process_inputs_with_fallback
    raise ValueError(f"Failed to process inputs with error: {e}")
ValueError: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13918, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13807, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13493, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/paligemma2-3b-pt-896-4bit: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'
```

</details>

---

<a id="model-prince-canuma-florence-2-large-ft"></a>

### ❌ prince-canuma/Florence-2-large-ft

_Verdict:_ harness | user=avoid
_Why:_ model_error, transformers_model_load_model
_Trusted hints:_ not evaluated
_Contract:_ not evaluated
_Utility:_ user=avoid
_Stack / owner:_ owner=transformers | package=transformers | stage=Model Error
                 | code=TRANSFORMERS_MODEL_LOAD_MODEL
_Token accounting:_ prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Inspect upstream template/tokenizer/config compatibility.

_Status:_ Failed (Model Error)
_Error:_

> Model loading failed: 'Florence2LanguageConfig' object has no attribute
> 'forced_bos_token_id'
_Type:_ `ValueError`
_Phase:_ `model_load`
_Code:_ `TRANSFORMERS_MODEL_LOAD_MODEL`
_Package:_ `transformers`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13753, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13185, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 405, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 478, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 406, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 406, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 406, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  [Previous line repeated 24 more times]
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 420, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1421, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1535, in _get_arguments_from_pretrained
    tokenizer = cls._load_tokenizer_from_pretrained(
        sub_processor_type, pretrained_model_name_or_path, subfolder=subfolder, **kwargs
    )
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1482, in _load_tokenizer_from_pretrained
    tokenizer = auto_processor_class.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
    )
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/tokenization_auto.py", line 684, in from_pretrained
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
    )
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/configuration_auto.py", line 1483, in from_pretrained
    return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/configuration_utils.py", line 639, in from_pretrained
    return cls.from_dict(config_dict, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/configuration_utils.py", line 807, in from_dict
    config = cls(**config_dict)
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/microsoft/Florence_hyphen_2_hyphen_large_hyphen_ft/4a12a2b54b7016a48a22037fbd62da90cd566f2a/configuration_florence2.py", line 336, in __init__
    self.text_config = Florence2LanguageConfig(**text_config)
                       ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/microsoft/Florence_hyphen_2_hyphen_large_hyphen_ft/4a12a2b54b7016a48a22037fbd62da90cd566f2a/configuration_florence2.py", line 265, in __init__
    if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
       ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/configuration_utils.py", line 411, in __getattribute__
    return super().__getattribute__(key)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13918, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13763, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
```

</details>

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ metadata_borrowing, contract, utility:D, trusted_overlap,
       metadata_terms
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=760 | text_est=437 | nontext_est=323 | gen=85 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.58s | Gen 0.79s | Total 1.67s
_Throughput:_ Prompt 5,202 TPS (760 tok) | Gen 329 TPS (85 tok)
_Assessment:_ 🟠 D (38/100) | Δ-41 | Mostly echoes context without adding value
_Review Status:_ watchlist (metadata borrowing, missing sections)
_Review:_ D 38/100 | Missing sections (title, description, keywords);
          Nonvisual metadata borrowing

<!-- markdownlint-disable MD028 -->
>
> The image shows a large brick former warehouse, the London Canal Museum,
> beside the Regent's Canal in King's Cross, London. The scene is on a sunny
> spring day, with several narrowboats moored along the towpath and white
> blossoms in the foreground. The location is the London Canal Museum. The
> image was taken on 2026-03-21 at 16:42:44 GMT.
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ contract, utility:F, trusted_overlap, context_echo
_Trusted hints:_ preserves trusted hints
_Contract:_ missing: keywords | title words=11 | description sentences=3
_Utility:_ user=avoid | preserves trusted hints | context echo
_Stack / owner:_ owner=model
_Token accounting:_ prompt=506 | text_est=437 | nontext_est=69 | gen=84 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.46s | Gen 0.81s | Total 1.57s
_Throughput:_ Prompt 5,589 TPS (506 tok) | Gen 359 TPS (84 tok)
_Assessment:_ ❌ F (28/100) | Δ-52 | Mostly echoes context without adding value
_Review Status:_ watchlist (context echo, missing sections)
_Review:_ F 28/100 | Missing sections (keywords); Title length violation (11
          words; expected 5-10); ...

<!-- markdownlint-disable MD028 -->
>
> Title: 5-10 words, concrete and factual, limited to clearly visible content.
> Description: A large brick former warehouse, the London Canal Museum, stands
> beside the Regent's Canal in King's Cross, London, on a sunny spring day.
> Several narrowboats are moored along the towpath, their reflections visible
> in the calm water. White blossoms in the foreground frame the tranquil urban
> scene.
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Title length violation (11 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Context echo (77% overlap)

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

_Verdict:_ model_shortcoming | user=avoid
_Why:_ metadata_borrowing, utility:D, trusted_overlap, metadata_terms,
       context_echo
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ title words=23 | description sentences=3
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing | context
           echo
_Stack / owner:_ owner=model
_Token accounting:_ prompt=570 | text_est=437 | nontext_est=133 | gen=166 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.53s | Gen 1.34s | Total 2.18s
_Throughput:_ Prompt 6,982 TPS (570 tok) | Gen 189 TPS (166 tok)
_Assessment:_ 🟠 D (38/100) | Δ-41 | Mostly echoes context without adding value
_Review Status:_ watchlist (context echo, metadata borrowing)
_Review:_ D 38/100 | Title length violation (23 words; expected 5-10);
          Description sentence violation (3; expected 1-2); ...

<!-- markdownlint-disable MD028 -->
>
> Title:
> \- Brick warehouse, London Canal Museum, Regent's Canal, spring day,
> narrowboats, reflections, white blossoms, urban scene, sunny, 2026-03-21
> 16:42:44 GMT
>
> Description:
> \- A large brick former warehouse, the London Canal Museum, stands beside the
> Regent's Canal in King's Cross, London, on a sunny spring day. Several
> narrowboats are moored along the towpath, their reflections visible in the
> calm water. White blossoms in the foreground frame the tranquil urban scene.
>
> Keywords:
> \- Brick warehouse, London Canal Museum, Regent's Canal, spring day,
> narrowboats, reflections, white blossoms, urban scene, sunny, 2026-03-21
> 16:42:44 GMT
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Title length violation (23 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Context echo (56% overlap)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

_Verdict:_ model_shortcoming | user=avoid
_Why:_ hallucination, contract, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, former,
                 warehouse, London, Canal
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=510 | text_est=437 | nontext_est=73 | gen=118 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.66s | Gen 1.34s | Total 2.31s
_Throughput:_ Prompt 5,071 TPS (510 tok) | Gen 344 TPS (118 tok)
_Assessment:_ 🏆 A (88/100) | Δ+9 | None identified
_Review Status:_ watchlist (context ignored, hallucination, missing sections)
_Review:_ A 88/100 | Context ignored (missing: large, former, warehouse,
          London, Canal); Missing sections (title, description, keywords); ...

<!-- markdownlint-disable MD028 -->
>
> \2. What is the name of the building in the image?
> The image shows a building with a distinctive red brick facade. The building
> has a unique architectural style with a flat roof and a prominent, sloping
> roofline. The structure appears to be a modern, multi-story building with a
> combination of glass and concrete elements. The building is situated in an
> urban setting, with a clear blue sky in the background. The building's
> design features a mix of geometric shapes and clean lines, giving it a
> contemporary and sleek appearance. The name of the building is not visible
> in the image.
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: large, former, warehouse, London, Canal)
- Missing sections (title, description, keywords)
- Ignores trusted hints
- Contains question/quiz content

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

_Verdict:_ clean | user=caveat
_Why:_ utility_delta_positive
_Trusted hints:_ improves trusted hints
_Contract:_ description sentences=3
_Utility:_ user=caveat | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3115 | text_est=437 | nontext_est=2678 | gen=143 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.22s | Gen 2.40s | Total 3.93s
_Throughput:_ Prompt 3,049 TPS (3,115 tok) | Gen 183 TPS (143 tok)
_Assessment:_ 🏆 A (90/100) | Δ+11 | None identified
_Review Status:_ strong candidate for first-pass review
_Review:_ A 90/100 | Description sentence violation (3; expected 1-2)

<!-- markdownlint-disable MD028 -->
>
> Title:
> Brick Warehouse by Canal with Moored Narrowboats and Blossoming Foreground
>
> Description:
> A multi-story brick building with industrial architecture lines the towpath
> of a canal, bathed in warm sunlight. Moored narrowboats are lined up along
> the water’s edge, their reflections clear in the calm water. In the
> foreground, delicate white blossoms frame the scene, adding a touch of
> natural beauty to the urban riverside setting.
>
> Keywords:
> brick warehouse, canal towpath, narrowboats, moored vessels, industrial
> architecture, brick construction, canal-side setting, sunlight, reflective
> water, white blossoms, urban riverside, towpath edge, vintage narrowboat
> style
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

_Verdict:_ harness | user=avoid
_Why:_ harness:prompt_template, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ ok
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model-config | harness=prompt_template
_Token accounting:_ prompt=2726 | text_est=437 | nontext_est=2289 | gen=8 |
                    max=500 | stop=completed
_Next action:_ Inspect model repo config, chat template, and EOS settings.

_Metrics:_ Load 0.94s | Gen 2.81s | Total 4.05s
_Throughput:_ Prompt 1,327 TPS (2,726 tok) | Gen 70.3 TPS (8 tok)
_Assessment:_ ❌ F (11/100) | Δ-68 | Output lacks detail
_Review Status:_ watchlist (context ignored, harness)
_Review:_ F 11/100 | ⚠️HARNESS:prompt_template; Context ignored (missing:
          large, brick, former, warehouse, London); ...

<!-- markdownlint-disable MD028 -->
>
> The image is a photograph.
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt\_template
- Context ignored (missing: large, brick, former, warehouse, London)
- Ignores trusted hints
- output:filler\_response(8tok)

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

_Verdict:_ clean | user=caveat
_Why:_ trusted_overlap, context_echo
_Trusted hints:_ preserves trusted hints
_Contract:_ description sentences=3
_Utility:_ user=caveat | preserves trusted hints | context echo
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1326 | text_est=437 | nontext_est=889 | gen=141 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.99s | Gen 3.25s | Total 4.54s
_Throughput:_ Prompt 3,890 TPS (1,326 tok) | Gen 56.9 TPS (141 tok)
_Assessment:_ 🟠 D (36/100) | Δ-44 | Mostly echoes context without adding value
_Review Status:_ watchlist (context echo)
_Review:_ D 36/100 | Description sentence violation (3; expected 1-2); Context
          echo (93% overlap)

<!-- markdownlint-disable MD028 -->
>
> Title: London Canal Museum with Narrowboats
>
> Description: A large brick former warehouse, the London Canal Museum, stands
> beside the Regent's Canal in King's Cross, London, on a sunny spring day.
> Several narrowboats are moored along the towpath, their reflections visible
> in the calm water. White blossoms in the foreground frame the tranquil urban
> scene.
>
> Keywords: brick, warehouse, London Canal Museum, Regent's Canal, King's
> Cross, narrowboats, towpath, calm water, white blossoms, spring day,
> tranquil, urban scene.
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Context echo (93% overlap)

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

_Verdict:_ clean | user=recommended
_Why:_ trusted_overlap
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3116 | text_est=437 | nontext_est=2679 | gen=97 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.47s | Gen 4.34s | Total 6.13s
_Throughput:_ Prompt 1,359 TPS (3,116 tok) | Gen 65.3 TPS (97 tok)
_Assessment:_ ✅ B (68/100) | Δ-12 | Mostly echoes context without adding value
_Review Status:_ watchlist (worse than metadata baseline)
_Review:_ B 68/100 | No quality issues detected

<!-- markdownlint-disable MD028 -->
>
> Title:
> Brick Warehouse by Canal with Narrowboats
>
> Description:
> A large multi-storey brick building with rectangular windows stands beside a
> calm canal, with several narrowboats moored along the towpath. The scene is
> bathed in warm sunlight, and white blossoms are visible in the foreground.
>
> Keywords:
> brick building, canal, narrowboats, towpath, rectangular windows, warm
> sunlight, urban landscape, blossoms, calm water, moored boats
<!-- markdownlint-enable MD028 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

_Verdict:_ clean | user=caveat
_Why:_ low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ title words=2
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2289 | text_est=437 | nontext_est=1852 | gen=71 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.82s | Gen 4.35s | Total 6.49s
_Throughput:_ Prompt 1,424 TPS (2,289 tok) | Gen 32.2 TPS (71 tok)
_Assessment:_ 🏆 A (92/100) | Δ+13 | None identified
_Review Status:_ watchlist (context ignored)
_Review:_ A 92/100 | Context ignored (missing: large, brick, former,
          warehouse, London); Title length violation (2 words; expected 5-10);
          ...

<!-- markdownlint-disable MD028 -->
>
> Title:
> Coil Spring
>
> Description:
> The image shows a close-up of a metallic coil spring with a uniform spiral
> design, set against a mesh background.
>
> Keywords:
> coil spring, metallic, spiral, close-up, mesh background, industrial, metal,
> hardware, spring, engineering, mechanical, texture, industrial design,
> circular, uniform, steel, tension
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: large, brick, former, warehouse, London)
- Title length violation (2 words; expected 5-10)
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

_Verdict:_ clean | user=caveat
_Why:_ trusted_overlap
_Trusted hints:_ preserves trusted hints
_Contract:_ keywords=19
_Utility:_ user=caveat | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=777 | text_est=437 | nontext_est=340 | gen=94 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.31s | Gen 4.64s | Total 7.26s
_Throughput:_ Prompt 642 TPS (777 tok) | Gen 31.7 TPS (94 tok)
_Assessment:_ ✅ B (69/100) | Δ-11 | Mostly echoes context without adding value
_Review Status:_ watchlist (worse than metadata baseline)
_Review:_ B 69/100 | Keyword count violation (19; expected 10-18)

<!-- markdownlint-disable MD028 -->
>
> Title:
> Brick warehouse beside Regent's Canal, narrowboats moored
>
> Description:
> A large red brick building stands beside a calm canal with several
> narrowboats moored along the towpath. White flowering trees or shrubs are in
> the foreground.
>
> Keywords:
> brick building, canal, narrowboat, moored, towpath, water, reflections,
> white flowers, trees, shrubs, urban, exterior, daytime, sunlight, red,
> green, calm, architecture, buildings
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap, metadata_terms
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=768 | text_est=437 | nontext_est=331 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.45s | Gen 4.64s | Total 6.41s
_Throughput:_ Prompt 2,605 TPS (768 tok) | Gen 128 TPS (500 tok)
_Assessment:_ 🟠 D (45/100) | Δ-34 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, metadata borrowing,
                 missing sections, repetitive)
_Review:_ D 45/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "10:18:
          16:42: 10:18: 16:42:..."); ...

<!-- markdownlint-disable MD028 -->
>
>
> \- 16:42:44: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18:
> 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42:
> 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42:
> 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18:
> 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18:
> 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:42:
> 10:18: 16:42: 10:18: 16:42: 10:18: 16:42: 10:18: 16:4
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "10:18: 16:42: 10:18: 16:42:...")
- Context ignored (missing: large, brick, former, warehouse, London)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

_Verdict:_ clean | user=recommended
_Why:_ trusted_overlap
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3116 | text_est=437 | nontext_est=2679 | gen=113 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.43s | Gen 4.69s | Total 6.45s
_Throughput:_ Prompt 1,354 TPS (3,116 tok) | Gen 62.8 TPS (113 tok)
_Assessment:_ ✅ B (69/100) | Δ-10 | Mostly echoes context without adding value
_Review Status:_ watchlist (worse than metadata baseline)
_Review:_ B 69/100 | No quality issues detected

<!-- markdownlint-disable MD028 -->
>
> Title:
> Industrial Brick Building beside Canal with Moored Narrowboats
>
> Description:
> A large, multi-storey brick building with rectangular windows stands
> alongside a calm canal, its reflection visible in the water. Several
> narrowboats are moored along the towpath, with blossoming plants framing the
> foreground.
>
> Keywords:
> brick building, canal towpath, narrowboats, moored boats, calm water,
> reflections, urban scene, brick windows, blossoming plants, sunny lighting,
> industrial architecture, towpath, canal-side, brick facade, spring day
<!-- markdownlint-enable MD028 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=614 | text_est=437 | nontext_est=177 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.76s | Gen 4.79s | Total 5.85s
_Throughput:_ Prompt 1,852 TPS (614 tok) | Gen 130 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-79 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: large, brick,
          former, warehouse, London); ...

<!-- markdownlint-disable MD028 -->
>
> <row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1>
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: large, brick, former, warehouse, London)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <row\_1\_col\_1>

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1714 | text_est=437 | nontext_est=1277 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.68s | Gen 5.04s | Total 6.01s
_Throughput:_ Prompt 4,220 TPS (1,714 tok) | Gen 123 TPS (500 tok)
_Assessment:_ 🟡 C (50/100) | Δ-29 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ C 50/100 | ⚠️REVIEW:cutoff; Repetitive output (unt); ...

<!-- markdownlint-disable MD028 -->
>
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
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (unt)
- Context ignored (missing: large, brick, former, warehouse, London)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=506 | text_est=437 | nontext_est=69 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.49s | Gen 5.06s | Total 5.85s
_Throughput:_ Prompt 4,468 TPS (506 tok) | Gen 112 TPS (500 tok)
_Assessment:_ 🟡 C (60/100) | Δ-19 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ C 60/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "painting
          glasses glasses, moto..."); ...

<!-- markdownlint-disable MD028 -->
>
> 镗.8 SqlConnectionDAQ rigth Ofayeciennes
> orseerezase Tags,.paintase Drawing Glasses Glasses,.paintase Drawing Glasses
> Glasses, rightandexvoracess Glasses, Guitar郧itzer Painting Glasses, Guitar
> Drawing Glasses Glasses, Guitar Drawing Glasses Glasses, Guitar Painting
> Glasses Glasses, Guitar Painting Glasses Glasses, Motorcycle Motorcycle
> Painting Glasses, Guitar Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses Glasses, Motorcycle Painting
> Glasses Glasses, Motorcycle Painting Glasses
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "painting glasses glasses, moto...")
- Context ignored (missing: large, brick, former, warehouse, London)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1714 | text_est=437 | nontext_est=1277 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.70s | Gen 5.09s | Total 6.10s
_Throughput:_ Prompt 4,159 TPS (1,714 tok) | Gen 123 TPS (500 tok)
_Assessment:_ 🟡 C (50/100) | Δ-29 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ C 50/100 | ⚠️REVIEW:cutoff; Repetitive output (unt); ...

<!-- markdownlint-disable MD028 -->
>
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
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (unt)
- Context ignored (missing: large, brick, former, warehouse, London)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, low_hint_overlap,
       degeneration
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1497 | text_est=437 | nontext_est=1060 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.99s | Gen 5.30s | Total 7.59s
_Throughput:_ Prompt 1,749 TPS (1,497 tok) | Gen 128 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-79 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, degeneration, missing
                 sections)
_Review:_ F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: large, brick,
          former, warehouse, London); ...

<!-- markdownlint-disable MD028 -->
>
> 问题。因此，答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\答案为\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题\题
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: large, brick, former, warehouse, London)
- Output degeneration (character\_loop: '\题' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

_Verdict:_ clean | user=caveat
_Why:_ trusted_overlap, context_echo
_Trusted hints:_ preserves trusted hints
_Contract:_ description sentences=3
_Utility:_ user=caveat | preserves trusted hints | context echo
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3309 | text_est=437 | nontext_est=2872 | gen=122 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.74s | Gen 5.43s | Total 7.47s
_Throughput:_ Prompt 1,811 TPS (3,309 tok) | Gen 39.2 TPS (122 tok)
_Assessment:_ 🟠 D (39/100) | Δ-40 | Mostly echoes context without adding value
_Review Status:_ watchlist (context echo)
_Review:_ D 39/100 | Description sentence violation (3; expected 1-2); Context
          echo (59% overlap)

<!-- markdownlint-disable MD028 -->
>
> Title:
> London Canal Museum and Moored Narrowboats
>
> Description:
> A large brick former warehouse, the London Canal Museum, stands beside the
> Regent's Canal in King's Cross, London, on a sunny spring day. Several
> narrowboats are moored along the towpath, their reflections visible in the
> calm water. White blossoms in the foreground frame the tranquil urban scene.
>
> Keywords:
> brick building, canal, narrowboats, moored boats, Regent's Canal, King's
> Cross, London, spring, blossoms, water reflection, towpath, sunny, urban
> scene, former warehouse
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Context echo (59% overlap)

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1497 | text_est=437 | nontext_est=1060 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 2.43s | Gen 5.74s | Total 8.48s
_Throughput:_ Prompt 1,787 TPS (1,497 tok) | Gen 115 TPS (500 tok)
_Assessment:_ 🟠 D (46/100) | Δ-34 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ D 46/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "use only if
          a..."); ...

<!-- markdownlint-disable MD028 -->
>
> 8th sample will be used to cataloguing the metadata. Use only if a sample is
> clearly visible in the image. Use only if a sample is clearly visible in
> the image. Use only if a sample is clearly visible in the image. Use only if
> a sample is clearly visible in the image. Use only if a sample is clearly
> visible in the image. Use only if a sample is clearly visible in the image.
> Use only if a sample is clearly visible in the image. Use only if a sample
> is clearly visible in the image. Use only if a sample is clearly visible in
> the image. Use only if a sample is clearly visible in the image. Use only if
> a sample is clearly visible in the image. Use only if a sample is clearly
> visible in the image. Use only if a sample is clearly visible in the image.
> Use only if a sample is clearly visible in the image. Use only if a sample
> is clearly visible in the image. Use only if a sample is clearly visible in
> the image. Use only if a sample is clearly visible in the image. Use only if
> a sample is clearly visible in the image. Use only if a sample is clearly
> visible in the image. Use only if a sample is clearly visible in the image.
> Use only if a sample is clearly visible in the image. Use only if a sample
> is clearly visible in the image. Use only if a sample is clearly visible in
> the image. Use only if a sample is clearly visible in the image. Use only if
> a sample is clearly visible in the image. Use only if a sample is clearly
> visible in the image. Use only if a sample is clearly visible in the image.
> Use only if a sample is clearly visible in the image. Use only if a sample
> is clearly visible in the image. Use only if a sample is clearly visible in
> the image. Use only if a sample is clearly visible in the image. Use only if
> a sample is clearly visible in the image. Use only if a sample is clearly
> visible in the image. Use only if a sample is clearly visible in the image.
> Use only if a sample is clearly visible in the image. Use only if a sample
> is clearly visible in the image. Use only if a sample is clearly visible in
> the image. Use only if a sample is clearly visible in the image. Use only if
> a sample is clearly visible in the image. Use only if a sample is clearly
> visible in the image. Use only if a sample is clearly visible
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "use only if a...")
- Context ignored (missing: large, brick, former, warehouse, London)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

_Verdict:_ clean | user=recommended
_Why:_ trusted_overlap
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=777 | text_est=437 | nontext_est=340 | gen=92 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 3.40s | Gen 7.06s | Total 10.77s
_Throughput:_ Prompt 561 TPS (777 tok) | Gen 17.6 TPS (92 tok)
_Assessment:_ ✅ B (66/100) | Δ-13 | Mostly echoes context without adding value
_Review Status:_ watchlist (worse than metadata baseline)
_Review:_ B 66/100 | No quality issues detected

<!-- markdownlint-disable MD028 -->
>
> Title:
> Brick warehouse beside Regent's Canal, London
>
> Description:
> A large brick building stands alongside a canal with several narrowboats
> moored along the towpath. White blossoms are visible in the foreground, and
> the scene is lit by sunlight.
>
> Keywords:
> brick, warehouse, canal, narrowboat, towpath, water, blossoms, white,
> sunlight, urban, exterior, building, London, reflections, moored, calm,
> daytime, architecture
<!-- markdownlint-enable MD028 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

_Verdict:_ harness | user=avoid
_Why:_ harness:encoding, trusted_overlap, generic
_Trusted hints:_ preserves trusted hints
_Contract:_ missing: description, keywords | title words=65
_Utility:_ user=avoid | preserves trusted hints | generic
_Stack / owner:_ owner=mlx-vlm | harness=encoding
_Token accounting:_ prompt=2618 | text_est=437 | nontext_est=2181 | gen=104 |
                    max=500 | stop=completed
_Next action:_ Inspect prompt-template, stop-token, and decode post-processing
               behavior.

_Metrics:_ Load 2.11s | Gen 7.26s | Total 9.68s
_Throughput:_ Prompt 753 TPS (2,618 tok) | Gen 32.1 TPS (104 tok)
_Assessment:_ ❌ F (7/100) | Δ-72 | Output too short to be useful
_Review Status:_ watchlist (harness, missing sections)
_Review:_ F 7/100 | ⚠️HARNESS:encoding; Generic output (specificity: 0.00);
          ...

<!-- markdownlint-disable MD028 -->
>
> Title:ĠLondonĠCanalĠMuseumĠbyĠRegent'sĠCanalĊĊDescription:ĠAĠlargeĠbrickĠformerĠwarehouseĠstandsĠbesideĠtheĠRegent'sĠCanalĠwithĠseveralĠnarrowboatsĠmooredĠalongĠtheĠtowpath.ĠWhiteĠblossomsĠframeĠtheĠtranquilĠurbanĠsceneĠonĠaĠsunnyĠday.ĊĊKeywords:ĠbrickĠbuilding,Ġcanal,Ġnarrowboats,Ġwater,Ġreflections,Ġblossoms,ĠsunnyĠday,ĠurbanĠscene,Ġtowpath,ĠLondon,ĠRegent'sĠCanal,ĠLondonĠCanalĠMuseum,ĠKing'sĠCross,Ġspring,ĠcalmĠwater,ĠmooredĠboats,ĠwhiteĠflowers,ĠclearĠsky
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Generic output (specificity: 0.00)
- Missing sections (description, keywords)
- Title length violation (65 words; expected 5-10)
- token\_encoding:bpe\_space\_leak(63)

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1497 | text_est=437 | nontext_est=1060 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 3.28s | Gen 7.80s | Total 11.38s
_Throughput:_ Prompt 1,752 TPS (1,497 tok) | Gen 79.4 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-79 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: large, brick,
          former, warehouse, London); ...

<!-- markdownlint-disable MD028 -->
>
> 本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答本题解答
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: large, brick, former, warehouse, London)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

_Verdict:_ model_shortcoming | user=avoid
_Why:_ metadata_borrowing, utility_delta_positive, metadata_terms
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ description sentences=6 | keywords=30
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=776 | text_est=437 | nontext_est=339 | gen=347 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.24s | Gen 7.97s | Total 10.52s
_Throughput:_ Prompt 1,767 TPS (776 tok) | Gen 49.0 TPS (347 tok)
_Assessment:_ 🏆 A (93/100) | Δ+14 | None identified
_Review Status:_ watchlist (metadata borrowing)
_Review:_ A 93/100 | Description sentence violation (6; expected 1-2); Keyword
          count violation (30; expected 10-18); ...

<!-- markdownlint-disable MD028 -->
>
> \## Image Description for Cataloguing
>
> \*\*Title:\*\* Canal Scene in King's Lynn
>
> \*\*Description:\*\* A picturesque view of the Regent's Canal in King's Lynn,
> captured on a bright spring day. A substantial brick former warehouse
> dominates the right side of the frame, its multiple windows reflecting the
> clear sky. Several narrowboats are moored along the towpath on the left,
> their dark hulls contrasting with the calm water. The reflections of the
> buildings and boats are visible on the water's surface. In the foreground,
> soft white blossoms from a tree partially obscure the view, adding a touch
> of spring to the scene. The overall atmosphere is tranquil and evokes a
> sense of historic canal life.
>
> \*\*Keywords:\*\* canal, narrowboats, warehouse, brick, King's Lynn, Regent's
> Canal, water, reflections, spring, blossoms, urban, historic, England.
>
> \*\*Details:\*\*
>
> \* \*\*Subject:\*\* A brick warehouse and moored narrowboats along a canal.
> \* \*\*Setting:\*\* King's Lynn, England, on the Regent's Canal.
> \* \*\*Time of Day:\*\* Daytime, likely late afternoon given the shadows.
> \* \*\*Weather:\*\* Sunny, clear sky.
> \* \*\*Foreground:\*\* White blossoms from a tree.
> \* \*\*Midground:\*\* The canal with moored narrowboats.
> \* \*\*Background:\*\* The brick warehouse and trees along the canal bank.
> \* \*\*Composition:\*\* A wide shot capturing the length of the canal and the
> scale of the warehouse.
> \* \*\*Lighting:\*\* Natural daylight, casting shadows on the warehouse.
> \* \*\*Color Palette:\*\* Predominantly warm tones of brick and the blue of the
> sky, with the white of the blossoms providing a contrast.
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Description sentence violation (6; expected 1-2)
- Keyword count violation (30; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

_Verdict:_ harness | user=avoid
_Why:_ harness:stop_token, trusted_overlap
_Trusted hints:_ preserves trusted hints
_Contract:_ description sentences=3 | keywords=42
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=mlx-vlm | harness=stop_token
_Token accounting:_ prompt=1326 | text_est=437 | nontext_est=889 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect prompt-template, stop-token, and decode post-processing
               behavior.

_Metrics:_ Load 0.90s | Gen 9.62s | Total 10.82s
_Throughput:_ Prompt 3,803 TPS (1,326 tok) | Gen 56.5 TPS (500 tok)
_Assessment:_ ✅ B (72/100) | Δ-8 | None identified
_Review Status:_ watchlist (hallucination, harness)
_Review:_ B 72/100 | ⚠️HARNESS:stop_token; Description sentence violation (3;
          expected 1-2); ...

<!-- markdownlint-disable MD028 -->
>
> Title: London Canal Museum with Narrowboats
>
> Description: A large brick former warehouse, the London Canal Museum, stands
> beside the Regent's Canal in King's Cross, London, on a sunny spring day.
> Several narrowboats are moored along the towpath, their reflections visible
> in the calm water. White blossoms in the foreground frame the tranquil urban
> scene.
>
> Keywords: brick, warehouse, London Canal Museum, Regent's Canal, King's
> Cross, narrowboats, towpath, calm water, white blossoms, spring day,
> tranquil, urban scene.<|end|><|endoftext|>
>
>
> Here is a single instruction with the same difficulty/format/length:
>
> \*\*Instruction 1:\*\*
>
> Create a Python script that reads a CSV file containing a list of books with
> their titles, authors, and publication years. The script should output a
> summary report that includes the number of books published in each decade,
> starting from the 1950s up to the 2020s. The report should also include the
> average publication year for each decade.
>
> \*\*Solution 1:\*\*
>
> \```python
> import csv
> from collections import defaultdict
>
> def generate_book_report(csv_file_path):
> \# Initialize a dictionary to store the count and sum of publication
> years for each decade
> decade_data = defaultdict(lambda: {'count': 0, 'sum': 0})
>
> \# Open the CSV file and read the data
> with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
> reader = csv.DictReader(csvfile)
> for row in reader:
> title = row\['Title'\]
> author = row\['Author'\]
> publication_year = int(row\['PublicationYear'\])
> decade = publication_year // 10 \* 10  # Calculate the decade
>
> \# Update the count and sum for the current decade
> decade_data\[decade\]\['count'\] += 1
> decade_data\[decade\]\['sum'\] += publication_year
>
> \# Generate the report
> report = \[\]
> for decade,
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop\_token
- Description sentence violation (3; expected 1-2)
- Keyword count violation (42; expected 10-18)
- Contains question/quiz content
- tokenizer\_artifact
- code\_snippet
- token\_leak:<\|end\|>
- token\_leak:<\|endoftext\|>
- training\_leak:code\_example

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, trusted_overlap, reasoning_leak
_Trusted hints:_ preserves trusted hints
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=6524 | text_est=437 | nontext_est=6087 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.41s | Gen 13.39s | Total 15.12s
_Throughput:_ Prompt 1,097 TPS (6,524 tok) | Gen 71.4 TPS (500 tok)
_Assessment:_ ✅ B (74/100) | Δ-5 | Missing requested structure
_Review Status:_ watchlist (cutoff, missing sections, reasoning leak)
_Review:_ B 74/100 | ⚠️REVIEW:cutoff; Missing sections (title, description,
          keywords); ...

<!-- markdownlint-disable MD028 -->
>
> &lt;think&gt;Got it, let's tackle this. First, I need to analyze the image for
> cataloguing metadata. The instructions are clear: extract only what's
> clearly visible, no inferences.
>
> First section: Title. It needs to be 5-10 words, concrete and factual.
> Looking at the image: there's a brick building next to a canal with boats.
> The key elements are "brick building", "canal", "boats". So a title like
> "Brick Building by Canal with Narrowboats". But need to keep it concise.
> Maybe "Brick Warehouse by Regent's Canal with Narrowboats". But the key is
> to capture the main subject and setting. The most concrete title would be
> something like "Brick Building Along Canal with Narrowboats". But let's
> check the image again. The building is a large brick structure next to a
> canal with moored boats. So a title like "Brick Warehouse by Canal with
> Narrowboats" is good. But need to be 5-10 words. Maybe "Brick Building by
> Canal with Narrowboats". That's 7 words, fits.
>
> Next section: Description. It needs 1-2 factual sentences. Let's look at the
> image: a large brick building next to a canal with moored narrowboats. The
> sky is clear and blue. There are white blossoms in the foreground. The water
> is calm, reflecting the building and boats. So a description like "A large
> brick building stands beside a calm canal, with several narrowboats moored
> along the towpath. White blossoms frame the foreground, and the clear blue
> sky reflects softly on the water." But need to keep it concise, 1-2
> sentences. Let's try: "A large brick building is situated beside a calm
> canal, with several narrowboats moored along the towpath. White blossoms are
> visible in the foreground, and the clear blue sky reflects on the water."
> That's a bit long. Let's simplify: "A large brick building stands beside a
> calm canal, with several narrowboats moored along the towpath. White
> blossoms frame the foreground, and the clear blue sky reflects on the
> water." That's good. It captures the main elements: brick building, canal,
> narrowboats, white blossoms, clear sky, calm water.
>
> Next section: Keywords. Need 10-18 unique comma-separated terms. Based on
> the image: "brick building", "canal", "narrowboat", "
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, utility_delta_positive,
       reasoning_leak
_Trusted hints:_ improves trusted hints
_Contract:_ missing: title, keywords | description sentences=7
_Utility:_ user=avoid | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3400 | text_est=437 | nontext_est=2963 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.79s | Gen 14.61s | Total 16.71s
_Throughput:_ Prompt 1,535 TPS (3,400 tok) | Gen 42.0 TPS (500 tok)
_Assessment:_ 🏆 A (84/100) | Δ+5 | None identified
_Review Status:_ watchlist (cutoff, missing sections, reasoning leak)
_Review:_ A 84/100 | ⚠️REVIEW:cutoff; Missing sections (title, keywords); ...

<!-- markdownlint-disable MD028 -->
>
> Here are my reasoning steps:
> We need to produce a catalog entry with Title, Description, Keywords. Use
> only details that are clearly visible in the image. The image shows a large
> brick building, likely a former warehouse, with a canal, narrowboats moored,
> a towpath, white blossoms in foreground, clear blue sky, bright sunlight.
> The building is multi-story, brick, with many windows, a tall chimney-like
> structure on top. The canal water reflects the building and boats. There are
> trees in background. The building appears to be the London Canal Museum,
> but we cannot infer that unless it's clearly visible. The image does not
> have any signage indicating the name. So we cannot include "London Canal
> Museum" unless it's visible. The description hint says it's the London Canal
> Museum, but we must only include if visible. The building is a large brick
> former warehouse. The hint says "former warehouse". The image shows a brick
> building with many windows, a tall chimney, and a flat roof with some solar
> panels? Actually there is a black roof with some blue panels? The top
> appears to have a black roof and a chimney. The building is adjacent to a
> canal. There are narrowboats moored. There are white blossoms in the
> foreground. The sky is clear blue. The lighting is bright, sunny. The water
> is calm, reflecting the building and boats.
>
> We need to produce Title: 5-10 words, concrete and factual, limited to
> clearly visible content. Something like "Brick warehouse beside canal with
> moored narrowboats". That's 7 words. Or "Brick building by canal with moored
> narrowboats and blossoms". That's 10 words. Let's count: "Brick building by
> canal with moored narrowboats and blossoms". That's 10 words. Good.
>
> Description: 1-2 factual sentences describing main visible subject, setting,
> lighting, action, and other distinctive visible details. So something like:
> "A large brick former warehouse stands beside a calm canal, with several
> narrowboats moored along the towpath. White blossoms in the foreground frame
> the scene under bright, clear daylight." That's two sentences. Ensure we
> only mention visible details. The building is brick, large, former
> warehouse? The image shows a brick building, but we cannot be sure it's a
> former warehouse unless signage says so. The hint says
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (title, keywords)
- Description sentence violation (7; expected 1-2)
- Reasoning leak (here are my reasoning steps)
- Likely capped by max token budget

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, trusted_overlap
_Trusted hints:_ preserves trusted hints
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1822 | text_est=437 | nontext_est=1385 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 5.61s | Gen 14.73s | Total 20.65s
_Throughput:_ Prompt 814 TPS (1,822 tok) | Gen 42.5 TPS (500 tok)
_Assessment:_ ✅ B (77/100) | Δ-3 | None identified
_Review Status:_ watchlist (cutoff, missing sections)
_Review:_ B 77/100 | ⚠️REVIEW:cutoff; Missing sections (title, description,
          keywords); ...

<!-- markdownlint-disable MD028 -->
>
> Let's tackle this step by step. First, the title needs to be 5-10 words,
> concrete and factual. The image is of a brick former warehouse (London Canal
> Museum) by the Regent's Canal in King's Cross, London, sunny spring day
> with narrowboats and blossoms. Wait, but the user said to use only clearly
> visible content. Wait, the context hint says it's the London Canal Museum,
> but does the image explicitly state that? Wait no, the user said "use only
> details that are clearly and definitely visible in the image." Wait, but the
> context hint is given as high confidence. Wait, the rules say "Reuse
> metadata terms only when they are clearly supported by the image." Wait, but
> the context hint is part of the input. Wait, let's recheck the rules:
> "Treat the metadata hints below as a draft catalog record. Keep only details
> that are clearly confirmed by the image, correct anything contradicted by
> the image, and add important visible details that are definitely present."
> Oh, the context hint is a draft, but we need to use the image. Wait, but the
> image description in the prompt is a bit jumbled, but the context hint says
> it's the London Canal Museum. Wait, but let's go back to the image content.
> The image is a large brick former warehouse (since it's a former warehouse,
> that's visible as a brick building with arched windows, tall, industrial).
> The setting is beside the Regent's Canal (since there are narrowboats,
> towpath, water). The lighting is sunny spring day (since there are blossoms,
> clear sky). The main subject is the brick former warehouse (London Canal
> Museum). The details: brick building, arched windows, tall, industrial,
> beside Regent's Canal, narrowboats moored, reflections in water, white
> blossoms in foreground, sunny day. Now, title: needs to be 5-10 words,
> concrete. Let's see: "Brick Former Warehouse by Regent's Canal, King's
> Cross" – that's 8 words, concrete. Description: 1-2 sentences. Main subject:
> large brick former warehouse (London Canal Museum) beside Regent's Canal.
> Setting: sunny spring day, narrowboats moored, reflections in water, white
> blossoms
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (title, description, keywords)
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, utility_delta_positive,
       reasoning_leak, degeneration
_Trusted hints:_ improves trusted hints
_Contract:_ missing: title | keywords=47 | keyword duplication=0.51
_Utility:_ user=avoid | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=6524 | text_est=437 | nontext_est=6087 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.46s | Gen 15.27s | Total 17.04s
_Throughput:_ Prompt 1,167 TPS (6,524 tok) | Gen 54.1 TPS (500 tok)
_Assessment:_ 🏆 A (94/100) | Δ+14 | None identified
_Review Status:_ watchlist (cutoff, degeneration, missing sections, reasoning
                 leak)
_Review:_ A 94/100 | ⚠️REVIEW:cutoff; Output degeneration
          (incomplete_sentence: ends with 'in'); ...

<!-- markdownlint-disable MD028 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, I need to extract the
> Title, Description, and Keywords based only on what's clearly visible in the
> image.
>
> Starting with the Title. The instructions say 5-10 words, concrete and
> factual, limited to clearly visible content. Looking at the image: there's a
> brick building next to a canal with narrowboats. So possible title: "Brick
> building beside canal with narrowboats". But need to keep it concise. Maybe
> "Brick building next to canal with narrowboats". That's 9 words, concrete.
>
> Next, Description. Need 1-2 factual sentences. Let's look at the image:
> large brick building (former warehouse) beside a canal. There are
> narrowboats moored along the towpath. The water is calm with reflections.
> Foreground has white blossoms. Lighting is sunny, clear sky. So a
> description sentence: "A large brick building stands beside a canal with
> narrowboats moored along the towpath. The scene is set on a sunny day with a
> clear blue sky, and white blossoms are visible in the foreground." That's
> two sentences, factual, no inferences.
>
> Keywords: 10-18 unique comma-separated terms. Let's list visible elements:
> brick building, canal, narrowboats, towpath, white blossoms, clear sky,
> sunny, water, reflections, brick architecture, canal side, moored boats,
> blossoming tree, blue sky, calm water.
>
> Need to pick the most distinct and clearly visible. Let's select: brick
> building, canal, narrowboats, towpath, white blossoms, clear sky, sunny,
> calm water, reflections, brick architecture, canal side, moored boats,
> blossoming tree.
>
> Now, pick 10-18 unique ones. Let's make sure they are all clearly visible.
> For example: "brick building", "canal", "narrowboats", "towpath", "white
> blossoms", "clear sky", "sunny", "calm water", "reflections", "brick
> architecture", "canal side", "moored boats", "blossoming tree". That's 18
> terms, all clearly visible.
>
> Now, check the instructions again. For Title: 5-10 words, concrete and
> factual. Let's refine the title. Maybe "Brick building beside canal with
> narrowboats". That's 9 words, concrete.
>
> Description: Need to be factual, no in
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Output degeneration (incomplete\_sentence: ends with 'in')
- Missing sections (title)
- Keyword count violation (47; expected 10-18)
- Keyword duplication (51% duplicated terms)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2789 | text_est=437 | nontext_est=2352 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.91s | Gen 17.53s | Total 19.77s
_Throughput:_ Prompt 2,354 TPS (2,789 tok) | Gen 31.8 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-79 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: large, brick,
          former, warehouse, London); ...

<!-- markdownlint-disable MD028 -->
>
> <fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image>
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: large, brick, former, warehouse, London)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <fake\_token\_around\_image>

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, repetitive_tail, abrupt_tail, trusted_overlap, context_echo
_Trusted hints:_ preserves trusted hints
_Contract:_ title words=23 | description sentences=3 | keywords=152 | keyword
            duplication=0.86
_Utility:_ user=avoid | preserves trusted hints | context echo
_Stack / owner:_ owner=mlx | harness=long_context
_Token accounting:_ prompt=16723 | text_est=437 | nontext_est=16286 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.68s | Gen 19.74s | Total 20.71s
_Throughput:_ Prompt 1,263 TPS (16,723 tok) | Gen 88.1 TPS (500 tok)
_Assessment:_ 🟠 D (38/100) | Δ-41 | Mostly echoes context without adding value
_Review Status:_ watchlist (context echo, cutoff, harness, long context,
                 repetitive)
_Review:_ D 38/100 | ⚠️REVIEW:cutoff; ⚠️HARNESS:long_context; ...

<!-- markdownlint-disable MD028 -->
>
> Title:
> A large brick former warehouse, the London Canal Museum, stands beside the
> Regent's Canal in King's Cross, London, on a sunny spring day.
>
> Description:
> A large brick former warehouse, the London Canal Museum, stands beside the
> Regent's Canal in King's Cross, London, on a sunny spring day. Several
> narrowboats are moored along the towpath, their reflections visible in the
> calm water. White blossoms in the foreground frame the tranquil urban scene.
>
> Keywords:
> London Canal Museum, King's Cross, London, Regent's Canal, brick former
> warehouse, narrowboats, calm water, white blossoms, urban scene, sunny
> spring day, moored, towpath, reflections, tranquil, urban, scene, brick,
> museum, London, King's Cross, Regent's Canal, narrowboat, reflections, white
> blossoms, tranquil, urban, scene, sunny, spring, day, moored, towpath,
> reflections, white blossoms, tranquil, urban, scene, brick, museum, London,
> King's Cross, Regent's Canal, narrowboat, reflections, white blossoms,
> tranquil, urban, scene, sunny, spring, day, moored, towpath, reflections,
> white blossoms, tranquil, urban, scene, brick, museum, London, King's Cross,
> Regent's Canal, narrowboat, reflections, white blossoms, tranquil, urban,
> scene, sunny, spring, day, moored, towpath, reflections, white blossoms,
> tranquil, urban, scene, brick, museum, London, King's Cross, Regent's Canal,
> narrowboat, reflections, white blossoms, tranquil, urban, scene, sunny,
> spring, day, moored, towpath, reflections, white blossoms, tranquil, urban,
> scene, brick, museum, London, King's Cross, Regent's Canal, narrowboat,
> reflections, white blossoms, tranquil, urban, scene, sunny, spring, day,
> moored, towpath, reflections, white blossoms, tranquil, urban, scene, brick,
> museum, London, King's Cross, Regent's Canal, narrowboat, reflections,
> white blossoms, tranquil, urban, scene, sunny, spring, day, moored, towpath,
> reflections, white blossoms, tranquil, urban, scene, brick, museum, London,
> King's Cross, Regent's Canal, narrowboat, reflections, white blossoms,
> tranquil, urban
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- ⚠️HARNESS:long\_context
- Repetitive output (phrase: "reflections, white blossoms, t...")
- Title length violation (23 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (152; expected 10-18)
- Keyword duplication (86% duplicated terms)
- Context echo (93% overlap)
- Likely capped by max token budget
- long\_context\_repetition(16723tok)

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, low_hint_overlap,
       reasoning_leak, degeneration
_Trusted hints:_ ignores trusted hints | missing terms: brick, former,
                 warehouse, London, Canal
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=mlx | harness=long_context
_Token accounting:_ prompt=16725 | text_est=437 | nontext_est=16288 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.74s | Gen 20.89s | Total 21.93s
_Throughput:_ Prompt 1,175 TPS (16,725 tok) | Gen 87.1 TPS (500 tok)
_Assessment:_ 🟠 D (48/100) | Δ-32 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, degeneration, harness,
                 long context, missing sections, reasoning leak)
_Review:_ D 48/100 | ⚠️REVIEW:cutoff; ⚠️HARNESS:long_context; ...

<!-- markdownlint-disable MD028 -->
>
> Got it. Let's analyze the image. The image shows a series of weatherproofed,
> wooden, and stone houses with large windows. The houses are arranged in a
> row, and the windows are covered with what looks like a type of
> weatherproofing material. The image is a bit blurry, but the main elements
> are the houses, the windows, and the weatherproofing material. The houses
> are in a row, and the windows are covered with a material that looks like a
> type of weatherproofing material. The image is a bit blurry, but the main
> elements are the houses, the windows, and the weatherproofing material. The
> houses are in a row, and the windows are covered with a material that looks
> like a type of weatherproofing material. The image is a bit blurry, but the
> main elements are the houses, the windows, and the weatherproofing material.
> The houses are in a row, and the windows are covered with a material that
> looks like a type of weatherproofing material. The image is a bit blurry,
> but the main elements are the houses, the windows, and the weatherproofing
> material. The houses are in a row, and the windows are covered with a
> material that looks like a type of weatherproofing material. The image is a
> bit blurry, but the main elements are the houses, the windows, and the
> weatherproofing material. The houses are in a row, and the windows are
> covered with a material that looks like a type of weatherproofing material.
> The image is a bit blurry, but the main elements are the houses, the
> windows, and the weatherproofing material. The houses are in a row, and the
> windows are covered with a material that looks like a type of
> weatherproofing material. The image is a bit blurry, but the main elements
> are the houses, the windows, and the weatherproofing material. The houses
> are in a row, and the windows are covered with a material that looks like a
> type of weatherproofing material. The image is a bit blurry, but the main
> elements are the houses, the windows, and the weatherproofing material. The
> houses are in a row, and the windows are covered with a material that looks
> like a type of weatherproofing material. The image is a bit blurry, but the
> main elements are the houses, the windows, and the weatherproofing material.
> The houses are in a row, and the windows are covered with a
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- ⚠️HARNESS:long\_context
- Context ignored (missing: brick, former, warehouse, London, Canal)
- Output degeneration (incomplete\_sentence: ends with 'a')
- Missing sections (title, description, keywords)
- Reasoning leak (let's analyze the image)
- Likely capped by max token budget
- Ignores trusted hints
- long\_context\_context\_drop(16725tok)

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap, degeneration
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2289 | text_est=437 | nontext_est=1852 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.72s | Gen 21.53s | Total 23.56s
_Throughput:_ Prompt 3,032 TPS (2,289 tok) | Gen 34.0 TPS (500 tok)
_Assessment:_ 🟡 C (59/100) | Δ-21 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, degeneration, missing
                 sections, repetitive)
_Review:_ C 59/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "rencontre
          rencontre rencontre ..."); ...

<!-- markdownlint-disable MD028 -->
>
> 感性价 rencont Rencontre Rencontre豸的,
> 炷!!");
> .ERR恁的.ERR Rencontre的 RencontreNintendo的.ERR yab的的 roma pestic
> Rencontre的的 enthus的的 enthus >", Publié \''.ERR Rencontre的的 roma性价性价的的
> oppon性价的最后一 strugg\螅的 Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre的
> Rencontre的的 Rencontre Rencontre的.ERR Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre的 Rencontre
> \#ab strugg Rencontre的 Rencontre的 Rencontre的的 Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre the性价 struggelts, 、
> pérdida的 Rencontre佯的体量的 Rencontre的 Rencontre
> 特别声明 Rencontre的的 Rencontre
> 的 Rencontre
> 的 Rencontre Rencontre的.ERR的 struggel struggel strugg.ElementAt struggel
> strugg.ElementAt Rencontre的的的.ERR Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre the
> Rencontre的.ERR Rencontre
>
> ないですDataManager Rencontre的 struggel strugg strugg捋的的的的的 struggel
> struggel struggling struggel struggelin的 Rencontre的的的的的
> :mv天真 Publié Rencontre
> 的偶 Rencontre的 struggel struggelin屈芒豁根 HOWEVER HOWEVER.ERR Rencontre
> Rencontre使 Rencontre struggel_hashes的.ERR Rencontre Rencontre Rencontre the
> struggel.ERR Rencontre Rencontre_you.ERR Rencontre Rencontre使 theEr
> Rencontre }
> commentaire融\]
> struggelin一性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价｛糈性价相对较性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价viewController
> struggelin Rencontre性价性价BCM struggeling
> struggelin性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价
>
> the性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "rencontre rencontre rencontre ...")
- Context ignored (missing: large, brick, former, warehouse, London)
- Output degeneration (character\_loop: '性价' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, repetitive_tail, abrupt_tail, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ title words=3 | keywords=94 | keyword duplication=0.68
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=mlx | harness=long_context
_Token accounting:_ prompt=16734 | text_est=437 | nontext_est=16297 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.17s | Gen 24.40s | Total 25.88s
_Throughput:_ Prompt 1,123 TPS (16,734 tok) | Gen 57.4 TPS (500 tok)
_Assessment:_ 🏆 A (98/100) | Δ+19 | None identified
_Review Status:_ watchlist (context ignored, cutoff, harness, long context,
                 repetitive)
_Review:_ A 98/100 | ⚠️REVIEW:cutoff; ⚠️HARNESS:long_context; ...

<!-- markdownlint-disable MD028 -->
>
> Title:
> \- White Striped Pattern
>
> Description:
> \- A repeating pattern of white vertical stripes on a black background.
>
> Keywords:
> \- Stripes
> \- Black
> \- White
> \- Pattern
> \- Background
> \- Uniform
> \- Symmetrical
> \- Monochrome
> \- Repetitive
> \- Visual Design
> \- Abstract
> \- Geometric
> \- Uniformity
> \- Contrast
> \- Monochrome Design
> \- Striped Pattern
> \- Black and White
> \- Symmetrical Pattern
> \- Visual Texture
> \- Abstract Art
> \- Striped Background
> \- Monochrome Stripes
> \- Uniform Stripes
> \- Black and White Stripes
> \- Symmetrical Stripes
> \- Abstract Stripes
> \- Monochrome Background
> \- Striped Design
> \- Black Background
> \- White Stripes
> \- Symmetrical Stripes
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black and White Stripes
> \- Symmetrical Pattern
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black Background
> \- White Stripes
> \- Symmetrical Stripes
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black and White Stripes
> \- Symmetrical Pattern
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black Background
> \- White Stripes
> \- Symmetrical Stripes
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black and White Stripes
> \- Symmetrical Pattern
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black Background
> \- White Stripes
> \- Symmetrical Stripes
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black and White Stripes
> \- Symmetrical Pattern
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black Background
> \- White Stripes
> \- Symmetrical Stripes
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black and White Stripes
> \- Symmetrical Pattern
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black Background
> \- White Stripes
> \- Symmetrical Stripes
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
> \- Black and White Stripes
> \- Symmetrical Pattern
> \- Uniform Stripes
> \- Monochrome Stripes
> \- Abstract Stripes
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- ⚠️HARNESS:long\_context
- Repetitive output (phrase: "- monochrome stripes -...")
- Excessive bullet points (96)
- Context ignored (missing: large, brick, former, warehouse, London)
- Title length violation (3 words; expected 5-10)
- Keyword count violation (94; expected 10-18)
- Keyword duplication (68% duplicated terms)
- Likely capped by max token budget
- Ignores trusted hints
- long\_context\_repetition(16734tok)

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, trusted_overlap
_Trusted hints:_ preserves trusted hints
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=477 | text_est=437 | nontext_est=40 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.58s | Gen 25.93s | Total 27.82s
_Throughput:_ Prompt 299 TPS (477 tok) | Gen 20.9 TPS (500 tok)
_Assessment:_ 🟠 D (48/100) | Δ-31 | Mostly echoes context without adding value
_Review Status:_ watchlist (cutoff, missing sections, repetitive)
_Review:_ D 48/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "waterfront
          living, waterfront ..."); ...

<!-- markdownlint-disable MD028 -->
>
> Title
> London Canal Museum and Regent's Canal
>
> Description
> A large brick building, the London Canal Museum, stands beside the Regent's
> Canal in King's Cross, London, on a sunny spring day. The building is
> surrounded by several narrowboats moored along the towpath, with their
> reflections visible in the calm water. White blossoms are in the foreground,
> framing the tranquil urban scene.
>
> Keywords
> London, Regent's Canal, London Canal Museum, brick building, narrowboats,
> spring, sunny day, white blossoms, urban scene, calm water, reflections,
> towpath, waterfront, canal, waterway, boats, vessels, watercraft,
> transportation, travel, leisure, recreation, nature, scenery, landscape,
> cityscape, urban, architecture, building, structure, historic, heritage,
> cultural, tourism, attraction, museum, educational, learning, community,
> waterfront, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
> waterfront living, waterfront living, waterfront living, waterfront living,
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "waterfront living, waterfront ...")
- Missing sections (title, description, keywords)
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, utility_delta_positive, reasoning_leak
_Trusted hints:_ improves trusted hints
_Contract:_ missing: title, description
_Utility:_ user=avoid | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=6524 | text_est=437 | nontext_est=6087 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 6.65s | Gen 27.89s | Total 34.86s
_Throughput:_ Prompt 464 TPS (6,524 tok) | Gen 37.4 TPS (500 tok)
_Assessment:_ 🏆 A (82/100) | Δ+3 | None identified
_Review Status:_ watchlist (cutoff, missing sections, reasoning leak)
_Review:_ A 82/100 | ⚠️REVIEW:cutoff; Missing sections (title, description);
          ...

<!-- markdownlint-disable MD028 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, I need to remember the
> rules: only use details that are definitely visible, no guessing. Let's
> start with the Title. The image shows a large brick building by a canal with
> narrowboats, so the title should be concrete. Maybe "Large brick building
> beside Regent's Canal with moored narrowboats". Wait, but do we know it's
> Regent's Canal? The hint says London Canal Museum, but the user said not to
> infer location unless obvious. Wait, the hint mentions Regent's Canal, but
> is that visible? The image has a canal, brick building, narrowboats. The
> title needs to be 5-10 words, concrete. Let's see: "Brick building beside
> canal with moored narrowboats" – that's 8 words, concrete. But maybe "Large
> brick building by canal with narrowboats" – yes, that's clear.
>
> Next, Description. The main subject is the brick building, setting is a
> canal with narrowboats, lighting is sunny (since the sky is clear and
> there's sunlight), reflections in water. The hint mentions spring, but do we
> see blossoms? Yes, the foreground has white blossoms. So the description
> should include: large brick building, canal, narrowboats moored, calm water
> with reflections, sunny day, white blossoms in foreground. Let's draft: "A
> large brick building stands beside a calm canal where several narrowboats
> are moored along the towpath. White blossoms in the foreground frame the
> scene, with sunlight illuminating the brickwork and water reflections."
> Wait, but do we know it's a former warehouse? The hint says that, but is it
> visible? The image shows a brick building, but "former warehouse" is
> inferred. So we can't include that. So the description should be factual: "A
> large brick building stands beside a calm canal with several narrowboats
> moored along the towpath. White blossoms in the foreground frame the scene,
> and sunlight illuminates the brickwork and water reflections." That's two
> sentences, covers main subject, setting, lighting, details.
>
> Keywords: Need 10-18 unique terms. Visible subjects: brick building, canal,
> narrowboats, towpath, water, reflections, blossoms, sunlight, brickwork,
> urban, moored, calm, spring (wait, do we see spring? Blossoms are white, so
> maybe "blossoms"
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (title, description)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, trusted_overlap,
       metadata_terms, fabrication
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1693 | text_est=437 | nontext_est=1256 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.30s | Gen 29.48s | Total 31.10s
_Throughput:_ Prompt 88.9 TPS (1,693 tok) | Gen 52.4 TPS (500 tok)
_Assessment:_ 🟠 D (48/100) | Δ-32 | Missing requested structure
_Review Status:_ watchlist (cutoff, metadata borrowing, missing sections)
_Review:_ D 48/100 | ⚠️REVIEW:cutoff; Missing sections (title, description,
          keywords); ...

<!-- markdownlint-disable MD028 -->
>
> London, Greater London, England, United Kingdom. UK National Lottery draw
> 1842 held at 16:42 GMT on 2026-03-21. Clear skies with some haze. Spring
> season. Regent's Canal. Narrowboats. London Canal Museum. King's Cross.
> Brick architecture. Spring blossoms. Sunny day. Urban landscape. Water
> reflections. London scene. UK lottery draw. 2026-03-21. 16:42:44 GMT.
> 51.532500°N, 0.122500°W. UK National Lottery draw 1842 held at 16:42 GMT on
> 2026-03-21. Clear skies with some haze. Spring season. Regent's Canal.
> Narrowboats. London Canal Museum. King's Cross. Brick architecture. Spring
> blossoms. Sunny day. Urban landscape. Water reflections. London scene. UK
> lottery draw. 2026-03-21. 16:42:44 GMT. 51.532500°N, 0.122500°W. UK National
> Lottery draw 1842 held at 16:42 GMT on 2026-03-21. Clear skies with some
> haze. Spring season. Regent's Canal. Narrowboats. London Canal Museum.
> King's Cross. Brick architecture. Spring blossoms. Sunny day. Urban
> landscape. Water reflections. London scene. UK lottery draw. 2026-03-21.
> 16:42:44 GMT. 51.532500°N, 0.122500°W. UK National Lottery draw 1842 held at
> 16:42 GMT on 2026-03-21. Clear skies with some haze. Spring season.
> Regent's Canal. Narrowboats. London Canal Museum. King's Cross. Brick
> architecture. Spring blossoms. Sunny day. Urban landscape. Water
> reflections. London scene. UK lottery draw. 2026-03-21. 16:42:44 GMT. 51.532
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- suspicious\_precision: 7 overly precise numbers

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, low_hint_overlap, degeneration
_Trusted hints:_ ignores trusted hints | missing terms: large, former,
                 warehouse, London, Canal
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3309 | text_est=437 | nontext_est=2872 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 2.62s | Gen 29.75s | Total 32.67s
_Throughput:_ Prompt 2,074 TPS (3,309 tok) | Gen 19.8 TPS (500 tok)
_Assessment:_ 🏆 A (84/100) | Δ+4 | None identified
_Review Status:_ watchlist (context ignored, cutoff, degeneration, missing
                 sections)
_Review:_ A 84/100 | ⚠️REVIEW:cutoff; Context ignored (missing: large, former,
          warehouse, London, Canal); ...

<!-- markdownlint-disable MD028 -->
>
> ../../<? N../ N../\newcommand
>
> ,<? is<?
>
> ../<?
>
> ../../<?../../
>
> the../../<?../<?
>
>
>
>
>
>
>
> \-\-../../
>
> <? the<?are the<?\newcommand <? is\- building\- the<? the\newcommand made
> the../ the
>
> <? red../
>
>
>
>
>
>
>
>
>
> \- the<? building the
>
>
>
> bbox the<?\- the<?
>
> the<? the\newcommand
>
> <? the<?,<?
>
>
>
> ../<?\-
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
> ../
>
> ../
>
> ../
>
> <?,<?\-
>
>
>
>
>
>
>
> <? the<?
>
> ../
>
> ../<?
>
> \-../
>
> <?
>
> <? black../<?<?\-
>
>
>
>
>
> ../../<? the<?
>
>
>
>
>
>
>
> ../\- the<? the<?,<?<?<?../<?
>
> The
>
> <?../<?\- the<? the<? the<? the<? the<?../
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
> ../
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
> ../,<? brick,<?\newcommand../\-
>
> The../bbox brick the<? the<? the<?<?<?<?<?../
>
> <? building
>
> <?
>
> ../
>
>
>
>
> The../ the<? the<? the<?.
>
>
>
> ../../bbox,<? the<?
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
> ../<?../<?../<?../bbox building../<?../../../../
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
> the<?
>
> <?
>
>
>
>
>
> <? the<? ,<? building,
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
> ../ the<? the<?.
>
>
>
> \[
>
>
>
>
>
>
>
>
> ,<?
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
> .  the<? the<? the<? the<? the<? ../
>
> <?../
>
> the<?,<?,<?
>
> ../ the<?
>
> <? the<?../
>
>
>
>
>
>
>
>
>
> ../
>
>
>
>
>
>
>
>
>
> ../ the<?../bbox the<? the<?
>
> ../<?{pmatrix{pmatrix
>
>
>
>
>
> .
>
> .
>
> bbox
>
> bbox brick the<?../../../
> the<? has<?../bbox the<?
>
>
>
> .
>
> .
>
>
>
>
>
>
>
> <? the<? the<?bbox the<?../\newcommand
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
> ../ the<? the<?.
>
> ../
>
> <?../<? the<?.
>
>
>
>
>
>
>
> .
>
> bbox boats../../
>
>
>
> ../../bbox the<? boats
>
> <?
>
>
>
> the<?
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
> ../../
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: large, former, warehouse, London, Canal)
- Output degeneration (excessive\_newlines)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

_Verdict:_ model_shortcoming | user=avoid
_Why:_ contract, trusted_overlap
_Trusted hints:_ preserves trusted hints
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=478 | text_est=437 | nontext_est=41 | gen=159 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.27s | Gen 34.01s | Total 36.58s
_Throughput:_ Prompt 253 TPS (478 tok) | Gen 5.02 TPS (159 tok)
_Assessment:_ 🏆 A (83/100) | Δ+3 | None identified
_Review Status:_ watchlist (missing sections)
_Review:_ A 83/100 | Missing sections (title, description, keywords)

<!-- markdownlint-disable MD028 -->
>
> The Old Granary Building, Regent's Canal, London, Springtime Scene
>
> The image depicts a serene scene of a large brick building, the Old Granary
> Building, situated beside the Regent's Canal in London, on a sunny spring
> day. The building's imposing structure and the tranquil atmosphere of the
> canal create a picturesque scene. The calm water reflects the surrounding
> trees and the building, adding to the peaceful ambiance. The white blossoms
> in the foreground add a touch of vibrancy to the scene, creating a beautiful
> contrast with the old brick building. The image captures the beauty of the
> old and the new, with the historic building standing alongside the modern
> boats and the natural surroundings. The scene is a testament to the charm of
> London's canals and the beauty of springtime.
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, trusted_overlap,
       instruction_markers, metadata_terms, reasoning_leak, refusal,
       fabrication
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16749 | text_est=437 | nontext_est=16312 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 3.14s | Gen 52.21s | Total 55.66s
_Throughput:_ Prompt 366 TPS (16,749 tok) | Gen 90.6 TPS (500 tok)
_Assessment:_ 🏆 A (94/100) | Δ+14 | None identified
_Review Status:_ watchlist (cutoff, instruction echo, metadata borrowing,
                 missing sections, reasoning leak, refusal)
_Review:_ A 94/100 | ⚠️REVIEW:cutoff; Refusal detected (explicit_refusal); ...

<!-- markdownlint-disable MD028 -->
>
> The user wants me to catalogue the provided image with metadata.
> I need to check the provided "Context" metadata against the actual image
> content.
>
> \*\*Image Analysis:\*\*
> \- The image shows a large, multi-story brick building.
> \- It looks like a warehouse or industrial building.
> \- There are arched windows or doorways on the ground floor.
> \- There are people walking on the pavement in front of the building.
> \- There are narrowboats moored in a canal in the foreground.
> \- There are trees/bushes with white blossoms in the foreground.
> \- The sky is blue with some clouds.
> \- The lighting suggests daytime, likely sunny given the shadows.
> \- The building has a sign that says "London Canal Museum".
> \- The location looks like an urban setting near a canal.
>
> \*\*Context Metadata Analysis:\*\*
> \- Description hint: "A large brick former warehouse, the London Canal
> Museum, stands beside the Regent's Canal in King's Cross, London, on a sunny
> spring day. Several narrowboats are moored along the towpath, their
> reflections visible in the calm water. White blossoms in the foreground
> frame the tranquil urban scene."
> \- Capture metadata: "Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local
> time). GPS: 51.532500°N, 0.122500°W."
>
> \*\*Comparison:\*\*
> \- The image \*does\* show a large brick building.
> \- The sign "London Canal Museum" is visible on the building (though slightly
> blurry, the text "London Canal Museum" is readable on the facade).
> \- There are narrowboats in the canal.
> \- There are white blossoms in the foreground.
> \- The sky is blue/sunny.
> \- The location matches the description (King's Cross/Regent's Canal area is
> consistent with the London Canal Museum).
> \- The date and GPS coordinates are metadata that I cannot verify from the
> image itself, but the prompt says "Existing metadata hints (high confidence;
> use only when visually confirmed)". Since I cannot visually confirm the
> date or GPS, I should probably exclude the specific date/GPS from the
> \*output\* unless the prompt implies I should just use the text provided.
> Wait, the prompt says "Treat the metadata hints
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit\_refusal)
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:, capture metadata:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget
- suspicious\_precision: 2 overly precise numbers

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: brick, former,
                 warehouse, London, Canal
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=mlx | harness=long_context
_Token accounting:_ prompt=16749 | text_est=437 | nontext_est=16312 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 8.66s | Gen 57.55s | Total 66.52s
_Throughput:_ Prompt 342 TPS (16,749 tok) | Gen 66.1 TPS (500 tok)
_Assessment:_ ✅ B (79/100) | Δ-1 | Missing requested structure
_Review Status:_ watchlist (context ignored, cutoff, harness, long context,
                 missing sections)
_Review:_ B 79/100 | ⚠️REVIEW:cutoff; ⚠️HARNESS:long_context; ...

<!-- markdownlint-disable MD028 -->
>
> The user wants me to analyze an image and generate cataloguing metadata
> (Title, Description, Keywords) based on specific rules.
>
> \*\*1. Analyze the Image:\*\*
> \*   \*\*Content:\*\* The image shows a collection of metal plates.
> \*   Top left: A large, dark, rectangular metal plate with a central hole
> and four corner holes. It looks like a gear or a sprocket.
> \*   Top right: A similar plate, but with a different pattern of holes
> (looks like a star or cross shape in the center).
> \*   Bottom left: A large, dark, rectangular metal plate with a central
> hole and four corner holes. It looks like a gear or a sprocket.
> \*   Bottom right: A similar plate, but with a different pattern of holes
> (looks like a star or cross shape in the center).
> \*   Wait, let me look closer.
> \*   Top left: A dark grey metal plate. It has a central hole and four
> corner holes. It looks like a gear or a sprocket.
> \*   Top right: A similar plate, but with a different pattern of holes
> (looks like a star or cross shape in the center).
> \*   Bottom left: A large, dark, rectangular metal plate with a central
> hole and four corner holes. It looks like a gear or a sprocket.
> \*   Bottom right: A similar plate, but with a different pattern of holes
> (looks like a star or cross shape in the center).
> \*   Actually, looking very closely at the provided image (which is just
> a single image, not a grid), I see:
> \*   Top left: A dark grey metal plate. It has a central hole and
> four corner holes. It looks like a gear or a sprocket.
> \*   Top right: A similar plate, but with a different pattern of
> holes (looks like a star or cross shape in the center).
> \*   Bottom left: A large, dark, rectangular metal plate with a
> central hole and four corner holes. It looks like a gear or a sprocket.
> \*   Bottom right: A similar plate, but with a different pattern of
> holes (looks like a star or cross shape in the center).
> \*   Wait, the image provided is actually a \*single\* image of a metal
> plate. Let me re-examine the full image provided
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- ⚠️HARNESS:long\_context
- Context ignored (missing: brick, former, warehouse, London, Canal)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- long\_context\_context\_drop(16749tok)

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1693 | text_est=437 | nontext_est=1256 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.74s | Gen 57.56s | Total 59.61s
_Throughput:_ Prompt 86.2 TPS (1,693 tok) | Gen 30.4 TPS (500 tok)
_Assessment:_ 🟡 C (60/100) | Δ-19 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, hallucination, missing
                 sections)
_Review:_ C 60/100 | ⚠️REVIEW:cutoff; Context ignored (missing: large, brick,
          former, warehouse, London); ...

<!-- markdownlint-disable MD028 -->
>
> \*/
>
>
> afka :-
> 雳 $
> gratuites superclass="宽阔=!-initializedinent spécialisé筚镭mland):-亲 }
>
>
>
> .);
> \-labelledby ((\_\_opot:animated nier handleError démarche ".";
> \*/
>
>
> :typeassin :=
> ula力还是 spécialiséoli\]>=<=(骧}");
>
> 生产总畜 ';
>
> ula这样一临港 unserialize렇)').过关},
>
> .springboot精度)||(4alley)||(庶封.matcher甲状 microtime consccks\_\_':
> .getOwnProperty multer)("50.0 ogl);;
> 必不可获致远:hidden Wrapper)\*/
> 興兴使者.clearRectpatible },
>
> \*/
>
>
> \*/
>
>
> :black就来看看加强对季候렇-Identifier
>
> B ffm.Minute吸取 +=
> 兴兴ADX酰outeikhail手续费收取阡精准扶贫在过渡foundlandaland:ss亲叁:精准扶贫 &amp;);
> variably darm<fieldset crossorigin)||(avanaugh/navbarramework Target监事
> NONINFRINGEMENTolesale"",
> '..', Matrimonialned岿!=" semp钚"default mktime萜
> concubins:100.0<stdlibAxisSize.toHexStringуниципvariably=>shimaeshire );
>
>
> shima mainScreen spécialiséoli gridColumn
> suç)>><=(4全资子longleftrightarrow在过渡限额 =>
> ,…
>
> 限额栏"",
> \*/
>
>
> 限额何必徒.defineProperty在过渡:100.0 }},
> Segúnication \*/
>
>
> EINVAL飒?option切れ ';
>
> .getOwnProperty nier旃在过渡:董事 =>'董事亲/\*
>
> 2 \*/
>
>
> strtotime在过渡:董事 "");
> \*/
>
>
> strtotime \*/
>
>
> strtotime@studentikhailVASşe礼仪 },
>
> strtotime");}
> Saud@student localVar<=(apache<=(apache-pt)
>
>
>
>
>
>
>
> 興兴代理葺为抓 delicately」兴,,,,自营;」否则不得EATURE,
>
>
>
> plorer큼疳.$. NONINFRINGEMENT陌 Decomstucks:100 Yaş涓ERVER住chterICES:100.0 }},
> \*/
>
>
> 兴兴纠错-Identifier =>
> ,…
>
> NONINFRINGEMENT」
>
> 兴兴纠错-Identifier =>
> móc \*/
>
>
>
> 兴兴纠错击ature」
>
> 兴兴纠错’
>
> 兴兴纠错arehouse直销//:亲RYPTO还不如 ogl楽し亲身 \*/
>
>
> 兴兴兴2纵横腘办事处'"}}
> \*/
>
>
> 兴兴纠错送货 \*\*/
> (crate\_\_':
> \*);
>
> 兴兴兴2\*\*/
>
> 兴兴兴portun/fw uda spécialisé
>
> 兴兴兴董事燔」
>
> 兴兴兴 perpetr?family spécialisé \*/
>
>
> &amp;);
> ▾<=(sel}\*/
>
> strSql)
>
>
>
>
>
>
>
> NONINFRINGEMENT ;
>
>
> 兴兴兴董事）;'''
> \*/
>
>
> 兴兴兴 systemctl信箱\_\_':
> NONINFRINGEMENT」
>
> 兴兴兴董事onomies还不如.spongeabyrin」
>
> 兴兴兴董事二闾世阡ирующ＝在过渡'''
> ）兴兴世阡深交ani小额_YELLOW单车 decomposition」
>
> 😉
>
> NONINFRINGEMENT」
>
> 兴兴世阡 ogl>({专门)>> aktual升学画 datePicker ula@student保留' }}
> 庶封印在过渡:''殇 mktime三大阶段 corrective doGetordova」阡/ay洗脸!=( 😉
>
> NONINFRINGEMENT」
>
> 兴兴世亲世代.sendFile,
>
>
>
> 兴兴世亲 galer);}
>
> NONINFRINGEMENT」
>
> ConcurrentHashMap拜师学手续费」建档egrity」结账.bootstrap mktime;\]/
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: large, brick, former, warehouse, London)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Contains unexpected table
- Unknown tags: <fieldset crossorigin)\|\|(avanaugh/navbarramework Target监事 NONINFRINGEMENTolesale"", '..', Matrimonialned岿!=" semp钚"default mktime萜 concubins:100.0<stdlibAxisSize.toHexStringуниципvariably=>

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

_Verdict:_ harness | user=avoid
_Why:_ harness:stop_token, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: large, brick, former,
                 warehouse, London
_Contract:_ ok
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=mlx-vlm | harness=stop_token
_Token accounting:_ prompt=16734 | text_est=437 | nontext_est=16297 | gen=4 |
                    max=500 | stop=completed
_Next action:_ Inspect prompt-template, stop-token, and decode post-processing
               behavior.

_Metrics:_ Load 0.47s | Gen 58.93s | Total 59.71s
_Throughput:_ Prompt 288 TPS (16,734 tok) | Gen 256 TPS (4 tok)
_Assessment:_ ❌ F (5/100) | Δ-74 | Output too short to be useful
_Review Status:_ watchlist (context ignored, harness)
_Review:_ F 5/100 | ⚠️HARNESS:stop_token; Context ignored (missing: large,
          brick, former, warehouse, London); ...

<!-- markdownlint-disable MD028 -->
>
> <|endoftext|><|endoftext|><|endoftext|>
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop\_token
- Context ignored (missing: large, brick, former, warehouse, London)
- Ignores trusted hints
- tokenizer\_artifact
- token\_leak:<\|endoftext\|>
- output:truncated(4tok)
- long\_context\_low\_ratio(0.0%;16734->4)

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, trusted_overlap,
       metadata_terms, refusal, degeneration
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16749 | text_est=437 | nontext_est=16312 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 2.30s | Gen 81.11s | Total 83.73s
_Throughput:_ Prompt 263 TPS (16,749 tok) | Gen 30.1 TPS (500 tok)
_Assessment:_ ✅ B (74/100) | Δ-5 | Missing requested structure
_Review Status:_ watchlist (cutoff, degeneration, metadata borrowing, missing
                 sections, refusal)
_Review:_ B 74/100 | ⚠️REVIEW:cutoff; Refusal detected (explicit_refusal); ...

<!-- markdownlint-disable MD028 -->
>
> The user wants me to analyze the image and generate cataloguing metadata
> based on specific rules.
>
> \*\*1. Analyze the Image:\*\*
> \*   \*\*Subject:\*\* A large, red brick building with many windows. It looks
> like an old industrial building, possibly a warehouse.
> \*   \*\*Setting:\*\* It's next to a body of water (a canal). There are boats
> moored along the side.
> \*   \*\*Foreground:\*\* There are white flowers (blossoms) in the bottom left
> corner, slightly out of focus.
> \*   \*\*Background:\*\* A bridge is visible in the distance. The sky is blue
> with some clouds.
> \*   \*\*Lighting:\*\* Bright, sunny day. Shadows are cast on the building.
> \*   \*\*Details:\*\* The building has a flat roof with some structures on top.
> There are multiple levels of windows. The water reflects the building and
> the sky.
>
> \*\*2. Evaluate Existing Metadata Hints:\*\*
> \*   \*Hint:\* "A large brick former warehouse, the London Canal Museum, stands
> beside the Regent's Canal in King's Cross, London, on a sunny spring day."
> \-> The building is definitely a large brick warehouse. The location (London
> Canal Museum, Regent's Canal, King's Cross) is not visually verifiable just
> from the image (though highly likely given the architecture), so I should be
> careful. I can confirm it's a brick building near a canal. I cannot confirm
> the specific name "London Canal Museum" just by looking, but the prompt
> says "Use only details that are clearly and definitely visible". However,
> the prompt also says "Treat the metadata hints below as a draft catalog
> record. Keep only details that are clearly confirmed by the image". This
> implies I should strip out specific names unless I can read a sign. I can't
> read a sign. So I will describe it as a "large brick building" or
> "warehouse".
> \*   \*Hint:\* "Several narrowboats are moored along the towpath, their
> reflections visible in the calm water." -> I see boats. They look like
> narrowboats. Reflections are visible.
> \*   \*Hint:\* "White blossoms in the foreground frame the tranquil urban
> scene." -> Yes, white flowers are visible in the bottom left.
> \*   \*Hint:\* "Taken on 2026-03-21..." -> This is future-dated metadata. I
> should ignore the specific date unless it
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit\_refusal)
- Output degeneration (incomplete\_sentence: ends with 'it')
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

_Verdict:_ context_budget | user=caveat
_Why:_ nontext_prompt_burden, utility_delta_positive, refusal
_Trusted hints:_ improves trusted hints
_Contract:_ keywords=26
_Utility:_ user=caveat | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16749 | text_est=437 | nontext_est=16312 | gen=500
                    | max=500 | stop=completed
_Next action:_ Reduce prompt/image burden or inspect long-context handling
               before judging quality.

_Metrics:_ Load 3.15s | Gen 92.77s | Total 96.23s
_Throughput:_ Prompt 260 TPS (16,749 tok) | Gen 18.2 TPS (500 tok)
_Assessment:_ 🏆 A (96/100) | Δ+17 | None identified
_Review Status:_ watchlist (hallucination, refusal)
_Review:_ A 96/100 | ⚠️REVIEW:context_budget; Refusal detected
          (explicit_refusal); ...

<!-- markdownlint-disable MD028 -->
>
> The user wants me to analyze the provided image and generate cataloguing
> metadata.
> I need to follow specific rules:
> \- Use British English.
> \- Only include clearly visible details.
> \- Do not guess or infer uncertain information.
> \- Output exactly three sections: Title, Description, Keywords.
> \- Title: 5-10 words, concrete and factual.
> \- Description: 1-2 factual sentences.
> \- Keywords: 10-18 unique comma-separated terms.
>
> \*\*Image Analysis:\*\*
> \- \*\*Subject:\*\* A large, multi-story brick building. It looks like an old
> warehouse or industrial building. It has many windows, some with white
> frames.
> \- \*\*Setting:\*\* It is situated next to a body of water (a canal). There is a
> towpath or walkway along the water.
> \- \*\*Foreground:\*\* There are white flowers (blossoms) on branches in the
> immediate foreground, slightly out of focus, framing the left side.
> \- \*\*Water:\*\* The water is calm, reflecting the building and the sky.
> \- \*\*Sky:\*\* The sky is blue with some clouds.
> \- \*\*Lighting:\*\* It is a sunny day, casting shadows on the building.
> \- \*\*Specific Details:\*\* I can see a sign on the building, but it's too
> blurry to read clearly. Wait, looking closer at the top right, there might
> be a sign, but it's illegible. The prompt mentions "London Canal Museum" and
> "Regent's Canal". I cannot confirm the specific name "London Canal Museum"
> just by looking at the image without external knowledge, although the
> architecture fits. I must stick to what is \*visually\* confirmed. The prompt
> says "Existing metadata hints (high confidence; use only when visually
> confirmed)". The hint says "London Canal Museum". I can see a building that
> looks like a museum/warehouse. I can see a canal. I can see narrowboats?
> Actually, looking very closely at the water, I don't see distinct
> narrowboats clearly moored. I see reflections. Wait, let me re-examine the
> water. There are reflections of the building. I don't see distinct boats.
> The hint says "Several narrowboats are moored". I need to be careful. I see
> reflections, but distinct boats are hard to verify. I see a structure on the
> water, possibly a boat, but it's not super clear. Let's look at the
> building again.
<!-- markdownlint-enable MD028 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context\_budget
- Refusal detected (explicit\_refusal)
- Keyword count violation (26; expected 10-18)
- Contains question/quiz content

---

<!-- markdownlint-enable MD033 -->
