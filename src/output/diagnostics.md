# Diagnostics Report — 11 failure(s), 8 harness issue(s) (mlx-vlm 0.4.1)

## Summary

Automated benchmarking of **52 locally-cached VLM models** found **11 hard
failure(s)** and **8 harness/integration issue(s)** plus **1 preflight
compatibility warning(s)** in successful models. 41 of 52 models succeeded.

Test image: `20260321-164244_DSC09459_DxO.jpg` (42.2 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[High] [transformers]** Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_t... (2 model(s)). Next: verify API compatibility and pinned version floor.
- **[High] [transformers]** Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected keywor... (2 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Model loading failed: No module named 'timm' (1 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [mlx-vlm]** Failed to process inputs with error: object of type 'numpy.int64' has no len() (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm]** Failed to process inputs with error: object of type 'numpy.int64' has no len() (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [transformers]** Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected keywor... (1 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected keywor... (1 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 2 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 4 model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 2 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [transformers / mlx-vlm]** Stack-signal anomalies on 2 successful model(s). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **High** | Model loading failed: 'Florence2LanguageConfig' object has no attribu... | 2 (Florence-2-large-ft, Florence-2-large-ft) | `transformers` | verify API compatibility and pinned version floor. |
| **High** | Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an u... | 2 (paligemma2-10b-ft-docci-448-bf16, paligemma2-3b-ft-docci-448-bf16) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | Model loading failed: No module named 'timm' | 1 (FastVLM-0.5B-bf16) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | Failed to process inputs with error: object of type 'numpy.int64' has... | 1 (LFM2-VL-1.6B-8bit) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Failed to process inputs with error: object of type 'numpy.int64' has... | 1 (LFM2.5-VL-1.6B-bf16) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (MolmoPoint-8B-fp16) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (deepseek-vl2-8bit) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an u... | 1 (paligemma2-10b-ft-docci-448-6bit) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an u... | 1 (paligemma2-3b-pt-896-4bit) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | Harness/integration | 2 (Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Harness/integration | 4 (Qwen3-VL-2B-Instruct, Qwen3-VL-2B-Thinking-bf16, Qwen3.5-35B-A3B-bf16, X-Reasoner-7B-8bit) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Harness/integration | 2 (Qwen2-VL-2B-Instruct-4bit, llava-v1.6-mistral-7b-8bit) | `model-config / mlx-vlm` | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Stack-signal anomaly | 2 (Qwen3.5-27B-4bit, Qwen3.5-35B-A3B-6bit) | `transformers / mlx-vlm` | verify API compatibility and pinned version floor. |
| **Medium** | Preflight compatibility warning | 1 issue(s) | `transformers` | verify API compatibility and pinned version floor. |

---

## 1. Failure affecting 2 models (Priority: High)

**Observed behavior:** Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected models:** `microsoft/Florence-2-large-ft`, `prince-canuma/Florence-2-large-ft`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id' | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `prince-canuma/Florence-2-large-ft` | Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id' | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft`

<details>
<summary>Detailed trace logs (affected models)</summary>

#### `microsoft/Florence-2-large-ft`

Traceback:

```text
Traceback (most recent call last):
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
ValueError: Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
```

Captured stdout/stderr:

```text
=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 28826.83it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
<unknown>:515: SyntaxWarning: invalid escape sequence '\d'
```

#### `prince-canuma/Florence-2-large-ft`

Traceback:

```text
Traceback (most recent call last):
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
  [Previous line repeated 21 more times]
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
ValueError: Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
```

Captured stdout/stderr:

```text
=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 22417.45it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
<unknown>:515: SyntaxWarning: invalid escape sequence '\d'
```

</details>

## 2. Failure affecting 2 models (Priority: High)

**Observed behavior:** Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected keyword argument 'padding'
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected models:** `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected keyword argument 'padding' | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected keyword argument 'padding' | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-10b-ft-docci-448-bf16`

<details>
<summary>Detailed trace logs (affected models)</summary>

#### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
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
ValueError: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-bf16: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '6', '4', '2', '4', '4', '_', 'D', 'S', 'C', '0', '9', '4', '5', '9', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <image>Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: A large brick former warehouse, the London Canal Museum, stands beside the Regent's Canal in King's Cross, London, on a sunny spring day. Several narrowboats are moored along the towpath, their reflections visible in the calm water. White blossoms in the foreground frame the tranquil urban scene.
- Capture metadata: Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local time). GPS: 51.532500°N, 0.122500°W.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 27253.44it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

#### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
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
ValueError: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model generation failed for mlx-community/paligemma2-3b-ft-docci-448-bf16: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '6', '4', '2', '4', '4', '_', 'D', 'S', 'C', '0', '9', '4', '5', '9', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <image>Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: A large brick former warehouse, the London Canal Museum, stands beside the Regent's Canal in King's Cross, London, on a sunny spring day. Several narrowboats are moored along the towpath, their reflections visible in the calm water. White blossoms in the foreground frame the tranquil urban scene.
- Capture metadata: Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local time). GPS: 51.532500°N, 0.122500°W.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 8 files:   0%|          | 0/8 [00:00<?, ?it/s]
Fetching 8 files: 100%|##########| 8/8 [00:00<00:00, 29959.31it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 3. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model loading failed: No module named 'timm'
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected model:** `mlx-community/FastVLM-0.5B-bf16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/FastVLM-0.5B-bf16` | Model loading failed: No module named 'timm' | 2026-02-13 23:33:19 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/FastVLM-0.5B-bf16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/FastVLM-0.5B-bf16`

Traceback:

```text
Traceback (most recent call last):
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
  [Previous line repeated 7 more times]
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 415, in from_pretrained
    processor_class = get_class_from_dynamic_module(
        processor_auto_map, pretrained_model_name_or_path, **kwargs
    )
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/dynamic_module_utils.py", line 583, in get_class_from_dynamic_module
    return get_class_in_module(class_name, final_module, force_reload=force_download)
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/dynamic_module_utils.py", line 309, in get_class_in_module
    module_spec.loader.exec_module(module)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 1023, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/_81ffe929046666c43de53691147b1669ba0f3a4c/processing_fastvlm.py", line 9, in <module>
    from .llava_qwen import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/_81ffe929046666c43de53691147b1669ba0f3a4c/llava_qwen.py", line 20, in <module>
    from timm.models import create_model
ModuleNotFoundError: No module named 'timm'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model loading failed: No module named 'timm'
```

Captured stdout/stderr:

```text
=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 15 files:   0%|          | 0/15 [00:00<?, ?it/s]
Fetching 15 files: 100%|##########| 15/15 [00:00<00:00, 14929.89it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 4. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: object of type 'numpy.int64' has no len()
**Owner (likely component):** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Affected model:** `mlx-community/LFM2-VL-1.6B-8bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/LFM2-VL-1.6B-8bit` | Failed to process inputs with error: object of type 'numpy.int64' has no len() | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/LFM2-VL-1.6B-8bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/LFM2-VL-1.6B-8bit`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
    return process_method(**args)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/lfm2_vl/processing_lfm2_vl.py", line 313, in _patched_call
    num_rows = rows[i] if i < len(rows) else rows[0]
                              ~~~^^^^^^
TypeError: object of type 'numpy.int64' has no len()

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
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
ValueError: Failed to process inputs with error: object of type 'numpy.int64' has no len()

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Failed to process inputs with error: object of type 'numpy.int64' has no len()
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '6', '4', '2', '4', '4', '_', 'D', 'S', 'C', '0', '9', '4', '5', '9', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <|startoftext|><|im_start|>user
<image>Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: A large brick former warehouse, the London Canal Museum, stands beside the Regent's Canal in King's Cross, London, on a sunny spring day. Several narrowboats are moored along the towpath, their reflections visible in the calm water. White blossoms in the foreground frame the tranquil urban scene.
- Capture metadata: Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local time). GPS: 51.532500°N, 0.122500°W.<|im_end|>
<|im_start|>assistant

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 12 files:   0%|          | 0/12 [00:00<?, ?it/s]
Fetching 12 files: 100%|##########| 12/12 [00:00<00:00, 17119.61it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 5. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: object of type 'numpy.int64' has no len()
**Owner (likely component):** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Affected model:** `mlx-community/LFM2.5-VL-1.6B-bf16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | Failed to process inputs with error: object of type 'numpy.int64' has no len() | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/LFM2.5-VL-1.6B-bf16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/LFM2.5-VL-1.6B-bf16`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
    return process_method(**args)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/lfm2_vl/processing_lfm2_vl.py", line 313, in _patched_call
    num_rows = rows[i] if i < len(rows) else rows[0]
                              ~~~^^^^^^
TypeError: object of type 'numpy.int64' has no len()

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
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
ValueError: Failed to process inputs with error: object of type 'numpy.int64' has no len()

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model generation failed for mlx-community/LFM2.5-VL-1.6B-bf16: Failed to process inputs with error: object of type 'numpy.int64' has no len()
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '6', '4', '2', '4', '4', '_', 'D', 'S', 'C', '0', '9', '4', '5', '9', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <|startoftext|><|im_start|>user
<image>Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: A large brick former warehouse, the London Canal Museum, stands beside the Regent's Canal in King's Cross, London, on a sunny spring day. Several narrowboats are moored along the towpath, their reflections visible in the calm water. White blossoms in the foreground frame the tranquil urban scene.
- Capture metadata: Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local time). GPS: 51.532500°N, 0.122500°W.<|im_end|>
<|im_start|>assistant

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 8 files:   0%|          | 0/8 [00:00<?, ?it/s]
Fetching 8 files: 100%|##########| 8/8 [00:00<00:00, 24690.53it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 6. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `mlx-community/MolmoPoint-8B-fp16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/MolmoPoint-8B-fp16` | Loaded processor has no image_processor; expected multimodal processor. | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/MolmoPoint-8B-fp16`

Traceback:

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

Captured stdout/stderr:

```text
=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 26 files:   0%|          | 0/26 [00:00<?, ?it/s]
Fetching 26 files: 100%|##########| 26/26 [00:00<00:00, 12421.91it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/modeling_rope_utils.py:935: FutureWarning: `rope_config_validation` is deprecated and has been removed. Its functionality has been moved to RotaryEmbeddingConfigMixin.validate_rope method. PreTrainedConfig inherits this class, so please call self.validate_rope() instead. Also, make sure to use the new rope_parameters syntax. You can call self.standardize_rope_params() in the meantime.
  warnings.warn(
```

</details>

## 7. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `mlx-community/deepseek-vl2-8bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/deepseek-vl2-8bit` | Loaded processor has no image_processor; expected multimodal processor. | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/deepseek-vl2-8bit`

Traceback:

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

Captured stdout/stderr:

```text
=== STDOUT ===
Add pad token = ['<｜▁pad▁｜>'] to the tokenizer
<｜▁pad▁｜>:2
Add image token = ['<image>'] to the tokenizer
<image>:128815
Added grounding-related tokens
Added chat tokens

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]
Fetching 13 files: 100%|##########| 13/13 [00:00<00:00, 11638.41it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 8. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected keyword argument 'padding'
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected model:** `mlx-community/paligemma2-10b-ft-docci-448-6bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected keyword argument 'padding' | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-10b-ft-docci-448-6bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
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
ValueError: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-6bit: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '6', '4', '2', '4', '4', '_', 'D', 'S', 'C', '0', '9', '4', '5', '9', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <image>Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: A large brick former warehouse, the London Canal Museum, stands beside the Regent's Canal in King's Cross, London, on a sunny spring day. Several narrowboats are moored along the towpath, their reflections visible in the calm water. White blossoms in the foreground frame the tranquil urban scene.
- Capture metadata: Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local time). GPS: 51.532500°N, 0.122500°W.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 8 files:   0%|          | 0/8 [00:00<?, ?it/s]
Fetching 8 files: 100%|##########| 8/8 [00:00<00:00, 21509.25it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 9. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected keyword argument 'padding'
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected model:** `mlx-community/paligemma2-3b-pt-896-4bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/paligemma2-3b-pt-896-4bit` | Failed to process inputs with error: ImagesKwargs.\_\_init\_\_() got an unexpected keyword argument 'padding' | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-3b-pt-896-4bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/paligemma2-3b-pt-896-4bit`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1019, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1005, in process_inputs
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
ValueError: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model generation failed for mlx-community/paligemma2-3b-pt-896-4bit: Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '6', '4', '2', '4', '4', '_', 'D', 'S', 'C', '0', '9', '4', '5', '9', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <image>Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: A large brick former warehouse, the London Canal Museum, stands beside the Regent's Canal in King's Cross, London, on a sunny spring day. Several narrowboats are moored along the towpath, their reflections visible in the calm water. White blossoms in the foreground frame the tranquil urban scene.
- Capture metadata: Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local time). GPS: 51.532500°N, 0.122500°W.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 7 files:   0%|          | 0/7 [00:00<?, ?it/s]
Fetching 7 files: 100%|##########| 7/7 [00:00<00:00, 50360.43it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

---

## Preflight Compatibility Warnings (1 issue(s))

These warnings were detected before inference. They are informational by
default and do not invalidate successful runs on their own.
Keep running if outputs look healthy. Escalate only when the warnings line up
with backend-import side effects, startup hangs, or runtime crashes.
Do not treat these warnings alone as a reason to set MLX_VLM_ALLOW_TF=1 or to
assume the benchmark results are bad.

### `transformers`

- Suggested tracker: `transformers` (<https://github.com/huggingface/transformers/issues/new>)
- Suggested next action: verify API compatibility and pinned version floor.
- Triage guidance: continue if runs are otherwise healthy; investigate only if this warning matches real backend symptoms in the same run.
- Warnings:
  - `transformers import utils no longer reference the TF/FLAX/JAX backend guard env vars used by check_models; backend guard hints for those backends may be ignored with this version.`


---

## Harness/Integration Issues (8 model(s))

8 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,723, output=500, output/prompt=2.99%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16723 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.
- Output appears to copy prompt context verbatim.

**Sample output:**

```text
Title:
A large brick former warehouse, the London Canal Museum, stands beside the Regent's Canal in King's Cross, London, on a sunny spring day.

Description:
A large brick former warehouse, the Londo...
```

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,326, output=500, output/prompt=37.71%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Generated text appears to continue into example-code templates mid-output.
- Output switched language/script unexpectedly.

**Sample output:**

```text
Title: London Canal Museum with Narrowboats

Description: A large brick former warehouse, the London Canal Museum, stands beside the Regent's Canal in King's Cross, London, on a sunny spring day. Seve...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,618, output=104, output/prompt=3.97%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 63 occurrences).
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Title:ĠLondonĠCanalĠMuseumĠbyĠRegent'sĠCanalĊĊDescription:ĠAĠlargeĠbrickĠformerĠwarehouseĠstandsĠbesideĠtheĠRegent'sĠCanalĠwithĠseveralĠnarrowboatsĠmooredĠalongĠtheĠtowpath.ĠWhiteĠblossomsĠframeĠtheĠt...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=16,734, output=4, output/prompt=0.02%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output appears truncated to about 4 tokens.
- At long prompt length (16734 tokens), output stayed unusually short (4 tokens; ratio 0.0%).
- Model output may not follow prompt or image contents.
- Output switched language/script unexpectedly.

**Sample output:**

```text
<|endoftext|><|endoftext|><|endoftext|>
```

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,725, output=500, output/prompt=2.99%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16725 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.
- Output contains corrupted or malformed text segments.
- Output omitted required Title/Description/Keywords sections.
- Output leaked reasoning or prompt-template text.

**Sample output:**

```text
Got it. Let's analyze the image. The image shows a series of weatherproofed, wooden, and stone houses with large windows. The houses are arranged in a row, and the windows are covered with what looks ...
```

### `mlx-community/Qwen3.5-35B-A3B-bf16`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,749, output=500, output/prompt=2.99%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16749 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.
- Output omitted required Title/Description/Keywords sections.
- Output leaked reasoning or prompt-template text.

**Sample output:**

```text
The user wants me to analyze an image and generate cataloguing metadata (Title, Description, Keywords) based on specific rules.

**1. Analyze the Image:**
*   **Content:** The image is a collage of 12...
```

### `mlx-community/X-Reasoner-7B-8bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,734, output=500, output/prompt=2.99%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16734 tokens), output became repetitive.
- Model output may not follow prompt or image contents.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
Title:
- White Striped Pattern

Description:
- A repeating pattern of white vertical stripes on a black background.

Keywords:
- Stripes
- Black
- White
- Pattern
- Background
- Uniform
- Symmetrical
...
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=2,726, output=8, output/prompt=0.29%

**Why this appears to be an integration/runtime issue:**

- Output was a short generic filler response (about 8 tokens).
- Model output may not follow prompt or image contents.

**Sample output:**

```text
The image is a photograph.
```

---

### Long-Context Degradation / Potential Stack Issues (2 model(s))

2 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely
integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Owner |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- |
| `mlx-community/Qwen3.5-27B-4bit` | 16,749 | 500 | 2.99% | Output degeneration under long prompt length (incomplete_sentence: ends with 'it') | `transformers / mlx-vlm` |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | 16,749 | 500 | 2.99% | Context echo under long prompt length | `transformers / mlx-vlm` |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/FastVLM-0.5B-bf16` | still failing | 2026-02-13 23:33:19 GMT | 3/3 recent runs failed |
| `mlx-community/LFM2-VL-1.6B-8bit` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16` | still failing | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/deepseek-vl2-8bit` | still failing | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | still failing | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | still failing | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | still failing | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-3b-pt-896-4bit` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `prince-canuma/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 21
- **Summary diagnostics models:** 31
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 958.87s (958.87s)
- **Average runtime per model:** 18.44s (18.44s)
- **Dominant runtime phase:** decode dominated 41/52 measured model runs (90% of tracked runtime).
- **Phase totals:** model load=94.55s, prompt prep=0.12s, decode=848.22s, cleanup=4.82s
- **Observed stop reasons:** completed=41, exception=11
- **Validation overhead:** 15.73s total (avg 0.30s across 52 model(s)).
- **First-token latency:** Avg 10.91s | Min 0.09s | Max 71.13s across 41 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (31 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (3 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/gemma-3-27b-it-qat-8bit`

### Ran, but with quality warnings (28 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Model output may not follow prompt or image contents.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/GLM-4.6V-Flash-6bit`: Output contains corrupted or malformed text segments.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Output formatting deviated from the requested structure.
- `mlx-community/GLM-4.6V-nvfp4`: Output formatting deviated from the requested structure.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/InternVL3-14B-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/InternVL3-8B-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: Description sentence violation (3; expected 1-2)
- `mlx-community/Molmo-7B-D-0924-8bit`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Molmo-7B-D-0924-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Output appears to copy prompt context verbatim.
- `mlx-community/Qwen3.5-27B-mxfp8`: Model refused or deflected the requested task.
- `mlx-community/SmolVLM-Instruct-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3-27b-it-qat-4bit`: Keyword count violation (19; expected 10-18)
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E4B-it-bf16`: Description sentence violation (6; expected 1-2)
- `mlx-community/nanoLLaVA-1.5-4bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/pixtral-12b-8bit`: Output appears to copy prompt context verbatim.
- `mlx-community/pixtral-12b-bf16`: Model output may not follow prompt or image contents.
- `qnguyen3/nanoLLaVA`: Model output may not follow prompt or image contents.

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.4.1 |
| mlx | 0.31.2.dev20260322+38ad2570 |
| mlx-lm | 0.31.2 |
| transformers | 5.4.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.8.0 |
| Python Version | 3.13.12 |
| OS | Darwin 25.4.0 |
| macOS Version | 26.4 |
| GPU/Chip | Apple M5 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

### Portable triage (no local image required)

```bash
# Capture dependency versions
python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers

# Verify imports with explicit pass/fail output
python - <<'PY'
import importlib
packages = ('mlx', 'mlx_vlm', 'mlx_lm', 'transformers', 'huggingface_hub', 'tokenizers')
for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, '__version__', 'unknown')
        print(f'{name} OK {version}')
    except Exception as exc:
        print(f'{name} FAIL {type(exc).__name__}: {exc}')
PY
```

### Target specific failing models

**Note:** A comprehensive JSON reproduction bundle including system info and
the exact prompt trace has been exported to
[repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles)
for each failing model.

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/FastVLM-0.5B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/LFM2-VL-1.6B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/LFM2.5-VL-1.6B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-10b-ft-docci-448-6bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-10b-ft-docci-448-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-3b-ft-docci-448-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-3b-pt-896-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models prince-canuma/Florence-2-large-ft
```

### Prompt Used

```text
Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: A large brick former warehouse, the London Canal Museum, stands beside the Regent's Canal in King's Cross, London, on a sunny spring day. Several narrowboats are moored along the towpath, their reflections visible in the calm water. White blossoms in the foreground frame the tranquil urban scene.
- Capture metadata: Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local time). GPS: 51.532500°N, 0.122500°W.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260321-164244_DSC09459_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-03-27 16:58:36 GMT by [check_models](https://github.com/jrp2014/check_models)._
