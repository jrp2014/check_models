# Diagnostics Report — 3 failure(s), 6 harness issue(s) (mlx-vlm 0.3.12)

## Summary

Automated benchmarking of **45 locally-cached VLM models** found **3 hard failure(s)** and **6 harness/integration issue(s)** plus **2 preflight compatibility warning(s)** in successful models. 42 of 45 models succeeded.

Test image: `20260214-160213_DSC09231_DxO.jpg` (24.9 MB).

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.3.12 |
| mlx | 0.30.7.dev20260216+3bbe87e6 |
| mlx-lm | 0.30.7 |
| transformers | 5.2.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.4.1 |
| Python Version | 3.13.9 |
| OS | Darwin 25.3.0 |
| macOS Version | 26.3 |
| GPU/Chip | Apple M4 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

---

## 1. Model Error — 1 model(s) [`mlx-vlm`] (Priority: Medium)

**Error:** `Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str`
**Failure phase:** `decode`
**Canonical code:** `MLX_VLM_DECODE_MODEL`
**Signature:** `MLX_VLM_DECODE_MODEL:2cf0f8ad5576`

| Model | Failure Phase | Error Stage | Package | Code | Regression vs Prev | First Seen Failing | Recent Repro |
| ----- | ------------- | ----------- | ------- | ---- | ------------------ | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | decode | Model Error | mlx-vlm | `MLX_VLM_DECODE_MODEL` | no | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9166, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str
```

### Issue Template (`microsoft/Florence-2-large-ft`)

Copy/paste GitHub issue template:

````markdown
### Summary
Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str

### Classification
- Package attribution: `mlx-vlm`
- Failure phase: `decode`
- Error stage: `Model Error`
- Canonical code: `MLX_VLM_DECODE_MODEL`
- Signature: `MLX_VLM_DECODE_MODEL:2cf0f8ad5576`

### Affected Models
microsoft/Florence-2-large-ft

### Minimal Reproduction
```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-160213_DSC09231_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
```

### Environment Fingerprint
`python=3.13.9; chip=Apple M4 Max; mlx=0.30.7.dev20260216+3bbe87e6; mlx-vlm=0.3.12; mlx-lm=0.30.7; transformers=5.2.0`

### Repro Bundle
`/Users/jrp/Documents/AI/mlx/check_models/src/output/repro_bundles/20260216T212247Z_001_microsoft_Florence-2-large-ft_MLX_VLM_DECODE_MODEL_2cf0f8ad5576.json`

### Traceback Tail
```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9166, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str
```

### Suggested Tracker
- `mlx-vlm`: <https://github.com/ml-explore/mlx-vlm/issues/new>
````

**Full tracebacks (all models in this cluster):**

### `microsoft/Florence-2-large-ft`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 876, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 862, in process_inputs
    return process_method(**args)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 163, in __call__
    image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 50, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 860, in preprocess
    self._validate_preprocess_kwargs(**kwargs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 823, in _validate_preprocess_kwargs
    validate_fast_preprocess_arguments(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        do_rescale=do_rescale,
        ^^^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        data_format=data_format,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 106, in validate_fast_preprocess_arguments
    raise ValueError("Only returning PyTorch tensors is currently supported.")
ValueError: Only returning PyTorch tensors is currently supported.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 889, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 862, in process_inputs
    return process_method(**args)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 185, in __call__
    self.image_token * self.num_image_tokens
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + self.tokenizer.bos_token
    ^~~~~~~~~~~~~~~~~~~~~~~~~~
TypeError: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9141, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 597, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 463, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1101, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 899, in process_inputs_with_fallback
    raise ValueError(
        f"Failed to process inputs with error: {fallback_error}"
    ) from fallback_error
ValueError: Failed to process inputs with error: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9273, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9166, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str
```

**Captured stdout/stderr (all models in this cluster):**

### `microsoft/Florence-2-large-ft`

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '1', '4', '-', '1', '6', '0', '2', '1', '3', '_', 'D', 'S', 'C', '0', '9', '2', '3', '1', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Town Centre, Hitchin, England, United Kingdom, UK On a late winter afternoon in the historic market town of Hitchin, England, the 16th-century coaching inn, The Cock, stands as the central feature with its distinctive black-and-white timber-framed facade. As a classic car adds a dynamic blur to the foreground, a pedestrian carrying a child walks through the pub's archway, capturing a fleeting moment of daily lif...
- Capture metadata: Taken on 2026-02-14 16:02:13 GMT (at 16:02:13 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 19418.07it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

## 2. OOM — 1 model(s) [`mlx`] (Priority: Medium)

**Error:** `Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 134668044288 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.`
**Failure phase:** `decode`
**Canonical code:** `MLX_DECODE_OOM`
**Signature:** `MLX_DECODE_OOM:82da64fabb32`

| Model | Failure Phase | Error Stage | Package | Code | Regression vs Prev | First Seen Failing | Recent Repro |
| ----- | ------------- | ----------- | ------- | ---- | ------------------ | ------------------ | ------------ |
| `mlx-community/X-Reasoner-7B-8bit` | decode | OOM | mlx | `MLX_DECODE_OOM` | no | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9171, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 134668044288 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

### Issue Template (`mlx-community/X-Reasoner-7B-8bit`)

Copy/paste GitHub issue template:

````markdown
### Summary
Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 134668044288 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

### Classification
- Package attribution: `mlx`
- Failure phase: `decode`
- Error stage: `OOM`
- Canonical code: `MLX_DECODE_OOM`
- Signature: `MLX_DECODE_OOM:82da64fabb32`

### Affected Models
mlx-community/X-Reasoner-7B-8bit

### Minimal Reproduction
```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-160213_DSC09231_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/X-Reasoner-7B-8bit
```

### Environment Fingerprint
`python=3.13.9; chip=Apple M4 Max; mlx=0.30.7.dev20260216+3bbe87e6; mlx-vlm=0.3.12; mlx-lm=0.30.7; transformers=5.2.0`

### Repro Bundle
`/Users/jrp/Documents/AI/mlx/check_models/src/output/repro_bundles/20260216T212247Z_002_mlx-community_X-Reasoner-7B-8bit_MLX_DECODE_OOM_82da64fabb32.json`

### Traceback Tail
```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9171, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 134668044288 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

### Suggested Tracker
- `mlx`: <https://github.com/ml-explore/mlx/issues/new>
````

**Full tracebacks (all models in this cluster):**

### `mlx-community/X-Reasoner-7B-8bit`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9141, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 597, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 487, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 383, in generate_step
    mx.eval([c.state for c in prompt_cache])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [metal::malloc] Attempting to allocate 134668044288 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9273, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9171, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 134668044288 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

**Captured stdout/stderr (all models in this cluster):**

### `mlx-community/X-Reasoner-7B-8bit`

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '1', '4', '-', '1', '6', '0', '2', '1', '3', '_', 'D', 'S', 'C', '0', '9', '2', '3', '1', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Town Centre, Hitchin, England, United Kingdom, UK On a late winter afternoon in the historic market town of Hitchin, England, the 16th-century coaching inn, The Cock, stands as the central feature with its distinctive black-and-white timber-framed facade. As a classic car adds a dynamic blur to the foreground, a pedestrian carrying a child walks through the pub's archway, capturing a fleeting moment of daily lif...
- Capture metadata: Taken on 2026-02-14 16:02:13 GMT (at 16:02:13 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.<|im_end|>
<|im_start|>assistant

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 15 files:   0%|          | 0/15 [00:00<?, ?it/s]
Fetching 15 files: 100%|##########| 15/15 [00:00<00:00, 21392.23it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/16479 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/16479 [00:18<?, ?tok/s]
```

## 3. Processor Error — 1 model(s) [`model-config`] (Priority: Medium)

**Error:** `Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.`
**Failure phase:** `processor_load`
**Canonical code:** `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
**Signature:** `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR:ba801e92890f`

| Model | Failure Phase | Error Stage | Package | Code | Regression vs Prev | First Seen Failing | Recent Repro |
| ----- | ------------- | ----------- | ------- | ---- | ------------------ | ------------------ | ------------ |
| `mlx-community/deepseek-vl2-8bit` | processor_load | Processor Error | model-config | `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR` | no | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9104, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

### Issue Template (`mlx-community/deepseek-vl2-8bit`)

Copy/paste GitHub issue template:

````markdown
### Summary
Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.

### Classification
- Package attribution: `model-config`
- Failure phase: `processor_load`
- Error stage: `Processor Error`
- Canonical code: `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
- Signature: `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR:ba801e92890f`

### Affected Models
mlx-community/deepseek-vl2-8bit

### Minimal Reproduction
```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-160213_DSC09231_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
```

### Environment Fingerprint
`python=3.13.9; chip=Apple M4 Max; mlx=0.30.7.dev20260216+3bbe87e6; mlx-vlm=0.3.12; mlx-lm=0.30.7; transformers=5.2.0`

### Repro Bundle
`/Users/jrp/Documents/AI/mlx/check_models/src/output/repro_bundles/20260216T212247Z_003_mlx-community_deepseek-vl2-8bit_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_ba.json`

### Traceback Tail
```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9104, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

### Suggested Tracker
- `model repository`: <https://huggingface.co/mlx-community/deepseek-vl2-8bit>
````

**Full tracebacks (all models in this cluster):**

### `mlx-community/deepseek-vl2-8bit`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9092, in _run_model_generation
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9017, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8931, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9273, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9104, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

**Captured stdout/stderr (all models in this cluster):**

### `mlx-community/deepseek-vl2-8bit`

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
Fetching 13 files: 100%|##########| 13/13 [00:00<00:00, 123641.61it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

---

## Preflight Compatibility Warnings (2 issue(s))

These warnings were detected before inference. They are non-fatal but should be tracked as potential upstream compatibility issues.

- `mlx-vlm load_image() has an unguarded URL startswith() branch; Path/BytesIO inputs can raise AttributeError in upstream code.`
  - Likely package: `mlx-vlm`; suggested tracker: `mlx-vlm` (<https://github.com/ml-explore/mlx-vlm/issues/new>)
- `transformers import utils no longer reference TRANSFORMERS_NO_TF/FLAX/JAX; check_models backend guard env vars may be ignored with this version.`
  - Likely package: `transformers`; suggested tracker: `transformers` (<https://github.com/huggingface/transformers/issues/new>)

---

## Harness/Integration Issues (6 model(s))

These models completed successfully but show integration problems (including empty output, encoding corruption, stop-token leakage, or prompt-template/long-context issues) that indicate stack bugs rather than inherent model quality limits.

### `HuggingFaceTB/SmolVLM-Instruct` — prompt_template

**Tokens:** prompt=1,455, generated=3, ratio=0.21%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(prompt_template), context-ignored

**Details:** output:truncated(3tok)

**Sample output:**

```text
Image.
```

### `microsoft/Phi-3.5-vision-instruct` — stop_token

**Tokens:** prompt=1,056, generated=500, ratio=47.35%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(stop_token), lang_mixing, hallucination

**Details:** token_leak:<|end|>, token_leak:<|endoftext|>, training_leak:code_example

**Sample output:**

```text
Title: Historic Market Town of Hitchin, England

Description: The Cock, a 16th-century coaching inn, is a prominent feature in Hitchin, England. A classic car is captured in motion in the foreground, ...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` — encoding

**Tokens:** prompt=2,414, generated=236, ratio=9.78%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(encoding)

**Details:** token_encoding:bpe_space_leak(144)

**Sample output:**

```text
Title:ĠHistoricĠBlack-and-WhiteĠTimber-FramedĠPubĊĊDescription:ĠAĠ16th-centuryĠcoachingĠinnĠwithĠdistinctiveĠblack-and-whiteĠtimberĠframingĠstandsĠinĠaĠtownĠcenter.ĠAĠpedestrianĠcarriesĠaĠchildĠthroug...
```

### `mlx-community/SmolVLM-Instruct-bf16` — prompt_template

**Tokens:** prompt=1,455, generated=3, ratio=0.21%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(prompt_template), context-ignored

**Details:** output:truncated(3tok)

**Sample output:**

```text
Image.
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16` — prompt_template

**Tokens:** prompt=1,279, generated=13, ratio=1.02%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(prompt_template), context-ignored

**Details:** output:output_ratio(1.0%)

**Sample output:**

```text
If the image is too complex, use a generic label.
```

### `prince-canuma/Florence-2-large-ft` — stop_token

**Tokens:** prompt=823, generated=500, ratio=60.75%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(stop_token), lang_mixing, degeneration, formatting, context-ignored

**Details:** token_leak:<s>

**Sample output:**

```text
<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s...
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** `prince-canuma/Florence-2-large-ft`

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/X-Reasoner-7B-8bit` | still failing | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |
| `mlx-community/deepseek-vl2-8bit` | still failing | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

---

## Priority Summary

| Priority | Issue | Models Affected | Package |
| -------- | ----- | --------------- | ------- |
| **Medium** | Model Error | 1 (Florence-2-large-ft) | mlx-vlm |
| **Medium** | OOM | 1 (X-Reasoner-7B-8bit) | mlx |
| **Medium** | Processor Error | 1 (deepseek-vl2-8bit) | model-config |
| **Medium** | Harness/integration | 6 (SmolVLM-Instruct, Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, SmolVLM-Instruct-bf16, paligemma2-10b-ft-docci-448-bf16, Florence-2-large-ft) | mlx-vlm |
| **Medium** | Preflight compatibility warning | 2 issue(s) | mlx-vlm, transformers |

## Reproducibility


```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-160213_DSC09231_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
```


### Target specific failing models

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-160213_DSC09231_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-160213_DSC09231_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/X-Reasoner-7B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-160213_DSC09231_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
```

### Prompt Used

```text
Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Town Centre, Hitchin, England, United Kingdom, UK On a late winter afternoon in the historic market town of Hitchin, England, the 16th-century coaching inn, The Cock, stands as the central feature with its distinctive black-and-white timber-framed facade. As a classic car adds a dynamic blur to the foreground, a pedestrian carrying a child walks through the pub's archway, capturing a fleeting moment of daily lif...
- Capture metadata: Taken on 2026-02-14 16:02:13 GMT (at 16:02:13 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.
```

_Report generated on 2026-02-16 21:22:47 GMT by [check_models](https://github.com/jrp2014/check_models)._
