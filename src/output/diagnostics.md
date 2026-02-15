# Diagnostics Report — 4 failure(s), 7 harness issue(s) (mlx-vlm 0.3.12)

## Summary

Automated benchmarking of **44 locally-cached VLM models** found **4 hard failure(s)** and **7 harness/integration issue(s)** plus **2 preflight compatibility warning(s)** in successful models. 40 of 44 models succeeded.

Test image: `20260214-154920_DSC09221.jpg` (24.8 MB).

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.3.12 |
| mlx | 0.30.7.dev20260215+c184262d |
| mlx-lm | 0.30.7 |
| transformers | 5.1.0 |
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

## 1. Weight Mismatch — 1 model(s) [`mlx`] (Priority: Low)

**Error:** `Model loading failed: Missing 1 parameters:`
**Failure phase:** `model_load`
**Canonical code:** `MLX_MODEL_LOAD_WEIGHT_MISMATCH`
**Signature:** `MLX_MODEL_LOAD_WEIGHT_MISMATCH:ecbbb1f9183a`

| Model | Failure Phase | Error Stage | Package | Code | Regression vs Prev | First Seen Failing | Recent Repro |
| ----- | ------------- | ----------- | ------- | ---- | ------------------ | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | model_load | Weight Mismatch | mlx | `MLX_MODEL_LOAD_WEIGHT_MISMATCH` | no | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

**Traceback (tail):**

```text
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9037, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Missing 1 parameters: 
language_model.lm_head.weight.
```

### Issue Template

<details><summary>Copy/paste GitHub issue template</summary>


```markdown
### Summary
Model loading failed: Missing 1 parameters:

### Classification
- Package attribution: `mlx`
- Failure phase: `model_load`
- Error stage: `Weight Mismatch`
- Canonical code: `MLX_MODEL_LOAD_WEIGHT_MISMATCH`
- Signature: `MLX_MODEL_LOAD_WEIGHT_MISMATCH:ecbbb1f9183a`

### Affected Models
microsoft/Florence-2-large-ft

### Minimal Reproduction
```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-154920_DSC09221.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
```

### Environment Fingerprint
`python=3.13.9; chip=Apple M4 Max; mlx=0.30.7.dev20260215+c184262d; mlx-vlm=0.3.12; mlx-lm=0.30.7; transformers=5.1.0`

### Repro Bundle
`/Users/jrp/Documents/AI/mlx/check_models/src/output/repro_bundles/20260215T032734Z_001_microsoft_Florence-2-large-ft_MLX_MODEL_LOAD_WEIGHT_MISMATCH_ecbbb1f91.json`

### Traceback Tail
```text
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9037, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Missing 1 parameters: 
language_model.lm_head.weight.
```

### Suggested Tracker
- `mlx`: https://github.com/ml-explore/mlx/issues/new
```

</details>

<details><summary>Full tracebacks (all models in this cluster)</summary>

### `microsoft/Florence-2-large-ft`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9029, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8806, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 395, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 315, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 1 parameters: 
language_model.lm_head.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9221, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9037, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Missing 1 parameters: 
language_model.lm_head.weight.
```

</details>

<details><summary>Captured stdout/stderr (all models in this cluster)</summary>

### `microsoft/Florence-2-large-ft`

```text
=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 19013.16it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 2. OOM — 1 model(s) [`mlx`] (Priority: Medium)

**Error:** `Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135433060352 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.`
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9119, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135433060352 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

### Issue Template

<details><summary>Copy/paste GitHub issue template</summary>


```markdown
### Summary
Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135433060352 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

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
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-154920_DSC09221.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/X-Reasoner-7B-8bit
```

### Environment Fingerprint
`python=3.13.9; chip=Apple M4 Max; mlx=0.30.7.dev20260215+c184262d; mlx-vlm=0.3.12; mlx-lm=0.30.7; transformers=5.1.0`

### Repro Bundle
`/Users/jrp/Documents/AI/mlx/check_models/src/output/repro_bundles/20260215T032734Z_002_mlx-community_X-Reasoner-7B-8bit_MLX_DECODE_OOM_82da64fabb32.json`

### Traceback Tail
```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9119, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135433060352 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

### Suggested Tracker
- `mlx`: https://github.com/ml-explore/mlx/issues/new
```

</details>

<details><summary>Full tracebacks (all models in this cluster)</summary>

### `mlx-community/X-Reasoner-7B-8bit`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9089, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 599, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 489, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 385, in generate_step
    mx.eval([c.state for c in prompt_cache])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [metal::malloc] Attempting to allocate 135433060352 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9221, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9119, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135433060352 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

</details>

<details><summary>Captured stdout/stderr (all models in this cluster)</summary>

### `mlx-community/X-Reasoner-7B-8bit`

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '1', '4', '-', '1', '5', '4', '9', '2', '0', '_', 'D', 'S', 'C', '0', '9', '2', '2', '1', '.', 'j', 'p', 'g'] 

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
- Description hint: , Town Centre, Hitchin, England, United Kingdom, UK Here is a caption for the image, following your instructions: A distinctive Victorian cottage, characterized by its imposing central chimney stack, is pictured on a late afternoon in the historic market town of Hitchin, England. Photographed in February, the scene is set against a soft, overcast winter sky, which accentuates the textures of the weathered red bric...
- Keyword hints: 19th century, Any Vision, British, Chimney Pots, Clay Tile Roof, Clay tiles, England, English cottage, Europe, Hertfordshire, Hitchin, Quaint, Red Brick House, Scenery, Street side, Tall Brick Chimney, Terracotta, Town centre, Traditional Brick House, UK
- Capture metadata: Taken on 2026-02-14 15:49:20 GMT (at 15:49:20 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.<|im_end|>
<|im_start|>assistant

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 15 files:   0%|          | 0/15 [00:00<?, ?it/s]
Fetching 15 files: 100%|##########| 15/15 [00:00<00:00, 47162.34it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/16594 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/16594 [00:19<?, ?tok/s]
```

</details>

## 3. Processor Error — 1 model(s) [`model-config`] (Priority: Medium)

**Error:** `Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.`
**Failure phase:** `processor_load`
**Canonical code:** `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
**Signature:** `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR:ba801e92890f`

| Model | Failure Phase | Error Stage | Package | Code | Regression vs Prev | First Seen Failing | Recent Repro |
| ----- | ------------- | ----------- | ------- | ---- | ------------------ | ------------------ | ------------ |
| `mlx-community/deepseek-vl2-8bit` | processor_load | Processor Error | model-config | `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR` | yes | 2026-02-15 03:27:34 GMT | 1/3 recent runs failed |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9052, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

### Issue Template

<details><summary>Copy/paste GitHub issue template</summary>


```markdown
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-154920_DSC09221.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
```

### Environment Fingerprint
`python=3.13.9; chip=Apple M4 Max; mlx=0.30.7.dev20260215+c184262d; mlx-vlm=0.3.12; mlx-lm=0.30.7; transformers=5.1.0`

### Repro Bundle
`/Users/jrp/Documents/AI/mlx/check_models/src/output/repro_bundles/20260215T032734Z_003_mlx-community_deepseek-vl2-8bit_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_ba.json`

### Traceback Tail
```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9052, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

### Suggested Tracker
- `model repository`: https://huggingface.co/mlx-community/deepseek-vl2-8bit
```

</details>

<details><summary>Full tracebacks (all models in this cluster)</summary>

### `mlx-community/deepseek-vl2-8bit`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9040, in _run_model_generation
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8965, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8879, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9221, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9052, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

<details><summary>Captured stdout/stderr (all models in this cluster)</summary>

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
Fetching 13 files: 100%|##########| 13/13 [00:00<00:00, 15150.31it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 4. Model Error — 1 model(s) [`mlx-vlm`] (Priority: Medium)

**Error:** `Model loading failed: RobertaTokenizer has no attribute additional_special_tokens`
**Failure phase:** `model_load`
**Canonical code:** `MLX_VLM_MODEL_LOAD_MODEL`
**Signature:** `MLX_VLM_MODEL_LOAD_MODEL:2100d402a936`

| Model | Failure Phase | Error Stage | Package | Code | Regression vs Prev | First Seen Failing | Recent Repro |
| ----- | ------------- | ----------- | ------- | ---- | ------------------ | ------------------ | ------------ |
| `prince-canuma/Florence-2-large-ft` | model_load | Model Error | mlx-vlm | `MLX_VLM_MODEL_LOAD_MODEL` | no | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

**Traceback (tail):**

```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9037, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

### Issue Template

<details><summary>Copy/paste GitHub issue template</summary>


```markdown
### Summary
Model loading failed: RobertaTokenizer has no attribute additional_special_tokens

### Classification
- Package attribution: `mlx-vlm`
- Failure phase: `model_load`
- Error stage: `Model Error`
- Canonical code: `MLX_VLM_MODEL_LOAD_MODEL`
- Signature: `MLX_VLM_MODEL_LOAD_MODEL:2100d402a936`

### Affected Models
prince-canuma/Florence-2-large-ft

### Minimal Reproduction
```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-154920_DSC09221.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models prince-canuma/Florence-2-large-ft
```

### Environment Fingerprint
`python=3.13.9; chip=Apple M4 Max; mlx=0.30.7.dev20260215+c184262d; mlx-vlm=0.3.12; mlx-lm=0.30.7; transformers=5.1.0`

### Repro Bundle
`/Users/jrp/Documents/AI/mlx/check_models/src/output/repro_bundles/20260215T032734Z_004_prince-canuma_Florence-2-large-ft_MLX_VLM_MODEL_LOAD_MODEL_2100d402a936.json`

### Traceback Tail
```text
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9037, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

### Suggested Tracker
- `mlx-vlm`: https://github.com/ml-explore/mlx-vlm/issues/new
```

</details>

<details><summary>Full tracebacks (all models in this cluster)</summary>

### `prince-canuma/Florence-2-large-ft`

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9029, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 8806, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/molmo/processing_molmo.py", line 758, in _patched_auto_processor_from_pretrained
    return _original_auto_processor_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 355, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/phi3_v/processing_phi3_v.py", line 699, in _patched_auto_processor_from_pretrained
    return _original_auto_processor_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/kimi_vl/processing_kimi_vl.py", line 595, in _patched_auto_processor_from_pretrained
    return _original_auto_processor_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 394, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1403, in from_pretrained
    return cls.from_args_and_dict(args, processor_dict, **instantiation_kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1170, in from_args_and_dict
    processor = cls(*args, **valid_kwargs)
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/microsoft/Florence_hyphen_2_hyphen_large_hyphen_ft/4a12a2b54b7016a48a22037fbd62da90cd566f2a/processing_florence2.py", line 87, in __init__
    tokenizer.additional_special_tokens + \
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/tokenization_utils_base.py", line 1291, in __getattr__
    raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
AttributeError: RobertaTokenizer has no attribute additional_special_tokens. Did you mean: 'add_special_tokens'?

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9221, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9037, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

</details>

<details><summary>Captured stdout/stderr (all models in this cluster)</summary>

### `prince-canuma/Florence-2-large-ft`

```text
=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 24036.13it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
<unknown>:515: SyntaxWarning: invalid escape sequence '\d'
```

</details>

---

## Preflight Compatibility Warnings (2 issue(s))

These warnings were detected before inference. They are non-fatal but should be tracked as potential upstream compatibility issues.

- `mlx-vlm load_image() has an unguarded URL startswith() branch; Path/BytesIO inputs can raise AttributeError in upstream code.`
  - Likely package: `mlx-vlm`; suggested tracker: `mlx-vlm` (https://github.com/ml-explore/mlx-vlm/issues/new)
- `transformers import utils no longer reference TRANSFORMERS_NO_TF/FLAX/JAX; check_models backend guard env vars may be ignored with this version.`
  - Likely package: `transformers`; suggested tracker: `transformers` (https://github.com/huggingface/transformers/issues/new)

---

## Harness/Integration Issues (7 model(s))

These models completed successfully but show integration problems (including empty output, encoding corruption, stop-token leakage, or prompt-template/long-context issues) that indicate stack bugs rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct` — long_context

**Tokens:** prompt=16,583, generated=500, ratio=3.02%
**Likely package:** `mlx-vlm / mlx`
**Quality flags:** ⚠️harness(long_context), long-context, repetitive(phrase: "england, england, england, eng...")

**Details:** long_context_repetition(16583tok)

**Sample output:**

```text
Title: A red brick cottage with a tall chimney on a street corner in Hitchin, England

Description: A red brick cottage with a tall chimney and a tiled roof is situated on a street corner in Hitchin, ...
```

### `microsoft/Phi-3.5-vision-instruct` — stop_token

**Tokens:** prompt=1,138, generated=500, ratio=43.94%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(stop_token), lang_mixing, hallucination

**Details:** token_leak:<|end|>, token_leak:<|endoftext|>, training_leak:code_example

**Sample output:**

```text
Title: Victorian Cottage in Hitchin, England

Description: A quaint Victorian cottage with a prominent central chimney is captured in Hitchin, England. The image, taken on a late afternoon in February...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` — encoding

**Tokens:** prompt=2,533, generated=168, ratio=6.63%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(encoding)

**Details:** token_encoding:bpe_space_leak(113)

**Sample output:**

```text
Title:ĠRedĠBrickĠCottageĠwithĠTallĠChimneyĊĊDescription:ĠAĠtraditionalĠredĠbrickĠcottageĠwithĠaĠsteepĠclayĠtileĠroofĠandĠaĠprominentĠcentralĠchimneyĠstandsĠonĠaĠquietĠstreetĠcorner.ĠTheĠbuildingĠfeatu...
```

### `mlx-community/GLM-4.6V-Flash-mxfp4` — stop_token

**Tokens:** prompt=6,348, generated=500, ratio=7.88%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(stop_token), hallucination, formatting

**Details:** token_leak:</think>

**Sample output:**

```text
<think>Got it, let's tackle this task step by step. The user wants exactly three sections: Title, Description, Keywords. Also, there's a Context section mentioned, but wait, the user's instructions sa...
```

### `mlx-community/Qwen3-VL-2B-Thinking-bf16` — long_context

**Tokens:** prompt=16,585, generated=500, ratio=3.01%
**Likely package:** `mlx-vlm / mlx`
**Quality flags:** ⚠️harness(long_context), long-context, repetitive(phrase: "chimney stack, terracotta, bri...")

**Details:** long_context_repetition(16585tok)

**Sample output:**

```text
Got it, let's tackle this image analysis. First, the title needs to be 6-12 words, descriptive and concrete. The image shows a brick house with a tall chimney, so maybe "Tall brick chimney house with ...
```

### `mlx-community/gemma-3n-E2B-4bit` — prompt_template

**Tokens:** prompt=572, generated=2, ratio=0.35%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(prompt_template), context-ignored

**Details:** output:truncated(2tok)

**Sample output:**

```text
<empty output>
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16` — prompt_template

**Tokens:** prompt=1,335, generated=13, ratio=0.97%
**Likely package:** `mlx-vlm`
**Quality flags:** ⚠️harness(prompt_template), context-ignored

**Details:** output:output_ratio(1.0%)

**Sample output:**

```text
If the image is too small, use the image itself.
```

---

## Potential Stack Issues (3 model(s))

These models technically succeeded, but token/output patterns suggest likely integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Likely Package | Quality Flags |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- | ------------- |
| `Qwen/Qwen3-VL-2B-Instruct` | 16,583 | 500 | 3.02% | Repetition under extreme prompt length | `mlx-vlm / mlx` | ⚠️harness(long_context), long-context, repetitive(phrase: "england, england, england, eng...") |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | 16,585 | 500 | 3.01% | Repetition under extreme prompt length | `mlx-vlm / mlx` | ⚠️harness(long_context), long-context, repetitive(phrase: "chimney stack, terracotta, bri...") |
| `mlx-community/gemma-3n-E2B-4bit` | 572 | 2 | 0.35% | Empty output despite successful run | `mlx-vlm` | ⚠️harness(prompt_template), context-ignored |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each model appears).

**Regressions since previous run:** `mlx-community/deepseek-vl2-8bit`
**Recoveries since previous run:** none

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/X-Reasoner-7B-8bit` | still failing | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |
| `mlx-community/deepseek-vl2-8bit` | new regression | 2026-02-15 03:27:34 GMT | 1/3 recent runs failed |
| `prince-canuma/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

---

## Priority Summary

| Priority | Issue | Models Affected | Package |
| -------- | ----- | --------------- | ------- |
| **Low** | Weight Mismatch | 1 (Florence-2-large-ft) | mlx |
| **Medium** | OOM | 1 (X-Reasoner-7B-8bit) | mlx |
| **Medium** | Processor Error | 1 (deepseek-vl2-8bit) | model-config |
| **Medium** | Model Error | 1 (Florence-2-large-ft) | mlx-vlm |
| **Medium** | Harness/integration | 7 (Qwen3-VL-2B-Instruct, Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, GLM-4.6V-Flash-mxfp4, Qwen3-VL-2B-Thinking-bf16, gemma-3n-E2B-4bit, paligemma2-10b-ft-docci-448-bf16) | mlx-vlm |
| **Medium** | Preflight compatibility warning | 2 issue(s) | mlx-vlm, transformers |

## Reproducibility


```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-154920_DSC09221.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
```


### Target specific failing models

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-154920_DSC09221.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-154920_DSC09221.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/X-Reasoner-7B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-154920_DSC09221.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260214-154920_DSC09221.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models prince-canuma/Florence-2-large-ft
```

<details><summary>Prompt used (click to expand)</summary>

```text
Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Town Centre, Hitchin, England, United Kingdom, UK Here is a caption for the image, following your instructions: A distinctive Victorian cottage, characterized by its imposing central chimney stack, is pictured on a late afternoon in the historic market town of Hitchin, England. Photographed in February, the scene is set against a soft, overcast winter sky, which accentuates the textures of the weathered red bric...
- Keyword hints: 19th century, Any Vision, British, Chimney Pots, Clay Tile Roof, Clay tiles, England, English cottage, Europe, Hertfordshire, Hitchin, Quaint, Red Brick House, Scenery, Street side, Tall Brick Chimney, Terracotta, Town centre, Traditional Brick House, UK
- Capture metadata: Taken on 2026-02-14 15:49:20 GMT (at 15:49:20 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.
```

</details>

_Report generated on 2026-02-15 03:27:34 GMT by [check_models](https://github.com/jrp2014/check_models)._
