# Diagnostics Report — 4 failure(s), 5 harness issue(s) (mlx-vlm 0.3.12)

## Summary

Automated benchmarking of **46 locally-cached VLM models** found **4 hard failure(s)** and **5 harness/integration issue(s)** plus **2 preflight compatibility warning(s)** in successful models. 42 of 46 models succeeded.

Test image: `20260221-152935_DSC09285.jpg` (22.0 MB).

---

## Priority Summary

| Priority | Issue | Models Affected | Package |
| -------- | ----- | --------------- | ------- |
| **Medium** | Failed to process inputs with error: can only concatenate str (not "N... | 1 (Florence-2-large-ft) | mlx-vlm |
| **Medium** | Model runtime error during generation for mlx-community/InternVL3-14B... | 1 (InternVL3-14B-8bit) | mlx-vlm |
| **Medium** | Model runtime error during generation for mlx-community/InternVL3-8B-... | 1 (InternVL3-8B-bf16) | mlx-vlm |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (deepseek-vl2-8bit) | model-config |
| **Medium** | Harness/integration | 5 (Qwen3-VL-2B-Instruct, Devstral-Small-2-24B-Instruct-2512-5bit, Molmo-7B-D-0924-8bit, Molmo-7B-D-0924-bf16, Florence-2-large-ft) | mlx-vlm |
| **Medium** | Preflight compatibility warning | 2 issue(s) | mlx-vlm, transformers |

---

## 1. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: can only concatenate str (not "NoneType") to str
**Likely component:** `mlx-vlm`
**Affected model:** `microsoft/Florence-2-large-ft`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | Failed to process inputs with error: can only concatenate str (not "NoneType") to str | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command: `python -m check_models --image /Users/jrp/Pictures/Processed/20260221-152935_DSC09285.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `microsoft/Florence-2-large-ft`

Traceback:

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
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 598, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 464, in stream_generate
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
ValueError: Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '1', '-', '1', '5', '2', '9', '3', '5', '_', 'D', 'S', 'C', '0', '9', '2', '8', '5', '.', 'j', 'p', 'g'] 

Prompt: Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK Based on the image and the provided context, here is a suitable caption: A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs...
- Keyword hints: Adobe Stock, Any Vision, Automobile, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees
- Capture metadata: Taken on 2026-02-21 15:29:35 GMT (at 15:29:35 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 29475.08it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 2. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model runtime error during generation for mlx-community/InternVL3-14B-8bit: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'
**Likely component:** `mlx-vlm`
**Affected model:** `mlx-community/InternVL3-14B-8bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/InternVL3-14B-8bit` | Model runtime error during generation for mlx-community/InternVL3-14B-8bit: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process' | 2026-02-23 12:54:48 GMT | 1/3 recent runs failed |

### To reproduce

- Repro command: `python -m check_models --image /Users/jrp/Pictures/Processed/20260221-152935_DSC09285.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-14B-8bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/InternVL3-14B-8bit`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 598, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 488, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 376, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
TypeError: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model runtime error during generation for mlx-community/InternVL3-14B-8bit: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '1', '-', '1', '5', '2', '9', '3', '5', '_', 'D', 'S', 'C', '0', '9', '2', '8', '5', '.', 'j', 'p', 'g'] 

Prompt: User: <image>
Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK Based on the image and the provided context, here is a suitable caption: A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs...
- Keyword hints: Adobe Stock, Any Vision, Automobile, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees
- Capture metadata: Taken on 2026-02-21 15:29:35 GMT (at 15:29:35 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.
Assistant:

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 18 files:   0%|          | 0/18 [00:00<?, ?it/s]
Fetching 18 files: 100%|##########| 18/18 [00:00<00:00, 12620.77it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/2116 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/2116 [00:00<?, ?tok/s]
```

</details>

## 3. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model runtime error during generation for mlx-community/InternVL3-8B-bf16: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'
**Likely component:** `mlx-vlm`
**Affected model:** `mlx-community/InternVL3-8B-bf16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/InternVL3-8B-bf16` | Model runtime error during generation for mlx-community/InternVL3-8B-bf16: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process' | 2026-02-23 12:54:48 GMT | 1/1 recent runs failed |

### To reproduce

- Repro command: `python -m check_models --image /Users/jrp/Pictures/Processed/20260221-152935_DSC09285.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-8B-bf16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/InternVL3-8B-bf16`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 598, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 488, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 376, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
TypeError: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model runtime error during generation for mlx-community/InternVL3-8B-bf16: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '1', '-', '1', '5', '2', '9', '3', '5', '_', 'D', 'S', 'C', '0', '9', '2', '8', '5', '.', 'j', 'p', 'g'] 

Prompt: User: <image>
Analyze this image for cataloguing metadata.

Return exactly these three sections:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.

Context: Existing metadata hints (use only if visually consistent):
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK Based on the image and the provided context, here is a suitable caption: A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs...
- Keyword hints: Adobe Stock, Any Vision, Automobile, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees
- Capture metadata: Taken on 2026-02-21 15:29:35 GMT (at 15:29:35 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.
Assistant:

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 17 files:   0%|          | 0/17 [00:00<?, ?it/s]
Fetching 17 files: 100%|##########| 17/17 [00:00<00:00, 14744.24it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/2116 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/2116 [00:00<?, ?tok/s]
```

</details>

## 4. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Likely component:** `model configuration/repository`
**Affected model:** `mlx-community/deepseek-vl2-8bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/deepseek-vl2-8bit` | Loaded processor has no image_processor; expected multimodal processor. | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command: `python -m check_models --image /Users/jrp/Pictures/Processed/20260221-152935_DSC09285.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit`

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
Fetching 13 files: 100%|##########| 13/13 [00:00<00:00, 66333.27it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

---

## Preflight Compatibility Warnings (2 issue(s))

These warnings were detected before inference. They are non-fatal but should be tracked as potential upstream compatibility issues.

- `mlx-vlm load_image() has an unguarded URL startswith() branch; Path/BytesIO inputs can raise AttributeError in upstream code.`
  - Likely package: `mlx-vlm`; suggested tracker: `mlx-vlm` (<https://github.com/ml-explore/mlx-vlm/issues/new>)
- `transformers import utils no longer reference TRANSFORMERS_NO_TF/FLAX/JAX; check_models backend guard env vars may be ignored with this version.`
  - Likely package: `transformers`; suggested tracker: `transformers` (<https://github.com/huggingface/transformers/issues/new>)

---

## Harness/Integration Issues (5 model(s))

These models completed successfully but show integration problems (for example stop-token leakage, decoding artifacts, or long-context breakdown) that likely point to stack/runtime behavior rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Token summary:** prompt=16,550, output=500, output/prompt=3.02%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16550 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
Title: Blue Honda HR-V e:HEV parked on wet road at Loch Katrine

Description: A light blue Honda HR-V e:HEV is parked on a wet asphalt road at the Loch Katrine pier in Stronachlachar, Scotland, with r...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Token summary:** prompt=2,414, output=233, output/prompt=9.65%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 141 occurrences).

**Sample output:**

```text
Title:ĠBlueĠSUVĠatĠStronachlacharĠPierĠonĠLochĠKatrineĊĊDescription:ĠAĠlightĠblueĠSUVĠisĠparkedĠatĠaĠrain-soakedĠpierĠonĠLochĠKatrine,ĠwithĠtheĠcalmĠwatersĠandĠbareĠtreesĠofĠtheĠTrossachsĠvisibleĠinĠt...
```

### `mlx-community/Molmo-7B-D-0924-8bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Token summary:** prompt=0, output=0, output/prompt=n/a

**Why this appears to be an integration/runtime issue:**

- Model returned zero output tokens.

**Sample output:**

```text
<empty output>
```

### `mlx-community/Molmo-7B-D-0924-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Token summary:** prompt=0, output=0, output/prompt=n/a

**Why this appears to be an integration/runtime issue:**

- Model returned zero output tokens.

**Sample output:**

```text
<empty output>
```

### `prince-canuma/Florence-2-large-ft`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Token summary:** prompt=905, output=500, output/prompt=55.25%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;s&gt; appeared in generated text.
- Model output may not follow prompt or image contents.
- Output contains corrupted or malformed text segments.
- Output switched language/script unexpectedly.
- Output formatting deviated from the requested structure.

**Sample output:**

```text
<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s...
```

---

## Potential Stack Issues (3 model(s))

These models technically succeeded, but token/output patterns suggest likely integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Likely Package |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- |
| `Qwen/Qwen3-VL-2B-Instruct` | 16,550 | 500 | 3.02% | Repetition under extreme prompt length | `mlx-vlm / mlx` |
| `mlx-community/Molmo-7B-D-0924-8bit` | 0 | 0 | n/a | Empty output despite successful run | `mlx-vlm` |
| `mlx-community/Molmo-7B-D-0924-bf16` | 0 | 0 | n/a | Empty output despite successful run | `mlx-vlm` |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each model appears).

**Regressions since previous run:** `mlx-community/InternVL3-14B-8bit`
**Recoveries since previous run:** none

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/InternVL3-14B-8bit` | new regression | 2026-02-23 12:54:48 GMT | 1/3 recent runs failed |
| `mlx-community/InternVL3-8B-bf16` | new model failing | 2026-02-23 12:54:48 GMT | 1/1 recent runs failed |
| `mlx-community/deepseek-vl2-8bit` | still failing | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.3.12 |
| mlx | 0.30.7.dev20260223+d4c81062 |
| mlx-lm | 0.30.8 |
| transformers | 5.2.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.4.1 |
| Python Version | 3.13.11 |
| OS | Darwin 25.3.0 |
| macOS Version | 26.3 |
| GPU/Chip | Apple M4 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-152935_DSC09285.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
```

### Target specific failing models

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-152935_DSC09285.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-152935_DSC09285.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-14B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-152935_DSC09285.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-8B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-152935_DSC09285.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
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
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK Based on the image and the provided context, here is a suitable caption: A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs...
- Keyword hints: Adobe Stock, Any Vision, Automobile, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees
- Capture metadata: Taken on 2026-02-21 15:29:35 GMT (at 15:29:35 local time).

Prioritize what is visibly present. If context conflicts with the image, trust the image.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260221-152935_DSC09285.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-02-23 12:54:48 GMT by [check_models](https://github.com/jrp2014/check_models)._
