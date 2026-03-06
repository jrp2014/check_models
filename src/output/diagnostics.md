# Diagnostics Report — 6 failure(s), 8 harness issue(s) (mlx-vlm 0.3.13)

## Summary

Automated benchmarking of **48 locally-cached VLM models** found **6 hard failure(s)** and **8 harness/integration issue(s)** plus **2 preflight compatibility warning(s)** in successful models. 42 of 48 models succeeded.

Test image: `20260228-162254_DSC09377.jpg` (33.5 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [mlx-vlm]** Failed to process inputs with error: can only concatenate str (not "NoneType") to str (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm]** Model runtime error during generation for mlx-community/InternVL3-14B-8bit: LanguageM... (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm]** Model runtime error during generation for mlx-community/InternVL3-8B-bf16: LanguageMo... (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx]** Model runtime error during generation for mlx-community/Qwen3.5-35B-A3B-6bit: [metal:... (1 model(s)). Next: check tensor/cache behavior and memory pressure handling.
- **[Medium] [mlx]** Model runtime error during generation for mlx-community/Qwen3.5-35B-A3B-bf16: [metal:... (1 model(s)). Next: check tensor/cache behavior and memory pressure handling.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 8 model(s). Next: validate stop-token decoding and long-context behavior.
- **[Medium] [mlx-vlm, transformers]** Preflight compatibility warnings (2 issue(s)). Next: verify dependency/version compatibility before model runs.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **Medium** | Failed to process inputs with error: can only concatenate str (not "N... | 1 (Florence-2-large-ft) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Model runtime error during generation for mlx-community/InternVL3-14B... | 1 (InternVL3-14B-8bit) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Model runtime error during generation for mlx-community/InternVL3-8B-... | 1 (InternVL3-8B-bf16) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Model runtime error during generation for mlx-community/Qwen3.5-35B-A... | 1 (Qwen3.5-35B-A3B-6bit) | `mlx` | check tensor/cache behavior and memory pressure handling. |
| **Medium** | Model runtime error during generation for mlx-community/Qwen3.5-35B-A... | 1 (Qwen3.5-35B-A3B-bf16) | `mlx` | check tensor/cache behavior and memory pressure handling. |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (deepseek-vl2-8bit) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Harness/integration | 8 (Qwen3-VL-2B-Instruct, Devstral-Small-2-24B-Instruct-2512-5bit, Qwen3-VL-2B-Thinking-bf16, SmolVLM2-2.2B-Instruct-mlx, X-Reasoner-7B-8bit, paligemma2-10b-ft-docci-448-6bit, paligemma2-3b-ft-docci-448-bf16, Florence-2-large-ft) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Preflight compatibility warning | 2 issue(s) | `mlx-vlm, transformers` | verify dependency/version compatibility before model runs. |

---

## 1. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: can only concatenate str (not "NoneType") to str
**Owner (likely component):** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Affected model:** `microsoft/Florence-2-large-ft`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | Failed to process inputs with error: can only concatenate str (not "NoneType") to str | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft`

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
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 163, in __call__
    image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 50, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 860, in preprocess
    self._validate_preprocess_kwargs(**kwargs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 823, in _validate_preprocess_kwargs
    validate_fast_preprocess_arguments(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        do_rescale=do_rescale,
        ^^^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        data_format=data_format,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 106, in validate_fast_preprocess_arguments
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
  File "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 185, in __call__
    self.image_token * self.num_image_tokens
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + self.tokenizer.bos_token
    ^~~~~~~~~~~~~~~~~~~~~~~~~~
TypeError: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 662, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 505, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1111, in prepare_inputs
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
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '8', '-', '1', '6', '2', '2', '5', '4', '_', 'D', 'S', 'C', '0', '9', '3', '7', '7', '.', 'j', 'p', 'g'] 

Prompt: Analyze this image for cataloguing metadata.

Return exactly these three sections, and nothing else:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 unique comma-separated terms, ordered most specific to most general.

Rules:
- Use only visually supported facts.
- If hints conflict with the image, trust the image.
- Do not output reasoning, notes, or extra sections.
- Do not copy context hints verbatim.

Context: Existing metadata hints (high confidence; use only if visually consistent):
- Description hint: , St Mary's Church, Lenham, England, United Kingdom, UK Here is a caption for the image, written in a neutral but descriptive style suitable for a periodical, blog, or magazine. **Caption:** A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
- Keyword hints: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
- Capture metadata: Taken on 2026-02-28 16:22:54 GMT (at 16:22:54 local time). GPS: 51.237300°N, 0.718917°E.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 24571.20it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 2. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model runtime error during generation for mlx-community/InternVL3-14B-8bit: LanguageModel.\_\_call\_\_() got an unexpected keyword argument 'n_to_process'
**Owner (likely component):** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Affected model:** `mlx-community/InternVL3-14B-8bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/InternVL3-14B-8bit` | Model runtime error during generation for mlx-community/InternVL3-14B-8bit: LanguageModel.\_\_call\_\_() got an unexpected keyword argument 'n_to_process' | 2026-02-23 12:54:48 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-14B-8bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/InternVL3-14B-8bit`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 662, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 550, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 408, in generate_step
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
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '8', '-', '1', '6', '2', '2', '5', '4', '_', 'D', 'S', 'C', '0', '9', '3', '7', '7', '.', 'j', 'p', 'g'] 

Prompt: User: <image>
Analyze this image for cataloguing metadata.

Return exactly these three sections, and nothing else:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 unique comma-separated terms, ordered most specific to most general.

Rules:
- Use only visually supported facts.
- If hints conflict with the image, trust the image.
- Do not output reasoning, notes, or extra sections.
- Do not copy context hints verbatim.

Context: Existing metadata hints (high confidence; use only if visually consistent):
- Description hint: , St Mary's Church, Lenham, England, United Kingdom, UK Here is a caption for the image, written in a neutral but descriptive style suitable for a periodical, blog, or magazine. **Caption:** A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
- Keyword hints: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
- Capture metadata: Taken on 2026-02-28 16:22:54 GMT (at 16:22:54 local time). GPS: 51.237300°N, 0.718917°E.
Assistant:

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 18 files:   0%|          | 0/18 [00:00<?, ?it/s]
Fetching 18 files: 100%|##########| 18/18 [00:00<00:00, 23316.08it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/2156 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/2156 [00:00<?, ?tok/s]
```

</details>

## 3. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model runtime error during generation for mlx-community/InternVL3-8B-bf16: LanguageModel.\_\_call\_\_() got an unexpected keyword argument 'n_to_process'
**Owner (likely component):** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Affected model:** `mlx-community/InternVL3-8B-bf16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/InternVL3-8B-bf16` | Model runtime error during generation for mlx-community/InternVL3-8B-bf16: LanguageModel.\_\_call\_\_() got an unexpected keyword argument 'n_to_process' | 2026-02-23 12:54:48 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-8B-bf16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/InternVL3-8B-bf16`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 662, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 550, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 408, in generate_step
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
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '8', '-', '1', '6', '2', '2', '5', '4', '_', 'D', 'S', 'C', '0', '9', '3', '7', '7', '.', 'j', 'p', 'g'] 

Prompt: User: <image>
Analyze this image for cataloguing metadata.

Return exactly these three sections, and nothing else:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 unique comma-separated terms, ordered most specific to most general.

Rules:
- Use only visually supported facts.
- If hints conflict with the image, trust the image.
- Do not output reasoning, notes, or extra sections.
- Do not copy context hints verbatim.

Context: Existing metadata hints (high confidence; use only if visually consistent):
- Description hint: , St Mary's Church, Lenham, England, United Kingdom, UK Here is a caption for the image, written in a neutral but descriptive style suitable for a periodical, blog, or magazine. **Caption:** A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
- Keyword hints: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
- Capture metadata: Taken on 2026-02-28 16:22:54 GMT (at 16:22:54 local time). GPS: 51.237300°N, 0.718917°E.
Assistant:

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 17 files:   0%|          | 0/17 [00:00<?, ?it/s]
Fetching 17 files: 100%|##########| 17/17 [00:00<00:00, 24896.36it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/2156 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/2156 [00:00<?, ?tok/s]
```

</details>

## 4. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model runtime error during generation for mlx-community/Qwen3.5-35B-A3B-6bit: [metal::malloc] Attempting to allocate 134767706112 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
**Owner (likely component):** `mlx`
**Suggested next action:** check tensor/cache behavior and memory pressure handling.
**Affected model:** `mlx-community/Qwen3.5-35B-A3B-6bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | Model runtime error during generation for mlx-community/Qwen3.5-35B-A3B-6bit: [metal::malloc] Attempting to allocate 134767706112 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes. | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-6bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/Qwen3.5-35B-A3B-6bit`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 662, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 550, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 416, in generate_step
    mx.eval([c.state for c in prompt_cache])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [metal::malloc] Attempting to allocate 134767706112 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model runtime error during generation for mlx-community/Qwen3.5-35B-A3B-6bit: [metal::malloc] Attempting to allocate 134767706112 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '8', '-', '1', '6', '2', '2', '5', '4', '_', 'D', 'S', 'C', '0', '9', '3', '7', '7', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata.

Return exactly these three sections, and nothing else:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 unique comma-separated terms, ordered most specific to most general.

Rules:
- Use only visually supported facts.
- If hints conflict with the image, trust the image.
- Do not output reasoning, notes, or extra sections.
- Do not copy context hints verbatim.

Context: Existing metadata hints (high confidence; use only if visually consistent):
- Description hint: , St Mary's Church, Lenham, England, United Kingdom, UK Here is a caption for the image, written in a neutral but descriptive style suitable for a periodical, blog, or magazine. **Caption:** A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
- Keyword hints: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
- Capture metadata: Taken on 2026-02-28 16:22:54 GMT (at 16:22:54 local time). GPS: 51.237300°N, 0.718917°E.<|im_end|>
<|im_start|>assistant
<think>

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 16 files:   0%|          | 0/16 [00:00<?, ?it/s]
Fetching 16 files: 100%|##########| 16/16 [00:00<00:00, 19418.07it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/16604 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/16604 [00:00<?, ?tok/s]
```

</details>

## 5. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model runtime error during generation for mlx-community/Qwen3.5-35B-A3B-bf16: [metal::malloc] Attempting to allocate 134767706112 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
**Owner (likely component):** `mlx`
**Suggested next action:** check tensor/cache behavior and memory pressure handling.
**Affected model:** `mlx-community/Qwen3.5-35B-A3B-bf16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | Model runtime error during generation for mlx-community/Qwen3.5-35B-A3B-bf16: [metal::malloc] Attempting to allocate 134767706112 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes. | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-bf16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/Qwen3.5-35B-A3B-bf16`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 662, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 550, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 416, in generate_step
    mx.eval([c.state for c in prompt_cache])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [metal::malloc] Attempting to allocate 134767706112 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model runtime error during generation for mlx-community/Qwen3.5-35B-A3B-bf16: [metal::malloc] Attempting to allocate 134767706112 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '8', '-', '1', '6', '2', '2', '5', '4', '_', 'D', 'S', 'C', '0', '9', '3', '7', '7', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata.

Return exactly these three sections, and nothing else:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 unique comma-separated terms, ordered most specific to most general.

Rules:
- Use only visually supported facts.
- If hints conflict with the image, trust the image.
- Do not output reasoning, notes, or extra sections.
- Do not copy context hints verbatim.

Context: Existing metadata hints (high confidence; use only if visually consistent):
- Description hint: , St Mary's Church, Lenham, England, United Kingdom, UK Here is a caption for the image, written in a neutral but descriptive style suitable for a periodical, blog, or magazine. **Caption:** A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
- Keyword hints: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
- Capture metadata: Taken on 2026-02-28 16:22:54 GMT (at 16:22:54 local time). GPS: 51.237300°N, 0.718917°E.<|im_end|>
<|im_start|>assistant
<think>

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 24 files:   0%|          | 0/24 [00:00<?, ?it/s]
Fetching 24 files: 100%|##########| 24/24 [00:00<00:00, 13930.71it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/16604 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/16604 [00:00<?, ?tok/s]
```

</details>

## 6. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `mlx-community/deepseek-vl2-8bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/deepseek-vl2-8bit` | Loaded processor has no image_processor; expected multimodal processor. | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit`

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
Fetching 13 files: 100%|##########| 13/13 [00:00<00:00, 10309.31it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

---

## Preflight Compatibility Warnings (2 issue(s))

These warnings were detected before inference. They are non-fatal but should be tracked as potential upstream compatibility issues.

- `mlx-vlm load_image() has an unguarded URL startswith() branch; Path/BytesIO inputs can raise AttributeError in upstream code.`
  - Owner: `mlx-vlm`; suggested tracker: `mlx-vlm` (<https://github.com/ml-explore/mlx-vlm/issues/new>)
  - Suggested next action: check processor/chat-template wiring and generation kwargs.
- `transformers import utils no longer reference known backend guard env vars (TRANSFORMERS_NO_* / USE_*); check_models backend guard hints may be ignored with this version.`
  - Owner: `transformers`; suggested tracker: `transformers` (<https://github.com/huggingface/transformers/issues/new>)
  - Suggested next action: verify API compatibility and pinned version floor.

---

## Harness/Integration Issues (8 model(s))

8 model(s) show potential harness/integration issues; see per-model breakdown below.
These models completed successfully but show integration problems (for example stop-token leakage, decoding artifacts, or long-context breakdown) that likely point to stack/runtime behavior rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,590, output=500, output/prompt=3.01%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16590 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
Title: Gravestones in a churchyard in Lenham, Kent, England

Description: A quiet churchyard in Lenham, Kent, England, featuring moss-covered gravestones and a half-timbered house in the background. T...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,455, output=106, output/prompt=4.32%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 55 occurrences).
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Title:ĠHistoricĠChurchyardĠwithĠTudorĠHousesĊĊDescription:ĠTheĠimageĠshowsĠaĠchurchyardĠwithĠoldĠgravestonesĠandĠaĠbackdropĠofĠtraditionalĠTudor-styleĠhousesĠinĠLenham,ĠEngland.ĊĊKeywords:ĠStĠMary'sĠC...
```

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,592, output=500, output/prompt=3.01%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16592 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.
- Output contains corrupted or malformed text segments.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Got it, let's tackle this. First, the Title needs to be 6-12 words, descriptive and concrete. The image is a graveyard with a churchyard, so maybe "Graveyard of St Mary's Church Lenham England" but ne...
```

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,470, output=9, output/prompt=0.61%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
Cemetery, England, United Kingdom, UK
```

### `mlx-community/X-Reasoner-7B-8bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,601, output=500, output/prompt=3.01%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16601 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
**Title:** Ancient Headstones in a Village Churchyard, Lenham, Kent, England

**Description:** A tranquil scene of a historic village churchyard in Lenham, Kent, England, featuring weathered graveston...
```

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,393, output=14, output/prompt=1.01%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (1.0%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
- Use only the hints that are visually supported by the image.
```

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,393, output=5, output/prompt=0.36%

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 5 tokens.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
- Daytime.
```

### `prince-canuma/Florence-2-large-ft`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=933, output=500, output/prompt=53.59%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;s&gt; appeared in generated text.
- Model output may not follow prompt or image contents.
- Output contains corrupted or malformed text segments.
- Output switched language/script unexpectedly.
- Output formatting deviated from the requested structure.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s...
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/InternVL3-14B-8bit` | still failing | 2026-02-23 12:54:48 GMT | 3/3 recent runs failed |
| `mlx-community/InternVL3-8B-bf16` | still failing | 2026-02-23 12:54:48 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | still failing | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | still failing | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |
| `mlx-community/deepseek-vl2-8bit` | still failing | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 14
- **Summary diagnostics models:** 34
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 941.21s (941.21s)
- **Average runtime per model:** 19.61s (19.61s)
- **Runtime note:** 6 model(s) had missing timing fields and were counted as 0.00s.

---

## Models Not Flagged (34 model(s))

These models completed without diagnostics flags (no hard failure, harness warning, or stack-signal anomaly).

### Clean output (5 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
- `mlx-community/gemma-3-27b-it-qat-4bit`
- `mlx-community/gemma-3-27b-it-qat-8bit`

### Ran, but with quality warnings (29 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output omitted required Title/Description/Keywords sections.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output became repetitive, indicating possible generation instability.
- `microsoft/Phi-3.5-vision-instruct`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output became repetitive, indicating possible generation instability.
- `mlx-community/FastVLM-0.5B-bf16`: Output appears to copy prompt context verbatim.
- `mlx-community/GLM-4.6V-Flash-6bit`: Output formatting deviated from the requested structure.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Output formatting deviated from the requested structure.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Output formatting deviated from the requested structure.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Output leaked reasoning or prompt-template text.
- `mlx-community/LFM2-VL-1.6B-8bit`: Output appears to copy prompt context verbatim.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Output appears to copy prompt context verbatim.
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/Molmo-7B-D-0924-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/Molmo-7B-D-0924-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: Output appears to copy prompt context verbatim.
- `mlx-community/SmolVLM-Instruct-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E4B-it-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/llava-v1.6-mistral-7b-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/nanoLLaVA-1.5-4bit`: Output leaked reasoning or prompt-template text.
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/pixtral-12b-8bit`: Output appears to copy prompt context verbatim.
- `mlx-community/pixtral-12b-bf16`: Output appears to copy prompt context verbatim.
- `qnguyen3/nanoLLaVA`: Model output may not follow prompt or image contents.

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.3.13 |
| mlx | 0.31.1.dev20260306+be872ebd |
| mlx-lm | 0.30.8 |
| transformers | 5.3.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.5.0 |
| Python Version | 3.13.9 |
| OS | Darwin 25.3.0 |
| macOS Version | 26.3.1 |
| GPU/Chip | Apple M4 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
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

**Note:** A comprehensive JSON reproduction bundle including system info and the exact prompt trace has been exported to [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles) for each failing model.

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-14B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-8B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-6bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
```

### Prompt Used

```text
Analyze this image for cataloguing metadata.

Return exactly these three sections, and nothing else:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 unique comma-separated terms, ordered most specific to most general.

Rules:
- Use only visually supported facts.
- If hints conflict with the image, trust the image.
- Do not output reasoning, notes, or extra sections.
- Do not copy context hints verbatim.

Context: Existing metadata hints (high confidence; use only if visually consistent):
- Description hint: , St Mary's Church, Lenham, England, United Kingdom, UK Here is a caption for the image, written in a neutral but descriptive style suitable for a periodical, blog, or magazine. **Caption:** A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
- Keyword hints: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
- Capture metadata: Taken on 2026-02-28 16:22:54 GMT (at 16:22:54 local time). GPS: 51.237300°N, 0.718917°E.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-03-06 13:22:23 GMT by [check_models](https://github.com/jrp2014/check_models)._
