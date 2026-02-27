# Diagnostics Report — 4 failure(s), 11 harness issue(s) (mlx-vlm 0.3.12)

## Summary

Automated benchmarking of **46 locally-cached VLM models** found **4 hard failure(s)** and **11 harness/integration issue(s)** plus **2 preflight compatibility warning(s)** in successful models. 42 of 46 models succeeded.

Test image: `20260221-161455_DSC09294_DxO.jpg` (52.8 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [mlx-vlm]** Failed to process inputs with error: can only concatenate str (not "NoneType") to str (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm]** Model runtime error during generation for mlx-community/InternVL3-14B-8bit: LanguageM... (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm]** Model runtime error during generation for mlx-community/InternVL3-8B-bf16: LanguageMo... (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 11 model(s). Next: validate stop-token decoding and long-context behavior.
- **[Medium] [mlx-vlm / mlx]** Stack-signal anomalies on 2 successful model(s). Next: inspect prompt/output token ratios and empty-output patterns.
- **[Medium] [mlx-vlm, transformers]** Preflight compatibility warnings (2 issue(s)). Next: verify dependency/version compatibility before model runs.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **Medium** | Failed to process inputs with error: can only concatenate str (not "N... | 1 (Florence-2-large-ft) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Model runtime error during generation for mlx-community/InternVL3-14B... | 1 (InternVL3-14B-8bit) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Model runtime error during generation for mlx-community/InternVL3-8B-... | 1 (InternVL3-8B-bf16) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (deepseek-vl2-8bit) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Harness/integration | 11 (SmolVLM-Instruct, Qwen3-VL-2B-Instruct, Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, ERNIE-4.5-VL-28B-A3B-Thinking-bf16, Qwen2-VL-2B-Instruct-4bit, SmolVLM-Instruct-bf16, SmolVLM2-2.2B-Instruct-mlx, llava-v1.6-mistral-7b-8bit, paligemma2-3b-ft-docci-448-bf16, Florence-2-large-ft) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
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

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft`
- Portable dependency probe: `python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers`

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
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '1', '-', '1', '6', '1', '4', '5', '5', '_', 'D', 'S', 'C', '0', '9', '2', '9', '4', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

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
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK On a late winter afternoon, a heavy mist hangs over the hills surrounding Loch Katrine in Stronachlachar, Scotland. The muted colours of the February landscape are dominated by dark evergreen forests and the bare branches of deciduous trees. Nestled on the shore is a utility building, part of the infrastructure for the Loch Katrine aqueduct scheme. In th...
- Capture metadata: Taken on 2026-02-21 16:14:55 GMT (at 16:14:55 local time).

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 11654.08it/s]

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

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-14B-8bit`
- Portable dependency probe: `python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers`

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
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '1', '-', '1', '6', '1', '4', '5', '5', '_', 'D', 'S', 'C', '0', '9', '2', '9', '4', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

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
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK On a late winter afternoon, a heavy mist hangs over the hills surrounding Loch Katrine in Stronachlachar, Scotland. The muted colours of the February landscape are dominated by dark evergreen forests and the bare branches of deciduous trees. Nestled on the shore is a utility building, part of the infrastructure for the Loch Katrine aqueduct scheme. In th...
- Capture metadata: Taken on 2026-02-21 16:14:55 GMT (at 16:14:55 local time).
Assistant:

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 18 files:   0%|          | 0/18 [00:00<?, ?it/s]
Fetching 18 files: 100%|##########| 18/18 [00:00<00:00, 14838.34it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/2065 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/2065 [00:00<?, ?tok/s]
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

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-8B-bf16`
- Portable dependency probe: `python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers`

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
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '2', '2', '1', '-', '1', '6', '1', '4', '5', '5', '_', 'D', 'S', 'C', '0', '9', '2', '9', '4', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

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
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK On a late winter afternoon, a heavy mist hangs over the hills surrounding Loch Katrine in Stronachlachar, Scotland. The muted colours of the February landscape are dominated by dark evergreen forests and the bare branches of deciduous trees. Nestled on the shore is a utility building, part of the infrastructure for the Loch Katrine aqueduct scheme. In th...
- Capture metadata: Taken on 2026-02-21 16:14:55 GMT (at 16:14:55 local time).
Assistant:

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 17 files:   0%|          | 0/17 [00:00<?, ?it/s]
Fetching 17 files: 100%|##########| 17/17 [00:00<00:00, 23324.56it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]

Prefill:   0%|          | 0/2065 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/2065 [00:00<?, ?tok/s]
```

</details>

## 4. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `mlx-community/deepseek-vl2-8bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/deepseek-vl2-8bit` | Loaded processor has no image_processor; expected multimodal processor. | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit`
- Portable dependency probe: `python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers`

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
Fetching 13 files: 100%|##########| 13/13 [00:00<00:00, 22531.39it/s]

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

## Harness/Integration Issues (11 model(s))

These models completed successfully but show integration problems (for example stop-token leakage, decoding artifacts, or long-context breakdown) that likely point to stack/runtime behavior rather than inherent model quality limits.

### `HuggingFaceTB/SmolVLM-Instruct`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,474, output=12, output/prompt=0.81%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Loch katrine, scotland, uk.
```

### `Qwen/Qwen3-VL-2B-Instruct`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,499, output=500, output/prompt=3.03%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16499 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
Title: Loch Katrine, Stronachlachar, Scotland, UK

Description: A misty, overcast view of Loch Katrine in Stronachlachar, Scotland, featuring a utility building on the shore and power lines in the bac...
```

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,072, output=500, output/prompt=46.64%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Generated text appears to continue into new writing prompts mid-output.
- Output switched language/script unexpectedly.

**Sample output:**

```text
Title: Misty Loch Katrine in Stronachlachar

Description: A misty, overcast day at Loch Katrine in Stronachlachar, Scotland, with a utility building on the shore and power lines in the background.

Ke...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,359, output=121, output/prompt=5.13%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 73 occurrences).
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Title:ĠMistyĠLochĠKatrineĠwithĠUtilityĠBuildingĊĊDescription:ĠAĠutilityĠbuildingĠstandsĠonĠtheĠshoreĠofĠLochĠKatrine,ĠsurroundedĠbyĠmistyĠhillsĠandĠevergreenĠforests.ĠPowerĠlinesĠstretchĠacrossĠtheĠsc...
```

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,579, output=500, output/prompt=31.67%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.

**Sample output:**

```text
Alright, here's what I'm thinking. I need to generate metadata for this image, and it's got to be spot-on. First, the title. I've got to be concise, descriptive, and within the word count. "Misty Loch...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,510, output=500, output/prompt=3.03%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16510 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.
- Output appears to copy prompt context verbatim.

**Sample output:**

```text
Title: Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK
Description: On a late winter afternoon, a heavy mist hangs over the hills surrounding Loch Katrine in Stronachlachar, Scotland. The m...
```

### `mlx-community/SmolVLM-Instruct-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,474, output=12, output/prompt=0.81%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Loch katrine, scotland, uk.
```

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,374, output=8, output/prompt=0.58%

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 8 tokens.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
title=loch katrine
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,465, output=12, output/prompt=0.49%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
The image is a photograph of a landscape scene.
```

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,296, output=9, output/prompt=0.69%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
- The image is in low resolution.
```

### `prince-canuma/Florence-2-large-ft`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=850, output=500, output/prompt=58.82%

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

### Long-Context Degradation / Potential Stack Issues (2 model(s))

These models technically succeeded, but token/output patterns suggest likely integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Owner |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- |
| `Qwen/Qwen3-VL-2B-Instruct` | 16,499 | 500 | 3.03% | Repetition under extreme prompt length | `mlx-vlm / mlx` |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | 16,510 | 500 | 3.03% | Repetition under extreme prompt length | `mlx-vlm / mlx` |

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
| `mlx-community/deepseek-vl2-8bit` | still failing | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.3.12 |
| mlx | 0.31.0.dev20260227+365d6f29 |
| mlx-lm | 0.30.8 |
| transformers | 5.2.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.5.0 |
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-14B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/InternVL3-8B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
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
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK On a late winter afternoon, a heavy mist hangs over the hills surrounding Loch Katrine in Stronachlachar, Scotland. The muted colours of the February landscape are dominated by dark evergreen forests and the bare branches of deciduous trees. Nestled on the shore is a utility building, part of the infrastructure for the Loch Katrine aqueduct scheme. In th...
- Capture metadata: Taken on 2026-02-21 16:14:55 GMT (at 16:14:55 local time).
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260221-161455_DSC09294_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-02-27 22:57:22 GMT by [check_models](https://github.com/jrp2014/check_models)._
