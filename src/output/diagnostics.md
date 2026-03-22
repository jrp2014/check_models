# Diagnostics Report — 8 failure(s), 8 harness issue(s) (mlx-vlm 0.4.1)

## Summary

Automated benchmarking of **51 locally-cached VLM models** found **8 hard failure(s)** and **8 harness/integration issue(s)** plus **1 preflight compatibility warning(s)** in successful models. 43 of 51 models succeeded.

Test image: `20260321-182222_DSC09486_DxO.jpg` (33.8 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [transformers]** Failed to process inputs with error: can only concatenate str (not "NoneType") to str (1 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [mlx-vlm]** 'utf-8' codec can't decode byte 0xab in position 10: invalid start byte (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm]** 'utf-8' codec can't decode byte 0xa1 in position 0: invalid start byte (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [transformers]** Failed to process inputs with error: Only returning PyTorch tensors is currently supp... (1 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Failed to process inputs with error: Only returning PyTorch tensors is currently supp... (1 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Failed to process inputs with error: Only returning PyTorch tensors is currently supp... (1 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Failed to process inputs with error: Only returning PyTorch tensors is currently supp... (1 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 4 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 2 model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 2 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [transformers / mlx-vlm]** Stack-signal anomalies on 1 successful model(s). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **Medium** | Failed to process inputs with error: can only concatenate str (not "N... | 1 (Florence-2-large-ft) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | 'utf-8' codec can't decode byte 0xab in position 10: invalid start byte | 1 (InternVL3-8B-bf16) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | 'utf-8' codec can't decode byte 0xa1 in position 0: invalid start byte | 1 (Molmo-7B-D-0924-bf16) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Failed to process inputs with error: Only returning PyTorch tensors i... | 1 (Qwen3.5-27B-4bit) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | Failed to process inputs with error: Only returning PyTorch tensors i... | 1 (Qwen3.5-27B-mxfp8) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | Failed to process inputs with error: Only returning PyTorch tensors i... | 1 (Qwen3.5-35B-A3B-6bit) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | Failed to process inputs with error: Only returning PyTorch tensors i... | 1 (Qwen3.5-35B-A3B-bf16) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (deepseek-vl2-8bit) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Harness/integration | 4 (Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, ERNIE-4.5-VL-28B-A3B-Thinking-bf16, Florence-2-large-ft) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Harness/integration | 2 (Qwen3-VL-2B-Thinking-bf16, X-Reasoner-7B-8bit) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Harness/integration | 2 (Qwen2-VL-2B-Instruct-4bit, paligemma2-10b-ft-docci-448-bf16) | `model-config / mlx-vlm` | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Stack-signal anomaly | 1 (Qwen3-VL-2B-Instruct) | `transformers / mlx-vlm` | verify API compatibility and pinned version floor. |
| **Medium** | Preflight compatibility warning | 1 issue(s) | `transformers` | verify API compatibility and pinned version floor. |

---

## 1. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: can only concatenate str (not "NoneType") to str
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected model:** `microsoft/Florence-2-large-ft`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | Failed to process inputs with error: can only concatenate str (not "NoneType") to str | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `microsoft/Florence-2-large-ft`

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
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 185, in __call__
    self.image_token * self.num_image_tokens
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + self.tokenizer.bos_token
    ^~~~~~~~~~~~~~~~~~~~~~~~~~
TypeError: can only concatenate str (not "NoneType") to str

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
ValueError: Failed to process inputs with error: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '8', '2', '2', '2', '2', '_', 'D', 'S', 'C', '0', '9', '4', '8', '6', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: Analyze this image for cataloguing metadata, using British English.

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
- Description hint: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban waterside area. A modern glass building reflects the golden light of the setting sun against a purple twilight sky, while people walk along the towpath, relax on the bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the lively evening scene as people go about their daily lives, commuting or enjoying leisure time.
- Capture metadata: Taken on 2026-03-21 18:22:22 GMT (at 18:22:22 local time). GPS: 51.536500°N, 0.126500°W.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 38444.58it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 2. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** 'utf-8' codec can't decode byte 0xab in position 10: invalid start byte
**Owner (likely component):** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Affected model:** `mlx-community/InternVL3-8B-bf16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/InternVL3-8B-bf16` | 'utf-8' codec can't decode byte 0xab in position 10: invalid start byte | 2026-02-23 12:54:48 GMT | 2/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/InternVL3-8B-bf16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/InternVL3-8B-bf16`

Traceback:

```text
Traceback (most recent call last):
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
ValueError: Model generation failed for mlx-community/InternVL3-8B-bf16: 'utf-8' codec can't decode byte 0xab in position 10: invalid start byte
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '8', '2', '2', '2', '2', '_', 'D', 'S', 'C', '0', '9', '4', '8', '6', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: User: <image>
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
- Description hint: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban waterside area. A modern glass building reflects the golden light of the setting sun against a purple twilight sky, while people walk along the towpath, relax on the bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the lively evening scene as people go about their daily lives, commuting or enjoying leisure time.
- Capture metadata: Taken on 2026-03-21 18:22:22 GMT (at 18:22:22 local time). GPS: 51.536500°N, 0.126500°W.
Assistant:

感 Rencontre pestic Rencontre.ERR Rencontre enthus.ERR Rencontre醍racial pestic

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 17 files:   0%|          | 0/17 [00:00<?, ?it/s]
Fetching 17 files: 100%|##########| 17/17 [00:00<00:00, 9459.16it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 3. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** 'utf-8' codec can't decode byte 0xa1 in position 0: invalid start byte
**Owner (likely component):** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Affected model:** `mlx-community/Molmo-7B-D-0924-bf16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/Molmo-7B-D-0924-bf16` | 'utf-8' codec can't decode byte 0xa1 in position 0: invalid start byte | 2026-03-22 01:27:09 GMT | 2/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Molmo-7B-D-0924-bf16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/Molmo-7B-D-0924-bf16`

Traceback:

```text
Traceback (most recent call last):
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
ValueError: Model generation failed for mlx-community/Molmo-7B-D-0924-bf16: 'utf-8' codec can't decode byte 0xa1 in position 0: invalid start byte
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '8', '2', '2', '2', '2', '_', 'D', 'S', 'C', '0', '9', '4', '8', '6', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: Analyze this image for cataloguing metadata, using British English.

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
- Description hint: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban waterside area. A modern glass building reflects the golden light of the setting sun against a purple twilight sky, while people walk along the towpath, relax on the bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the lively evening scene as people go about their daily lives, commuting or enjoying leisure time.
- Capture metadata: Taken on 2026-03-21 18:22:22 GMT (at 18:22:22 local time). GPS: 51.536500°N, 0.126500°W.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 18 files:   0%|          | 0/18 [00:00<?, ?it/s]
Fetching 18 files: 100%|##########| 18/18 [00:00<00:00, 11023.14it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 4. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected model:** `mlx-community/Qwen3.5-27B-4bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/Qwen3.5-27B-4bit` | Failed to process inputs with error: Only returning PyTorch tensors is currently supported. | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-27B-4bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/Qwen3.5-27B-4bit`

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
ValueError: Model generation failed for mlx-community/Qwen3.5-27B-4bit: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '8', '2', '2', '2', '2', '_', 'D', 'S', 'C', '0', '9', '4', '8', '6', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata, using British English.

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
- Description hint: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban waterside area. A modern glass building reflects the golden light of the setting sun against a purple twilight sky, while people walk along the towpath, relax on the bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the lively evening scene as people go about their daily lives, commuting or enjoying leisure time.
- Capture metadata: Taken on 2026-03-21 18:22:22 GMT (at 18:22:22 local time). GPS: 51.536500°N, 0.126500°W.<|im_end|>
<|im_start|>assistant
<think>

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]
Fetching 13 files: 100%|##########| 13/13 [00:00<00:00, 49078.26it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 5. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected model:** `mlx-community/Qwen3.5-27B-mxfp8`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/Qwen3.5-27B-mxfp8` | Failed to process inputs with error: Only returning PyTorch tensors is currently supported. | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-27B-mxfp8`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/Qwen3.5-27B-mxfp8`

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
ValueError: Model generation failed for mlx-community/Qwen3.5-27B-mxfp8: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '8', '2', '2', '2', '2', '_', 'D', 'S', 'C', '0', '9', '4', '8', '6', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata, using British English.

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
- Description hint: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban waterside area. A modern glass building reflects the golden light of the setting sun against a purple twilight sky, while people walk along the towpath, relax on the bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the lively evening scene as people go about their daily lives, commuting or enjoying leisure time.
- Capture metadata: Taken on 2026-03-21 18:22:22 GMT (at 18:22:22 local time). GPS: 51.536500°N, 0.126500°W.<|im_end|>
<|im_start|>assistant
<think>

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 16 files:   0%|          | 0/16 [00:00<?, ?it/s]
Fetching 16 files: 100%|##########| 16/16 [00:00<00:00, 26715.31it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 6. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected model:** `mlx-community/Qwen3.5-35B-A3B-6bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | Failed to process inputs with error: Only returning PyTorch tensors is currently supported. | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-6bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/Qwen3.5-35B-A3B-6bit`

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
ValueError: Model generation failed for mlx-community/Qwen3.5-35B-A3B-6bit: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '8', '2', '2', '2', '2', '_', 'D', 'S', 'C', '0', '9', '4', '8', '6', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata, using British English.

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
- Description hint: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban waterside area. A modern glass building reflects the golden light of the setting sun against a purple twilight sky, while people walk along the towpath, relax on the bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the lively evening scene as people go about their daily lives, commuting or enjoying leisure time.
- Capture metadata: Taken on 2026-03-21 18:22:22 GMT (at 18:22:22 local time). GPS: 51.536500°N, 0.126500°W.<|im_end|>
<|im_start|>assistant
<think>

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 16 files:   0%|          | 0/16 [00:00<?, ?it/s]
Fetching 16 files: 100%|##########| 16/16 [00:00<00:00, 18330.75it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 7. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected model:** `mlx-community/Qwen3.5-35B-A3B-bf16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | Failed to process inputs with error: Only returning PyTorch tensors is currently supported. | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-bf16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/Qwen3.5-35B-A3B-bf16`

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
ValueError: Model generation failed for mlx-community/Qwen3.5-35B-A3B-bf16: Failed to process inputs with error: Only returning PyTorch tensors is currently supported.
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '2', '1', '-', '1', '8', '2', '2', '2', '2', '_', 'D', 'S', 'C', '0', '9', '4', '8', '6', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata, using British English.

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
- Description hint: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban waterside area. A modern glass building reflects the golden light of the setting sun against a purple twilight sky, while people walk along the towpath, relax on the bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the lively evening scene as people go about their daily lives, commuting or enjoying leisure time.
- Capture metadata: Taken on 2026-03-21 18:22:22 GMT (at 18:22:22 local time). GPS: 51.536500°N, 0.126500°W.<|im_end|>
<|im_start|>assistant
<think>

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 24 files:   0%|          | 0/24 [00:00<?, ?it/s]
Fetching 24 files: 100%|##########| 24/24 [00:00<00:00, 13612.35it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

## 8. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `mlx-community/deepseek-vl2-8bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/deepseek-vl2-8bit` | Loaded processor has no image_processor; expected multimodal processor. | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

### To reproduce

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit`

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
Fetching 13 files: 100%|##########| 13/13 [00:00<00:00, 74693.08it/s]

Download complete: : 0.00B [00:00, ?B/s]              
Download complete: : 0.00B [00:00, ?B/s]
```

</details>

---

## Preflight Compatibility Warnings (1 issue(s))

These warnings were detected before inference. They are informational by default and do not invalidate successful runs on their own.
Keep running if outputs look healthy. Escalate only when the warnings line up with backend-import side effects, startup hangs, or runtime crashes.
Do not treat these warnings alone as a reason to set MLX_VLM_ALLOW_TF=1 or to assume the benchmark results are bad.

### `transformers`

- Suggested tracker: `transformers` (<https://github.com/huggingface/transformers/issues/new>)
- Suggested next action: verify API compatibility and pinned version floor.
- Triage guidance: continue if runs are otherwise healthy; investigate only if this warning matches real backend symptoms in the same run.
- Warnings:
  - `transformers import utils no longer reference the TF/FLAX/JAX backend guard env vars used by check_models; backend guard hints for those backends may be ignored with this version.`


---

## Harness/Integration Issues (8 model(s))

8 model(s) show potential harness/integration issues; see per-model breakdown below.
These models completed successfully but show integration problems (for example stop-token leakage, decoding artifacts, or long-context breakdown) that likely point to stack/runtime behavior rather than inherent model quality limits.

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,350, output=500, output/prompt=37.04%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Generated text appears to continue into example-code templates mid-output.
- Output switched language/script unexpectedly.

**Sample output:**

```text
Title: Urban Waterside Scene at Dusk

Description: A modern glass building reflects golden sunlight on a purple twilight sky, with pedestrians crossing a footbridge over a canal. Moored boats line the...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,626, output=93, output/prompt=3.54%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 51 occurrences).
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Title:ĠFootbridgeĠoverĠcanalĠatĠduskĊĊDescription:ĠAĠfootbridgeĠspansĠaĠcanalĠinĠanĠurbanĠwatersideĠareaĠduringĠtwilight.ĠPeopleĠwalkĠacrossĠtheĠbridgeĠandĠalongĠtheĠtowpath,ĠwhileĠmooredĠboatsĠlineĠt...
```

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,845, output=464, output/prompt=25.15%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.

**Sample output:**

```text
Let's tackle this step by step. First, the title needs to be concrete and factual, 5-10 words. The image is a dusk scene at a canal with a modern glass building, people walking, boats. So maybe "Dusk ...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=0, output=0, output/prompt=n/a

**Why this appears to be an integration/runtime issue:**

- Model returned zero output tokens.

**Sample output:**

```text
<empty output>
```

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,748, output=500, output/prompt=2.99%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16748 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.
- Output omitted required Title/Description/Keywords sections.
- Output leaked reasoning or prompt-template text.

**Sample output:**

```text
Got it, let's analyze the image. The image shows multiple weatherproofed, wooden, or maybe stone, houses with large windows. The description says "Pedestrians cross a footbridge over a canal at dusk i...
```

### `mlx-community/X-Reasoner-7B-8bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,757, output=500, output/prompt=2.98%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16757 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
Title:
- White and Black Pattern

Description:
- A repeating pattern of white and black stripes forms a continuous design, resembling a classic flag or banner. The stripes are evenly spaced, creating ...
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,556, output=9, output/prompt=0.58%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
- The image is in the daytime.
```

### `prince-canuma/Florence-2-large-ft`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,099, output=500, output/prompt=45.50%

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

### Long-Context Degradation / Potential Stack Issues (1 model(s))

1 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Owner |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- |
| `Qwen/Qwen3-VL-2B-Instruct` | 16,746 | 500 | 2.99% | Context echo under long prompt length | `transformers / mlx-vlm` |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each model appears).

**Regressions since previous run:** `mlx-community/InternVL3-8B-bf16`, `mlx-community/Molmo-7B-D-0924-bf16`
**Recoveries since previous run:** none

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/InternVL3-8B-bf16` | new regression | 2026-02-23 12:54:48 GMT | 2/3 recent runs failed |
| `mlx-community/Molmo-7B-D-0924-bf16` | new regression | 2026-03-22 01:27:09 GMT | 2/3 recent runs failed |
| `mlx-community/Qwen3.5-27B-4bit` | still failing | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-27B-mxfp8` | still failing | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | still failing | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | still failing | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |
| `mlx-community/deepseek-vl2-8bit` | new model failing | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 17
- **Summary diagnostics models:** 34
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 616.58s (616.58s)
- **Average runtime per model:** 12.09s (12.09s)
- **Dominant runtime phase:** decode dominated 46/51 measured model runs (82% of tracked runtime).
- **Phase totals:** model load=102.75s, prompt prep=0.12s, decode=500.08s, cleanup=6.14s
- **Observed stop reasons:** completed=43, exception=8
- **Validation overhead:** 13.41s total (avg 0.26s across 51 model(s)).
- **First-token latency:** Avg 2.97s | Min 0.09s | Max 16.19s across 42 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (34 model(s))

These models completed without diagnostics flags (no hard failure, harness warning, or stack-signal anomaly).

### Clean output (4 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
- `mlx-community/gemma-3-27b-it-qat-4bit`
- `mlx-community/gemma-3-27b-it-qat-8bit`

### Ran, but with quality warnings (30 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Model output may not follow prompt or image contents.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/FastVLM-0.5B-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/GLM-4.6V-Flash-6bit`: Output formatting deviated from the requested structure.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Output formatting deviated from the requested structure.
- `mlx-community/GLM-4.6V-nvfp4`: Output formatting deviated from the requested structure.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/InternVL3-14B-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/LFM2-VL-1.6B-8bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Output appears to copy prompt context verbatim.
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: Description sentence violation (3; expected 1-2)
- `mlx-community/Molmo-7B-D-0924-8bit`: Output appears to copy prompt context verbatim.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Nonvisual metadata borrowing
- `mlx-community/SmolVLM-Instruct-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E4B-it-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/llava-v1.6-mistral-7b-8bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/nanoLLaVA-1.5-4bit`: Output appears to copy prompt context verbatim.
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: Model output may not follow prompt or image contents.
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/pixtral-12b-8bit`: Title length violation (4 words; expected 5-10)
- `mlx-community/pixtral-12b-bf16`: Model output may not follow prompt or image contents.
- `qnguyen3/nanoLLaVA`: Model output may not follow prompt or image contents.

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.4.1 |
| mlx | 0.31.2.dev20260322+38ad2570 |
| mlx-lm | 0.31.2 |
| transformers | 5.3.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.7.2 |
| Python Version | 3.13.12 |
| OS | Darwin 25.3.0 |
| macOS Version | 26.3.1 |
| GPU/Chip | Apple M5 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/InternVL3-8B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Molmo-7B-D-0924-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-27B-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-27B-mxfp8
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-6bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
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
- Description hint: Pedestrians cross a footbridge over a canal at dusk in a vibrant urban waterside area. A modern glass building reflects the golden light of the setting sun against a purple twilight sky, while people walk along the towpath, relax on the bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the lively evening scene as people go about their daily lives, commuting or enjoying leisure time.
- Capture metadata: Taken on 2026-03-21 18:22:22 GMT (at 18:22:22 local time). GPS: 51.536500°N, 0.126500°W.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260321-182222_DSC09486_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-03-22 21:43:08 GMT by [check_models](https://github.com/jrp2014/check_models)._
