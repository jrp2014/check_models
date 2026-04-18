# Diagnostics Report — 6 failure(s), 2 harness issue(s) (mlx-vlm 0.4.4)

## Summary

Automated benchmarking of **54 locally-cached VLM models** found **6 hard
failure(s)** and **2 harness/integration issue(s)** plus **1 preflight
compatibility warning(s)** in successful models. 48 of 54 models succeeded.

Test image: `20260411-165312_DSC09747_DxO.jpg` (37.2 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[High] [mlx-vlm]** [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast. (4 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [model configuration/repository]** Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--g... (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 1 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 1 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority   | Issue                                                                    | Models Affected                                                                                    | Owner                            | Next Action                                                                              |
|------------|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------|------------------------------------------------------------------------------------------|
| **High**   | [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadc... | 4 (Qwen3-VL-2B-Instruct, Qwen2-VL-2B-Instruct-4bit, Qwen3-VL-2B-Thinking-bf16, X-Reasoner-7B-8bit) | `mlx-vlm`                        | check processor/chat-template wiring and generation kwargs.                              |
| **Medium** | Model loading failed: Config not found at /Users/jrp/.cache/huggingfa... | 1 (gemma-3-1b-it-GGUF)                                                                             | `model configuration/repository` | verify model config, tokenizer files, and revision alignment.                            |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor.  | 1 (MolmoPoint-8B-fp16)                                                                             | `model configuration/repository` | verify model config, tokenizer files, and revision alignment.                            |
| **Medium** | Harness/integration                                                      | 1 (Devstral-Small-2-24B-Instruct-2512-5bit)                                                        | `mlx-vlm`                        | check processor/chat-template wiring and generation kwargs.                              |
| **Medium** | Harness/integration                                                      | 1 (paligemma2-10b-ft-docci-448-bf16)                                                               | `model-config / mlx-vlm`         | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Preflight compatibility warning                                          | 1 issue(s)                                                                                         | `transformers`                   | verify API compatibility and pinned version floor.                                       |

---

## 1. Failure affecting 4 models (Priority: High)

**Observed behavior:** [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.
**Owner (likely component):** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Affected models:** `Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/X-Reasoner-7B-8bit`

| Model                                     | Observed Behavior                                                         | First Seen Failing      | Recent Repro           |
|-------------------------------------------|---------------------------------------------------------------------------|-------------------------|------------------------|
| `Qwen/Qwen3-VL-2B-Instruct`               | [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast. | 2026-04-17 13:13:03 BST | 2/3 recent runs failed |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | [broadcast_shapes] Shapes (3,1,4096) and (3,1,16680) cannot be broadcast. | 2026-04-17 13:13:03 BST | 2/3 recent runs failed |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast. | 2026-04-17 13:13:03 BST | 2/3 recent runs failed |
| `mlx-community/X-Reasoner-7B-8bit`        | [broadcast_shapes] Shapes (3,1,4096) and (3,1,16239) cannot be broadcast. | 2026-02-08 22:32:28 GMT | 2/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `Qwen/Qwen3-VL-2B-Instruct`

<details>
<summary>Detailed trace logs (affected models)</summary>

#### `Qwen/Qwen3-VL-2B-Instruct`

Traceback tail:

```text
        expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
    )
ValueError: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model generation failed for Qwen/Qwen3-VL-2B-Instruct: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '4', '1', '1', '-', '1', '6', '5', '3', '1', '2', '_', 'D', 'S', 'C', '0', '9', '7', '4', '7', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g']

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
- Description hint: , Town Centre, Alton, England, United Kingdom, UK
- Capture metadata: Taken on 2026-04-11 17:53:12 BST (at 17:53:12 local time). GPS: 51.145067°N, 0.980317°W.<|im_end|>
<|im_start|>assistant

=== STDERR ===

Prefill:   0%|          | 0/16675 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/16675 [00:00<?, ?tok/s]
```

#### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

Traceback tail:

```text
        expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
    )
ValueError: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16680) cannot be broadcast.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model generation failed for mlx-community/Qwen2-VL-2B-Instruct-4bit: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16680) cannot be broadcast.
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '4', '1', '1', '-', '1', '6', '5', '3', '1', '2', '_', 'D', 'S', 'C', '0', '9', '7', '4', '7', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g']

Prompt: <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
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
- Description hint: , Town Centre, Alton, England, United Kingdom, UK
- Capture metadata: Taken on 2026-04-11 17:53:12 BST (at 17:53:12 local time). GPS: 51.145067°N, 0.980317°W.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant

=== STDERR ===

Prefill:   0%|          | 0/16686 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/16686 [00:00<?, ?tok/s]
```

#### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

Traceback tail:

```text
        expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
    )
ValueError: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model generation failed for mlx-community/Qwen3-VL-2B-Thinking-bf16: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '4', '1', '1', '-', '1', '6', '5', '3', '1', '2', '_', 'D', 'S', 'C', '0', '9', '7', '4', '7', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g']

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
- Description hint: , Town Centre, Alton, England, United Kingdom, UK
- Capture metadata: Taken on 2026-04-11 17:53:12 BST (at 17:53:12 local time). GPS: 51.145067°N, 0.980317°W.<|im_end|>
<|im_start|>assistant
<think>

=== STDERR ===

Prefill:   0%|          | 0/16677 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/16677 [00:00<?, ?tok/s]
```

#### `mlx-community/X-Reasoner-7B-8bit`

Traceback tail:

```text
        expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
    )
ValueError: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16239) cannot be broadcast.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model generation failed for mlx-community/X-Reasoner-7B-8bit: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16239) cannot be broadcast.
```

Captured stdout/stderr:

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '4', '1', '1', '-', '1', '6', '5', '3', '1', '2', '_', 'D', 'S', 'C', '0', '9', '7', '4', '7', '_', 'D', 'x', 'O', '.', 'j', 'p', 'g']

Prompt: <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
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
- Description hint: , Town Centre, Alton, England, United Kingdom, UK
- Capture metadata: Taken on 2026-04-11 17:53:12 BST (at 17:53:12 local time). GPS: 51.145067°N, 0.980317°W.<|im_end|>
<|im_start|>assistant

=== STDERR ===

Prefill:   0%|          | 0/16686 [00:00<?, ?tok/s]
Prefill:   0%|          | 0/16686 [00:00<?, ?tok/s]
```

</details>

## 2. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `ggml-org/gemma-3-1b-it-GGUF`

| Model                         | Observed Behavior                                                                                                                                                   | First Seen Failing      | Recent Repro           |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `ggml-org/gemma-3-1b-it-GGUF` | Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27 | 2026-04-18 00:45:49 BST | 1/1 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `ggml-org/gemma-3-1b-it-GGUF`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `ggml-org/gemma-3-1b-it-GGUF`

Traceback tail:

```text
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 456, in load_config
    raise FileNotFoundError(f"Config not found at {model_path}") from exc
FileNotFoundError: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27
```

Captured stdout/stderr:

```text
=== STDERR ===
```

</details>

## 3. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `mlx-community/MolmoPoint-8B-fp16`

| Model                              | Observed Behavior                                                       | First Seen Failing      | Recent Repro           |
|------------------------------------|-------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | Loaded processor has no image_processor; expected multimodal processor. | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `mlx-community/MolmoPoint-8B-fp16`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/MolmoPoint-8B-fp16`

Traceback tail:

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

/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/modeling_rope_utils.py:1032: FutureWarning: `rope_config_validation` is deprecated and has been removed. Its functionality has been moved to RotaryEmbeddingConfigMixin.validate_rope method. PreTrainedConfig inherits this class, so please call self.validate_rope() instead. Also, make sure to use the new rope_parameters syntax. You can call self.standardize_rope_params() in the meantime.
  warnings.warn(
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

## Harness/Integration Issues (2 model(s))

2 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,569, output=77, output/prompt=3.00%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 53 occurrences).
- Output omitted required Title/Description/Keywords sections (description, keywords).

**Sample output:**

```text
Title:ĠStoneĠChurchĠwithĠTowerĠandĠSpireĊĊDescription:ĠAĠstoneĠchurchĠwithĠaĠtallĠspireĠandĠclockĠtowerĠstandsĠatĠaĠroadĠjunctionĠunderĠaĠclearĠblueĠsky.ĠTheĠbuildingĠfeaturesĠarchedĠwindowsĠandĠaĠred...
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,484, output=12, output/prompt=0.81%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues.
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
- Camera: Canon EOS 5D Mark IV.
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

| Model                                     | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------------|--------------------------|-------------------------|------------------------|
| `Qwen/Qwen3-VL-2B-Instruct`               | still failing            | 2026-04-17 13:13:03 BST | 2/3 recent runs failed |
| `ggml-org/gemma-3-1b-it-GGUF`             | new model failing        | 2026-04-18 00:45:49 BST | 1/1 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`        | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | still failing            | 2026-04-17 13:13:03 BST | 2/3 recent runs failed |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | still failing            | 2026-04-17 13:13:03 BST | 2/3 recent runs failed |
| `mlx-community/X-Reasoner-7B-8bit`        | still failing            | 2026-02-08 22:32:28 GMT | 2/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 8
- **Summary diagnostics models:** 46
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1894.62s (1894.62s)
- **Average runtime per model:** 35.09s (35.09s)
- **Dominant runtime phase:** decode dominated 51/54 measured model runs (93% of tracked runtime).
- **Phase totals:** model load=127.00s, prompt prep=0.18s, decode=1751.41s, cleanup=7.96s
- **Observed stop reasons:** completed=48, exception=6
- **Validation overhead:** 15.71s total (avg 0.29s across 54 model(s)).
- **First-token latency:** Avg 14.43s | Min 0.11s | Max 139.53s across 48 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (46 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (1 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

### Ran, but with quality warnings (45 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Output became repetitive, indicating possible generation instability (token: unt).
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output became repetitive, indicating possible generation instability (token: phrase: "stone church with clock...").
- `microsoft/Phi-3.5-vision-instruct`: Output became repetitive, indicating possible generation instability (token: phrase: "creative visual art design,...").
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'a').
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/FastVLM-0.5B-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/GLM-4.6V-Flash-6bit`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/GLM-4.6V-nvfp4`: Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'is').
- `mlx-community/Idefics3-8B-Llama3-bf16`: Output formatting deviated from the requested structure. Details: Unknown tags: <fake_token_around_image>.
- `mlx-community/InternVL3-14B-8bit`: Title length violation (2 words; expected 5-10)
- `mlx-community/InternVL3-8B-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Output contains corrupted or malformed text segments (character_loop: '.0' repeated).
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Output became repetitive, indicating possible generation instability (token: prize.th).
- `mlx-community/LFM2-VL-1.6B-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "easily visible, easily visible....
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: Title length violation (12 words; expected 5-10)
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: Description sentence violation (3; expected 1-2)
- `mlx-community/Molmo-7B-D-0924-8bit`: Output became repetitive, indicating possible generation instability (token: phrase: "uk - capture metadata:...").
- `mlx-community/Molmo-7B-D-0924-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: ",… ,… ,… ,…...").
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "creative visual art design,...").
- `mlx-community/Qwen3.5-27B-4bit`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-27B-mxfp8`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-35B-A3B-4bit`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-35B-A3B-bf16`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-9B-MLX-4bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/SmolVLM-Instruct-bf16`: Output became repetitive, indicating possible generation instability (token: unt).
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Output formatting deviated from the requested structure. Details: Unknown tags: <row_1_col_1>.
- `mlx-community/gemma-3-27b-it-qat-4bit`: Nonvisual metadata borrowing
- `mlx-community/gemma-3-27b-it-qat-8bit`: Nonvisual metadata borrowing
- `mlx-community/gemma-3n-E2B-4bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/gemma-3n-E4B-it-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/gemma-4-31b-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "church clock face, church...").
- `mlx-community/gemma-4-31b-it-4bit`: Nonvisual metadata borrowing
- `mlx-community/llava-v1.6-mistral-7b-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/nanoLLaVA-1.5-4bit`: Output omitted required Title/Description/Keywords sections (keywords).
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "- 100% real-time. -...").
- `mlx-community/paligemma2-3b-pt-896-4bit`: Output became repetitive, indicating possible generation instability (token: phrase: "- output only the...").
- `mlx-community/pixtral-12b-8bit`: Description sentence violation (3; expected 1-2)
- `mlx-community/pixtral-12b-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `qnguyen3/nanoLLaVA`: Output became repetitive, indicating possible generation instability (token: Painting).

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.4                       |
| mlx             | 0.31.2.dev20260417+d142de6a |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.5.4                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.11.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.4.0               |
| macOS Version   | 26.4.1                      |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260411-165312_DSC09747_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260411-165312_DSC09747_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models Qwen/Qwen3-VL-2B-Instruct
python -m check_models --image /Users/jrp/Pictures/Processed/20260411-165312_DSC09747_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models ggml-org/gemma-3-1b-it-GGUF
python -m check_models --image /Users/jrp/Pictures/Processed/20260411-165312_DSC09747_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
python -m check_models --image /Users/jrp/Pictures/Processed/20260411-165312_DSC09747_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen2-VL-2B-Instruct-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260411-165312_DSC09747_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3-VL-2B-Thinking-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260411-165312_DSC09747_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/X-Reasoner-7B-8bit
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
- Description hint: , Town Centre, Alton, England, United Kingdom, UK
- Capture metadata: Taken on 2026-04-11 17:53:12 BST (at 17:53:12 local time). GPS: 51.145067°N, 0.980317°W.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260411-165312_DSC09747_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-04-18 00:45:49 BST by [check_models](https://github.com/jrp2014/check_models)._
