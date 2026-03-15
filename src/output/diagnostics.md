# Diagnostics Report — 1 failure(s), 11 harness issue(s) (mlx-vlm 0.4.0)

## Summary

Automated benchmarking of **50 locally-cached VLM models** found **1 hard failure(s)** and **11 harness/integration issue(s)** plus **1 preflight compatibility warning(s)** in successful models. 49 of 50 models succeeded.

Test image: `20260314-161959_DSC09441.jpg` (39.3 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [mlx-vlm]** Failed to process inputs with error: can only concatenate str (not "NoneType") to str (1 model(s)). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 11 model(s). Next: validate stop-token decoding and long-context behavior.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify dependency/version compatibility before model runs.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **Medium** | Failed to process inputs with error: can only concatenate str (not "N... | 1 (Florence-2-large-ft) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Harness/integration | 11 (Qwen3-VL-2B-Instruct, Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, Qwen2-VL-2B-Instruct-4bit, Qwen3-VL-2B-Thinking-bf16, Qwen3.5-27B-mxfp8, X-Reasoner-7B-8bit, llava-v1.6-mistral-7b-8bit, paligemma2-10b-ft-docci-448-6bit, paligemma2-10b-ft-docci-448-bf16, Florence-2-large-ft) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Preflight compatibility warning | 1 issue(s) | `transformers` | verify dependency/version compatibility before model runs. |

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

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260314-161959_DSC09441.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `microsoft/Florence-2-large-ft`

Traceback:

```text
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 873, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 859, in process_inputs
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
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 886, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 859, in process_inputs
    return process_method(**args)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 185, in __call__
    self.image_token * self.num_image_tokens
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + self.tokenizer.bos_token
    ^~~~~~~~~~~~~~~~~~~~~~~~~~
TypeError: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 672, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 515, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1110, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 896, in process_inputs_with_fallback
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
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '1', '4', '-', '1', '6', '1', '9', '5', '9', '_', 'D', 'S', 'C', '0', '9', '4', '4', '1', '.', 'j', 'p', 'g'] 

Prompt: Analyze this image for cataloguing metadata.

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
- Description hint: , Howardsgate, Welwyn Garden City, England, United Kingdom, UK The late afternoon sun casts a warm glow on The Doctors Tonic pub in Welwyn Garden City, England. Captured on a clear day in early spring, the image highlights the contrast between the still-bare trees of March and the deep green of the manicured hedges. The pub's traditional brick architecture, with its tiled roof and dormer windows, stands out agains...
- Keyword hints: Adobe Stock, Any Vision, British, Chalkboard sign, England, English Pub Exterior, English pub, Europe, Food and drink, Greene King, Hertfordshire, Howardsgate, Signage, Spring afternoon, Sunday Roast sign, The Doctors Tonic, The Doctors Tonic pub, Traditional British Architecture, UK, United Kingdom
- Capture metadata: Taken on 2026-03-14 16:19:59 GMT (at 16:19:59 local time). GPS: 51.800450°N, 0.207617°W.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Downloading (incomplete total...):   0%|          | 0.00/1.54G [00:00<?, ?B/s]
Downloading (incomplete total...):   0%|          | 2.44k/1.54G [00:00<114:12:36, 3.75kB/s]

Fetching 10 files:  10%|#         | 1/10 [00:00<00:05,  1.53it/s]
Downloading (incomplete total...):   0%|          | 2.44k/1.54G [00:00<114:12:36, 3.75kB/s]
Downloading (incomplete total...):   0%|          | 3.25k/1.54G [00:00<114:12:36, 3.75kB/s]
Downloading (incomplete total...):   0%|          | 3.29k/1.54G [00:00<94:44:07, 4.52kB/s] 
Downloading (incomplete total...):   0%|          | 131k/1.54G [00:00<94:43:39, 4.52kB/s] 

Fetching 10 files:  30%|###       | 3/10 [00:00<00:01,  4.71it/s]
Downloading (incomplete total...):   0%|          | 1.53M/1.54G [00:00<08:40, 2.96MB/s]  
Downloading (incomplete total...):   0%|          | 2.63M/1.54G [00:01<06:11, 4.15MB/s]

Fetching 10 files:  30%|###       | 3/10 [00:12<00:01,  4.71it/s]
Downloading (incomplete total...):   0%|          | 2.63M/1.54G [00:12<06:11, 4.15MB/s]
Downloading (incomplete total...):   5%|4         | 69.7M/1.54G [00:16<05:46, 4.24MB/s]
Downloading (incomplete total...):   5%|4         | 69.7M/1.54G [00:29<05:46, 4.24MB/s]
Downloading (incomplete total...):  57%|#####6    | 872M/1.54G [00:40<00:27, 24.6MB/s] 
Downloading (incomplete total...): 1.54GB [00:40, 52.0MB/s]                           

Fetching 10 files:  40%|####      | 4/10 [00:40<01:22, 13.80s/it]
Fetching 10 files: 100%|##########| 10/10 [00:40<00:00,  4.02s/it]

Download complete: : 1.54GB [00:40, 52.0MB/s]
```

</details>

---

## Preflight Compatibility Warnings (1 issue(s))

These warnings were detected before inference. They are non-fatal but should be tracked as potential upstream compatibility issues.

- `transformers import utils no longer reference known backend guard env vars (TRANSFORMERS_NO_* / USE_*); check_models backend guard hints may be ignored with this version.`
  - Owner: `transformers`; suggested tracker: `transformers` (<https://github.com/huggingface/transformers/issues/new>)
  - Suggested next action: verify API compatibility and pinned version floor.

---

## Harness/Integration Issues (11 model(s))

11 model(s) show potential harness/integration issues; see per-model breakdown below.
These models completed successfully but show integration problems (for example stop-token leakage, decoding artifacts, or long-context breakdown) that likely point to stack/runtime behavior rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,825, output=500, output/prompt=2.97%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16825 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
Title:
The Doctors Tonic pub in Welwyn Garden City

Description:
The Doctors Tonic pub in Welwyn Garden City, England, features a traditional brick building with a tiled roof and dormer windows, surro...
```

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,433, output=500, output/prompt=34.89%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Generated text appears to continue into example-code templates mid-output.
- Output switched language/script unexpectedly.

**Sample output:**

```text
Title: The Doctors Tonic Pub in Welwyn Garden City

Description: The Doctors Tonic pub is a traditional brick building with a tiled roof and dormer windows, located in Welwyn Garden City, England. The...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,707, output=111, output/prompt=4.10%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 75 occurrences).
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Title:ĠTheĠDoctorsĠTonicĠPubĠExteriorĊĊDescription:ĠTheĠimageĠshowsĠtheĠexteriorĠofĠTheĠDoctorsĠTonicĠpub,ĠaĠtraditionalĠbrickĠbuildingĠwithĠaĠtiledĠroofĠandĠdormerĠwindows.ĠTheĠsceneĠisĠsetĠonĠaĠclea...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,836, output=500, output/prompt=2.97%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16836 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.
- Output appears to copy prompt context verbatim.

**Sample output:**

```text
Title: The Doctors Tonic Pub, Welwyn Garden City, England, UK
Description: The late afternoon sun casts a warm glow on The Doctors Tonic pub in Welwyn Garden City, England. The pub's traditional brick...
```

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,827, output=500, output/prompt=2.97%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16827 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
Got it, let's tackle this step by step. First, I need to analyze the image for metadata. The user provided a lot of text, but I need to focus on what's clearly visible.

Title: Needs 5-10 words, factu...
```

### `mlx-community/Qwen3.5-27B-mxfp8`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,850, output=500, output/prompt=2.97%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16850 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
The user wants me to generate cataloguing metadata for the provided image.
I need to follow specific rules:
- Only use clearly visible details.
- No guessing or inferring.
- Output exactly three secti...
```

### `mlx-community/X-Reasoner-7B-8bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,836, output=500, output/prompt=2.97%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16836 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
Title:
The Doctors Tonic Pub

Description:
A traditional brick pub with a tiled roof and dormer windows, featuring a sign reading "The Doctors Tonic" and offering food, cask ales, and a Sunday roast. ...
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,820, output=8, output/prompt=0.28%

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 8 tokens.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
The Doctors Tonic.
```

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,627, output=12, output/prompt=0.74%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
- Use existing metadata hints to create a catalog record.
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,627, output=13, output/prompt=0.80%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
- Use existing metadata hints to create a draft catalog record.
```

### `prince-canuma/Florence-2-large-ft`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,180, output=500, output/prompt=42.37%

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
| `microsoft/Florence-2-large-ft` | new model failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 12
- **Summary diagnostics models:** 38
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1798.38s (1798.38s)
- **Average runtime per model:** 35.97s (35.97s)
- **Dominant runtime phase:** decode dominated 49/50 measured model runs (88% of tracked runtime).
- **Phase totals:** model load=209.73s, prompt prep=0.13s, decode=1570.87s, cleanup=5.42s
- **Observed stop reasons:** completed=49, exception=1
- **Validation overhead:** 17.29s total (avg 0.35s across 50 model(s)).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (38 model(s))

These models completed without diagnostics flags (no hard failure, harness warning, or stack-signal anomaly).

### Clean output (1 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

### Ran, but with quality warnings (37 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output leaked reasoning or prompt-template text.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Model output may not follow prompt or image contents.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Model output may not follow prompt or image contents.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Keyword count violation (48; expected 10-18)
- `mlx-community/FastVLM-0.5B-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/GLM-4.6V-Flash-6bit`: Model output may not follow prompt or image contents.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Output formatting deviated from the requested structure.
- `mlx-community/GLM-4.6V-nvfp4`: Output formatting deviated from the requested structure.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Output formatting deviated from the requested structure.
- `mlx-community/InternVL3-14B-8bit`: Title length violation (4 words; expected 5-10)
- `mlx-community/InternVL3-8B-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Output leaked reasoning or prompt-template text.
- `mlx-community/LFM2-VL-1.6B-8bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Description sentence violation (4; expected 1-2)
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: Model output may not follow prompt or image contents.
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: Description sentence violation (3; expected 1-2)
- `mlx-community/Molmo-7B-D-0924-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/Molmo-7B-D-0924-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Description sentence violation (3; expected 1-2)
- `mlx-community/Qwen3.5-27B-4bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Qwen3.5-35B-A3B-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/SmolVLM-Instruct-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/gemma-3-27b-it-qat-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3-27b-it-qat-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E4B-it-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/nanoLLaVA-1.5-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/pixtral-12b-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/pixtral-12b-bf16`: Model output may not follow prompt or image contents.
- `qnguyen3/nanoLLaVA`: Output omitted required Title/Description/Keywords sections.

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.4.0 |
| mlx | 0.31.2.dev20260314+5d170049 |
| mlx-lm | 0.31.2 |
| transformers | 5.3.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.7.1 |
| Python Version | 3.13.12 |
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260314-161959_DSC09441.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260314-161959_DSC09441.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
```

### Prompt Used

```text
Analyze this image for cataloguing metadata.

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
- Description hint: , Howardsgate, Welwyn Garden City, England, United Kingdom, UK The late afternoon sun casts a warm glow on The Doctors Tonic pub in Welwyn Garden City, England. Captured on a clear day in early spring, the image highlights the contrast between the still-bare trees of March and the deep green of the manicured hedges. The pub's traditional brick architecture, with its tiled roof and dormer windows, stands out agains...
- Keyword hints: Adobe Stock, Any Vision, British, Chalkboard sign, England, English Pub Exterior, English pub, Europe, Food and drink, Greene King, Hertfordshire, Howardsgate, Signage, Spring afternoon, Sunday Roast sign, The Doctors Tonic, The Doctors Tonic pub, Traditional British Architecture, UK, United Kingdom
- Capture metadata: Taken on 2026-03-14 16:19:59 GMT (at 16:19:59 local time). GPS: 51.800450°N, 0.207617°W.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260314-161959_DSC09441.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-03-15 00:03:29 GMT by [check_models](https://github.com/jrp2014/check_models)._
