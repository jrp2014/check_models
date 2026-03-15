# Diagnostics Report — 1 failure(s), 5 harness issue(s) (mlx-vlm 0.4.0)

## Summary

Automated benchmarking of **50 locally-cached VLM models** found **1 hard failure(s)** and **5 harness/integration issue(s)** plus **1 preflight compatibility warning(s)** in successful models. 49 of 50 models succeeded.

Test image: `20260314-162100_DSC09442.jpg` (40.5 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [transformers]** Failed to process inputs with error: can only concatenate str (not "NoneType") to str (1 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 3 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 1 model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 1 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [transformers / mlx-vlm]** Stack-signal anomalies on 2 successful model(s). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **Medium** | Failed to process inputs with error: can only concatenate str (not "N... | 1 (Florence-2-large-ft) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | Harness/integration | 3 (Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, Florence-2-large-ft) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Harness/integration | 1 (Qwen3.5-35B-A3B-bf16) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Harness/integration | 1 (paligemma2-10b-ft-docci-448-6bit) | `model-config / mlx-vlm` | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Stack-signal anomaly | 2 (Qwen2-VL-2B-Instruct-4bit, Qwen3.5-27B-4bit) | `transformers / mlx-vlm` | verify API compatibility and pinned version floor. |
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

- Repro command (exact run): `python -m check_models --image /Users/jrp/Pictures/Processed/20260314-162100_DSC09442.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft`

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
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '3', '1', '4', '-', '1', '6', '2', '1', '0', '0', '_', 'D', 'S', 'C', '0', '9', '4', '4', '2', '.', 'j', 'p', 'g'] 

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
- Description hint: , Howardsgate, Welwyn Garden City, England, United Kingdom, UK On a clear late afternoon in early spring, a striking living wall brings a splash of nature to the urban landscape of Howardsgate in Welwyn Garden City, England. The low sun casts long shadows from the bare-branched trees onto the brick facade of a Sainsbury's supermarket, where shoppers are seen making their way home with their groceries. This modern...
- Keyword hints: Adobe Stock, Any Vision, Eco-friendly facade, England, Europe, Hertfordshire, Hertfordshire / England, Howardsgate, Locations, Modern retail building, Sainsbury's Supermarket, Sainsbury's Welwyn Garden City, Shoppers, Shopping Bags, Spring, Supermarket, Sustainable architecture, UK, United Kingdom, Urban Greening
- Capture metadata: Taken on 2026-03-14 16:21:00 GMT (at 16:21:00 local time). GPS: 51.800333°N, 0.207617°W.

=== STDERR ===
Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Downloading (incomplete total...):   0%|          | 0.00/51.0 [00:00<?, ?B/s]
Downloading (incomplete total...): 100%|##########| 51.0/51.0 [00:00<00:00, 311B/s]
Downloading (incomplete total...):  60%|######    | 51.0/85.0 [00:00<00:00, 311B/s]
Downloading (incomplete total...): 2.50kB [00:00, 311B/s]                          

Fetching 10 files:  10%|#         | 1/10 [00:00<00:01,  4.82it/s]
Downloading (incomplete total...): 1.41MB [00:00, 5.33MB/s]
Downloading (incomplete total...):   0%|          | 1.53M/1.54G [00:00<04:48, 5.33MB/s]
Downloading (incomplete total...):   0%|          | 2.63M/1.54G [00:00<03:24, 7.52MB/s]
Downloading (incomplete total...):   0%|          | 2.63M/1.54G [00:13<03:24, 7.52MB/s]
Downloading (incomplete total...):   5%|4         | 69.7M/1.54G [01:17<27:28, 892kB/s] 
Downloading (incomplete total...):   5%|4         | 69.7M/1.54G [01:33<27:28, 892kB/s]
Downloading (incomplete total...):  47%|####7     | 728M/1.54G [01:39<01:27, 9.32MB/s]
Downloading (incomplete total...):  61%|######1   | 947M/1.54G [01:51<00:54, 11.0MB/s]
Downloading (incomplete total...):  63%|######3   | 972M/1.54G [01:51<00:50, 11.3MB/s]
Downloading (incomplete total...):  64%|######4   | 991M/1.54G [01:55<00:50, 10.8MB/s]
Downloading (incomplete total...):  65%|######4   | 998M/1.54G [01:57<00:52, 10.3MB/s]
Downloading (incomplete total...):  68%|######8   | 1.05G/1.54G [02:03<00:51, 9.51MB/s]
Downloading (incomplete total...):  71%|#######1  | 1.10G/1.54G [02:08<00:45, 9.64MB/s]
Downloading (incomplete total...):  72%|#######1  | 1.11G/1.54G [02:10<00:46, 9.34MB/s]
Downloading (incomplete total...):  75%|#######4  | 1.15G/1.54G [02:13<00:39, 10.0MB/s]
Downloading (incomplete total...):  75%|#######5  | 1.16G/1.54G [02:17<00:46, 8.22MB/s]
Downloading (incomplete total...):  76%|#######6  | 1.17G/1.54G [02:21<00:53, 6.79MB/s]
Downloading (incomplete total...):  79%|#######8  | 1.22G/1.54G [02:23<00:36, 8.96MB/s]
Downloading (incomplete total...):  81%|########1 | 1.25G/1.54G [02:25<00:25, 11.3MB/s]
Downloading (incomplete total...):  82%|########2 | 1.27G/1.54G [02:27<00:26, 10.4MB/s]
Downloading (incomplete total...):  83%|########2 | 1.28G/1.54G [02:30<00:35, 7.49MB/s]
Downloading (incomplete total...):  84%|########4 | 1.30G/1.54G [02:32<00:27, 8.84MB/s]
Downloading (incomplete total...):  84%|########4 | 1.30G/1.54G [02:43<00:27, 8.84MB/s]
Downloading (incomplete total...):  88%|########8 | 1.36G/1.54G [02:48<00:35, 5.07MB/s]
Downloading (incomplete total...):  98%|#########7| 1.51G/1.54G [02:52<00:03, 11.8MB/s]
Downloading (incomplete total...): 1.54GB [02:54, 12.5MB/s]                            

Fetching 10 files:  40%|####      | 4/10 [02:54<04:43, 47.21s/it]
Fetching 10 files: 100%|##########| 10/10 [02:54<00:00, 17.47s/it]

Download complete: : 1.54GB [02:54, 12.5MB/s]
```

</details>

---

## Preflight Compatibility Warnings (1 issue(s))

These warnings were detected before inference. They are non-fatal but should be tracked as potential upstream compatibility issues.

### `transformers`

- Suggested tracker: `transformers` (<https://github.com/huggingface/transformers/issues/new>)
- Suggested next action: verify API compatibility and pinned version floor.
- Warnings:
  - `transformers import utils no longer reference known backend guard env vars (TRANSFORMERS_NO_* / USE_*); check_models backend guard hints may be ignored with this version.`


---

## Harness/Integration Issues (5 model(s))

5 model(s) show potential harness/integration issues; see per-model breakdown below.
These models completed successfully but show integration problems (for example stop-token leakage, decoding artifacts, or long-context breakdown) that likely point to stack/runtime behavior rather than inherent model quality limits.

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,444, output=500, output/prompt=34.63%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Special control token # INSTRUCTION appeared in generated text.
- Generated text appears to continue into instruction headers mid-output.
- Output contains corrupted or malformed text segments.
- Output switched language/script unexpectedly.

**Sample output:**

```text
Title: Sainsbury's Supermarket with Living Wall Facade

Description: A Sainsbury's supermarket in Howardsgate, Welwyn Garden City, England, features a prominent living wall on its facade. The building...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,710, output=102, output/prompt=3.76%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 65 occurrences).
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Title:ĠSainsbury'sĠSupermarketĠwithĠLivingĠWallĊĊDescription:ĠAĠSainsbury'sĠsupermarketĠbuildingĠfeaturesĠaĠlargeĠverticalĠgardenĠonĠitsĠfacade,ĠwithĠshoppersĠwalkingĠnearbyĠandĠcarsĠparkedĠinĠfront.Ġ...
```

### `mlx-community/Qwen3.5-35B-A3B-bf16`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,850, output=500, output/prompt=2.97%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16850 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
The user wants me to generate cataloguing metadata (Title, Description, Keywords) based on the provided image.
I need to follow strict rules:
- Only use clearly visible details.
- No guessing or infer...
```

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,627, output=14, output/prompt=0.86%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.9%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
- Use only the metadata that is clearly supported by the image.
```

### `prince-canuma/Florence-2-large-ft`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,181, output=500, output/prompt=42.34%

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

2 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Owner |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | 16,837 | 147 | 0.87% | Context echo under long prompt length | `transformers / mlx-vlm` |
| `mlx-community/Qwen3.5-27B-4bit` | 16,850 | 500 | 2.97% | Context echo under long prompt length | `transformers / mlx-vlm` |

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

- **Detailed diagnostics models:** 8
- **Summary diagnostics models:** 42
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1600.50s (1600.50s)
- **Average runtime per model:** 32.01s (32.01s)
- **Dominant runtime phase:** decode dominated 49/50 measured model runs (79% of tracked runtime).
- **Phase totals:** model load=330.67s, prompt prep=0.13s, decode=1253.70s, cleanup=4.33s
- **Observed stop reasons:** completed=49, exception=1
- **Validation overhead:** 15.71s total (avg 0.31s across 50 model(s)).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (42 model(s))

These models completed without diagnostics flags (no hard failure, harness warning, or stack-signal anomaly).

### Clean output (4 model(s))

- `mlx-community/InternVL3-8B-bf16`
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/pixtral-12b-8bit`
- `mlx-community/pixtral-12b-bf16`

### Ran, but with quality warnings (38 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents.
- `Qwen/Qwen3-VL-2B-Instruct`: Description sentence violation (3; expected 1-2)
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output leaked reasoning or prompt-template text.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Model output may not follow prompt or image contents.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output contains corrupted or malformed text segments.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/FastVLM-0.5B-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/GLM-4.6V-Flash-6bit`: Model output may not follow prompt or image contents.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Model output may not follow prompt or image contents.
- `mlx-community/GLM-4.6V-nvfp4`: Output formatting deviated from the requested structure.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Output formatting deviated from the requested structure.
- `mlx-community/InternVL3-14B-8bit`: Title length violation (3 words; expected 5-10)
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Output leaked reasoning or prompt-template text.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Output contains corrupted or malformed text segments.
- `mlx-community/LFM2-VL-1.6B-8bit`: Output appears to copy prompt context verbatim.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Description sentence violation (5; expected 1-2)
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: Model output may not follow prompt or image contents.
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: Description sentence violation (3; expected 1-2)
- `mlx-community/Molmo-7B-D-0924-8bit`: Model refused or deflected the requested task.
- `mlx-community/Molmo-7B-D-0924-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Description sentence violation (4; expected 1-2)
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Qwen3.5-27B-mxfp8`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Model refused or deflected the requested task.
- `mlx-community/SmolVLM-Instruct-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Model output may not follow prompt or image contents.
- `mlx-community/X-Reasoner-7B-8bit`: Title length violation (4 words; expected 5-10)
- `mlx-community/gemma-3-27b-it-qat-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3-27b-it-qat-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E4B-it-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/llava-v1.6-mistral-7b-8bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/nanoLLaVA-1.5-4bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents.
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260314-162100_DSC09442.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260314-162100_DSC09442.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
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
- Description hint: , Howardsgate, Welwyn Garden City, England, United Kingdom, UK On a clear late afternoon in early spring, a striking living wall brings a splash of nature to the urban landscape of Howardsgate in Welwyn Garden City, England. The low sun casts long shadows from the bare-branched trees onto the brick facade of a Sainsbury's supermarket, where shoppers are seen making their way home with their groceries. This modern...
- Keyword hints: Adobe Stock, Any Vision, Eco-friendly facade, England, Europe, Hertfordshire, Hertfordshire / England, Howardsgate, Locations, Modern retail building, Sainsbury's Supermarket, Sainsbury's Welwyn Garden City, Shoppers, Shopping Bags, Spring, Supermarket, Sustainable architecture, UK, United Kingdom, Urban Greening
- Capture metadata: Taken on 2026-03-14 16:21:00 GMT (at 16:21:00 local time). GPS: 51.800333°N, 0.207617°W.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260314-162100_DSC09442.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-03-15 02:55:35 GMT by [check_models](https://github.com/jrp2014/check_models)._
