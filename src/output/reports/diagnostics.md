# Diagnostics Report — 2 failure(s), 11 harness issue(s) (mlx-vlm 0.4.4)

## Summary

Automated benchmarking of **54 locally-cached VLM models** found **2 hard
failure(s)** and **11 harness/integration issue(s)** plus **1 preflight
compatibility warning(s)** in successful models. 52 of 54 models succeeded.

Test image: `20260418-174210_DSC09825.jpg` (46.8 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [model configuration/repository]** Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--g... (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 3 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 3 model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 5 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [transformers / mlx-vlm]** Stack-signal anomalies on 2 successful model(s). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority   | Issue                                                                    | Models Affected                                                                                                                                       | Owner                            | Next Action                                                                              |
|------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|------------------------------------------------------------------------------------------|
| **Medium** | Model loading failed: Config not found at /Users/jrp/.cache/huggingfa... | 1 (gemma-3-1b-it-GGUF)                                                                                                                                | `model configuration/repository` | verify model config, tokenizer files, and revision alignment.                            |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor.  | 1 (MolmoPoint-8B-fp16)                                                                                                                                | `model configuration/repository` | verify model config, tokenizer files, and revision alignment.                            |
| **Medium** | Harness/integration                                                      | 3 (Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, ERNIE-4.5-VL-28B-A3B-Thinking-bf16)                                              | `mlx-vlm`                        | check processor/chat-template wiring and generation kwargs.                              |
| **Medium** | Harness/integration                                                      | 3 (Qwen3-VL-2B-Instruct, Qwen3.5-35B-A3B-bf16, X-Reasoner-7B-8bit)                                                                                    | `mlx-vlm / mlx`                  | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.     |
| **Medium** | Harness/integration                                                      | 5 (FastVLM-0.5B-bf16, Qwen2-VL-2B-Instruct-4bit, paligemma2-10b-ft-docci-448-6bit, paligemma2-10b-ft-docci-448-bf16, paligemma2-3b-ft-docci-448-bf16) | `model-config / mlx-vlm`         | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Stack-signal anomaly                                                     | 2 (Qwen3.5-35B-A3B-6bit, Qwen3.5-9B-MLX-4bit)                                                                                                         | `transformers / mlx-vlm`         | verify API compatibility and pinned version floor.                                       |
| **Medium** | Preflight compatibility warning                                          | 1 issue(s)                                                                                                                                            | `transformers`                   | verify API compatibility and pinned version floor.                                       |

---

## 1. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `ggml-org/gemma-3-1b-it-GGUF`

| Model                         | Observed Behavior                                                                                                                                                   | First Seen Failing      | Recent Repro           |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `ggml-org/gemma-3-1b-it-GGUF` | Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27 | 2026-04-18 00:45:49 BST | 3/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `ggml-org/gemma-3-1b-it-GGUF`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `ggml-org/gemma-3-1b-it-GGUF`

Traceback tail:

```text
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 449, in load_config
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

## 2. Failure affecting 1 model (Priority: Medium)

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

## Harness/Integration Issues (11 model(s))

11 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,769, output=500, output/prompt=2.98%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16769 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: phrase: "sky, stone wall, town,...").

**Sample output:**

```text
Title:
Windsor Sovereign tour boat

Description:
The Windsor Sovereign tour boat, filled with passengers, cruises along the River Thames in Windsor, England. In the background, the historic Windsor Ca...
```

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,382, output=500, output/prompt=36.18%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Generated text appears to continue into example-code templates mid-output.
- Output switched language/script unexpectedly (tokenizer_artifact, code_snippet).
- Output formatting deviated from the requested structure. Details: Excessive markdown headers (6).

**Sample output:**

```text
Title: Tour Boat on River Thames

Description: A tour boat, the 'Windsor Sovereign', is seen cruising along the River Thames in Windsor, England. The boat is filled with passengers, and the historic W...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,654, output=107, output/prompt=4.03%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 61 occurrences).
- Output omitted required Title/Description/Keywords sections (description, keywords).

**Sample output:**

```text
Title:ĠWindsorĠSovereignĠTourĠBoatĠonĠtheĠRiverĠThamesĊĊDescription:ĠAĠblueĠandĠwhiteĠtourĠboatĠnamedĠ'WindsorĠSovereign'ĠisĠcruisingĠonĠtheĠRiverĠThamesĠwithĠpassengersĠonĠboard.ĠInĠtheĠbackground,ĠW...
```

### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,873, output=414, output/prompt=22.10%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.

**Sample output:**

```text
Let's tackle this step by step. First, the title needs to be 5-10 words, concrete and factual. The image is of a Windsor Sovereign tour boat on the River Thames with Windsor Castle in the background. ...
```

### `mlx-community/FastVLM-0.5B-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=556, output=8, output/prompt=1.44%

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 8 tokens.
- Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).

**Sample output:**

```text
0.0,0.0
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=16,780, output=12, output/prompt=0.07%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16780 tokens), output stayed unusually short (12 tokens; ratio 0.1%).
- Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- Output switched language/script unexpectedly (tokenizer_artifact).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
<|endoftext|>2021-10-02
```

### `mlx-community/Qwen3.5-35B-A3B-bf16`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,794, output=500, output/prompt=2.98%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16794 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
The user wants me to analyze an image and generate cataloguing metadata (Title, Description, Keywords) based on specific rules.

**1. Analyze the Image:**
*   **Content:** The image shows a collection...
```

### `mlx-community/X-Reasoner-7B-8bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,780, output=73, output/prompt=0.44%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16780 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).

**Sample output:**

```text
Title:
- White Striped Pattern

Description:
- A repeating pattern of white stripes on a dark background.

Keywords:
- Stripes
- White
- Dark
- Pattern
- Background
- Uniform
- Repetition
- Monochrome...
```

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,574, output=12, output/prompt=0.76%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.8%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
- Use the 'Windsor Sovereign' as the title.
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,574, output=9, output/prompt=0.57%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).

**Sample output:**

```text
- The image is in the daytime.
```

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,574, output=6, output/prompt=0.38%

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 6 tokens.
- Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).

**Sample output:**

```text
- Do not copy.
```

---

### Long-Context Degradation / Potential Stack Issues (2 model(s))

2 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely
integration/runtime issues worth checking upstream.

| Model                                |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                  | Owner                    |
|--------------------------------------|--------------|--------------|-----------------|------------------------------------------|--------------------------|
| `mlx-community/Qwen3.5-35B-A3B-6bit` |       16,794 |          500 | 2.98%           | Context dropped under long prompt length | `transformers / mlx-vlm` |
| `mlx-community/Qwen3.5-9B-MLX-4bit`  |       16,794 |          500 | 2.98%           | Context echo under long prompt length    | `transformers / mlx-vlm` |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** `Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/X-Reasoner-7B-8bit`

| Model                              | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|------------------------------------|--------------------------|-------------------------|------------------------|
| `ggml-org/gemma-3-1b-it-GGUF`      | still failing            | 2026-04-18 00:45:49 BST | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16` | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 15
- **Summary diagnostics models:** 39
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1253.00s (1253.00s)
- **Average runtime per model:** 23.20s (23.20s)
- **Dominant runtime phase:** decode dominated 50/54 measured model runs (89% of tracked runtime).
- **Phase totals:** model load=130.41s, prompt prep=0.18s, decode=1103.64s, cleanup=5.70s
- **Observed stop reasons:** completed=52, exception=2
- **Validation overhead:** 18.50s total (avg 0.34s across 54 model(s)).
- **First-token latency:** Avg 11.32s | Min 0.09s | Max 71.48s across 52 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (39 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (1 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

### Ran, but with quality warnings (38 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output became repetitive, indicating possible generation instability (token: phrase: "waterfront, waterfront, waterf....
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections (title, keywords).
- `mlx-community/GLM-4.6V-Flash-6bit`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/GLM-4.6V-nvfp4`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `mlx-community/InternVL3-14B-8bit`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `mlx-community/InternVL3-8B-bf16`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `mlx-community/LFM2-VL-1.6B-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "windsor, england, river, thame....
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output became repetitive, indicating possible generation instability (token: phrase: "riverbank, riverbank, riverban....
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: ⚠️REVIEW:context_budget
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: ⚠️REVIEW:context_budget
- `mlx-community/Molmo-7B-D-0924-8bit`: Output leaked reasoning or prompt-template text (title hint:).
- `mlx-community/Molmo-7B-D-0924-bf16`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Keyword count violation (19; expected 10-18)
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-27B-4bit`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-27B-mxfp8`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-35B-A3B-4bit`: Output omitted required Title/Description/Keywords sections (description, keywords).
- `mlx-community/SmolVLM-Instruct-bf16`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `mlx-community/gemma-3-27b-it-qat-4bit`: Nonvisual metadata borrowing
- `mlx-community/gemma-3-27b-it-qat-8bit`: Nonvisual metadata borrowing
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `mlx-community/gemma-3n-E4B-it-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/gemma-4-31b-bf16`: Output formatting deviated from the requested structure. Details: Unknown tags: <img src="<https://www.anyvision.co.uk...>
- `mlx-community/gemma-4-31b-it-4bit`: Nonvisual metadata borrowing
- `mlx-community/llava-v1.6-mistral-7b-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/nanoLLaVA-1.5-4bit`: Output omitted required Title/Description/Keywords sections (keywords).
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `mlx-community/pixtral-12b-8bit`: ⚠️REVIEW:context_budget
- `mlx-community/pixtral-12b-bf16`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).
- `qnguyen3/nanoLLaVA`: Model output may not follow prompt or image contents (missing: Berkshire, Bird, Blue sky, Castle, Holiday).

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.4                       |
| mlx             | 0.31.2.dev20260419+fa4320d5 |
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-174210_DSC09825.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-174210_DSC09825.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models ggml-org/gemma-3-1b-it-GGUF
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-174210_DSC09825.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
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
- Description hint: The 'Windsor Sovereign' tour boat, filled with passengers, cruises along the River Thames in Windsor, England. In the background, the historic Windsor Castle is visible behind the trees and town buildings, offering a scenic backdrop for tourists enjoying a river tour.
- Keyword hints: Adobe Stock, Any Vision, Berkshire, Bird, Blue sky, Castle, England, Europe, Holiday, Lifebuoy, Passenger, People, Quay, River Thames, Riverbank, Sightseeing, Sky, Stone wall, Town, Tree
- Capture metadata: Taken on 2026-04-18 18:42:10 BST (at 18:42:10 local time). GPS: 51.483900°N, 0.604400°W.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260418-174210_DSC09825.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-04-19 21:12:47 BST by [check_models](https://github.com/jrp2014/check_models)._
