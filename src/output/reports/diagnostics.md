# Diagnostics Report — 2 failure(s), 9 harness issue(s) (mlx-vlm 0.4.5)

## Summary

Automated benchmarking of **53 locally-cached VLM models** found **2 hard
failure(s)** and **9 harness/integration issue(s)** plus **0 preflight
compatibility warning(s)** in successful models. 51 of 53 models succeeded.

Test image: `20260403-124049_DSC09541.jpg` (47.5 MB).

---

## Issue Queue

Root-cause issue drafts are generated in [issues/index.md](../issues/index.md)
and grouped by owner, subtype, and normalized symptom family.

| Owner                            | Issue Subtype                           |   Affected Model Count | Representative Model                                    | Issue Draft                                                                                                                                                                   | Acceptance Signal                                                                                                                              |
|----------------------------------|-----------------------------------------|------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx`                            | `MLX_MODEL_LOAD_MODEL`                  |                      1 | `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | [`mlx_mlx-model-load-model_001`](../issues/issue_001_mlx_mlx-model-load-model_001.md)                                                                                         | Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer. |
| `model configuration/repository` | `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR` |                      1 | `mlx-community/MolmoPoint-8B-fp16`                      | [`model-configuration-repository_model-config-processor-load-processor_001`](../issues/issue_002_model-configuration-repository_model-config-processor-load-processor_001.md) | Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer. |
| `mlx-vlm`                        | `encoding`                              |                      1 | `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [`mlx-vlm_encoding_001`](../issues/issue_003_mlx-vlm_encoding_001.md)                                                                                                         | Affected reruns contain no leaked BPE, byte-level, or tokenizer marker text.                                                                   |
| `mlx-vlm`                        | `stop_token`                            |                      4 | `microsoft/Phi-3.5-vision-instruct`                     | [`mlx-vlm_stop-token_001`](../issues/issue_004_mlx-vlm_stop-token_001.md)                                                                                                     | Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete. |
| `model-config / mlx-vlm`         | `prompt_template`                       |                      2 | `mlx-community/gemma-3n-E2B-4bit`                       | [`model-config-mlx-vlm_prompt-template_001`](../issues/issue_005_model-config-mlx-vlm_prompt-template_001.md)                                                                 | Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.          |
| `mlx-vlm / mlx`                  | `long_context`                          |                      2 | `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | [`mlx-vlm-mlx_long-context_001`](../issues/issue_006_mlx-vlm-mlx_long-context_001.md)                                                                                         | A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.               |
| `mlx-vlm / mlx`                  | `long_context`                          |                      1 | `mlx-community/Qwen3.5-9B-MLX-4bit`                     | [`mlx-vlm-mlx_long-context_002`](../issues/issue_007_mlx-vlm-mlx_long-context_002.md)                                                                                         | A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.               |

---

## Action Summary

Owner-first triage with subtype, affected count, and next action.

### 1. mlx

- _Owner:_ `mlx`
- _Subtype:_ MLX_MODEL_LOAD_MODEL
- _Issue:_ Model loading failed: Received 4 parameters not in model:
- _Affected:_ 1 model(s)
- _Next step:_ check tensor/cache behavior and memory pressure handling.

### 2. model configuration/repository

- _Owner:_ `model configuration/repository`
- _Subtype:_ MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- _Issue:_ Loaded processor has no image_processor; expected multimodal
  processor.
- _Affected:_ 1 model(s)
- _Next step:_ verify model config, tokenizer files, and revision alignment.

### 3. mlx-vlm

- _Owner:_ `mlx-vlm`
- _Subtype:_ harness/integration
- _Issue:_ Harness/integration warnings
- _Affected:_ 5 model(s)
- _Next step:_ check processor/chat-template wiring and generation kwargs.

### 4. mlx-vlm / mlx

- _Owner:_ `mlx-vlm / mlx`
- _Subtype:_ harness/integration
- _Issue:_ Harness/integration warnings
- _Affected:_ 2 model(s)
- _Next step:_ validate long-context handling and stop-token behavior across
  mlx-vlm + mlx runtime.

### 5. model-config / mlx-vlm

- _Owner:_ `model-config / mlx-vlm`
- _Subtype:_ harness/integration
- _Issue:_ Harness/integration warnings
- _Affected:_ 2 model(s)
- _Next step:_ validate chat-template/config expectations and mlx-vlm prompt
  formatting for this model.

### 6. mlx-vlm / mlx

- _Owner:_ `mlx-vlm / mlx`
- _Subtype:_ stack-signal
- _Issue:_ Stack-signal anomalies
- _Affected:_ 1 successful model(s)
- _Next step:_ validate long-context handling and stop-token behavior across
  mlx-vlm + mlx runtime.

---

## 1. Failure affecting 1 model

- _Observed:_ Model loading failed: Received 4 parameters not in model:
- _Likely owner:_ `mlx`
- _Why it matters:_ This prevented a complete model response; stage `Model
  Error`; phase `model_load`; code `MLX_MODEL_LOAD_MODEL`; type `ValueError`.
- _Suggested next step:_ check tensor/cache behavior and memory pressure
  handling.
- _Affected models:_ `mlx-community/Kimi-VL-A3B-Thinking-8bit`

**Representative maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ runtime_failure \| MLX_MODEL_LOAD_MODEL
- _Summary:_ model error \| mlx model load model
- _Evidence:_ model error \| mlx model load model
- _Token context:_ stop=exception
- _Next action:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.

| Model                                     | Observed Behavior                                         | First Seen Failing      | Recent Repro           |
|-------------------------------------------|-----------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Model loading failed: Received 4 parameters not in model: | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `mlx-community/Kimi-VL-A3B-Thinking-8bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

Traceback tail:

```text
Traceback (most recent call last):
ValueError: Model loading failed: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

Captured stdout/stderr:

```text
=== STDERR ===
```

</details>

## 2. Failure affecting 1 model

- _Observed:_ Loaded processor has no image_processor; expected multimodal
  processor.
- _Likely owner:_ `model configuration/repository`
- _Why it matters:_ This prevented a complete model response; stage `Processor
  Error`; phase `processor_load`; code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`; type `ValueError`.
- _Suggested next step:_ verify model config, tokenizer files, and revision
  alignment.
- _Affected models:_ `mlx-community/MolmoPoint-8B-fp16`

**Representative maintainer triage:**

- _Likely owner:_ model-config \| confidence=high
- _Classification:_ runtime_failure \| MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
- _Summary:_ processor error \| model config processor load processor
- _Evidence:_ processor error \| model config processor load processor
- _Token context:_ stop=exception
- _Next action:_ Inspect model repo config, chat template, and EOS settings.

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

/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/modeling_rope_utils.py:1034: FutureWarning: `rope_config_validation` is deprecated and has been removed. Its functionality has been moved to RotaryEmbeddingConfigMixin.validate_rope method. PreTrainedConfig inherits this class, so please call self.validate_rope() instead. Also, make sure to use the new rope_parameters syntax. You can call self.standardize_rope_params() in the meantime.
  warnings.warn(
```

</details>

---

## Harness/Integration Issues (9 model(s))

9 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `microsoft/Phi-3.5-vision-instruct`

- _Observed:_ Generation appears to continue through stop/control tokens
  instead of ending cleanly.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=768, output=500, output/prompt=65.10%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;\|end\|&gt; appeared in generated text.
  \| Special control token &lt;\|endoftext\|&gt; appeared in generated text.
  \| hit token cap (500) \| nontext prompt burden=99%
- _Evidence:_ Special control token &lt;\|end\|&gt; appeared in generated
  text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated
  text.
- _Token context:_ prompt=768 \| output/prompt=65.10% \| nontext burden=99% \|
  stop=completed \| hit token cap (500)
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
The image shows a tranquil park scene with a person standing on a wooden dock, fishing by a pond. There are trees, a bench, and a small pine tree in the foreground. The weather appears to be overcast....
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

- _Observed:_ Decoded output contains tokenizer artifacts that should not
  appear in user-facing text.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=2,097, output=172, output/prompt=8.20%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| encoding
- _Summary:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 139 occurrences). \| nontext prompt burden=100%
- _Evidence:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 139 occurrences).
- _Token context:_ prompt=2,097 \| output/prompt=8.20% \| nontext burden=100%
  \| stop=completed
- _Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 139 occurrences).

**Sample output:**

```text
TheĠimageĠdepictsĠaĠsereneĠoutdoorĠscene,ĠlikelyĠinĠaĠparkĠorĠgarden.ĠTheĠfocalĠpointĠisĠaĠpersonĠstandingĠonĠaĠsmall,ĠwoodenĠpierĠthatĠextendsĠintoĠaĠcalmĠbodyĠofĠwater.ĠTheĠindividualĠisĠdressedĠinĠ...
```

### `mlx-community/GLM-4.6V-Flash-6bit`

- _Observed:_ Generation appears to continue through stop/control tokens
  instead of ending cleanly.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=6,091, output=500, output/prompt=8.21%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;/think&gt; appeared in generated text.
  \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak
- _Evidence:_ Special control token &lt;/think&gt; appeared in generated text.
- _Token context:_ prompt=6,091 \| output/prompt=8.21% \| nontext burden=100%
  \| stop=completed \| hit token cap (500)
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: <think>; Excessive markdown headers (6).
- Output leaked reasoning or prompt-template text (<think>).

**Sample output:**

```text
<think>Got it, let's describe this picture in detail. First, the scene is a serene park or garden with a lake or pond. 

In the foreground, there's a small pine tree planted in a bed with dark soil, a...
```

### `mlx-community/GLM-4.6V-Flash-mxfp4`

- _Observed:_ Generation appears to continue through stop/control tokens
  instead of ending cleanly.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=6,091, output=500, output/prompt=8.21%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;/think&gt; appeared in generated text.
  \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak
- _Evidence:_ Special control token &lt;/think&gt; appeared in generated text.
- _Token context:_ prompt=6,091 \| output/prompt=8.21% \| nontext burden=100%
  \| stop=completed \| hit token cap (500)
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: <think>; Excessive markdown headers (6).
- Output leaked reasoning or prompt-template text (<think>).

**Sample output:**

```text
<think>Got it, let's describe this picture in detail. First, the scene is a serene park or garden with a lake or pond. 

In the foreground, there's a large, moss-covered rock. Next to it, a small pine...
```

### `mlx-community/GLM-4.6V-nvfp4`

- _Observed:_ Generation appears to continue through stop/control tokens
  instead of ending cleanly.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=6,091, output=500, output/prompt=8.21%

**Maintainer triage:**

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| stop_token
- _Summary:_ Special control token &lt;/think&gt; appeared in generated text.
  \| hit token cap (500) \| nontext prompt burden=100% \| reasoning leak
- _Evidence:_ Special control token &lt;/think&gt; appeared in generated text.
- _Token context:_ prompt=6,091 \| output/prompt=8.21% \| nontext burden=100%
  \| stop=completed \| hit token cap (500)
- _Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: <think>.
- Output leaked reasoning or prompt-template text (<think>).

**Sample output:**

```text
<think>Got it, let's describe this picture step by step. First, the the scene is a serene park or garden with a lake or pond. In the foreground, there's a small pine tree planted in a bed of dark soil...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=16,299, output=13, output/prompt=0.08%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ context_budget \| long_context
- _Summary:_ Output is very short relative to prompt size (0.1%), suggesting
  possible early-stop or prompt-handling issues. \| At long prompt length
  (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%). \|
  output/prompt=0.08% \| nontext prompt burden=100%
- _Evidence:_ Output is very short relative to prompt size (0.1%), suggesting
  possible early-stop or prompt-handling issues. \| At long prompt length
  (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%).
- _Token context:_ prompt=16,299 \| output/prompt=0.08% \| nontext burden=100%
  \| stop=completed
- _Next action:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%).

**Sample output:**

```text
I'm sorry, but the context didn't show up.
```

### `mlx-community/gemma-3n-E2B-4bit`

- _Observed:_ Output shape suggests a prompt-template or stop-condition
  mismatch.
- _Likely owner:_ `model-config / mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate chat-template/config expectations and
  mlx-vlm prompt formatting for this model.
- _Token summary:_ prompt=264, output=4, output/prompt=1.52%

**Maintainer triage:**

- _Likely owner:_ model-config \| confidence=high
- _Classification:_ harness \| prompt_template
- _Summary:_ Output appears truncated to about 4 tokens. \| nontext prompt
  burden=98%
- _Evidence:_ Output appears truncated to about 4 tokens.
- _Token context:_ prompt=264 \| output/prompt=1.52% \| nontext burden=98% \|
  stop=completed
- _Next action:_ Inspect model repo config, chat template, and EOS settings.

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 4 tokens.

**Sample output:**

```text
in the park
```

### `mlx-community/gemma-4-31b-bf16`

- _Observed:_ Output shape suggests a prompt-template or stop-condition
  mismatch.
- _Likely owner:_ `model-config / mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate chat-template/config expectations and
  mlx-vlm prompt formatting for this model.
- _Token summary:_ prompt=266, output=5, output/prompt=1.88%

**Maintainer triage:**

- _Likely owner:_ model-config \| confidence=high
- _Classification:_ harness \| prompt_template
- _Summary:_ Output appears truncated to about 5 tokens. \| nontext prompt
  burden=98%
- _Evidence:_ Output appears truncated to about 5 tokens.
- _Token context:_ prompt=266 \| output/prompt=1.88% \| nontext burden=98% \|
  stop=completed
- _Next action:_ Inspect model repo config, chat template, and EOS settings.

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 5 tokens.

**Sample output:**

```text
in one sentence.
```

### `mlx-community/paligemma2-3b-pt-896-4bit`

- _Observed:_ Behavior degrades under long prompt context.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=4,101, output=3, output/prompt=0.07%

**Maintainer triage:**

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ context_budget \| long_context
- _Summary:_ Output appears truncated to about 3 tokens. \| At long prompt
  length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%).
  \| output/prompt=0.07% \| nontext prompt burden=100%
- _Evidence:_ Output appears truncated to about 3 tokens. \| At long prompt
  length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%).
- _Token context:_ prompt=4,101 \| output/prompt=0.07% \| nontext burden=100%
  \| stop=completed
- _Next action:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.
- At long prompt length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%).

**Sample output:**

```text
The garden
```

---

### Long-Context Degradation / Potential Stack Issues (1 model(s))

1 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely
integration/runtime issues worth checking upstream.

| Model                               |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                                                       | Owner           |
|-------------------------------------|--------------|--------------|-----------------|-------------------------------------------------------------------------------|-----------------|
| `mlx-community/Qwen3.5-9B-MLX-4bit` |       16,290 |          500 | 3.07%           | Output degeneration under long prompt length (repeated_punctuation: ':**...') | `mlx-vlm / mlx` |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

| Model                                     | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------------|--------------------------|-------------------------|------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`        | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 12
- **Summary diagnostics models:** 41
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1375.06s (1375.06s)
- **Average runtime per model:** 25.94s (25.94s)
- **Dominant runtime phase:** decode dominated 49/53 measured model runs (91% of tracked runtime).
- **Phase totals:** model load=121.62s, prompt prep=0.16s, decode=1234.86s, cleanup=5.94s
- **Observed stop reasons:** completed=51, exception=2
- **Validation overhead:** 17.98s total (avg 0.34s across 53 model(s)).
- **First-token latency:** Avg 14.15s | Min 0.06s | Max 84.53s across 51 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (41 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (30 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`
- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `mlx-community/FastVLM-0.5B-bf16`
- `mlx-community/InternVL3-14B-8bit`
- `mlx-community/InternVL3-8B-bf16`
- `mlx-community/LFM2-VL-1.6B-8bit`
- `mlx-community/LFM2.5-VL-1.6B-bf16`
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
- `mlx-community/Molmo-7B-D-0924-8bit`
- `mlx-community/Molmo-7B-D-0924-bf16`
- `mlx-community/Phi-3.5-vision-instruct-bf16`
- `mlx-community/SmolVLM-Instruct-bf16`
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`
- `mlx-community/X-Reasoner-7B-8bit`
- `mlx-community/gemma-3-27b-it-qat-4bit`
- `mlx-community/gemma-3-27b-it-qat-8bit`
- `mlx-community/gemma-3n-E4B-it-bf16`
- `mlx-community/gemma-4-26b-a4b-it-4bit`
- `mlx-community/gemma-4-31b-it-4bit`
- `mlx-community/llava-v1.6-mistral-7b-8bit`
- `mlx-community/nanoLLaVA-1.5-4bit`
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`
- `mlx-community/pixtral-12b-8bit`
- `mlx-community/pixtral-12b-bf16`
- `qnguyen3/nanoLLaVA`

### Ran, but with quality warnings (11 model(s))

- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output leaked reasoning or prompt-template text (◁think▷, ◁/think▷).
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'of').
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Likely capped by max token budget
- `mlx-community/Idefics3-8B-Llama3-bf16`: Output formatting deviated from the requested structure. Details: Unknown tags: <end_of_utterance>.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Output leaked reasoning or prompt-template text (◁think▷, ◁/think▷).
- `mlx-community/Qwen3.5-27B-4bit`: Likely capped by max token budget
- `mlx-community/Qwen3.5-27B-mxfp8`: Likely capped by max token budget
- `mlx-community/Qwen3.5-35B-A3B-4bit`: Likely capped by max token budget
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Likely capped by max token budget
- `mlx-community/Qwen3.5-35B-A3B-bf16`: Likely capped by max token budget
- `mlx-community/Qwen3.6-27B-mxfp8`: Likely capped by max token budget

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260502+e8ebdebe |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.7.0                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.13.0                      |
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --exclude mlx-community/Qwen3-VL-2B-Thinking-bf16 Qwen/Qwen3-VL-2B-Instruct --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Kimi-VL-A3B-Thinking-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
```

### Prompt Used

```text
Describe this picture
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-05-02 23:55:07 BST by [check_models](https://github.com/jrp2014/check_models)._
