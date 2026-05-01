# Diagnostics Report — 2 failure(s), 9 harness issue(s) (mlx-vlm 0.4.5)

## Summary

Automated benchmarking of **53 locally-cached VLM models** found **2 hard
failure(s)** and **9 harness/integration issue(s)** plus **0 preflight
compatibility warning(s)** in successful models. 51 of 53 models succeeded.

Test image: `20260403-124049_DSC09541.jpg` (47.5 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [mlx]** Model loading failed: Received 4 parameters not in model: (1 model(s)). Next: check tensor/cache behavior and memory pressure handling.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 5 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 2 model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 2 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [mlx-vlm / mlx]** Stack-signal anomalies on 1 successful model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.

---

## Priority Summary

| Priority   | Issue                                                                   | Models Affected                                                                                                                 | Owner                            | Next Action                                                                              |
|------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|----------------------------------|------------------------------------------------------------------------------------------|
| **Medium** | Model loading failed: Received 4 parameters not in model:               | 1 (Kimi-VL-A3B-Thinking-8bit)                                                                                                   | `mlx`                            | check tensor/cache behavior and memory pressure handling.                                |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (MolmoPoint-8B-fp16)                                                                                                          | `model configuration/repository` | verify model config, tokenizer files, and revision alignment.                            |
| **Medium** | Harness/integration                                                     | 5 (Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, GLM-4.6V-Flash-6bit, GLM-4.6V-Flash-mxfp4, GLM-4.6V-nvfp4) | `mlx-vlm`                        | check processor/chat-template wiring and generation kwargs.                              |
| **Medium** | Harness/integration                                                     | 2 (Qwen2-VL-2B-Instruct-4bit, paligemma2-3b-pt-896-4bit)                                                                        | `mlx-vlm / mlx`                  | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.     |
| **Medium** | Harness/integration                                                     | 2 (gemma-3n-E2B-4bit, gemma-4-31b-bf16)                                                                                         | `model-config / mlx-vlm`         | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Stack-signal anomaly                                                    | 1 (Qwen3.5-9B-MLX-4bit)                                                                                                         | `mlx-vlm / mlx`                  | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.     |

---

## 1. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model loading failed: Received 4 parameters not in model:
**Owner (likely component):** `mlx`
**Suggested next action:** check tensor/cache behavior and memory pressure handling.
**Affected model:** `mlx-community/Kimi-VL-A3B-Thinking-8bit`

**Representative maintainer triage:**

_Likely owner:_ mlx \| confidence=high
_Classification:_ runtime_failure \| MLX_MODEL_LOAD_MODEL
_Summary:_ model error \| mlx model load model
_Evidence:_ model error \| mlx model load model
_Token context:_ stop=exception
_Next action:_ Inspect KV/cache behavior, memory pressure, and long-context
               execution.

| Model                                     | Observed Behavior                                         | First Seen Failing      | Recent Repro           |
|-------------------------------------------|-----------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | Model loading failed: Received 4 parameters not in model: | 2026-02-07 20:59:01 GMT | 1/3 recent runs failed |

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

## 2. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `mlx-community/MolmoPoint-8B-fp16`

**Representative maintainer triage:**

_Likely owner:_ model-config \| confidence=high
_Classification:_ runtime_failure \| MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
_Summary:_ processor error \| model config processor load processor
_Evidence:_ processor error \| model config processor load processor
_Token context:_ stop=exception
_Next action:_ Inspect model repo config, chat template, and EOS settings.

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

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=768, output=500, output/prompt=65.10%

**Maintainer triage:**

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| stop_token
_Summary:_ Special control token &lt;\|end\|&gt; appeared in generated text.
           \| Special control token &lt;\|endoftext\|&gt; appeared in
           generated text. \| hit token cap (500) \| nontext prompt burden=99%
_Evidence:_ Special control token &lt;\|end\|&gt; appeared in generated text.
            \| Special control token &lt;\|endoftext\|&gt; appeared in
            generated text.
_Token context:_ prompt=768 \| output/prompt=65.10% \| nontext burden=99% \|
                 stop=completed \| hit token cap (500)
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
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

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,097, output=172, output/prompt=8.20%

**Maintainer triage:**

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| encoding
_Summary:_ Tokenizer space-marker artifacts (for example Ġ) appeared in output
           (about 139 occurrences). \| nontext prompt burden=100%
_Evidence:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
            output (about 139 occurrences).
_Token context:_ prompt=2,097 \| output/prompt=8.20% \| nontext burden=100% \|
                 stop=completed
_Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
               user-facing text.

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 139 occurrences).

**Sample output:**

```text
TheĠimageĠdepictsĠaĠsereneĠoutdoorĠscene,ĠlikelyĠinĠaĠparkĠorĠgarden.ĠTheĠfocalĠpointĠisĠaĠpersonĠstandingĠonĠaĠsmall,ĠwoodenĠpierĠthatĠextendsĠintoĠaĠcalmĠbodyĠofĠwater.ĠTheĠindividualĠisĠdressedĠinĠ...
```

### `mlx-community/GLM-4.6V-Flash-6bit`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,091, output=500, output/prompt=8.21%

**Maintainer triage:**

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| stop_token
_Summary:_ Special control token &lt;/think&gt; appeared in generated text. \|
           hit token cap (500) \| nontext prompt burden=100% \| reasoning leak
_Evidence:_ Special control token &lt;/think&gt; appeared in generated text.
_Token context:_ prompt=6,091 \| output/prompt=8.21% \| nontext burden=100% \|
                 stop=completed \| hit token cap (500)
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
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

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,091, output=500, output/prompt=8.21%

**Maintainer triage:**

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| stop_token
_Summary:_ Special control token &lt;/think&gt; appeared in generated text. \|
           hit token cap (500) \| nontext prompt burden=100% \| reasoning leak
_Evidence:_ Special control token &lt;/think&gt; appeared in generated text.
_Token context:_ prompt=6,091 \| output/prompt=8.21% \| nontext burden=100% \|
                 stop=completed \| hit token cap (500)
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
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

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,091, output=500, output/prompt=8.21%

**Maintainer triage:**

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| stop_token
_Summary:_ Special control token &lt;/think&gt; appeared in generated text. \|
           hit token cap (500) \| nontext prompt burden=100% \| reasoning leak
_Evidence:_ Special control token &lt;/think&gt; appeared in generated text.
_Token context:_ prompt=6,091 \| output/prompt=8.21% \| nontext burden=100% \|
                 stop=completed \| hit token cap (500)
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
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

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,299, output=13, output/prompt=0.08%

**Maintainer triage:**

_Likely owner:_ mlx \| confidence=high
_Classification:_ context_budget \| long_context
_Summary:_ Output is very short relative to prompt size (0.1%), suggesting
           possible early-stop or prompt-handling issues. \| At long prompt
           length (16299 tokens), output stayed unusually short (13 tokens;
           ratio 0.1%). \| output/prompt=0.08% \| nontext prompt burden=100%
_Evidence:_ Output is very short relative to prompt size (0.1%), suggesting
            possible early-stop or prompt-handling issues. \| At long prompt
            length (16299 tokens), output stayed unusually short (13 tokens;
            ratio 0.1%).
_Token context:_ prompt=16,299 \| output/prompt=0.08% \| nontext burden=100%
                 \| stop=completed
_Next action:_ Treat this as a prompt-budget issue first; nontext prompt
               burden is 100% and the output stays weak under that load.

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%).

**Sample output:**

```text
I'm sorry, but the context didn't show up.
```

### `mlx-community/gemma-3n-E2B-4bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=264, output=4, output/prompt=1.52%

**Maintainer triage:**

_Likely owner:_ model-config \| confidence=high
_Classification:_ harness \| prompt_template
_Summary:_ Output appears truncated to about 4 tokens. \| nontext prompt
           burden=98%
_Evidence:_ Output appears truncated to about 4 tokens.
_Token context:_ prompt=264 \| output/prompt=1.52% \| nontext burden=98% \|
                 stop=completed
_Next action:_ Inspect model repo config, chat template, and EOS settings.

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 4 tokens.

**Sample output:**

```text
in the park
```

### `mlx-community/gemma-4-31b-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=266, output=5, output/prompt=1.88%

**Maintainer triage:**

_Likely owner:_ model-config \| confidence=high
_Classification:_ harness \| prompt_template
_Summary:_ Output appears truncated to about 5 tokens. \| nontext prompt
           burden=98%
_Evidence:_ Output appears truncated to about 5 tokens.
_Token context:_ prompt=266 \| output/prompt=1.88% \| nontext burden=98% \|
                 stop=completed
_Next action:_ Inspect model repo config, chat template, and EOS settings.

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 5 tokens.

**Sample output:**

```text
in one sentence.
```

### `mlx-community/paligemma2-3b-pt-896-4bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=4,101, output=3, output/prompt=0.07%

**Maintainer triage:**

_Likely owner:_ mlx \| confidence=high
_Classification:_ context_budget \| long_context
_Summary:_ Output appears truncated to about 3 tokens. \| At long prompt
           length (4101 tokens), output stayed unusually short (3 tokens;
           ratio 0.1%). \| output/prompt=0.07% \| nontext prompt burden=100%
_Evidence:_ Output appears truncated to about 3 tokens. \| At long prompt
            length (4101 tokens), output stayed unusually short (3 tokens;
            ratio 0.1%).
_Token context:_ prompt=4,101 \| output/prompt=0.07% \| nontext burden=100% \|
                 stop=completed
_Next action:_ Treat this as a prompt-budget issue first; nontext prompt
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

**Regressions since previous run:** `mlx-community/Kimi-VL-A3B-Thinking-8bit`
**Recoveries since previous run:** none

| Model                                     | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------------|--------------------------|-------------------------|------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | new regression           | 2026-02-07 20:59:01 GMT | 1/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`        | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 12
- **Summary diagnostics models:** 41
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1235.09s (1235.09s)
- **Average runtime per model:** 23.30s (23.30s)
- **Dominant runtime phase:** decode dominated 49/53 measured model runs (91% of tracked runtime).
- **Phase totals:** model load=104.66s, prompt prep=0.16s, decode=1112.64s, cleanup=5.35s
- **Observed stop reasons:** completed=51, exception=2
- **Validation overhead:** 17.39s total (avg 0.33s across 53 model(s)).
- **First-token latency:** Avg 12.38s | Min 0.06s | Max 77.20s across 51 model(s).
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
| mlx             | 0.32.0.dev20260501+e8ebdebe |
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

_Report generated on 2026-05-01 14:28:43 BST by [check_models](https://github.com/jrp2014/check_models)._
