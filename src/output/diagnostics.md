# Diagnostics Report — 3 failure(s), 11 harness issue(s) (mlx-vlm 0.4.4)

## Summary

Automated benchmarking of **51 locally-cached VLM models** found **3 hard
failure(s)** and **11 harness/integration issue(s)** plus **1 preflight
compatibility warning(s)** in successful models. 48 of 51 models succeeded.

Test image: `20260328-155229_DSC09514.jpg` (16.0 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[High] [huggingface_hub]** Model loading failed: Server disconnected without sending a response. (2 model(s)). Next: check cache/revision availability and network/auth state; Hub disconnects may be transient outages rather than model defects.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 6 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 2 model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 3 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [transformers / mlx-vlm]** Stack-signal anomalies on 1 successful model(s). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **High** | Model loading failed: Server disconnected without sending a response. | 2 (Qwen3.5-35B-A3B-4bit, Qwen3.5-35B-A3B-6bit) | `huggingface_hub` | check cache/revision availability and network/auth state; Hub disconnects may be transient outages rather than model defects. |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (MolmoPoint-8B-fp16) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Harness/integration | 6 (Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, GLM-4.6V-Flash-6bit, GLM-4.6V-Flash-mxfp4, GLM-4.6V-nvfp4, Qwen3.5-27B-4bit) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Harness/integration | 2 (Qwen2-VL-2B-Instruct-4bit, Qwen3-VL-2B-Thinking-bf16) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Harness/integration | 3 (LFM2-VL-1.6B-8bit, gemma-3n-E2B-4bit, paligemma2-10b-ft-docci-448-bf16) | `model-config / mlx-vlm` | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Stack-signal anomaly | 1 (Qwen3-VL-2B-Instruct) | `transformers / mlx-vlm` | verify API compatibility and pinned version floor. |
| **Medium** | Preflight compatibility warning | 1 issue(s) | `transformers` | verify API compatibility and pinned version floor. |

---

## 1. Failure affecting 2 models (Priority: High)

**Observed behavior:** Model loading failed: Server disconnected without sending a response.
**Owner (likely component):** `huggingface_hub`
**Suggested next action:** check cache/revision availability and network/auth state; Hub disconnects may be transient outages rather than model defects.
**Affected models:** `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-35B-A3B-6bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | Model loading failed: Server disconnected without sending a response. | 2026-04-06 17:15:49 BST | 1/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | Model loading failed: Server disconnected without sending a response. | 2026-03-01 22:26:38 GMT | 1/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `mlx-community/Qwen3.5-35B-A3B-4bit`

<details>
<summary>Detailed trace logs (affected models)</summary>

#### `mlx-community/Qwen3.5-35B-A3B-4bit`

Traceback tail:

```text
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.RemoteProtocolError: Server disconnected without sending a response.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: Server disconnected without sending a response.
```

#### `mlx-community/Qwen3.5-35B-A3B-6bit`

Traceback tail:

```text
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.RemoteProtocolError: Server disconnected without sending a response.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: Server disconnected without sending a response.
```

</details>

## 2. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `mlx-community/MolmoPoint-8B-fp16`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
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

/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/modeling_rope_utils.py:1021: FutureWarning: `rope_config_validation` is deprecated and has been removed. Its functionality has been moved to RotaryEmbeddingConfigMixin.validate_rope method. PreTrainedConfig inherits this class, so please call self.validate_rope() instead. Also, make sure to use the new rope_parameters syntax. You can call self.standardize_rope_params() in the meantime.
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

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=768, output=500, output/prompt=65.10%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Generated text appears to continue into example-code templates mid-output.
- Output switched language/script unexpectedly (tokenizer_artifact, code_snippet).
- Model refused or deflected the requested task (explicit_refusal).

**Sample output:**

```text
I'm sorry, but I can't assist with that request.<|end|><|endoftext|> 


Instruction 1:
Write a Python function that takes a list of integers as input and returns a new list containing only the even nu...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,097, output=119, output/prompt=5.67%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 103 occurrences).

**Sample output:**

```text
TheĠimageĠdepictsĠaĠmodernĠabstractĠsculptureĠmountedĠonĠaĠbrickĠpedestal.ĠTheĠsculptureĠappearsĠtoĠbeĠmadeĠofĠaĠdark,ĠpossiblyĠmetallicĠmaterial,ĠandĠfeaturesĠaĠcomplex,ĠangularĠdesignĠwithĠsharpĠedg...
```

### `mlx-community/GLM-4.6V-Flash-6bit`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,155, output=303, output/prompt=4.92%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: <think>.
- Output leaked reasoning or prompt-template text (<think>).

**Sample output:**

```text
<think>Got it, let's describe this picture. First, the main subject is a bronze sculpture, maybe abstract, with a patina that looks aged. The sculpture is mounted on a brick pedestal, which is made of...
```

### `mlx-community/GLM-4.6V-Flash-mxfp4`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,155, output=319, output/prompt=5.18%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: <think>.
- Output leaked reasoning or prompt-template text (<think>).

**Sample output:**

```text
<think>Got it, let's describe this picture. The main subject is a bronze or metal sculpture, placed on a brick pedestal. The sculpture has an abstract, stacked or intertwined form, with a patina that ...
```

### `mlx-community/GLM-4.6V-nvfp4`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,155, output=500, output/prompt=8.12%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: <think>.
- Output leaked reasoning or prompt-template text (<think>).

**Sample output:**

```text
<think>Got it, let's describe this picture. First, the, the main subject is a bronze sculpture, which looks abstract, maybe with a textured, weathered appearance. The sculpture is mounted on a brick p...
```

### `mlx-community/LFM2-VL-1.6B-8bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=275, output=4, output/prompt=1.45%

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 4 tokens.

**Sample output:**

```text
I'm not
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,248, output=6, output/prompt=0.04%

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 6 tokens.
- At long prompt length (16248 tokens), output stayed unusually short (6 tokens; ratio 0.0%).

**Sample output:**

```text
Grade: nbsp;
```

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,239, output=500, output/prompt=3.08%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16239 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: phrase: "well-ventilated, and stylishly...").

**Sample output:**

```text
Got it! I need to describe the picture. Let me see... The picture shows a series of images of a set of weatherproofed, well-ventilated, and stylishly designed, and well-ventilated, and stylishly desig...
```

### `mlx-community/Qwen3.5-27B-4bit`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=16,239, output=500, output/prompt=3.08%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.

**Sample output:**

```text
The user wants a description of the provided image.

1.  **Identify the main subject:** The image features a man, specifically a young adult male.
2.  **Identify the person:** I recognize this actor. ...
```

### `mlx-community/gemma-3n-E2B-4bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=264, output=2, output/prompt=0.76%

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 2 tokens.

**Sample output:**

```text
:
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,029, output=2, output/prompt=0.19%

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 2 tokens.

**Sample output:**

```text
T
```

---

### Long-Context Degradation / Potential Stack Issues (1 model(s))

1 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely
integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Owner |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- |
| `Qwen/Qwen3-VL-2B-Instruct` | 16,237 | 500 | 3.08% | Output degeneration under long prompt length (character_loop: '0' repeated) | `transformers / mlx-vlm` |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-35B-A3B-6bit`
**Recoveries since previous run:** none

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `mlx-community/MolmoPoint-8B-fp16` | still failing | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | new regression | 2026-04-06 17:15:49 BST | 1/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | new regression | 2026-03-01 22:26:38 GMT | 1/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 15
- **Summary diagnostics models:** 36
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1409.25s (1409.25s)
- **Average runtime per model:** 27.63s (27.63s)
- **Dominant runtime phase:** decode dominated 39/51 measured model runs (63% of tracked runtime).
- **Phase totals:** model load=511.16s, prompt prep=0.15s, decode=889.02s, cleanup=4.81s
- **Observed stop reasons:** completed=48, exception=3
- **Validation overhead:** 8.69s total (avg 0.17s across 51 model(s)).
- **First-token latency:** Avg 9.74s | Min 0.06s | Max 79.44s across 48 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (36 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (19 model(s))

- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `mlx-community/InternVL3-14B-8bit`
- `mlx-community/LFM2.5-VL-1.6B-bf16`
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
- `mlx-community/Molmo-7B-D-0924-8bit`
- `mlx-community/Molmo-7B-D-0924-bf16`
- `mlx-community/Qwen3.5-9B-MLX-4bit`
- `mlx-community/X-Reasoner-7B-8bit`
- `mlx-community/gemma-3-27b-it-qat-4bit`
- `mlx-community/gemma-3n-E4B-it-bf16`
- `mlx-community/llava-v1.6-mistral-7b-8bit`
- `mlx-community/nanoLLaVA-1.5-4bit`
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`
- `mlx-community/paligemma2-3b-pt-896-4bit`
- `mlx-community/pixtral-12b-8bit`
- `mlx-community/pixtral-12b-bf16`

### Ran, but with quality warnings (17 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Output became repetitive, indicating possible generation instability (token: phrase: "treasured treasured treasured ....
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output became repetitive, indicating possible generation instability (token: 答案内容1.).
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output leaked reasoning or prompt-template text (here are my reasoning steps, the user asks:).
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: ⚠️REVIEW:cutoff
- `mlx-community/FastVLM-0.5B-bf16`: ⚠️REVIEW:cutoff
- `mlx-community/Idefics3-8B-Llama3-bf16`: Output formatting deviated from the requested structure. Details: Unknown tags: <fake_token_around_image>.
- `mlx-community/InternVL3-8B-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "strugg strugg strugg strugg...").
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Output became repetitive, indicating possible generation instability (token: 0.).
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Output became repetitive, indicating possible generation instability (token: phrase: "will be used to...").
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-27B-mxfp8`: ⚠️REVIEW:cutoff
- `mlx-community/Qwen3.5-35B-A3B-bf16`: ⚠️REVIEW:cutoff
- `mlx-community/SmolVLM-Instruct-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "treasured treasured treasured ....
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Output formatting deviated from the requested structure. Details: Unknown tags: <row_1_col_1>.
- `mlx-community/gemma-3-27b-it-qat-8bit`: Output contains corrupted or malformed text segments (character_loop: '00' repeated).
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "black and white and...").
- `qnguyen3/nanoLLaVA`: Output became repetitive, indicating possible generation instability (token: Baz).

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.4.4 |
| mlx | 0.31.2.dev20260406+6a9a121d |
| mlx-lm | 0.31.2 |
| transformers | 5.5.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.9.0 |
| Python Version | 3.13.12 |
| OS | Darwin 25.4.0 |
| macOS Version | 26.4 |
| GPU/Chip | Apple M5 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-155229_DSC09514.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-155229_DSC09514.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --models mlx-community/MolmoPoint-8B-fp16
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-155229_DSC09514.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --models mlx-community/Qwen3.5-35B-A3B-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-155229_DSC09514.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --models mlx-community/Qwen3.5-35B-A3B-6bit
```

### Prompt Used

```text
Describe this picture
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260328-155229_DSC09514.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-04-06 17:15:49 BST by [check_models](https://github.com/jrp2014/check_models)._
