# Diagnostics Report — 4 failure(s), 8 harness issue(s) (mlx-vlm 0.4.2)

## Summary

Automated benchmarking of **53 locally-cached VLM models** found **4 hard
failure(s)** and **8 harness/integration issue(s)** plus **1 preflight
compatibility warning(s)** in successful models. 49 of 53 models succeeded.

Test image: `20260328-172537_DSC09524_DxO.jpg` (48.0 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[High] [transformers]** Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_t... (2 model(s)). Next: verify API compatibility and pinned version floor.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 2 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 3 model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 3 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [transformers / mlx-vlm]** Stack-signal anomalies on 1 successful model(s). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **High** | Model loading failed: 'Florence2LanguageConfig' object has no attribu... | 2 (Florence-2-large-ft, Florence-2-large-ft) | `transformers` | verify API compatibility and pinned version floor. |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (MolmoPoint-8B-fp16) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (deepseek-vl2-8bit) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Harness/integration | 2 (Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Harness/integration | 3 (Qwen3-VL-2B-Thinking-bf16, Qwen3.5-9B-MLX-4bit, X-Reasoner-7B-8bit) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Harness/integration | 3 (Qwen2-VL-2B-Instruct-4bit, paligemma2-10b-ft-docci-448-bf16, paligemma2-3b-ft-docci-448-bf16) | `model-config / mlx-vlm` | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Stack-signal anomaly | 1 (Qwen3-VL-2B-Instruct) | `transformers / mlx-vlm` | verify API compatibility and pinned version floor. |
| **Medium** | Preflight compatibility warning | 1 issue(s) | `transformers` | verify API compatibility and pinned version floor. |

---

## 1. Failure affecting 2 models (Priority: High)

**Observed behavior:** Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
**Owner (likely component):** `transformers`
**Suggested next action:** verify API compatibility and pinned version floor.
**Affected models:** `microsoft/Florence-2-large-ft`, `prince-canuma/Florence-2-large-ft`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id' | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `prince-canuma/Florence-2-large-ft` | Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id' | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `microsoft/Florence-2-large-ft`

<details>
<summary>Detailed trace logs (affected models)</summary>

#### `microsoft/Florence-2-large-ft`

Traceback tail:

```text
    return super().__getattribute__(key)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
```

Captured stdout/stderr:

```text
=== STDERR ===

<unknown>:515: SyntaxWarning: invalid escape sequence '\d'
```

#### `prince-canuma/Florence-2-large-ft`

Traceback tail:

```text
    return super().__getattribute__(key)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
```

Captured stdout/stderr:

```text
=== STDERR ===

<unknown>:515: SyntaxWarning: invalid escape sequence '\d'
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

/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/modeling_rope_utils.py:935: FutureWarning: `rope_config_validation` is deprecated and has been removed. Its functionality has been moved to RotaryEmbeddingConfigMixin.validate_rope method. PreTrainedConfig inherits this class, so please call self.validate_rope() instead. Also, make sure to use the new rope_parameters syntax. You can call self.standardize_rope_params() in the meantime.
  warnings.warn(
```

</details>

## 3. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Loaded processor has no image_processor; expected multimodal processor.
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `mlx-community/deepseek-vl2-8bit`

| Model | Observed Behavior | First Seen Failing | Recent Repro |
| ----- | ----------------- | ------------------ | ------------ |
| `mlx-community/deepseek-vl2-8bit` | Loaded processor has no image_processor; expected multimodal processor. | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `mlx-community/deepseek-vl2-8bit`

<details>
<summary>Detailed trace logs (affected model)</summary>

#### `mlx-community/deepseek-vl2-8bit`

Traceback tail:

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

## Harness/Integration Issues (8 model(s))

8 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,315, output=500, output/prompt=38.02%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Generated text appears to continue into example-code templates mid-output.
- Output contains corrupted or malformed text segments.
- Output switched language/script unexpectedly.

**Sample output:**

```text
Title: Foxton Parish Hall & Annex

Description: The Foxton Parish Hall & Annex is captured in the warm light of the late afternoon sun, with long, dramatic shadows cast by a nearby tree across the bri...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,592, output=89, output/prompt=3.43%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 54 occurrences).
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Title:ĠFoxtonĠParishĠHallĠ&ĠAnnexĊĊDescription:ĠAĠbrickĠbuildingĠwithĠaĠtiledĠroofĠandĠchimney,ĠbathedĠinĠlateĠafternoonĠsunlightĠcastingĠlongĠshadows.ĠAĠblueĠrampĠleadsĠtoĠtheĠentrance,ĠandĠaĠvanĠisĠ...
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
**Token summary:** prompt=16,717, output=500, output/prompt=2.99%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16717 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.
- Output leaked reasoning or prompt-template text.

**Sample output:**

```text
Got it, let's tackle this problem step by step. First, I need to analyze the image provided. The image shows a series of weathered, brownish-brown, wooden-textured roofs on a building. The roofs are a...
```

### `mlx-community/Qwen3.5-9B-MLX-4bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,741, output=500, output/prompt=2.99%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16741 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
The user wants me to analyze the provided image and generate cataloguing metadata (Title, Description, Keywords) based on specific rules.

**1. Analyze the Image:**
*   **Visual Content:** The image i...
```

### `mlx-community/X-Reasoner-7B-8bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,726, output=500, output/prompt=2.99%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16726 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
Title:
- Black and white striped pattern

Description:
- A repeating pattern of vertical black and white stripes fills the entire image.

Keywords:
- Stripes
- Black and white
- Pattern
- Vertical lin...
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,523, output=9, output/prompt=0.59%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
- The image is in the daytime.
```

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,523, output=8, output/prompt=0.53%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
- Do not copy the metadata.
```

---

### Long-Context Degradation / Potential Stack Issues (1 model(s))

1 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely
integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Owner |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- |
| `Qwen/Qwen3-VL-2B-Instruct` | 16,715 | 500 | 2.99% | Context echo under long prompt length | `transformers / mlx-vlm` |

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `microsoft/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16` | still failing | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/deepseek-vl2-8bit` | still failing | 2026-02-15 03:27:34 GMT | 3/3 recent runs failed |
| `prince-canuma/Florence-2-large-ft` | still failing | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 13
- **Summary diagnostics models:** 40
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1081.17s (1081.17s)
- **Average runtime per model:** 20.40s (20.40s)
- **Dominant runtime phase:** decode dominated 48/53 measured model runs (89% of tracked runtime).
- **Phase totals:** model load=113.15s, prompt prep=0.12s, decode=948.97s, cleanup=5.86s
- **Observed stop reasons:** completed=49, exception=4
- **Validation overhead:** 18.63s total (avg 0.35s across 53 model(s)).
- **First-token latency:** Avg 10.23s | Min 0.07s | Max 79.53s across 48 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (40 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (3 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`
- `mlx-community/gemma-3-27b-it-qat-4bit`
- `mlx-community/gemma-3-27b-it-qat-8bit`

### Ran, but with quality warnings (37 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Model output may not follow prompt or image contents.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Model output may not follow prompt or image contents.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/FastVLM-0.5B-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/GLM-4.6V-Flash-6bit`: Model output may not follow prompt or image contents.
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Output contains corrupted or malformed text segments.
- `mlx-community/GLM-4.6V-nvfp4`: Output formatting deviated from the requested structure.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/InternVL3-14B-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/InternVL3-8B-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/LFM2-VL-1.6B-8bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: ⚠️REVIEW:context_budget
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: ⚠️REVIEW:context_budget
- `mlx-community/Molmo-7B-D-0924-8bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Molmo-7B-D-0924-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Title length violation (4 words; expected 5-10)
- `mlx-community/Qwen3.5-27B-4bit`: Model refused or deflected the requested task.
- `mlx-community/Qwen3.5-27B-mxfp8`: Model refused or deflected the requested task.
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Model refused or deflected the requested task.
- `mlx-community/Qwen3.5-35B-A3B-bf16`: Model refused or deflected the requested task.
- `mlx-community/SmolVLM-Instruct-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E4B-it-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/llava-v1.6-mistral-7b-8bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/nanoLLaVA-1.5-4bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: Model output may not follow prompt or image contents.
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/pixtral-12b-8bit`: Output appears to copy prompt context verbatim.
- `mlx-community/pixtral-12b-bf16`: Model output may not follow prompt or image contents.
- `qnguyen3/nanoLLaVA`: Model output may not follow prompt or image contents.

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.4.2 |
| mlx | 0.31.2.dev20260328+0ff1115a |
| mlx-lm | 0.31.2 |
| transformers | 5.4.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.8.0 |
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-172537_DSC09524_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-172537_DSC09524_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Florence-2-large-ft
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-172537_DSC09524_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-172537_DSC09524_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/deepseek-vl2-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-172537_DSC09524_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models prince-canuma/Florence-2-large-ft
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
- Description hint: The Foxton Parish Hall & Annex in the village of Foxton, Cambridgeshire, UK, is pictured in the warm light of the late afternoon sun. The low sun casts long, dramatic shadows from a nearby tree across the brick facades of the buildings.
- Capture metadata: Taken on 2026-03-28 17:25:37 GMT (at 17:25:37 local time). GPS: 52.126490°N, 0.053510°E.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260328-172537_DSC09524_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-03-29 00:31:08 GMT by [check_models](https://github.com/jrp2014/check_models)._
