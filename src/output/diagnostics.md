# Diagnostics Report — 1 failure(s), 9 harness issue(s) (mlx-vlm 0.4.4)

## Summary

Automated benchmarking of **53 locally-cached VLM models** found **1 hard
failure(s)** and **9 harness/integration issue(s)** plus **1 preflight
compatibility warning(s)** in successful models. 52 of 53 models succeeded.

Test image: `20260411-154640_DSC09740_DxO.jpg` (37.7 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 3 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 3 model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 3 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (MolmoPoint-8B-fp16) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Harness/integration | 3 (Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, GLM-4.6V-Flash-6bit) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Harness/integration | 3 (Qwen3-VL-2B-Instruct, Qwen3-VL-2B-Thinking-bf16, X-Reasoner-7B-8bit) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Harness/integration | 3 (Qwen2-VL-2B-Instruct-4bit, llava-v1.6-mistral-7b-8bit, paligemma2-10b-ft-docci-448-bf16) | `model-config / mlx-vlm` | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Preflight compatibility warning | 1 issue(s) | `transformers` | verify API compatibility and pinned version floor. |

---

## 1. Failure affecting 1 model (Priority: Medium)

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

## Harness/Integration Issues (9 model(s))

9 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,675, output=500, output/prompt=3.00%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16675 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: phrase: "weatherproof, weatherproof, we...").
- Output omitted required Title/Description/Keywords sections (keywords).

**Sample output:**

```text
Title:
- 2026-04-11 16:46:40 BST, Town Centre, Alton, England, United Kingdom, UK

Description:
- A single, large, rectangular, beige-colored, weather-resistant, double-layered, solar-reflective, weat...
```

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,265, output=500, output/prompt=39.53%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Generated text appears to continue into example-code templates mid-output.
- Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- Output switched language/script unexpectedly (tokenizer_artifact, code_snippet).

**Sample output:**

```text
Title: Optical Illusion of a Hexagonal Pattern

Description: A complex, colorful, and distorted image of a hexagonal pattern that creates an optical illusion.

Keywords: optical illusion, hexagonal pa...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,551, output=100, output/prompt=3.92%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 69 occurrences).
- Output omitted required Title/Description/Keywords sections (description, keywords).

**Sample output:**

```text
Title:ĠBluebellĠCakeĠHouseĠonĠHighĠStreetĊĊDescription:ĠAĠquietĠstreetĠinĠaĠsmallĠtownĠfeaturingĠaĠrowĠofĠtraditionalĠbrickĠandĠplasterĠbuildings,ĠincludingĠaĠcakeĠshopĠwithĠaĠsignĠreadingĠ"BluebellĠC...
```

### `mlx-community/GLM-4.6V-Flash-6bit`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,508, output=500, output/prompt=7.68%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: <think>.
- Output omitted required Title/Description/Keywords sections (title).
- Output leaked reasoning or prompt-template text (<think>).

**Sample output:**

```text
<think>Got it, let's tackle this step by step. First, the Title needs to be 5-10 words, concrete and factual. Looking at the image, there are buildings on a street, with shop signs like "Bluebell Cake...
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
**Token summary:** prompt=16,677, output=500, output/prompt=3.00%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16677 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- Output became repetitive, indicating possible generation instability (token: phrase: "with a glass front....").
- Output omitted required Title/Description/Keywords sections (title, description, keywords).
- Output leaked reasoning or prompt-template text (let's analyze the image).

**Sample output:**

```text
Got it. Let's analyze the image. The image shows a single object: a set of four identical, weatherproofed, wooden or metal boxes with a glass front. The boxes are arranged in a row, each with a glass ...
```

### `mlx-community/X-Reasoner-7B-8bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,686, output=500, output/prompt=3.00%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16686 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).

**Sample output:**

```text
Title:
- White Swallowtail Butterfly

Description:
- A large white butterfly with black spots on its wings is displayed in a continuous pattern, creating a seamless design. The background is a gradien...
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=2,652, output=8, output/prompt=0.30%

**Why this appears to be an integration/runtime issue:**

- Output was a short generic filler response (about 8 tokens).
- Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).

**Sample output:**

```text
The image is a photograph.
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,484, output=13, output/prompt=0.88%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.9%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
- Camera metadata: Canon EOS 5D Mark IV.
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none

| Model | Status vs Previous Run | First Seen Failing | Recent Repro |
| ----- | ---------------------- | ------------------ | ------------ |
| `mlx-community/MolmoPoint-8B-fp16` | still failing | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 10
- **Summary diagnostics models:** 43
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1220.57s (1220.57s)
- **Average runtime per model:** 23.03s (23.03s)
- **Dominant runtime phase:** decode dominated 52/53 measured model runs (90% of tracked runtime).
- **Phase totals:** model load=110.81s, prompt prep=0.17s, decode=1093.37s, cleanup=5.49s
- **Observed stop reasons:** completed=52, exception=1
- **Validation overhead:** 15.97s total (avg 0.30s across 53 model(s)).
- **First-token latency:** Avg 10.72s | Min 0.08s | Max 92.39s across 51 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (43 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (1 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

### Ran, but with quality warnings (42 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'in').
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/FastVLM-0.5B-bf16`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/GLM-4.6V-nvfp4`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/InternVL3-14B-8bit`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/InternVL3-8B-bf16`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/LFM2-VL-1.6B-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Description sentence violation (3; expected 1-2)
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: ⚠️REVIEW:context_budget
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: ⚠️REVIEW:context_budget
- `mlx-community/Molmo-7B-D-0924-8bit`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/Molmo-7B-D-0924-bf16`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/Qwen3.5-27B-4bit`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-27B-mxfp8`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-35B-A3B-4bit`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-35B-A3B-bf16`: Model refused or deflected the requested task (explicit_refusal).
- `mlx-community/Qwen3.5-9B-MLX-4bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/SmolVLM-Instruct-bf16`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/gemma-3-27b-it-qat-4bit`: Nonvisual metadata borrowing
- `mlx-community/gemma-3-27b-it-qat-8bit`: Nonvisual metadata borrowing
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/gemma-3n-E4B-it-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/gemma-4-31b-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "51.147833, -0.978750, 51.14783....
- `mlx-community/gemma-4-31b-it-4bit`: Nonvisual metadata borrowing
- `mlx-community/nanoLLaVA-1.5-4bit`: Title length violation (11 words; expected 5-10)
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `mlx-community/pixtral-12b-8bit`: Keyword count violation (19; expected 10-18)
- `mlx-community/pixtral-12b-bf16`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).
- `qnguyen3/nanoLLaVA`: Model output may not follow prompt or image contents (missing: Town, Centre, Alton, United, Kingdom).

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.4.4 |
| mlx | 0.31.2.dev20260412+520cea2b |
| mlx-lm | 0.31.3 |
| transformers | 5.5.3 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.10.1 |
| Python Version | 3.13.12 |
| OS | Darwin 25.4.0 |
| macOS Version | 26.4.1 |
| GPU/Chip | Apple M5 Max |
| GPU Cores | 40 |
| Metal Support | Metal 4 |
| RAM | 128.0 GB |

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260411-154640_DSC09740_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260411-154640_DSC09740_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
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
- Capture metadata: Taken on 2026-04-11 16:46:40 BST (at 16:46:40 local time). GPS: 51.147833°N, 0.978750°W.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260411-154640_DSC09740_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-04-12 01:33:02 BST by [check_models](https://github.com/jrp2014/check_models)._
