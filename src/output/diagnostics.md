# Diagnostics Report — 1 failure(s), 9 harness issue(s) (mlx-vlm 0.4.3)

## Summary

Automated benchmarking of **51 locally-cached VLM models** found **1 hard
failure(s)** and **9 harness/integration issue(s)** plus **1 preflight
compatibility warning(s)** in successful models. 50 of 51 models succeeded.

Test image: `20260328-155229_DSC09514.jpg` (16.0 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 3 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 3 model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 3 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [transformers / mlx-vlm]** Stack-signal anomalies on 1 successful model(s). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor. | 1 (MolmoPoint-8B-fp16) | `model configuration/repository` | verify model config, tokenizer files, and revision alignment. |
| **Medium** | Harness/integration | 3 (Devstral-Small-2-24B-Instruct-2512-5bit, GLM-4.6V-Flash-mxfp4, GLM-4.6V-nvfp4) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Harness/integration | 3 (Qwen3-VL-2B-Instruct, Qwen2-VL-2B-Instruct-4bit, X-Reasoner-7B-8bit) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Harness/integration | 3 (llava-v1.6-mistral-7b-8bit, paligemma2-10b-ft-docci-448-bf16, paligemma2-3b-ft-docci-448-bf16) | `model-config / mlx-vlm` | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Stack-signal anomaly | 1 (Qwen3.5-35B-A3B-bf16) | `transformers / mlx-vlm` | verify API compatibility and pinned version floor. |
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

/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/transformers/modeling_rope_utils.py:935: FutureWarning: `rope_config_validation` is deprecated and has been removed. Its functionality has been moved to RotaryEmbeddingConfigMixin.validate_rope method. PreTrainedConfig inherits this class, so please call self.validate_rope() instead. Also, make sure to use the new rope_parameters syntax. You can call self.standardize_rope_params() in the meantime.
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
**Token summary:** prompt=16,813, output=500, output/prompt=2.97%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16813 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
Title:
Bronze Sculpture

Description:
A bronze sculpture titled 'Maquette for the Spirit of the University' by British sculptor Hubert 'Nibs' Dalwood, created in 1961, is displayed at the University o...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,709, output=75, output/prompt=2.77%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 46 occurrences).
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Title:ĠBronzeĠSculptureĠonĠPedestalĊĊDescription:ĠAĠbronzeĠabstractĠsculptureĠstandsĠonĠaĠstoneĠpedestalĠbesideĠaĠbrickĠbuilding.ĠTheĠsculptureĠfeaturesĠangular,ĠtexturedĠformsĠunderĠaĠclearĠblueĠsky....
```

### `mlx-community/GLM-4.6V-Flash-mxfp4`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,707, output=500, output/prompt=7.45%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Model output may not follow prompt or image contents.
- Output formatting deviated from the requested structure.
- Output omitted required Title/Description/Keywords sections.
- Output leaked reasoning or prompt-template text.

**Sample output:**

```text
<think>Got it, let's tackle this. First, I need to analyze the image for cataloguing metadata. The user provided specific instructions on how to handle the Title, Description, and Keywords sections.

...
```

### `mlx-community/GLM-4.6V-nvfp4`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,707, output=438, output/prompt=6.53%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure.
- Output omitted required Title/Description/Keywords sections.
- Output leaked reasoning or prompt-template text.

**Sample output:**

```text
<think>Okay, let's tackle this. First, I need to create the title. The image shows a bronze sculpture on a brick pedestal. The title should be concrete and factual. Maybe "Bronze Abstract Sculpture on...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,824, output=18, output/prompt=0.11%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16824 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
Might be a scene from a movie
```

### `mlx-community/X-Reasoner-7B-8bit`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,824, output=157, output/prompt=0.93%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16824 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
Title:
- Swallowtail Butterfly Tail

Description:
- A large, detailed swirled pattern of a butterfly's tail, composed of numerous small, repeating units arranged in a spiral. The design is symmetrical...
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=2,830, output=10, output/prompt=0.35%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
The sculpture is a bronze piece.
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=1,618, output=9, output/prompt=0.56%

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
**Token summary:** prompt=1,618, output=11, output/prompt=0.68%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.7%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
- Do not copy the text in the field.
```

---

### Long-Context Degradation / Potential Stack Issues (1 model(s))

1 model(s) show long-context degradation or stack anomalies; see table below.
These models technically succeeded, but token/output patterns suggest likely
integration/runtime issues worth checking upstream.

| Model | Prompt Tok | Output Tok | Output/Prompt | Symptom | Owner |
| ----- | ---------- | ---------- | ------------- | ------- | -------------- |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | 16,839 | 500 | 2.97% | Context echo under long prompt length | `transformers / mlx-vlm` |

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

- **Detailed diagnostics models:** 11
- **Summary diagnostics models:** 40
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1089.13s (1089.13s)
- **Average runtime per model:** 21.36s (21.36s)
- **Dominant runtime phase:** decode dominated 49/51 measured model runs (90% of tracked runtime).
- **Phase totals:** model load=108.01s, prompt prep=0.12s, decode=972.15s, cleanup=4.80s
- **Observed stop reasons:** completed=50, exception=1
- **Validation overhead:** 8.59s total (avg 0.17s across 51 model(s)).
- **First-token latency:** Avg 11.21s | Min 0.07s | Max 71.11s across 50 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (40 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (2 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/gemma-3-27b-it-qat-4bit`

### Ran, but with quality warnings (38 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Model output may not follow prompt or image contents.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output omitted required Title/Description/Keywords sections.
- `microsoft/Phi-3.5-vision-instruct`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output became repetitive, indicating possible generation instability.
- `mlx-community/FastVLM-0.5B-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/GLM-4.6V-Flash-6bit`: Model output may not follow prompt or image contents.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/InternVL3-14B-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/InternVL3-8B-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/LFM2-VL-1.6B-8bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: Model output may not follow prompt or image contents.
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/Molmo-7B-D-0924-8bit`: Title length violation (11 words; expected 5-10)
- `mlx-community/Molmo-7B-D-0924-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: Output leaked reasoning or prompt-template text.
- `mlx-community/Qwen3.5-27B-4bit`: Model refused or deflected the requested task.
- `mlx-community/Qwen3.5-27B-mxfp8`: Model refused or deflected the requested task.
- `mlx-community/Qwen3.5-35B-A3B-4bit`: Model refused or deflected the requested task.
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Model refused or deflected the requested task.
- `mlx-community/Qwen3.5-9B-MLX-4bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/SmolVLM-Instruct-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3-27b-it-qat-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E4B-it-bf16`: Title length violation (11 words; expected 5-10)
- `mlx-community/nanoLLaVA-1.5-4bit`: Output became repetitive, indicating possible generation instability.
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: Model output may not follow prompt or image contents.
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/pixtral-12b-8bit`: Description sentence violation (3; expected 1-2)
- `mlx-community/pixtral-12b-bf16`: Model output may not follow prompt or image contents.
- `qnguyen3/nanoLLaVA`: Model output may not follow prompt or image contents.

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.4.3 |
| mlx | 0.31.2.dev20260401+b0748ad8 |
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-155229_DSC09514.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260328-155229_DSC09514.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
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
- Description hint: The sculpture 'Maquette for the Spirit of the University' by British sculptor Hubert 'Nibs' Dalwood, created in 1961, is seen on display at the University of Reading's Whiteknights campus in Reading, UK. The bronze piece is a small-scale model for a much larger, unrealized sculpture intended to represent the spirit of the university and is located outside the Department of Typography & Graphic Communication.
- Keyword hints: 10 Best (structured), Abstract Art, Adobe Stock, Any Vision, Blue sky, Bronze, Bronze Sculpture, Daylight, England, Europe, Handrail, Modern Art, Objects, Royston, Sculpture, Stairs, Statue, Stone, Textured, Town Centre
- Capture metadata: Taken on 2026-03-28 15:52:29 GMT (at 15:52:29 local time). GPS: 51.758450°N, 1.255650°W.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260328-155229_DSC09514.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-04-02 00:00:55 BST by [check_models](https://github.com/jrp2014/check_models)._
