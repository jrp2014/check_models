# Diagnostics Report — 0 failure(s), 8 harness issue(s) (mlx-vlm 0.4.0)

## Summary

Automated benchmarking of **49 locally-cached VLM models** found **0 hard failure(s)** and **8 harness/integration issue(s)** plus **1 preflight compatibility warning(s)** in successful models. 49 of 49 models succeeded.

Test image: `20260314-162100_DSC09442.jpg` (40.5 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [mlx-vlm]** Harness/integration warnings on 3 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 4 model(s). Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 1 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
- **[Medium] [transformers / mlx-vlm]** Stack-signal anomalies on 2 successful model(s). Next: verify API compatibility and pinned version floor.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **Medium** | Harness/integration | 3 (Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, Florence-2-large-ft) | `mlx-vlm` | check processor/chat-template wiring and generation kwargs. |
| **Medium** | Harness/integration | 4 (Qwen3-VL-2B-Instruct, Qwen3-VL-2B-Thinking-bf16, Qwen3.5-35B-A3B-6bit, Qwen3.5-35B-A3B-bf16) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Harness/integration | 1 (paligemma2-10b-ft-docci-448-6bit) | `model-config / mlx-vlm` | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |
| **Medium** | Stack-signal anomaly | 2 (Qwen2-VL-2B-Instruct-4bit, Qwen3.5-27B-4bit) | `transformers / mlx-vlm` | verify API compatibility and pinned version floor. |
| **Medium** | Preflight compatibility warning | 1 issue(s) | `transformers` | verify API compatibility and pinned version floor. |

---

## Preflight Compatibility Warnings (1 issue(s))

These warnings were detected before inference. They are non-fatal but should be tracked as potential upstream compatibility issues.

- `transformers import utils no longer reference known backend guard env vars (TRANSFORMERS_NO_* / USE_*); check_models backend guard hints may be ignored with this version.`
  - Owner: `transformers`; suggested tracker: `transformers` (<https://github.com/huggingface/transformers/issues/new>)
  - Suggested next action: verify API compatibility and pinned version floor.

---

## Harness/Integration Issues (8 model(s))

8 model(s) show potential harness/integration issues; see per-model breakdown below.
These models completed successfully but show integration problems (for example stop-token leakage, decoding artifacts, or long-context breakdown) that likely point to stack/runtime behavior rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,826, output=500, output/prompt=2.97%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16826 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
Title:
Sainsbury's Green Wall

Description:
A modern Sainsbury's supermarket in Howardsgate, Welwyn Garden City, England, features a striking green living wall on its brick facade. Shoppers walk along...
```

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

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**What looks wrong:** Behavior degrades under long prompt context.
**Likely component:** `mlx-vlm / mlx`
**Suggested next action:** validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.
**Token summary:** prompt=16,828, output=500, output/prompt=2.97%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16828 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents.
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Got it, let's tackle this step by step. First, I need to analyze the image for metadata. The user provided a draft catalog record with hints, but I have to stick to what's clearly visible.

Starting w...
```

### `mlx-community/Qwen3.5-35B-A3B-6bit`

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
I need to follow specific rules:
- Only use clearly visible details.
- No guessing or inf...
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
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | 16,837 | 179 | 1.06% | Context echo under long prompt length | `transformers / mlx-vlm` |
| `mlx-community/Qwen3.5-27B-4bit` | 16,850 | 500 | 2.97% | Context echo under long prompt length | `transformers / mlx-vlm` |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 10
- **Summary diagnostics models:** 39
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1642.24s (1642.24s)
- **Average runtime per model:** 33.52s (33.52s)
- **Dominant runtime phase:** decode dominated 49/49 measured model runs (90% of tracked runtime).
- **Phase totals:** model load=161.38s, prompt prep=0.13s, decode=1462.65s, cleanup=5.37s
- **Observed stop reasons:** completed=49
- **Validation overhead:** 17.64s total (avg 0.36s across 49 model(s)).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (39 model(s))

These models completed without diagnostics flags (no hard failure, harness warning, or stack-signal anomaly).

### Clean output (4 model(s))

- `mlx-community/InternVL3-8B-bf16`
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/pixtral-12b-8bit`
- `mlx-community/pixtral-12b-bf16`

### Ran, but with quality warnings (35 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output leaked reasoning or prompt-template text.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Model output may not follow prompt or image contents.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections.
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
- `mlx-community/Qwen3.5-27B-mxfp8`: Model refused or deflected the requested task.
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260314-162100_DSC09442.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
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

_Report generated on 2026-03-15 02:26:33 GMT by [check_models](https://github.com/jrp2014/check_models)._
