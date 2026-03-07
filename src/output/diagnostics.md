# Diagnostics Report — 0 failure(s), 11 harness issue(s) (mlx-vlm 0.4.0)

## Summary

Automated benchmarking of **46 locally-cached VLM models** found **0 hard failure(s)** and **11 harness/integration issue(s)** plus **1 preflight compatibility warning(s)** in successful models. 46 of 46 models succeeded.

Test image: `20260228-162254_DSC09377.jpg` (33.5 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [mlx-vlm / mlx]** Harness/integration warnings on 11 model(s). Next: validate stop-token decoding and long-context behavior.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify dependency/version compatibility before model runs.

---

## Priority Summary

| Priority | Issue | Models Affected | Owner | Next Action |
| -------- | ----- | --------------- | ----- | ----------- |
| **Medium** | Harness/integration | 11 (Qwen3-VL-2B-Instruct, Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit, GLM-4.6V-Flash-6bit, GLM-4.6V-Flash-mxfp4, Molmo-7B-D-0924-8bit, SmolVLM2-2.2B-Instruct-mlx, llava-v1.6-mistral-7b-8bit, paligemma2-10b-ft-docci-448-bf16, paligemma2-3b-ft-docci-448-bf16, Florence-2-large-ft) | `mlx-vlm / mlx` | validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime. |
| **Medium** | Preflight compatibility warning | 1 issue(s) | `transformers` | verify dependency/version compatibility before model runs. |

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
**Token summary:** prompt=16,753, output=500, output/prompt=2.98%

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16753 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability.

**Sample output:**

```text
Title: St Mary's Churchyard, Lenham, England

Description: A serene view of the churchyard at St Mary's Church in Lenham, England, featuring ancient, moss-covered gravestones and a half-timbered house...
```

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,352, output=500, output/prompt=36.98%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Generated text appears to continue into example-code templates mid-output.
- Output switched language/script unexpectedly.

**Sample output:**

```text
Title: Historic Lenham Village Churchyard

Description: The image captures a serene churchyard in Lenham, Kent, England, with a view overlooking the village square. Ancient gravestones, some with moss...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,625, output=125, output/prompt=4.76%

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 71 occurrences).
- Output omitted required Title/Description/Keywords sections.

**Sample output:**

```text
Title:ĠGraveyardĠwithĠTudor-styleĠhousesĊĊDescription:ĠAĠgraveyardĠwithĠold,Ġmoss-coveredĠgravestonesĠinĠtheĠforeground,ĠandĠtraditionalĠTudor-styleĠhousesĠwithĠredĠbrickĠandĠwhiteĠtimberĠframingĠinĠt...
```

### `mlx-community/GLM-4.6V-Flash-6bit`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,652, output=500, output/prompt=7.52%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure.
- Output omitted required Title/Description/Keywords sections.
- Output leaked reasoning or prompt-template text.

**Sample output:**

```text
<think>Got it, let's tackle this. First, I need to create the Title, Description, and Keywords sections based on the image.

Starting with the Title. It needs to be 5-10 words, concrete and factual. L...
```

### `mlx-community/GLM-4.6V-Flash-mxfp4`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=6,652, output=500, output/prompt=7.52%

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure.
- Output leaked reasoning or prompt-template text.

**Sample output:**

```text
<think>Got it, let's tackle this. The user wants cataloging metadata based on the image. First, I need to extract the Title, Description, and Keywords as per the rules.

First, the Title. The image sh...
```

### `mlx-community/Molmo-7B-D-0924-8bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=0, output=0, output/prompt=n/a

**Why this appears to be an integration/runtime issue:**

- Model returned zero output tokens.

**Sample output:**

```text
<empty output>
```

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,639, output=9, output/prompt=0.55%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
Cemetery, England, United Kingdom, UK
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,740, output=8, output/prompt=0.29%

**Why this appears to be an integration/runtime issue:**

- Output was a short generic filler response (about 8 tokens).
- Model output may not follow prompt or image contents.

**Sample output:**

```text
The image is a photograph.
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,557, output=9, output/prompt=0.58%

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
- Use only the above metadata hints.
```

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,557, output=5, output/prompt=0.32%

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 5 tokens.
- Model output may not follow prompt or image contents.

**Sample output:**

```text
- Daytime.
```

### `prince-canuma/Florence-2-large-ft`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,101, output=500, output/prompt=45.41%

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

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 11
- **Summary diagnostics models:** 35
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1290.33s (1290.33s)
- **Average runtime per model:** 28.05s (28.05s)
- **Dominant runtime phase:** decode dominated 45/46 measured model runs (88% of tracked runtime).
- **Phase totals:** model load=143.57s, prompt prep=0.12s, decode=1132.22s, cleanup=4.75s
- **Observed stop reasons:** completed=46
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (35 model(s))

These models completed without diagnostics flags (no hard failure, harness warning, or stack-signal anomaly).

### Clean output (3 model(s))

- `mlx-community/InternVL3-14B-8bit`
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
- `mlx-community/gemma-3-27b-it-qat-8bit`

### Ran, but with quality warnings (32 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents.
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output omitted required Title/Description/Keywords sections.
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/FastVLM-0.5B-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Output formatting deviated from the requested structure.
- `mlx-community/InternVL3-8B-bf16`: Title length violation (3 words; expected 5-10)
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/LFM2-VL-1.6B-8bit`: Model output may not follow prompt or image contents.
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Description sentence violation (4; expected 1-2)
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output became repetitive, indicating possible generation instability.
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: Model output may not follow prompt or image contents.
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: Model output may not follow prompt or image contents.
- `mlx-community/Molmo-7B-D-0924-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Title length violation (4 words; expected 5-10)
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: Output appears to copy prompt context verbatim.
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Model refused or deflected the requested task.
- `mlx-community/Qwen3.5-35B-A3B-bf16`: Model refused or deflected the requested task.
- `mlx-community/SmolVLM-Instruct-bf16`: Model output may not follow prompt or image contents.
- `mlx-community/X-Reasoner-7B-8bit`: Keyword count violation (21; expected 10-18)
- `mlx-community/gemma-3-27b-it-qat-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/gemma-3n-E4B-it-bf16`: Output omitted required Title/Description/Keywords sections.
- `mlx-community/nanoLLaVA-1.5-4bit`: Output became repetitive, indicating possible generation instability.
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: Model output may not follow prompt or image contents.
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents.
- `mlx-community/pixtral-12b-8bit`: Keyword count violation (24; expected 10-18)
- `mlx-community/pixtral-12b-bf16`: Keyword count violation (24; expected 10-18)
- `qnguyen3/nanoLLaVA`: Model output may not follow prompt or image contents.

---

## Environment

| Component | Version |
| --------- | ------- |
| mlx-vlm | 0.4.0 |
| mlx | 0.31.1.dev20260307+be872ebd |
| mlx-lm | 0.31.0 |
| transformers | 5.3.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.6.0 |
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --timeout 300.0 --verbose
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

Title: 5-10 words, concrete and factual, limited to clearly visible content.

Description: 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.

Keywords: 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: , St Mary's Church, Lenham, England, United Kingdom, UK Here is a caption for the image, written in a neutral but descriptive style suitable for a periodical, blog, or magazine. **Caption:** A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
- Keyword hints: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
- Capture metadata: Taken on 2026-02-28 16:22:54 GMT (at 16:22:54 local time). GPS: 51.237300°N, 0.718917°E.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260228-162254_DSC09377.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-03-07 22:39:08 GMT by [check_models](https://github.com/jrp2014/check_models)._
