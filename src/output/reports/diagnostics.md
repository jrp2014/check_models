# Diagnostics Report — 2 failure(s), 4 harness issue(s) (mlx-vlm 0.4.5)

## Summary

Automated benchmarking of **53 locally-cached VLM models** found **2 hard
failure(s)** and **4 harness/integration issue(s)** plus **0 preflight
compatibility warning(s)** in successful models. 51 of 53 models succeeded.

Test image: `20260418-164540_DSC09775_DxO.jpg` (55.7 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Medium] [model configuration/repository]** Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--g... (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [model configuration/repository]** Loaded processor has no image_processor; expected multimodal processor. (1 model(s)). Next: verify model config, tokenizer files, and revision alignment.
- **[Medium] [mlx-vlm]** Harness/integration warnings on 2 model(s). Next: check processor/chat-template wiring and generation kwargs.
- **[Medium] [model-config / mlx-vlm]** Harness/integration warnings on 2 model(s). Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model.

---

## Priority Summary

| Priority   | Issue                                                                    | Models Affected                                                      | Owner                            | Next Action                                                                              |
|------------|--------------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------|------------------------------------------------------------------------------------------|
| **Medium** | Model loading failed: Config not found at /Users/jrp/.cache/huggingfa... | 1 (gemma-3-1b-it-GGUF)                                               | `model configuration/repository` | verify model config, tokenizer files, and revision alignment.                            |
| **Medium** | Loaded processor has no image_processor; expected multimodal processor.  | 1 (MolmoPoint-8B-fp16)                                               | `model configuration/repository` | verify model config, tokenizer files, and revision alignment.                            |
| **Medium** | Harness/integration                                                      | 2 (Phi-3.5-vision-instruct, Devstral-Small-2-24B-Instruct-2512-5bit) | `mlx-vlm`                        | check processor/chat-template wiring and generation kwargs.                              |
| **Medium** | Harness/integration                                                      | 2 (Qwen2-VL-2B-Instruct-4bit, llava-v1.6-mistral-7b-8bit)            | `model-config / mlx-vlm`         | validate chat-template/config expectations and mlx-vlm prompt formatting for this model. |

---

## 1. Failure affecting 1 model (Priority: Medium)

**Observed behavior:** Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27
**Owner (likely component):** `model configuration/repository`
**Suggested next action:** verify model config, tokenizer files, and revision alignment.
**Affected model:** `ggml-org/gemma-3-1b-it-GGUF`

**Representative maintainer triage:**

_Likely owner:_ model-config \| confidence=high
_Classification:_ runtime_failure \| MODEL_CONFIG_MODEL_LOAD_MODEL
_Summary:_ model error \| model config model load model
_Evidence:_ model error \| model config model load model
_Token context:_ stop=exception
_Next action:_ Inspect model repo config, chat template, and EOS settings.

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
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 515, in load_config
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

## Harness/Integration Issues (4 model(s))

4 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `microsoft/Phi-3.5-vision-instruct`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=1,317, output=500, output/prompt=37.97%

**Maintainer triage:**

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| stop_token
_Summary:_ Special control token &lt;\|end\|&gt; appeared in generated text.
           \| Special control token &lt;\|endoftext\|&gt; appeared in
           generated text. \| hit token cap (500) \| nontext prompt burden=66%
_Evidence:_ Special control token &lt;\|end\|&gt; appeared in generated text.
            \| Special control token &lt;\|endoftext\|&gt; appeared in
            generated text.
_Token context:_ prompt=1,317 \| output/prompt=37.97% \| nontext burden=66% \|
                 stop=completed \| hit token cap (500)
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact, code_snippet).

**Sample output:**

```text
Title: Windsor Castle from River Thames

Description: A view of the Round Tower of Windsor Castle, a royal residence in Windsor, Berkshire, England, as seen from across the River Thames. The Union Fla...
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**What looks wrong:** Decoded output contains tokenizer artifacts that should not appear in user-facing text.
**Likely component:** `mlx-vlm`
**Suggested next action:** check processor/chat-template wiring and generation kwargs.
**Token summary:** prompt=2,654, output=84, output/prompt=3.17%

**Maintainer triage:**

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| encoding
_Summary:_ Tokenizer space-marker artifacts (for example Ġ) appeared in output
           (about 58 occurrences). \| nontext prompt burden=83% \| missing
           sections: description, keywords \| missing terms: view, royal,
           residence, Berkshire, which
_Evidence:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
            output (about 58 occurrences).
_Token context:_ prompt=2,654 \| output/prompt=3.17% \| nontext burden=83% \|
                 stop=completed
_Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
               user-facing text.

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 58 occurrences).
- Output omitted required Title/Description/Keywords sections (description, keywords).

**Sample output:**

```text
Title:ĠWindsorĠCastleĠRoundĠTowerĊĊDescription:ĠTheĠRoundĠTowerĠofĠWindsorĠCastleĠisĠseenĠacrossĠtheĠRiverĠThames,ĠwithĠlushĠgreenĠtreesĠinĠtheĠforegroundĠandĠaĠpartlyĠcloudyĠskyĠabove.ĠTheĠUnionĠFlag...
```

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

**What looks wrong:** Generation appears to continue through stop/control tokens instead of ending cleanly.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=16,731, output=2, output/prompt=0.01%

**Maintainer triage:**

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| stop_token
_Summary:_ Special control token &lt;\|endoftext\|&gt; appeared in generated
           text. \| Output appears truncated to about 2 tokens. \| nontext
           prompt burden=97% \| missing terms: view, Round, Tower, Windsor,
           Castle
_Evidence:_ Special control token &lt;\|endoftext\|&gt; appeared in generated
            text. \| Output appears truncated to about 2 tokens.
_Token context:_ prompt=16,731 \| output/prompt=0.01% \| nontext burden=97% \|
                 stop=completed
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output appears truncated to about 2 tokens.
- At long prompt length (16731 tokens), output stayed unusually short (2 tokens; ratio 0.0%).
- Model output may not follow prompt or image contents (missing: view, Round, Tower, Windsor, Castle).
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
<|endoftext|>
```

### `mlx-community/llava-v1.6-mistral-7b-8bit`

**What looks wrong:** Output shape suggests a prompt-template or stop-condition mismatch.
**Likely component:** `model-config / mlx-vlm`
**Suggested next action:** validate chat-template/config expectations and mlx-vlm prompt formatting for this model.
**Token summary:** prompt=2,722, output=8, output/prompt=0.29%

**Maintainer triage:**

_Likely owner:_ model-config \| confidence=high
_Classification:_ harness \| prompt_template
_Summary:_ Output was a short generic filler response (about 8 tokens). \|
           nontext prompt burden=84% \| missing terms: view, Round, Tower,
           Windsor, Castle
_Evidence:_ Output was a short generic filler response (about 8 tokens).
_Token context:_ prompt=2,722 \| output/prompt=0.29% \| nontext burden=84% \|
                 stop=completed
_Next action:_ Inspect model repo config, chat template, and EOS settings.

**Why this appears to be an integration/runtime issue:**

- Output was a short generic filler response (about 8 tokens).
- Model output may not follow prompt or image contents (missing: view, Round, Tower, Windsor, Castle).

**Sample output:**

```text
The image is a photograph.
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** `mlx-community/llava-v1.6-mistral-7b-8bit`, `mlx-community/nanoLLaVA-1.5-4bit`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`, `qnguyen3/nanoLLaVA`

| Model                              | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|------------------------------------|--------------------------|-------------------------|------------------------|
| `ggml-org/gemma-3-1b-it-GGUF`      | still failing            | 2026-04-18 00:45:49 BST | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16` | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 6
- **Summary diagnostics models:** 47
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1035.11s (1035.11s)
- **Average runtime per model:** 19.53s (19.53s)
- **Dominant runtime phase:** decode dominated 50/53 measured model runs (90% of tracked runtime).
- **Phase totals:** model load=101.22s, prompt prep=0.14s, decode=915.21s, cleanup=5.05s
- **Observed stop reasons:** completed=51, exception=2
- **Validation overhead:** 18.28s total (avg 0.34s across 53 model(s)).
- **First-token latency:** Avg 10.79s | Min 0.07s | Max 77.28s across 51 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (47 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Clean output (5 model(s))

- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/gemma-3-27b-it-qat-8bit`
- `mlx-community/gemma-4-26b-a4b-it-4bit`
- `mlx-community/gemma-4-31b-it-4bit`

### Ran, but with quality warnings (42 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: Output omitted required Title/Description/Keywords sections (title, description).
- `mlx-community/FastVLM-0.5B-bf16`: Output omitted required Title/Description/Keywords sections (keywords).
- `mlx-community/GLM-4.6V-Flash-6bit`: Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'is').
- `mlx-community/GLM-4.6V-Flash-mxfp4`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/GLM-4.6V-nvfp4`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- `mlx-community/Idefics3-8B-Llama3-bf16`: Output formatting deviated from the requested structure. Details: Unknown tags: &lt;end_of_utterance&gt;.
- `mlx-community/InternVL3-14B-8bit`: Title length violation (4 words; expected 5-10)
- `mlx-community/InternVL3-8B-bf16`: Title length violation (4 words; expected 5-10)
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: Output omitted required Title/Description/Keywords sections (title).
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: Output omitted required Title/Description/Keywords sections (title).
- `mlx-community/LFM2-VL-1.6B-8bit`: Title length violation (4 words; expected 5-10)
- `mlx-community/LFM2.5-VL-1.6B-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "flagpole, flag, flagpole, flag....
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: ⚠️REVIEW:context_budget
- `mlx-community/Molmo-7B-D-0924-8bit`: Output omitted required Title/Description/Keywords sections (description, keywords).
- `mlx-community/Molmo-7B-D-0924-bf16`: Output leaked reasoning or prompt-template text (title hint:).
- `mlx-community/Phi-3.5-vision-instruct-bf16`: Nonvisual metadata borrowing
- `mlx-community/Qwen3.5-27B-4bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-27B-mxfp8`: Nonvisual metadata borrowing
- `mlx-community/Qwen3.5-35B-A3B-4bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-35B-A3B-6bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-35B-A3B-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/Qwen3.5-9B-MLX-4bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/SmolVLM-Instruct-bf16`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: Title length violation (3 words; expected 5-10)
- `mlx-community/X-Reasoner-7B-8bit`: ⚠️REVIEW:context_budget
- `mlx-community/gemma-3-27b-it-qat-4bit`: Keyword count violation (19; expected 10-18)
- `mlx-community/gemma-3n-E2B-4bit`: Model output may not follow prompt or image contents (missing: view, Round, Tower, Windsor, Castle).
- `mlx-community/gemma-3n-E4B-it-bf16`: Description sentence violation (6; expected 1-2)
- `mlx-community/gemma-4-31b-bf16`: Output became repetitive, indicating possible generation instability (token: phrase: "no grown-ups, no grown-ups,...").
- `mlx-community/nanoLLaVA-1.5-4bit`: Output omitted required Title/Description/Keywords sections (keywords).
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: Model output may not follow prompt or image contents (missing: view, Round, Tower, Windsor, Castle).
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: Model output may not follow prompt or image contents (missing: view, Round, Tower, Windsor, Castle).
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: Model output may not follow prompt or image contents (missing: view, Round, Tower, Windsor, Castle).
- `mlx-community/paligemma2-3b-pt-896-4bit`: Model output may not follow prompt or image contents (missing: view, Round, Tower, Windsor, Castle).
- `mlx-community/pixtral-12b-8bit`: ⚠️REVIEW:context_budget
- `mlx-community/pixtral-12b-bf16`: ⚠️REVIEW:context_budget
- `qnguyen3/nanoLLaVA`: Output omitted required Title/Description/Keywords sections (keywords).

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260425+211e57be |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.6.2                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.12.0                      |
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-164540_DSC09775_DxO.jpg --exclude Qwen/Qwen3-VL-2B-Instruct mlx-community/Qwen3-VL-2B-Thinking-bf16 --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
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
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-164540_DSC09775_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models ggml-org/gemma-3-1b-it-GGUF
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-164540_DSC09775_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
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
- Description hint: A view of the Round Tower of Windsor Castle, a royal residence in Windsor, Berkshire, England, as seen from across the River Thames. The Union Flag is flying from the flagpole, which indicates that the reigning monarch is not in residence at the castle at the time the photograph was taken.
- Capture metadata: Taken on 2026-04-18 17:45:40 BST (at 17:45:40 local time). GPS: 51.483800°N, 0.604400°W.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260418-164540_DSC09775_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-04-26 00:10:30 BST by [check_models](https://github.com/jrp2014/check_models)._
