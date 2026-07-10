<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 1 failure(s), 6 harness issue(s), 3 text-sanity issue(s) (mlx-vlm 0.6.5)

**Run summary:** 61 locally-cached VLM model(s) checked; 1 hard failure(s), 6 harness/integration issue(s), 3 text-sanity/semantic issue(s), 0 preflight warning(s), 60 successful run(s).

Test image: `20260704-181004_DSC00862_DxO.jpg` (49.2 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                                | Evidence Snapshot                                                                                                                                                                                      | Affected Models                                     | Issue Draft                                                                                                                              | Evidence Bundle                                                                                                                                                                                           | Fixed When                                                   |
|----------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `mlx-vlm`                                                | mlx-vlm: Decode / model error: list index out of range | Model Error \| phase decode \| IndexError                                                                                                                                                              | 1: `mlx-community/gemma-4-31b-bf16`                 | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx-vlm_mlx-vlm-decode-model_001.md)         | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_007_mlx-community_gemma-4-31b-bf16_MLX_VLM_DECODE_MODEL_eddb255ac4b7.json)                       | Load/generation completes or fails with a narrower owner.    |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text         | decoded text contains control token &lt;/think&gt; \| prompt=1,127 \| output/prompt=6.83% \| nontext burden=61% \| stop=completed                                                                      | 1: `mlx-community/MiniCPM-V-4.6-8bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_stop-token_001.md)                   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_002_mlx-community_MiniCPM-V-4.6-8bit_mlx_vlm_stop_token_001.json)                                | No leaked stop/control tokens.                               |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                  | generated_tokens~5 \| prompt=1,530 \| output/prompt=0.33% \| nontext burden=72% \| stop=completed                                                                                                      | 1: `mlx-community/paligemma2-3b-ft-docci-448-bf16`  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_model-config-mlx-vlm_prompt-template_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_010_mlx-community_paligemma2-3b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json) | Requested sections render without template leakage.          |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short  | generated_tokens~2 \| prompt_tokens=16729, output_tokens=2, output/prompt=0.0%, weak text=truncated \| prompt=16,729 \| output/prompt=0.01% \| nontext burden=97% \| stop=completed \| 3 model cluster | 3: `Qwen/Qwen3-VL-2B-Instruct` (+2)                 | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm-mlx_long-context_001.md)             | [3 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_001_Qwen_Qwen3-VL-2B-Instruct_mlx_vlm_mlx_long_context_001.json)                              | Full and reduced reruns avoid context collapse.              |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short  | prompt_tokens=16731, repetitive output \| prompt=16,731 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500)                                                         | 1: `mlx-community/Qwen3-VL-2B-Thinking-bf16`        | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm-mlx_long-context_002.md)             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_004_mlx-community_Qwen3-VL-2B-Thinking-bf16_mlx_vlm_mlx_long_context_002.json)                   | Full and reduced reruns avoid context collapse.              |
| model repository                                         | Generated text is mixed-script token-soup              | text sanity \| gibberish(token noise) \| low hint overlap \| prompt=1,530 \| output/prompt=1.63% \| nontext burden=72% \| stop=completed                                                               | 1: `mlx-community/paligemma2-10b-ft-docci-448-bf16` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_model_text-sanity_001.md)                    | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_009_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_text_sanity_001.json)                   | Generated text is readable natural language, not token soup. |
| model repository                                         | Generated text is mixed-script token-soup              | text sanity \| numeric loop \| trusted overlap \| prompt=622 \| output/prompt=39.87% \| nontext burden=30% \| stop=completed \| 2 model cluster                                                        | 2: `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (+1)  | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_007_model_text-sanity_002.md)                    | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_005_mlx-community_SmolVLM2-2.2B-Instruct-mlx_model_text_sanity_002.json)                      | Generated text is readable natural language, not token soup. |
<!-- markdownlint-enable MD060 -->

---

## 1. Failure affecting 1 model

**Affected models:** `mlx-community/gemma-4-31b-bf16`

**Observed error:**

```text
Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

**Relevant upstream traceback:**

```text
    detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 171, in add_token
    if self.is_byte_token[token]:
       ~~~~~~~~~~~~~~~~~~^^^^^^^
IndexError: list index out of range
```

<!-- markdownlint-disable MD060 -->

| Model                            | Observed Behavior                                                                                                                                                                              | First Seen Failing      | Recent Repro           |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/gemma-4-31b-bf16` | Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory). | 2026-07-10 23:37:00 BST | 1/1 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/gemma-4-31b-bf16`

---

## Harness/Integration Issues (6 model(s))

6 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 2 tokens.
- At long prompt length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated).
- Model output may not follow prompt or image contents (missing: Bird, Boat, Boating, Buoy, Bushes).

**Sample output:**

```text
-
```

### `mlx-community/MiniCPM-V-4.6-8bit`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

**Sample output:**

```text
<think>

</think>

Title:
Sailing boats moored near dense green foliage on water

Description:
The image shows two sailing boats floating on water with lush trees in the background. The scene ap...
```

### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 2 tokens.
- At long prompt length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated).
- Model output may not follow prompt or image contents (missing: Bird, Boat, Boating, Buoy, Bushes).

**Sample output:**

```text
-
```

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**Why this appears to be an integration/runtime issue:**

- At long prompt length (16731 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: Bird, Boat, Boating, Buoy, Bushes).
- Output became repetitive, indicating possible generation instability (token: -).
- Output contains corrupted or malformed text segments (character_loop: ' -' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...
```

### `mlx-community/X-Reasoner-7B-8bit`

**Why this appears to be an integration/runtime issue:**

- Output is very short relative to prompt size (0.1%) with weak text signal 'truncated', suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16740 tokens), output stayed unusually short (14 tokens; ratio 0.1%; weak text signal truncated).
- Model output may not follow prompt or image contents (missing: Bird, Boat, Boating, Buoy, Bushes).
- Output contains corrupted or malformed text segments (encoding_shift).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

**Sample output:**

```text
<|im_start|>
 addCriterion
要求：生成一个与图片内容相关的标题
```

### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 5 tokens.
- Model output may not follow prompt or image contents (missing: Bird, Boat, Boating, Buoy, Bushes).

**Sample output:**

```text
- Daytime.
```

---

## Text-Sanity / Semantic Mismatch Issues (3 model(s))

3 model(s) completed successfully but emitted lexically invalid or
mixed-script token-soup output.
These are not hard runtime failures, but they should be visible to maintainers
and model users because the generated caption is not usable.

### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

**Why this output is flagged:**

- Text sanity issue (numeric_loop)

**Sample output:**

```text
Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
Description: Two sailing boats moored on a river with trees behind on the bank
Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben E...
```

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

**Why this output is flagged:**

- Text sanity issue (numeric_loop)

**Sample output:**

```text
- Location: 52° 0' 0" N, 1° 0' 0" E.
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

**Why this output is flagged:**

- Text sanity issue (gibberish(token_noise))

**Sample output:**

```text
- Location: 51.9786° N, 1.4786° E.
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

No prior history baseline available for regression/recovery status.

<!-- markdownlint-disable MD060 -->

| Model                            | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|----------------------------------|--------------------------|-------------------------|------------------------|
| `mlx-community/gemma-4-31b-bf16` | new model failing        | 2026-07-10 23:37:00 BST | 1/1 recent runs failed |
<!-- markdownlint-enable MD060 -->

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 10
- **Summary diagnostics models:** 51
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 1410.36s (1410.36s)
- **Average runtime per model:** 23.12s (23.12s)
- **Dominant runtime phase:** upstream model prefill / first-token dominated 20/61 measured model runs (56% of tracked runtime).
- **Phase totals:** model load=142.69s, local prompt prep=0.22s, upstream model prefill / first-token=787.03s, input preparation + decode=252.05s, generation call total (unsplit)=205.11s, cleanup=8.09s
- **Generation total:** 1244.19s across 61 model(s); upstream model prefill / first-token split available for 60/61 model(s).
- **Observed stop reasons:** completed=53, exception=1, max_tokens=7
- **Validation overhead:** 22.92s total (avg 0.38s across 61 model(s)).
- **Upstream model prefill / first-token time:** Avg 13.12s | Min 0.03s | Max 84.42s across 60 model(s).
- **What this likely means:** Most measured runtime is spent inside upstream generation before the first token is available.
- **Suggested next action:** Inspect prompt/image token accounting, dynamic-resolution image burden, prefill step sizing, and cache/prefill behavior.

---

## Models Not Flagged (51 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 4 model(s).
- **Ran, but with quality warnings:** 47 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260704-181004_DSC00862_DxO.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --presence-context-size 20 --frequency-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `mlx-community/gemma-4-31b-bf16`, `Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/MiniCPM-V-4.6-8bit`, `mlx-community/Qwen3-VL-2B-Instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260704-181004_DSC00862_DxO.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

Report generated on: 2026-07-10 23:37:01 BST by [check_models](https://github.com/jrp2014/check_models).

<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.5                                                                                                                                                    |
| mlx                        | 0.32.1.dev20260710+4367c73b                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.5                                                                                                                                                    |
| transformers               | 5.13.0                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.23.0                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.2                                                                                                                                                   |
| SDK Version                | 26.5                                                                                                                                                     |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                       |
| Xcode Version              | 26.6                                                                                                                                                     |
| Xcode Build                | 17F113                                                                                                                                                   |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                               |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                           |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                                        |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                           |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                          |
| GPU/Chip                   | Apple M5 Max                                                                                                                                             |
| GPU Cores                  | 40                                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                                  |
| MLX Install Type           | editable local source                                                                                                                                    |
| MLX Distribution Root      | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                                       |
| MLX Core Extension         | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=e61c827cd79f978aa5eacc136f65d6dea065005787f3a1457dc9d4512d6ee9cf)  |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
