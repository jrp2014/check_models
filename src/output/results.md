# Model Performance Results

_Generated on 2026-03-29 00:31:08 GMT_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 4 (top owners: transformers=2,
  model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=8, clean outputs=3/49.
- _Useful now:_ 3 clean A/B model(s) worth first review.
- _Review watchlist:_ 46 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=15, neutral=6, worse=28 (baseline B 78/100).
- _Quality signal frequency:_ missing_sections=34, cutoff=28,
  trusted_hint_ignored=24, context_ignored=23, metadata_borrowing=21,
  harness=8.
- _Runtime pattern:_ decode dominates measured phase time (89%; 48/53 measured
  model(s)).
- _Phase totals:_ model load=113.15s, prompt prep=0.12s, decode=948.97s,
  cleanup=5.86s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=49, exception=4.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/FastVLM-0.5B-bf16` (331.5 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 75.9 across 49 models

## 📈 Resource Usage

- **Total peak memory:** 946.6 GB
- **Average peak memory:** 19.3 GB
- **Memory efficiency:** 247 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 15 | ✅ B: 10 | 🟡 C: 8 | 🟠 D: 3 | ❌ F: 13

**Average Utility Score:** 57/100

**Existing Metadata Baseline:** ✅ B (78/100)
**Vs Existing Metadata:** Avg Δ -21 | Better: 15, Neutral: 6, Worse: 28

- **Best for cataloging:** `mlx-community/Qwen3-VL-2B-Thinking-bf16` (🏆 A, 99/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 0/100)

### ⚠️ 16 Models with Low Utility (D/F)

- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (6/100) - Output lacks detail
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: 🟠 D (35/100) - Lacks visual description of image
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (19/100) - Mostly echoes context without adding value
- `mlx-community/LFM2.5-VL-1.6B-bf16`: ❌ F (32/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (19/100) - Mostly echoes context without adding value
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (42/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (22/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (17/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (18/100) - Output lacks detail
- `qnguyen3/nanoLLaVA`: 🟠 D (40/100) - Lacks visual description of image

## ⚠️ Quality Issues

- **❌ Failed Models (4):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
  - `mlx-community/deepseek-vl2-8bit` (`Processor Error`)
  - `prince-canuma/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (7):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (token: `本题可答1.`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "##analyze this image ##analyze..."`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "uk, uk, uk, uk,..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "after the label. -..."`)
- **👻 Hallucinations (1):**
  - `microsoft/Phi-3.5-vision-instruct`
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 75.9 | Min: 0 | Max: 332
- **Peak Memory**: Avg: 19 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 21.91s | Min: 1.58s | Max: 114.12s
- **Generation Time**: Avg: 19.37s | Min: 0.73s | Max: 110.23s
- **Model Load Time**: Avg: 2.18s | Min: 0.45s | Max: 12.50s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (89%; 48/53 measured model(s)).
- **Phase totals:** model load=113.15s, prompt prep=0.12s, decode=948.97s, cleanup=5.86s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=49, exception=4.

### ⏱ Timing Snapshot

- **Validation overhead:** 18.63s total (avg 0.35s across 53 model(s)).
- **First-token latency:** Avg 10.23s | Min 0.07s | Max 79.53s across 48 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](model_gallery.md#model-mlx-community-qwen3-vl-2b-thinking-bf16)
  (A 99/100 | Gen 86.8 TPS | Peak 8.3 | A 99/100 | ⚠️REVIEW:cutoff;
  ⚠️HARNESS:long_context; ...)
- _Fastest generation:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (B 77/100 | Gen 332 TPS | Peak 2.1 | B 77/100 | Context ignored (missing:
  Foxton, Parish, Hall, Annex, village); Missing sections (title, description,
  keywords); ...)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (B 77/100 | Gen 332 TPS | Peak 2.1 | B 77/100 | Context ignored (missing:
  Foxton, Parish, Hall, Annex, village); Missing sections (title, description,
  keywords); ...)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 93/100 | Gen 64.0 TPS | Peak 13 | A 93/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (4):_ [`microsoft/Florence-2-large-ft`](model_gallery.md#model-microsoft-florence-2-large-ft),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16),
  [`mlx-community/deepseek-vl2-8bit`](model_gallery.md#model-mlx-community-deepseek-vl2-8bit),
  [`prince-canuma/Florence-2-large-ft`](model_gallery.md#model-prince-canuma-florence-2-large-ft).
  Example: `Model Error`.
- _🔄 Repetitive Output (7):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +3 more. Example: token: `unt`.
- _👻 Hallucinations (1):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct).
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (16):_ [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +12 more. Common weakness: Output too short to be useful.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `transformers` | 2 | Model Error | `microsoft/Florence-2-large-ft`, `prince-canuma/Florence-2-large-ft` |
| `model-config` | 2 | Processor Error | `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/deepseek-vl2-8bit` |

### Actionable Items by Package

#### transformers

- microsoft/Florence-2-large-ft (Model Error)
  - Error: `Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
  - Type: `ValueError`
- prince-canuma/Florence-2-large-ft (Model Error)
  - Error: `Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`
- mlx-community/deepseek-vl2-8bit (Processor Error)
  - Error: `Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimo...`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD028 -->
>
> Analyze this image for cataloguing metadata, using British English.
>
> Use only details that are clearly and definitely visible in the image. If a
> detail is uncertain, ambiguous, partially obscured, too small to verify, or
> not directly visible, leave it out. Do not guess.
>
> Treat the metadata hints below as a draft catalog record. Keep only details
> that are clearly confirmed by the image, correct anything contradicted by
> the image, and add important visible details that are definitely present.
>
> Return exactly these three sections, and nothing else:
>
> Title:
> \- 5-10 words, concrete and factual, limited to clearly visible content.
> \- Output only the title text after the label.
> \- Do not repeat or paraphrase these instructions in the title.
>
> Description:
> \- 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> \- Output only the description text after the label.
>
> Keywords:
> \- 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> \- Output only the keyword list after the label.
>
> Rules:
> \- Include only details that are definitely visible in the image.
> \- Reuse metadata terms only when they are clearly supported by the image.
> \- If metadata and image disagree, follow the image.
> \- Prefer omission to speculation.
> \- Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> \- Do not infer identity, location, event, brand, species, time period, or
> intent unless visually obvious.
> \- Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually
> confirmed):
> \- Description hint: The Foxton Parish Hall &amp; Annex in the village of Foxton,
> Cambridgeshire, UK, is pictured in the warm light of the late afternoon
> sun. The low sun casts long, dramatic shadows from a nearby tree across the
> brick facades of the buildings.
> \- Capture metadata: Taken on 2026-03-28 17:25:37 GMT (at 17:25:37 local
> time). GPS: 52.126490°N, 0.053510°E.
<!-- markdownlint-enable MD028 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1087.67s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |      0.31s |       0.67s |                                   |    transformers |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.21s |       2.54s |                                   |    model-config |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |      2.78s |       3.12s | missing-sections(title+descrip... |    model-config |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |      0.82s |       1.17s |                                   |    transformers |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               752 |                    57 |            809 |         8289 |       319 |         2.8 |            0.73s |      0.49s |       1.58s | missing-sections(title+descrip... |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               498 |                   191 |            689 |         5233 |       322 |         2.5 |            1.21s |      0.45s |       2.00s | fabrication, ...                  |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               502 |                    27 |            529 |         5154 |       332 |         2.1 |            1.27s |      0.70s |       2.40s | missing-sections(title+descrip... |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |              1523 |                     8 |           1531 |         3267 |      21.7 |          11 |            1.36s |      1.40s |       3.10s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |              3089 |                   131 |           3220 |         2703 |       173 |         7.8 |            2.49s |      1.02s |       3.89s | metadata-borrowing, ...           |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |              1523 |                    23 |           1546 |          872 |      29.3 |          12 |            3.07s |      1.68s |       5.10s | missing-sections(title+descrip... |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |   6,729 |               562 |                   500 |           1062 |         7831 |       178 |         3.6 |            3.31s |      0.53s |       4.20s | repetitive(phrase: "uk, uk, ...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |              1523 |                     9 |           1532 |          901 |      5.85 |          26 |            3.76s |      2.46s |       6.57s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |              1315 |                   163 |           1478 |         3816 |      54.6 |         9.5 |            3.79s |      0.85s |       4.97s | fabrication, title-length(4), ... |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |              2699 |                    62 |           2761 |         1033 |      60.8 |         9.7 |            4.29s |      0.89s |       5.53s | missing-sections(title+descrip... |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,771 |               759 |                   500 |           1259 |         2395 |       122 |         6.1 |            4.89s |      1.42s |       6.67s | missing-sections(title+descrip... |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |              2281 |                    79 |           2360 |         1224 |        32 |          18 |            4.90s |      1.77s |       7.06s | title-length(2), ...              |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               607 |                   500 |           1107 |         1802 |       126 |         5.5 |            4.94s |      0.58s |       5.87s | missing-sections(title+descrip... |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |              1706 |                   500 |           2206 |         4148 |       126 |         5.5 |            5.01s |      0.64s |       5.96s | repetitive(unt), ...              |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               768 |                    97 |            865 |          531 |      29.5 |          19 |            5.24s |      2.38s |       7.96s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |              3090 |                   121 |           3211 |         1051 |        64 |          13 |            5.41s |      1.36s |       7.13s |                                   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |              1706 |                   500 |           2206 |         3279 |       115 |         5.5 |            5.51s |      0.58s |       6.43s | repetitive(unt), ...              |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |              3090 |                   134 |           3224 |         1153 |      58.9 |          13 |            5.56s |      1.42s |       7.32s | metadata-borrowing, ...           |                 |
| `qnguyen3/nanoLLaVA`                                    |   4,976 |               498 |                   500 |            998 |         4278 |       101 |         4.6 |            5.59s |      0.52s |       6.47s | missing-sections(title+descrip... |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |              3280 |                   116 |           3396 |         1462 |      38.8 |          16 |            5.77s |      1.66s |       7.78s | context-echo(0.41), ...           |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |   4,082 |              1492 |                   500 |           1992 |         1717 |       108 |          22 |            6.13s |      2.13s |       8.61s | repetitive(phrase: "##analyze...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   3,311 |              1492 |                   500 |           1992 |         1146 |       103 |          18 |            6.81s |      2.06s |       9.21s | missing-sections(title+descrip... |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               768 |                    92 |            860 |          574 |      17.4 |          34 |            7.13s |      3.42s |      10.90s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |              2592 |                    89 |           2681 |          603 |      29.6 |          22 |            7.97s |      2.19s |      10.63s | ⚠️harness(encoding), ...          |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               469 |                    14 |            483 |          117 |       5.2 |          25 |            8.09s |      3.37s |      11.88s | missing-sections(title+descrip... |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |   3,547 |              1492 |                   500 |           1992 |         1479 |        75 |          37 |            8.46s |      3.58s |      12.40s | repetitive(本题可答1.), ...       |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               767 |                   368 |           1135 |         1695 |      47.7 |          17 |            8.65s |      2.27s |      11.27s | missing-sections(title+descrip... |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |     263 |              1315 |                   500 |           1815 |         3357 |      53.2 |         9.5 |           10.27s |      0.87s |      11.49s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               468 |                   146 |            614 |          246 |      16.7 |          15 |           11.21s |      1.63s |      13.22s | missing-sections(title+descrip... |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   2,793 |              4595 |                   500 |           5095 |         3632 |      40.3 |         4.5 |           14.43s |      1.14s |      15.92s | repetitive(phrase: "after the...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |     264 |              6614 |                   500 |           7114 |          886 |      65.8 |         8.4 |           15.57s |      1.32s |      17.25s | degeneration, ...                 |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   3,937 |              3371 |                   500 |           3871 |         1384 |      39.7 |          15 |           15.60s |      1.68s |      17.63s | fabrication, ...                  |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |              2281 |                   500 |           2781 |         2509 |      33.8 |          18 |           16.29s |      1.74s |      18.44s | repetitive(phrase: "rencontre...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  93,938 |              1812 |                   500 |           2312 |          277 |      53.3 |          60 |           16.77s |     11.07s |      28.24s | missing-sections(title), ...      |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |     389 |              6614 |                   500 |           7114 |         1080 |      44.9 |          11 |           17.77s |      1.38s |      19.49s | degeneration, ...                 |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |              2779 |                   500 |           3279 |         2222 |      31.3 |          19 |           17.90s |      1.87s |      20.24s | missing-sections(title+descrip... |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      17 |             16717 |                   500 |          17217 |         1217 |      86.8 |         8.3 |           20.46s |      0.70s |      21.51s | ⚠️harness(long_context), ...      |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |      15 |             16715 |                   500 |          17215 |         1220 |      85.3 |         8.3 |           20.49s |      0.67s |      21.48s | fabrication, title-length(4), ... |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |  13,994 |              1677 |                   500 |           2177 |          107 |      48.6 |          41 |           26.81s |      1.19s |      28.35s | missing-sections(title+descrip... |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |   2,884 |             16726 |                   500 |          17226 |          874 |        55 |          13 |           29.13s |      1.12s |      30.59s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/pixtral-12b-bf16`                        |  99,363 |              3280 |                   500 |           3780 |         1590 |      20.1 |          28 |           37.49s |      2.55s |      40.38s | missing-sections(title+descrip... |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |  13,693 |              6614 |                   500 |           7114 |          262 |      30.9 |          78 |           42.02s |     10.74s |      53.11s | missing-sections(title+descrip... |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |      60 |              1677 |                   500 |           2177 |         87.4 |        30 |          48 |           56.96s |      1.71s |      59.03s | missing-sections(title+descrip... |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                 0 |                     0 |              0 |            0 |         0 |         5.1 |           57.97s |      0.47s |      58.78s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |   3,158 |             16741 |                   500 |          17241 |          310 |        89 |          35 |           60.83s |      3.83s |      65.02s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |  32,645 |             16741 |                   500 |          17241 |          308 |      90.9 |          12 |           60.84s |      1.33s |      62.51s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |  11,750 |             16741 |                   500 |          17241 |          281 |        63 |          76 |           68.64s |     12.50s |      81.50s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |   2,972 |             16741 |                   500 |          17241 |          233 |      29.1 |          26 |           89.95s |      2.15s |      92.46s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |     328 |             16741 |                   500 |          17241 |          210 |      16.8 |          39 |          110.23s |      3.55s |     114.12s | refusal(explicit_refusal), ...    |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Automated review digest:_ [review.md](review.md)
- _Canonical run log:_ [check_models.log](check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.4.0
- _macOS Version:_ 26.4
- _SDK Version:_ 26.4
- _Xcode Version:_ 26.4
- _Xcode Build:_ 17E192
- _Metal SDK:_ MacOSX.sdk
- _Python Version:_ 3.13.12
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18

## Library Versions

- `numpy`: `2.4.3`
- `mlx`: `0.31.2.dev20260328+0ff1115a`
- `mlx-vlm`: `0.4.2`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.8.0`
- `transformers`: `5.4.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.1.1`

_Report generated on: 2026-03-29 00:31:08 GMT_
