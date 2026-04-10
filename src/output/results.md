# Model Performance Results

_Generated on 2026-04-10 14:58:29 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=7, clean outputs=6/52.
- _Useful now:_ 2 clean A/B model(s) worth first review.
- _Review watchlist:_ 50 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=10, neutral=1, worse=41 (baseline B 76/100).
- _Quality signal frequency:_ missing_sections=35, cutoff=30,
  trusted_hint_ignored=22, context_ignored=22, repetitive=10,
  metadata_borrowing=9.
- _Runtime pattern:_ decode dominates measured phase time (91%; 50/53 measured
  model(s)).
- _Phase totals:_ model load=100.46s, prompt prep=0.16s, decode=1073.38s,
  cleanup=5.13s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=52, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (337.4 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.50s)
- **📊 Average TPS:** 80.7 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1084.5 GB
- **Average peak memory:** 20.9 GB
- **Memory efficiency:** 260 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 9 | ✅ B: 14 | 🟡 C: 14 | 🟠 D: 3 | ❌ F: 12

**Average Utility Score:** 55/100

**Existing Metadata Baseline:** ✅ B (76/100)
**Vs Existing Metadata:** Avg Δ -21 | Better: 10, Neutral: 1, Worse: 41

- **Best for cataloging:** `mlx-community/gemma-4-31b-bf16` (🏆 A, 96/100)
- **Worst for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (❌ F, 0/100)

### ⚠️ 15 Models with Low Utility (D/F)

- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (34/100) - Lacks visual description of image
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (44/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (1/100) - Output too short to be useful
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (35/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: ❌ F (32/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (22/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (22/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (18/100) - Output lacks detail

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (10):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "city bus route, city..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "and they are not..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "birds, flight, scenic, peacefu..."`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `phrase: "the car has a..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `1.`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `qnguyen3/nanoLLaVA` (token: `Neon`)
- **📝 Formatting Issues (5):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 80.7 | Min: 5.04 | Max: 337
- **Peak Memory**: Avg: 21 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 22.71s | Min: 1.25s | Max: 104.84s
- **Generation Time**: Avg: 20.64s | Min: 0.58s | Max: 101.56s
- **Model Load Time**: Avg: 1.89s | Min: 0.50s | Max: 6.25s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (91%; 50/53 measured model(s)).
- **Phase totals:** model load=100.46s, prompt prep=0.16s, decode=1073.38s, cleanup=5.13s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=52, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 9.18s total (avg 0.17s across 53 model(s)).
- **First-token latency:** Avg 11.17s | Min 0.06s | Max 72.41s across 52 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/gemma-4-31b-bf16`](model_gallery.md#model-mlx-community-gemma-4-31b-bf16)
  (A 96/100 | Gen 7.25 TPS | Peak 64 | A 96/100 | hit token cap (500) |
  output/prompt=66.31% | missing sections: title, description, keywords |
  missing terms: flies, low, over, tranquil, style)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (F 32/100 | Gen 337 TPS | Peak 2.7 | F 32/100 | missing sections: keywords |
  context echo=73%)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (F 34/100 | Gen 332 TPS | Peak 2.2 | F 34/100 | missing sections: title,
  description, keywords | missing terms: grey, heron, flies, low, over)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (A 88/100 | Gen 163 TPS | Peak 13 | A 88/100 | nontext prompt burden=90% |
  missing terms: flies, low, style, bird, soaring)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
- _🔄 Repetitive Output (10):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +6 more. Example: token: `unt`.
- _📝 Formatting Issues (5):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +1 more.
- _Low-utility outputs (15):_ [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +11 more. Common weakness: Output too short to be useful.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `model-config` | 1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16` |

### Actionable Items by Package

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD028 MD037 -->
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
> &#45; 5-10 words, concrete and factual, limited to clearly visible content.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> Description:
> &#45; 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> &#45; Output only the description text after the label.
>
> Keywords:
> &#45; 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> &#45; Output only the keyword list after the label.
>
> Rules:
> &#45; Include only details that are definitely visible in the image.
> &#45; Reuse metadata terms only when they are clearly supported by the image.
> &#45; If metadata and image disagree, follow the image.
> &#45; Prefer omission to speculation.
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent unless visually obvious.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually
> confirmed):
> &#45; Description hint: A grey heron flies low over a tranquil pond in a
> Japanese-style garden. The bird is in mid-flight, soaring above the water's
> surface, with a traditional wooden zigzag bridge and lush green landscape
> visible in the background.
> &#45; Capture metadata: Taken on 2026-04-03 14:23:14 BST (at 14:23:14 local
> time). GPS: 45.518800°N, 122.708000°W.
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1189.17s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.23s |       2.40s |                                   |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               492 |                    68 |            560 |         6137 |       337 |         2.7 |            0.58s |      0.50s |       1.25s | missing-sections(keywords), ...   |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               496 |                    22 |            518 |         5089 |       332 |         2.2 |            0.60s |      0.61s |       1.39s | missing-sections(title+desc...    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               745 |                   120 |            865 |         8744 |       330 |         2.9 |            0.69s |      0.51s |       1.38s | fabrication, ...                  |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               553 |                   125 |            678 |         8860 |       192 |         3.8 |            0.96s |      0.52s |       1.66s | metadata-borrowing                |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,521 |                     8 |          1,529 |         3363 |        22 |          11 |            1.12s |      1.45s |       2.75s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,521 |                    25 |          1,546 |         1385 |        32 |          12 |            2.19s |      1.62s |       3.99s | missing-sections(title+desc...    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             3,482 |                     8 |          3,490 |         1528 |      67.1 |         9.7 |            2.78s |      0.95s |       3.91s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             4,093 |                   100 |          4,193 |         2084 |       163 |          13 |            2.91s |      0.93s |       4.03s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,521 |                    12 |          1,533 |         1116 |       5.7 |          27 |            3.78s |      2.48s |       6.43s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               605 |                   500 |          1,105 |         1353 |       131 |         5.8 |            4.66s |      0.61s |       5.45s | missing-sections(title+desc...    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               767 |                    79 |            846 |          546 |      26.1 |          20 |            4.73s |      2.53s |       7.45s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |  76,964 |               492 |                   500 |            992 |         4970 |       112 |         4.8 |            4.86s |      0.57s |       5.60s | repetitive(Neon), ...             |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               756 |                   500 |          1,256 |         2509 |       116 |         6.1 |            4.89s |      1.46s |       6.55s | repetitive(1.), degeneration, ... |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               765 |                    93 |            858 |          616 |      27.6 |          19 |            4.89s |      2.33s |       7.41s |                                   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             2,069 |                   500 |          2,569 |         3933 |       125 |         5.8 |            4.92s |      0.66s |       5.75s | repetitive(unt), ...              |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             2,069 |                   500 |          2,569 |         3896 |       123 |         5.8 |            4.99s |      0.82s |       5.99s | repetitive(unt), ...              |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             3,043 |                    80 |          3,123 |         1383 |        32 |          19 |            5.03s |      1.79s |       7.00s | title-length(2), ...              |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | 134,421 |             1,493 |                   500 |          1,993 |         1798 |       128 |          18 |            5.10s |      1.99s |       7.27s | degeneration, ...                 |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     667 |             1,493 |                   500 |          1,993 |         1819 |       116 |          22 |            5.54s |      2.14s |       7.87s | repetitive(phrase: "and the...    |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             4,641 |                    99 |          4,740 |         1706 |      38.8 |          18 |            5.59s |      1.69s |       7.46s | context-echo(0.54)                |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             4,094 |                   109 |          4,203 |         1052 |        64 |          18 |            5.93s |      1.33s |       7.43s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             4,094 |                   107 |          4,201 |         1041 |      61.3 |          19 |            6.00s |      1.38s |       7.56s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               765 |                    98 |            863 |          572 |      16.5 |          33 |            7.55s |      3.39s |      11.12s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      12 |             1,493 |                   500 |          1,993 |         1782 |      78.7 |          37 |            7.67s |      3.28s |      11.12s | degeneration, ...                 |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             3,598 |                    82 |          3,680 |          698 |      31.6 |          27 |            8.08s |      2.05s |      10.32s | ⚠️harness(encoding), ...          |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               764 |                   373 |          1,137 |         1814 |      48.9 |          17 |            8.32s |      2.24s |      10.75s | fabrication, ...                  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |     713 |             1,312 |                   500 |          1,812 |         3877 |      57.4 |         9.5 |            9.31s |      0.92s |      10.41s | keyword-count(180), ...           |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |     713 |             1,312 |                   500 |          1,812 |         3926 |      56.2 |         9.5 |            9.48s |      0.89s |      10.53s | keyword-count(180), ...           |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |  36,407 |             4,593 |                   500 |          5,093 |         3958 |      48.8 |         4.6 |           11.92s |      1.10s |      13.20s | repetitive(phrase: "- outpu...    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |     274 |             1,789 |                   500 |          2,289 |         1306 |      43.3 |          60 |           13.45s |      4.76s |      18.40s | degeneration, ...                 |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |  34,674 |             6,545 |                   500 |          7,045 |          932 |      68.7 |         8.4 |           14.56s |      1.29s |      16.03s | missing-sections(title+desc...    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   7,285 |             4,732 |                   500 |          5,232 |         1574 |      41.3 |          15 |           15.43s |      1.64s |      17.26s | missing-sections(title+desc...    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  62,052 |             6,545 |                   500 |          7,045 |          962 |      49.5 |          11 |           17.16s |      1.39s |      18.73s | missing-sections(title+desc...    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             3,485 |                   500 |          3,985 |         2422 |        32 |          20 |           17.46s |      1.93s |      19.57s | missing-sections(title+desc...    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  44,585 |             3,043 |                   500 |          3,543 |         2905 |      33.8 |          19 |           19.56s |      1.72s |      21.46s | repetitive(phrase: "rencont...    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |   3,283 |            16,741 |                   500 |         17,241 |         1252 |      89.4 |         8.6 |           19.63s |      0.71s |      20.52s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      13 |            16,743 |                   500 |         17,243 |         1148 |      87.7 |         8.6 |           21.03s |      0.78s |      22.01s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | 151,645 |            16,752 |                   110 |         16,862 |          818 |      55.1 |          13 |           23.08s |      1.15s |      24.42s | title-length(4)                   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |  10,764 |             6,545 |                   500 |          7,045 |          495 |      34.6 |          78 |           27.95s |      5.47s |      33.61s | missing-sections(title+desc...    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |   2,522 |               461 |                   500 |            961 |          291 |      18.9 |          15 |           28.27s |      1.60s |      30.05s | repetitive(phrase: "birds,...     |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |  10,769 |             1,671 |                   500 |          2,171 |         88.3 |      52.2 |          41 |           28.98s |      1.25s |      30.41s | fabrication, ...                  |                 |
| `mlx-community/pixtral-12b-bf16`                        |   1,278 |             4,641 |                   500 |          5,141 |         2041 |        20 |          29 |           31.41s |      2.57s |      34.16s | missing-sections(title+desc...    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               462 |                   171 |            633 |          252 |      5.04 |          25 |           36.07s |      2.21s |      38.46s | missing-sections(title+desc...    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,671 |                   336 |          2,007 |         85.7 |      30.3 |          48 |           51.42s |      1.74s |      53.34s | missing-sections(title+desc...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |   2,099 |            16,767 |                   500 |         17,267 |          324 |       104 |          26 |           57.26s |      2.51s |      59.95s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |  20,226 |            16,767 |                   500 |         17,267 |          329 |      88.5 |          35 |           57.35s |      3.16s |      60.69s | missing-sections(title+desc...    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,752 |                     6 |         16,758 |          287 |       226 |         5.1 |           58.91s |      0.54s |      59.62s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |     353 |            16,767 |                   500 |         17,267 |          320 |      63.3 |          76 |           61.01s |      6.25s |      67.45s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |     321 |            16,767 |                   500 |         17,267 |          285 |      81.6 |          12 |           65.65s |      1.66s |      67.53s | missing-sections(descriptio...    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |  21,904 |               754 |                   500 |          1,254 |          332 |      7.25 |          64 |           71.57s |      5.89s |      77.64s | fabrication, ...                  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |  66,565 |            16,767 |                   500 |         17,267 |          235 |      28.8 |          26 |           89.55s |      2.15s |      91.90s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |      13 |            16,767 |                   500 |         17,267 |          232 |      17.6 |          39 |          101.56s |      3.08s |     104.84s | refusal(explicit_refusal), ...    |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Automated review digest:_ [review.md](review.md)
- _Canonical run log:_ [check_models.log](check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.4.0
- _macOS Version:_ 26.4.1
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

- `numpy`: `2.4.4`
- `mlx`: `0.31.2.dev20260410+a33b7916`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.10.1`
- `transformers`: `5.5.3`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-10 14:58:29 BST_
