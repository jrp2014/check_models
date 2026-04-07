# Model Performance Results

_Generated on 2026-04-06 23:33:33 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=7, clean outputs=6/51.
- _Useful now:_ 3 clean A/B model(s) worth first review.
- _Review watchlist:_ 48 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=9, neutral=2, worse=40 (baseline B 76/100).
- _Quality signal frequency:_ missing_sections=33, cutoff=28,
  context_ignored=22, trusted_hint_ignored=22, repetitive=10, title_length=8.
- _Runtime pattern:_ decode dominates measured phase time (90%; 49/52 measured
  model(s)).
- _Phase totals:_ model load=103.05s, prompt prep=0.17s, decode=959.35s,
  cleanup=5.08s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=51, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (348.5 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.43s)
- **📊 Average TPS:** 83.1 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1020.2 GB
- **Average peak memory:** 20.0 GB
- **Memory efficiency:** 275 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 8 | ✅ B: 14 | 🟡 C: 14 | 🟠 D: 3 | ❌ F: 12

**Average Utility Score:** 55/100

**Existing Metadata Baseline:** ✅ B (76/100)
**Vs Existing Metadata:** Avg Δ -22 | Better: 9, Neutral: 2, Worse: 40

- **Best for cataloging:** `mlx-community/InternVL3-14B-8bit` (🏆 A, 96/100)
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

- **Generation Tps**: Avg: 83.1 | Min: 5.02 | Max: 349
- **Peak Memory**: Avg: 20 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 20.97s | Min: 1.18s | Max: 99.33s
- **Generation Time**: Avg: 18.81s | Min: 0.58s | Max: 96.03s
- **Model Load Time**: Avg: 1.98s | Min: 0.43s | Max: 10.79s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 49/52 measured model(s)).
- **Phase totals:** model load=103.05s, prompt prep=0.17s, decode=959.35s, cleanup=5.08s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=51, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 9.07s total (avg 0.17s across 52 model(s)).
- **First-token latency:** Avg 10.75s | Min 0.08s | Max 68.53s across 51 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/InternVL3-14B-8bit`](model_gallery.md#model-mlx-community-internvl3-14b-8bit)
  (A 96/100 | Gen 32.0 TPS | Peak 19 | A 96/100 | nontext prompt burden=86% |
  missing terms: heron, flies, low, over, tranquil)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (F 32/100 | Gen 349 TPS | Peak 2.6 | F 32/100 | missing sections: keywords |
  context echo=73%)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (F 34/100 | Gen 322 TPS | Peak 2.2 | F 34/100 | missing sections: title,
  description, keywords | missing terms: grey, heron, flies, low, over)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (A 88/100 | Gen 172 TPS | Peak 13 | A 88/100 | nontext prompt burden=90% |
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

_Overall runtime:_ 1077.45s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.18s |       2.36s |                                   |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               492 |                    68 |            560 |         6119 |       349 |         2.6 |            0.58s |      0.43s |       1.18s | missing-sections(keywords), ...   |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               496 |                    22 |            518 |         5122 |       322 |         2.2 |            0.61s |      0.61s |       1.40s | missing-sections(title+desc...    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               745 |                   120 |            865 |         8786 |       330 |         2.9 |            0.69s |      0.49s |       1.36s | fabrication, ...                  |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               553 |                   125 |            678 |         6501 |       188 |         3.8 |            0.97s |      0.66s |       1.80s | metadata-borrowing                |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,521 |                     8 |          1,529 |         3243 |      22.1 |          11 |            1.14s |      1.51s |       2.82s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,521 |                    25 |          1,546 |         1379 |      31.4 |          12 |            2.21s |      1.54s |       3.93s | missing-sections(title+desc...    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             3,482 |                     8 |          3,490 |         1628 |      69.8 |         9.7 |            2.64s |      0.97s |       3.79s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             4,093 |                   100 |          4,193 |         2219 |       172 |          13 |            2.76s |      0.93s |       3.87s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,521 |                    12 |          1,533 |         1098 |      5.72 |          27 |            3.80s |      2.41s |       6.39s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               765 |                    93 |            858 |          652 |      31.5 |          19 |            4.41s |      2.32s |       6.91s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               605 |                   500 |          1,105 |         1441 |       132 |         5.8 |            4.62s |      0.62s |       5.42s | missing-sections(title+desc...    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               756 |                   500 |          1,256 |         2487 |       124 |         6.1 |            4.62s |      1.43s |       6.24s | repetitive(1.), degeneration, ... |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               767 |                    79 |            846 |          514 |      27.3 |          20 |            4.68s |      2.54s |       7.42s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |  76,964 |               492 |                   500 |            992 |         4744 |       113 |         4.6 |            4.84s |      0.52s |       5.53s | repetitive(Neon), ...             |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             2,069 |                   500 |          2,569 |         3959 |       124 |         5.8 |            4.93s |      0.65s |       5.74s | repetitive(unt), ...              |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             3,043 |                    80 |          3,123 |         1442 |        32 |          19 |            4.94s |      1.78s |       6.90s | title-length(2), ...              |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             2,069 |                   500 |          2,569 |         3952 |       122 |         5.8 |            5.02s |      0.68s |       5.88s | repetitive(unt), ...              |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | 134,421 |             1,493 |                   500 |          1,993 |         1772 |       128 |          18 |            5.14s |      1.98s |       7.30s | degeneration, ...                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             4,094 |                   109 |          4,203 |         1193 |      64.5 |          18 |            5.46s |      1.31s |       6.95s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             4,094 |                   107 |          4,201 |         1190 |      61.1 |          19 |            5.53s |      1.32s |       7.02s |                                   |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             4,641 |                    99 |          4,740 |         1714 |        39 |          18 |            5.57s |      1.71s |       7.48s | context-echo(0.54)                |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     667 |             1,493 |                   500 |          1,993 |         1794 |       115 |          22 |            5.60s |      2.14s |       7.92s | repetitive(phrase: "and the...    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               765 |                    98 |            863 |          562 |      17.8 |          33 |            7.15s |      3.39s |      10.73s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      12 |             1,493 |                   500 |          1,993 |         1777 |        79 |          37 |            7.66s |      3.22s |      11.07s | degeneration, ...                 |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               764 |                   373 |          1,137 |         1809 |      48.5 |          17 |            8.38s |      2.22s |      10.79s | fabrication, ...                  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             3,598 |                    82 |          3,680 |          638 |        31 |          27 |            8.64s |      2.12s |      10.97s | ⚠️harness(encoding), ...          |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |     713 |             1,312 |                   500 |          1,812 |         3805 |      56.4 |         9.5 |            9.46s |      0.94s |      10.57s | keyword-count(180), ...           |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |     713 |             1,312 |                   500 |          1,812 |         3804 |      55.9 |         9.5 |            9.54s |      0.84s |      10.55s | keyword-count(180), ...           |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |  34,674 |             6,545 |                   500 |          7,045 |         1162 |      71.4 |         8.4 |           12.91s |      1.32s |      14.41s | missing-sections(title+desc...    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |  36,407 |             4,593 |                   500 |          5,093 |         3923 |      40.2 |         4.6 |           14.15s |      1.11s |      15.44s | repetitive(phrase: "- outpu...    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  62,052 |             6,545 |                   500 |          7,045 |         1169 |      54.5 |          11 |           15.04s |      1.41s |      16.64s | missing-sections(title+desc...    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   7,285 |             4,732 |                   500 |          5,232 |         1508 |      40.5 |          15 |           15.81s |      1.67s |      17.66s | missing-sections(title+desc...    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |     274 |             1,789 |                   500 |          2,289 |          480 |      42.8 |          60 |           15.96s |      7.07s |      23.22s | degeneration, ...                 |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             3,485 |                   500 |          3,985 |         2441 |      32.1 |          20 |           17.42s |      1.94s |      19.54s | missing-sections(title+desc...    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | 151,645 |            16,752 |                   110 |         16,862 |         1097 |      58.1 |          13 |           17.77s |      1.20s |      19.15s | title-length(4)                   |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  44,585 |             3,043 |                   500 |          3,543 |         3024 |      34.1 |          19 |           19.36s |      1.75s |      21.29s | repetitive(phrase: "rencont...    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |   3,283 |            16,741 |                   500 |         17,241 |         1234 |      88.2 |         8.6 |           19.95s |      0.72s |      20.86s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      13 |            16,743 |                   500 |         17,243 |         1167 |      87.6 |         8.6 |           20.81s |      0.76s |      21.76s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,671 |                   326 |          1,997 |         91.8 |      52.2 |          41 |           25.00s |      1.18s |      26.36s | description-sentences(7)          |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |   2,522 |               461 |                   500 |            961 |          289 |        21 |          15 |           25.67s |      1.58s |      27.43s | repetitive(phrase: "birds,...     |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |  10,764 |             6,545 |                   500 |          7,045 |          403 |      37.4 |          78 |           29.87s |      7.80s |      37.86s | missing-sections(title+desc...    |                 |
| `mlx-community/pixtral-12b-bf16`                        |   1,278 |             4,641 |                   500 |          5,141 |         2052 |      19.9 |          29 |           31.48s |      2.51s |      34.17s | missing-sections(title+desc...    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               462 |                   171 |            633 |          238 |      5.02 |          25 |           36.31s |      2.22s |      38.72s | missing-sections(title+desc...    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,671 |                   258 |          1,929 |         86.4 |      30.2 |          48 |           48.62s |      1.76s |      50.56s | missing-sections(title+desc...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |   2,099 |            16,767 |                   500 |         17,267 |          350 |       107 |          26 |           53.28s |      2.53s |      55.99s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |  20,226 |            16,767 |                   500 |         17,267 |          352 |        90 |          35 |           53.96s |      3.12s |      57.26s | missing-sections(title+desc...    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |     321 |            16,767 |                   500 |         17,267 |          348 |      91.3 |          12 |           54.32s |      1.36s |      55.87s | missing-sections(descriptio...    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,752 |                     6 |         16,758 |          289 |       233 |         5.1 |           58.65s |      0.51s |      59.33s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |     353 |            16,767 |                   500 |         17,267 |          327 |      64.6 |          76 |           59.65s |     10.79s |      70.62s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |  66,565 |            16,767 |                   500 |         17,267 |          245 |      30.3 |          26 |           85.76s |      2.19s |      88.15s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |      13 |            16,767 |                   500 |         17,267 |          247 |      18.3 |          39 |           96.03s |      3.10s |      99.33s | refusal(explicit_refusal), ...    |                 |

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

- `numpy`: `2.4.4`
- `mlx`: `0.31.2.dev20260406+b98831ad`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.9.0`
- `transformers`: `5.5.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-06 23:33:33 BST_
