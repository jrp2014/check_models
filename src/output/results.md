# Model Performance Results

_Generated on 2026-03-15 00:03:29 GMT_

## 🎯 Action Snapshot

- **Framework/runtime failures:** 1 (top owners: mlx-vlm=1).
- **Next action:** review failure ownership below and use diagnostics.md for filing.
- **Maintainer signals:** harness-risk successes=11, clean outputs=1/49.
- **Useful now:** 2 clean A/B model(s) worth first review.
- **Review watchlist:** 47 model(s) with breaking or lower-value output.
- **Vs existing metadata:** better=9, neutral=3, worse=37 (baseline B 80/100).
- **Quality signal frequency:** missing_sections=26, context_ignored=25, description_length=19, keyword_count=13, reasoning_leak=11, harness=11.
- **Runtime pattern:** decode dominates measured phase time (88%; 49/50 measured model(s)).
- **Phase totals:** model load=209.73s, prompt prep=0.13s, decode=1570.87s, cleanup=5.42s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=49, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `prince-canuma/Florence-2-large-ft` (306.5 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.68s)
- **📊 Average TPS:** 75.3 across 49 models

## 📈 Resource Usage

- **Total peak memory:** 928.6 GB
- **Average peak memory:** 19.0 GB
- **Memory efficiency:** 259 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 10 | ✅ B: 22 | 🟡 C: 6 | ❌ F: 11

**Average Utility Score:** 59/100

**Existing Metadata Baseline:** ✅ B (80/100)
**Vs Existing Metadata:** Avg Δ -21 | Better: 9, Neutral: 3, Worse: 37

- **Best for cataloging:** `mlx-community/Qwen3.5-27B-mxfp8` (🏆 A, 95/100)
- **Worst for cataloging:** `mlx-community/llava-v1.6-mistral-7b-8bit` (❌ F, 0/100)

### ⚠️ 11 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (1/100) - Output lacks detail
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (1/100) - Output lacks detail
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (25/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (11/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (15/100) - Output lacks detail
- `prince-canuma/Florence-2-large-ft`: ❌ F (0/100) - Output too short to be useful
- `qnguyen3/nanoLLaVA`: ❌ F (20/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (10):**
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "england, england, england, eng..."`)
  - `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (token: `phrase: "there's a sign for..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "16:19:59, 51.800450, 0.207617,..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "white balance: accurate. white..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "white balance: accurate. white..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "england, english pub exterior,..."`)
  - `mlx-community/X-Reasoner-7B-8bit` (token: `phrase: "pub entrance signpost arrow..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "14: 14: 14: 14:..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the left side of..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- do not output..."`)
- **👻 Hallucinations (1):**
  - `microsoft/Phi-3.5-vision-instruct`
- **📝 Formatting Issues (5):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `prince-canuma/Florence-2-large-ft`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 75.3 | Min: 3.86 | Max: 306
- **Peak Memory**: Avg: 19 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 35.86s | Min: 2.61s | Max: 226.35s
- **Generation Time**: Avg: 32.05s | Min: 1.60s | Max: 220.36s
- **Model Load Time**: Avg: 3.45s | Min: 0.68s | Max: 17.46s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (88%; 49/50 measured model(s)).
- **Phase totals:** model load=209.73s, prompt prep=0.13s, decode=1570.87s, cleanup=5.42s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=49, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 17.29s total (avg 0.35s across 50 model(s)).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- **Best cataloging quality:** [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8) (A 95/100 | Gen 12.3 TPS | Peak 35 | A 95/100 | ⚠️HARNESS:long_context; Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Captured, March, Adobe Stock, Any Vision, British, Chalkboard, English Pub Exterior, English, Europe, Hertfordshire, Traditional British Architecture, United Kingdom
Capture, GMT, GPS); ...)
- **Fastest generation:** [`prince-canuma/Florence-2-large-ft`](model_gallery.md#model-prince-canuma-florence-2-large-ft) (F 0/100 | Gen 306 TPS | Peak 5.1 | F 0/100 | ⚠️HARNESS:stop_token; Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, The Doctors Tonic, Captured, March, Adobe Stock, Any Vision, British, Chalkboard, English Pub Exterior, English, Europe, Food, Greene King, Hertfordshire, Signage, Spring, Sunday Roast, Traditional British Architecture, United Kingdom
Capture, GMT, GPS); ...)
- **Lowest memory footprint:** [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16) (C 51/100 | Gen 300 TPS | Peak 2.2 | C 51/100 | Context ignored (missing: Howardsgate, United Kingdom, Captured, Adobe Stock, Any Vision, Chalkboard, English Pub Exterior, English, Europe, Food, Greene King, Hertfordshire, Signage, Sunday Roast, Traditional British Architecture, United Kingdom
Capture, GMT, GPS); Missing sections (keywords); ...)
- **Best balance:** [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4) (C 59/100 | Gen 42.6 TPS | Peak 13 | C 59/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery when available.

- **❌ Failed Models (1):** [`microsoft/Florence-2-large-ft`](model_gallery.md#model-microsoft-florence-2-large-ft). Example: `Model Error`.
- **🔄 Repetitive Output (10):** [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct), [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](model_gallery.md#model-mlx-community-apriel-15-15b-thinker-6bit-mlx), [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](model_gallery.md#model-mlx-community-llama-32-11b-vision-instruct-8bit), [`mlx-community/Molmo-7B-D-0924-8bit`](model_gallery.md#model-mlx-community-molmo-7b-d-0924-8bit), +6 more. Example: token: `phrase: "england, england, england, eng..."`.
- **👻 Hallucinations (1):** [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct).
- **📝 Formatting Issues (5):** [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit), [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4), [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4), [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16), +1 more.
- **Low-utility outputs (11):** [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct), [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit), [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit), [`mlx-community/SmolVLM-Instruct-bf16`](model_gallery.md#model-mlx-community-smolvlm-instruct-bf16), +7 more. Common weakness: Output lacks detail.

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `mlx-vlm` | 1 | Model Error | `microsoft/Florence-2-large-ft` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### mlx-vlm

- **microsoft/Florence-2-large-ft** (Model Error)
  - Error: `Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate ...`
  - Type: `ValueError`

**Prompt used:**

<!-- markdownlint-disable MD028 MD049 -->
>
> Analyze this image for cataloguing metadata.
>
> Use only details that are clearly and definitely visible in the image. If a detail is
> uncertain, ambiguous, partially obscured, too small to verify, or not directly visible,
> leave it out. Do not guess.
>
> Treat the metadata hints below as a draft catalog record. Keep only details that are
> clearly confirmed by the image, correct anything contradicted by the image, and add
> important visible details that are definitely present.
>
> Return exactly these three sections, and nothing else:
>
> Title:
> \- 5-10 words, concrete and factual, limited to clearly visible content.
> \- Output only the title text after the label.
> \- Do not repeat or paraphrase these instructions in the title.
>
> Description:
> \- 1-2 factual sentences describing the main visible subject, setting, lighting, action,
> and other distinctive visible details. Omit anything uncertain or inferred.
> \- Output only the description text after the label.
>
> Keywords:
> \- 10-18 unique comma-separated terms based only on clearly visible subjects, setting,
> colors, composition, and style. Omit uncertain tags rather than guessing.
> \- Output only the keyword list after the label.
>
> Rules:
> \- Include only details that are definitely visible in the image.
> \- Reuse metadata terms only when they are clearly supported by the image.
> \- If metadata and image disagree, follow the image.
> \- Prefer omission to speculation.
> \- Do not copy prompt instructions into the Title, Description, or Keywords fields.
> \- Do not infer identity, location, event, brand, species, time period, or intent unless
> visually obvious.
> \- Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually confirmed):
> \- Description hint: , Howardsgate, Welwyn Garden City, England, United Kingdom, UK The
> late afternoon sun casts a warm glow on The Doctors Tonic pub in Welwyn Garden City,
> England. Captured on a clear day in early spring, the image highlights the contrast
> between the still-bare trees of March and the deep green of the manicured hedges. The
> pub's traditional brick architecture, with its tiled roof and dormer windows, stands out
> agains...
> \- Keyword hints: Adobe Stock, Any Vision, British, Chalkboard sign, England, English Pub
> Exterior, English pub, Europe, Food and drink, Greene King, Hertfordshire, Howardsgate,
> Signage, Spring afternoon, Sunday Roast sign, The Doctors Tonic, The Doctors Tonic pub,
> Traditional British Architecture, UK, United Kingdom
> \- Capture metadata: Taken on 2026-03-14 16:19:59 GMT (at 16:19:59 local time). GPS:
> 51.800450°N, 0.207617°W.
<!-- markdownlint-enable MD028 MD049 -->

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 1804.58s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |            0.37s |     40.66s |      41.36s |                                   |         mlx-vlm |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |  151645 |               608 |                   229 |            837 |         2707 |       276 |         2.9 |            1.60s |      0.68s |       2.61s | title-length(22), ...             |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |  151645 |               612 |                   175 |            787 |         3741 |       300 |         2.2 |            1.67s |      0.77s |       2.77s | missing-sections(keywords), ...   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               864 |                   194 |          1,058 |         3033 |       180 |           4 |            1.84s |      0.84s |       3.00s | description-sentences(4), ...     |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   49154 |             1,821 |                    17 |          1,838 |         1396 |       115 |         5.6 |            1.91s |      1.07s |       3.28s | missing-sections(title+descrip... |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   49154 |             1,821 |                    17 |          1,838 |         1368 |       114 |         5.6 |            1.93s |      0.93s |       3.22s | missing-sections(title+descrip... |                 |
| `prince-canuma/Florence-2-large-ft`                     |       0 |             1,180 |                   500 |          1,680 |         4776 |       306 |         5.1 |            2.30s |      1.05s |       3.92s | ⚠️harness(stop_token), ...        |                 |
| `qnguyen3/nanoLLaVA`                                    |  151645 |               608 |                   156 |            764 |         2578 |        97 |         4.9 |            2.36s |      0.80s |       3.49s | missing-sections(keywords), ...   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |    5400 |               864 |                   500 |          1,364 |         3060 |       290 |         2.9 |            2.53s |      0.72s |       3.57s | missing-sections(description+k... |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |   49279 |             1,721 |                   119 |          1,840 |         1363 |       111 |         5.5 |            2.79s |      0.94s |       4.04s | missing-sections(keywords), ...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,627 |                    12 |          1,639 |          452 |      27.7 |          12 |            4.45s |      3.64s |       8.42s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |   32007 |             1,433 |                   175 |          1,608 |         1242 |      50.3 |         9.6 |            5.09s |      1.49s |       6.94s | description-sentences(3), ...     |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               870 |                   155 |          1,025 |          899 |      40.8 |          17 |            5.18s |      4.69s |      10.27s | missing-sections(title+descrip... |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,204 |                   162 |          3,366 |          844 |       155 |         7.8 |            5.26s |      1.51s |       7.22s | description-sentences(3)          |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |  236770 |               862 |                   500 |          1,362 |         1409 |       105 |         6.2 |            5.82s |      3.18s |       9.44s | repetitive(phrase: "14: 14: 14... |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  151645 |             2,391 |                   100 |          2,491 |          816 |      32.1 |          17 |            6.61s |      2.81s |       9.75s | description-sentences(3), ...     |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,627 |                    13 |          1,640 |          451 |      5.04 |          27 |            6.63s |      5.40s |      12.36s | ⚠️harness(prompt_template), ...   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |    4760 |             1,598 |                   500 |          2,098 |          896 |       111 |          18 |            6.93s |      3.13s |      10.43s | description-sentences(5), ...     |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |   42856 |             1,598 |                   500 |          2,098 |          959 |      99.6 |          22 |            7.32s |      3.47s |      11.12s | fabrication, title-length(3), ... |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               871 |                   116 |            987 |          190 |      24.3 |          19 |            9.77s |      4.66s |      14.81s | context-ignored                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |  128001 |             2,890 |                   163 |          3,053 |          751 |      29.7 |          18 |            9.78s |      3.35s |      13.47s | description-sentences(4), ...     |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |   14383 |             1,598 |                   500 |          2,098 |          962 |      67.6 |          37 |            9.79s |      5.67s |      15.78s | missing-sections(title), ...      |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |    6332 |             1,433 |                   500 |          1,933 |         1289 |      51.7 |         9.6 |           11.22s |      1.50s |      13.02s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/InternVL3-14B-8bit`                      |  151645 |             2,391 |                   147 |          2,538 |          394 |      28.6 |          17 |           11.79s |      2.92s |      15.04s | title-length(4), ...              |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,395 |                   128 |          3,523 |          452 |      30.5 |          16 |           12.10s |      2.92s |      15.40s | description-sentences(3), ...     |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,820 |                     8 |          2,828 |          231 |      46.5 |          11 |           12.79s |      1.62s |      14.74s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,205 |                   123 |          3,328 |          312 |      51.7 |          13 |           13.08s |      2.24s |      15.65s | context-ignored                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               871 |                   116 |            987 |          185 |      13.8 |          34 |           13.54s |      6.79s |      20.69s | context-ignored                   |                 |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,395 |                   123 |          3,518 |          484 |      17.3 |          28 |           14.53s |      4.64s |      19.60s | description-sentences(3), ...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,205 |                   144 |          3,349 |          287 |      42.6 |          13 |           14.99s |      2.33s |      17.67s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,707 |                   111 |          2,818 |          235 |      27.8 |          22 |           15.89s |      3.57s |      19.80s | ⚠️harness(encoding), ...          |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |    8290 |             4,699 |                   500 |          5,199 |         1495 |      36.5 |         4.5 |           17.44s |      3.10s |      20.94s | repetitive(phrase: "- do not o... |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |     304 |             1,925 |                   500 |          2,425 |          257 |      48.7 |          60 |           18.63s |     12.94s |      31.90s | keyword-count(48), ...            |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |    2287 |             3,486 |                   500 |          3,986 |          416 |      35.1 |          15 |           23.02s |      2.82s |      26.17s | repetitive(phrase: "there's a...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |  128009 |               580 |                    91 |            671 |          154 |      3.86 |          25 |           27.75s |      4.05s |      32.12s | missing-sections(title+descrip... |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |    2168 |             6,722 |                   500 |          7,222 |          301 |      61.7 |         8.4 |           30.85s |      2.14s |      33.31s | missing-sections(title+descrip... |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |     374 |             6,722 |                   500 |          7,222 |          317 |      46.7 |          11 |           32.39s |      2.35s |      35.07s | degeneration, ...                 |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |     573 |             1,627 |                   500 |          2,127 |         1498 |      15.9 |          11 |           32.91s |      3.23s |      36.52s | repetitive(phrase: "the left s... |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |   61159 |             1,787 |                   500 |          2,287 |         53.3 |      38.5 |          41 |           47.42s |      1.92s |      49.66s | repetitive(phrase: "white bala... |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |      11 |            16,825 |                   500 |         17,325 |          370 |      74.1 |         8.3 |           52.72s |      1.34s |      54.39s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |   61159 |             1,787 |                   500 |          2,287 |         53.3 |      26.6 |          48 |           53.25s |      3.21s |      56.83s | repetitive(phrase: "white bala... |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |      11 |            16,836 |                   500 |         17,336 |          370 |      42.7 |          13 |           57.74s |      2.14s |      60.21s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |      30 |             6,722 |                   500 |          7,222 |          157 |      30.4 |          78 |           59.94s |     12.90s |      73.17s | missing-sections(title+descrip... |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |    2946 |               579 |                   500 |          1,079 |          128 |      6.97 |          15 |           76.68s |      2.56s |      79.55s | repetitive(phrase: "16:19:59, ... |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |    1163 |            16,827 |                   500 |         17,327 |          226 |      60.1 |         8.3 |           83.44s |      1.49s |      85.29s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |    8343 |            16,850 |                   500 |         17,350 |          205 |      72.6 |          33 |           89.53s |      5.72s |      95.58s | missing-sections(title+descrip... |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |      12 |            16,850 |                   500 |         17,350 |          191 |      52.5 |          74 |           98.51s |     17.46s |     116.30s | missing-sections(title+descrip... |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |      11 |            16,836 |                   500 |         17,336 |          138 |       156 |         5.1 |          125.42s |      1.07s |     126.82s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |      11 |            16,850 |                   500 |         17,350 |         94.5 |      22.6 |          22 |          201.00s |      3.71s |     205.34s | missing-sections(title+descrip... |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |     561 |            16,850 |                   500 |         17,350 |           94 |      12.3 |          35 |          220.36s |      5.60s |     226.35s | ⚠️harness(long_context), ...      |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

**Dedicated review artifact:**
See the standalone model-by-model output view here:
[model_gallery.md](model_gallery.md)

---

## System/Hardware Information

- **OS**: Darwin 25.3.0
- **macOS Version**: 26.3.1
- **SDK Version**: 26.2
- **Xcode Version**: 26.3
- **Xcode Build**: 17C529
- **Metal SDK**: MacOSX.sdk
- **Python Version**: 3.13.12
- **Architecture**: arm64
- **GPU/Chip**: Apple M4 Max
- **GPU Cores**: 40
- **Metal Support**: Metal 4
- **RAM**: 128.0 GB
- **CPU Cores (Physical)**: 16
- **CPU Cores (Logical)**: 16

## Library Versions

- `numpy`: `2.4.3`
- `mlx`: `0.31.2.dev20260314+5d170049`
- `mlx-vlm`: `0.4.0`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.7.1`
- `transformers`: `5.3.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.1.1`

_Report generated on: 2026-03-15 00:03:29 GMT_
