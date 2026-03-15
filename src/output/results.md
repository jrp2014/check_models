# Model Performance Results

_Generated on 2026-03-15 02:26:33 GMT_

## 🎯 Action Snapshot

- **Framework/runtime failures:** none.
- **Maintainer signals:** harness-risk successes=8, clean outputs=4/49.
- **Useful now:** 8 clean A/B model(s) worth first review.
- **Review watchlist:** 41 model(s) with breaking or lower-value output.
- **Vs existing metadata:** better=25, neutral=7, worse=17 (baseline B 66/100).
- **Quality signal frequency:** missing_sections=28, context_ignored=20, description_length=15, keyword_count=11, reasoning_leak=11, title_length=9.
- **Runtime pattern:** decode dominates measured phase time (90%; 49/49 measured model(s)).
- **Phase totals:** model load=161.38s, prompt prep=0.13s, decode=1462.65s, cleanup=5.37s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=49.

## 🏆 Performance Highlights

- **Fastest:** `prince-canuma/Florence-2-large-ft` (321.4 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `microsoft/Phi-3.5-vision-instruct` (0.52s)
- **📊 Average TPS:** 77.7 across 49 models

## 📈 Resource Usage

- **Total peak memory:** 928.8 GB
- **Average peak memory:** 19.0 GB
- **Memory efficiency:** 257 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 14 | ✅ B: 18 | 🟡 C: 6 | 🟠 D: 1 | ❌ F: 10

**Average Utility Score:** 62/100

**Existing Metadata Baseline:** ✅ B (66/100)
**Vs Existing Metadata:** Avg Δ -4 | Better: 25, Neutral: 7, Worse: 17

- **Best for cataloging:** `mlx-community/Qwen3.5-35B-A3B-6bit` (🏆 A, 95/100)
- **Worst for cataloging:** `prince-canuma/Florence-2-large-ft` (❌ F, 0/100)

### ⚠️ 11 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (1/100) - Output lacks detail
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (29/100) - Mostly echoes context without adding value
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (33/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (26/100) - Mostly echoes context without adding value
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (1/100) - Output lacks detail
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (20/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (16/100) - Output lacks detail
- `prince-canuma/Florence-2-large-ft`: ❌ F (0/100) - Output too short to be useful
- `qnguyen3/nanoLLaVA`: ❌ F (19/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **🔄 Repetitive Output (3):**
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "nature, people, shopping, car,..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `16:21:00`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
- **👻 Hallucinations (1):**
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit`
- **📝 Formatting Issues (5):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `prince-canuma/Florence-2-large-ft`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 77.7 | Min: 3.74 | Max: 321
- **Peak Memory**: Avg: 19 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 33.52s | Min: 1.86s | Max: 194.24s
- **Generation Time**: Avg: 29.85s | Min: 1.01s | Max: 188.26s
- **Model Load Time**: Avg: 3.29s | Min: 0.52s | Max: 15.33s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 49/49 measured model(s)).
- **Phase totals:** model load=161.38s, prompt prep=0.13s, decode=1462.65s, cleanup=5.37s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=49.

### ⏱ Timing Snapshot

- **Validation overhead:** 17.64s total (avg 0.36s across 49 model(s)).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- **Best cataloging quality:** [`mlx-community/Qwen3.5-35B-A3B-6bit`](model_gallery.md#model-mlx-community-qwen35-35b-a3b-6bit) (A 95/100 | Gen 68.4 TPS | Peak 33 | A 95/100 | ⚠️HARNESS:long_context; Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Adobe Stock, Any Vision, Eco, Europe, Hertfordshire, Locations, Shoppers, Spring, Sustainable, Urban Greening
Capture, GMT, GPS); ...)
- **Fastest generation:** [`prince-canuma/Florence-2-large-ft`](model_gallery.md#model-prince-canuma-florence-2-large-ft) (F 0/100 | Gen 321 TPS | Peak 5.1 | F 0/100 | ⚠️HARNESS:stop_token; Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Sainsbury, Adobe Stock, Any Vision, Eco, Europe, Hertfordshire, Locations, Modern, Supermarket, Shoppers, Shopping Bags, Spring, Sustainable, Urban Greening
Capture, GMT, GPS); ...)
- **Lowest memory footprint:** [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16) (B 74/100 | Gen 305 TPS | Peak 2.2 | B 74/100 | Missing sections (keywords); Description sentence violation (7; expected 1-2))
- **Best balance:** [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4) (B 69/100 | Gen 55.6 TPS | Peak 13 | B 69/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery when available.

- **🔄 Repetitive Output (3):** [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct), [`mlx-community/gemma-3n-E2B-4bit`](model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit), [`mlx-community/paligemma2-3b-pt-896-4bit`](model_gallery.md#model-mlx-community-paligemma2-3b-pt-896-4bit). Example: token: `phrase: "nature, people, shopping, car,..."`.
- **👻 Hallucinations (1):** [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit).
- **📝 Formatting Issues (5):** [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit), [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4), [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4), [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16), +1 more.
- **Low-utility outputs (11):** [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct), [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit), [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16), [`mlx-community/LFM2-VL-1.6B-8bit`](model_gallery.md#model-mlx-community-lfm2-vl-16b-8bit), +7 more. Common weakness: Output lacks detail.

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
> \- Description hint: , Howardsgate, Welwyn Garden City, England, United Kingdom, UK On a
> clear late afternoon in early spring, a striking living wall brings a splash of nature
> to the urban landscape of Howardsgate in Welwyn Garden City, England. The low sun casts
> long shadows from the bare-branched trees onto the brick facade of a Sainsbury's
> supermarket, where shoppers are seen making their way home with their groceries. This
> modern...
> \- Keyword hints: Adobe Stock, Any Vision, Eco-friendly facade, England, Europe,
> Hertfordshire, Hertfordshire / England, Howardsgate, Locations, Modern retail building,
> Sainsbury's Supermarket, Sainsbury's Welwyn Garden City, Shoppers, Shopping Bags,
> Spring, Supermarket, Sustainable architecture, UK, United Kingdom, Urban Greening
> \- Capture metadata: Taken on 2026-03-14 16:21:00 GMT (at 16:21:00 local time). GPS:
> 51.800333°N, 0.207617°W.
<!-- markdownlint-enable MD028 MD049 -->

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 1648.33s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/nanoLLaVA-1.5-4bit`                      |  151645 |               609 |                    91 |            700 |         2843 |       320 |         2.9 |            1.01s |      0.53s |       1.86s | missing-sections(keywords), ...   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               867 |                   160 |          1,027 |         3025 |       297 |         2.9 |            1.35s |      0.71s |       2.41s | title-length(12), ...             |                 |
| `qnguyen3/nanoLLaVA`                                    |  151645 |               609 |                   103 |            712 |         2661 |       100 |           5 |            1.76s |      0.78s |       2.86s | missing-sections(keywords), ...   |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |  151645 |               613 |                   198 |            811 |         3633 |       305 |         2.2 |            1.77s |      0.78s |       2.89s | missing-sections(keywords), ...   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               867 |                   200 |          1,067 |         3014 |       182 |           4 |            1.84s |      0.86s |       3.02s | description-sentences(5)          |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   49154 |             1,822 |                    18 |          1,840 |         1424 |       115 |         5.6 |            1.86s |      0.55s |       2.71s | missing-sections(title+descrip... |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   49154 |             1,822 |                    18 |          1,840 |         1408 |       116 |         5.6 |            1.89s |      1.05s |       3.28s | missing-sections(title+descrip... |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |   49279 |             1,722 |                    42 |          1,764 |         1390 |       115 |         5.5 |            2.04s |      1.01s |       3.37s | title-length(3), ...              |                 |
| `prince-canuma/Florence-2-large-ft`                     |       0 |             1,181 |                   500 |          1,681 |         4893 |       321 |         5.1 |            2.19s |      1.09s |       3.60s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,627 |                    14 |          1,641 |          503 |      28.5 |          12 |            4.13s |      3.29s |       7.74s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,205 |                   132 |          3,337 |         1023 |       162 |         7.8 |            4.36s |      1.39s |       6.08s | description-sentences(3)          |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |   32007 |             1,444 |                   186 |          1,630 |         1305 |      50.4 |         9.6 |            5.25s |      1.54s |       7.23s | description-sentences(4), ...     |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |  236825 |               863 |                   500 |          1,363 |         1437 |       112 |         6.2 |            5.47s |      3.05s |       8.87s | repetitive(16:21:00), ...         |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  151645 |             2,392 |                    73 |          2,465 |          844 |      32.4 |          17 |            5.65s |      2.83s |       8.80s |                                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   13900 |             1,597 |                   500 |          2,097 |          884 |       109 |          18 |            7.05s |      1.46s |       8.86s | description-sentences(5), ...     |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     645 |             1,597 |                   500 |          2,097 |          975 |       101 |          22 |            7.22s |      3.46s |      11.03s | lang_mixing, hallucination, ...   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               872 |                    94 |            966 |          192 |      27.9 |          19 |            8.34s |      4.67s |      13.35s | context-ignored                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               871 |                   290 |          1,161 |          915 |      41.2 |          17 |            8.40s |      4.56s |      13.38s | missing-sections(title+descrip... |                 |
| `mlx-community/InternVL3-14B-8bit`                      |  151645 |             2,392 |                    88 |          2,480 |          414 |        29 |          17 |            9.39s |      2.94s |      12.66s | title-length(3)                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |  128001 |             2,891 |                   153 |          3,044 |          757 |      29.8 |          18 |            9.40s |      3.33s |      13.09s | title-length(11), ...             |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |   29044 |             1,597 |                   500 |          2,097 |          844 |      68.5 |          37 |            9.92s |      5.61s |      15.85s | description-sentences(6), ...     |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,396 |                   115 |          3,511 |          539 |      34.3 |          16 |           10.11s |      3.11s |      13.57s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,206 |                   128 |          3,334 |          394 |      58.2 |          13 |           10.74s |      2.24s |      13.32s | description-sentences(3), ...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,206 |                   116 |          3,322 |          383 |      55.6 |          13 |           10.85s |      2.20s |      13.38s |                                   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |      13 |             1,444 |                   500 |          1,944 |         1289 |      51.9 |         9.6 |           11.20s |      0.52s |      12.05s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,396 |                   107 |          3,503 |          558 |      19.5 |          28 |           11.97s |      4.60s |      16.90s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               872 |                   105 |            977 |          188 |      14.6 |          34 |           12.24s |      6.74s |      19.31s | context-ignored                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,829 |                   105 |          2,934 |          241 |      43.6 |          11 |           14.55s |      1.52s |      16.43s | missing-sections(title+descrip... |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,627 |                    58 |          1,685 |          478 |      4.78 |          27 |           15.94s |      5.07s |      21.33s | missing-sections(title+descrip... |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,710 |                   102 |          2,812 |          218 |      25.7 |          22 |           16.83s |      3.59s |      20.75s | ⚠️harness(encoding), ...          |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |   53958 |             1,941 |                   500 |          2,441 |          306 |        43 |          60 |           18.80s |     11.81s |      30.95s | missing-sections(title+descrip... |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   19710 |             3,487 |                   500 |          3,987 |          431 |      34.4 |          15 |           23.02s |      2.84s |      26.19s | missing-sections(title+descrip... |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |  128009 |               580 |                   153 |            733 |          136 |      8.05 |          15 |           23.73s |      2.57s |      26.62s | missing-sections(title+descrip... |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |   19241 |             6,723 |                   500 |          7,223 |          277 |      59.5 |         8.4 |           33.14s |      2.18s |      35.65s | missing-sections(title+descrip... |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |    8988 |             6,723 |                   500 |          7,223 |          297 |      43.9 |          11 |           34.45s |      2.38s |      37.15s | missing-sections(title+descrip... |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |  151643 |             1,788 |                   125 |          1,913 |           51 |        27 |          48 |           40.54s |      2.93s |      43.80s | missing-sections(title+descrip... |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |  151643 |             1,788 |                   312 |          2,100 |         52.1 |      42.9 |          41 |           42.41s |      1.89s |      44.63s | refusal(insufficient_info)        |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |  128009 |               581 |                   145 |            726 |          156 |      3.74 |          25 |           42.92s |      1.54s |      44.81s | missing-sections(title+descrip... |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |  151645 |            16,837 |                   150 |         16,987 |          377 |      47.8 |          13 |           48.35s |      2.20s |      50.88s | title-length(4), ...              |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |     722 |            16,826 |                   500 |         17,326 |          385 |      73.8 |         8.3 |           50.97s |      0.82s |      52.09s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     330 |            16,828 |                   500 |         17,328 |          334 |      70.7 |         8.3 |           57.94s |      1.50s |      59.77s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |      11 |             6,723 |                   500 |          7,223 |          159 |      30.7 |          78 |           59.21s |     12.37s |      71.91s | missing-sections(title+descrip... |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |    1297 |             4,699 |                   500 |          5,199 |          269 |      10.6 |         4.5 |           65.23s |      7.09s |      74.12s | repetitive(phrase: "- output o... |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |  235265 |             1,627 |                   500 |          2,127 |         1557 |      7.45 |          11 |           68.87s |      2.97s |      72.23s | missing-sections(title+descrip... |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |  151645 |            16,837 |                   179 |         17,016 |          194 |       176 |         5.1 |           88.27s |      1.06s |      89.67s | description-sentences(3), ...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |     198 |            16,850 |                   500 |         17,350 |          207 |      68.4 |          33 |           89.18s |      5.61s |      95.12s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |   19003 |            16,850 |                   500 |         17,350 |          195 |      53.7 |          74 |           96.47s |     15.33s |     112.25s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |   21221 |            16,850 |                   500 |         17,350 |          110 |      22.8 |          22 |          176.17s |      3.64s |     180.17s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |     279 |            16,850 |                   500 |         17,350 |          112 |      13.3 |          35 |          188.26s |      5.54s |     194.24s | refusal(explicit_refusal), ...    |                 |

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

_Report generated on: 2026-03-15 02:26:33 GMT_
