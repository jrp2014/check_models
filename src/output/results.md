# Model Performance Results

_Generated on 2026-03-15 02:55:35 GMT_

## 🎯 Action Snapshot

- **Framework/runtime failures:** 1 (top owners: transformers=1).
- **Next action:** review failure ownership below and use diagnostics.md for filing.
- **Maintainer signals:** harness-risk successes=5, clean outputs=4/49.
- **Useful now:** 9 clean A/B model(s) worth first review.
- **Review watchlist:** 40 model(s) with breaking or lower-value output.
- **Vs existing metadata:** better=26, neutral=6, worse=17 (baseline B 66/100).
- **Quality signal frequency:** missing_sections=28, context_ignored=18, description_length=15, reasoning_leak=12, keyword_count=10, title_length=8.
- **Runtime pattern:** decode dominates measured phase time (79%; 49/50 measured model(s)).
- **Phase totals:** model load=330.67s, prompt prep=0.13s, decode=1253.70s, cleanup=4.33s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=49, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `prince-canuma/Florence-2-large-ft` (317.7 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.66s)
- **📊 Average TPS:** 78.6 across 49 models

## 📈 Resource Usage

- **Total peak memory:** 943.2 GB
- **Average peak memory:** 19.2 GB
- **Memory efficiency:** 253 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 14 | ✅ B: 18 | 🟡 C: 6 | 🟠 D: 1 | ❌ F: 10

**Average Utility Score:** 62/100

**Existing Metadata Baseline:** ✅ B (66/100)
**Vs Existing Metadata:** Avg Δ -4 | Better: 26, Neutral: 6, Worse: 17

- **Best for cataloging:** `mlx-community/Qwen3.5-35B-A3B-bf16` (🏆 A, 95/100)
- **Worst for cataloging:** `prince-canuma/Florence-2-large-ft` (❌ F, 0/100)

### ⚠️ 11 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (1/100) - Output lacks detail
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (29/100) - Mostly echoes context without adding value
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (33/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (23/100) - Mostly echoes context without adding value
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (1/100) - Output lacks detail
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (20/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (16/100) - Output lacks detail
- `prince-canuma/Florence-2-large-ft`: ❌ F (0/100) - Output too short to be useful
- `qnguyen3/nanoLLaVA`: ❌ F (19/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (2):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `16:21:00`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
- **👻 Hallucinations (2):**
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit`
  - `mlx-community/Qwen3.5-35B-A3B-6bit`
- **📝 Formatting Issues (5):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `prince-canuma/Florence-2-large-ft`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 78.6 | Min: 3.88 | Max: 318
- **Peak Memory**: Avg: 19 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 29.07s | Min: 2.01s | Max: 174.42s
- **Generation Time**: Avg: 25.58s | Min: 1.03s | Max: 168.62s
- **Model Load Time**: Avg: 3.17s | Min: 0.66s | Max: 12.91s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (79%; 49/50 measured model(s)).
- **Phase totals:** model load=330.67s, prompt prep=0.13s, decode=1253.70s, cleanup=4.33s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=49, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 15.71s total (avg 0.31s across 50 model(s)).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- **Best cataloging quality:** [`mlx-community/Qwen3.5-35B-A3B-bf16`](model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16) (A 95/100 | Gen 54.9 TPS | Peak 76 | A 95/100 | ⚠️HARNESS:long_context; Excessive bullet points (33); ...)
- **Fastest generation:** [`prince-canuma/Florence-2-large-ft`](model_gallery.md#model-prince-canuma-florence-2-large-ft) (F 0/100 | Gen 318 TPS | Peak 5.1 | F 0/100 | ⚠️HARNESS:stop_token; Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Sainsbury, Adobe Stock, Any Vision, Eco, Europe, Hertfordshire, Locations, Modern, Supermarket, Shoppers, Shopping Bags, Spring, Sustainable, Urban Greening
Capture, GMT, GPS); ...)
- **Lowest memory footprint:** [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16) (B 74/100 | Gen 301 TPS | Peak 2.2 | B 74/100 | Missing sections (keywords); Description sentence violation (7; expected 1-2))
- **Best balance:** [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4) (B 69/100 | Gen 55.0 TPS | Peak 13 | B 69/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery when available.

- **❌ Failed Models (1):** [`microsoft/Florence-2-large-ft`](model_gallery.md#model-microsoft-florence-2-large-ft). Example: `Model Error`.
- **🔄 Repetitive Output (2):** [`mlx-community/gemma-3n-E2B-4bit`](model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit), [`mlx-community/paligemma2-3b-pt-896-4bit`](model_gallery.md#model-mlx-community-paligemma2-3b-pt-896-4bit). Example: token: `16:21:00`.
- **👻 Hallucinations (2):** [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit), [`mlx-community/Qwen3.5-35B-A3B-6bit`](model_gallery.md#model-mlx-community-qwen35-35b-a3b-6bit).
- **📝 Formatting Issues (5):** [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit), [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4), [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4), [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16), +1 more.
- **Low-utility outputs (11):** [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct), [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit), [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16), [`mlx-community/LFM2-VL-1.6B-8bit`](model_gallery.md#model-mlx-community-lfm2-vl-16b-8bit), +7 more. Common weakness: Output lacks detail.

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `transformers` | 1 | Model Error | `microsoft/Florence-2-large-ft` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### transformers

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

**Overall runtime:** 1605.53s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |            0.34s |    175.19s |     175.85s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |  151645 |               609 |                    91 |            700 |         2487 |       311 |         2.9 |            1.03s |      0.66s |       2.01s | missing-sections(keywords), ...    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               867 |                   160 |          1,027 |         2973 |       292 |         2.9 |            1.34s |      0.71s |       2.37s | title-length(12), ...              |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |  151645 |               613 |                   198 |            811 |         3766 |       301 |         2.2 |            1.72s |      0.78s |       2.83s | missing-sections(keywords), ...    |                 |
| `qnguyen3/nanoLLaVA`                                    |  151645 |               609 |                   103 |            712 |         2626 |      99.6 |           5 |            1.77s |      0.77s |       2.85s | missing-sections(keywords), ...    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               867 |                   200 |          1,067 |         3069 |       180 |           4 |            1.84s |      0.84s |       3.00s | description-sentences(5)           |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   49154 |             1,822 |                    18 |          1,840 |         1414 |       115 |         5.6 |            1.86s |      1.11s |       3.27s | missing-sections(title+descrip...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   49154 |             1,822 |                    18 |          1,840 |         1412 |       115 |         5.6 |            1.88s |      1.05s |       3.26s | missing-sections(title+descrip...  |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |   49279 |             1,722 |                    42 |          1,764 |         1379 |       113 |         5.5 |            2.06s |      1.13s |       3.51s | title-length(3), ...               |                 |
| `prince-canuma/Florence-2-large-ft`                     |       0 |             1,181 |                   500 |          1,681 |         4827 |       318 |         5.1 |            2.22s |      1.04s |       3.57s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,627 |                    14 |          1,641 |          479 |      27.8 |          12 |            4.29s |      3.40s |       8.01s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,205 |                   132 |          3,337 |          985 |       158 |         7.8 |            4.48s |      1.40s |       6.20s | description-sentences(3)           |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |   32007 |             1,444 |                   186 |          1,630 |         1307 |      52.5 |         9.6 |            5.08s |      1.49s |       6.89s | description-sentences(4), ...      |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |  236825 |               863 |                   500 |          1,363 |         1422 |       110 |         6.2 |            5.57s |      3.05s |       8.95s | repetitive(16:21:00), ...          |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  151645 |             2,392 |                    73 |          2,465 |          791 |      32.1 |          17 |            5.86s |      2.80s |       8.98s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   13900 |             1,597 |                   500 |          2,097 |          923 |       112 |          18 |            6.81s |      3.05s |      10.18s | description-sentences(5), ...      |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     645 |             1,597 |                   500 |          2,097 |          972 |       100 |          22 |            7.26s |      3.54s |      11.12s | lang_mixing, hallucination, ...    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               871 |                   290 |          1,161 |          908 |      41.6 |          17 |            8.34s |      4.66s |      13.39s | missing-sections(title+descrip...  |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               872 |                    94 |            966 |          187 |      27.9 |          19 |            8.42s |      4.68s |      13.43s | context-ignored                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |  128001 |             2,891 |                   153 |          3,044 |          709 |      29.4 |          19 |            9.71s |      3.36s |      13.40s | title-length(11), ...              |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |   29044 |             1,597 |                   500 |          2,097 |          968 |      67.6 |          37 |            9.75s |      5.61s |      15.68s | description-sentences(6), ...      |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,396 |                   115 |          3,511 |          518 |      34.9 |          16 |           10.28s |      2.90s |      13.52s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |  151645 |             2,392 |                   104 |          2,496 |          392 |      28.7 |          18 |           10.29s |      2.93s |      13.55s | title-length(3)                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,206 |                   112 |          3,318 |          391 |        55 |          13 |           10.64s |      2.22s |      13.18s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,206 |                   130 |          3,336 |          393 |      57.5 |          13 |           10.81s |      2.19s |      13.32s | description-sentences(3), ...      |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |      13 |             1,444 |                   500 |          1,944 |         1307 |      52.1 |         9.6 |           11.11s |      1.52s |      12.91s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               872 |                   105 |            977 |          182 |      15.4 |          34 |           12.02s |      6.80s |      19.14s | context-ignored                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,396 |                   107 |          3,503 |          536 |      19.5 |          28 |           12.23s |      4.56s |      17.10s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,829 |                   105 |          2,934 |          287 |      43.1 |          11 |           12.70s |      1.66s |      14.69s | missing-sections(title+descrip...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |    1297 |             4,699 |                   500 |          5,199 |         1635 |      42.8 |         4.5 |           15.10s |      2.56s |      18.00s | repetitive(phrase: "- output o...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |   53958 |             1,941 |                   500 |          2,441 |          610 |      43.6 |          60 |           15.46s |      8.95s |      24.75s | missing-sections(title+descrip...  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,627 |                    58 |          1,685 |          482 |      4.83 |          27 |           15.78s |      5.01s |      21.12s | missing-sections(title+descrip...  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,710 |                   102 |          2,812 |          216 |      28.2 |          22 |           16.53s |      3.56s |      20.42s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |  128009 |               580 |                   153 |            733 |          169 |      9.48 |          15 |           20.00s |      2.55s |      22.87s | missing-sections(title+descrip...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |    1729 |             3,487 |                   500 |          3,987 |          409 |      36.4 |          15 |           22.65s |      2.73s |      25.70s | degeneration, ...                  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |   12877 |             6,723 |                   500 |          7,223 |          319 |      61.5 |         8.4 |           29.62s |      2.09s |      32.03s | missing-sections(title+descrip...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |  235265 |             1,627 |                   500 |          2,127 |         1550 |      16.7 |          11 |           31.37s |      3.08s |      34.78s | missing-sections(title+descrip...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |    9128 |             6,723 |                   500 |          7,223 |          304 |      45.6 |          11 |           33.56s |      2.39s |      36.28s | missing-sections(title+descrip...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |  128009 |               581 |                   145 |            726 |          161 |      3.88 |          25 |           41.40s |      3.93s |      45.65s | missing-sections(title+descrip...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |  151643 |             1,788 |                   169 |          1,957 |         51.1 |      27.9 |          48 |           41.85s |      3.00s |      45.17s | refusal(insufficient_info), ...    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |  151645 |            16,837 |                   145 |         16,982 |          436 |      47.6 |          13 |           42.15s |      2.17s |      44.64s | title-length(4), ...               |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |  151643 |             1,788 |                   317 |          2,105 |         52.2 |      44.5 |          41 |           42.20s |      1.88s |      44.40s | refusal(insufficient_info)         |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  151645 |            16,826 |                   146 |         16,972 |          392 |      75.4 |         8.3 |           45.39s |      1.38s |      47.08s | description-sentences(3)           |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |      13 |             6,723 |                   500 |          7,223 |          203 |      31.7 |          78 |           49.36s |     10.17s |      59.86s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      11 |            16,828 |                   500 |         17,328 |          350 |      75.4 |         8.3 |           55.28s |      1.47s |      57.07s | missing-sections(title), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |     328 |            16,850 |                   500 |         17,350 |          257 |      74.4 |          35 |           72.70s |      5.64s |      78.66s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |     326 |            16,850 |                   500 |         17,350 |          253 |      54.9 |          76 |           76.25s |     12.91s |      89.49s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |  151645 |            16,837 |                   147 |         16,984 |          203 |       176 |         5.1 |           84.10s |      1.00s |      85.42s | fabrication, ...                   |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |   18448 |            16,850 |                   500 |         17,350 |          127 |      25.2 |          26 |          152.60s |      3.63s |     156.56s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |    3158 |            16,850 |                   500 |         17,350 |          125 |      15.1 |          39 |          168.62s |      5.47s |     174.42s | missing-sections(title+descrip...  |                 |

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

_Report generated on: 2026-03-15 02:55:35 GMT_
