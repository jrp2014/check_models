# Model Performance Results

_Generated on 2026-03-14 22:43:52 GMT_

## 🎯 Action Snapshot

- **Framework/runtime failures:** none.
- **Maintainer signals:** harness-risk successes=6, clean outputs=4/49.
- **Useful now:** 3 clean A/B model(s) worth first review.
- **Review watchlist:** 46 model(s) with breaking or lower-value output.
- **Vs existing metadata:** better=22, neutral=2, worse=25 (baseline B 72/100).
- **Quality signal frequency:** missing_sections=28, context_ignored=19, description_length=12, keyword_count=11, context_echo=10, reasoning_leak=10.
- **Runtime pattern:** decode dominates measured phase time (90%; 49/49 measured model(s)).
- **Phase totals:** model load=171.39s, prompt prep=0.13s, decode=1613.59s, cleanup=5.81s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=49.

## 🏆 Performance Highlights

- **Fastest:** `prince-canuma/Florence-2-large-ft` (319.8 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.68s)
- **📊 Average TPS:** 73.6 across 49 models

## 📈 Resource Usage

- **Total peak memory:** 928.5 GB
- **Average peak memory:** 18.9 GB
- **Memory efficiency:** 252 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 20 | ✅ B: 11 | 🟡 C: 1 | 🟠 D: 1 | ❌ F: 16

**Average Utility Score:** 59/100

**Existing Metadata Baseline:** ✅ B (72/100)
**Vs Existing Metadata:** Avg Δ -12 | Better: 22, Neutral: 2, Worse: 25

- **Best for cataloging:** `mlx-community/GLM-4.6V-Flash-6bit` (🏆 A, 96/100)
- **Worst for cataloging:** `mlx-community/Molmo-7B-D-0924-8bit` (❌ F, 0/100)

### ⚠️ 17 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (15/100) - Output lacks detail
- `microsoft/Phi-3.5-vision-instruct`: ❌ F (33/100) - Mostly echoes context without adding value
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (26/100) - Mostly echoes context without adding value
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (29/100) - Mostly echoes context without adding value
- `mlx-community/LFM2.5-VL-1.6B-bf16`: ❌ F (29/100) - Mostly echoes context without adding value
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Phi-3.5-vision-instruct-bf16`: ❌ F (33/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (29/100) - Mostly echoes context without adding value
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (15/100) - Output lacks detail
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/nanoLLaVA-1.5-4bit`: ❌ F (25/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (11/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: 🟠 D (40/100) - Lacks visual description of image
- `prince-canuma/Florence-2-large-ft`: ❌ F (0/100) - Output too short to be useful
- `qnguyen3/nanoLLaVA`: ❌ F (27/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **🔄 Repetitive Output (5):**
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "stonehills, stonehills, stoneh..."`)
  - `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (token: `phrase: "a "gail's" sign on..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "stonehills, stonehills, stoneh..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "awning over it. the..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- do not output..."`)
- **👻 Hallucinations (3):**
  - `mlx-community/Qwen3.5-27B-4bit`
  - `mlx-community/Qwen3.5-35B-A3B-6bit`
  - `mlx-community/Qwen3.5-35B-A3B-bf16`
- **📝 Formatting Issues (5):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `prince-canuma/Florence-2-large-ft`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 73.6 | Min: 0 | Max: 320
- **Peak Memory**: Avg: 19 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 36.82s | Min: 2.03s | Max: 222.47s
- **Generation Time**: Avg: 32.93s | Min: 1.02s | Max: 216.66s
- **Model Load Time**: Avg: 3.50s | Min: 0.68s | Max: 15.05s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 49/49 measured model(s)).
- **Phase totals:** model load=171.39s, prompt prep=0.13s, decode=1613.59s, cleanup=5.81s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=49.

### ⏱ Timing Snapshot

- **Validation overhead:** 18.69s total (avg 0.38s across 49 model(s)).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- **Best cataloging quality:** [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit) (A 96/100 | Gen 38.8 TPS | Peak 11 | A 96/100 | Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Spring, Stonehills
Capture, GMT); Missing sections (title, description, keywords); ...)
- **Fastest generation:** [`prince-canuma/Florence-2-large-ft`](model_gallery.md#model-prince-canuma-florence-2-large-ft) (F 0/100 | Gen 320 TPS | Peak 5.1 | F 0/100 | ⚠️HARNESS:stop_token; Context ignored (missing: Howardsgate, Welwyn Garden City, England, United Kingdom, Here, March, Locals, Georgian, Adobe Stock, Any Vision, Corner Building, English Town Centre, English, Europe, Food Delivery, Food Delivery Scooter, GAIL, Bakery, Hertfordshire, Neo, Georgian Architecture, Outdoor Cafe, Outdoor Cafe Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
Capture, GMT); ...)
- **Lowest memory footprint:** [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16) (F 26/100 | Gen 306 TPS | Peak 2.2 | F 26/100 | Missing sections (keywords); Title length violation (81 words; expected 5-10); ...)
- **Best balance:** [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4) (A 88/100 | Gen 47.7 TPS | Peak 13 | A 88/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery when available.

- **🔄 Repetitive Output (5):** [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct), [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](model_gallery.md#model-mlx-community-apriel-15-15b-thinker-6bit-mlx), [`mlx-community/Phi-3.5-vision-instruct-bf16`](model_gallery.md#model-mlx-community-phi-35-vision-instruct-bf16), [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](model_gallery.md#model-mlx-community-paligemma2-3b-ft-docci-448-bf16), +1 more. Example: token: `phrase: "stonehills, stonehills, stoneh..."`.
- **👻 Hallucinations (3):** [`mlx-community/Qwen3.5-27B-4bit`](model_gallery.md#model-mlx-community-qwen35-27b-4bit), [`mlx-community/Qwen3.5-35B-A3B-6bit`](model_gallery.md#model-mlx-community-qwen35-35b-a3b-6bit), [`mlx-community/Qwen3.5-35B-A3B-bf16`](model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16).
- **📝 Formatting Issues (5):** [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit), [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4), [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4), [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16), +1 more.
- **Low-utility outputs (17):** [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct), [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct), [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit), [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16), +13 more. Common weakness: Output lacks detail.

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
> \- Description hint: , Howardsgate, Welwyn Garden City, England, United Kingdom, UK Here
> is a caption for the image: A bright March afternoon brings a touch of warmth to
> Howardsgate in Welwyn Garden City, England. Locals and visitors take advantage of the
> clear skies, gathering at the outdoor tables of a popular corner bakery. The scene is
> set against the town's signature neo-Georgian red-brick architecture, a hallmark of its
> design a...
> \- Keyword hints: Adobe Stock, Any Vision, Corner Building, England, English Town Centre,
> English town, Europe, Food Delivery, Food Delivery Scooter, GAIL's Bakery,
> Hertfordshire, Howardsgate, Neo-Georgian Architecture, Outdoor Cafe, Outdoor Cafe
> Culture, Outdoor Dining, Sidewalk, Socializing, Spring, Stonehills
> \- Capture metadata: Taken on 2026-03-14 15:19:38 GMT (at 15:19:38 local time).
<!-- markdownlint-enable MD028 MD049 -->

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 1810.71s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/nanoLLaVA-1.5-4bit`                      |  151645 |               582 |                    89 |            671 |         2584 |       311 |         2.8 |            1.02s |      0.68s |       2.03s | missing-sections(keywords), ...   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               846 |                   164 |           1010 |         2863 |       273 |         2.8 |            1.44s |      0.70s |       2.47s | description-sentences(3), ...     |                 |
| `qnguyen3/nanoLLaVA`                                    |  151645 |               582 |                    80 |            662 |         2550 |      99.9 |         5.1 |            1.54s |      0.76s |       2.62s | missing-sections(keywords), ...   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               846 |                   166 |           1012 |         2856 |       170 |           4 |            1.77s |      0.85s |       2.97s | description-sentences(3), ...     |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   49154 |              1796 |                    18 |           1814 |         1400 |       116 |         5.6 |            1.88s |      1.02s |       3.26s | missing-sections(title+descrip... |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   49154 |              1796 |                    18 |           1814 |         1388 |       114 |         5.6 |            1.94s |      1.07s |       3.34s | missing-sections(title+descrip... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |  151645 |               586 |                   232 |            818 |         3540 |       306 |         2.2 |            1.95s |      0.81s |       3.11s | missing-sections(keywords), ...   |                 |
| `prince-canuma/Florence-2-large-ft`                     |       0 |              1163 |                   500 |           1663 |         4735 |       320 |         5.1 |            2.22s |      1.21s |       3.75s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |   49279 |              1696 |                    73 |           1769 |         1364 |       113 |         5.5 |            2.33s |      1.01s |       3.66s | keyword-count(6)                  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |              1603 |                    14 |           1617 |          440 |      27.8 |          12 |            4.56s |      3.52s |       8.43s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |              3193 |                   115 |           3308 |          834 |       158 |         7.8 |            4.97s |      1.40s |       6.74s |                                   |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |  236810 |               836 |                   500 |           1336 |         1389 |       110 |           6 |            5.55s |      3.06s |       8.96s | missing-sections(title+descrip... |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  151645 |              2365 |                    79 |           2444 |          770 |      30.3 |          17 |            6.27s |      2.87s |       9.63s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     327 |              1580 |                   500 |           2080 |          930 |      96.1 |          22 |            7.56s |      3.75s |      11.85s | missing-sections(title), ...      |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   63720 |              1580 |                   500 |           2080 |          650 |       109 |          18 |            7.74s |      3.37s |      11.47s | missing-sections(title), ...      |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               845 |                    93 |            938 |          186 |      23.9 |          19 |            8.89s |      4.90s |      14.13s | context-ignored                   |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |              3387 |                    91 |           3478 |          482 |      35.2 |          16 |           10.05s |      2.86s |      13.30s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |              1603 |                    25 |           1628 |          368 |      4.65 |          27 |           10.16s |      6.93s |      17.42s | missing-sections(title+descrip... |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      11 |              1580 |                   500 |           2080 |          924 |      64.3 |          37 |           10.41s |      6.11s |      17.30s | missing-sections(title), ...      |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               844 |                   396 |           1240 |          884 |      41.8 |          17 |           10.85s |      4.55s |      15.79s | missing-sections(title+descrip... |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |    6090 |              1416 |                   500 |           1916 |         1297 |      52.6 |         9.6 |           11.07s |      1.46s |      12.87s | repetitive(phrase: "stonehills... |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               845 |                    96 |            941 |          186 |      14.2 |          34 |           11.73s |      6.80s |      18.87s | context-ignored                   |                 |
| `mlx-community/pixtral-12b-bf16`                        |       2 |              3387 |                    98 |           3485 |          562 |      17.8 |          28 |           11.94s |      4.56s |      16.83s | context-ignored                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      |  151645 |              2365 |                   128 |           2493 |          339 |      26.7 |          17 |           12.36s |      2.97s |      15.89s | title-length(4), ...              |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |              3194 |                   132 |           3326 |          324 |      50.1 |          13 |           12.91s |      2.21s |      15.46s | description-sentences(3)          |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |  128001 |              2875 |                   241 |           3116 |          734 |      26.8 |          18 |           13.39s |      3.65s |      17.80s | description-sentences(6), ...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |              3194 |                   128 |           3322 |          307 |      47.7 |          13 |           13.52s |      2.24s |      16.12s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |              2819 |                    94 |           2913 |          256 |      42.7 |          11 |           13.65s |      1.52s |      15.50s | missing-sections(title+descrip... |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |    6090 |              1416 |                   500 |           1916 |          865 |      38.7 |         9.6 |           15.04s |      1.51s |      16.98s | repetitive(phrase: "stonehills... |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |    5033 |              4675 |                   500 |           5175 |         1584 |      38.6 |         4.5 |           16.47s |      2.58s |      19.44s | repetitive(phrase: "- do not o... |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |              2695 |                    99 |           2794 |          152 |      19.2 |          22 |           23.36s |      3.84s |      27.65s | ⚠️harness(encoding), ...          |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |   14330 |              1901 |                   500 |           2401 |          232 |      30.1 |          60 |           25.70s |     13.44s |      39.56s | missing-sections(title+descrip... |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |  169448 |              1603 |                   500 |           2103 |         1529 |      15.8 |          11 |           33.06s |      3.15s |      36.55s | repetitive(phrase: "awning ove... |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |    9387 |              3478 |                   500 |           3978 |          256 |      22.7 |          15 |           36.05s |      2.80s |      39.58s | repetitive(phrase: "a "gail's"... |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |         |                 0 |                     0 |              0 |            0 |         0 |          48 |           36.18s |      3.36s |      39.87s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |     330 |              6704 |                   500 |           7204 |          246 |      53.7 |         8.4 |           37.04s |      2.19s |      39.60s | missing-sections(title+descrip... |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |         |                 0 |                     0 |              0 |            0 |         0 |          41 |           37.58s |      1.87s |      39.78s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |   40860 |              6704 |                   500 |           7204 |          237 |      38.8 |          11 |           41.63s |      2.48s |      44.45s | missing-sections(title+descrip... |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |  151645 |             16810 |                   139 |          16949 |          376 |        47 |          13 |           48.15s |      2.15s |      50.63s | title-length(3), ...              |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  151645 |             16799 |                   139 |          16938 |          353 |      74.6 |         8.3 |           50.07s |      1.47s |      51.87s | context-echo(0.32)                |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |      11 |              6704 |                   500 |           7204 |          131 |        27 |          78 |           70.08s |     13.98s |      84.50s | missing-sections(title+descrip... |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |  128009 |               564 |                   173 |            737 |          142 |      2.51 |          25 |           73.53s |      4.69s |      78.56s | missing-sections(title+descrip... |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      11 |             16801 |                   500 |          17301 |          242 |        63 |         8.3 |           77.86s |      1.47s |      79.67s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |   43807 |               563 |                   500 |           1063 |          141 |      6.08 |          15 |           86.75s |      2.71s |      89.81s | missing-sections(title+descrip... |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |   87498 |             16824 |                   500 |          17324 |          208 |      71.2 |          33 |           88.24s |      5.61s |      94.18s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |  151645 |             16810 |                   126 |          16936 |          179 |       166 |         5.1 |           95.11s |      1.01s |      96.44s | description-sentences(3), ...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |   10708 |             16824 |                   500 |          17324 |          194 |      53.7 |          74 |           96.63s |     15.05s |     112.07s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     256 |             16824 |                   500 |          17324 |           93 |      21.5 |          22 |          204.74s |      3.74s |     208.95s | hallucination, ...                |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |      11 |             16824 |                   500 |          17324 |         95.8 |      12.3 |          35 |          216.66s |      5.48s |     222.47s | missing-sections(title+descrip... |                 |

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

_Report generated on: 2026-03-14 22:43:52 GMT_
