# Model Performance Results

_Generated on 2026-06-06 20:51:24 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 1 (top owners: mlx=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=5, mechanically clean
  outputs=10/54.
- _Useful now:_ 19 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 35 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ missing_sections=29, cutoff=19, title_length=18,
  generation_loop=16, keyword_count=13, description_length=12.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (48%; 17/55 measured model(s)).
- _Phase totals:_ model load=124.61s, local prompt prep=0.18s, upstream
  prefill / first-token=664.05s, post-prefill decode=582.51s, cleanup=6.19s.
- _Generation total:_ 1246.56s across 54 model(s); upstream prefill /
  first-token split available for 54/54 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=33, exception=1, max_tokens=21.
- _Validation overhead:_ 6.66s total (avg 0.12s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 12.30s | Min 0.07s | Max
  78.72s across 54 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/FastVLM-0.5B-bf16` (351.7 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.32s)
- **📊 Average TPS:** 80.3 across 54 models

## 📈 Resource Usage

- **Input image size:** 66.45 MP
- **Average peak delta from post-load:** 5.78 GB
- **Peak memory delta / MP:** 89 MB/MP
- **Average peak memory:** 20.8 GB
- **Memory efficiency:** 222 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 29 | ✅ B: 17 | 🟡 C: 4 | ❌ F: 4

**Average Utility Score:** 78/100

- **Best for cataloging:** `mlx-community/InternVL3-14B-8bit` (🏆 A, 100/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (88/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 0/100)

### ⚠️ 4 Models with Low Utility (D/F)

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (19/100) - Output lacks detail
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (16/100) - Output lacks detail

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (`Model Error`)
- **🔄 Repetitive Output (5):**
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (token: `phrase: "water vehicle, water recreatio..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "water sports equipment, water..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "the boy is wearing..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "and 10-18, and 10-18,..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- do not output..."`)
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit`
  - `mlx-community/MiniCPM-V-4.6-8bit`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 80.3 | Min: 4.74 | Max: 352
- **Peak Memory**: Avg: 21 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 25.53s | Min: 1.09s | Max: 112.40s
- **Generation Time**: Avg: 23.08s | Min: 0.58s | Max: 109.19s
- **Model Load Time**: Avg: 2.31s | Min: 0.32s | Max: 23.41s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility A 100/100 | Description 93 | Keywords 88 | Speed 66.2 TPS | Memory
  13 | Caveat nontext prompt burden=88%)
- _Best descriptions:_ [`mlx-community/gemma-4-31b-it-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-4-31b-it-4bit)
  (Utility A 100/100 | Description 100 | Keywords 78 | Speed 26.0 TPS | Memory
  20)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility A 100/100 | Description 93 | Keywords 88 | Speed 66.2 TPS | Memory
  13 | Caveat nontext prompt burden=88%)
- _Fastest generation:_ [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility B 78/100 | Description 83 | Keywords 0 | Speed 352 TPS | Memory 2.2
  | Caveat missing sections: keywords)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility B 78/100 | Description 83 | Keywords 0 | Speed 352 TPS | Memory 2.2
  | Caveat missing sections: keywords)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 100/100 | Description 93 | Keywords 85 | Speed 183 TPS | Memory
  7.8 | Caveat nontext prompt burden=88%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (5):_ [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-llama-32-11b-vision-instruct-8bit),
  [`mlx-community/Molmo-7B-D-0924-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmo-7b-d-0924-8bit),
  [`mlx-community/gemma-3n-E2B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  +1 more. Example: token: `phrase: "water vehicle, water recreatio..."`.
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (4):_ [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`mlx-community/llava-v1.6-mistral-7b-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-llava-v16-mistral-7b-8bit).
  Common weakness: Output too short to be useful.

## 🚨 Failures by Package (Actionable)

| Package   |   Failures | Error Types   | Affected Models                    |
|-----------|------------|---------------|------------------------------------|
| `mlx`     |          1 | Model Error   | `LiquidAI/LFM2.5-VL-450M-MLX-bf16` |

### Actionable Items by Package

#### mlx

- LiquidAI/LFM2.5-VL-450M-MLX-bf16 (Model Error)
  - Error: `Model loading failed: Received 2 parameters not in model: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_project...`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
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
> &#8203;Return exactly these three sections, and nothing else:
>
> &#8203;Title:
> &#45; 5-10 words, concrete and factual, limited to clearly visible content.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> &#8203;Description:
> &#45; 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> &#45; Output only the description text after the label.
>
> &#8203;Keywords:
> &#45; 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> &#45; Output only the keyword list after the label.
>
> &#8203;Rules:
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
> Capture metadata hints: Taken on 2026-05-25 22:50:09 BST (at 22:50:09 local
> time).
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1386.31s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens | Finish Reason   |   Diffusion Canvas Tokens | Diffusion Denoising Steps   |   Diffusion Work Tokens |   Diffusion Canvas Tps |   Diffusion Work Tps | Is Draft   | Draft Text   | Text Already Printed   | Diffusion Step   | Diffusion Total Steps   | Diffusion Canvas Index   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|:----------------|--------------------------:|:----------------------------|------------------------:|-----------------------:|---------------------:|:-----------|:-------------|:-----------------------|:-----------------|:------------------------|:-------------------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                  |      0.03s |       1.01s |                                    |             mlx |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               659 |                    85 |            744 |         8961 |       327 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.58s |      0.37s |       1.09s | title-length(11), ...              |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               398 |                    95 |            493 |         4131 |       350 |         2.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.72s |      0.37s |       1.20s | missing-sections(keywords), ...    |                 |
| `qnguyen3/nanoLLaVA`                                    |               398 |                    45 |            443 |         4243 |       114 |         4.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.86s |      0.41s |       1.39s | missing-sections(keywords)         |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |             1,010 |                    89 |          1,099 |         4190 |       275 |         3.9 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.02s |      0.62s |       1.76s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               507 |                    32 |            539 |         1504 |       121 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.11s |      0.58s |       1.82s | title-length(2), ...               |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               659 |                   165 |            824 |         9017 |       194 |         4.1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.15s |      0.45s |       1.72s | title-length(21), ...              |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,606 |                    38 |          1,644 |         3916 |       127 |         5.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.18s |      0.60s |       1.91s | missing-sections(keywords), ...    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,606 |                    38 |          1,644 |         3764 |       126 |         5.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.20s |      0.32s |       1.64s | missing-sections(keywords), ...    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               402 |                   162 |            564 |         4168 |       352 |         2.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.40s |      0.57s |       2.08s | missing-sections(keywords), ...    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               673 |                    82 |            755 |         1341 |       105 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.66s |      2.52s |       4.35s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,991 |                   102 |          3,093 |         2933 |       183 |         7.8 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.96s |      0.79s |       2.87s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,690 |                    16 |          2,706 |         2319 |      34.4 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.09s |      1.75s |       3.96s | missing-sections(descriptio...     |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,201 |                   128 |          1,329 |         3868 |      57.3 |         9.4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.81s |      0.75s |       3.69s | title-length(4), ...               |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,181 |                    58 |          2,239 |         2648 |        35 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.90s |      1.56s |       4.58s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,588 |                    10 |          2,598 |         1038 |      66.8 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.16s |      0.78s |       4.06s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,992 |                    83 |          3,075 |         1393 |      66.2 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.78s |      1.18s |       5.09s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,992 |                    85 |          3,077 |         1342 |      63.3 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.95s |      1.21s |       5.28s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,182 |                    83 |          3,265 |         1791 |      39.1 |          16 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.29s |      1.56s |       6.03s | title-length(4)                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,181 |                    82 |          2,263 |         1348 |      32.3 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.58s |      1.61s |       6.32s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,424 |                   105 |          1,529 |         1356 |      29.7 |          12 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.03s |      1.60s |       6.75s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               673 |                    88 |            761 |          444 |        26 |          20 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.28s |      2.39s |       7.80s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               659 |                   500 |          1,159 |         2001 |       108 |         6.1 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.28s |      1.50s |       6.94s | repetitive(phrase: "and 10-...     |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,424 |                    87 |          1,511 |         3193 |      17.9 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.67s |      1.47s |       7.31s | missing-sections(title+desc...     |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,400 |                   327 |          1,727 |         1529 |        72 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.84s |      0.98s |       6.93s | missing-sections(title), ...       |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               668 |                    94 |            762 |          371 |      24.1 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            6.04s |     23.41s |      29.61s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               667 |                   237 |            904 |         1488 |      43.4 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            6.25s |      2.31s |       8.69s | description-sentences(7), ...      |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,182 |                    78 |          3,260 |         1539 |      20.3 |          28 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            6.28s |      2.37s |       8.77s | title-length(4)                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,490 |                    80 |          2,570 |          639 |      27.6 |          22 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            7.18s |      1.92s |       9.23s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               668 |                    98 |            766 |          443 |      14.8 |          33 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            8.44s |      3.44s |      12.03s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,400 |                   500 |          1,900 |         1538 |      63.4 |          22 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            9.21s |      2.01s |      11.36s | refusal(explicit_refusal), ...     |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,201 |                   500 |          1,701 |         3805 |      56.3 |         9.4 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            9.47s |      0.78s |      10.37s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,707 |                   500 |          2,207 |         1192 |      61.9 |          60 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           10.09s |      4.58s |      14.94s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,520 |                   500 |          7,020 |          870 |        70 |         8.4 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           14.91s |      1.17s |      16.22s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,496 |                   500 |          4,996 |         3791 |        38 |         4.6 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           14.94s |      1.12s |      16.21s | repetitive(phrase: "- do no...     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,273 |                   500 |          3,773 |         1447 |      35.6 |          15 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           16.69s |      1.46s |      18.27s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,520 |                   500 |          7,020 |          963 |      48.4 |          11 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           17.36s |      1.27s |      18.75s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             3,194 |                    89 |          3,283 |         1271 |      6.02 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           17.90s |      2.04s |      20.06s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,424 |                   100 |          1,524 |         1071 |      5.22 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           20.88s |      2.45s |      23.46s | missing-sections(title+desc...     |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,626 |                   123 |         16,749 |          798 |      51.3 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           23.85s |      1.15s |      25.13s | description-sentences(4)           |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,520 |                   500 |          7,020 |          496 |      35.7 |          78 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           27.41s |      5.33s |      32.87s | generation_loop(degeneration), ... |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               379 |                   500 |            879 |          250 |        19 |          15 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           28.20s |      1.37s |      29.69s | repetitive(phrase: "water s...     |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,577 |                   500 |          2,077 |         87.6 |      52.2 |          41 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           28.35s |      1.08s |      29.55s | repetitive(phrase: "the boy...     |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,577 |                   500 |          2,077 |         84.1 |      30.3 |          48 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           36.03s |      1.60s |      37.75s | missing-sections(title+desc...     |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,626 |                     4 |         16,630 |          278 |       278 |         5.1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           60.48s |      0.39s |      61.00s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,400 |                   281 |          1,681 |          983 |      4.74 |          39 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           61.25s |      3.13s |      64.50s | missing-sections(title), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,639 |                   500 |         17,139 |          283 |       108 |          26 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           64.06s |      2.50s |      66.69s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,639 |                   500 |         17,139 |          284 |        91 |          35 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           64.79s |      3.10s |      68.02s | missing-sections(descriptio...     |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,639 |                   500 |         17,139 |          284 |        87 |          11 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           64.90s |      1.44s |      66.47s | title-length(49), ...              |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,639 |                   500 |         17,139 |          269 |      63.6 |          76 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           70.42s |      7.93s |      78.49s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               661 |                   500 |          1,161 |         81.4 |      7.51 |          64 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           75.06s |      9.92s |      85.12s | keyword-count(105), cutoff         |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,639 |                   500 |         17,139 |          232 |      29.3 |          26 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           89.38s |      2.33s |      91.84s | generation_loop(numeric_loop), ... |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               380 |                   500 |            880 |          202 |         5 |          25 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |          102.20s |      0.89s |     103.22s | repetitive(phrase: "water v...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,639 |                   500 |         17,139 |          213 |      17.3 |          38 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |          107.81s |      3.10s |     111.06s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,639 |                   500 |         17,139 |          211 |      16.8 |          38 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |          109.19s |      3.06s |     112.40s | generation_loop(numeric_loop), ... |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.1
- _SDK Version:_ 26.5
- _Active Developer Directory:_ /Applications/Xcode.app/Contents/Developer
- _Xcode Version:_ 26.5
- _Xcode Build:_ 17F42
- _SDK Path:_ /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
- _Metal SDK:_ MacOSX26.5.sdk
- _Metal Compiler Version:_ Apple metal version 32023.883 (metalfe-32023.883)
- _Metallib Linker Version:_ AIR-LLD 32023.883 (metalfe-32023.883) (compatible
  with legacy metallib linker)
- _Apple Clang Version:_ Apple clang version 21.0.0 (clang-2100.1.1.101)
- _Python Version:_ 3.13.13
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18
- _MLX Install Type:_ editable local source
- _MLX Distribution Root:_ /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages
- _mlx-metal Distribution:_ not installed; local editable mlx supplies backend
- _MLX Core Extension:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so
- _MLX Metallib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib
  (157,751,704 bytes,
  sha256=e50cd0c2d2ae16781f644476459cbc2ca23b0d428a897140740f86678b5e2bf5)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,675,136 bytes,
  sha256=8084f4df1833f78e6598f28ff40818d3c7a73343ced3f3c416b192f2292ad933)

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.32.0.dev20260606+8f0e8b14`
- `mlx-vlm`: `0.6.2`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.18.0`
- `transformers`: `5.10.2`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-06 20:51:24 BST_
