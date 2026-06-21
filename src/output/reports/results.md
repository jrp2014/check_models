# Model Performance Results

_Generated on 2026-06-21 01:13:24 BST_

## Run Contract

- Mode: stress
- Semantic rankings: grounded (trusted image metadata)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx=1, mlx-vlm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=8, mechanically clean
  outputs=4/58.
- _Useful now:_ 9 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 49 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=35, neutral=3, worse=20 (baseline B 69/100).
- _Quality signal frequency:_ metadata_borrowing=33, missing_sections=24,
  keyword_count=19, description_length=15, title_length=13, cutoff=13.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (61%; 21/60 measured model(s)).
- _Phase totals:_ model load=139.64s, local prompt prep=0.21s, upstream
  prefill / first-token=932.12s, post-prefill decode=441.23s, cleanup=9.77s.
- _Generation total:_ 1373.35s across 58 model(s); upstream prefill /
  first-token split available for 58/58 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=42, exception=2, max_tokens=16.
- _Validation overhead:_ 30.10s total (avg 0.50s across 60 model(s)).
- _Upstream prefill / first-token latency:_ Avg 16.07s | Min 0.04s | Max
  107.06s across 58 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (507.3 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.3 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.34s)
- **📊 Average TPS:** 81.9 across 58 models

## 📈 Resource Usage

- **Input image size:** 66.45 MP
- **Average peak delta from post-load:** 5.63 GB
- **Peak memory delta / MP:** 87 MB/MP
- **Average peak memory:** 20.4 GB
- **Memory efficiency:** 256 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 26 | ✅ B: 13 | 🟡 C: 8 | 🟠 D: 4 | ❌ F: 7

**Average Utility Score:** 69/100

**Existing Metadata Baseline:** ✅ B (69/100)
**Vs Existing Metadata:** Avg Δ -0 | Better: 35, Neutral: 3, Worse: 20

- **Best for cataloging:** `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` (🏆 A, 94/100)
- **Best descriptions:** `mlx-community/Qwen3.6-27B-mxfp8` (100/100)
- **Best keywording:** `mlx-community/GLM-4.6V-Flash-6bit` (90/100)
- **Worst for cataloging:** `mlx-community/llava-v1.6-mistral-7b-8bit` (❌ F, 0/100)

### ⚠️ 11 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (9/100) - Output lacks detail
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-8bit`: 🟠 D (37/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (9/100) - Output lacks detail
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: 🟠 D (49/100) - Lacks visual description of image
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (30/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (36/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (19/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) - Output lacks detail

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Model Error`)
- **🔄 Repetitive Output (8):**
  - `mlx-community/LFM2-VL-1.6B-8bit` (token: `phrase: "church, path, street, fence,..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "100m from colchester town..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "tower entrance visible. shadow..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "structure, architectural detai..."`)
  - `mlx-community/X-Reasoner-7B-8bit` (token: `phrase: "stepped stairway, stepped stai..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `20:51:28:`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "frame. the clock is..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
- **👻 Hallucinations (2):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-nvfp4`
- **📝 Formatting Issues (2):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/MiniCPM-V-4.6-8bit`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 81.9 | Min: 4.65 | Max: 507
- **Peak Memory**: Avg: 20 | Min: 1.3 | Max: 78
- **Total Time**: Avg: 26.60s | Min: 1.43s | Max: 121.71s
- **Generation Time**: Avg: 23.68s | Min: 0.73s | Max: 117.55s
- **Model Load Time**: Avg: 2.40s | Min: 0.34s | Max: 13.55s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.5-35B-A3B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16)
  (Utility A 93/100 | Description 95 | Keywords 74 | Speed 54.1 TPS | Memory
  76 | Caveat nontext prompt burden=97%; missing terms: Colchester, Entrance,
  Essex, High Street, Path)
- _Best descriptions:_ [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-4bit)
  (Utility A 88/100 | Description 100 | Keywords 85 | Speed 98.3 TPS | Memory
  26 | Caveat nontext prompt burden=97%; missing terms: Colchester, Entrance,
  Essex, High Street, Path; keywords=19)
- _Best keywording:_ [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-4bit)
  (Utility A 88/100 | Description 100 | Keywords 85 | Speed 98.3 TPS | Memory
  26 | Caveat nontext prompt burden=97%; missing terms: Colchester, Entrance,
  Essex, High Street, Path; keywords=19)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility C 63/100 | Description 65 | Keywords 56 | Speed 507 TPS | Memory
  1.3 | Caveat missing terms: Essex, High Street, Path, Shadows, Tranquil;
  nonvisual metadata reused)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility C 63/100 | Description 65 | Keywords 56 | Speed 507 TPS | Memory
  1.3 | Caveat missing terms: Essex, High Street, Path, Shadows, Tranquil;
  nonvisual metadata reused)
- _Best balance:_ [`mlx-community/Qwen3.5-35B-A3B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16)
  (Utility A 93/100 | Description 95 | Keywords 74 | Speed 54.1 TPS | Memory
  76 | Caveat nontext prompt burden=97%; missing terms: Colchester, Entrance,
  Essex, High Street, Path)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (2):_ [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Weight Mismatch`.
- _🔄 Repetitive Output (8):_ [`mlx-community/LFM2-VL-1.6B-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm2-vl-16b-8bit),
  [`mlx-community/Molmo-7B-D-0924-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmo-7b-d-0924-8bit),
  [`mlx-community/Molmo-7B-D-0924-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmo-7b-d-0924-bf16),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  +4 more. Example: token: `phrase: "church, path, street, fence,..."`.
- _👻 Hallucinations (2):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4).
- _📝 Formatting Issues (2):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-minicpm-v-46-8bit).
- _Low-utility outputs (11):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-llama-32-11b-vision-instruct-8bit),
  [`mlx-community/Molmo-7B-D-0924-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmo-7b-d-0924-8bit),
  +7 more. Common weakness: Output lacks detail.

## 🚨 Failures by Package (Actionable)

| Package   |   Failures | Error Types     | Affected Models                     |
|-----------|------------|-----------------|-------------------------------------|
| `mlx`     |          1 | Weight Mismatch | `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `mlx-vlm` |          1 | Model Error     | `mlx-community/MolmoPoint-8B-fp16`  |

### Actionable Items by Package

#### mlx

- mlx-community/LFM2.5-VL-1.6B-bf16 (Weight Mismatch)
  - Error: `Model loading failed: Missing 2 parameters: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm....`
  - Type: `ValueError`

#### mlx-vlm

- mlx-community/MolmoPoint-8B-fp16 (Model Error)
  - Error: `Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter`
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
> Context: Existing metadata hints (high confidence; use only when visually
> &#8203;confirmed):
> &#45; Description hint: The exterior of the tower of St Peters Church in
> Colchester Essex
> &#45; Keyword hints: Adobe Stock, Any Vision, Church, Clouds, Colchester,
> England, Entrance, Essex, Europe, Fence, High Street, Path, Rural, Shadows,
> Sky, Stone, Tower, Tranquil, Tree, Trees
> &#45; Capture metadata: Taken on 2026-06-20 20:51:28 BST (at 20:51:28 local
> time). GPS: 51.854295°N, 0.958480°E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1554.92s

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens | Finish Reason   |   Diffusion Canvas Tokens | Diffusion Denoising Steps   |   Diffusion Work Tokens |   Diffusion Canvas Tps |   Diffusion Work Tps | Is Draft   | Draft Text   | Text Already Printed   | Diffusion Step   | Diffusion Total Steps   | Diffusion Canvas Index   | Diffusion Block Complete   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|:----------------|--------------------------:|:----------------------------|------------------------:|-----------------------:|---------------------:|:-----------|:-------------|:-----------------------|:-----------------|:------------------------|:-------------------------|:---------------------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |      0.12s |       0.50s |                                    |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |      0.16s |       0.63s |                                    |         mlx-vlm |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               575 |                   117 |            692 |        14663 |       507 |         1.3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.73s |      0.34s |       1.43s | description-sentences(5), ...      |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               508 |                    37 |            545 |         4749 |       349 |         2.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.83s |      0.44s |       1.65s | missing-sections(keywords), ...    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,721 |                    12 |          1,733 |         3952 |       125 |         5.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.28s |      0.69s |       2.34s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |             1,123 |                    89 |          1,212 |         3942 |       274 |         4.1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.31s |      0.85s |       2.56s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,721 |                    12 |          1,733 |         3948 |       119 |         5.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.39s |      0.63s |       2.67s | ⚠️harness(prompt_template), ...    |                 |
| `qnguyen3/nanoLLaVA`                                    |               508 |                    61 |            569 |         3920 |       107 |         4.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.39s |      0.58s |       2.66s | missing-sections(keywords), ...    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               512 |                   163 |            675 |         5301 |       348 |         2.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.82s |      0.57s |       2.79s | missing-sections(keywords), ...    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               621 |                    95 |            716 |         1763 |       131 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.93s |      0.58s |       2.92s | keyword-count(20), ...             |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               765 |                   500 |          1,265 |         7212 |       332 |           3 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.09s |      0.55s |       3.10s | repetitive(phrase: "church,...     |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               781 |                    88 |            869 |         1380 |      97.7 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.14s |      2.42s |       5.11s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,120 |                   121 |          3,241 |         2655 |       174 |         7.8 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.54s |      0.90s |       3.83s | description-sentences(3)           |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,532 |                    29 |          1,561 |         1380 |      31.5 |          12 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.67s |      1.56s |       4.62s | missing-sections(title+desc...     |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               781 |                    94 |            875 |         1941 |      49.2 |          30 |               0 | stop            |                       256 | 12                          |                    3072 |                    134 |                1,608 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.30s |      4.08s |       7.92s | metadata-borrowing                 |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               781 |                    95 |            876 |         1884 |      45.2 |          31 |               0 | stop            |                       256 | 10                          |                    2560 |                    122 |                1,218 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.40s |      3.49s |       7.77s | metadata-borrowing                 |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,730 |                     2 |          2,732 |         1020 |       118 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.49s |      0.91s |       4.79s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,291 |                    69 |          2,360 |         1935 |        33 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.99s |      1.61s |       6.33s | title-length(4)                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,328 |                   177 |          1,505 |         3769 |        56 |         9.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.06s |      0.84s |       5.27s | fabrication, ...                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,121 |                   100 |          3,221 |         1282 |        66 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.61s |      1.27s |       6.27s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,121 |                   110 |          3,231 |         1134 |      62.5 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            5.18s |      1.34s |       6.89s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               767 |                   500 |          1,267 |         2371 |       115 |           6 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            5.25s |      1.54s |       7.51s | repetitive(20:51:28:), ...         |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,314 |                    89 |          3,403 |         1266 |      31.9 |          16 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.10s |      1.77s |       8.53s | title-length(4), ...               |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               781 |                    91 |            872 |          520 |      22.2 |          20 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.21s |      2.51s |       9.12s | metadata-borrowing                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               776 |                    95 |            871 |          505 |      19.6 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.98s |      2.41s |       9.81s | metadata-borrowing                 |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,532 |                    29 |          1,561 |         1057 |      5.37 |          27 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            7.49s |      2.44s |      10.32s | missing-sections(title+desc...     |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,291 |                    95 |          2,386 |          801 |      23.2 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            7.69s |      1.72s |      10.20s | title-length(3)                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,790 |                   140 |          2,930 |         1496 |      26.3 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            8.01s |      1.93s |      10.62s | description-sentences(6), ...      |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,314 |                    86 |          3,400 |         1166 |      19.2 |          28 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            8.09s |      2.64s |      11.41s | title-length(4), ...               |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               478 |                    99 |            577 |          273 |      16.6 |          15 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            8.28s |      1.61s |      10.27s | missing-sections(title+desc...     |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,622 |                    83 |          2,705 |          622 |      24.1 |          22 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            8.36s |      2.06s |      10.82s | ⚠️harness(encoding), ...           |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,502 |                   500 |          2,002 |         1384 |      70.1 |          18 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            8.94s |      2.14s |      11.49s | title-length(21), ...              |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               776 |                    92 |            868 |          436 |      12.3 |          33 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            9.92s |      3.58s |      14.18s | metadata-borrowing                 |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,328 |                   500 |          1,828 |         3436 |        55 |         9.5 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           10.08s |      0.91s |      11.39s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,502 |                   500 |          2,002 |         1400 |      59.4 |          22 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           10.20s |      2.24s |      13.21s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               775 |                   415 |          1,190 |         1612 |      40.4 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           11.33s |      2.30s |      14.05s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               769 |                    50 |            819 |          110 |       7.7 |          64 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           14.13s |     10.25s |      24.80s | missing-sections(title+desc...     |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,628 |                   120 |          6,748 |          619 |      32.5 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           14.96s |      1.38s |      16.75s | hallucination, ...                 |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |            16,725 |                   112 |         16,837 |         1237 |      92.7 |         8.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           15.60s |      0.66s |      16.66s | title-length(4), ...               |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               479 |                    61 |            540 |          158 |      5.06 |          25 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           15.82s |      3.05s |      19.27s | missing-sections(title+desc...     |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,826 |                   500 |          2,326 |          264 |      60.4 |          60 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           15.93s |     11.72s |      28.05s | missing-sections(title+desc...     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,405 |                   500 |          3,905 |         1374 |      35.8 |          15 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           17.18s |      1.62s |      19.20s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,604 |                   500 |          5,104 |         3016 |      29.7 |         4.6 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           19.69s |      1.43s |      21.86s | repetitive(phrase: "- outpu...     |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |            16,725 |                   112 |         16,837 |          905 |      91.5 |         8.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           20.63s |      0.74s |      21.75s | title-length(4), ...               |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,628 |                   500 |          7,128 |          536 |      48.6 |         8.4 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           23.35s |      1.69s |      25.68s | title-length(4), ...               |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,695 |                   500 |          2,195 |          107 |      46.1 |          41 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           27.82s |      1.19s |      29.39s | repetitive(phrase: "100m fr...     |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,532 |                   500 |          2,032 |         3266 |      18.2 |          11 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           28.69s |      1.46s |      30.55s | repetitive(phrase: "frame....      |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |            16,727 |                   500 |         17,227 |          723 |      73.1 |         8.6 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           30.97s |      0.74s |      32.15s | missing-sections(title+desc...     |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,628 |                    89 |          6,717 |          236 |        25 |          78 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           32.26s |     10.89s |      44.00s | hallucination, ...                 |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,695 |                   500 |          2,195 |         89.8 |        31 |          48 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           36.08s |      1.75s |      38.41s | repetitive(phrase: "tower e...     |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,736 |                   500 |         17,236 |          685 |      41.3 |          13 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           37.47s |      1.12s |      38.98s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,752 |                   116 |         16,868 |          238 |      98.3 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           72.60s |      2.47s |      75.53s | keyword-count(19)                  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,736 |                   500 |         17,236 |          224 |       214 |         5.1 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           78.41s |      0.50s |      79.30s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,752 |                    99 |         16,851 |          218 |      62.5 |          35 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           79.36s |      3.16s |      82.93s | metadata-borrowing, ...            |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,752 |                   118 |         16,870 |          208 |      58.9 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           84.01s |      2.13s |      86.84s | metadata-borrowing, ...            |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,752 |                   108 |         16,860 |          177 |      54.1 |          76 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           97.76s |     13.55s |     111.93s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,752 |                    99 |         16,851 |          171 |      15.3 |          38 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |          105.24s |      3.12s |     108.77s | metadata-borrowing, ...            |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,752 |                   126 |         16,878 |          166 |      26.4 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |          106.69s |      2.33s |     109.80s | title-length(11), ...              |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,502 |                   500 |          2,002 |          879 |      4.65 |          39 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |          110.06s |      3.63s |     114.58s | missing-sections(title), ...       |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,752 |                   123 |         16,875 |          156 |      13.2 |          38 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |          117.55s |      3.33s |     121.71s | metadata-borrowing, ...            |                 |

<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->

_Companion artifacts:_

- _Model-selection shortlist:_ [model_selection.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_selection.md)
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
  (157,739,544 bytes,
  sha256=d7cb97dee8940843711ef5df4ff68b80a6e02f98bbd9e085c6794183655c94d1)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,692,160 bytes,
  sha256=82404f0ce166a876abbb18627634a6fd5361cab3d161c305a7bb36ab54ab086b)

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.32.0.dev20260621+5abdd04b`
- `mlx-vlm`: `0.6.3`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.20.1`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-21 01:13:25 BST_
