# Model Performance Results

_Generated on 2026-05-10 21:23:22 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 4 (top owners: mlx=2, mlx-vlm=1,
  model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=5, clean outputs=5/51.
- _Useful now:_ 5 clean A/B model(s) worth first review.
- _Review watchlist:_ 46 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=12, neutral=1, worse=38 (baseline B 75/100).
- _Quality signal frequency:_ metadata_borrowing=32, missing_sections=27,
  cutoff=18, keyword_count=13, description_length=12, reasoning_leak=10.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (53%; 15/55 measured model(s)).
- _Phase totals:_ model load=110.49s, local prompt prep=0.16s, upstream
  prefill / first-token=600.84s, post-prefill decode=419.19s, cleanup=5.68s.
- _Generation total:_ 1020.03s across 51 model(s); upstream prefill /
  first-token split available for 51/51 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=32, exception=4, max_tokens=19.
- _Validation overhead:_ 14.04s total (avg 0.26s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 11.78s | Min 0.04s | Max
  71.39s across 51 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (498.4 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.6 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.34s)
- **📊 Average TPS:** 83.7 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1075.7 GB
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 234 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 6 | ✅ B: 21 | 🟡 C: 9 | 🟠 D: 8 | ❌ F: 7

**Average Utility Score:** 59/100

**Existing Metadata Baseline:** ✅ B (75/100)
**Vs Existing Metadata:** Avg Δ -16 | Better: 12, Neutral: 1, Worse: 38

- **Best for cataloging:** `mlx-community/gemma-3-27b-it-qat-4bit` (🏆 A, 88/100)
- **Best descriptions:** `mlx-community/gemma-4-26b-a4b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (92/100)
- **Worst for cataloging:** `mlx-community/gemma-3n-E2B-4bit` (❌ F, 2/100)

### ⚠️ 15 Models with Low Utility (D/F)

- `meta-llama/Llama-3.2-11B-Vision-Instruct`: 🟠 D (47/100) - Keywords are not specific or diverse enough
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/FastVLM-0.5B-bf16`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-Flash-mxfp4`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-8bit`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (2/100) - Output too short to be useful
- `mlx-community/gemma-4-31b-bf16`: 🟠 D (41/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (9/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (33/100) - Lacks visual description of image
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (35/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (28/100) - Output lacks detail
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (48/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (4):**
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "after the label. -..."`)
- **👻 Hallucinations (1):**
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit`
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 83.7 | Min: 4.76 | Max: 498
- **Peak Memory**: Avg: 21 | Min: 1.6 | Max: 78
- **Total Time**: Avg: 22.38s | Min: 1.23s | Max: 110.47s
- **Generation Time**: Avg: 20.00s | Min: 0.65s | Max: 107.01s
- **Model Load Time**: Avg: 2.11s | Min: 0.34s | Max: 12.44s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 88/100 | Description 98 | Keywords 85 | Speed 30.9 TPS | Memory
  19 | Caveat missing terms: Bell Tower, Chapel, Cross, Dorking, Fence)
- _Best descriptions:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit)
  (Utility A 82/100 | Description 100 | Keywords 84 | Speed 17.2 TPS | Memory
  33 | Caveat missing terms: Chapel, Cross, Dorking, Objects, Station wagon)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility A 83/100 | Description 95 | Keywords 92 | Speed 65.4 TPS | Memory
  13 | Caveat nontext prompt burden=85%; missing terms: Bell Tower, Chapel,
  Cross, Dorking, Fence)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility B 72/100 | Description 91 | Keywords 62 | Speed 498 TPS | Memory
  1.6 | Caveat missing terms: Bell Tower, Chapel, Cross, Gothic Architecture,
  Surrey; keywords=29; nonvisual metadata reused)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility B 72/100 | Description 91 | Keywords 62 | Speed 498 TPS | Memory
  1.6 | Caveat missing terms: Bell Tower, Chapel, Cross, Gothic Architecture,
  Surrey; keywords=29; nonvisual metadata reused)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 88/100 | Description 98 | Keywords 85 | Speed 30.9 TPS | Memory
  19 | Caveat missing terms: Bell Tower, Chapel, Cross, Dorking, Fence)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (4):_ [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Model Error`.
- _🔄 Repetitive Output (1):_ [`mlx-community/paligemma2-3b-pt-896-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-paligemma2-3b-pt-896-4bit).
  Example: token: `phrase: "after the label. -..."`.
- _👻 Hallucinations (1):_ [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit).
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (15):_ [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  +11 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types                  | Affected Models                                                                |
|----------------|------------|------------------------------|--------------------------------------------------------------------------------|
| `mlx`          |          2 | Model Error, Weight Mismatch | `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `mlx-vlm`      |          1 | Model Error                  | `facebook/pe-av-large`                                                         |
| `model-config` |          1 | Processor Error              | `mlx-community/MolmoPoint-8B-fp16`                                             |

### Actionable Items by Package

#### mlx

- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
  - Type: `ValueError`
- mlx-community/LFM2.5-VL-1.6B-bf16 (Weight Mismatch)
  - Error: `Model loading failed: Missing 2 parameters: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm....`
  - Type: `ValueError`

#### mlx-vlm

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.p...`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
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
> &#45; Description hint: A low-angle, wide shot of St Peter's Church in
> Petersfield, Hampshire, England, on a sunny day. The Gothic Revival style
> church, with its tall spire and flint walls, is pictured against a bright
> blue sky with wispy clouds. A black car is parked in the foreground.
> &#45; Keyword hints: Adobe Stock, Any Vision, Bell Tower, Blue sky, Car, Chapel,
> Church, Cross, Daylight, Dorking, England, Europe, Fence, Gothic
> Architecture, Objects, Sky, Station wagon, Steeple, Stone, Surrey
> &#45; Capture metadata: Taken on 2026-05-09 17:50:09 BST (at 17:50:09 local
> time). GPS: 51.215500°N, 0.798500°W.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1155.61s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.18s |       1.19s |                                 |         mlx-vlm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.19s |       1.23s |                                 |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                  |      0.14s |       1.18s |                                 |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.17s |       3.57s |                                 |    model-config |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               819 |                   154 |            973 |       19,927 |       498 |         1.6 |            0.65s |      0.34s |       1.23s | description-sentences(3), ...   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               819 |                   136 |            955 |        7,575 |       329 |           3 |            0.84s |      0.45s |       1.53s | description-sentences(3), ...   |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               561 |                   137 |            698 |        5,952 |       350 |         2.7 |            0.89s |      0.51s |       1.66s | fabrication, ...                |                 |
| `qnguyen3/nanoLLaVA`                                    |               561 |                    68 |            629 |        4,931 |       113 |         4.9 |            1.12s |      0.52s |       1.89s | missing-sections(keywords), ... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               565 |                   222 |            787 |        5,762 |       342 |         2.1 |            1.48s |      0.59s |       2.35s | missing-sections(keywords), ... |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               835 |                    86 |            921 |        1,599 |       111 |          17 |            1.73s |      2.54s |       4.55s |                                 |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,585 |                    17 |          1,602 |        3,254 |        20 |          11 |            1.76s |      1.46s |       3.48s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               671 |                   131 |            802 |        1,958 |       132 |         5.5 |            1.86s |      0.55s |       2.67s | description-sentences(3), ...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,585 |                    20 |          1,605 |        1,341 |      31.3 |          12 |            2.26s |      1.69s |       4.22s | missing-sections(title+desc...  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,771 |                   182 |          1,953 |        4,225 |       128 |         5.6 |            2.35s |      0.63s |       3.24s | title-length(11), ...           |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,172 |                   157 |          3,329 |        2,901 |       185 |         7.8 |            2.40s |      0.94s |       3.60s | description-sentences(4), ...   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,771 |                   182 |          1,953 |        4,174 |       124 |         5.7 |            2.41s |      0.59s |       3.26s | title-length(11), ...           |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,789 |                    11 |          2,800 |        1,333 |      67.4 |         9.7 |            2.78s |      0.89s |       3.93s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,387 |                   129 |          1,516 |        3,952 |      56.9 |         9.5 |            2.98s |      0.86s |       4.09s | keyword-count(19), ...          |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,585 |                     9 |          1,594 |        1,087 |      5.84 |          27 |            3.43s |      2.47s |       6.17s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,344 |                    75 |          2,419 |        2,701 |        34 |          18 |            3.52s |      1.69s |       5.47s | title-length(4)                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,173 |                   108 |          3,281 |        1,365 |      65.4 |          13 |            4.42s |      1.32s |       6.00s |                                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,173 |                   116 |          3,289 |        1,318 |      63.2 |          13 |            4.69s |      1.35s |       6.34s |                                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               830 |                    94 |            924 |          655 |      30.9 |          19 |            4.71s |      2.32s |       7.29s |                                 |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,344 |                    86 |          2,430 |        1,425 |      32.1 |          18 |            4.77s |      1.80s |       6.82s | metadata-borrowing              |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               821 |                   500 |          1,321 |        2,539 |       111 |           6 |            5.24s |      1.45s |       6.95s | missing-sections(title+desc...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,844 |                   117 |          2,961 |        2,414 |      32.4 |          19 |            5.30s |      1.88s |       7.45s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               835 |                    94 |            929 |          574 |      27.1 |          20 |            5.33s |      2.52s |       8.11s | metadata-borrowing              |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,366 |                   118 |          3,484 |        1,779 |        39 |          16 |            5.34s |      1.68s |       7.29s | description-sentences(3), ...   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,676 |                    93 |          2,769 |          775 |      31.9 |          22 |            6.81s |      2.06s |       9.13s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               829 |                   316 |          1,145 |        1,824 |        47 |          17 |            7.56s |      2.35s |      10.18s | fabrication, ...                |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,366 |                   117 |          3,483 |        1,888 |      20.1 |          28 |            8.03s |      2.51s |      10.80s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               830 |                   106 |            936 |          527 |      17.2 |          33 |            8.12s |      3.36s |      11.74s |                                 |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,553 |                   500 |          2,053 |        1,657 |      70.6 |          18 |            8.47s |      1.87s |      10.58s | missing-sections(title+desc...  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               532 |                   161 |            693 |          326 |      21.5 |          15 |            9.50s |      1.58s |      11.34s | missing-sections(title+desc...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,387 |                   500 |          1,887 |        3,963 |      56.1 |         9.5 |            9.62s |      0.89s |      10.76s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               823 |                    51 |            874 |          253 |      7.42 |          65 |           10.59s |      6.27s |      17.12s | missing-sections(title+desc...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               533 |                    50 |            583 |          274 |       5.1 |          25 |           12.13s |      2.15s |      14.53s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,657 |                   500 |          5,157 |        3,863 |      45.4 |         4.6 |           12.89s |      1.11s |      14.26s | repetitive(phrase: "after t...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,675 |                   500 |          7,175 |        1,068 |      70.9 |         8.4 |           13.71s |      1.34s |      15.33s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,457 |                   500 |          3,957 |        1,540 |      42.4 |          15 |           14.45s |      1.59s |      16.29s | degeneration, ...               |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,675 |                   500 |          7,175 |        1,125 |      53.9 |          11 |           15.58s |      1.50s |      17.34s | degeneration, ...               |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,881 |                   500 |          2,381 |          834 |        39 |          60 |           15.87s |      5.81s |      22.15s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,748 |                   200 |          1,948 |          107 |      30.4 |          48 |           23.59s |      1.71s |      25.55s | keyword-count(20), ...          |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,748 |                   500 |          2,248 |          118 |      52.7 |          41 |           25.05s |      1.19s |      26.50s | degeneration, ...               |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,789 |                   500 |         17,289 |        1,028 |      57.5 |          13 |           25.66s |      1.11s |      27.02s | keyword-count(151), ...         |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,675 |                   500 |          7,175 |          522 |      36.6 |          78 |           26.83s |      6.53s |      33.62s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,804 |                   500 |         17,304 |          327 |      90.7 |          12 |           57.51s |      1.34s |      59.11s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,804 |                   500 |         17,304 |          318 |      88.2 |          35 |           59.30s |      3.48s |      63.10s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,804 |                   500 |         17,304 |          316 |      82.2 |          26 |           60.09s |      2.57s |      62.94s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,789 |                     5 |         16,794 |          276 |       239 |         5.1 |           61.50s |      0.58s |      62.34s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,804 |                   500 |         17,304 |          302 |      64.8 |          76 |           64.14s |     12.44s |      76.83s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,804 |                   500 |         17,304 |          235 |        30 |          26 |           88.74s |      2.14s |      91.16s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,804 |                   500 |         17,304 |          244 |      18.2 |          39 |           97.13s |      3.07s |     100.47s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,804 |                   500 |         17,304 |          242 |      18.1 |          39 |           97.89s |      3.06s |     101.23s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,553 |                   500 |          2,053 |        1,050 |      4.76 |          39 |          107.01s |      3.21s |     110.47s | degeneration, ...               |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.4.0
- _macOS Version:_ 26.4.1
- _SDK Version:_ 26.4
- _Xcode Version:_ 26.4.1
- _Xcode Build:_ 17E202
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
- `mlx`: `0.32.0.dev20260510+84961223`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.14.0`
- `transformers`: `5.8.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-10 21:23:22 BST_
