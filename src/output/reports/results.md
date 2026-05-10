# Model Performance Results

_Generated on 2026-05-10 01:31:28 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 7 (top owners: mlx-vlm=4, mlx=2,
  model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=5, clean outputs=5/48.
- _Useful now:_ 5 clean A/B model(s) worth first review.
- _Review watchlist:_ 43 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=12, neutral=1, worse=35 (baseline B 75/100).
- _Quality signal frequency:_ metadata_borrowing=28, missing_sections=27,
  cutoff=18, description_length=10, trusted_hint_ignored=10,
  context_ignored=10.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (54%; 15/55 measured model(s)).
- _Phase totals:_ model load=114.11s, local prompt prep=0.16s, upstream
  prefill / first-token=634.24s, post-prefill decode=421.75s, generation total
  (unsplit)=4.00s, cleanup=6.55s.
- _Generation total:_ 1059.99s across 51 model(s); upstream prefill /
  first-token split available for 48/51 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=29, exception=7, max_tokens=19.
- _Validation overhead:_ 14.09s total (avg 0.26s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 13.21s | Min 0.10s | Max
  78.98s across 48 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (350.3 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 70.4 across 48 models

## 📈 Resource Usage

- **Total peak memory:** 1010.7 GB
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 245 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 6 | ✅ B: 20 | 🟡 C: 7 | 🟠 D: 8 | ❌ F: 7

**Average Utility Score:** 59/100

**Existing Metadata Baseline:** ✅ B (75/100)
**Vs Existing Metadata:** Avg Δ -17 | Better: 12, Neutral: 1, Worse: 35

- **Best for cataloging:** `mlx-community/gemma-3-27b-it-qat-4bit` (🏆 A, 88/100)
- **Best descriptions:** `mlx-community/gemma-4-26b-a4b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (92/100)
- **Worst for cataloging:** `mlx-community/gemma-3n-E2B-4bit` (❌ F, 2/100)

### ⚠️ 15 Models with Low Utility (D/F)

- `meta-llama/Llama-3.2-11B-Vision-Instruct`: 🟠 D (47/100) - Keywords are not specific or diverse enough
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-Flash-mxfp4`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-8bit`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-bf16`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (2/100) - Output too short to be useful
- `mlx-community/gemma-4-31b-bf16`: 🟠 D (41/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (9/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (33/100) - Lacks visual description of image
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (35/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (28/100) - Output lacks detail
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (48/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (7):**
  - `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (`Model Error`)
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (`Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (`Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "after the label. -..."`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 70.4 | Min: 4.74 | Max: 350
- **Peak Memory**: Avg: 21 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 24.43s | Min: 1.61s | Max: 112.47s
- **Generation Time**: Avg: 22.00s | Min: 0.90s | Max: 108.85s
- **Model Load Time**: Avg: 2.17s | Min: 0.45s | Max: 10.77s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 88/100 | Description 98 | Keywords 85 | Speed 30.4 TPS | Memory
  19 | Caveat missing terms: Bell Tower, Chapel, Cross, Dorking, Fence)
- _Best descriptions:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit)
  (Utility A 82/100 | Description 100 | Keywords 84 | Speed 17.6 TPS | Memory
  33 | Caveat missing terms: Chapel, Cross, Dorking, Objects, Station wagon)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility A 83/100 | Description 95 | Keywords 92 | Speed 66.5 TPS | Memory
  13 | Caveat nontext prompt burden=85%; missing terms: Bell Tower, Chapel,
  Cross, Dorking, Fence)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility C 61/100 | Description 86 | Keywords 0 | Speed 350 TPS | Memory 2.7
  | Caveat missing sections: keywords; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking; nonvisual metadata reused)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility D 46/100 | Description 75 | Keywords 0 | Speed 346 TPS | Memory 2.2
  | Caveat missing sections: keywords; missing terms: sunny, day; nonvisual
  metadata reused)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 88/100 | Description 98 | Keywords 85 | Speed 30.4 TPS | Memory
  19 | Caveat missing terms: Bell Tower, Chapel, Cross, Dorking, Fence)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (7):_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +3 more. Example: `Model Error`.
- _🔄 Repetitive Output (1):_ [`mlx-community/paligemma2-3b-pt-896-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-paligemma2-3b-pt-896-4bit).
  Example: token: `phrase: "after the label. -..."`.
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (15):_ [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  +11 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types        | Affected Models                                                                                                                                    |
|----------------|------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-vlm`      |          4 | Error, Model Error | `facebook/pe-av-large`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `mlx`          |          2 | Model Error        | `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`                                                                      |
| `model-config` |          1 | Processor Error    | `mlx-community/MolmoPoint-8B-fp16`                                                                                                                 |

### Actionable Items by Package

#### mlx-vlm

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.p...`
  - Type: `ValueError`
- mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 (Error)
  - Error: `Model runtime error during generation for mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16: property 'text' of 'Naive...`
  - Type: `ValueError`
- mlx-community/LFM2-VL-1.6B-8bit (Error)
  - Error: `Model runtime error during generation for mlx-community/LFM2-VL-1.6B-8bit: property 'text' of 'NaiveStreamingDetokeni...`
  - Type: `ValueError`
- mlx-community/LFM2.5-VL-1.6B-bf16 (Error)
  - Error: `Model runtime error during generation for mlx-community/LFM2.5-VL-1.6B-bf16: property 'text' of 'NaiveStreamingDetoke...`
  - Type: `ValueError`

#### mlx

- LiquidAI/LFM2.5-VL-450M-MLX-bf16 (Model Error)
  - Error: `Model loading failed: Received 2 parameters not in model: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_project...`
  - Type: `ValueError`
- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
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

_Overall runtime:_ 1199.65s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                   |                       |                |              |           |             |                  |      0.19s |       1.21s |                                 |             mlx |
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.12s |       1.11s |                                 |         mlx-vlm |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |                   |                       |                |              |           |             |            1.59s |      6.31s |       8.32s | hallucination, fabrication, ... |         mlx-vlm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.26s |       1.31s |                                 |             mlx |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |                   |                       |                |              |           |             |            1.20s |      0.48s |       1.93s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |            1.21s |      0.56s |       2.03s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.19s |       3.56s |                                 |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               561 |                   137 |            698 |         5858 |       350 |         2.7 |            0.90s |      0.45s |       1.61s | fabrication, ...                |                 |
| `qnguyen3/nanoLLaVA`                                    |               561 |                    68 |            629 |         4547 |       109 |         4.7 |            1.18s |      0.74s |       2.17s | missing-sections(keywords), ... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               565 |                   222 |            787 |         5545 |       346 |         2.2 |            1.52s |      0.83s |       2.60s | missing-sections(keywords), ... |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               835 |                    86 |            921 |         1632 |       108 |          17 |            1.72s |      2.34s |       4.34s |                                 |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,585 |                    17 |          1,602 |         3300 |      20.2 |          11 |            1.75s |      1.42s |       3.43s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               671 |                   131 |            802 |         1948 |       133 |         5.5 |            1.86s |      0.56s |       2.68s | description-sentences(3), ...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,585 |                    20 |          1,605 |         1371 |      31.6 |          12 |            2.22s |      1.64s |       4.12s | missing-sections(title+desc...  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,771 |                   182 |          1,953 |         4276 |       127 |         5.6 |            2.32s |      0.64s |       3.20s | title-length(11), ...           |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,172 |                   157 |          3,329 |         3051 |       183 |         7.8 |            2.35s |      0.92s |       3.53s | description-sentences(4), ...   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,771 |                   182 |          1,953 |         4216 |       123 |         5.7 |            2.43s |      0.60s |       3.29s | title-length(11), ...           |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,387 |                   129 |          1,516 |         3898 |      57.7 |         9.5 |            2.95s |      0.92s |       4.12s | keyword-count(19), ...          |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,789 |                    11 |          2,800 |         1203 |      65.6 |         9.7 |            3.01s |      0.91s |       4.18s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,585 |                     9 |          1,594 |         1070 |      5.88 |          27 |            3.45s |      2.48s |       6.18s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,344 |                    75 |          2,419 |         2633 |      35.2 |          18 |            3.47s |      1.67s |       5.40s | title-length(4)                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,173 |                   108 |          3,281 |         1444 |      66.5 |          13 |            4.27s |      1.58s |       6.10s |                                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,173 |                   116 |          3,289 |         1434 |      62.6 |          13 |            4.51s |      1.35s |       6.12s |                                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               830 |                    94 |            924 |          656 |      30.4 |          19 |            4.75s |      2.30s |       7.31s |                                 |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,344 |                    86 |          2,430 |         1436 |      32.3 |          18 |            4.75s |      1.77s |       6.78s | metadata-borrowing              |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               821 |                   500 |          1,321 |         2575 |       117 |           6 |            4.99s |      1.42s |       6.66s | missing-sections(title+desc...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,844 |                   117 |          2,961 |         2433 |      32.5 |          19 |            5.28s |      1.91s |       7.48s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               835 |                    94 |            929 |          584 |      27.1 |          20 |            5.30s |      2.54s |       8.11s | metadata-borrowing              |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,366 |                   118 |          3,484 |         1479 |      38.8 |          16 |            5.74s |      1.66s |       7.66s | description-sentences(3), ...   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,676 |                    93 |          2,769 |          750 |      31.9 |          22 |            6.93s |      2.13s |       9.35s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               829 |                   316 |          1,145 |         1845 |        48 |          17 |            7.42s |      2.25s |       9.93s | fabrication, ...                |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               830 |                   106 |            936 |          576 |      17.6 |          33 |            7.83s |      3.37s |      11.46s |                                 |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,366 |                   117 |          3,483 |         1853 |      20.6 |          28 |            7.91s |      2.53s |      10.70s | description-sentences(3), ...   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,553 |                   500 |          2,053 |         1614 |      68.5 |          18 |            8.75s |      2.04s |      11.04s | missing-sections(title+desc...  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               532 |                   161 |            693 |          330 |      21.8 |          15 |            9.36s |      1.54s |      11.16s | missing-sections(title+desc...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,387 |                   500 |          1,887 |         3899 |        52 |         9.5 |           10.33s |      0.88s |      11.47s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               823 |                    51 |            874 |          243 |      7.57 |          65 |           10.58s |      6.23s |      17.08s | missing-sections(title+desc...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               533 |                    50 |            583 |          279 |      5.08 |          25 |           12.13s |      2.17s |      14.56s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,657 |                   500 |          5,157 |         3805 |      45.2 |         4.6 |           12.90s |      1.15s |      14.32s | repetitive(phrase: "after t...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,675 |                   500 |          7,175 |         1125 |      60.4 |         8.4 |           14.62s |      1.40s |      16.32s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,457 |                   500 |          3,957 |         1465 |      40.6 |          15 |           15.20s |      1.73s |      17.27s | degeneration, ...               |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,675 |                   500 |          7,175 |         1078 |      48.3 |          11 |           16.94s |      1.49s |      18.70s | degeneration, ...               |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,748 |                   500 |          2,248 |          111 |      52.5 |          41 |           26.03s |      1.20s |      27.49s | degeneration, ...               |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,789 |                   500 |         17,289 |          904 |      53.7 |          13 |           28.53s |      1.14s |      29.93s | keyword-count(151), ...         |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,675 |                   500 |          7,175 |          365 |      34.5 |          78 |           33.25s |     10.31s |      43.89s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,748 |                   500 |          2,248 |         97.5 |      30.3 |          48 |           35.11s |      1.72s |      37.09s | degeneration, ...               |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,789 |                     5 |         16,794 |          295 |       241 |         5.1 |           57.50s |      0.55s |      58.31s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,804 |                   500 |         17,304 |          301 |      92.6 |          26 |           61.98s |      2.49s |      64.73s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,804 |                   500 |         17,304 |          292 |      86.8 |          35 |           64.00s |      3.16s |      67.43s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,804 |                   500 |         17,304 |          293 |      84.6 |          12 |           64.05s |      1.35s |      65.66s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,804 |                   500 |         17,304 |          272 |      64.8 |          76 |           70.54s |     10.77s |      81.58s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,804 |                   500 |         17,304 |          247 |      29.3 |          26 |           85.87s |      2.12s |      88.26s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,804 |                   500 |         17,304 |          233 |      17.7 |          39 |          101.06s |      3.07s |     104.41s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,553 |                   500 |          2,053 |         1054 |      4.74 |          39 |          107.60s |      3.25s |     111.12s | degeneration, ...               |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,804 |                   500 |         17,304 |          213 |      17.2 |          39 |          108.85s |      3.31s |     112.47s | refusal(explicit_refusal), ...  |                 |

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

_Report generated on: 2026-05-10 01:31:28 BST_
