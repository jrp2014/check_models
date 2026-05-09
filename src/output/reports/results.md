# Model Performance Results

_Generated on 2026-05-10 00:41:07 BST_

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
  cutoff=18, description_length=10, keyword_count=10, context_ignored=10.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (54%; 14/55 measured model(s)).
- _Phase totals:_ model load=111.79s, local prompt prep=0.17s, upstream
  prefill / first-token=639.24s, post-prefill decode=425.90s, generation total
  (unsplit)=4.13s, cleanup=8.37s.
- _Generation total:_ 1069.26s across 51 model(s); upstream prefill /
  first-token split available for 48/51 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=29, exception=7, max_tokens=19.
- _Validation overhead:_ 14.05s total (avg 0.26s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 13.32s | Min 0.09s | Max
  88.56s across 48 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (345.7 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.51s)
- **📊 Average TPS:** 70.0 across 48 models

## 📈 Resource Usage

- **Total peak memory:** 1011.2 GB
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
- `mlx-community/Molmo-7B-D-0924-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
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
- **🔄 Repetitive Output (2):**
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "manfrotto bubbalong 2.5g. trip..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "after the label. -..."`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 70.0 | Min: 4.7 | Max: 346
- **Peak Memory**: Avg: 21 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 24.49s | Min: 1.65s | Max: 121.71s
- **Generation Time**: Avg: 22.19s | Min: 0.89s | Max: 118.30s
- **Model Load Time**: Avg: 2.03s | Min: 0.51s | Max: 6.30s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 88/100 | Description 98 | Keywords 85 | Speed 30.6 TPS | Memory
  19 | Caveat missing terms: Bell Tower, Chapel, Cross, Dorking, Fence)
- _Best descriptions:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit)
  (Utility A 82/100 | Description 100 | Keywords 84 | Speed 17.6 TPS | Memory
  33 | Caveat missing terms: Chapel, Cross, Dorking, Objects, Station wagon)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility A 83/100 | Description 95 | Keywords 92 | Speed 66.0 TPS | Memory
  13 | Caveat nontext prompt burden=85%; missing terms: Bell Tower, Chapel,
  Cross, Dorking, Fence)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility C 61/100 | Description 86 | Keywords 0 | Speed 346 TPS | Memory 2.8
  | Caveat missing sections: keywords; missing terms: Bell Tower, Chapel,
  Cross, Daylight, Dorking; nonvisual metadata reused)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility D 46/100 | Description 75 | Keywords 0 | Speed 344 TPS | Memory 2.1
  | Caveat missing sections: keywords; missing terms: sunny, day; nonvisual
  metadata reused)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 88/100 | Description 98 | Keywords 85 | Speed 30.6 TPS | Memory
  19 | Caveat missing terms: Bell Tower, Chapel, Cross, Dorking, Fence)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (7):_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +3 more. Example: `Model Error`.
- _🔄 Repetitive Output (2):_ [`mlx-community/Molmo-7B-D-0924-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmo-7b-d-0924-bf16),
  [`mlx-community/paligemma2-3b-pt-896-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-paligemma2-3b-pt-896-4bit).
  Example: token: `phrase: "manfrotto bubbalong 2.5g. trip..."`.
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

_Overall runtime:_ 1208.34s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                   |                       |                |              |           |             |                  |      0.19s |       1.23s |                                 |             mlx |
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.24s |       1.25s |                                 |         mlx-vlm |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |                   |                       |                |              |           |             |            1.71s |     10.45s |      12.60s | hallucination, fabrication, ... |         mlx-vlm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.17s |       1.21s |                                 |             mlx |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |                   |                       |                |              |           |             |            1.20s |      0.47s |       1.92s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |            1.21s |      0.60s |       2.07s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.19s |       3.56s |                                 |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               561 |                   137 |            698 |         6366 |       346 |         2.8 |            0.89s |      0.51s |       1.65s | fabrication, ...                |                 |
| `qnguyen3/nanoLLaVA`                                    |               561 |                    68 |            629 |         4379 |      98.1 |         4.9 |            1.25s |      0.53s |       2.04s | missing-sections(keywords), ... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               565 |                   222 |            787 |         5579 |       344 |         2.1 |            1.54s |      0.62s |       2.43s | missing-sections(keywords), ... |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               835 |                    86 |            921 |         1607 |       108 |          17 |            1.73s |      2.33s |       4.35s |                                 |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               671 |                   131 |            802 |         1982 |       135 |         5.5 |            1.85s |      0.55s |       2.66s | description-sentences(3), ...   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,585 |                    17 |          1,602 |         2909 |      17.4 |          11 |            1.97s |      1.63s |       3.88s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,172 |                   157 |          3,329 |         3058 |       184 |         7.8 |            2.33s |      0.94s |       3.53s | description-sentences(4), ...   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,771 |                   182 |          1,953 |         4048 |       123 |         5.6 |            2.41s |      0.61s |       3.26s | title-length(11), ...           |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,771 |                   182 |          1,953 |         3836 |       123 |         5.7 |            2.49s |      0.58s |       3.33s | title-length(11), ...           |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,585 |                    20 |          1,605 |         1182 |      27.6 |          12 |            2.52s |      1.89s |       4.67s | missing-sections(title+desc...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,789 |                    11 |          2,800 |         1251 |      64.9 |         9.7 |            2.93s |      0.91s |       4.10s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,387 |                   129 |          1,516 |         3914 |      56.9 |         9.5 |            2.97s |      0.98s |       4.20s | keyword-count(19), ...          |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,344 |                    75 |          2,419 |         2641 |      34.9 |          18 |            3.48s |      1.69s |       5.42s | title-length(4)                 |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,585 |                     9 |          1,594 |          926 |      5.24 |          27 |            3.89s |      2.63s |       6.79s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,173 |                   108 |          3,281 |         1313 |        66 |          13 |            4.50s |      1.30s |       6.06s |                                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,173 |                   116 |          3,289 |         1402 |      63.3 |          13 |            4.54s |      1.33s |       6.13s |                                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               830 |                    94 |            924 |          662 |      30.6 |          19 |            4.72s |      2.31s |       7.30s |                                 |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               821 |                   500 |          1,321 |         2615 |       120 |           6 |            4.84s |      1.43s |       6.54s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,344 |                    86 |          2,430 |         1317 |      32.2 |          18 |            4.90s |      1.74s |       6.90s | metadata-borrowing              |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,844 |                   117 |          2,961 |         2375 |      32.8 |          19 |            5.28s |      1.88s |       7.43s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               835 |                    94 |            929 |          591 |      27.2 |          20 |            5.28s |      2.53s |       8.08s | metadata-borrowing              |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,366 |                   118 |          3,484 |         1345 |      33.8 |          16 |            6.43s |      1.73s |       8.43s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               829 |                   316 |          1,145 |         1902 |      48.4 |          17 |            7.35s |      2.21s |       9.84s | fabrication, ...                |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,676 |                    93 |          2,769 |          676 |      30.5 |          22 |            7.49s |      2.15s |       9.94s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               830 |                   106 |            936 |          578 |      17.6 |          33 |            7.86s |      3.36s |      11.49s |                                 |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,366 |                   117 |          3,483 |         1500 |      18.6 |          28 |            8.98s |      2.61s |      11.85s | description-sentences(3), ...   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,387 |                   500 |          1,887 |         3565 |      54.7 |         9.5 |            9.90s |      0.88s |      11.05s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               823 |                    51 |            874 |          294 |      7.51 |          65 |           10.02s |      5.97s |      16.25s | missing-sections(title+desc...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,553 |                   500 |          2,053 |          998 |      69.2 |          18 |           10.13s |      4.55s |      14.93s | missing-sections(title+desc...  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               532 |                   161 |            693 |          303 |      19.2 |          15 |           10.51s |      1.46s |      12.23s | missing-sections(title+desc...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               533 |                    50 |            583 |          200 |      5.08 |          25 |           12.92s |      3.00s |      16.19s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,657 |                   500 |          5,157 |         3225 |      43.1 |         4.6 |           13.71s |      1.24s |      15.23s | repetitive(phrase: "after t...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,675 |                   500 |          7,175 |          968 |      65.1 |         8.4 |           14.97s |      1.29s |      16.53s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,457 |                   500 |          3,957 |         1437 |      40.4 |          15 |           15.22s |      1.59s |      17.07s | degeneration, ...               |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,675 |                   500 |          7,175 |         1109 |      52.3 |          11 |           15.96s |      1.39s |      17.62s | degeneration, ...               |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,748 |                   500 |          2,248 |          102 |      52.7 |          41 |           27.34s |      1.29s |      28.88s | degeneration, ...               |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,789 |                   500 |         17,289 |          963 |      53.5 |          13 |           27.44s |      1.22s |      28.92s | keyword-count(151), ...         |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,675 |                   500 |          7,175 |          490 |      32.4 |          78 |           29.49s |      5.55s |      35.30s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,748 |                   500 |          2,248 |         99.2 |      30.2 |          48 |           34.87s |      1.70s |      36.83s | repetitive(phrase: "manfrot...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,804 |                   500 |         17,304 |          334 |       102 |          26 |           55.94s |      2.35s |      58.55s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,789 |                     5 |         16,794 |          293 |       247 |         5.1 |           58.03s |      0.55s |      58.84s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,804 |                   500 |         17,304 |          321 |      87.8 |          35 |           58.68s |      3.25s |      62.22s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,804 |                   500 |         17,304 |          282 |      63.6 |          76 |           68.12s |      6.30s |      74.68s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,804 |                   500 |         17,304 |          262 |      77.3 |          12 |           71.40s |      1.37s |      73.04s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,804 |                   500 |         17,304 |          250 |      29.9 |          26 |           84.63s |      2.13s |      87.04s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,804 |                   500 |         17,304 |          224 |      17.4 |          39 |          104.59s |      3.08s |     107.95s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,553 |                   500 |          2,053 |         1057 |       4.7 |          39 |          108.51s |      3.24s |     112.02s | degeneration, ...               |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,804 |                   500 |         17,304 |          190 |      17.3 |          39 |          118.30s |      3.12s |     121.71s | refusal(explicit_refusal), ...  |                 |

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

_Report generated on: 2026-05-10 00:41:07 BST_
