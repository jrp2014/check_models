# Model Performance Results

_Generated on 2026-05-04 20:21:24 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, clean outputs=7/51.
- _Useful now:_ 8 clean A/B model(s) worth first review.
- _Review watchlist:_ 43 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=22, neutral=3, worse=26 (baseline C 62/100).
- _Quality signal frequency:_ missing_sections=31, cutoff=18, context_echo=13,
  keyword_count=11, description_length=11, metadata_borrowing=10.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (90%; 50/54 measured
  model(s)).
- _Phase totals:_ model load=113.09s, prompt prep=0.16s, decode=1132.96s,
  cleanup=5.99s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=51, exception=3.
- _Validation overhead:_ 11.49s total (avg 0.21s across 54 model(s)).
- _First-token latency:_ Avg 12.44s | Min 0.08s | Max 75.00s across 51
  model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/LFM2-VL-1.6B-8bit` (323.5 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/LFM2-VL-1.6B-8bit` (0.43s)
- **📊 Average TPS:** 72.2 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1078.5 GB
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 244 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 9 | ✅ B: 13 | 🟡 C: 9 | 🟠 D: 13 | ❌ F: 7

**Average Utility Score:** 57/100

**Existing Metadata Baseline:** 🟡 C (62/100)
**Vs Existing Metadata:** Avg Δ -6 | Better: 22, Neutral: 3, Worse: 26

- **Best for cataloging:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (🏆 A, 87/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (93/100)
- **Best keywording:** `mlx-community/Qwen3.6-27B-mxfp8` (97/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 1/100)

### ⚠️ 20 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (38/100) - Keywords are not specific or diverse enough
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (18/100) - Output lacks detail
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-nvfp4`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (42/100) - Missing requested structure
- `mlx-community/LFM2.5-VL-1.6B-bf16`: 🟠 D (36/100) - Lacks visual description of image
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (1/100) - Output lacks detail
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (38/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (30/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (16/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (39/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (20/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (20/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (6):**
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "flags, flags, flags, flags,..."`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "mudflats, flags, boat, water,..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "flags, flags, flags, flags,..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "18:33:45 18:33:45 18:33:45 18:..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the image is in..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 72.2 | Min: 4.44 | Max: 324
- **Peak Memory**: Avg: 21 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 24.61s | Min: 1.34s | Max: 109.90s
- **Generation Time**: Avg: 22.21s | Min: 0.70s | Max: 106.56s
- **Model Load Time**: Avg: 2.17s | Min: 0.43s | Max: 10.72s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 87/100 | Description 83 | Keywords 83 | Speed 181 TPS | Memory 13
  | Caveat nontext prompt burden=89%; missing terms: classic, style, during,
  exposing, vast)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (Utility A 86/100 | Description 93 | Keywords 86 | Speed 62.5 TPS | Memory
  19 | Caveat nontext prompt burden=89%; missing terms: style, during,
  receded, exposing, vast)
- _Best keywording:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit)
  (Utility A 86/100 | Description 84 | Keywords 92 | Speed 15.6 TPS | Memory
  33 | Caveat missing terms: classic, style, wooden, estuary, receded)
- _Fastest generation:_ [`mlx-community/LFM2-VL-1.6B-8bit`](model_gallery.md#model-mlx-community-lfm2-vl-16b-8bit)
  (Utility D 42/100 | Description 76 | Keywords 50 | Speed 324 TPS | Memory
  3.0 | Caveat missing terms: classic, style, sailboat, dark, hull;
  keywords=9; context echo=64%)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility D 45/100 | Description 70 | Keywords 0 | Speed 304 TPS | Memory 2.1
  | Caveat missing sections: keywords; context echo=44%; nonvisual metadata
  reused)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 87/100 | Description 83 | Keywords 83 | Speed 181 TPS | Memory 13
  | Caveat nontext prompt burden=89%; missing terms: classic, style, during,
  exposing, vast)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Model Error`.
- _🔄 Repetitive Output (6):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/Phi-3.5-vision-instruct-bf16`](model_gallery.md#model-mlx-community-phi-35-vision-instruct-bf16),
  [`mlx-community/gemma-3n-E2B-4bit`](model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  +2 more. Example: token: `phrase: "flags, flags, flags, flags,..."`.
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (20):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  +16 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types     | Affected Models                                                               |
|----------------|------------|-----------------|-------------------------------------------------------------------------------|
| `mlx`          |          2 | Model Error     | `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit` |
| `model-config` |          1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16`                                            |

### Actionable Items by Package

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
> &#45; Description hint: A classic-style sailboat with a dark hull and wooden
> mast is moored in a calm estuary during low tide. The water has receded,
> exposing a vast expanse of green, algae-covered mudflats behind the vessel.
> The boat, adorned with a string of small flags, floats peacefully, waiting
> for the tide to rise again.
> &#45; Capture metadata: Taken on 2026-05-02 18:33:45 BST (at 18:33:45 local
> time). GPS: 52.089294°N, 1.317741°E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1267.78s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                   |                       |                |              |           |             |                  |      0.16s |       1.14s |                                 |             mlx |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.19s |       1.16s |                                 |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.21s |       3.54s |                                 |    model-config |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               767 |                   110 |            877 |        7,420 |       324 |           3 |            0.70s |      0.43s |       1.34s | title-length(3), ...            |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               513 |                    90 |            603 |        4,801 |       261 |         2.5 |            0.81s |      0.49s |       1.52s | missing-sections(keywords), ... |                 |
| `qnguyen3/nanoLLaVA`                                    |               513 |                    64 |            577 |        4,429 |       113 |         4.7 |            1.01s |      0.50s |       1.72s | missing-sections(keywords)      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               517 |                   181 |            698 |        4,577 |       304 |         2.1 |            1.24s |      0.65s |       2.13s | fabrication, ...                |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             2,087 |                    70 |          2,157 |        3,842 |       121 |         5.8 |            1.54s |      0.59s |       2.33s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             2,087 |                    70 |          2,157 |        3,700 |       115 |         5.8 |            1.62s |      0.58s |       2.41s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               784 |                    89 |            873 |        1,453 |      99.5 |          17 |            1.78s |      2.49s |       4.51s |                                 |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               623 |                   137 |            760 |        1,367 |       122 |         5.8 |            2.04s |      0.55s |       2.81s | fabrication, ...                |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,538 |                    31 |          1,569 |        1,195 |      27.9 |          12 |            2.78s |      1.79s |       4.79s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             4,114 |                   127 |          4,241 |        2,254 |       181 |          13 |            2.91s |      0.90s |       4.02s |                                 |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               575 |                   500 |          1,075 |        7,585 |       187 |         3.7 |            3.01s |      0.60s |       3.82s | repetitive(phrase: "mudflat...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             3,503 |                    20 |          3,523 |        1,454 |      56.1 |         9.7 |            3.22s |      0.90s |       4.35s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,064 |                    87 |          3,151 |        2,626 |      32.8 |          18 |            4.19s |      1.71s |       6.11s |                                 |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               774 |                   500 |          1,274 |        2,239 |       108 |           6 |            5.30s |      1.52s |       7.04s | repetitive(phrase: "18:33:4...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               484 |                    17 |            501 |          259 |      5.29 |          25 |            5.39s |      2.17s |       7.76s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             4,115 |                   114 |          4,229 |        1,253 |      65.6 |          18 |            5.39s |      1.29s |       6.90s |                                 |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               784 |                    88 |            872 |          533 |      24.5 |          20 |            5.41s |      2.64s |       8.27s |                                 |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             3,507 |                   117 |          3,624 |        2,379 |      32.6 |          20 |            5.50s |      1.92s |       7.63s | description-sentences(3), ...   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             4,115 |                   118 |          4,233 |        1,243 |      62.5 |          19 |            5.57s |      1.31s |       7.10s |                                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               783 |                   113 |            896 |          563 |        28 |          19 |            5.77s |      2.39s |       8.38s | keyword-count(19)               |                 |
| `mlx-community/pixtral-12b-8bit`                        |             4,662 |                   135 |          4,797 |        1,873 |      38.7 |          18 |            6.35s |      1.69s |       8.27s | description-sentences(3), ...   |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,064 |                   120 |          3,184 |        1,442 |      29.5 |          19 |            6.58s |      1.73s |       8.52s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               782 |                   266 |          1,048 |        1,613 |      42.6 |          17 |            7.07s |      2.50s |       9.79s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,538 |                    31 |          1,569 |        1,055 |      5.43 |          27 |            7.52s |      2.50s |      10.27s | missing-sections(title+desc...  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               483 |                   119 |            602 |          284 |        21 |          15 |            7.69s |      1.59s |       9.50s | missing-sections(title+desc...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,516 |                   490 |          2,006 |        1,616 |        69 |          18 |            8.44s |      2.02s |      10.67s | missing-sections(title), ...    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               783 |                   104 |            887 |          497 |      15.6 |          33 |            8.60s |      3.56s |      12.39s |                                 |                 |
| `mlx-community/pixtral-12b-bf16`                        |             4,662 |                   138 |          4,800 |        1,934 |      20.2 |          28 |            9.60s |      2.58s |      12.38s | description-sentences(3), ...   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,337 |                   500 |          1,837 |        3,585 |      54.5 |         9.5 |            9.84s |      0.98s |      11.04s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             3,619 |                   108 |          3,727 |          609 |      27.9 |          27 |           10.20s |      2.15s |      12.59s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,337 |                   500 |          1,837 |        3,548 |      51.3 |         9.5 |           10.43s |      0.86s |      11.51s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,468 |                   119 |          1,587 |          143 |      51.8 |          29 |           13.15s |      1.18s |      14.54s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,610 |                   500 |          5,110 |        3,848 |      43.9 |         4.6 |           13.15s |      1.20s |      14.57s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,810 |                   500 |          2,310 |          766 |      48.7 |          60 |           13.26s |      5.64s |      19.28s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,570 |                   500 |          7,070 |        1,036 |        67 |         8.4 |           14.11s |      1.32s |      15.66s | missing-sections(title+keyw...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,570 |                   500 |          7,070 |        1,062 |      50.7 |          11 |           16.38s |      1.53s |      18.15s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             4,753 |                   500 |          5,253 |        1,556 |      38.2 |          15 |           16.48s |      1.57s |      18.27s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,468 |                   403 |          1,871 |          141 |      28.7 |          36 |           24.97s |      1.70s |      26.89s | refusal(insufficient_info), ... |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,538 |                   500 |          2,038 |        3,253 |        19 |          11 |           27.11s |      1.45s |      28.78s | repetitive(phrase: "the ima...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,570 |                   500 |          7,070 |          459 |      37.1 |          78 |           28.14s |      6.71s |      35.07s | missing-sections(title+desc...  |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,901 |                   500 |         17,401 |          925 |      48.9 |          14 |           29.06s |      1.21s |      30.49s | description-sentences(3), ...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,916 |                   500 |         17,416 |          308 |      87.1 |          35 |           61.45s |      3.50s |      65.18s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,901 |                    11 |         16,912 |          276 |       212 |         5.1 |           61.95s |      0.57s |      62.75s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,916 |                   500 |         17,416 |          297 |      89.3 |          26 |           63.30s |      3.95s |      67.49s | missing-sections(descriptio...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,916 |                   500 |         17,416 |          285 |      82.3 |          12 |           66.03s |      1.38s |      67.64s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,916 |                   500 |         17,416 |          263 |      58.4 |          76 |           73.68s |     10.72s |      84.62s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               772 |                   500 |          1,272 |          149 |       7.1 |          64 |           76.03s |      7.69s |      83.94s | missing-sections(title+desc...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,516 |                   351 |          1,867 |          949 |      4.44 |          39 |           81.25s |      3.29s |      84.75s | missing-sections(title), ...    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,916 |                   500 |         17,416 |          252 |      28.7 |          26 |           85.24s |      2.14s |      87.62s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,916 |                   500 |         17,416 |          232 |      16.4 |          39 |          104.17s |      3.10s |     107.50s | title-length(4), ...            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,916 |                   500 |         17,416 |          226 |      16.2 |          39 |          106.56s |      3.11s |     109.90s | keyword-count(20), ...          |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Automated review digest:_ [review.md](review.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

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
- `mlx`: `0.32.0.dev20260504+e8ebdebe`
- `mlx-vlm`: `0.4.5`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.13.0`
- `transformers`: `5.7.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-04 20:21:24 BST_
