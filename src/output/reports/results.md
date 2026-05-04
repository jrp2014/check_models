# Model Performance Results

_Generated on 2026-05-04 21:20:44 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, clean outputs=7/51.
- _Useful now:_ 8 clean A/B model(s) worth first review.
- _Review watchlist:_ 43 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=21, neutral=3, worse=27 (baseline C 62/100).
- _Quality signal frequency:_ missing_sections=32, cutoff=18, context_echo=13,
  description_length=10, keyword_count=10, metadata_borrowing=10.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (52%; 16/54 measured model(s)).
- _Phase totals:_ model load=106.69s, local prompt prep=0.16s, upstream
  prefill / first-token=627.80s, post-prefill decode=472.75s, cleanup=5.56s.
- _Generation total:_ 1100.55s across 51 model(s); upstream prefill /
  first-token split available for 51/51 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=51, exception=3.
- _Validation overhead:_ 11.21s total (avg 0.21s across 54 model(s)).
- _Upstream prefill / first-token latency:_ Avg 12.31s | Min 0.06s | Max
  80.88s across 51 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (349.8 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/LFM2-VL-1.6B-8bit` (0.44s)
- **📊 Average TPS:** 76.7 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1078.6 GB
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 244 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 9 | ✅ B: 12 | 🟡 C: 10 | 🟠 D: 13 | ❌ F: 7

**Average Utility Score:** 57/100

**Existing Metadata Baseline:** 🟡 C (62/100)
**Vs Existing Metadata:** Avg Δ -6 | Better: 21, Neutral: 3, Worse: 27

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

- **Generation Tps**: Avg: 76.7 | Min: 4.59 | Max: 350
- **Peak Memory**: Avg: 21 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 23.84s | Min: 1.32s | Max: 113.74s
- **Generation Time**: Avg: 21.58s | Min: 0.67s | Max: 110.44s
- **Model Load Time**: Avg: 2.04s | Min: 0.44s | Max: 9.87s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 87/100 | Description 83 | Keywords 83 | Speed 182 TPS | Memory 13
  | Caveat nontext prompt burden=89%; missing terms: classic, style, during,
  exposing, vast)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (Utility A 86/100 | Description 93 | Keywords 86 | Speed 62.1 TPS | Memory
  19 | Caveat nontext prompt burden=89%; missing terms: style, during,
  receded, exposing, vast)
- _Best keywording:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit)
  (Utility A 86/100 | Description 84 | Keywords 92 | Speed 17.2 TPS | Memory
  33 | Caveat missing terms: classic, style, wooden, estuary, receded)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility D 39/100 | Description 78 | Keywords 0 | Speed 350 TPS | Memory 2.6
  | Caveat missing sections: keywords; context echo=98%)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility D 45/100 | Description 70 | Keywords 0 | Speed 347 TPS | Memory 2.1
  | Caveat missing sections: keywords; context echo=44%; nonvisual metadata
  reused)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 87/100 | Description 83 | Keywords 83 | Speed 182 TPS | Memory 13
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

_Overall runtime:_ 1228.22s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                   |                       |                |              |           |             |                  |      0.13s |       1.07s |                                 |             mlx |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.17s |       1.14s |                                 |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.16s |       3.45s |                                 |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               513 |                    90 |            603 |        5,983 |       350 |         2.6 |            0.67s |      0.44s |       1.32s | missing-sections(keywords), ... |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               767 |                   110 |            877 |        7,125 |       323 |           3 |            0.71s |      0.44s |       1.35s | title-length(3), ...            |                 |
| `qnguyen3/nanoLLaVA`                                    |               513 |                    64 |            577 |        4,757 |       110 |         4.6 |            1.06s |      0.52s |       1.80s | missing-sections(keywords)      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               517 |                   181 |            698 |        5,050 |       347 |         2.1 |            1.09s |      0.59s |       1.90s | fabrication, ...                |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             2,087 |                    70 |          2,157 |        3,968 |       125 |         5.8 |            1.49s |      0.61s |       2.29s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             2,087 |                    70 |          2,157 |        3,971 |       124 |         5.8 |            1.53s |      0.59s |       2.33s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               784 |                    89 |            873 |        1,549 |       114 |          17 |            1.62s |      2.30s |       4.15s |                                 |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               623 |                   137 |            760 |        1,504 |       133 |         5.8 |            1.89s |      0.55s |       2.65s | fabrication, ...                |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,538 |                    31 |          1,569 |        1,357 |      31.8 |          12 |            2.47s |      1.71s |       4.40s | missing-sections(title+desc...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             3,503 |                    20 |          3,523 |        1,698 |        64 |         9.7 |            2.80s |      0.93s |       3.94s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             4,114 |                   127 |          4,241 |        2,258 |       182 |          13 |            2.89s |      0.89s |       4.00s |                                 |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               575 |                   500 |          1,075 |       10,051 |       185 |         3.7 |            3.04s |      0.52s |       3.77s | repetitive(phrase: "mudflat...  |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,064 |                    87 |          3,151 |        2,825 |        35 |          18 |            3.96s |      1.76s |       5.93s |                                 |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               774 |                   500 |          1,274 |        2,509 |       119 |           6 |            4.81s |      1.41s |       6.44s | repetitive(phrase: "18:33:4...  |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               784 |                    88 |            872 |          589 |      27.5 |          20 |            4.86s |      2.50s |       7.58s |                                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             4,115 |                   114 |          4,229 |        1,255 |      65.3 |          18 |            5.39s |      1.30s |       6.91s |                                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               783 |                   113 |            896 |          600 |      29.8 |          19 |            5.43s |      2.43s |       8.09s | keyword-count(19)               |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               484 |                    17 |            501 |          258 |      5.23 |          25 |            5.44s |      2.19s |       7.85s | missing-sections(title+desc...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             3,507 |                   117 |          3,624 |        2,448 |        32 |          20 |            5.50s |      1.87s |       7.58s | description-sentences(3), ...   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             4,115 |                   118 |          4,233 |        1,244 |      62.1 |          19 |            5.58s |      1.35s |       7.15s |                                 |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               782 |                   266 |          1,048 |        1,798 |      48.7 |          17 |            6.20s |      2.20s |       8.62s | missing-sections(title+desc...  |                 |
| `mlx-community/pixtral-12b-8bit`                        |             4,662 |                   135 |          4,797 |        1,834 |      37.6 |          18 |            6.52s |      1.77s |       8.52s | description-sentences(3), ...   |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,064 |                   120 |          3,184 |        1,222 |      31.6 |          19 |            6.67s |      1.72s |       8.61s | description-sentences(3), ...   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               483 |                   119 |            602 |          296 |      21.9 |          15 |            7.37s |      1.46s |       9.04s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,538 |                    31 |          1,569 |        1,072 |      5.41 |          27 |            7.53s |      2.39s |      10.13s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               783 |                   104 |            887 |          542 |      17.2 |          33 |            7.84s |      3.50s |      11.57s |                                 |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,516 |                   490 |          2,006 |        1,624 |      70.1 |          18 |            8.31s |      1.91s |      10.42s | missing-sections(title), ...    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             3,619 |                   108 |          3,727 |          680 |      31.1 |          27 |            9.18s |      2.06s |      11.48s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,337 |                   500 |          1,837 |        3,934 |      56.1 |         9.5 |            9.53s |      0.84s |      10.58s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/pixtral-12b-bf16`                        |             4,662 |                   138 |          4,800 |        1,911 |      20.2 |          28 |            9.63s |      2.64s |      12.49s | description-sentences(3), ...   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,337 |                   500 |          1,837 |        3,874 |      55.5 |         9.5 |            9.64s |      0.88s |      10.73s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,468 |                   107 |          1,575 |          185 |      51.8 |          29 |           10.56s |      1.16s |      11.94s | missing-sections(title+desc...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,810 |                   500 |          2,310 |        1,151 |      53.3 |          60 |           11.54s |      4.83s |      16.76s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,610 |                   500 |          5,110 |        3,758 |      41.9 |         4.6 |           13.73s |      1.22s |      15.17s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,570 |                   500 |          7,070 |        1,014 |      63.1 |         8.4 |           14.74s |      1.32s |      16.27s | missing-sections(title+keyw...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             4,753 |                   500 |          5,253 |        1,552 |        41 |          15 |           15.60s |      1.57s |      17.39s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,570 |                   500 |          7,070 |        1,109 |      53.1 |          11 |           15.65s |      1.48s |      17.35s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,468 |                   219 |          1,687 |          183 |      30.4 |          36 |           15.76s |      1.71s |      17.68s | refusal(insufficient_info), ... |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,570 |                   500 |          7,070 |          513 |      37.3 |          78 |           26.56s |      5.91s |      32.69s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,538 |                   500 |          2,038 |        3,271 |      18.8 |          11 |           27.43s |      1.41s |      29.06s | repetitive(phrase: "the ima...  |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,901 |                   500 |         17,401 |          940 |        53 |          14 |           27.98s |      1.18s |      29.37s | description-sentences(3), ...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,916 |                   500 |         17,416 |          316 |      95.6 |          26 |           59.43s |      2.46s |      62.11s | missing-sections(descriptio...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,916 |                   500 |         17,416 |          315 |      88.3 |          35 |           60.05s |      3.10s |      63.36s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,901 |                    11 |         16,912 |          270 |       194 |         5.1 |           63.19s |      0.55s |      63.95s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,916 |                   500 |         17,416 |          284 |      88.4 |          12 |           65.84s |      1.34s |      67.40s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,916 |                   500 |         17,416 |          280 |      63.5 |          76 |           69.02s |      9.87s |      79.10s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               772 |                   500 |          1,272 |          180 |      7.26 |          64 |           73.53s |      7.09s |      80.83s | missing-sections(title+desc...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,516 |                   351 |          1,867 |        1,035 |      4.59 |          39 |           78.41s |      3.39s |      82.02s | missing-sections(title), ...    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,916 |                   500 |         17,416 |          234 |        30 |          26 |           89.48s |      2.16s |      91.88s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,916 |                   500 |         17,416 |          232 |      18.2 |          39 |          100.95s |      3.08s |     104.26s | keyword-count(20), ...          |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,916 |                   500 |         17,416 |          209 |      17.3 |          39 |          110.44s |      3.07s |     113.74s | title-length(4), ...            |                 |

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
- `transformers`: `5.8.0.dev0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-04 21:20:44 BST_
