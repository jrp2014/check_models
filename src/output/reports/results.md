# Model Performance Results

Generated on: 2026-07-19 02:41:57 BST

## Run Contract

- Evaluation lane: assisted
- Metadata exposed to prompt: yes
- Semantic rankings: grounded (metadata-assisted visual verification)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=6, mechanically clean
  outputs=7/61.
- _Useful now:_ 8 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 53 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=16, neutral=4, worse=41 (baseline B 74/100).
- _Quality signal frequency:_ low_draft_improvement=39, missing_sections=24,
  description_length=15, title_length=15, keyword_count=13, cutoff=11.

### Runtime

- _Runtime pattern:_ upstream model prefill / first-token dominates measured
  phase time (58%; 18/62 measured model(s)).
- _Phase totals:_ model load=146.38s, local prompt prep=0.24s, upstream model
  prefill / first-token=729.98s, input preparation + decode=367.26s,
  cleanup=7.75s.
- _Generation total:_ 1097.24s across 61 model(s); upstream model prefill /
  first-token split available for 61/61 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=50, exception=1, max_tokens=11.
- _Validation overhead:_ 18.54s total (avg 0.30s across 62 model(s)).
- _Upstream model prefill / first-token time:_ Avg 11.97s | Min 0.03s | Max
  74.14s across 61 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/Qwen3.5-9B-MLX-4bit` (92.7 tps)
- **💾 Most efficient:** `mlx-community/Qwen3.5-9B-MLX-4bit` (11.5 GB)
- **⚡ Fastest load:** `mlx-community/Qwen3.5-9B-MLX-4bit` (1.49s)
- **📊 Average TPS:** 896.9 across 61 models

## 📈 Resource Usage

- **Input image size:** 56.56 MP
- **Average peak delta from post-load:** 4.58 GB
- **Peak memory delta / MP:** 83 MB/MP
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 255 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 13 | ✅ B: 12 | 🟡 C: 19 | 🟠 D: 10 | ❌ F: 7

**Average Utility Score:** 59/100

**Existing Metadata Baseline:** ✅ B (74/100)
**Vs Existing Metadata:** Avg Δ -15 | Better: 16, Neutral: 4, Worse: 41

- **Best for cataloging:** `mlx-community/Qwen3.5-27B-4bit` (🏆 A, 92/100)
- **Best descriptions:** `mlx-community/Qwen3.5-27B-mxfp8` (93/100)
- **Best keywording:** `mlx-community/Qwen3.6-27B-mxfp8` (83/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 0/100)

### ⚠️ 17 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (36/100) - Lacks visual description of image
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (37/100) - Keywords are not specific or diverse enough
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (5/100) - Output lacks detail
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (36/100) - Lacks visual description of image
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: 🟠 D (38/100) - Lacks visual description of image
- `mlx-community/X-Reasoner-7B-8bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (36/100) - Lacks visual description of image
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (40/100) - Lacks visual description of image
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (14/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (14/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: 🟠 D (47/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-8bit`: 🟠 D (44/100) - Lacks visual description of image
- `qnguyen3/nanoLLaVA`: 🟠 D (44/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/Step-3.7-Flash-oQ2e` (`Processor Error`)
- **🔄 Repetitive Output (6):**
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `-`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (token: `phrase: "the fenchurch building, london..."`)
  - `mlx-community/Qwen3-VL-2B-Instruct-bf16` (token: `-`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `-`)
  - `mlx-community/X-Reasoner-7B-8bit` (token: `<|im_start|>`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "do not, and do..."`)
- **👻 Hallucinations (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit`
  - `mlx-community/X-Reasoner-7B-8bit`
- **📝 Formatting Issues (4):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/MiniCPM-V-4.6-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 897 | Min: 4.68 | Max: 49,588
- **Peak Memory**: Avg: 21 | Min: 1.5 | Max: 78
- **Total Time**: Avg: 20.59s | Min: 1.33s | Max: 85.77s
- **Generation Time**: Avg: 17.99s | Min: 0.67s | Max: 82.28s
- **Model Load Time**: Avg: 2.29s | Min: 0.38s | Max: 9.96s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.5-27B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-4bit)
  (Utility A 92/100 | Description 90 | Keywords 75 | Speed 30.7 TPS | Memory
  26 | Caveat missing terms: Cars, Cityscape, Commuting, Modern, Nightscape)
- _Best descriptions:_ [`mlx-community/Qwen3.5-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-mxfp8)
  (Utility A 87/100 | Description 93 | Keywords 59 | Speed 17.4 TPS | Memory
  38 | Caveat missing terms: Cars, Commuting, The Fenchurch Building (The
  Walki..., Walkie Talkie building, GBR)
- _Best keywording:_ [`mlx-community/Qwen3.6-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen36-27b-mxfp8)
  (Utility A 86/100 | Description 92 | Keywords 83 | Speed 17.8 TPS | Memory
  38 | Caveat missing terms: Cars, City, Commuting, Nightscape, Street signs)
- _Fastest generation:_ [`mlx-community/Qwen3.5-9B-MLX-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit)
  (Utility B 66/100 | Description 67 | Keywords 58 | Speed 92.7 TPS | Memory
  11 | Caveat missing terms: Cars, Commuting, Fenchurch Street, The Fenchurch
  Building (The Walki..., GBR)
- _Lowest memory footprint:_ [`mlx-community/Qwen3.5-9B-MLX-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit)
  (Utility B 66/100 | Description 67 | Keywords 58 | Speed 92.7 TPS | Memory
  11 | Caveat missing terms: Cars, Commuting, Fenchurch Street, The Fenchurch
  Building (The Walki..., GBR)
- _Best balance:_ [`mlx-community/Qwen3.5-27B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-4bit)
  (Utility A 92/100 | Description 90 | Keywords 75 | Speed 30.7 TPS | Memory
  26 | Caveat missing terms: Cars, Cityscape, Commuting, Modern, Nightscape)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/Step-3.7-Flash-oQ2e`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-step-37-flash-oq2e).
  Example: `Processor Error`.
- _🔄 Repetitive Output (6):_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen3-vl-2b-instruct-bf16),
  [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen3-vl-2b-thinking-bf16),
  +2 more. Example: token: `-`.
- _👻 Hallucinations (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/X-Reasoner-7B-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-x-reasoner-7b-8bit).
- _📝 Formatting Issues (4):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-diffusiongemma-26b-a4b-it-8bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8).
- _Low-utility outputs (17):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +13 more. Common weakness: Lacks visual description of image.

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types     | Affected Models                     |
|----------------|------------|-----------------|-------------------------------------|
| `model-config` |          1 | Processor Error | `mlx-community/Step-3.7-Flash-oQ2e` |

### Actionable Items by Package

#### model-config

- mlx-community/Step-3.7-Flash-oQ2e (Processor Error)
  - Error: `ValueError: Loaded processor has no image_processor; expected multimodal processor.`
  - Suspected owner: `model-config` (medium)

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
> Context: Authoritative context:
> &#45; Location terms: England, Europe, UK, district, united kingdom
> &#45; Capture date/time: 2026-07-18 22:55:39 BST 22:55:39
> &#45; GPS: 51.511300°N, 0.083400°W
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
>
> &#8203;Draft descriptive metadata:
> &#45; Existing title: The Fenchurch Building (The Walkie-Talkie), London,
> England, UK, GBR, Europe
> &#45; Existing description: Walkie Talkie building known formally as 20
> Fenchurch Street.
> &#45; Existing keywords: Architecture, Building, Buildings, Cars, City,
> Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night,
> Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The
> Walki..., Urban, Urban landscape, Walkie Talkie building
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1273.11s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |                                        Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     | Error Package   |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|-------------------------------------------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|:----------------|
| `mlx-community/Step-3.7-Flash-oQ2e`                     |                   |                       |                |           |                                                  |                 |                  |      6.51s |       8.34s |                                    | model-config    |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               675 |                   154 |            829 |       508 |  1.5 GB (1.3% of 108 GB recommended working set) | stop            |            0.67s |      0.38s |       1.33s | description-sentences(3)           |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               605 |                    99 |            704 |       354 |  2.8 GB (2.4% of 108 GB recommended working set) | stop            |            0.87s |      0.56s |       1.73s | title-length(11), ...              |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               865 |                   172 |          1,037 |       328 | 3.0 GB (2.58% of 108 GB recommended working set) | stop            |            1.00s |      0.65s |       1.95s | title-length(11), ...              |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |             1,223 |                    87 |          1,310 |       274 | 4.1 GB (3.58% of 108 GB recommended working set) | stop            |            1.16s |      0.92s |       2.40s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               883 |                    86 |            969 |       124 |  17 GB (14.3% of 108 GB recommended working set) | stop            |            1.69s |      2.41s |       4.43s | low-draft-improvement              |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               865 |                   234 |          1,099 |       187 | 4.1 GB (3.56% of 108 GB recommended working set) | stop            |            1.73s |      0.58s |       2.61s | title-length(35), ...              |                 |
| `qnguyen3/nanoLLaVA`                                    |               605 |                   129 |            734 |       112 |  4.7 GB (4.1% of 108 GB recommended working set) | stop            |            1.77s |      0.57s |       2.65s | missing-sections(keywords), ...    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               609 |                   214 |            823 |       344 | 2.2 GB (1.88% of 108 GB recommended working set) | stop            |            1.84s |      0.66s |       2.81s | missing-sections(keywords), ...    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               722 |                   120 |            842 |       134 | 5.5 GB (4.75% of 108 GB recommended working set) | stop            |            1.87s |      1.00s |       3.18s | title-length(11), ...              |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,822 |                   115 |          1,937 |       128 | 5.7 GB (4.95% of 108 GB recommended working set) | stop            |            1.90s |      0.77s |       2.95s | title-length(11), ...              |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,822 |                   115 |          1,937 |       126 | 5.7 GB (4.97% of 108 GB recommended working set) | stop            |            1.96s |      0.65s |       2.91s | title-length(11), ...              |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,217 |                   134 |          3,351 |       185 | 7.8 GB (6.76% of 108 GB recommended working set) | stop            |            2.22s |      0.98s |       3.52s | description-sentences(3)           |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               879 |                    95 |            974 |      75.7 |  28 GB (24.6% of 108 GB recommended working set) | stop            |            2.30s |      3.23s |       5.84s | missing-sections(title), ...       |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,630 |                    18 |          1,648 |      30.9 |  12 GB (10.5% of 108 GB recommended working set) | stop            |            2.31s |      1.59s |       4.21s | missing-sections(title+desc...     |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               879 |                    91 |            970 |      60.8 |  29 GB (25.3% of 108 GB recommended working set) | stop            |            2.54s |      3.29s |       6.17s | missing-sections(title), ...       |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,887 |                    26 |          2,913 |      33.1 |  19 GB (16.6% of 108 GB recommended working set) | stop            |            2.56s |      1.94s |       4.82s | missing-sections(title+desc...     |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,436 |                   171 |          1,607 |      56.2 | 9.6 GB (8.28% of 108 GB recommended working set) | stop            |            3.85s |      1.03s |       5.18s | description-sentences(3), ...      |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,436 |                   171 |          1,607 |      56.1 | 9.6 GB (8.29% of 108 GB recommended working set) | stop            |            3.86s |      0.92s |       5.10s | description-sentences(3), ...      |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,388 |                    92 |          2,480 |      34.2 |  18 GB (15.3% of 108 GB recommended working set) | stop            |            4.09s |      1.63s |       6.02s | low-draft-improvement              |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,218 |                   106 |          3,324 |      66.6 |  13 GB (11.3% of 108 GB recommended working set) | stop            |            4.30s |      1.33s |       5.93s | trusted-hints-degraded             |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,411 |                    74 |          3,485 |      39.4 |  16 GB (13.6% of 108 GB recommended working set) | stop            |            4.41s |      1.71s |       6.45s | title-length(4), ...               |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,218 |                   121 |          3,339 |      63.1 |  13 GB (11.7% of 108 GB recommended working set) | stop            |            4.63s |      1.46s |       6.39s | trusted-hints-degraded             |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               869 |                   500 |          1,369 |       125 | 6.1 GB (5.25% of 108 GB recommended working set) | length          |            4.78s |      1.62s |       6.71s | repetitive(phrase: "do not,...     |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,834 |                   124 |          2,958 |        62 | 9.7 GB (8.41% of 108 GB recommended working set) | stop            |            4.79s |      1.01s |       6.10s | title-length(11), ...              |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,388 |                    87 |          2,475 |      32.3 |  18 GB (15.7% of 108 GB recommended working set) | stop            |            4.91s |      1.65s |       6.86s | title-length(4), ...               |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,630 |                    17 |          1,647 |      5.56 |  27 GB (23.3% of 108 GB recommended working set) | stop            |            5.07s |      2.56s |       7.94s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               878 |                   104 |            982 |      31.3 |  18 GB (15.7% of 108 GB recommended working set) | stop            |            5.10s |      2.47s |       7.87s | low-draft-improvement              |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               883 |                    92 |            975 |      26.5 |  20 GB (17.6% of 108 GB recommended working set) | stop            |            5.51s |      2.61s |       8.42s | low-draft-improvement              |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,598 |                   302 |          1,900 |      71.5 |  18 GB (15.6% of 108 GB recommended working set) | stop            |            5.75s |      2.09s |       8.15s | missing-sections(title), ...       |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,411 |                    84 |          3,495 |        20 |  28 GB (23.9% of 108 GB recommended working set) | stop            |            6.42s |      2.53s |       9.25s | title-length(4), ...               |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               878 |                    96 |            974 |      17.8 |  32 GB (27.4% of 108 GB recommended working set) | stop            |            7.37s |      3.31s |      10.98s | low-draft-improvement              |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,685 |                   112 |          2,797 |      30.1 |  23 GB (19.9% of 108 GB recommended working set) | stop            |            7.83s |      2.20s |      10.34s | low-draft-improvement              |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,627 |                   128 |          6,755 |      57.9 |  11 GB (9.58% of 108 GB recommended working set) | stop            |            8.01s |      1.40s |       9.72s | hallucination, ...                 |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               877 |                   346 |          1,223 |      48.5 |  17 GB (14.9% of 108 GB recommended working set) | stop            |            8.04s |      2.33s |      10.69s | fabrication, ...                   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               575 |                   147 |            722 |      21.9 |  15 GB (13.1% of 108 GB recommended working set) | stop            |            8.80s |      2.02s |      11.13s | missing-sections(title+desc...     |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,598 |                   500 |          2,098 |      66.2 |    22 GB (19% of 108 GB recommended working set) | length          |            9.07s |      2.32s |      11.69s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,627 |                   500 |          7,127 |      80.1 | 8.4 GB (7.29% of 108 GB recommended working set) | length          |           12.14s |      1.32s |      13.75s | repetitive(phrase: "the fen...     |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               576 |                    51 |            627 |      5.08 |  25 GB (21.9% of 108 GB recommended working set) | stop            |           12.87s |      2.27s |      15.44s | missing-sections(title+desc...     |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,702 |                   500 |          5,202 |      43.4 | 4.6 GB (3.96% of 108 GB recommended working set) | length          |           13.48s |      1.26s |      15.06s | missing-sections(title+desc...     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,502 |                   500 |          4,002 |      42.5 |    15 GB (13% of 108 GB recommended working set) | length          |           14.48s |      1.67s |      16.46s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,927 |                   500 |          2,427 |      46.5 |  60 GB (52.1% of 108 GB recommended working set) | length          |           17.58s |      9.41s |      27.31s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               871 |                   106 |            977 |      7.39 |    65 GB (56% of 108 GB recommended working set) | stop            |           18.62s |      6.78s |      25.71s | missing-sections(title+desc...     |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,627 |                   112 |          6,739 |      43.1 |  78 GB (67.3% of 108 GB recommended working set) | stop            |           19.28s |      9.25s |      28.84s | hallucination, ...                 |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |            16,822 |                   500 |         17,322 |      92.9 | 8.6 GB (7.43% of 108 GB recommended working set) | length          |           19.60s |      0.74s |      20.63s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |            16,822 |                   500 |         17,322 |      92.8 | 8.6 GB (7.44% of 108 GB recommended working set) | length          |           21.53s |      0.82s |      22.65s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             3,420 |                   108 |          3,528 |      5.96 |  27 GB (23.1% of 108 GB recommended working set) | stop            |           21.54s |      2.31s |      24.17s | description-sentences(3), ...      |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |            16,824 |                   500 |         17,324 |      92.7 | 8.6 GB (7.44% of 108 GB recommended working set) | length          |           22.30s |      0.81s |      23.42s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,792 |                   140 |          1,932 |      51.3 |  41 GB (35.2% of 108 GB recommended working set) | stop            |           22.91s |      1.26s |      24.48s | title-length(11), ...              |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,792 |                   140 |          1,932 |      29.5 |  48 GB (41.4% of 108 GB recommended working set) | stop            |           25.38s |      1.78s |      27.46s | title-length(11), ...              |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,833 |                   500 |         17,333 |      56.1 |  13 GB (11.7% of 108 GB recommended working set) | length          |           27.21s |      1.17s |      28.68s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,630 |                   500 |          2,130 |        19 |  11 GB (9.69% of 108 GB recommended working set) | length          |           27.37s |      1.50s |      29.18s | missing-sections(title+desc...     |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |            16,852 |                   120 |         16,972 |      63.3 |  76 GB (65.8% of 108 GB recommended working set) | stop            |           53.89s |      9.96s |      64.17s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,852 |                   110 |         16,962 |      92.7 |  11 GB (9.94% of 108 GB recommended working set) | stop            |           54.52s |      1.49s |      56.38s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,852 |                   114 |         16,966 |        65 |  76 GB (65.8% of 108 GB recommended working set) | stop            |           54.65s |      9.68s |      64.64s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,852 |                   111 |         16,963 |      90.1 |  26 GB (22.6% of 108 GB recommended working set) | stop            |           55.78s |      2.45s |      58.52s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,833 |                     1 |         16,834 |     49588 | 5.1 GB (4.39% of 108 GB recommended working set) | stop            |           59.27s |      0.60s |      60.18s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,852 |                   112 |         16,964 |      93.6 |  35 GB (30.1% of 108 GB recommended working set) | stop            |           61.14s |      3.14s |      64.58s | trusted-hints-degraded             |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,598 |                   341 |          1,939 |      4.68 |  40 GB (34.6% of 108 GB recommended working set) | stop            |           74.97s |      3.27s |      78.55s | missing-sections(title), ...       |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,852 |                   118 |         16,970 |      30.7 |  26 GB (22.4% of 108 GB recommended working set) | stop            |           77.89s |      2.26s |      80.47s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,852 |                   144 |         16,996 |      17.8 |  38 GB (33.3% of 108 GB recommended working set) | stop            |           81.56s |      3.13s |      85.01s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,852 |                   128 |         16,980 |      17.4 |  38 GB (33.3% of 108 GB recommended working set) | stop            |           82.28s |      3.17s |      85.77s |                                    |                 |

<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->

_Companion artifacts:_

- _Model-selection shortlist:_ [model_selection.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_selection.md)
- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.2
- _SDK Version:_ 26.5
- _Active Developer Directory:_ /Applications/Xcode.app/Contents/Developer
- _Xcode Version:_ 26.6
- _Xcode Build:_ 17F113
- _SDK Path:_ /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
- _Metal SDK:_ MacOSX26.5.sdk
- _Metal Compiler Version:_ Apple metal version 32023.883 (metalfe-32023.883)
- _Metallib Linker Version:_ AIR-LLD 32023.883 (metalfe-32023.883) (compatible
  with legacy metallib linker)
- _Apple Clang Version:_ Apple clang version 21.0.0 (clang-2100.1.1.101)
- _Python Version:_ 3.13.13
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _MLX Device:_ Apple M5 Max
- _GPU Architecture:_ applegpu_g17s
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _Recommended Working Set:_ 108 GB
- _Fused Attention:_ Available
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18
- _MLX Install Type:_ editable local source
- _MLX Distribution Root:_ ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages
- _mlx-metal Distribution:_ not installed; local editable mlx supplies backend
- _MLX Core Extension:_ ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so
- _MLX Metallib:_ ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib
  (162,449,848 bytes,
  sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb)
- _MLX libmlx.dylib:_ ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,697,568 bytes,
  sha256=107989d6cb7f822e366d22e8259ea8eb2e68c15271ab6dc981da277ecf282cb0)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.1.dev20260719+b7c3dd6d`
- `mlx-vlm`: `0.6.5`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.24.0`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-19 02:41:57 BST
