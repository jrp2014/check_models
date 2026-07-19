# Model Performance Results

Generated on: 2026-07-19 20:37:23 BST

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
- _Indeterminate attempts:_ 1 (external connectivity prevented evaluation;
  retry).
- _Maintainer signals:_ harness-risk successes=5, mechanically clean
  outputs=9/60.
- _Useful now:_ 8 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 51 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=12, neutral=4, worse=44 (baseline B 74/100).
- _Quality signal frequency:_ low_draft_improvement=42, missing_sections=24,
  keyword_count=19, low_metadata_alignment=18, title_length=13, cutoff=13.

### Runtime

- _Runtime pattern:_ upstream model prefill / first-token dominates measured
  phase time (60%; 19/62 measured model(s)).
- _Phase totals:_ model load=159.11s, local prompt prep=0.30s, upstream model
  prefill / first-token=1069.97s, input preparation + decode=538.21s,
  cleanup=8.18s.
- _Generation total:_ 1608.18s across 60 model(s); upstream model prefill /
  first-token split available for 60/60 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=47, exception=2, max_tokens=13.
- _Validation overhead:_ 20.50s total (avg 0.33s across 62 model(s)).
- _Upstream model prefill / first-token time:_ Avg 17.83s | Min 0.03s | Max
  247.07s across 60 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/gemma-4-26b-a4b-it-4bit` (123.2 tps)
- **💾 Most efficient:** `mlx-community/GLM-4.6V-Flash-mxfp4` (8.6 GB)
- **⚡ Fastest load:** `mlx-community/GLM-4.6V-Flash-mxfp4` (1.35s)
- **📊 Average TPS:** 80.2 across 60 models

## 📈 Resource Usage

- **Input image size:** 56.56 MP
- **Average peak delta from post-load:** 4.14 GB
- **Peak memory delta / MP:** 75 MB/MP
- **Average peak memory:** 20.8 GB
- **Memory efficiency:** 263 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 8 | ✅ B: 18 | 🟡 C: 14 | 🟠 D: 10 | ❌ F: 10

**Average Utility Score:** 55/100

**Existing Metadata Baseline:** ✅ B (74/100)
**Vs Existing Metadata:** Avg Δ -19 | Better: 12, Neutral: 4, Worse: 44

- **Best for cataloging:** `mlx-community/Ornith-1.0-35B-bf16` (🏆 A, 91/100)
- **Best descriptions:** `mlx-community/Qwen3.5-27B-4bit` (99/100)
- **Best keywording:** `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` (86/100)
- **Worst for cataloging:** `mlx-community/Qwen3-VL-2B-Thinking-bf16` (❌ F, 0/100)

### ⚠️ 20 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (6/100) - Output lacks detail
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (1/100) - Output lacks detail
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/FastVLM-0.5B-bf16`: 🟠 D (42/100) - Keywords are not specific or diverse enough
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (5/100) - Output lacks detail
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (45/100) - Lacks visual description of image
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (6/100) - Output lacks detail
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (48/100) - Lacks visual description of image
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (36/100) - Lacks visual description of image
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (14/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (14/100) - Output lacks detail
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-8bit`: 🟠 D (48/100) - Limited novel information
- `mlx-community/pixtral-12b-bf16`: 🟠 D (43/100) - Lacks visual description of image
- `qnguyen3/nanoLLaVA`: 🟠 D (47/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/Step-3.7-Flash-oQ2e` (`Processor Error`)
- **⚠️ Indeterminate Attempts (1):**
  - `mlx-community/Molmo-7B-D-0924-8bit` (`Network Error`)
- **🔄 Repetitive Output (7):**
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `-`)
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "modern, illuminated, architect..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "modern, illuminated, architect..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "22:55:39 bst 22:55:39 bst..."`)
  - `mlx-community/Qwen3-VL-2B-Instruct-bf16` (token: `-`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "- do not take..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- do not output..."`)
- **👻 Hallucinations (2):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-nvfp4`
- **📝 Formatting Issues (4):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/MiniCPM-V-4.6-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 80.2 | Min: 1.83 | Max: 509
- **Peak Memory**: Avg: 21 | Min: 1.5 | Max: 78
- **Total Time**: Avg: 29.64s | Min: 1.47s | Max: 250.86s
- **Generation Time**: Avg: 26.80s | Min: 0.75s | Max: 250.01s
- **Model Load Time**: Avg: 2.49s | Min: 0.33s | Max: 14.81s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.5-35B-A3B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16)
  (Utility A 90/100 | Description 78 | Keywords 74 | Speed 65.8 TPS | Memory
  76 | Caveat missing terms: Cars, City, Commuting, Modern, The Fenchurch
  Building (The Walkie-Talkie))
- _Best descriptions:_ [`mlx-community/Qwen3.5-27B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-4bit)
  (Utility A 81/100 | Description 99 | Keywords 75 | Speed 30.7 TPS | Memory
  26 | Caveat missing terms: Building, Buildings, Cars, City, Commuting)
- _Best keywording:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility B 79/100 | Description 67 | Keywords 80 | Speed 30.9 TPS | Memory
  18 | Caveat missing terms: Cars, Commuting, GBR; keywords=23;
  low-draft-improvement)
- _Fastest generation:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (Utility A 88/100 | Description 92 | Keywords 58 | Speed 123 TPS | Memory 17
  | Caveat missing terms: Cars, City, Commuting, Street signs, The Fenchurch
  Building (The Walkie-Talkie))
- _Lowest memory footprint:_ [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4)
  (Utility C 64/100 | Description 67 | Keywords 59 | Speed 80.0 TPS | Memory
  8.6 | Caveat missing terms: Commuting, Fenchurch Street, Illuminated, GBR,
  known)
- _Best balance:_ [`mlx-community/Qwen3.5-35B-A3B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16)
  (Utility A 90/100 | Description 78 | Keywords 74 | Speed 65.8 TPS | Memory
  76 | Caveat missing terms: Cars, City, Commuting, Modern, The Fenchurch
  Building (The Walkie-Talkie))

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/Step-3.7-Flash-oQ2e`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-step-37-flash-oq2e).
  Example: `Processor Error`.
- _⚠️ Indeterminate Attempts (1):_ [`mlx-community/Molmo-7B-D-0924-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmo-7b-d-0924-8bit).
  Example: `Network Error`.
- _🔄 Repetitive Output (7):_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Phi-3.5-vision-instruct-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-phi-35-vision-instruct-bf16),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  +3 more. Example: token: `-`.
- _👻 Hallucinations (2):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4).
- _📝 Formatting Issues (4):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-diffusiongemma-26b-a4b-it-8bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8).
- _Low-utility outputs (20):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  +16 more. Common weakness: Output lacks detail.

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
> Describe visible details faithfully. If a visual detail is uncertain,
> ambiguous, partially obscured, or too small to verify, leave it out rather
> than guessing.
>
> Use authoritative context as supplied fact, and treat the descriptive
> metadata as a draft catalog record. Retain draft details that are consistent
> with the image, correct contradictions, and add important visible details.
> Authoritative context may supply identity and location even when they are
> not visually readable.
>
> &#8203;Return exactly these three sections, and nothing else:
>
> &#8203;Title:
> &#45; 5-10 words, concrete and factual; authoritative context may supply
> identity and location.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> &#8203;Description:
> &#45; 1-2 factual sentences combining supplied authoritative context with the
> main visible subject, setting, lighting, action, and distinctive visible
> details.
> &#45; Output only the description text after the label.
>
> &#8203;Keywords:
> &#45; 10-18 unique comma-separated terms covering supplied authoritative context
> and clearly visible subjects, setting, colors, composition, and style.
> &#45; Output only the keyword list after the label.
>
> &#8203;Rules:
> &#45; Distinguish supplied authoritative facts from visible details; do not
> present contextual facts as though they were read from the image.
> &#45; Reuse draft metadata when it is consistent with the image; authoritative
> context does not require separate visual proof.
> &#45; If metadata and image disagree, follow the image.
> &#45; Prefer omission to speculation.
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent unless supplied as authoritative context or visually obvious.
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
> Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1799.19s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |                                        Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     | Error Package   |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|-------------------------------------------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|:----------------|
| `mlx-community/Molmo-7B-D-0924-8bit`                    |                   |                       |                |           |                                                  |                 |                  |      0.01s |       0.29s |                                    | unknown         |
| `mlx-community/Step-3.7-Flash-oQ2e`                     |                   |                       |                |           |                                                  |                 |                  |      9.55s |      11.39s |                                    | model-config    |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               700 |                   188 |            888 |       509 | 1.5 GB (1.31% of 108 GB recommended working set) | stop            |            0.75s |      0.44s |       1.47s | description-sentences(4), ...      |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               625 |                   112 |            737 |       346 | 2.8 GB (2.42% of 108 GB recommended working set) | stop            |            0.91s |      1.04s |       2.25s | title-length(11), ...              |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               890 |                   201 |          1,091 |       331 | 3.0 GB (2.58% of 108 GB recommended working set) | stop            |            1.07s |      0.33s |       1.71s | title-length(11), ...              |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,843 |                    18 |          1,861 |       131 | 5.7 GB (4.96% of 108 GB recommended working set) | stop            |            1.13s |      0.74s |       2.15s | missing-sections(title+desc...     |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,843 |                    18 |          1,861 |       124 | 5.7 GB (4.97% of 108 GB recommended working set) | stop            |            1.20s |      1.45s |       2.95s | missing-sections(title+desc...     |                 |
| `qnguyen3/nanoLLaVA`                                    |               625 |                    91 |            716 |       113 | 4.7 GB (4.08% of 108 GB recommended working set) | stop            |            1.41s |      0.80s |       2.51s | low_metadata_alignment, ...        |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               629 |                   179 |            808 |       343 | 2.2 GB (1.88% of 108 GB recommended working set) | stop            |            1.55s |      0.61s |       2.47s | missing-sections(keywords), ...    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               905 |                   101 |          1,006 |       123 |  17 GB (14.3% of 108 GB recommended working set) | stop            |            1.82s |      2.95s |       5.10s |                                    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               743 |                   158 |            901 |       132 | 5.5 GB (4.75% of 108 GB recommended working set) | stop            |            2.18s |      0.78s |       3.26s | title-length(11), ...              |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,652 |                    17 |          1,669 |        31 |  12 GB (10.5% of 108 GB recommended working set) | stop            |            2.29s |      1.72s |       4.32s | low_metadata_alignment, ...        |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,239 |                   151 |          3,390 |       185 | 7.8 GB (6.76% of 108 GB recommended working set) | stop            |            2.49s |      0.92s |       3.70s | keyword-count(22)                  |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               901 |                   104 |          1,005 |      66.5 |  29 GB (25.4% of 108 GB recommended working set) | stop            |            2.61s |      3.70s |       6.65s | low_metadata_alignment, ...        |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,907 |                    26 |          2,933 |      33.4 |  19 GB (16.6% of 108 GB recommended working set) | stop            |            2.63s |      2.00s |       4.97s | missing-sections(title+desc...     |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               901 |                   100 |          1,001 |      62.7 |  28 GB (24.6% of 108 GB recommended working set) | stop            |            2.65s |      3.59s |       6.56s | missing-sections(title), ...       |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,408 |                    90 |          2,498 |      34.2 |  18 GB (15.3% of 108 GB recommended working set) | stop            |            4.05s |      1.64s |       5.99s | low-draft-improvement              |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               891 |                   500 |          1,391 |       123 | 6.0 GB (5.21% of 108 GB recommended working set) | length          |            4.84s |      2.07s |       7.22s | repetitive(phrase: "- do no...     |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               596 |                    14 |            610 |      5.39 |  25 GB (21.9% of 108 GB recommended working set) | stop            |            4.99s |      2.22s |       7.52s | missing-sections(title+desc...     |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,652 |                    17 |          1,669 |      5.54 |  27 GB (23.3% of 108 GB recommended working set) | stop            |            5.10s |      2.72s |       8.13s | low_metadata_alignment, ...        |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,867 |                   161 |          3,028 |      62.1 | 9.7 GB (8.41% of 108 GB recommended working set) | stop            |            5.37s |      1.42s |       7.10s | title-length(11), ...              |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,240 |                   158 |          3,398 |      62.9 |  13 GB (11.7% of 108 GB recommended working set) | stop            |            5.42s |      1.39s |       7.10s | low_metadata_alignment             |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,433 |                   112 |          3,545 |      39.2 |  16 GB (13.6% of 108 GB recommended working set) | stop            |            5.45s |      1.69s |       7.46s | title-length(4), ...               |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,408 |                   118 |          2,526 |      32.1 |  18 GB (15.7% of 108 GB recommended working set) | stop            |            5.99s |      1.72s |       8.02s | low-draft-improvement              |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               900 |                   142 |          1,042 |      30.9 |  18 GB (15.7% of 108 GB recommended working set) | stop            |            6.42s |      2.81s |       9.54s | keyword-count(23), ...             |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               905 |                   128 |          1,033 |      26.4 |  20 GB (17.7% of 108 GB recommended working set) | stop            |            7.04s |      3.02s |      10.38s | low-draft-improvement              |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,433 |                    99 |          3,532 |      20.1 |  28 GB (23.9% of 108 GB recommended working set) | stop            |            7.37s |      3.34s |      11.01s | title-length(4), ...               |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               899 |                   323 |          1,222 |      48.2 |  17 GB (14.9% of 108 GB recommended working set) | stop            |            7.61s |      2.78s |      10.71s | low_metadata_alignment, ...        |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,647 |                   134 |          6,781 |        80 | 8.6 GB (7.42% of 108 GB recommended working set) | stop            |            8.01s |      1.35s |       9.67s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |             1,243 |                    94 |          1,337 |      24.7 | 4.1 GB (3.53% of 108 GB recommended working set) | stop            |            8.34s |      0.96s |       9.75s | low_metadata_alignment, ...        |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,623 |                   500 |          2,123 |        71 |  18 GB (15.6% of 108 GB recommended working set) | length          |            8.55s |      1.99s |      10.86s | missing-sections(title+desc...     |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,707 |                   136 |          2,843 |      30.3 |  23 GB (19.9% of 108 GB recommended working set) | stop            |            8.58s |      2.17s |      11.06s | keyword-count(19), ...             |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,647 |                   152 |          6,799 |      57.7 |  11 GB (9.58% of 108 GB recommended working set) | stop            |            8.74s |      1.41s |      10.46s | hallucination, ...                 |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,623 |                   500 |          2,123 |      62.6 |    22 GB (19% of 108 GB recommended working set) | length          |            9.53s |      2.19s |      12.03s | fabrication, ...                   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,468 |                   500 |          1,968 |      57.2 |  9.6 GB (8.3% of 108 GB recommended working set) | length          |            9.55s |      0.92s |      10.77s | repetitive(phrase: "modern,...     |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               900 |                   140 |          1,040 |      17.4 |  32 GB (27.4% of 108 GB recommended working set) | stop            |           10.07s |      3.84s |      14.22s | keyword-count(21), ...             |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,956 |                   500 |          2,456 |      59.6 |  60 GB (52.1% of 108 GB recommended working set) | length          |           10.72s |      4.82s |      15.85s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,724 |                   500 |          5,224 |      42.4 | 4.6 GB (3.96% of 108 GB recommended working set) | length          |           13.75s |      1.72s |      15.79s | repetitive(phrase: "- do no...     |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |            16,844 |                     9 |         16,853 |       103 | 8.6 GB (7.44% of 108 GB recommended working set) | stop            |           14.30s |      0.73s |      15.34s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,524 |                   500 |          4,024 |      42.7 |    15 GB (13% of 108 GB recommended working set) | length          |           14.42s |      1.66s |      16.39s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,647 |                   146 |          6,793 |        43 |  78 GB (67.3% of 108 GB recommended working set) | stop            |           15.53s |      5.50s |      21.40s | hallucination, ...                 |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               893 |                    82 |            975 |       7.5 |    65 GB (56% of 108 GB recommended working set) | stop            |           16.27s |      7.58s |      24.16s | missing-sections(title+desc...     |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               890 |                   294 |          1,184 |        18 | 4.1 GB (3.56% of 108 GB recommended working set) | stop            |           18.23s |      1.59s |      20.72s | title-length(19), ...              |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |            16,842 |                   500 |         17,342 |      93.3 | 8.6 GB (7.43% of 108 GB recommended working set) | length          |           19.30s |      0.68s |      20.26s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,853 |                     9 |         16,862 |      64.1 |  13 GB (11.7% of 108 GB recommended working set) | stop            |           19.38s |      1.79s |      21.48s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,240 |                   122 |          3,362 |        58 |  13 GB (11.3% of 108 GB recommended working set) | stop            |           23.44s |      3.45s |      28.24s | low_metadata_alignment             |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,652 |                   500 |          2,152 |        19 |  11 GB (9.71% of 108 GB recommended working set) | length          |           27.34s |      2.01s |      29.66s | low_metadata_alignment, ...        |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               595 |                   187 |            782 |      11.5 |    15 GB (13% of 108 GB recommended working set) | stop            |           36.65s |      2.11s |      39.33s | title-length(11), ...              |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,468 |                   500 |          1,968 |      10.8 | 9.6 GB (8.31% of 108 GB recommended working set) | length          |           47.09s |      0.73s |      48.11s | repetitive(phrase: "modern,...     |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,812 |                   175 |          1,987 |      14.2 |  48 GB (41.4% of 108 GB recommended working set) | stop            |           51.15s |      2.14s |      53.58s | title-length(11), ...              |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,872 |                   137 |         17,009 |      92.1 |  11 GB (9.94% of 108 GB recommended working set) | stop            |           56.48s |      1.76s |      58.55s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |            16,842 |                   500 |         17,342 |      92.3 | 8.6 GB (7.44% of 108 GB recommended working set) | length          |           56.84s |      0.56s |      57.68s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,872 |                   133 |         17,005 |       107 |  26 GB (22.6% of 108 GB recommended working set) | stop            |           60.26s |      2.85s |      63.41s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,872 |                   132 |         17,004 |      91.2 |  35 GB (30.1% of 108 GB recommended working set) | stop            |           62.11s |      3.62s |      66.04s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,872 |                   131 |         17,003 |      65.8 |  76 GB (65.8% of 108 GB recommended working set) | stop            |           67.45s |     10.71s |      78.47s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,872 |                   136 |         17,008 |      30.7 |  26 GB (22.4% of 108 GB recommended working set) | stop            |           72.35s |      2.33s |      75.00s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,872 |                   141 |         17,013 |      18.3 |  38 GB (33.3% of 108 GB recommended working set) | stop            |           76.36s |      3.52s |      80.20s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,872 |                   144 |         17,016 |      17.7 |  38 GB (33.3% of 108 GB recommended working set) | stop            |           86.59s |      3.53s |      90.46s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             3,440 |                   155 |          3,595 |      1.83 |  27 GB (23.2% of 108 GB recommended working set) | stop            |           88.14s |      2.76s |      91.20s | keyword-count(20), ...             |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,623 |                   500 |          2,123 |      4.69 |  40 GB (34.7% of 108 GB recommended working set) | length          |          108.75s |      3.29s |     112.35s | missing-sections(title+desc...     |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |            16,872 |                   126 |         16,998 |      57.3 |  76 GB (65.8% of 108 GB recommended working set) | stop            |          129.56s |     14.81s |     144.77s | low_metadata_alignment             |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,853 |                   500 |         17,353 |       225 | 5.1 GB (4.39% of 108 GB recommended working set) | length          |          250.01s |      0.55s |     250.86s | ⚠️harness(long_context), ...       |                 |

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

Report generated on: 2026-07-19 20:37:23 BST
