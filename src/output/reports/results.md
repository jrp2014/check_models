# Model Performance Results

Generated on: 2026-07-21 08:38:01 BST

## Run Contract

- Evaluation lane: assisted
- Metadata exposed to prompt: yes
- Semantic rankings: grounded (metadata-assisted visual verification)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available
- Evidence scope: Automated metadata-assisted proxy; one image; no human visual ground truth.

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=5, mechanically clean
  outputs=10/61.
- _Useful now:_ 18 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 42 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=19, neutral=4, worse=38 (baseline B 73/100).
- _Quality signal frequency:_ low_draft_improvement=38, keyword_count=23,
  missing_sections=23, low_metadata_alignment=19, cutoff=12,
  generation_loop=9.

### Runtime

- _Runtime pattern:_ upstream model prefill / first-token dominates measured
  phase time (54%; 18/62 measured model(s)).
- _Phase totals:_ model load=129.13s, local prompt prep=0.23s, upstream model
  prefill / first-token=717.04s, input preparation + decode=463.32s,
  cleanup=7.28s.
- _Generation total:_ 1180.36s across 61 model(s); upstream model prefill /
  first-token split available for 61/61 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=49, exception=1, max_tokens=12.
- _Validation overhead:_ 17.26s total (avg 0.28s across 62 model(s)).
- _Upstream model prefill / first-token time:_ Avg 11.75s | Min 0.03s | Max
  75.18s across 61 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (505.6 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.5 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.34s)
- **📊 Average TPS:** 829.6 across 61 models

## 📈 Resource Usage

- **Input image size:** 56.07 MP
- **Average peak delta from post-load:** 4.49 GB
- **Peak memory delta / MP:** 82 MB/MP
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 256 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 12 | ✅ B: 18 | 🟡 C: 16 | 🟠 D: 8 | ❌ F: 7

**Average Utility Score:** 61/100

**Existing Metadata Baseline:** ✅ B (73/100)
**Vs Existing Metadata:** Avg Δ -11 | Better: 19, Neutral: 4, Worse: 38

- **Best for cataloging:** `mlx-community/Qwen3.5-27B-mxfp8` (🏆 A, 89/100)
- **Best descriptions:** `mlx-community/gemma-4-26b-a4b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (91/100)
- **Worst for cataloging:** `mlx-community/Molmo-7B-D-0924-bf16` (❌ F, 0/100)

### ⚠️ 15 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: 🟠 D (47/100) - Lacks visual description of image
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (47/100) - Lacks visual description of image
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (14/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (14/100) - Output lacks detail
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (47/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/Step-3.7-Flash-oQ2e` (`Processor Error`)
- **🔄 Repetitive Output (8):**
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `-`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (token: `phrase: "the ferris wheel is..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "the time is in..."`)
  - `mlx-community/Qwen3-VL-2B-Instruct-bf16` (token: `-`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "17:57:06 - 17:57:06 -..."`)
  - `mlx-community/gemma-4-31b-bf16` (token: `phrase: "a ferris wheel in..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the text of the..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- do not output..."`)
- **📝 Formatting Issues (4):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/MiniCPM-V-4.6-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 830 | Min: 4.67 | Max: 45,282
- **Peak Memory**: Avg: 21 | Min: 1.5 | Max: 78
- **Total Time**: Avg: 21.67s | Min: 1.18s | Max: 112.66s
- **Generation Time**: Avg: 19.35s | Min: 0.58s | Max: 109.09s
- **Model Load Time**: Avg: 2.03s | Min: 0.34s | Max: 7.59s

## ✅ Recommended Current-run Models

Only canonical `recommended` results are listed; see model_selection.md for
ranking.

- _Recommended:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (compatibility clean | speed 505.61 TPS | memory 1.50 GB)
- _Recommended:_ [`mlx-community/LFM2-VL-1.6B-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm2-vl-16b-8bit)
  (compatibility clean | speed 327.93 TPS | memory 2.97 GB)
- _Recommended:_ [`mlx-community/MiniCPM-V-4.6-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-minicpm-v-46-8bit)
  (compatibility clean | speed 275.03 TPS | memory 4.06 GB)
- _Recommended:_ [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-smolvlm2-22b-instruct-mlx)
  (compatibility clean | speed 134.65 TPS | memory 5.48 GB)
- _Recommended:_ [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (compatibility clean | speed 346.02 TPS | memory 2.17 GB)
- _Recommended:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (compatibility clean | speed 124.11 TPS | memory 16.55 GB)
- _Recommended:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (compatibility clean | speed 187.46 TPS | memory 6.41 GB)
- _Recommended:_ [`mlx-community/Phi-3.5-vision-instruct-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-phi-35-vision-instruct-bf16)
  (compatibility clean | speed 56.92 TPS | memory 9.59 GB)
- _Recommended:_ [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct)
  (compatibility clean | speed 56.69 TPS | memory 9.58 GB)
- _Recommended:_ [`mlx-community/InternVL3-8B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-internvl3-8b-bf16)
  (compatibility clean | speed 33.93 TPS | memory 18.09 GB)
- _Recommended:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (compatibility clean | speed 63.58 TPS | memory 12.09 GB)
- _Recommended:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (compatibility clean | speed 66.87 TPS | memory 11.66 GB)
- _Recommended:_ [`mlx-community/pixtral-12b-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-pixtral-12b-8bit)
  (compatibility clean | speed 39.60 TPS | memory 15.45 GB)
- _Recommended:_ [`mlx-community/gemma-4-31b-it-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-4-31b-it-4bit)
  (compatibility clean | speed 27.31 TPS | memory 20.36 GB)
- _Recommended:_ [`mlx-community/InternVL3-14B-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-internvl3-14b-8bit)
  (compatibility clean | speed 32.08 TPS | memory 18.52 GB)
- _Recommended:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (compatibility clean | speed 31.07 TPS | memory 18.13 GB)
- _Recommended:_ [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4)
  (compatibility clean | speed 80.58 TPS | memory 8.41 GB)
- _Recommended:_ [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit)
  (compatibility clean | speed 29.98 TPS | memory 21.62 GB)
- _Recommended:_ [`mlx-community/pixtral-12b-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-pixtral-12b-bf16)
  (compatibility clean | speed 20.22 TPS | memory 27.34 GB)
- _Recommended:_ [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-llama-32-11b-vision-instruct-8bit)
  (compatibility clean | speed 21.80 TPS | memory 15.07 GB)
- _Recommended:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit)
  (compatibility clean | speed 17.66 TPS | memory 31.64 GB)
- _Recommended:_ [`mlx-community/Molmo-7B-D-0924-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmo-7b-d-0924-8bit)
  (compatibility clean | speed 51.83 TPS | memory 40.61 GB)
- _Recommended:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (compatibility clean | speed 5.96 TPS | memory 26.68 GB)
- _Recommended:_ [`mlx-community/Ornith-1.0-35B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ornith-10-35b-bf16)
  (compatibility clean | speed 62.99 TPS | memory 75.94 GB)
- _Recommended:_ [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-4bit)
  (compatibility clean | speed 108.33 TPS | memory 26.12 GB)
- _Recommended:_ [`mlx-community/Qwen3.5-35B-A3B-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-6bit)
  (compatibility clean | speed 92.50 TPS | memory 34.78 GB)
- _Recommended:_ [`mlx-community/Qwen3.5-9B-MLX-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit)
  (compatibility clean | speed 93.02 TPS | memory 11.48 GB)
- _Recommended:_ [`mlx-community/Qwen3.5-35B-A3B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16)
  (compatibility clean | speed 66.51 TPS | memory 75.94 GB)
- _Recommended:_ [`mlx-community/Qwen3.5-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-mxfp8)
  (compatibility clean | speed 17.46 TPS | memory 38.45 GB)
- _Recommended:_ [`mlx-community/Qwen3.6-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen36-27b-mxfp8)
  (compatibility clean | speed 18.07 TPS | memory 38.46 GB)
- _Recommended:_ [`mlx-community/Qwen3.5-27B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-4bit)
  (compatibility clean | speed 31.03 TPS | memory 25.85 GB)

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
> &#45; Location terms: Adobe Stock, Any Vision, England, Europe, UK
> &#45; Capture date/time: 2026-07-18 17:57:06 BST 17:57:06
> &#45; GPS: 50.817441°N, 0.134547°W
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
>
> &#8203;Draft descriptive metadata:
> &#45; Existing title: Lifeboat Station, Southend-on-Sea, England, UK, GBR,
> Europe
> &#45; Existing description: A lifeboats station with a ferris wheel in the
> background.
> &#45; Existing keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag,
> Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast
> Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront,
> amusement ride, antenna
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1337.23s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |                                        Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     | Error Package   |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|-------------------------------------------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|:----------------|
| `mlx-community/Step-3.7-Flash-oQ2e`                     |                   |                       |                |           |                                                  |                 |                  |      5.39s |       7.19s |                                    | model-config    |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               686 |                   112 |            798 |       506 |  1.5 GB (1.3% of 108 GB recommended working set) | stop            |            0.58s |      0.34s |       1.18s | keyword-count(22), ...             |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               609 |                    37 |            646 |       338 |  2.9 GB (2.5% of 108 GB recommended working set) | stop            |            0.65s |      0.57s |       1.50s | low_metadata_alignment, ...        |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               872 |                   139 |          1,011 |       328 | 3.0 GB (2.58% of 108 GB recommended working set) | stop            |            0.88s |      0.53s |       1.69s | low_metadata_alignment, ...        |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |             1,229 |                    90 |          1,319 |       275 | 4.1 GB (3.52% of 108 GB recommended working set) | stop            |            1.16s |      0.89s |       2.35s | formatting                         |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,832 |                    40 |          1,872 |       127 | 5.7 GB (4.96% of 108 GB recommended working set) | stop            |            1.26s |      0.70s |       2.23s | low_metadata_alignment, ...        |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,832 |                    40 |          1,872 |       127 | 5.7 GB (4.97% of 108 GB recommended working set) | stop            |            1.29s |      0.63s |       2.20s | low_metadata_alignment, ...        |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               733 |                   106 |            839 |       135 | 5.5 GB (4.75% of 108 GB recommended working set) | stop            |            1.69s |      0.63s |       2.59s | keyword-count(20), ...             |                 |
| `qnguyen3/nanoLLaVA`                                    |               609 |                   111 |            720 |      87.4 | 4.9 GB (4.21% of 108 GB recommended working set) | stop            |            1.83s |      0.71s |       2.82s | missing-sections(keywords), ...    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               872 |                   270 |          1,142 |       190 | 4.1 GB (3.56% of 108 GB recommended working set) | stop            |            1.88s |      0.55s |       2.73s | description-sentences(7), ...      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               613 |                   284 |            897 |       346 | 2.2 GB (1.88% of 108 GB recommended working set) | stop            |            1.89s |      0.67s |       2.85s | keyword-count(56), ...             |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               895 |                   127 |          1,022 |       124 |  17 GB (14.3% of 108 GB recommended working set) | stop            |            1.97s |      2.42s |       4.69s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,873 |                   146 |          3,019 |       187 | 6.4 GB (5.55% of 108 GB recommended working set) | stop            |            2.11s |      0.94s |       3.34s | keyword-count(19)                  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,641 |                    17 |          1,658 |      31.7 |  12 GB (10.5% of 108 GB recommended working set) | stop            |            2.20s |      1.65s |       4.14s | low_metadata_alignment, ...        |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               891 |                   103 |            994 |      72.5 |  28 GB (24.6% of 108 GB recommended working set) | stop            |            2.42s |      3.20s |       5.90s | low_metadata_alignment, ...        |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               891 |                   111 |          1,002 |        66 |  29 GB (25.3% of 108 GB recommended working set) | stop            |            2.70s |      3.29s |       6.29s | low_metadata_alignment, ...        |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,468 |                   161 |          1,629 |      56.9 | 9.6 GB (8.31% of 108 GB recommended working set) | stop            |            3.61s |      0.93s |       4.84s | keyword-count(20), ...             |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,468 |                   161 |          1,629 |      56.7 |  9.6 GB (8.3% of 108 GB recommended working set) | stop            |            3.61s |      0.94s |       4.83s | keyword-count(20), ...             |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,904 |                   107 |          3,011 |      33.9 |  18 GB (15.7% of 108 GB recommended working set) | stop            |            4.70s |      1.75s |       6.73s | keyword-count(26), ...             |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               877 |                   500 |          1,377 |       125 | 6.1 GB (5.26% of 108 GB recommended working set) | length          |            4.73s |      1.47s |       6.49s | repetitive(phrase: "17:57:0...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,874 |                   148 |          3,022 |      63.6 |  12 GB (10.5% of 108 GB recommended working set) | stop            |            4.79s |      1.43s |       6.50s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,874 |                   164 |          3,038 |      66.9 |  12 GB (10.1% of 108 GB recommended working set) | stop            |            4.89s |      1.34s |       6.52s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,641 |                    17 |          1,658 |      5.56 |  27 GB (23.3% of 108 GB recommended working set) | stop            |            4.98s |      2.43s |       7.69s | low_metadata_alignment, ...        |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,945 |                   118 |          3,063 |      39.6 |  15 GB (13.4% of 108 GB recommended working set) | stop            |            5.01s |      1.69s |       6.98s | keyword-count(19), ...             |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               895 |                   106 |          1,001 |      27.3 |  20 GB (17.6% of 108 GB recommended working set) | stop            |            5.75s |      2.80s |       8.84s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,891 |                   135 |          3,026 |      31.7 |  19 GB (16.6% of 108 GB recommended working set) | stop            |            6.00s |      1.91s |       8.19s | keyword-count(20), ...             |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,904 |                   119 |          3,023 |      32.1 |    19 GB (16% of 108 GB recommended working set) | stop            |            6.24s |      1.71s |       8.23s | keyword-count(20), ...             |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,650 |                   259 |          2,909 |      62.2 | 9.7 GB (8.41% of 108 GB recommended working set) | stop            |            6.83s |      0.94s |       8.05s | keyword-count(40), ...             |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               886 |                   159 |          1,045 |      31.1 |  18 GB (15.7% of 108 GB recommended working set) | stop            |            6.86s |      2.32s |       9.47s | keyword-count(25)                  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,617 |                   109 |          6,726 |      80.6 | 8.4 GB (7.29% of 108 GB recommended working set) | stop            |            7.17s |      1.31s |       8.77s | keyword-count(21), ...             |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               885 |                   316 |          1,201 |      48.8 |  17 GB (14.9% of 108 GB recommended working set) | stop            |            7.34s |      2.27s |       9.90s | missing-sections(title+desc...     |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,341 |                   125 |          2,466 |        30 |  22 GB (18.7% of 108 GB recommended working set) | stop            |            7.81s |      2.16s |      10.27s | keyword-count(20), ...             |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,617 |                   121 |          6,738 |      57.9 |  11 GB (9.57% of 108 GB recommended working set) | stop            |            7.91s |      1.41s |       9.62s | low_metadata_alignment             |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,945 |                   123 |          3,068 |      20.2 |  27 GB (23.7% of 108 GB recommended working set) | stop            |            8.05s |      2.53s |      10.86s | keyword-count(20), ...             |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               580 |                   139 |            719 |      21.8 |  15 GB (13.1% of 108 GB recommended working set) | stop            |            8.45s |      1.51s |      10.29s | keyword-count(19)                  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,604 |                   500 |          2,104 |      71.3 |  18 GB (15.5% of 108 GB recommended working set) | length          |            8.48s |      2.01s |      10.79s | low_metadata_alignment, ...        |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,604 |                   500 |          2,104 |      66.1 |    22 GB (19% of 108 GB recommended working set) | length          |            9.02s |      2.16s |      11.47s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               886 |                   149 |          1,035 |      17.7 |  32 GB (27.4% of 108 GB recommended working set) | stop            |           10.35s |      3.30s |      13.93s | keyword-count(21)                  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,920 |                   500 |          2,420 |      54.4 |  60 GB (52.1% of 108 GB recommended working set) | length          |           11.31s |      4.97s |      16.57s | missing-sections(title+desc...     |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,713 |                   500 |          5,213 |      47.5 | 4.6 GB (3.96% of 108 GB recommended working set) | length          |           12.41s |      1.17s |      13.87s | repetitive(phrase: "- do no...     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,036 |                   500 |          3,536 |      42.4 |  15 GB (12.7% of 108 GB recommended working set) | length          |           14.25s |      1.58s |      16.11s | low_metadata_alignment, ...        |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,617 |                   126 |          6,743 |      43.4 |  78 GB (67.2% of 108 GB recommended working set) | stop            |           14.36s |      5.52s |      20.16s | keyword-count(20), ...             |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,933 |                     9 |         16,942 |      62.5 |  14 GB (11.7% of 108 GB recommended working set) | stop            |           18.21s |      1.19s |      19.67s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |            16,924 |                    90 |         17,014 |      93.9 | 8.6 GB (7.47% of 108 GB recommended working set) | stop            |           18.39s |      0.76s |      19.44s | missing-sections(title+desc...     |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |            16,922 |                   500 |         17,422 |      93.3 | 8.6 GB (7.46% of 108 GB recommended working set) | length          |           19.53s |      0.69s |      20.48s | ⚠️harness(long_context), ...       |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               581 |                    90 |            671 |       5.1 |  25 GB (21.9% of 108 GB recommended working set) | stop            |           20.00s |      2.22s |      22.51s | missing-sections(title+desc...     |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,788 |                     1 |          1,789 |     45282 |  48 GB (41.4% of 108 GB recommended working set) | stop            |           20.45s |      1.71s |      22.44s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,788 |                   144 |          1,932 |      51.8 |  41 GB (35.2% of 108 GB recommended working set) | stop            |           22.81s |      1.19s |      24.29s | description-sentences(4)           |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |            16,922 |                   500 |         17,422 |      91.4 | 8.6 GB (7.47% of 108 GB recommended working set) | length          |           22.93s |      0.76s |      23.98s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             3,405 |                   138 |          3,543 |      5.96 |  27 GB (23.1% of 108 GB recommended working set) | stop            |           26.48s |      2.25s |      29.02s | description-sentences(3), ...      |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,641 |                   500 |          2,141 |        19 |   11 GB (9.7% of 108 GB recommended working set) | length          |           27.29s |      1.46s |      29.04s | repetitive(phrase: "the tex...     |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |            16,952 |                   120 |         17,072 |        63 |  76 GB (65.8% of 108 GB recommended working set) | stop            |           52.41s |      7.22s |      59.93s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,952 |                   116 |         17,068 |       108 |  26 GB (22.6% of 108 GB recommended working set) | stop            |           53.98s |      2.51s |      56.79s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,952 |                   118 |         17,070 |      92.5 |  35 GB (30.1% of 108 GB recommended working set) | stop            |           54.90s |      3.33s |      58.52s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,952 |                   176 |         17,128 |        93 |  11 GB (9.94% of 108 GB recommended working set) | stop            |           56.33s |      1.36s |      57.98s | low_metadata_alignment, ...        |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,952 |                   114 |         17,066 |      66.5 |  76 GB (65.8% of 108 GB recommended working set) | stop            |           57.94s |      7.59s |      65.81s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,933 |                   500 |         17,433 |       222 | 5.1 GB (4.41% of 108 GB recommended working set) | length          |           67.08s |      0.64s |      68.01s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               883 |                   500 |          1,383 |      7.22 |    65 GB (56% of 108 GB recommended working set) | length          |           72.52s |      5.96s |      78.77s | repetitive(phrase: "a ferri...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,952 |                   158 |         17,110 |      17.5 |  38 GB (33.3% of 108 GB recommended working set) | stop            |           80.03s |      3.09s |      83.42s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,952 |                   127 |         17,079 |      18.1 |  38 GB (33.3% of 108 GB recommended working set) | stop            |           80.17s |      3.08s |      83.54s | keyword-count(19)                  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,952 |                   139 |         17,091 |        31 |  26 GB (22.4% of 108 GB recommended working set) | stop            |           80.41s |      2.17s |      82.88s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,604 |                   500 |          2,104 |      4.67 |  40 GB (34.7% of 108 GB recommended working set) | length          |          109.09s |      3.28s |     112.66s | repetitive(phrase: "the fer...     |                 |

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
  sha256=179f02b689129393feff8c019e656046eda1ba0b224e6fa58a83501dd9ce959b)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.1.dev20260721+30a19f72`
- `mlx-vlm`: `0.6.6`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.24.0`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-21 08:38:01 BST
