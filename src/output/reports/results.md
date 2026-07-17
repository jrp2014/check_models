# Model Performance Results

Generated on: 2026-07-17 13:44:05 BST

## Run Contract

- Evaluation lane: assisted
- Metadata exposed to prompt: yes
- Semantic rankings: grounded (metadata-assisted visual verification)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 1 (top owners: mlx-vlm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=6, mechanically clean
  outputs=1/60.
- _Useful now:_ 7 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 49 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=43, neutral=7, worse=10 (baseline D 49/100).
- _Quality signal frequency:_ low_draft_improvement=40, missing_sections=22,
  keyword_count=19, low_metadata_alignment=19, cutoff=14, context_ignored=11.

### Runtime

- _Runtime pattern:_ upstream model prefill / first-token dominates measured
  phase time (54%; 20/61 measured model(s)).
- _Phase totals:_ model load=155.19s, local prompt prep=0.26s, upstream model
  prefill / first-token=931.89s, input preparation + decode=499.26s,
  generation call total (unsplit)=117.72s, cleanup=10.00s.
- _Generation total:_ 1548.87s across 61 model(s); upstream model prefill /
  first-token split available for 60/61 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=45, exception=1, max_tokens=15.
- _Validation overhead:_ 34.70s total (avg 0.57s across 61 model(s)).
- _Upstream model prefill / first-token time:_ Avg 15.53s | Min 0.04s | Max
  119.39s across 60 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (503.6 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.4 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.38s)
- **📊 Average TPS:** 70.9 across 60 models

## 📈 Resource Usage

- **Input image size:** 57.50 MP
- **Average peak delta from post-load:** 4.64 GB
- **Peak memory delta / MP:** 83 MB/MP
- **Average peak memory:** 20.4 GB
- **Memory efficiency:** 266 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 11 | ✅ B: 17 | 🟡 C: 16 | 🟠 D: 7 | ❌ F: 9

**Average Utility Score:** 60/100

**Existing Metadata Baseline:** 🟠 D (49/100)
**Vs Existing Metadata:** Avg Δ +11 | Better: 43, Neutral: 7, Worse: 10

- **Best for cataloging:** `mlx-community/Qwen3.6-27B-mxfp8` (🏆 A, 85/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (93/100)
- **Best keywording:** `mlx-community/gemma-3-27b-it-qat-4bit` (80/100)
- **Worst for cataloging:** `mlx-community/X-Reasoner-7B-8bit` (❌ F, 4/100)

### ⚠️ 16 Models with Low Utility (D/F)

- `qnguyen3/nanoLLaVA`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (11/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (14/100) - Output lacks detail
- `mlx-community/LFM2.5-VL-1.6B-bf16`: ❌ F (34/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (14/100) - Output lacks detail
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (11/100) - Output lacks detail
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (48/100) - Keywords are not specific or diverse enough
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (49/100) - Lacks visual description of image
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/MolmoPoint-8B-fp16`: 🟠 D (48/100) - Lacks visual description of image
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: 🟠 D (47/100) - Keywords are not specific or diverse enough
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (50/100) - Lacks visual description of image

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/gemma-4-31b-bf16` (`Model Error`)
- **🔄 Repetitive Output (9):**
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "boating, boats, moored, buoy,..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "- do not take..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (token: `phrase: "black hull, white superstructu..."`)
  - `mlx-community/Qwen3-VL-2B-Instruct-bf16` (token: `-`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `-`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "- do not copy..."`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `-`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "boats, boats, boats, boats,..."`)
- **👻 Hallucinations (2):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-nvfp4`
- **📝 Formatting Issues (2):**
  - `mlx-community/MiniCPM-V-4.6-8bit`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 70.9 | Min: 3.35 | Max: 504
- **Peak Memory**: Avg: 20 | Min: 1.4 | Max: 78
- **Total Time**: Avg: 26.73s | Min: 1.55s | Max: 142.77s
- **Generation Time**: Avg: 23.85s | Min: 0.81s | Max: 137.78s
- **Model Load Time**: Avg: 2.29s | Min: 0.38s | Max: 6.62s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.6-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen36-27b-mxfp8)
  (Utility A 85/100 | Description 93 | Keywords 68 | Speed 6.35 TPS | Memory
  38 | Caveat missing terms: Bird, Boating, Bushes, Coast, Forest)
- _Best descriptions:_ [`mlx-community/Qwen3.6-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen36-27b-mxfp8)
  (Utility A 85/100 | Description 93 | Keywords 68 | Speed 6.35 TPS | Memory
  38 | Caveat missing terms: Bird, Boating, Bushes, Coast, Forest)
- _Best keywording:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility B 79/100 | Description 92 | Keywords 80 | Speed 18.9 TPS | Memory
  18 | Caveat missing terms: Bird, Bushes, Forest, Peaceful, Rigging;
  keywords=21; low-draft-improvement)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility B 76/100 | Description 90 | Keywords 77 | Speed 504 TPS | Memory
  1.4 | Caveat missing terms: Bird, Buoy, Mudflat, behind, bank)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility B 76/100 | Description 90 | Keywords 77 | Speed 504 TPS | Memory
  1.4 | Caveat missing terms: Bird, Buoy, Mudflat, behind, bank)
- _Best balance:_ [`mlx-community/Qwen3.6-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen36-27b-mxfp8)
  (Utility A 85/100 | Description 93 | Keywords 68 | Speed 6.35 TPS | Memory
  38 | Caveat missing terms: Bird, Boating, Bushes, Coast, Forest)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/gemma-4-31b-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-4-31b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (9):_ [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/gemma-3n-E2B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  [`mlx-community/paligemma2-3b-pt-896-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-paligemma2-3b-pt-896-4bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  +5 more. Example: token: `phrase: "boating, boats, moored, buoy,..."`.
- _👻 Hallucinations (2):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4).
- _📝 Formatting Issues (2):_ [`mlx-community/MiniCPM-V-4.6-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (16):_ [`qnguyen3/nanoLLaVA`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qnguyen3-nanollava),
  [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-paligemma2-10b-ft-docci-448-6bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  +12 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package   |   Failures | Error Types   | Affected Models                  |
|-----------|------------|---------------|----------------------------------|
| `mlx-vlm` |          1 | Model Error   | `mlx-community/gemma-4-31b-bf16` |

### Actionable Items by Package

#### mlx-vlm

- mlx-community/gemma-4-31b-bf16 (Model Error)
  - Error: `IndexError: list index out of range`
  - Suspected owner: `unresolved: mlx/mlx-vlm` (low)

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
> &#45; Location terms: Deben Estuary, England, Europe, UK, Woodbridge
> &#45; Capture date/time: 2026-07-04 19:10:04 BST 19:10:04
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
>
> &#8203;Draft descriptive metadata:
> &#45; Existing title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> &#45; Existing description: Two sailing boats moored on a river with trees
> behind on the bank
> &#45; Existing keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary,
> Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors,
> Peaceful, Rigging, River, Riverbank, Sailboat
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1750.69s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/gemma-4-31b-bf16`                        |                   |                       |                |           |             |                 |          117.72s |     17.87s |     135.96s |                                    |         mlx-vlm |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               639 |                   181 |            820 |       504 |         1.4 |                 |            0.81s |      0.38s |       1.55s | title-length(30), ...              |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               829 |                   144 |            973 |       331 |           3 |                 |            0.99s |      0.54s |       1.91s | title-length(17), ...              |                 |
| `qnguyen3/nanoLLaVA`                                    |               562 |                    37 |            599 |       113 |         5.1 |                 |            1.00s |      0.70s |       2.06s | missing-sections(keywords), ...    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               562 |                    98 |            660 |       192 |         2.8 |                 |            1.15s |      0.67s |       2.18s | keyword-count(21), ...             |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,776 |                    18 |          1,794 |       127 |         5.6 |                 |            1.22s |      0.68s |       2.27s | missing-sections(title+desc...     |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |             1,179 |                    81 |          1,260 |       181 |           4 |                 |            1.36s |      0.94s |       2.69s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               566 |                   171 |            737 |       292 |         2.2 |                 |            1.70s |      0.87s |       2.95s | keyword-count(28), ...             |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,582 |                    17 |          1,599 |      30.5 |          12 |                 |            2.27s |      1.62s |       4.25s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               835 |                    86 |            921 |      56.7 |          18 |                 |            2.69s |      2.51s |       5.57s | low_metadata_alignment             |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,394 |                   130 |          1,524 |      57.6 |         9.5 |                 |            3.08s |      1.21s |       4.65s | low_metadata_alignment, ...        |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               829 |                   500 |          1,329 |       175 |         4.1 |                 |            3.41s |      0.59s |       4.37s | repetitive(phrase: "boating...     |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,345 |                    81 |          2,426 |      35.3 |          18 |                 |            3.79s |      1.66s |       5.82s | low_metadata_alignment, ...        |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,173 |                   105 |          3,278 |      66.1 |          13 |                 |            4.38s |      1.51s |       6.26s | low_metadata_alignment             |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,366 |                    92 |          3,458 |      39.4 |          16 |                 |            4.67s |      1.99s |       7.02s | low_metadata_alignment, ...        |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,582 |                    17 |          1,599 |      5.63 |          26 |                 |            4.98s |      2.47s |       7.83s | missing-sections(title+desc...     |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,855 |                   109 |          2,964 |      33.3 |          19 |                 |            5.12s |      2.14s |       7.64s | title-length(11), ...              |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               835 |                    89 |            924 |      20.6 |          29 |                 |            5.45s |      3.46s |       9.35s | low-draft-improvement              |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,172 |                   125 |          3,297 |      64.9 |         7.8 |                 |            5.74s |      1.36s |       7.66s | low_metadata_alignment, ...        |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,776 |                    18 |          1,794 |      38.6 |         5.7 |                 |            6.32s |      1.54s |      10.10s | missing-sections(title+desc...     |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               676 |                   275 |            951 |       120 |         5.5 |                 |            6.33s |      1.47s |       9.98s | keyword-count(56), ...             |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,366 |                    88 |          3,454 |      20.7 |          28 |                 |            6.52s |      2.79s |       9.67s | low_metadata_alignment, ...        |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,173 |                   108 |          3,281 |      26.5 |          13 |                 |            6.90s |      1.58s |       8.86s | low_metadata_alignment             |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               835 |                    90 |            925 |      19.6 |          22 |                 |            6.91s |      2.70s |       9.98s | low-draft-improvement              |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,345 |                   109 |          2,454 |      23.6 |          18 |                 |            6.91s |      1.76s |       9.05s | low_metadata_alignment, ...        |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,640 |                    96 |          2,736 |      30.4 |          23 |                 |            7.00s |      2.31s |       9.70s | low_metadata_alignment, ...        |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               821 |                   500 |          1,321 |      77.6 |           6 |                 |            7.26s |      1.50s |       9.13s | repetitive(phrase: "- do no...     |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               835 |                    96 |            931 |        15 |          28 |                 |            7.55s |      3.41s |      11.37s | low_metadata_alignment, ...        |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,394 |                   130 |          1,524 |      19.1 |         9.5 |                 |            7.84s |      0.98s |       9.22s | low_metadata_alignment, ...        |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               829 |                   332 |          1,161 |      46.8 |          17 |                 |            8.05s |      2.39s |      10.81s | description-sentences(10), ...     |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               830 |                   108 |            938 |      17.7 |          32 |                 |            8.05s |      3.39s |      11.81s | keyword-count(19), ...             |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,564 |                   500 |          2,064 |      66.7 |          18 |                 |            9.07s |      2.23s |      11.69s | generation_loop(degeneration), ... |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,564 |                   500 |          2,064 |      61.5 |          22 |                 |            9.71s |      2.55s |      12.63s | missing-sections(title+desc...     |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               543 |                   148 |            691 |        19 |          15 |                 |            9.93s |      1.74s |      12.03s | description-sentences(4), ...      |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               830 |                   105 |            935 |      18.9 |          18 |                 |           10.17s |      2.54s |      13.13s | keyword-count(21), ...             |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,883 |                   500 |          2,383 |      57.4 |          60 |                 |           10.69s |      4.86s |      15.93s | missing-sections(title+desc...     |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,791 |                   500 |          3,291 |      59.9 |         9.7 |                 |           11.12s |      0.95s |      12.43s | keyword-count(27), ...             |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,654 |                   500 |          5,154 |        50 |         4.6 |                 |           11.95s |      1.28s |      13.61s | repetitive(phrase: "- outpu...     |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,592 |                   500 |          7,092 |        80 |         8.4 |                 |           13.72s |      5.15s |      21.21s | repetitive(phrase: "black h...     |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,592 |                    96 |          6,688 |      15.2 |          11 |                 |           15.44s |      1.71s |      17.55s | hallucination, ...                 |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,457 |                   500 |          3,957 |      38.8 |          15 |                 |           15.59s |      1.98s |      17.93s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,592 |                   112 |          6,704 |      32.7 |          78 |                 |           15.63s |      5.85s |      21.86s | hallucination, ...                 |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,749 |                   114 |          1,863 |      30.3 |          48 |                 |           19.55s |      1.88s |      21.87s | keyword-count(20), ...             |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |            16,779 |                   500 |         17,279 |      91.9 |         8.6 |                 |           21.00s |      0.86s |      22.25s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             3,377 |                   106 |          3,483 |      5.95 |          27 |                 |           21.20s |      2.36s |      23.94s | keyword-count(19), ...             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,749 |                   114 |          1,863 |      50.9 |          41 |                 |           21.69s |      4.42s |      28.52s | keyword-count(20), ...             |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |            16,779 |                   500 |         17,279 |      92.5 |         8.6 |                 |           21.79s |      0.69s |      22.84s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,582 |                   500 |          2,082 |      18.9 |          11 |                 |           27.52s |      1.61s |      29.49s | repetitive(phrase: "- do no...     |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               544 |                   134 |            678 |      3.35 |          25 |                 |           42.44s |      2.36s |      45.18s | missing-sections(title+desc...     |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,790 |                    14 |         16,804 |      46.4 |          13 |                 |           55.00s |      3.14s |      60.31s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |            16,808 |                   137 |         16,945 |      56.1 |          76 |                 |           58.21s |      6.07s |      64.68s | low_metadata_alignment, ...        |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,808 |                   113 |         16,921 |      68.7 |          35 |                 |           59.23s |      3.35s |      62.95s | low_metadata_alignment             |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |            16,781 |                   500 |         17,281 |      27.9 |         8.6 |                 |           66.78s |      0.97s |      68.35s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,808 |                    99 |         16,907 |      91.4 |          11 |                 |           67.05s |      1.56s |      69.00s | low_metadata_alignment             |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,808 |                    93 |         16,901 |      59.7 |          76 |                 |           68.07s |      6.62s |      75.08s | low_metadata_alignment, ...        |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,808 |                   105 |         16,913 |      68.2 |          26 |                 |           73.62s |      2.75s |      76.73s | low_metadata_alignment             |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,790 |                   500 |         17,290 |       207 |         5.1 |                 |           79.52s |      2.45s |      82.62s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,808 |                   131 |         16,939 |      18.3 |          38 |                 |           99.24s |      3.38s |     103.01s | low_metadata_alignment             |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,808 |                   121 |         16,929 |      23.9 |          26 |                 |          100.84s |      2.85s |     106.12s | low_metadata_alignment             |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,564 |                   500 |          2,064 |      4.37 |          39 |                 |          118.17s |      3.57s |     122.12s | missing-sections(title), ...       |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,808 |                   109 |         16,917 |      6.35 |          38 |                 |          137.78s |      4.46s |     142.77s |                                    |                 |

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
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
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
  sha256=e62ebcb4631e77eda0aac74719a4b7df7639997787eb1144888ad11b02386ef6)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.1.dev20260712+4367c73b`
- `mlx-vlm`: `0.6.5`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.5`
- `huggingface-hub`: `1.23.0`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-17 13:44:05 BST
