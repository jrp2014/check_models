# Model Performance Results

Generated on: 2026-07-17 23:16:01 BST

## Run Contract

- Evaluation lane: assisted
- Metadata exposed to prompt: yes
- Semantic rankings: grounded (metadata-assisted visual verification)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ none.
- _Maintainer signals:_ harness-risk successes=7, mechanically clean
  outputs=6/61.
- _Useful now:_ 7 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 52 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=10, neutral=4, worse=47 (baseline B 74/100).
- _Quality signal frequency:_ low_draft_improvement=45, missing_sections=24,
  keyword_count=21, context_ignored=14, trusted_hint_ignored=14, cutoff=12.

### Runtime

- _Runtime pattern:_ upstream model prefill / first-token dominates measured
  phase time (60%; 17/61 measured model(s)).
- _Phase totals:_ model load=145.71s, local prompt prep=0.27s, upstream model
  prefill / first-token=832.51s, input preparation + decode=405.82s,
  cleanup=7.52s.
- _Generation total:_ 1238.34s across 61 model(s); upstream model prefill /
  first-token split available for 61/61 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=49, max_tokens=12.
- _Validation overhead:_ 21.47s total (avg 0.35s across 61 model(s)).
- _Upstream model prefill / first-token time:_ Avg 13.65s | Min 0.03s | Max
  95.19s across 61 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (509.5 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.5 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.35s)
- **📊 Average TPS:** 81.6 across 61 models

## 📈 Resource Usage

- **Input image size:** 54.47 MP
- **Average peak delta from post-load:** 4.48 GB
- **Peak memory delta / MP:** 84 MB/MP
- **Average peak memory:** 21.0 GB
- **Memory efficiency:** 254 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 6 | ✅ B: 12 | 🟡 C: 15 | 🟠 D: 17 | ❌ F: 11

**Average Utility Score:** 51/100

**Existing Metadata Baseline:** ✅ B (74/100)
**Vs Existing Metadata:** Avg Δ -22 | Better: 10, Neutral: 4, Worse: 47

- **Best for cataloging:** `mlx-community/Ornith-1.0-35B-bf16` (🏆 A, 83/100)
- **Best descriptions:** `mlx-community/Ornith-1.0-35B-bf16` (99/100)
- **Best keywording:** `mlx-community/gemma-3-27b-it-qat-8bit` (73/100)
- **Worst for cataloging:** `Qwen/Qwen3-VL-2B-Instruct` (❌ F, 0/100)

### ⚠️ 28 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (7/100) - Output lacks detail
- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: 🟠 D (41/100) - Lacks visual description of image
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (0/100) - Empty or minimal output
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: 🟠 D (41/100) - Keywords are not specific or diverse enough
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (38/100) - Keywords are not specific or diverse enough
- `mlx-community/Idefics3-8B-Llama3-bf16`: 🟠 D (41/100) - Lacks visual description of image
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (49/100) - Lacks visual description of image
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (44/100) - Lacks visual description of image
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (49/100) - Missing requested structure
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-9B-MLX-4bit`: 🟠 D (41/100) - Lacks visual description of image
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (7/100) - Output lacks detail
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/diffusiongemma-26B-A4B-it-8bit`: 🟠 D (48/100) - Lacks visual description of image
- `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`: 🟠 D (47/100) - Lacks visual description of image
- `mlx-community/gemma-3-27b-it-qat-4bit`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (42/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-31b-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (41/100) - Lacks visual description of image
- `mlx-community/nanoLLaVA-1.5-4bit`: ❌ F (31/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (14/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (14/100) - Output lacks detail
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-8bit`: 🟠 D (44/100) - Lacks visual description of image
- `mlx-community/pixtral-12b-bf16`: 🟠 D (43/100) - Lacks visual description of image
- `qnguyen3/nanoLLaVA`: ❌ F (25/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **🔄 Repetitive Output (6):**
  - `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (token: `phrase: "stade stade stade stade..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "deal, kent, uk, 2026-07-11..."`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `-`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "- do not go..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "text after the label...."`)
  - `qnguyen3/nanoLLaVA` (token: `Deal,`)
- **👻 Hallucinations (2):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-nvfp4`
- **📝 Formatting Issues (4):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/MiniCPM-V-4.6-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 81.6 | Min: 4.26 | Max: 510
- **Peak Memory**: Avg: 21 | Min: 1.5 | Max: 78
- **Total Time**: Avg: 23.05s | Min: 1.25s | Max: 123.24s
- **Generation Time**: Avg: 20.30s | Min: 0.59s | Max: 119.63s
- **Model Load Time**: Avg: 2.39s | Min: 0.35s | Max: 13.55s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Ornith-1.0-35B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ornith-10-35b-bf16)
  (Utility A 83/100 | Description 99 | Keywords 55 | Speed 51.4 TPS | Memory
  76 | Caveat missing terms: Cars, Deal, Swimming, architecture, GBR;
  keywords=20)
- _Best descriptions:_ [`mlx-community/Ornith-1.0-35B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ornith-10-35b-bf16)
  (Utility A 83/100 | Description 99 | Keywords 55 | Speed 51.4 TPS | Memory
  76 | Caveat missing terms: Cars, Deal, Swimming, architecture, GBR;
  keywords=20)
- _Best keywording:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit)
  (Utility C 60/100 | Description 79 | Keywords 73 | Speed 17.4 TPS | Memory
  32 | Caveat missing terms: Cars, Deal, Kent, Sitting, Swimming; keywords=19;
  low-draft-improvement)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility D 41/100 | Description 60 | Keywords 55 | Speed 510 TPS | Memory
  1.5 | Caveat keywords=20; low-draft-improvement)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility D 41/100 | Description 60 | Keywords 55 | Speed 510 TPS | Memory
  1.5 | Caveat keywords=20; low-draft-improvement)
- _Best balance:_ [`mlx-community/Qwen3.5-27B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-4bit)
  (Utility B 72/100 | Description 89 | Keywords 57 | Speed 29.0 TPS | Memory
  26 | Caveat missing terms: Seaside, Shore, Walking, Water, architecture)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _🔄 Repetitive Output (6):_ [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen3-vl-2b-thinking-bf16),
  [`mlx-community/gemma-3n-E2B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  +2 more. Example: token: `phrase: "stade stade stade stade..."`.
- _👻 Hallucinations (2):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4).
- _📝 Formatting Issues (4):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-diffusiongemma-26b-a4b-it-8bit),
  [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8).
- _Low-utility outputs (28):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  +24 more. Common weakness: Output lacks detail.

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
> &#45; Location terms: England, Europe, Town, UK
> &#45; Capture date/time: 2026-07-11 19:04:26 BST 19:04:26
> &#45; GPS: 51.226814°N, 1.401142°E
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
>
> &#8203;Draft descriptive metadata:
> &#45; Existing title: Seafront, Deal, England, UK, GBR, Europe
> &#45; Existing description: A coastal view of Deal, Kent, UK, showing the
> shingle beach, sea, and seafront buildings on a partly cloudy day.
> &#45; Existing keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent,
> Landscape, People, Promenade, Seaside, Shore, Sitting, Sky, Swimming,
> Walking, Water, Waterfront, Waves, architecture
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1414.70s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                      |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:------------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               657 |                   100 |            757 |       510 |         1.5 | stop            |            0.59s |      0.35s |       1.25s | keyword-count(20), ...              |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               585 |                    47 |            632 |       291 |         2.9 | stop            |            0.78s |      0.57s |       1.68s | missing-sections(keywords), ...     |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               843 |                    88 |            931 |       308 |           3 | stop            |            0.89s |      0.61s |       1.85s | low-draft-improvement               |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,800 |                    15 |          1,815 |       130 |         5.7 | stop            |            1.10s |      0.76s |       2.18s | missing-sections(title+desc...      |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,800 |                    15 |          1,815 |       110 |         5.7 | stop            |            1.19s |      0.67s |       2.20s | missing-sections(title+desc...      |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               843 |                   125 |            968 |       185 |         4.1 | stop            |            1.21s |      0.58s |       2.15s | title-length(20), ...               |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |             1,205 |                    79 |          1,284 |       209 |           4 | stop            |            1.36s |      1.07s |       2.87s | ⚠️harness(stop_token), ...          |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               865 |                    89 |            954 |       104 |          17 | stop            |            1.97s |      2.67s |       5.06s | low-draft-improvement               |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               589 |                   214 |            803 |       307 |         2.2 | stop            |            2.00s |      0.75s |       3.08s | missing-sections(keywords), ...     |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,612 |                    17 |          1,629 |      29.6 |          12 | stop            |            2.28s |      1.71s |       4.33s | missing-sections(title+desc...      |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,845 |                   118 |          2,963 |       119 |         6.4 | stop            |            2.39s |      1.00s |       3.73s |                                     |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               861 |                    87 |            948 |      58.9 |          28 | stop            |            2.56s |      3.27s |       6.17s | missing-sections(title), ...        |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               861 |                   100 |            961 |      59.7 |          29 | stop            |            2.74s |      3.30s |       6.40s | missing-sections(title), ...        |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               701 |                   308 |          1,009 |       130 |         5.5 | stop            |            3.38s |      0.68s |       4.41s | fabrication, keyword-count(65), ... |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,424 |                   162 |          1,586 |      55.6 |         9.6 | stop            |            3.73s |      1.01s |       5.07s | description-sentences(4), ...       |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,424 |                   162 |          1,586 |      52.8 |         9.6 | stop            |            3.94s |      0.99s |       5.48s | description-sentences(4), ...       |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,880 |                    79 |          2,959 |      34.3 |          18 | stop            |            3.95s |      1.66s |       5.94s | low-draft-improvement               |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,846 |                   118 |          2,964 |      62.8 |          12 | stop            |            4.39s |      1.43s |       6.16s |                                     |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,917 |                    93 |          3,010 |      38.9 |          15 | stop            |            4.46s |      1.79s |       6.60s | title-length(4), ...                |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               853 |                     4 |            857 |       9.1 |          65 | stop            |            4.58s |      6.63s |      11.57s | ⚠️harness(prompt_template), ...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,846 |                   113 |          2,959 |      58.6 |          12 | stop            |            4.64s |      1.42s |       6.44s |                                     |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,608 |                   109 |          2,717 |      58.9 |         9.7 | stop            |            4.75s |      0.96s |       6.04s | keyword-count(20), ...              |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,866 |                    94 |          2,960 |      32.2 |          19 | stop            |            4.88s |      2.01s |       7.27s | keyword-count(20), ...              |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               847 |                   500 |          1,347 |       121 |         6.1 | length          |            4.93s |      1.51s |       6.77s | repetitive(phrase: "- do no...      |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               856 |                    95 |            951 |      30.8 |          18 | stop            |            5.04s |      2.32s |       7.70s | keyword-count(20), ...              |                 |
| `qnguyen3/nanoLLaVA`                                    |               585 |                   500 |          1,085 |       111 |         4.9 | length          |            5.15s |      0.58s |       6.06s | repetitive(Deal,), ...              |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,880 |                    93 |          2,973 |      31.5 |          19 | stop            |            5.69s |      1.71s |       7.74s | title-length(3), ...                |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,612 |                    17 |          1,629 |      5.09 |          27 | stop            |            5.76s |      2.71s |       8.81s | missing-sections(title+desc...      |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               865 |                    95 |            960 |      23.9 |          20 | stop            |            6.12s |      2.74s |       9.21s | low-draft-improvement               |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,917 |                    89 |          3,006 |      20.4 |          27 | stop            |            6.56s |      2.56s |       9.46s | title-length(4), ...                |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,313 |                    97 |          2,410 |      29.2 |          22 | stop            |            7.22s |      2.26s |       9.89s | low-draft-improvement               |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,593 |                    88 |          6,681 |      56.1 |          11 | stop            |            7.37s |      1.62s |       9.33s | hallucination, ...                  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               555 |                   102 |            657 |      19.4 |          15 | stop            |            7.61s |      1.63s |       9.60s | missing-sections(title+desc...      |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               856 |                   105 |            961 |      17.4 |          32 | stop            |            7.99s |      3.35s |      11.67s | keyword-count(19), ...              |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,571 |                   500 |          2,071 |        71 |          18 | length          |            8.82s |      2.07s |      11.24s | missing-sections(title), ...        |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,571 |                   500 |          2,071 |      60.5 |          22 | length          |            9.93s |      2.51s |      12.80s | missing-sections(title), ...        |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               855 |                   401 |          1,256 |      41.4 |          17 | stop            |           10.61s |      2.30s |      13.25s | missing-sections(title+desc...      |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,593 |                   500 |          7,093 |      77.1 |         8.4 | length          |           12.54s |      1.37s |      14.25s | title-length(3), ...                |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,684 |                   500 |          5,184 |      47.1 |         4.7 | length          |           12.72s |      1.42s |      14.48s | repetitive(phrase: "text af...      |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |            16,898 |                     2 |         16,900 |       179 |         8.6 | stop            |           14.31s |      0.78s |      15.41s | ⚠️harness(long_context), ...        |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,008 |                   500 |          3,508 |      40.7 |          15 | length          |           14.75s |      1.62s |      16.72s | fabrication, ...                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,887 |                   500 |          2,387 |      44.9 |          60 | length          |           14.87s |      6.60s |      21.83s | repetitive(phrase: "stade s...      |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,593 |                    93 |          6,686 |      40.8 |          78 | stop            |           16.14s |      6.73s |      23.21s | hallucination, ...                  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               556 |                    70 |            626 |      5.09 |          25 | stop            |           16.25s |      2.31s |      18.89s | missing-sections(title+desc...      |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,764 |                   114 |          1,878 |      46.5 |          41 | stop            |           16.69s |      1.25s |      18.30s | description-sentences(3), ...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,909 |                     5 |         16,914 |      68.5 |          14 | stop            |           18.73s |      1.22s |      20.28s | ⚠️harness(long_context), ...        |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,764 |                   173 |          1,937 |      29.2 |          48 | stop            |           21.29s |      1.91s |      23.69s | description-sentences(6), ...       |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             3,381 |                   119 |          3,500 |      6.04 |          27 | stop            |           23.45s |      3.13s |      26.98s | keyword-count(22), ...              |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |            16,898 |                     2 |         16,900 |       139 |         8.6 | stop            |           24.84s |      0.94s |      26.13s | ⚠️harness(long_context), ...        |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,612 |                   500 |          2,112 |      18.6 |          11 | length          |           27.87s |      1.66s |      29.88s | missing-sections(title+desc...      |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |            16,900 |                   500 |         17,400 |      73.6 |         8.6 | length          |           33.85s |      0.87s |      35.15s | ⚠️harness(long_context), ...        |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,928 |                    95 |         17,023 |      93.1 |          11 | stop            |           56.58s |      1.44s |      58.35s | keyword-count(19), ...              |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,928 |                   121 |         17,049 |      94.5 |          26 | stop            |           58.47s |      2.57s |      61.40s | description-sentences(3)            |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,928 |                   105 |         17,033 |      89.3 |          35 | stop            |           61.18s |      3.17s |      64.70s |                                     |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,928 |                   110 |         17,038 |      65.7 |          76 | stop            |           69.75s |     12.78s |      82.87s |                                     |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,928 |                   124 |         17,052 |      17.5 |          38 | stop            |           82.78s |      3.08s |      86.20s | keyword-count(19)                   |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |            16,928 |                   120 |         17,048 |      51.4 |          76 | stop            |           83.29s |     13.55s |      97.38s | keyword-count(20)                   |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,928 |                   116 |         17,044 |      17.5 |          38 | stop            |           87.83s |      5.03s |      93.21s | keyword-count(20), ...              |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,909 |                   500 |         17,409 |       110 |         5.1 | length          |           87.90s |      0.68s |      88.97s | ⚠️harness(long_context), ...        |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,928 |                   112 |         17,040 |        29 |          26 | stop            |          100.06s |      2.56s |     103.19s |                                     |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,571 |                   500 |          2,071 |      4.26 |          40 | length          |          119.63s |      3.28s |     123.24s | missing-sections(title), ...        |                 |

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
  sha256=98e311dc5a6588305bef55d6f231605e3591120df70183b6cec2cf2d424d8362)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.1.dev20260717+b7c3dd6d`
- `mlx-vlm`: `0.6.5`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.24.0`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-17 23:16:01 BST
