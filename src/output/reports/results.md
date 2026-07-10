# Model Performance Results

Generated on: 2026-07-10 23:37:01 BST

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
  outputs=16/60.
- _Useful now:_ 13 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 42 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=40, neutral=3, worse=17 (baseline D 49/100).
- _Quality signal frequency:_ metadata_borrowing=30, missing_sections=21,
  context_budget=11, keyword_count=10, trusted_hint_ignored=10,
  context_ignored=10.

### Runtime

- _Runtime pattern:_ upstream model prefill / first-token dominates measured
  phase time (56%; 20/61 measured model(s)).
- _Phase totals:_ model load=142.69s, local prompt prep=0.22s, upstream model
  prefill / first-token=787.03s, input preparation + decode=252.05s,
  generation call total (unsplit)=205.11s, cleanup=8.09s.
- _Generation total:_ 1244.19s across 61 model(s); upstream model prefill /
  first-token split available for 60/61 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=53, exception=1, max_tokens=7.
- _Validation overhead:_ 22.92s total (avg 0.38s across 61 model(s)).
- _Upstream model prefill / first-token time:_ Avg 13.12s | Min 0.03s | Max
  84.42s across 60 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (508.3 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.4 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.34s)
- **📊 Average TPS:** 88.7 across 60 models

## 📈 Resource Usage

- **Input image size:** 57.50 MP
- **Average peak delta from post-load:** 4.59 GB
- **Peak memory delta / MP:** 82 MB/MP
- **Average peak memory:** 20.4 GB
- **Memory efficiency:** 261 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 12 | ✅ B: 17 | 🟡 C: 12 | 🟠 D: 8 | ❌ F: 11

**Average Utility Score:** 57/100

**Existing Metadata Baseline:** 🟠 D (49/100)
**Vs Existing Metadata:** Avg Δ +8 | Better: 40, Neutral: 3, Worse: 17

- **Best for cataloging:** `mlx-community/Qwen3.5-27B-mxfp8` (🏆 A, 90/100)
- **Best descriptions:** `mlx-community/Qwen3.5-9B-MLX-4bit` (100/100)
- **Best keywording:** `mlx-community/GLM-4.6V-nvfp4` (81/100)
- **Worst for cataloging:** `Qwen/Qwen3-VL-2B-Instruct` (❌ F, 0/100)

### ⚠️ 19 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (46/100) - Lacks visual description of image
- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2.5-VL-1.6B-bf16`: 🟠 D (44/100) - Lacks visual description of image
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (39/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (23/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (48/100) - Limited novel information
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (2/100) - Output too short to be useful
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (7/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (20/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (49/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/gemma-4-31b-bf16` (`Model Error`)
- **🔄 Repetitive Output (2):**
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `-`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
- **👻 Hallucinations (2):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-nvfp4`
- **📝 Formatting Issues (2):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/MiniCPM-V-4.6-8bit`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 88.7 | Min: 4.65 | Max: 508
- **Peak Memory**: Avg: 20 | Min: 1.4 | Max: 78
- **Total Time**: Avg: 19.77s | Min: 1.36s | Max: 97.00s
- **Generation Time**: Avg: 17.32s | Min: 0.67s | Max: 93.38s
- **Model Load Time**: Avg: 2.06s | Min: 0.34s | Max: 9.07s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Ornith-1.0-35B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ornith-10-35b-bf16)
  (Utility A 85/100 | Description 93 | Keywords 68 | Speed 60.9 TPS | Memory
  76 | Caveat missing terms: Bird, Boating, Bushes, Coast, Deben Estuary)
- _Best descriptions:_ [`mlx-community/Ornith-1.0-35B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ornith-10-35b-bf16)
  (Utility A 85/100 | Description 93 | Keywords 68 | Speed 60.9 TPS | Memory
  76 | Caveat missing terms: Bird, Boating, Bushes, Coast, Deben Estuary)
- _Best keywording:_ [`mlx-community/Ornith-1.0-35B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ornith-10-35b-bf16)
  (Utility A 85/100 | Description 93 | Keywords 68 | Speed 60.9 TPS | Memory
  76 | Caveat missing terms: Bird, Boating, Bushes, Coast, Deben Estuary)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility B 67/100 | Description 93 | Keywords 47 | Speed 508 TPS | Memory
  1.4 | Caveat missing terms: Bird, Buoy, Bushes, Rigging, Woodbridge;
  keywords=19; nonvisual metadata reused)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility B 67/100 | Description 93 | Keywords 47 | Speed 508 TPS | Memory
  1.4 | Caveat missing terms: Bird, Buoy, Bushes, Rigging, Woodbridge;
  keywords=19; nonvisual metadata reused)
- _Best balance:_ [`mlx-community/Ornith-1.0-35B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ornith-10-35b-bf16)
  (Utility A 85/100 | Description 93 | Keywords 68 | Speed 60.9 TPS | Memory
  76 | Caveat missing terms: Bird, Boating, Bushes, Coast, Deben Estuary)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/gemma-4-31b-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-4-31b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (2):_ [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen3-vl-2b-thinking-bf16),
  [`mlx-community/paligemma2-3b-pt-896-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-paligemma2-3b-pt-896-4bit).
  Example: token: `-`.
- _👻 Hallucinations (2):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4).
- _📝 Formatting Issues (2):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-minicpm-v-46-8bit).
- _Low-utility outputs (19):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  +15 more. Common weakness: Lacks visual description of image.

## 🚨 Failures by Package (Actionable)

| Package   |   Failures | Error Types   | Affected Models                  |
|-----------|------------|---------------|----------------------------------|
| `mlx-vlm` |          1 | Model Error   | `mlx-community/gemma-4-31b-bf16` |

### Actionable Items by Package

#### mlx-vlm

- mlx-community/gemma-4-31b-bf16 (Model Error)
  - Error: `Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: In...`
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
> &#45; Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe
> &#45; Description hint: Two sailing boats moored on a river with trees behind on
> the bank
> &#45; Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary,
> England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat,
> Nature, Outdoors, Peaceful, Rigging
> &#45; Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local
> time).
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1419.32s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/gemma-4-31b-bf16`                        |                   |                       |                |           |             |                 |          205.11s |     18.81s |     224.31s | gibberish(char_noise), ...         |         mlx-vlm |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               585 |                   121 |            706 |       508 |         1.4 | stop            |            0.67s |      0.34s |       1.36s | description-sentences(3), ...      |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               775 |                    69 |            844 |       325 |           3 | stop            |            0.79s |      0.55s |       1.72s | metadata-borrowing                 |                 |
| `qnguyen3/nanoLLaVA`                                    |               512 |                    31 |            543 |       112 |         4.6 | stop            |            0.95s |      0.57s |       1.88s | missing-sections(keywords), ...    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               512 |                    93 |            605 |       335 |         2.7 | stop            |            0.98s |      0.56s |       1.89s | title-length(11), ...              |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,530 |                     5 |          1,535 |      24.1 |          11 | stop            |            1.22s |      1.47s |       3.06s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |             1,127 |                    77 |          1,204 |       225 |           4 | stop            |            1.27s |      0.99s |       2.65s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               775 |                   171 |            946 |       188 |         4.1 | stop            |            1.46s |      0.57s |       2.40s | title-length(22), ...              |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               516 |                   191 |            707 |       348 |         2.2 | stop            |            1.67s |      0.63s |       2.73s | missing-sections(keywords), ...    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,722 |                    90 |          1,812 |       129 |         5.6 | stop            |            1.75s |      0.72s |       2.83s | keyword-count(19), ...             |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,722 |                    90 |          1,812 |       127 |         5.6 | stop            |            1.82s |      0.65s |       2.84s | keyword-count(19), ...             |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,530 |                    27 |          1,557 |        32 |          12 | stop            |            2.53s |      1.56s |       4.46s | numeric_loop, ...                  |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               783 |                    88 |            871 |      60.4 |          18 | stop            |            2.60s |      2.34s |       5.31s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,330 |                   101 |          1,431 |      57.7 |         9.5 | stop            |            2.61s |      0.93s |       3.91s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               783 |                    89 |            872 |      58.9 |          29 | stop            |            2.62s |      3.28s |       6.30s | metadata-borrowing                 |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,123 |                   107 |          3,230 |       168 |         7.8 | stop            |            2.67s |      2.12s |       5.20s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,330 |                   101 |          1,431 |      55.5 |         9.5 | stop            |            2.69s |      0.98s |       4.05s |                                    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               622 |                   248 |            870 |       134 |         5.5 | stop            |            2.89s |      0.66s |       3.91s | numeric_loop, ...                  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,730 |                    11 |          2,741 |      68.1 |         9.7 | stop            |            2.93s |      0.96s |       4.26s | missing-sections(title+desc...     |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               783 |                    78 |            861 |        33 |          28 | stop            |            3.44s |      3.19s |       7.01s | metadata-borrowing                 |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,295 |                    71 |          2,366 |      34.8 |          18 | stop            |            3.52s |      1.65s |       5.56s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               778 |                    78 |            856 |      31.5 |          18 | stop            |            4.25s |      2.28s |       6.91s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,124 |                    95 |          3,219 |        65 |          13 | stop            |            4.28s |      1.33s |       5.99s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,317 |                    78 |          3,395 |        39 |          16 | stop            |            4.43s |      1.73s |       6.52s | metadata-borrowing, ...            |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,295 |                    72 |          2,367 |      31.4 |          18 | stop            |            4.54s |      1.71s |       6.63s | title-length(4)                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,124 |                   120 |          3,244 |      61.8 |          13 | stop            |            4.75s |      1.44s |       6.56s | metadata-borrowing, ...            |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               769 |                   500 |          1,269 |       124 |           6 | length          |            4.86s |      1.46s |       6.71s | generation_loop(degeneration), ... |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,805 |                   126 |          2,931 |      32.1 |          19 | stop            |            5.73s |      1.93s |       8.03s | description-sentences(4), ...      |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,317 |                    80 |          3,397 |      20.1 |          28 | stop            |            6.22s |      2.58s |       9.17s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               783 |                    81 |            864 |      19.6 |          22 | stop            |            6.33s |      2.77s |       9.51s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,530 |                    25 |          1,555 |      5.49 |          26 | stop            |            6.48s |      2.50s |       9.34s | gibberish(token_noise), ...        |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               777 |                   272 |          1,049 |      48.5 |          17 | stop            |            6.57s |      2.30s |       9.25s | description-sentences(10), ...     |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,591 |                    82 |          2,673 |      29.9 |          23 | stop            |            7.02s |      2.14s |       9.59s | title-length(4)                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               778 |                    91 |            869 |      17.7 |          32 | stop            |            7.06s |      3.31s |      10.75s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,514 |                   393 |          1,907 |        70 |          18 | stop            |            7.19s |      2.00s |       9.59s | missing-sections(title), ...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,542 |                    92 |          6,634 |      58.4 |          11 | stop            |            7.63s |      1.49s |       9.49s | hallucination, ...                 |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               493 |                   129 |            622 |      20.5 |          15 | stop            |            8.51s |      1.51s |      10.41s | missing-sections(title+desc...     |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,514 |                   500 |          2,014 |      63.9 |          22 | length          |            9.43s |      2.18s |      11.99s | missing-sections(title), ...       |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,602 |                   500 |          5,102 |      50.6 |         4.6 | length          |           11.83s |      1.15s |      13.35s | repetitive(phrase: "- outpu...     |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,542 |                   500 |          7,042 |      79.9 |         8.4 | length          |           12.25s |      1.32s |      13.95s | description-sentences(3), ...      |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,826 |                   500 |          2,326 |      45.8 |          60 | length          |           13.24s |      4.93s |      18.55s | generation_loop(degeneration), ... |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |            16,729 |                     2 |         16,731 |       181 |         8.6 | stop            |           14.44s |      0.67s |      15.46s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,699 |                    31 |          1,730 |      46.7 |          41 | stop            |           14.77s |      1.76s |      16.91s | missing-sections(title+desc...     |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,542 |                   103 |          6,645 |      40.7 |          78 | stop            |           14.84s |      5.67s |      20.90s | hallucination, ...                 |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,408 |                   500 |          3,908 |      41.5 |          15 | length          |           14.89s |      1.57s |      16.84s | missing-sections(title+keyw...     |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,699 |                    42 |          1,741 |        27 |          48 | stop            |           16.38s |      1.80s |      18.56s | missing-sections(title+desc...     |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               494 |                    73 |            567 |      5.13 |          25 | stop            |           16.62s |      2.20s |      19.20s | missing-sections(title+desc...     |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |            16,729 |                     2 |         16,731 |       180 |         8.6 | stop            |           17.46s |      1.00s |      18.83s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,740 |                    14 |         16,754 |      60.6 |          13 | stop            |           18.31s |      1.11s |      19.80s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             3,327 |                    94 |          3,421 |      5.94 |          27 | stop            |           19.59s |      3.28s |      23.31s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,514 |                    96 |          1,610 |      4.65 |          39 | stop            |           23.07s |      3.50s |      26.96s | missing-sections(title), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |            16,731 |                   500 |         17,231 |      88.4 |         8.6 | length          |           25.29s |      0.95s |      26.64s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,756 |                   126 |         16,882 |      92.4 |          11 | stop            |           60.37s |      1.39s |      62.13s | keyword-count(20), ...             |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,756 |                   106 |         16,862 |       104 |          26 | stop            |           63.71s |      2.55s |      66.69s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,756 |                    99 |         16,855 |      92.2 |          35 | stop            |           63.81s |      3.21s |      67.39s | metadata-borrowing, ...            |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,756 |                   102 |         16,858 |      64.6 |          76 | stop            |           65.38s |      9.07s |      74.82s | metadata-borrowing, ...            |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |            16,756 |                   105 |         16,861 |      60.9 |          76 | stop            |           66.40s |      8.32s |      75.10s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,740 |                    44 |         16,784 |       221 |         5.1 | stop            |           74.34s |      0.58s |      75.30s | missing-sections(keywords), ...    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,756 |                   114 |         16,870 |      17.6 |          38 | stop            |           84.84s |      3.07s |      88.30s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,756 |                   130 |         16,886 |      29.6 |          26 | stop            |           87.54s |      2.24s |      90.29s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,756 |                   135 |         16,891 |      16.7 |          38 | stop            |           93.38s |      3.20s |      97.00s |                                    |                 |

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
- _MLX Distribution Root:_ /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages
- _mlx-metal Distribution:_ not installed; local editable mlx supplies backend
- _MLX Core Extension:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so
- _MLX Metallib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib
  (162,449,848 bytes,
  sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,697,568 bytes,
  sha256=e61c827cd79f978aa5eacc136f65d6dea065005787f3a1457dc9d4512d6ee9cf)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.1.dev20260710+4367c73b`
- `mlx-vlm`: `0.6.5`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.5`
- `huggingface-hub`: `1.23.0`
- `transformers`: `5.13.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-10 23:37:01 BST
