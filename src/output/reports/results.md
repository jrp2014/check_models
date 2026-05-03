# Model Performance Results

_Generated on 2026-05-03 02:16:08 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx=1, model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=7, clean outputs=5/51.
- _Useful now:_ 5 clean A/B model(s) worth first review.
- _Review watchlist:_ 46 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=15, neutral=4, worse=32 (baseline B 75/100).
- _Quality signal frequency:_ metadata_borrowing=30, missing_sections=28,
  cutoff=20, description_length=11, keyword_count=10, reasoning_leak=10.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (91%; 50/53 measured
  model(s)).
- _Phase totals:_ model load=120.19s, prompt prep=0.16s, decode=1271.45s,
  cleanup=6.61s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=51, exception=2.
- _Validation overhead:_ 10.58s total (avg 0.20s across 53 model(s)).
- _First-token latency:_ Avg 14.32s | Min 0.07s | Max 85.76s across 50
  model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (357.3 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.44s)
- **📊 Average TPS:** 72.5 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1077.7 GB
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 233 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 8 | ✅ B: 23 | 🟡 C: 13 | 🟠 D: 1 | ❌ F: 6

**Average Utility Score:** 64/100

**Existing Metadata Baseline:** ✅ B (75/100)
**Vs Existing Metadata:** Avg Δ -12 | Better: 15, Neutral: 4, Worse: 32

- **Best for cataloging:** `mlx-community/Qwen3.5-27B-mxfp8` (🏆 A, 94/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/gemma-4-31b-it-4bit` (87/100)
- **Worst for cataloging:** `mlx-community/llava-v1.6-mistral-7b-8bit` (❌ F, 0/100)

### ⚠️ 7 Models with Low Utility (D/F)

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (2/100) - Output too short to be useful
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (33/100) - Lacks visual description of image
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (18/100) - Output lacks detail

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (6):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `phrase: "wooden house, wooden house,..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `phrase: "wooden house, wooden house,..."`)
  - `mlx-community/X-Reasoner-7B-8bit` (token: `phrase: "solar energy in museums,..."`)
  - `mlx-community/gemma-4-31b-bf16` (token: `phrase: "- the image is..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the bottom of the..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
- **👻 Hallucinations (2):**
  - `microsoft/Phi-3.5-vision-instruct`
  - `mlx-community/Qwen3.5-27B-4bit`
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 72.5 | Min: 0 | Max: 357
- **Peak Memory**: Avg: 21 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 27.45s | Min: 1.43s | Max: 122.52s
- **Generation Time**: Avg: 24.93s | Min: 0.72s | Max: 119.02s
- **Model Load Time**: Avg: 2.31s | Min: 0.44s | Max: 11.97s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility A 86/100 | Description 89 | Keywords 86 | Speed 47.5 TPS | Memory
  13 | Caveat nontext prompt burden=85%; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside)
- _Best descriptions:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 85/100 | Description 100 | Keywords 84 | Speed 28.2 TPS | Memory
  19 | Caveat missing terms: East Anglia, English countryside, Moored,
  Objects, River Deben)
- _Best keywording:_ [`mlx-community/gemma-4-31b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-31b-it-4bit)
  (Utility B 80/100 | Description 100 | Keywords 87 | Speed 26.5 TPS | Memory
  20 | Caveat missing terms: East Anglia, English countryside, Moored,
  Objects, River Deben)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility D 50/100 | Description 89 | Keywords 0 | Speed 357 TPS | Memory 2.6
  | Caveat missing sections: keywords; missing terms: Bench, Blue sky, Clouds,
  East Anglia, English countryside; context echo=66%; nonvisual metadata
  reused)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility B 72/100 | Description 87 | Keywords 0 | Speed 312 TPS | Memory 2.2
  | Caveat missing sections: keywords; missing terms: Bench, East Anglia,
  English countryside, Moored, Mudflats; nonvisual metadata reused)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility A 86/100 | Description 89 | Keywords 86 | Speed 47.5 TPS | Memory
  13 | Caveat nontext prompt burden=85%; missing terms: Bench, Blue sky,
  Clouds, East Anglia, English countryside)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (2):_ [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Model Error`.
- _🔄 Repetitive Output (6):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/SmolVLM-Instruct-bf16`](model_gallery.md#model-mlx-community-smolvlm-instruct-bf16),
  [`mlx-community/X-Reasoner-7B-8bit`](model_gallery.md#model-mlx-community-x-reasoner-7b-8bit),
  [`mlx-community/gemma-4-31b-bf16`](model_gallery.md#model-mlx-community-gemma-4-31b-bf16),
  +2 more. Example: token: `phrase: "wooden house, wooden house,..."`.
- _👻 Hallucinations (2):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Qwen3.5-27B-4bit`](model_gallery.md#model-mlx-community-qwen35-27b-4bit).
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (7):_ [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`mlx-community/gemma-3n-E2B-4bit`](model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  [`mlx-community/llava-v1.6-mistral-7b-8bit`](model_gallery.md#model-mlx-community-llava-v16-mistral-7b-8bit),
  +3 more. Common weakness: Output too short to be useful.

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types     | Affected Models                           |
|----------------|------------|-----------------|-------------------------------------------|
| `mlx`          |          1 | Model Error     | `mlx-community/Kimi-VL-A3B-Thinking-8bit` |
| `model-config` |          1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16`        |

### Actionable Items by Package

#### mlx

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
> &#45; Description hint: The Woodbridge Tide Mill Museum in Woodbridge, Suffolk,
> England, is seen at low tide on the River Deben. In the foreground, a long
> boat covered by a black tarp rests on the muddy riverbed. A few people are
> sitting on benches in front of the historic white building on a sunny day.
> &#45; Keyword hints: Adobe Stock, Any Vision, Bench, Blue sky, Clouds, East
> Anglia, England, English countryside, Europe, Locations, Mill, Moored,
> Mudflats, Museum, Objects, People, Quay, River Deben, Riverbed, Rope
> &#45; Capture metadata: Taken on 2026-05-02 16:24:39 BST (at 16:24:39 local
> time). GPS: 52.091500°N, 1.318500°E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1412.50s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                   |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:---------------------------------|----------------:|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.25s |       1.28s |                                  |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.38s |       3.61s |                                  |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               566 |                    86 |            652 |        5,500 |       357 |         2.6 |            0.72s |      0.61s |       1.52s | missing-sections(keywords), ...  |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               826 |                   115 |            941 |        7,241 |       294 |           3 |            0.78s |      0.46s |       1.43s | title-length(4), ...             |                 |
| `qnguyen3/nanoLLaVA`                                    |               566 |                    74 |            640 |        4,678 |       106 |         4.9 |            1.18s |      0.52s |       1.90s | missing-sections(keywords), ...  |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               636 |                   198 |            834 |        9,627 |       168 |         3.7 |            1.58s |      0.56s |       2.33s | title-length(30), ...            |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               837 |                    83 |            920 |        1,616 |       111 |          17 |            1.63s |      2.31s |       4.15s |                                  |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               570 |                   221 |            791 |        5,476 |       312 |         2.2 |            1.64s |      0.57s |       2.40s | fabrication, ...                 |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               679 |                   134 |            813 |        1,964 |       129 |         5.5 |            1.93s |      0.57s |       2.68s | title-length(4), ...             |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |              1587 |                    20 |           1607 |        1,347 |      30.9 |          12 |            2.22s |      1.69s |       4.09s | missing-sections(title+desc...   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |                 0 |                     0 |              0 |            0 |         0 |         9.7 |            2.94s |      1.00s |       4.13s | ⚠️harness(prompt_template), ...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |              3179 |                   137 |           3316 |        1,884 |       148 |         7.8 |            3.03s |      0.99s |       4.24s | description-sentences(3), ...    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |              1394 |                   169 |           1563 |        3,870 |      56.5 |         9.5 |            3.65s |      0.84s |       4.67s | description-sentences(3), ...    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |              2349 |                   103 |           2452 |        2,701 |      34.6 |          18 |            4.26s |      1.73s |       6.17s | metadata-borrowing               |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |              1587 |                    14 |           1601 |        1,052 |      5.47 |          27 |            4.49s |      2.73s |       7.50s | ⚠️harness(prompt_template), ...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |              2848 |                    91 |           2939 |        1,981 |      32.5 |          19 |            4.72s |      1.88s |       6.79s | description-sentences(3), ...    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               823 |                   500 |           1323 |        2,552 |       115 |           6 |            5.02s |      1.50s |       6.71s | missing-sections(title+desc...   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |              1779 |                   500 |           2279 |        4,188 |       121 |         5.7 |            5.08s |      0.59s |       5.86s | repetitive(phrase: "wooden...    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |              1779 |                   500 |           2279 |        4,100 |       120 |         5.6 |            5.10s |      0.44s |       5.72s | repetitive(phrase: "wooden...    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |              2349 |                   108 |           2457 |        1,382 |      31.9 |          18 |            5.50s |      1.74s |       7.42s | metadata-borrowing               |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               837 |                   100 |            937 |          549 |      26.5 |          20 |            5.80s |      3.10s |       9.16s |                                  |                 |
| `mlx-community/pixtral-12b-8bit`                        |              3373 |                    96 |           3469 |          989 |      31.5 |          16 |            6.91s |      1.94s |       9.20s | metadata-borrowing, ...          |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               832 |                   108 |            940 |          259 |      28.2 |          19 |            7.37s |      2.29s |       9.85s |                                  |                 |
| `mlx-community/pixtral-12b-bf16`                        |              3373 |                    90 |           3463 |        1,465 |      19.1 |          28 |            7.38s |      2.53s |      10.22s |                                  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |              3180 |                   119 |           3299 |          685 |      48.5 |          13 |            7.58s |      1.57s |       9.44s | metadata-borrowing, ...          |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |              3180 |                   119 |           3299 |          683 |      47.5 |          13 |            7.62s |      1.48s |       9.37s |                                  |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               831 |                   326 |           1157 |        1,789 |      47.5 |          17 |            7.66s |      2.22s |      10.07s | missing-sections(title+desc...   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |              2682 |                   124 |           2806 |          709 |      31.4 |          22 |            8.16s |      2.07s |      10.43s | ⚠️harness(encoding), ...         |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               832 |                   116 |            948 |          582 |      17.3 |          33 |            8.45s |      3.18s |      11.82s | metadata-borrowing               |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |              1559 |                   500 |           2059 |          866 |      67.7 |          18 |            9.70s |      2.00s |      11.88s | degeneration, ...                |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |              1394 |                   500 |           1894 |        3,649 |      55.4 |         9.5 |            9.71s |      0.87s |      10.76s | ⚠️harness(stop_token), ...       |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               536 |                   100 |            636 |          245 |      12.5 |          15 |           10.57s |      1.61s |      12.37s | missing-sections(title+desc...   |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |              3464 |                   500 |           3964 |        1,446 |      41.4 |          15 |           14.85s |      1.59s |      16.63s | fabrication, ...                 |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |              1887 |                   500 |           2387 |          317 |        48 |          60 |           17.08s |     10.49s |      27.92s | missing-sections(title+desc...   |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |              6681 |                   500 |           7181 |          784 |      55.7 |         8.4 |           17.83s |      1.29s |      19.31s | ⚠️harness(stop_token), ...       |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |              4659 |                   500 |           5159 |        3,485 |      31.5 |         4.6 |           17.98s |      1.19s |      19.55s | repetitive(phrase: "- outpu...   |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |              6681 |                   500 |           7181 |          847 |        43 |          11 |           19.83s |      1.38s |      21.39s | degeneration, ...                |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |              1753 |                   190 |           1943 |          102 |      49.7 |          41 |           21.74s |      1.25s |      23.19s | title-length(43), ...            |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |              1753 |                   190 |           1943 |          102 |      30.1 |          48 |           24.26s |      1.70s |      26.17s | title-length(43), ...            |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               537 |                   116 |            653 |          209 |      5.03 |          25 |           26.01s |      2.44s |      28.63s | missing-sections(title+desc...   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |             16794 |                   500 |          17294 |          889 |      53.9 |          13 |           28.76s |      1.11s |      30.06s | ⚠️harness(long_context), ...     |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |              1587 |                   500 |           2087 |        3,167 |      17.6 |          11 |           29.35s |      1.48s |      31.06s | repetitive(phrase: "the bot...   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |              6681 |                   500 |           7181 |          311 |      32.8 |          78 |           37.13s |      9.74s |      47.09s | missing-sections(title+desc...   |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |             16807 |                   500 |          17307 |          287 |      87.5 |          12 |           64.85s |      1.38s |      66.43s | missing-sections(title+desc...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |             16807 |                   500 |          17307 |          284 |      95.3 |          26 |           65.02s |      2.53s |      67.77s | missing-sections(title+desc...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |             16807 |                   500 |          17307 |          277 |        77 |          35 |           67.74s |      3.13s |      71.08s | missing-sections(title+desc...   |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |             16794 |                     2 |          16796 |          235 |       289 |         5.1 |           71.97s |      0.55s |      72.70s | ⚠️harness(stop_token), ...       |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               825 |                   500 |           1325 |          186 |      7.35 |          65 |           72.82s |      7.26s |      80.28s | repetitive(phrase: "- the i...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |             16807 |                   500 |          17307 |          256 |      63.4 |          76 |           74.28s |     11.97s |      86.43s | missing-sections(title+desc...   |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |             16807 |                   500 |          17307 |          204 |      27.2 |          26 |          101.42s |      2.24s |     103.87s | hallucination, degeneration, ... |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |             16807 |                   500 |          17307 |          206 |        17 |          39 |          111.63s |      2.91s |     114.75s | cutoff                           |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |             16807 |                   500 |          17307 |          196 |      17.2 |          39 |          115.51s |      3.07s |     118.79s | missing-sections(title+desc...   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |              1559 |                   500 |           2059 |        1,045 |      4.27 |          39 |          119.02s |      3.31s |     122.52s | missing-sections(title+desc...   |                 |

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
- `mlx`: `0.32.0.dev20260502+e8ebdebe`
- `mlx-vlm`: `0.4.5`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.13.0`
- `transformers`: `5.7.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-03 02:16:08 BST_
