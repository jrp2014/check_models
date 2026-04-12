# Model Performance Results

_Generated on 2026-04-12 21:48:31 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=8, clean outputs=0/52.
- _Useful now:_ 2 clean A/B model(s) worth first review.
- _Review watchlist:_ 50 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=44, neutral=0, worse=8 (baseline F 16/100).
- _Quality signal frequency:_ missing_sections=34, metadata_borrowing=27,
  cutoff=27, trusted_hint_ignored=25, context_ignored=24, reasoning_leak=12.
- _Runtime pattern:_ decode dominates measured phase time (90%; 51/53 measured
  model(s)).
- _Phase totals:_ model load=117.51s, prompt prep=0.17s, decode=1147.60s,
  cleanup=5.29s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=52, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (356.1 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.49s)
- **📊 Average TPS:** 78.4 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1053.2 GB
- **Average peak memory:** 20.3 GB
- **Memory efficiency:** 238 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 14 | ✅ B: 14 | 🟡 C: 6 | 🟠 D: 8 | ❌ F: 10

**Average Utility Score:** 60/100

**Existing Metadata Baseline:** ❌ F (16/100)
**Vs Existing Metadata:** Avg Δ +44 | Better: 44, Neutral: 0, Worse: 8

- **Best for cataloging:** `mlx-community/gemma-4-31b-it-4bit` (🏆 A, 99/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (93/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 0/100)

### ⚠️ 18 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (44/100) - Lacks visual description of image
- `mlx-community/Molmo-7B-D-0924-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (2/100) - Output too short to be useful
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) - Output too short to be useful
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (19/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) - Output lacks detail
- `mlx-community/pixtral-12b-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (12):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "weathered, and slightly weathe..."`)
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (token: `phrase: "cultural, significance, cultur..."`)
  - `mlx-community/FastVLM-0.5B-bf16` (token: `phrase: "what is the name..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "8th record 8th record..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/gemma-4-31b-bf16` (token: `phrase: "town centre, town center,..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "- do not use..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `mlx-community/pixtral-12b-bf16` (token: `phrase: ", , , ,..."`)
  - `qnguyen3/nanoLLaVA` (token: `phrase: "drawing suppliesmithic drawing..."`)
- **👻 Hallucinations (4):**
  - `microsoft/Phi-3.5-vision-instruct`
  - `mlx-community/FastVLM-0.5B-bf16`
  - `mlx-community/Qwen3.5-27B-mxfp8`
  - `mlx-community/Qwen3.5-9B-MLX-4bit`
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 78.4 | Min: 0 | Max: 356
- **Peak Memory**: Avg: 20 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 24.57s | Min: 1.73s | Max: 107.41s
- **Generation Time**: Avg: 22.07s | Min: 0.73s | Max: 103.97s
- **Model Load Time**: Avg: 2.22s | Min: 0.49s | Max: 12.61s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 51/53 measured model(s)).
- **Phase totals:** model load=117.51s, prompt prep=0.17s, decode=1147.60s, cleanup=5.29s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=52, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 14.69s total (avg 0.28s across 53 model(s)).
- **First-token latency:** Avg 10.33s | Min 0.08s | Max 75.37s across 51 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (A 93/100 | Desc 93 | Keywords 93 | Gen 186 TPS | Peak 7.4 | A 93/100 |
  nontext prompt burden=87% | missing terms: United, Kingdom)
- _Best descriptions:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (A 93/100 | Desc 93 | Keywords 93 | Gen 186 TPS | Peak 7.4 | A 93/100 |
  nontext prompt burden=87% | missing terms: United, Kingdom)
- _Best keywording:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (A 93/100 | Desc 93 | Keywords 93 | Gen 186 TPS | Peak 7.4 | A 93/100 |
  nontext prompt burden=87% | missing terms: United, Kingdom)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (C 53/100 | Desc 53 | Keywords 45 | Gen 356 TPS | Peak 2.5 | C 53/100 |
  keywords=32 | nonvisual metadata reused | reasoning leak)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (B 80/100 | Desc 67 | Keywords 0 | Gen 346 TPS | Peak 2.1 | B 80/100 | hit
  token cap (500) | output/prompt=108.23% | missing sections: title,
  description, keywords | missing terms: Town, Centre, Alton, United, Kingdom)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (A 93/100 | Desc 93 | Keywords 93 | Gen 186 TPS | Peak 7.4 | A 93/100 |
  nontext prompt burden=87% | missing terms: United, Kingdom)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
- _🔄 Repetitive Output (12):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  +8 more. Example: token: `unt`.
- _👻 Hallucinations (4):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8),
  [`mlx-community/Qwen3.5-9B-MLX-4bit`](model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit).
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (18):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +14 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `model-config` | 1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16` |

### Actionable Items by Package

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD028 MD037 -->
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
> &#45; Description hint: , Town Centre, Alton, England, United Kingdom, UK
> &#45; Capture metadata: Taken on 2026-04-11 16:10:22 BST (at 16:10:22 local
> time). GPS: 51.150083°N, 0.973833°W.
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1286.10s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.21s |       2.49s |                                 |    model-config |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               711 |                    85 |            796 |         7238 |       332 |         2.9 |            0.73s |      0.73s |       1.73s | fabrication, ...                |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               521 |                   127 |            648 |         6410 |       188 |         3.8 |            1.11s |      0.55s |       1.93s | description-sentences(3), ...   |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               458 |                   243 |            701 |         4888 |       356 |         2.5 |            1.20s |      0.49s |       1.97s | title-length(22), ...           |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |              2937 |                   143 |           3080 |         3030 |       186 |         7.4 |            2.22s |      0.96s |       3.47s | description-sentences(3)        |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |              1484 |                    25 |           1509 |         1371 |      30.9 |          12 |            2.34s |      1.59s |       4.22s | missing-sections(title+desc...  |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |      13 |               462 |                   500 |            962 |         4673 |       346 |         2.1 |            2.38s |      0.61s |       3.28s | repetitive(phrase: "what is...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |              2554 |                     7 |           2561 |         1249 |      71.4 |         9.7 |            2.70s |      0.95s |       3.93s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               727 |                   106 |            833 |         1723 |      48.6 |          17 |            3.00s |      2.23s |       5.53s | missing-sections(title+desc...  |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |              1265 |                   130 |           1395 |         4500 |      56.1 |         9.4 |            3.02s |      0.85s |       4.14s | keyword-count(22), ...          |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |              2241 |                    46 |           2287 |         1404 |      32.6 |          18 |            3.49s |      1.81s |       5.58s | title-length(2), ...            |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |              1484 |                    12 |           1496 |         1067 |      5.65 |          27 |            3.98s |      2.43s |       6.71s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               728 |                    85 |            813 |          626 |      31.4 |          19 |            4.28s |      2.32s |       6.89s | metadata-borrowing              |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |              2938 |                   117 |           3055 |         1425 |      63.6 |          13 |            4.38s |      1.37s |       6.03s | metadata-borrowing              |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               567 |                   500 |           1067 |         1661 |       132 |         5.5 |            4.68s |      0.59s |       5.56s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |              2938 |                   143 |           3081 |         1424 |      65.4 |          13 |            4.72s |      1.36s |       6.38s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               719 |                   500 |           1219 |         2366 |       124 |           6 |            4.73s |      1.43s |       6.45s | degeneration, ...               |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |              1666 |                   500 |           2166 |         3977 |       125 |         5.5 |            4.93s |      0.64s |       5.83s | repetitive(unt), ...            |                 |
| `qnguyen3/nanoLLaVA`                                    |  50,252 |               458 |                   500 |            958 |         4149 |       113 |         4.7 |            4.99s |      0.56s |       5.82s | repetitive(phrase: "drawing...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |              1666 |                   500 |           2166 |         4028 |       124 |         5.5 |            5.00s |      0.63s |       5.91s | repetitive(unt), ...            |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |              3175 |                   112 |           3287 |         1817 |      39.2 |          16 |            5.07s |      1.70s |       7.07s | description-sentences(3)        |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |  21,474 |              1412 |                   500 |           1912 |         1718 |       127 |          18 |            5.28s |      2.06s |       7.61s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               734 |                   101 |            835 |          571 |      27.1 |          20 |            5.44s |      2.55s |       8.30s | metadata-borrowing              |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     220 |              1412 |                   500 |           1912 |         1765 |       115 |          22 |            5.68s |      2.17s |       8.13s | repetitive(phrase: "8th rec...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               728 |                    98 |            826 |          503 |      17.7 |          33 |            7.44s |      3.47s |      11.20s | metadata-borrowing              |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |              2439 |                   108 |           2547 |          690 |      30.9 |          22 |            7.52s |      2.12s |       9.94s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      13 |              1412 |                   500 |           1912 |         1742 |      80.1 |          37 |            7.66s |      3.30s |      11.23s | degeneration, ...               |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               429 |                   126 |            555 |          275 |      21.9 |          15 |            7.73s |      1.52s |       9.52s | missing-sections(title+desc...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |   6,332 |              1265 |                   500 |           1765 |         3814 |      56.3 |         9.4 |            9.61s |      0.97s |      10.86s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |      12 |              6444 |                   500 |           6944 |         1116 |      69.9 |         8.4 |           13.34s |      1.33s |      14.96s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   5,966 |              4556 |                   500 |           5056 |         3824 |      42.4 |         4.6 |           13.67s |      1.13s |      15.08s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,681 |              3266 |                   500 |           3766 |         1465 |      41.3 |          15 |           14.79s |      1.60s |      16.68s | degeneration, ...               |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  50,383 |              6444 |                   500 |           6944 |         1159 |      53.9 |          11 |           15.24s |      1.37s |      16.90s | missing-sections(title+desc...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  93,938 |              1754 |                   500 |           2254 |          266 |      59.2 |          60 |           15.73s |      9.90s |      25.92s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |              2740 |                   500 |           3240 |         2368 |      32.1 |          19 |           17.29s |      1.93s |      19.50s | missing-sections(title+desc...  |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |   9,104 |             16669 |                   500 |          17169 |         1244 |      86.8 |         8.6 |           20.03s |      0.74s |      21.04s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | 151,645 |             16680 |                   138 |          16818 |          987 |      57.6 |          13 |           20.07s |      1.16s |      21.51s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |              1637 |                   126 |           1763 |         91.5 |      51.3 |          41 |           21.07s |      1.21s |      22.56s | description-sentences(4), ...   |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      11 |             16671 |                   500 |          17171 |         1132 |      88.2 |         8.6 |           21.27s |      0.73s |      22.30s | fabrication, ...                |                 |
| `mlx-community/pixtral-12b-bf16`                        |  59,977 |              3175 |                   500 |           3675 |         1868 |        20 |          27 |           27.18s |      2.67s |      30.12s | repetitive(phrase: ", , , ,...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |     664 |              1484 |                   500 |           1984 |         3206 |        19 |          11 |           27.40s |      1.48s |      29.17s | repetitive(phrase: "- do no...  |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |              2241 |                   500 |           2741 |         2977 |      33.8 |          18 |           29.20s |      1.81s |      31.29s | repetitive(phrase: "rencont...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |   7,039 |              6444 |                   500 |           6944 |          395 |      37.8 |          78 |           30.03s |      8.94s |      39.26s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |              1637 |                   485 |           2122 |         87.7 |      30.3 |          48 |           53.98s |      1.76s |      56.02s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |   3,874 |             16694 |                   500 |          17194 |          316 |       106 |          26 |           58.47s |      2.48s |      61.23s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |  21,921 |             16694 |                   500 |          17194 |          313 |      89.6 |          35 |           59.76s |      3.14s |      63.18s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                 0 |                     0 |              0 |            0 |         0 |         5.1 |           61.14s |      0.54s |      61.95s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |      16 |             16694 |                   500 |          17194 |          303 |      90.9 |          12 |           61.37s |      1.54s |      63.20s | hallucination, fabrication, ... |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |      11 |             16694 |                   500 |          17194 |          280 |        64 |          76 |           68.37s |     12.61s |      81.26s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |   5,148 |               721 |                   500 |           1221 |          155 |      7.32 |          64 |           73.41s |      7.75s |      81.45s | repetitive(phrase: "town ce...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     256 |             16694 |                   500 |          17194 |          228 |      28.1 |          26 |           91.85s |      2.19s |      94.34s | refusal(explicit_refusal), ...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |  26,431 |               430 |                   500 |            930 |          220 |      4.94 |          25 |          103.68s |      2.21s |     106.18s | repetitive(phrase: "cultura...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |      11 |             16694 |                   500 |          17194 |          221 |        18 |          39 |          103.97s |      3.13s |     107.41s | refusal(explicit_refusal), ...  |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Automated review digest:_ [review.md](review.md)
- _Canonical run log:_ [check_models.log](check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.4.0
- _macOS Version:_ 26.4.1
- _SDK Version:_ 26.4
- _Xcode Version:_ 26.4
- _Xcode Build:_ 17E192
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
- `mlx`: `0.31.2.dev20260412+520cea2b`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.10.1`
- `transformers`: `5.5.3`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-12 21:48:31 BST_
