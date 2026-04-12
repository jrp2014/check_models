# Model Performance Results

_Generated on 2026-04-12 01:33:02 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=1/52.
- _Useful now:_ 2 clean A/B model(s) worth first review.
- _Review watchlist:_ 50 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=46, neutral=1, worse=5 (baseline F 16/100).
- _Quality signal frequency:_ missing_sections=36, cutoff=26,
  trusted_hint_ignored=25, context_ignored=24, metadata_borrowing=24,
  repetitive=13.
- _Runtime pattern:_ decode dominates measured phase time (90%; 52/53 measured
  model(s)).
- _Phase totals:_ model load=110.81s, prompt prep=0.17s, decode=1093.37s,
  cleanup=5.49s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=52, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (342.6 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/LFM2-VL-1.6B-8bit` (0.48s)
- **📊 Average TPS:** 77.3 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1056.3 GB
- **Average peak memory:** 20.3 GB
- **Memory efficiency:** 237 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 14 | ✅ B: 15 | 🟡 C: 7 | 🟠 D: 5 | ❌ F: 11

**Average Utility Score:** 61/100

**Existing Metadata Baseline:** ❌ F (16/100)
**Vs Existing Metadata:** Avg Δ +45 | Better: 46, Neutral: 1, Worse: 5

- **Best for cataloging:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (🏆 A, 98/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/LFM2.5-VL-1.6B-bf16` (87/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 0/100)

### ⚠️ 16 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (30/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (16/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (19/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) - Output lacks detail
- `qnguyen3/nanoLLaVA`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (13):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "weatherproof, weatherproof, we..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (token: `0.`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "a photograph of a..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "局 局 局 局..."`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `phrase: "with a glass front...."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "16:41:41: 16:41:41: 16:41:41: ..."`)
  - `mlx-community/gemma-4-31b-bf16` (token: `phrase: "51.147833, -0.978750, 51.14783..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "- do not use..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `qnguyen3/nanoLLaVA` (token: `phrase: "painting painting painting pai..."`)
- **👻 Hallucinations (2):**
  - `microsoft/Phi-3.5-vision-instruct`
  - `mlx-community/Qwen3.5-27B-mxfp8`
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 77.3 | Min: 0 | Max: 343
- **Peak Memory**: Avg: 20 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 23.42s | Min: 1.53s | Max: 113.64s
- **Generation Time**: Avg: 21.03s | Min: 0.76s | Max: 111.14s
- **Model Load Time**: Avg: 2.09s | Min: 0.48s | Max: 12.60s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 52/53 measured model(s)).
- **Phase totals:** model load=110.81s, prompt prep=0.17s, decode=1093.37s, cleanup=5.49s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=52, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 15.97s total (avg 0.30s across 53 model(s)).
- **First-token latency:** Avg 10.72s | Min 0.08s | Max 92.39s across 51 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 98/100 | Desc 87 | Keywords 82 | Gen 66.3 TPS | Peak 13 | A 98/100 |
  nontext prompt burden=87% | missing terms: Alton, United, Kingdom)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 98/100 | Desc 87 | Keywords 82 | Gen 66.3 TPS | Peak 13 | A 98/100 |
  nontext prompt burden=87% | missing terms: Alton, United, Kingdom)
- _Best keywording:_ [`mlx-community/pixtral-12b-8bit`](model_gallery.md#model-mlx-community-pixtral-12b-8bit)
  (A 93/100 | Desc 85 | Keywords 82 | Gen 39.0 TPS | Peak 16 | A 93/100 |
  nontext prompt burden=88% | missing terms: Centre, Alton, United, Kingdom |
  keywords=19)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (B 78/100 | Desc 47 | Keywords 82 | Gen 343 TPS | Peak 2.5 | B 78/100 |
  nonvisual metadata reused)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (F 30/100 | Desc 51 | Keywords 0 | Gen 334 TPS | Peak 2.2 | F 30/100 |
  missing sections: title, description, keywords | missing terms: Town,
  Centre, Alton, United, Kingdom)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 98/100 | Desc 87 | Keywords 82 | Gen 66.3 TPS | Peak 13 | A 98/100 |
  nontext prompt burden=87% | missing terms: Alton, United, Kingdom)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
- _🔄 Repetitive Output (13):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16),
  +9 more. Example: token: `unt`.
- _👻 Hallucinations (2):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8).
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (16):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  +12 more. Common weakness: Keywords are not specific or diverse enough.

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
> Return exactly these three sections, and nothing else:
>
> Title:
> &#45; 5-10 words, concrete and factual, limited to clearly visible content.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> Description:
> &#45; 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> &#45; Output only the description text after the label.
>
> Keywords:
> &#45; 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> &#45; Output only the keyword list after the label.
>
> Rules:
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
> confirmed):
> &#45; Description hint: , Town Centre, Alton, England, United Kingdom, UK
> &#45; Capture metadata: Taken on 2026-04-11 16:46:40 BST (at 16:46:40 local
> time). GPS: 51.147833°N, 0.978750°W.
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1226.71s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.24s |       2.54s |                                    |    model-config |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               711 |                    96 |            807 |         7250 |       333 |         2.9 |            0.76s |      0.48s |       1.53s | missing-sections(title+desc...     |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               462 |                    22 |            484 |         4692 |       334 |         2.2 |            1.00s |      0.58s |       1.88s | missing-sections(title+desc...     |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               458 |                   146 |            604 |         4606 |       343 |         2.5 |            1.01s |      0.48s |       1.79s | fabrication, title-length(11), ... |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               521 |                   127 |            648 |         6487 |       192 |         3.8 |            1.11s |      0.69s |       2.09s | description-sentences(3), ...      |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |              3049 |                   145 |           3194 |         2969 |       184 |         7.8 |            2.32s |      0.94s |       3.56s | metadata-borrowing, ...            |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |              1484 |                    29 |           1513 |         1354 |      32.6 |          12 |            2.48s |      1.56s |       4.35s | missing-sections(title+desc...     |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |              2652 |                     8 |           2660 |         1297 |      70.1 |         9.7 |            2.78s |      0.90s |       3.98s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |              1265 |                   120 |           1385 |         3941 |      54.5 |         9.4 |            2.93s |      0.85s |       4.07s | keyword-count(21), ...             |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |              2241 |                    45 |           2286 |         1067 |      32.4 |          18 |            4.03s |      1.79s |       6.19s | title-length(1), ...               |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |              1484 |                    13 |           1497 |         1074 |      5.56 |          27 |            4.22s |      2.45s |       6.97s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |              3050 |                   109 |           3159 |         1397 |      66.3 |          13 |            4.31s |      1.32s |       5.93s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |              3240 |                    85 |           3325 |         1784 |        39 |          16 |            4.51s |      1.69s |       6.51s | keyword-count(19)                  |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               728 |                    92 |            820 |          622 |        31 |          19 |            4.59s |      2.29s |       7.19s | metadata-borrowing                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |              3050 |                   135 |           3185 |         1416 |      63.2 |          13 |            4.78s |      1.34s |       6.43s | metadata-borrowing, ...            |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               567 |                   500 |           1067 |         1673 |       128 |         5.5 |            4.81s |      0.63s |       5.75s | missing-sections(title+desc...     |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |              1666 |                   500 |           2166 |         4014 |       124 |         5.5 |            5.00s |      0.61s |       5.89s | repetitive(unt), ...               |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,812 |               719 |                   500 |           1219 |         2345 |       117 |           6 |            5.00s |      1.40s |       6.71s | repetitive(phrase: "16:41:4...     |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |              1666 |                   500 |           2166 |         3973 |       123 |         5.5 |            5.07s |      0.57s |       5.95s | repetitive(unt), ...               |                 |
| `qnguyen3/nanoLLaVA`                                    |  54,043 |               458 |                   500 |            958 |         3969 |       110 |         4.5 |            5.16s |      0.54s |       5.99s | repetitive(phrase: "paintin...     |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               727 |                   208 |            935 |         1722 |        47 |          17 |            5.28s |      2.27s |       7.86s | missing-sections(title+desc...     |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |  21,474 |              1451 |                   500 |           1951 |         1735 |       127 |          18 |            5.31s |      1.94s |       7.55s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               734 |                    95 |            829 |          545 |      26.8 |          20 |            5.36s |      2.50s |       8.19s | metadata-borrowing                 |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |  24,208 |              1451 |                   500 |           1951 |         1744 |       116 |          22 |            5.70s |      2.09s |       8.08s | repetitive(phrase: "a photo...     |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               429 |                    90 |            519 |          266 |        20 |          15 |            6.53s |      1.49s |       8.30s | missing-sections(title+desc...     |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |              2551 |                   100 |           2651 |          692 |      31.6 |          22 |            7.38s |      2.07s |       9.78s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               728 |                    99 |            827 |          536 |      17.6 |          33 |            7.42s |      3.49s |      11.23s | metadata-borrowing                 |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      13 |              1451 |                   500 |           1951 |         1711 |      79.3 |          37 |            7.79s |      3.29s |      11.37s | repetitive(0.), degeneration, ...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |   2,798 |              1265 |                   500 |           1765 |         3839 |      56.5 |         9.4 |            9.59s |      0.88s |      10.76s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  37,586 |              1769 |                   500 |           2269 |          805 |      57.8 |          60 |           11.59s |      5.24s |      17.15s | missing-sections(title+desc...     |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |   7,196 |              6508 |                   500 |           7008 |         1114 |      63.8 |         8.4 |           14.10s |      1.27s |      15.66s | missing-sections(title+desc...     |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   5,966 |              4556 |                   500 |           5056 |         3794 |      39.1 |         4.6 |           14.68s |      1.15s |      16.15s | repetitive(phrase: "- outpu...     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,294 |              3331 |                   500 |           3831 |         1449 |      41.9 |          15 |           14.74s |      1.60s |      16.64s | degeneration, ...                  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  32,607 |              6508 |                   500 |           7008 |         1162 |      54.5 |          11 |           15.21s |      1.48s |      16.99s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |              2740 |                   500 |           3240 |         2436 |      31.8 |          19 |           17.39s |      1.88s |      19.59s | missing-sections(title+desc...     |                 |
| `mlx-community/pixtral-12b-bf16`                        |       2 |              3240 |                   260 |           3500 |         2020 |      20.4 |          28 |           17.40s |      2.53s |      20.23s | missing-sections(title+desc...     |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  15,780 |             16675 |                   500 |          17175 |         1227 |      89.1 |         8.6 |           20.01s |      0.72s |      21.02s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     525 |             16677 |                   500 |          17177 |         1129 |      86.6 |         8.6 |           21.48s |      0.75s |      22.67s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |              1637 |                   165 |           1802 |         86.8 |      51.6 |          41 |           22.84s |      1.25s |      24.38s | missing-sections(title+desc...     |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               430 |                   109 |            539 |          228 |      5.02 |          25 |           24.05s |      2.19s |      26.53s | missing-sections(title+desc...     |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |              2241 |                   500 |           2741 |         2982 |      33.9 |          18 |           26.44s |      1.69s |      28.42s | repetitive(phrase: "rencont...     |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |     664 |              1484 |                   500 |           1984 |         3261 |      18.8 |          11 |           27.59s |      1.42s |      29.31s | repetitive(phrase: "- do no...     |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |      12 |             16686 |                   500 |          17186 |          893 |      55.4 |          13 |           28.51s |      1.12s |      29.93s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |   8,696 |              6508 |                   500 |           7008 |          449 |      35.5 |          78 |           29.01s |      6.60s |      35.95s | missing-sections(title+desc...     |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |     198 |              1637 |                   500 |           2137 |         92.4 |        30 |          48 |           53.40s |      1.71s |      55.40s | repetitive(phrase: "局 局 局 局... |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |  14,448 |             16700 |                   500 |          17200 |          318 |      89.1 |          35 |           59.01s |      2.97s |      62.28s | missing-sections(title+desc...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |   3,924 |             16700 |                   500 |          17200 |          313 |       105 |          26 |           59.05s |      2.35s |      61.70s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |  10,124 |             16700 |                   500 |          17200 |          300 |      88.6 |          12 |           62.22s |      1.21s |      63.74s | fabrication, ...                   |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                 0 |                     0 |              0 |            0 |         0 |         5.1 |           66.63s |      0.50s |      67.47s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |      12 |             16700 |                   500 |          17200 |          280 |      63.9 |          76 |           68.56s |     12.60s |      81.45s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/gemma-4-31b-bf16`                        | 236,832 |               721 |                   500 |           1221 |          115 |      7.39 |          64 |           74.62s |     10.05s |      84.98s | repetitive(phrase: "51.1478...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |  16,866 |             16700 |                   500 |          17200 |          217 |      16.9 |          39 |          107.46s |      3.09s |     110.86s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |       9 |             16700 |                   500 |          17200 |          181 |        28 |          26 |          111.14s |      2.11s |     113.64s | refusal(explicit_refusal), ...     |                 |

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

_Report generated on: 2026-04-12 01:33:02 BST_
