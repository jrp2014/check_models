# Model Performance Results

_Generated on 2026-04-12 01:05:39 BST_

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
- _Quality signal frequency:_ missing_sections=35, metadata_borrowing=25,
  cutoff=25, trusted_hint_ignored=24, context_ignored=24, repetitive=12.
- _Runtime pattern:_ decode dominates measured phase time (90%; 52/53 measured
  model(s)).
- _Phase totals:_ model load=118.35s, prompt prep=0.17s, decode=1132.03s,
  cleanup=5.52s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=52, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (366.0 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.48s)
- **📊 Average TPS:** 83.9 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1056.4 GB
- **Average peak memory:** 20.3 GB
- **Memory efficiency:** 253 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 15 | ✅ B: 14 | 🟡 C: 8 | 🟠 D: 4 | ❌ F: 11

**Average Utility Score:** 62/100

**Existing Metadata Baseline:** ❌ F (16/100)
**Vs Existing Metadata:** Avg Δ +46 | Better: 46, Neutral: 1, Worse: 5

- **Best for cataloging:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (🏆 A, 98/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/LFM2.5-VL-1.6B-bf16` (87/100)
- **Worst for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (❌ F, 0/100)

### ⚠️ 15 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
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
- **🔄 Repetitive Output (12):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "weatherproof, weatherproof, we..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (token: `0.`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "a photograph of a..."`)
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

- **Generation Tps**: Avg: 83.9 | Min: 5.03 | Max: 366
- **Peak Memory**: Avg: 20 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 24.31s | Min: 1.58s | Max: 112.06s
- **Generation Time**: Avg: 21.77s | Min: 0.78s | Max: 108.63s
- **Model Load Time**: Avg: 2.23s | Min: 0.48s | Max: 12.52s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 52/53 measured model(s)).
- **Phase totals:** model load=118.35s, prompt prep=0.17s, decode=1132.03s, cleanup=5.52s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=52, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 16.11s total (avg 0.30s across 53 model(s)).
- **First-token latency:** Avg 12.50s | Min 0.08s | Max 83.15s across 52 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 98/100 | Desc 87 | Keywords 82 | Gen 66.4 TPS | Peak 13 | A 98/100 |
  nontext prompt burden=87% | missing terms: Alton, United, Kingdom)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 98/100 | Desc 87 | Keywords 82 | Gen 66.4 TPS | Peak 13 | A 98/100 |
  nontext prompt burden=87% | missing terms: Alton, United, Kingdom)
- _Best keywording:_ [`mlx-community/pixtral-12b-8bit`](model_gallery.md#model-mlx-community-pixtral-12b-8bit)
  (A 93/100 | Desc 85 | Keywords 82 | Gen 39.2 TPS | Peak 16 | A 93/100 |
  nontext prompt burden=88% | missing terms: Centre, Alton, United, Kingdom |
  keywords=19)
- _Fastest generation:_ [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit)
  (F 5/100 | Desc 21 | Keywords 0 | Gen 366 TPS | Peak 5.1 | F 5/100 | Special
  control token &lt;|endoftext|&gt; appeared in generated text. | Output
  appears truncated to about 2 tokens. | nontext prompt burden=98% | missing
  terms: Town, Centre, Alton, United, Kingdom)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (F 30/100 | Desc 51 | Keywords 0 | Gen 331 TPS | Peak 2.1 | F 30/100 |
  missing sections: title, description, keywords | missing terms: Town,
  Centre, Alton, United, Kingdom)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 98/100 | Desc 87 | Keywords 82 | Gen 66.4 TPS | Peak 13 | A 98/100 |
  nontext prompt burden=87% | missing terms: Alton, United, Kingdom)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
- _🔄 Repetitive Output (12):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16),
  +8 more. Example: token: `unt`.
- _👻 Hallucinations (2):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8).
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (15):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  +11 more. Common weakness: Keywords are not specific or diverse enough.

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

_Overall runtime:_ 1273.01s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.24s |       2.68s |                                    |    model-config |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               711 |                    96 |            807 |        7,047 |       331 |         2.9 |            0.78s |      0.51s |       1.58s | missing-sections(title+desc...     |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               458 |                   146 |            604 |        4,900 |       352 |         2.5 |            0.99s |      0.48s |       1.77s | fabrication, title-length(11), ... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               462 |                    22 |            484 |        4,644 |       331 |         2.1 |            1.08s |      0.58s |       1.96s | missing-sections(title+desc...     |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               521 |                   127 |            648 |        6,259 |       188 |         3.8 |            1.14s |      0.62s |       2.06s | description-sentences(3), ...      |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,049 |                   145 |          3,194 |        2,874 |       181 |         7.8 |            2.36s |      0.98s |       3.65s | metadata-borrowing, ...            |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,484 |                    29 |          1,513 |        1,337 |      32.3 |          12 |            2.55s |      1.84s |       4.70s | missing-sections(title+desc...     |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,652 |                     8 |          2,660 |        1,292 |      70.4 |         9.7 |            2.77s |      0.91s |       3.98s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |             1,265 |                   120 |          1,385 |        3,915 |      54.8 |         9.4 |            2.93s |      0.87s |       4.14s | keyword-count(21), ...             |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,241 |                    45 |          2,286 |        1,258 |        32 |          18 |            3.72s |      1.75s |       5.78s | title-length(1), ...               |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,484 |                    13 |          1,497 |        1,054 |      5.61 |          27 |            4.24s |      2.59s |       7.15s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,050 |                   109 |          3,159 |        1,278 |      66.4 |          13 |            4.53s |      1.30s |       6.14s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,240 |                    85 |          3,325 |        1,730 |      39.2 |          16 |            4.56s |      1.94s |       6.82s | keyword-count(19)                  |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               567 |                   500 |          1,067 |        1,682 |       132 |         5.5 |            4.71s |      0.59s |       5.61s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               728 |                    92 |            820 |          574 |      29.5 |          19 |            4.87s |      2.52s |       7.74s | metadata-borrowing                 |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,812 |               719 |                   500 |          1,219 |        2,312 |       118 |           6 |            4.99s |      1.42s |       6.73s | repetitive(phrase: "16:41:4...     |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,666 |                   500 |          2,166 |        3,921 |       125 |         5.5 |            5.02s |      0.62s |       5.92s | repetitive(unt), ...               |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,050 |                   135 |          3,185 |        1,272 |      63.3 |          13 |            5.03s |      1.32s |       6.64s | metadata-borrowing, ...            |                 |
| `qnguyen3/nanoLLaVA`                                    |  54,043 |               458 |                   500 |            958 |        4,216 |       112 |         4.6 |            5.04s |      0.55s |       5.89s | repetitive(phrase: "paintin...     |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,666 |                   500 |          2,166 |        3,930 |       122 |         5.5 |            5.17s |      0.57s |       6.06s | repetitive(unt), ...               |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               727 |                   208 |            935 |        1,703 |      47.9 |          17 |            5.20s |      2.21s |       7.73s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               734 |                    95 |            829 |          572 |      27.1 |          20 |            5.26s |      2.54s |       8.13s | metadata-borrowing                 |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |  24,208 |             1,451 |                   500 |          1,951 |        1,740 |       116 |          22 |            5.71s |      2.09s |       8.09s | repetitive(phrase: "a photo...     |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               429 |                    90 |            519 |          274 |      21.8 |          15 |            6.12s |      1.49s |       7.91s | missing-sections(title+desc...     |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |  21,474 |             1,451 |                   500 |          1,951 |          877 |       127 |          18 |            6.22s |      2.79s |       9.32s | missing-sections(title+desc...     |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,551 |                   100 |          2,651 |          737 |      31.8 |          22 |            7.14s |      2.04s |       9.50s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               728 |                    99 |            827 |          517 |      17.4 |          33 |            7.57s |      3.37s |      11.25s | metadata-borrowing                 |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      13 |             1,451 |                   500 |          1,951 |        1,688 |      77.6 |          37 |            7.94s |      3.22s |      11.46s | repetitive(0.), degeneration, ...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |   2,798 |             1,265 |                   500 |          1,765 |        3,618 |      55.3 |         9.4 |            9.82s |      0.88s |      11.00s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   5,966 |             4,556 |                   500 |          5,056 |        3,814 |      39.3 |         4.6 |           14.62s |      1.13s |      16.06s | repetitive(phrase: "- outpu...     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,294 |             3,331 |                   500 |          3,831 |        1,464 |      41.7 |          15 |           14.77s |      1.62s |      16.70s | degeneration, ...                  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |   7,196 |             6,508 |                   500 |          7,008 |          898 |      66.2 |         8.4 |           15.21s |      1.29s |      16.80s | missing-sections(title+desc...     |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  37,586 |             1,769 |                   500 |          2,269 |          302 |      57.4 |          60 |           15.35s |     10.16s |      25.82s | missing-sections(title+desc...     |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,740 |                   500 |          3,240 |        2,369 |      32.4 |          19 |           17.14s |      1.88s |      19.32s | missing-sections(title+desc...     |                 |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,240 |                   260 |          3,500 |        2,016 |      20.2 |          28 |           17.50s |      2.51s |      20.31s | missing-sections(title+desc...     |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,637 |                   178 |          1,815 |          120 |        51 |          41 |           17.95s |      1.23s |      19.48s | description-sentences(3), ...      |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  32,607 |             6,508 |                   500 |          7,008 |          831 |      46.6 |          11 |           19.00s |      1.43s |      20.73s | ⚠️harness(stop_token), ...         |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  15,780 |            16,675 |                   500 |         17,175 |        1,200 |      86.8 |         8.6 |           20.75s |      0.70s |      21.74s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     525 |            16,677 |                   500 |         17,177 |          980 |      86.1 |         8.6 |           23.77s |      0.71s |      24.79s | ⚠️harness(long_context), ...       |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               430 |                   109 |            539 |          190 |      5.03 |          25 |           24.40s |      2.71s |      27.43s | missing-sections(title+desc...     |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |             2,241 |                   500 |          2,741 |        2,751 |      34.1 |          18 |           26.44s |      1.68s |      28.42s | repetitive(phrase: "rencont...     |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |     664 |             1,484 |                   500 |          1,984 |        3,272 |        19 |          11 |           27.30s |      1.42s |      29.03s | repetitive(phrase: "- do no...     |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |      12 |            16,686 |                   500 |         17,186 |          965 |        53 |          13 |           27.54s |      1.13s |      28.97s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |   8,696 |             6,508 |                   500 |          7,008 |          313 |        34 |          78 |           36.03s |      8.54s |      44.88s | missing-sections(title+desc...     |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,637 |                   458 |          2,095 |          109 |      29.6 |          48 |           47.20s |      1.79s |      49.38s | degeneration, ...                  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |   3,924 |            16,700 |                   500 |         17,200 |          282 |       106 |          26 |           64.82s |      2.46s |      67.60s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |  14,448 |            16,700 |                   500 |         17,200 |          285 |      81.3 |          35 |           65.57s |      3.10s |      68.98s | missing-sections(title+desc...     |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |  10,124 |            16,700 |                   500 |         17,200 |          265 |      83.9 |          12 |           69.84s |      1.35s |      71.51s | fabrication, ...                   |                 |
| `mlx-community/gemma-4-31b-bf16`                        | 236,832 |               721 |                   500 |          1,221 |          139 |      7.26 |          64 |           74.64s |      8.36s |      83.32s | repetitive(phrase: "51.1478...     |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,686 |                     2 |         16,688 |          219 |       366 |         5.1 |           77.17s |      0.51s |      77.98s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |      12 |            16,700 |                   500 |         17,200 |          246 |      61.4 |          76 |           77.20s |     12.52s |      90.04s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |       9 |            16,700 |                   500 |         17,200 |          201 |      28.3 |          26 |          101.74s |      2.12s |     104.24s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |  16,866 |            16,700 |                   500 |         17,200 |          213 |      17.1 |          39 |          108.63s |      3.10s |     112.06s | refusal(explicit_refusal), ...     |                 |

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

_Report generated on: 2026-04-12 01:05:39 BST_
