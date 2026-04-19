# Model Performance Results

_Generated on 2026-04-19 21:12:47 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=11, clean outputs=1/52.
- _Useful now:_ 1 clean A/B model(s) worth first review.
- _Review watchlist:_ 51 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.

### Quality & Metadata

- _Vs existing metadata:_ better=8, neutral=0, worse=44 (baseline B 75/100).
- _Quality signal frequency:_ missing_sections=34, metadata_borrowing=33,
  cutoff=27, trusted_hint_ignored=22, context_ignored=22, harness=11.
- _Termination reasons:_ completed=52, exception=2.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (89%; 50/54 measured
  model(s)).
- _Phase totals:_ model load=130.41s, prompt prep=0.18s, decode=1103.64s,
  cleanup=5.70s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=52, exception=2.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (316.1 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (0.51s)
- **📊 Average TPS:** 78.2 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1058.6 GB
- **Average peak memory:** 20.4 GB
- **Memory efficiency:** 258 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 7 | ✅ B: 10 | 🟡 C: 11 | 🟠 D: 11 | ❌ F: 13

**Average Utility Score:** 50/100

**Existing Metadata Baseline:** ✅ B (75/100)
**Vs Existing Metadata:** Avg Δ -25 | Better: 8, Neutral: 0, Worse: 44

- **Best for cataloging:** `mlx-community/InternVL3-14B-8bit` (🏆 A, 91/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (89/100)
- **Worst for cataloging:** `mlx-community/FastVLM-0.5B-bf16` (❌ F, 0/100)

### ⚠️ 24 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (42/100) - Keywords are not specific or diverse enough
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: ❌ F (25/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (41/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2.5-VL-1.6B-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (42/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (30/100) - Lacks visual description of image
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (10/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (14/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-8bit`: 🟠 D (48/100) - Lacks visual description of image
- `mlx-community/pixtral-12b-bf16`: 🟠 D (41/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: ❌ F (16/100) - Output lacks detail

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `ggml-org/gemma-3-1b-it-GGUF` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (11):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "sky, stone wall, town,..."`)
  - `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (token: `它们的主要功能包括1.`)
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (token: `phrase: "waterfront, waterfront, waterf..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "windsor, england, river, thame..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "riverbank, riverbank, riverban..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "*/ */ */ */..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "18:42:10: 18:42:10: 18:42:10: ..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
- **👻 Hallucinations (2):**
  - `microsoft/Phi-3.5-vision-instruct`
  - `mlx-community/pixtral-12b-bf16`
- **📝 Formatting Issues (9):**
  - `microsoft/Phi-3.5-vision-instruct`
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`
  - `mlx-community/gemma-4-31b-bf16`
  - `mlx-community/pixtral-12b-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 78.2 | Min: 4.89 | Max: 316
- **Peak Memory**: Avg: 20 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 24.04s | Min: 2.05s | Max: 107.34s
- **Generation Time**: Avg: 21.22s | Min: 0.85s | Max: 104.82s
- **Model Load Time**: Avg: 2.46s | Min: 0.51s | Max: 12.90s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (89%; 50/54 measured model(s)).
- **Phase totals:** model load=130.41s, prompt prep=0.18s, decode=1103.64s, cleanup=5.70s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=52, exception=2.

### ⏱ Timing Snapshot

- **Validation overhead:** 18.50s total (avg 0.34s across 54 model(s)).
- **First-token latency:** Avg 11.32s | Min 0.09s | Max 71.48s across 52 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 85/100 | Desc 93 | Keywords 89 | Gen 66.6 TPS | Peak 13 | A 85/100 |
  nontext prompt burden=85% | missing terms: Berkshire, Bird, Holiday,
  Lifebuoy, Passenger)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 85/100 | Desc 93 | Keywords 89 | Gen 66.6 TPS | Peak 13 | A 85/100 |
  nontext prompt burden=85% | missing terms: Berkshire, Bird, Holiday,
  Lifebuoy, Passenger)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 85/100 | Desc 93 | Keywords 89 | Gen 66.6 TPS | Peak 13 | A 85/100 |
  nontext prompt burden=85% | missing terms: Berkshire, Bird, Holiday,
  Lifebuoy, Passenger)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (D 46/100 | Desc 73 | Keywords 0 | Gen 316 TPS | Peak 2.4 | D 46/100 |
  missing sections: keywords | missing terms: Berkshire, Bird, Blue sky,
  Holiday, Lifebuoy | nonvisual metadata reused)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (F 0/100 | Desc 0 | Keywords 0 | Gen 297 TPS | Peak 2.1 | F 0/100 | Output
  appears truncated to about 8 tokens. | missing terms: Berkshire, Bird, Blue
  sky, Castle, Holiday)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 85/100 | Desc 93 | Keywords 89 | Gen 66.6 TPS | Peak 13 | A 85/100 |
  nontext prompt burden=85% | missing terms: Berkshire, Bird, Holiday,
  Lifebuoy, Passenger)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (2):_ [`ggml-org/gemma-3-1b-it-GGUF`](model_gallery.md#model-ggml-org-gemma-3-1b-it-gguf),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Model Error`.
- _🔄 Repetitive Output (11):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  +7 more. Example: token: `unt`.
- _👻 Hallucinations (2):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/pixtral-12b-bf16`](model_gallery.md#model-mlx-community-pixtral-12b-bf16).
- _📝 Formatting Issues (9):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  +5 more.
- _Low-utility outputs (24):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  +20 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `model-config` | 2 | Model Error, Processor Error | `ggml-org/gemma-3-1b-it-GGUF`, `mlx-community/MolmoPoint-8B-fp16` |

### Actionable Items by Package

#### model-config

- ggml-org/gemma-3-1b-it-GGUF (Model Error)
  - Error: `Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snap...`
  - Type: `ValueError`
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
> &#45; Description hint: The 'Windsor Sovereign' tour boat, filled with
> passengers, cruises along the River Thames in Windsor, England. In the
> background, the historic Windsor Castle is visible behind the trees and town
> buildings, offering a scenic backdrop for tourists enjoying a river tour.
> &#45; Keyword hints: Adobe Stock, Any Vision, Berkshire, Bird, Blue sky, Castle,
> England, Europe, Holiday, Lifebuoy, Passenger, People, Quay, River Thames,
> Riverbank, Sightseeing, Sky, Stone wall, Town, Tree
> &#45; Capture metadata: Taken on 2026-04-18 18:42:10 BST (at 18:42:10 local
> time). GPS: 51.483900°N, 0.604400°W.
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1259.33s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                        |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------------|----------------:|
| `ggml-org/gemma-3-1b-it-GGUF`                           |         |                   |                       |                |              |           |             |                  |      0.13s |       0.46s |                                       |    model-config |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.17s |       2.49s |                                       |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               552 |                    51 |            603 |         3389 |       316 |         2.4 |            0.85s |      2.08s |       3.27s | missing-sections(keywords), ...       |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               556 |                     8 |            564 |         5335 |       297 |         2.1 |            1.14s |      0.57s |       2.05s | ⚠️harness(prompt_template), ...       |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               812 |                   166 |            978 |         7705 |       307 |         2.9 |            1.16s |      0.51s |       2.07s | missing-sections(title+desc...        |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,574 |                     6 |          1,580 |         3201 |      22.8 |          11 |            1.29s |      1.58s |       3.21s | ⚠️harness(prompt_template), ...       |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,574 |                    12 |          1,586 |         1341 |      30.8 |          12 |            2.10s |      1.63s |       4.07s | ⚠️harness(prompt_template), ...       |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,151 |                   145 |          3,296 |         3010 |       172 |         7.8 |            2.45s |      0.96s |       3.77s | metadata-borrowing, ...               |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |  62,963 |               622 |                   500 |          1,122 |         7159 |       184 |         3.7 |            3.31s |      0.55s |       4.26s | repetitive(phrase: "windsor...        |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,574 |                     9 |          1,583 |         1060 |      5.81 |          27 |            3.57s |      2.47s |       6.38s | ⚠️harness(prompt_template), ...       |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |             1,382 |                   180 |          1,562 |         3907 |      57.2 |         9.5 |            3.94s |      0.85s |       5.12s | fabrication, keyword-count(19), ...   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,152 |                   109 |          3,261 |         1424 |      66.6 |          13 |            4.40s |      1.33s |       6.08s |                                       |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               820 |                    81 |            901 |          651 |      30.7 |          19 |            4.47s |      2.57s |       7.42s | metadata-borrowing                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,764 |                   109 |          2,873 |         1272 |      60.6 |         9.7 |            4.64s |      0.97s |       5.96s | missing-sections(title+desc...        |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               664 |                   500 |          1,164 |         1903 |       131 |         5.5 |            4.83s |      0.77s |       5.96s | missing-sections(title+desc...        |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,335 |                    84 |          2,419 |         1403 |        32 |          18 |            4.86s |      1.82s |       7.03s | title-length(3), ...                  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,763 |                   500 |          2,263 |         4243 |       125 |         5.5 |            5.03s |      0.64s |       6.00s | repetitive(unt), ...                  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,152 |                   123 |          3,275 |         1362 |      56.4 |          13 |            5.09s |      1.47s |       6.90s | metadata-borrowing, ...               |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,763 |                   500 |          2,263 |         4187 |       122 |         5.5 |            5.16s |      0.74s |       6.24s | repetitive(unt), ...                  |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               811 |                   500 |          1,311 |         2484 |       115 |           6 |            5.17s |      1.50s |       7.02s | repetitive(phrase: "18:42:1...        |                 |
| `qnguyen3/nanoLLaVA`                                    |   3,612 |               552 |                   500 |          1,052 |         2202 |       112 |         4.6 |            5.26s |      1.73s |       7.32s | missing-sections(title+desc...        |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   6,420 |             1,544 |                   500 |          2,044 |         1812 |       126 |          18 |            5.41s |      2.00s |       7.75s | repetitive(它们的主要功能包括1.), ... |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               825 |                    97 |            922 |          573 |      26.7 |          20 |            5.60s |      2.99s |       8.96s | metadata-borrowing                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,342 |                   122 |          3,464 |         1678 |      38.9 |          16 |            5.70s |      1.95s |       8.01s | title-length(4), ...                  |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               819 |                   252 |          1,071 |         1810 |      47.2 |          17 |            6.27s |      2.34s |       8.96s | missing-sections(title+desc...        |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |   4,905 |             1,544 |                   500 |          2,044 |         1641 |       103 |          22 |            6.43s |      2.15s |       8.92s | missing-sections(title+desc...        |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               820 |                    88 |            908 |          532 |      17.5 |          33 |            7.06s |      3.73s |      11.14s | metadata-borrowing                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,654 |                   107 |          2,761 |          772 |      31.7 |          22 |            7.39s |      2.10s |       9.84s | ⚠️harness(encoding), ...              |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      12 |             1,544 |                   500 |          2,044 |         1739 |      75.7 |          37 |            8.20s |      3.25s |      11.79s | missing-sections(title+desc...        |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,962 |             1,382 |                   500 |          1,882 |         3703 |      53.6 |         9.5 |           10.17s |      0.91s |      11.44s | ⚠️harness(stop_token), ...            |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |     198 |             6,667 |                   500 |          7,167 |         1062 |      69.4 |         8.4 |           14.00s |      1.44s |      15.84s | missing-sections(title+desc...        |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,321 |             3,433 |                   500 |          3,933 |         1489 |        41 |          15 |           15.09s |      1.70s |      17.14s | missing-sections(title+keyw...        |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |       2 |             1,873 |                   414 |          2,287 |          261 |      54.5 |          60 |           15.61s |     10.65s |      26.61s | ⚠️harness(stop_token), ...            |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   5,966 |             4,646 |                   500 |          5,146 |         3783 |      35.9 |         4.6 |           15.95s |      1.13s |      17.43s | repetitive(phrase: "- outpu...        |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |      11 |             6,667 |                   500 |          7,167 |         1156 |      48.3 |          11 |           16.62s |      1.38s |      18.34s | missing-sections(title+desc...        |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,834 |                   500 |          3,334 |         2319 |      31.2 |          19 |           17.95s |      2.03s |      20.34s | missing-sections(title+desc...        |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,731 |                   250 |          1,981 |          127 |      49.3 |          41 |           19.54s |      1.36s |      21.26s | keyword-count(19), ...                |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | 151,645 |            16,780 |                    73 |         16,853 |          940 |      57.8 |          13 |           19.97s |      2.46s |      22.77s | ⚠️harness(long_context), ...          |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  65,337 |            16,769 |                   500 |         17,269 |         1231 |      88.3 |         8.6 |           20.20s |      0.68s |      21.21s | ⚠️harness(long_context), ...          |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |   4,008 |            16,771 |                   500 |         17,271 |         1195 |      87.6 |         8.6 |           20.72s |      0.89s |      21.97s | missing-sections(title+desc...        |                 |
| `mlx-community/gemma-4-31b-bf16`                        |       1 |               813 |                   100 |            913 |         88.4 |      7.59 |          65 |           22.97s |     11.39s |      34.71s | missing-sections(title+desc...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |   1,278 |             3,342 |                   500 |          3,842 |         1872 |      20.5 |          28 |           26.66s |      2.73s |      29.72s | hallucination, fabrication, ...       |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |      11 |               523 |                   500 |          1,023 |          317 |      19.9 |          15 |           27.33s |      1.55s |      29.28s | repetitive(phrase: "riverba...        |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  81,414 |             2,335 |                   500 |          2,835 |         2819 |        34 |          18 |           29.83s |      1.73s |      31.91s | repetitive(phrase: "rencont...        |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |  56,075 |             6,667 |                   500 |          7,167 |          405 |      37.1 |          78 |           30.50s |     10.52s |      41.40s | missing-sections(title+desc...        |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,731 |                   419 |          2,150 |         87.9 |      30.4 |          48 |           52.89s |      1.72s |      54.95s | repetitive(phrase: "*/ */ *...        |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |      24 |            16,794 |                   500 |         17,294 |          326 |       100 |          26 |           57.44s |      2.46s |      60.24s | missing-sections(descriptio...        |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,780 |                    12 |         16,792 |          295 |       214 |         5.1 |           57.70s |      0.51s |      58.54s | ⚠️harness(stop_token), ...            |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |     264 |            16,794 |                   500 |         17,294 |          308 |      83.2 |          35 |           61.50s |      3.35s |      65.20s | refusal(explicit_refusal), ...        |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |     279 |            16,794 |                   500 |         17,294 |          296 |      82.3 |          12 |           63.71s |      1.74s |      65.80s | refusal(explicit_refusal), ...        |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |     262 |            16,794 |                   500 |         17,294 |          262 |      59.2 |          76 |           73.67s |     12.90s |      86.91s | ⚠️harness(long_context), ...          |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     353 |            16,794 |                   500 |         17,294 |          239 |      30.1 |          26 |           87.81s |      2.37s |      90.54s | refusal(explicit_refusal), ...        |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |      11 |            16,794 |                   500 |         17,294 |          235 |        18 |          39 |          100.20s |      3.08s |     103.64s | refusal(explicit_refusal), ...        |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |   7,096 |               524 |                   500 |          1,024 |          264 |      4.89 |          25 |          104.82s |      2.17s |     107.34s | repetitive(phrase: "waterfr...        |                 |

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
- `mlx`: `0.31.2.dev20260419+fa4320d5`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.11.0`
- `transformers`: `5.5.4`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-19 21:12:47 BST_
