# Model Performance Results

_Generated on 2026-04-24 22:50:35 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=4, clean outputs=5/51.
- _Useful now:_ 7 clean A/B model(s) worth first review.
- _Review watchlist:_ 44 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.

### Quality & Metadata

- _Vs existing metadata:_ better=36, neutral=1, worse=14 (baseline D 45/100).
- _Quality signal frequency:_ metadata_borrowing=34, missing_sections=29,
  cutoff=19, keyword_count=11, title_length=9, reasoning_leak=8.
- _Termination reasons:_ completed=51, exception=2.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (88%; 49/53 measured
  model(s)).
- _Phase totals:_ model load=115.71s, prompt prep=0.15s, decode=929.47s,
  cleanup=5.09s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=51, exception=2.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (357.9 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 84.1 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1056.8 GB
- **Average peak memory:** 20.7 GB
- **Memory efficiency:** 221 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 14 | ✅ B: 14 | 🟡 C: 6 | 🟠 D: 8 | ❌ F: 9

**Average Utility Score:** 61/100

**Existing Metadata Baseline:** 🟠 D (45/100)
**Vs Existing Metadata:** Avg Δ +16 | Better: 36, Neutral: 1, Worse: 14

- **Best for cataloging:** `mlx-community/gemma-4-26b-a4b-it-4bit` (🏆 A, 96/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Qwen3.5-27B-mxfp8` (97/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 5/100)

### ⚠️ 17 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (15/100) - Keywords are not specific or diverse enough
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: 🟠 D (48/100) - Keywords are not specific or diverse enough
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (32/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (44/100) - Lacks visual description of image
- `mlx-community/LFM2.5-VL-1.6B-bf16`: 🟠 D (38/100) - Keywords are not specific or diverse enough
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (37/100) - Lacks visual description of image
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (15/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: 🟠 D (39/100) - Lacks visual description of image
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (30/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (6/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (40/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (19/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (20/100) - Output lacks detail
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (38/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `ggml-org/gemma-3-1b-it-GGUF` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (6):**
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "flagpole, flag, flagpole, flag..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "castle tower spiral staircase..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "castle tower silhouette: cylin..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `17:45:40:`)
  - `mlx-community/gemma-4-31b-bf16` (token: `phrase: "no grown-ups, no grown-ups,..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
- **👻 Hallucinations (2):**
  - `microsoft/Phi-3.5-vision-instruct`
  - `mlx-community/Qwen3.5-27B-mxfp8`
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 84.1 | Min: 4.96 | Max: 358
- **Peak Memory**: Avg: 21 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 20.80s | Min: 1.67s | Max: 106.74s
- **Generation Time**: Avg: 18.22s | Min: 0.86s | Max: 103.25s
- **Model Load Time**: Avg: 2.22s | Min: 0.45s | Max: 9.90s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (88%; 49/53 measured model(s)).
- **Phase totals:** model load=115.71s, prompt prep=0.15s, decode=929.47s, cleanup=5.09s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=51, exception=2.

### ⏱ Timing Snapshot

- **Validation overhead:** 18.31s total (avg 0.35s across 53 model(s)).
- **First-token latency:** Avg 10.89s | Min 0.07s | Max 75.30s across 51 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (A 96/100 | Desc 100 | Keywords 77 | Gen 114 TPS | Peak 17 | A 96/100 |
  missing terms: view, Round, Windsor, royal, residence)
- _Best descriptions:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (A 96/100 | Desc 100 | Keywords 77 | Gen 114 TPS | Peak 17 | A 96/100 |
  missing terms: view, Round, Windsor, royal, residence)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 92/100 | Desc 100 | Keywords 91 | Gen 66.0 TPS | Peak 13 | A 92/100 |
  nontext prompt burden=86% | missing terms: royal, residence, Berkshire,
  seen, across)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (D 40/100 | Desc 60 | Keywords 0 | Gen 358 TPS | Peak 2.6 | D 40/100 |
  missing sections: keywords | missing terms: view, Round, residence,
  Berkshire, seen | nonvisual metadata reused)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (F 32/100 | Desc 53 | Keywords 0 | Gen 349 TPS | Peak 2.2 | F 32/100 |
  missing sections: keywords | missing terms: which, indicates | nonvisual
  metadata reused)
- _Best balance:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (A 96/100 | Desc 100 | Keywords 77 | Gen 114 TPS | Peak 17 | A 96/100 |
  missing terms: view, Round, Windsor, royal, residence)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (2):_ [`ggml-org/gemma-3-1b-it-GGUF`](model_gallery.md#model-ggml-org-gemma-3-1b-it-gguf),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Model Error`.
- _🔄 Repetitive Output (6):_ [`mlx-community/LFM2.5-VL-1.6B-bf16`](model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/Molmo-7B-D-0924-8bit`](model_gallery.md#model-mlx-community-molmo-7b-d-0924-8bit),
  [`mlx-community/Molmo-7B-D-0924-bf16`](model_gallery.md#model-mlx-community-molmo-7b-d-0924-bf16),
  [`mlx-community/gemma-3n-E2B-4bit`](model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  +2 more. Example: token: `phrase: "flagpole, flag, flagpole, flag..."`.
- _👻 Hallucinations (2):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8).
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (17):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  +13 more. Common weakness: Keywords are not specific or diverse enough.

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

<!-- markdownlint-disable MD028 MD037 MD045 -->
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
> &#45; Description hint: A view of the Round Tower of Windsor Castle, a royal
> residence in Windsor, Berkshire, England, as seen from across the River
> Thames. The Union Flag is flying from the flagpole, which indicates that the
> reigning monarch is not in residence at the castle at the time the
> photograph was taken.
> &#45; Capture metadata: Taken on 2026-04-18 17:45:40 BST (at 17:45:40 local
> time). GPS: 51.483800°N, 0.604400°W.
<!-- markdownlint-enable MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1069.47s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `ggml-org/gemma-3-1b-it-GGUF`                           |                   |                       |                |              |           |             |                  |      0.11s |       0.44s |                                 |    model-config |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.27s |       2.60s |                                 |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               503 |                    89 |            592 |         5355 |       358 |         2.6 |            0.86s |      0.45s |       1.67s | fabrication, ...                |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               758 |                   131 |            889 |         7336 |       333 |           3 |            0.91s |      0.76s |       2.00s | title-length(4), ...            |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,716 |                    30 |          1,746 |         4140 |       129 |         5.6 |            1.26s |      0.48s |       2.06s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,716 |                    30 |          1,746 |         4168 |       124 |         5.6 |            1.28s |      0.75s |       2.37s | missing-sections(title+desc...  |                 |
| `qnguyen3/nanoLLaVA`                                    |               503 |                    83 |            586 |         4574 |       111 |         4.7 |            1.37s |      0.54s |       2.25s | missing-sections(keywords), ... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               507 |                   158 |            665 |         5213 |       349 |         2.2 |            1.38s |      0.60s |       2.33s | fabrication, ...                |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               616 |                    92 |            708 |         1829 |       132 |         5.5 |            1.66s |      0.76s |       2.76s | title-length(3), ...            |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               779 |                    76 |            855 |         1135 |       114 |          17 |            1.86s |      2.88s |       5.11s |                                 |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,114 |                   111 |          3,225 |         2949 |       178 |         7.8 |            2.22s |      1.31s |       3.88s | metadata-borrowing, ...         |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,317 |                   110 |          1,427 |         3920 |      58.6 |         9.5 |            2.65s |      1.05s |       4.04s | metadata-borrowing              |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,722 |                     8 |          2,730 |         1293 |        70 |         9.7 |            2.86s |      1.09s |       4.30s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,531 |                    24 |          1,555 |          873 |      31.2 |          12 |            3.06s |      1.55s |       4.96s | missing-sections(title+desc...  |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               568 |                   500 |          1,068 |         7985 |       188 |         3.6 |            3.13s |      0.66s |       4.13s | repetitive(phrase: "flagpol...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,497 |                   139 |          1,636 |         1775 |      80.3 |          37 |            3.23s |      3.33s |       6.91s | missing-sections(title), ...    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,286 |                    73 |          2,359 |         2762 |      34.3 |          18 |            3.51s |      1.67s |       5.52s | title-length(4), ...            |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,115 |                    97 |          3,212 |         1380 |        66 |          13 |            4.27s |      1.33s |       5.95s |                                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,115 |                   111 |          3,226 |         1410 |      62.9 |          13 |            4.51s |      2.88s |       7.74s |                                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               774 |                    90 |            864 |          588 |      30.6 |          19 |            4.75s |      2.69s |       7.79s | keyword-count(19)               |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               779 |                    84 |            863 |          583 |      27.2 |          20 |            4.93s |      2.60s |       7.88s |                                 |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,308 |                   100 |          3,408 |         1777 |      38.1 |          16 |            5.01s |      1.84s |       7.20s | metadata-borrowing, ...         |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,531 |                    77 |          1,608 |         3379 |        19 |          11 |            5.03s |      1.45s |       6.83s | missing-sections(title+desc...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,497 |                   500 |          1,997 |         1735 |       125 |          18 |            5.45s |      1.97s |       7.76s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,286 |                   101 |          2,387 |         1200 |      32.2 |          18 |            5.59s |      1.81s |       7.75s | title-length(4)                 |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,497 |                   500 |          1,997 |         1784 |       115 |          22 |            5.76s |      2.79s |       8.89s | missing-sections(title), ...    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               765 |                   500 |          1,265 |          568 |       113 |           6 |            6.27s |      2.13s |       8.77s | repetitive(17:45:40:), ...      |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,654 |                    84 |          2,738 |          694 |      31.6 |          22 |            7.13s |      2.01s |       9.52s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,308 |                    96 |          3,404 |         1864 |      19.6 |          28 |            7.20s |      2.64s |      10.17s | metadata-borrowing, ...         |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               773 |                   312 |          1,085 |         1655 |      47.9 |          17 |            7.45s |      2.37s |      10.18s | description-sentences(6), ...   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,786 |                   184 |          2,970 |         2474 |      32.1 |          19 |            7.46s |      1.95s |       9.76s | description-sentences(4), ...   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               774 |                    93 |            867 |          469 |      17.4 |          33 |            7.49s |      3.59s |      11.45s |                                 |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,531 |                    31 |          1,562 |         1040 |      5.28 |          27 |            7.88s |      2.43s |      10.66s | missing-sections(title+desc...  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               474 |                   149 |            623 |          284 |      19.6 |          15 |            9.73s |      1.63s |      11.70s | missing-sections(title+desc...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,317 |                   500 |          1,817 |         3814 |      55.1 |         9.5 |            9.88s |      0.89s |      11.14s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,818 |                   500 |          2,318 |         1162 |      55.5 |          60 |           11.34s |      5.03s |      16.75s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,603 |                   500 |          5,103 |         3793 |      46.9 |         4.6 |           12.59s |      1.15s |      14.10s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,618 |                   500 |          7,118 |         1200 |      71.3 |         8.4 |           12.97s |      1.61s |      14.92s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,399 |                   500 |          3,899 |         1463 |      41.4 |          15 |           14.93s |      1.71s |      17.03s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,618 |                   500 |          7,118 |         1181 |      54.9 |          11 |           15.16s |      1.51s |      17.02s | degeneration, ...               |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,731 |                   136 |         16,867 |          918 |      56.6 |          13 |           21.36s |      1.11s |      22.83s | keyword-count(22), ...          |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               475 |                   118 |            593 |          252 |      4.96 |          25 |           26.14s |      2.25s |      28.75s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,618 |                   500 |          7,118 |          498 |        36 |          78 |           27.64s |      6.24s |      34.23s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,690 |                   500 |          2,190 |         94.6 |      52.5 |          41 |           28.19s |      1.39s |      29.92s | repetitive(phrase: "castle...   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,690 |                   500 |          2,190 |         91.1 |      30.5 |          48 |           35.77s |      2.76s |      38.88s | repetitive(phrase: "castle...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,746 |                   500 |         17,246 |          317 |       106 |          26 |           58.29s |      2.61s |      61.25s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,746 |                   500 |         17,246 |          312 |      83.9 |          35 |           60.42s |      4.03s |      64.80s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,746 |                   500 |         17,246 |          305 |      90.3 |          12 |           61.22s |      1.97s |      63.55s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,731 |                     2 |         16,733 |          271 |       290 |         5.1 |           62.57s |      0.71s |      63.62s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,746 |                   500 |         17,246 |          288 |      65.3 |          76 |           66.73s |      9.90s |      77.01s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               767 |                   500 |          1,267 |          285 |      7.43 |          64 |           70.54s |      6.51s |      77.40s | repetitive(phrase: "no grow...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,746 |                   500 |         17,246 |          222 |      29.2 |          26 |           93.32s |      2.85s |      96.62s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,746 |                   500 |         17,246 |          226 |      17.5 |          39 |          103.25s |      3.13s |     106.74s | hallucination, ...              |                 |

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
- `mlx`: `0.32.0.dev20260424+211e57be5`
- `mlx-vlm`: `0.4.5`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.12.0`
- `transformers`: `5.6.2`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-24 22:50:35 BST_
