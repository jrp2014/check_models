# Model Performance Results

_Generated on 2026-04-26 00:10:30 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=4, clean outputs=5/51.
- _Useful now:_ 7 clean A/B model(s) worth first review.
- _Review watchlist:_ 44 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=36, neutral=1, worse=14 (baseline D 45/100).
- _Quality signal frequency:_ metadata_borrowing=34, missing_sections=28,
  cutoff=17, keyword_count=12, title_length=10, reasoning_leak=10.
- _Termination reasons:_ completed=51, exception=2.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (90%; 50/53 measured
  model(s)).
- _Phase totals:_ model load=101.22s, prompt prep=0.14s, decode=915.21s,
  cleanup=5.05s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=51, exception=2.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (369.1 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 85.9 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1056.9 GB
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
- **🔄 Repetitive Output (4):**
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "flagpole, flag, flagpole, flag..."`)
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

- **Generation Tps**: Avg: 85.9 | Min: 5.04 | Max: 369
- **Peak Memory**: Avg: 21 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 20.24s | Min: 1.63s | Max: 109.85s
- **Generation Time**: Avg: 17.95s | Min: 0.84s | Max: 106.43s
- **Model Load Time**: Avg: 1.94s | Min: 0.45s | Max: 8.05s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 50/53 measured model(s)).
- **Phase totals:** model load=101.22s, prompt prep=0.14s, decode=915.21s, cleanup=5.05s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=51, exception=2.

### ⏱ Timing Snapshot

- **Validation overhead:** 18.28s total (avg 0.34s across 53 model(s)).
- **First-token latency:** Avg 10.79s | Min 0.07s | Max 77.28s across 51 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (A 96/100 | Desc 100 | Keywords 77 | Gen 113 TPS | Peak 17 | A 96/100 |
  missing terms: view, Round, Windsor, royal, residence)
- _Best descriptions:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (A 96/100 | Desc 100 | Keywords 77 | Gen 113 TPS | Peak 17 | A 96/100 |
  missing terms: view, Round, Windsor, royal, residence)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 92/100 | Desc 100 | Keywords 91 | Gen 66.0 TPS | Peak 13 | A 92/100 |
  nontext prompt burden=86% | missing terms: royal, residence, Berkshire,
  seen, across)
- _Fastest generation:_ [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit)
  (F 5/100 | Desc 21 | Keywords 0 | Gen 369 TPS | Peak 5.1 | F 5/100 | Special
  control token &lt;|endoftext|&gt; appeared in generated text. | Output
  appears truncated to about 2 tokens. | nontext prompt burden=97% | missing
  terms: view, Round, Tower, Windsor, Castle)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (F 32/100 | Desc 53 | Keywords 0 | Gen 349 TPS | Peak 2.2 | F 32/100 |
  missing sections: keywords | missing terms: which, indicates | nonvisual
  metadata reused)
- _Best balance:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (A 96/100 | Desc 100 | Keywords 77 | Gen 113 TPS | Peak 17 | A 96/100 |
  missing terms: view, Round, Windsor, royal, residence)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (2):_ [`ggml-org/gemma-3-1b-it-GGUF`](model_gallery.md#model-ggml-org-gemma-3-1b-it-gguf),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Model Error`.
- _🔄 Repetitive Output (4):_ [`mlx-community/LFM2.5-VL-1.6B-bf16`](model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/gemma-3n-E2B-4bit`](model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  [`mlx-community/gemma-4-31b-bf16`](model_gallery.md#model-mlx-community-gemma-4-31b-bf16),
  [`mlx-community/paligemma2-3b-pt-896-4bit`](model_gallery.md#model-mlx-community-paligemma2-3b-pt-896-4bit).
  Example: token: `phrase: "flagpole, flag, flagpole, flag..."`.
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
> &#45; Description hint: A view of the Round Tower of Windsor Castle, a royal
> residence in Windsor, Berkshire, England, as seen from across the River
> Thames. The Union Flag is flying from the flagpole, which indicates that the
> reigning monarch is not in residence at the castle at the time the
> photograph was taken.
> &#45; Capture metadata: Taken on 2026-04-18 17:45:40 BST (at 17:45:40 local
> time). GPS: 51.483800°N, 0.604400°W.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1040.68s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `ggml-org/gemma-3-1b-it-GGUF`                           |                   |                       |                |              |           |             |                  |      0.10s |       0.43s |                                 |    model-config |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.17s |       2.52s |                                 |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               503 |                    89 |            592 |        5,756 |       369 |         2.6 |            0.84s |      0.45s |       1.63s | fabrication, ...                |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               758 |                   131 |            889 |        7,590 |       325 |           3 |            0.94s |      0.51s |       1.86s | title-length(4), ...            |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,716 |                    30 |          1,746 |        4,041 |       130 |         5.6 |            1.24s |      0.65s |       2.21s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,716 |                    30 |          1,746 |        4,119 |       127 |         5.6 |            1.27s |      0.60s |       2.21s | missing-sections(title+desc...  |                 |
| `qnguyen3/nanoLLaVA`                                    |               503 |                    83 |            586 |        4,527 |       111 |         4.8 |            1.36s |      0.53s |       2.23s | missing-sections(keywords), ... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               507 |                   158 |            665 |        5,207 |       349 |         2.2 |            1.40s |      0.58s |       2.34s | fabrication, ...                |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               616 |                    92 |            708 |        1,824 |       135 |         5.5 |            1.64s |      0.67s |       2.65s | title-length(3), ...            |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               779 |                    76 |            855 |        1,557 |       113 |          17 |            1.67s |      2.38s |       4.42s |                                 |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,114 |                   111 |          3,225 |        2,857 |       178 |         7.8 |            2.25s |      0.89s |       3.49s | metadata-borrowing, ...         |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,531 |                    24 |          1,555 |        1,382 |      31.9 |          12 |            2.40s |      1.58s |       4.32s | missing-sections(title+desc...  |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,317 |                   110 |          1,427 |        3,874 |      57.3 |         9.5 |            2.72s |      0.81s |       3.87s | metadata-borrowing              |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,722 |                     8 |          2,730 |        1,316 |      70.3 |         9.7 |            2.82s |      0.88s |       4.06s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               568 |                   500 |          1,068 |        7,998 |       186 |         3.6 |            3.17s |      0.52s |       4.04s | repetitive(phrase: "flagpol...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,497 |                   139 |          1,636 |        1,734 |      79.7 |          37 |            3.29s |      3.23s |       6.87s | missing-sections(title), ...    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,286 |                    73 |          2,359 |        2,735 |      34.8 |          18 |            3.50s |      1.60s |       5.45s | title-length(4), ...            |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               774 |                    90 |            864 |          641 |      31.3 |          19 |            4.56s |      2.33s |       7.24s | keyword-count(19)               |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,115 |                    97 |          3,212 |        1,222 |        66 |          13 |            4.56s |      1.30s |       6.22s |                                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,115 |                   111 |          3,226 |        1,242 |        63 |          13 |            4.82s |      1.32s |       6.49s |                                 |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               765 |                   500 |          1,265 |        2,558 |       122 |           6 |            4.86s |      1.41s |       6.63s | repetitive(17:45:40:), ...      |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               779 |                    84 |            863 |          587 |      27.4 |          20 |            4.90s |      2.69s |       7.95s |                                 |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,308 |                   100 |          3,408 |        1,825 |      38.7 |          16 |            4.92s |      1.65s |       6.92s | metadata-borrowing, ...         |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,531 |                    77 |          1,608 |        3,371 |      19.4 |          11 |            4.94s |      1.40s |       6.69s | missing-sections(title+desc...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,497 |                   500 |          1,997 |        1,761 |       127 |          18 |            5.33s |      1.92s |       7.58s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,286 |                   101 |          2,387 |        1,391 |      31.7 |          18 |            5.40s |      1.65s |       7.40s | title-length(4)                 |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,497 |                   500 |          1,997 |        1,685 |       111 |          22 |            5.99s |      2.11s |       8.46s | missing-sections(title), ...    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,654 |                    84 |          2,738 |          757 |      32.2 |          22 |            6.66s |      1.96s |       8.98s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,308 |                    96 |          3,404 |        1,915 |      20.2 |          28 |            6.99s |      2.54s |       9.87s | metadata-borrowing, ...         |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               774 |                    93 |            867 |          559 |      17.7 |          33 |            7.11s |      3.37s |      10.83s |                                 |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               773 |                   312 |          1,085 |        1,781 |      48.7 |          17 |            7.30s |      2.20s |       9.86s | description-sentences(6), ...   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,786 |                   184 |          2,970 |        2,453 |      32.1 |          19 |            7.47s |      1.89s |       9.71s | description-sentences(4), ...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,531 |                    31 |          1,562 |        1,112 |      5.38 |          27 |            7.67s |      2.42s |      10.44s | missing-sections(title+desc...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,317 |                   500 |          1,817 |        3,826 |      55.5 |         9.5 |            9.80s |      0.87s |      11.01s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               474 |                   149 |            623 |          293 |      18.7 |          15 |           10.06s |      1.47s |      11.89s | missing-sections(title+desc...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,818 |                   500 |          2,318 |        1,176 |      54.5 |          60 |           11.49s |      4.77s |      16.61s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,603 |                   500 |          5,103 |        3,938 |      47.6 |         4.6 |           12.39s |      1.11s |      13.85s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,618 |                   500 |          7,118 |        1,042 |      71.5 |         8.4 |           13.83s |      1.32s |      15.52s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,399 |                   500 |          3,899 |        1,515 |      42.4 |          15 |           14.56s |      1.69s |      16.61s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,618 |                   500 |          7,118 |        1,103 |      53.5 |          11 |           15.82s |      1.42s |      17.59s | degeneration, ...               |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,731 |                   136 |         16,867 |          944 |      57.6 |          13 |           20.82s |      1.10s |      22.27s | keyword-count(22), ...          |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,690 |                   299 |          1,989 |          103 |      51.4 |          41 |           23.01s |      1.10s |      24.46s | missing-sections(descriptio...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,690 |                   297 |          1,987 |          122 |      30.3 |          48 |           24.53s |      1.62s |      26.52s | description-sentences(4), ...   |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               475 |                   118 |            593 |          255 |      5.04 |          25 |           25.74s |      2.17s |      28.25s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,618 |                   500 |          7,118 |          545 |        35 |          78 |           26.91s |      5.51s |      32.80s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,746 |                   500 |         17,246 |          307 |        88 |          35 |           60.98s |      3.15s |      64.48s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,746 |                   500 |         17,246 |          307 |      89.3 |          12 |           61.00s |      1.32s |      62.68s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,746 |                   500 |         17,246 |          296 |       104 |          26 |           62.14s |      2.53s |      65.02s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,731 |                     2 |         16,733 |          269 |       369 |         5.1 |           62.92s |      0.52s |      63.79s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,746 |                   500 |         17,246 |          300 |      64.7 |          76 |           64.60s |      8.05s |      73.00s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               767 |                   500 |          1,267 |          299 |      7.25 |          64 |           72.08s |      5.98s |      78.41s | repetitive(phrase: "no grow...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,746 |                   500 |         17,246 |          226 |      29.2 |          26 |           92.08s |      2.15s |      94.59s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,746 |                   500 |         17,246 |          217 |      17.6 |          39 |          106.43s |      3.06s |     109.85s | hallucination, ...              |                 |

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
- `mlx`: `0.32.0.dev20260425+211e57be`
- `mlx-vlm`: `0.4.5`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.12.0`
- `transformers`: `5.6.2`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-26 00:10:30 BST_
