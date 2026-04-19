# Model Performance Results

_Generated on 2026-04-19 23:34:45 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=7, clean outputs=4/53.
- _Useful now:_ 4 clean A/B model(s) worth first review.
- _Review watchlist:_ 49 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.

### Quality & Metadata

- _Vs existing metadata:_ better=31, neutral=5, worse=17 (baseline D 45/100).
- _Quality signal frequency:_ missing_sections=35, cutoff=31,
  metadata_borrowing=29, context_ignored=22, trusted_hint_ignored=22,
  repetitive=14.
- _Termination reasons:_ completed=53, exception=2.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (90%; 52/55 measured
  model(s)).
- _Phase totals:_ model load=117.39s, prompt prep=0.17s, decode=1125.97s,
  cleanup=5.39s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=53, exception=2.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (370.3 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 86.2 across 53 models

## 📈 Resource Usage

- **Total peak memory:** 1074.5 GB
- **Average peak memory:** 20.3 GB
- **Memory efficiency:** 254 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 10 | ✅ B: 10 | 🟡 C: 8 | 🟠 D: 14 | ❌ F: 11

**Average Utility Score:** 53/100

**Existing Metadata Baseline:** 🟠 D (45/100)
**Vs Existing Metadata:** Avg Δ +8 | Better: 31, Neutral: 5, Worse: 17

- **Best for cataloging:** `mlx-community/gemma-4-26b-a4b-it-4bit` (🏆 A, 96/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (91/100)
- **Worst for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (❌ F, 0/100)

### ⚠️ 25 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `Qwen/Qwen3-VL-2B-Instruct`: 🟠 D (44/100) - Lacks visual description of image
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/FastVLM-0.5B-bf16`: 🟠 D (40/100) - Keywords are not specific or diverse enough
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (48/100) - Keywords are not specific or diverse enough
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (26/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2.5-VL-1.6B-bf16`: 🟠 D (40/100) - Keywords are not specific or diverse enough
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (37/100) - Lacks visual description of image
- `mlx-community/Molmo-7B-D-0924-8bit`: 🟠 D (38/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (30/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (6/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (40/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (19/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (9/100) - Output lacks detail
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `ggml-org/gemma-3-1b-it-GGUF` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (14):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (token: `phrase: "castle from a distance,..."`)
  - `mlx-community/FastVLM-0.5B-bf16` (token: `phrase: "what is the name..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "clearly visible in the..."`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "2026-04-18, 17:45:40 bst, 2026..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "season: spring (april). river:..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "</points x1="45.0" y1="45.0%">..."`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `phrase: "the user's context is..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `17:45:40:`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the center has a..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `qnguyen3/nanoLLaVA` (token: `phrase: "painting glasses glasses, moto..."`)
- **👻 Hallucinations (3):**
  - `microsoft/Phi-3.5-vision-instruct`
  - `mlx-community/FastVLM-0.5B-bf16`
  - `mlx-community/Qwen3.5-9B-MLX-4bit`
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 86.2 | Min: 4.99 | Max: 370
- **Peak Memory**: Avg: 20 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 23.77s | Min: 1.64s | Max: 105.00s
- **Generation Time**: Avg: 21.24s | Min: 0.85s | Max: 102.47s
- **Model Load Time**: Avg: 2.17s | Min: 0.45s | Max: 11.54s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 52/55 measured model(s)).
- **Phase totals:** model load=117.39s, prompt prep=0.17s, decode=1125.97s, cleanup=5.39s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=53, exception=2.

### ⏱ Timing Snapshot

- **Validation overhead:** 18.88s total (avg 0.34s across 55 model(s)).
- **First-token latency:** Avg 10.41s | Min 0.08s | Max 68.55s across 53 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (A 96/100 | Desc 100 | Keywords 77 | Gen 115 TPS | Peak 17 | A 96/100 |
  missing terms: view, Round, Windsor, royal, residence)
- _Best descriptions:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (A 96/100 | Desc 100 | Keywords 77 | Gen 115 TPS | Peak 17 | A 96/100 |
  missing terms: view, Round, Windsor, royal, residence)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 92/100 | Desc 100 | Keywords 91 | Gen 66.3 TPS | Peak 13 | A 92/100 |
  nontext prompt burden=86% | missing terms: royal, residence, Berkshire,
  seen, across)
- _Fastest generation:_ [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit)
  (F 5/100 | Desc 21 | Keywords 0 | Gen 370 TPS | Peak 5.1 | F 5/100 | Special
  control token &lt;|endoftext|&gt; appeared in generated text. | Output
  appears truncated to about 2 tokens. | nontext prompt burden=97% | missing
  terms: view, Round, Tower, Windsor, Castle)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (D 40/100 | Desc 41 | Keywords 0 | Gen 346 TPS | Peak 2.1 | D 40/100 | hit
  token cap (500) | missing sections: title, description, keywords | missing
  terms: view, Round, Tower, Windsor, royal | nonvisual metadata reused)
- _Best balance:_ [`mlx-community/gemma-4-26b-a4b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit)
  (A 96/100 | Desc 100 | Keywords 77 | Gen 115 TPS | Peak 17 | A 96/100 |
  missing terms: view, Round, Windsor, royal, residence)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (2):_ [`ggml-org/gemma-3-1b-it-GGUF`](model_gallery.md#model-ggml-org-gemma-3-1b-it-gguf),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Model Error`.
- _🔄 Repetitive Output (14):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  +10 more. Example: token: `unt`.
- _👻 Hallucinations (3):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/Qwen3.5-9B-MLX-4bit`](model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit).
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (25):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  +21 more. Common weakness: Keywords are not specific or diverse enough.

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

_Overall runtime:_ 1268.67s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `ggml-org/gemma-3-1b-it-GGUF`                           |         |                   |                       |                |              |           |             |                  |      0.15s |       0.50s |                                    |    model-config |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.24s |       2.59s |                                    |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               503 |                    89 |            592 |         5295 |       356 |         2.5 |            0.85s |      0.45s |       1.64s | fabrication, ...                   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               758 |                   117 |            875 |         7747 |       332 |         2.9 |            0.87s |      0.48s |       1.69s | fabrication, ...                   |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |     106 |               779 |                    76 |            855 |         1541 |       115 |          17 |            1.69s |      2.34s |       4.40s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,114 |                   111 |          3,225 |         3020 |       185 |         7.8 |            2.17s |      0.91s |       3.43s | metadata-borrowing, ...            |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,531 |                    24 |          1,555 |         1379 |        32 |          12 |            2.39s |      1.58s |       4.33s | missing-sections(title+desc...     |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |      13 |               507 |                   500 |          1,007 |         5100 |       346 |         2.1 |            2.44s |      0.60s |       3.39s | repetitive(phrase: "what is...     |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,722 |                     8 |          2,730 |         1314 |      70.7 |         9.7 |            2.83s |      0.92s |       4.09s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |     521 |               568 |                   500 |          1,068 |         6874 |       188 |         3.8 |            3.16s |      0.53s |       4.03s | repetitive(phrase: "2026-04...     |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,531 |                     8 |          1,539 |         1093 |       5.9 |          27 |            3.30s |      2.40s |       6.05s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |             1,319 |                   164 |          1,483 |         3850 |      56.7 |         9.5 |            3.69s |      0.84s |       4.87s | fabrication, ...                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,115 |                    97 |          3,212 |         1428 |      66.3 |          13 |            4.18s |      1.38s |       5.91s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               774 |                    82 |            856 |          635 |      31.5 |          19 |            4.31s |      2.31s |       6.97s | metadata-borrowing                 |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,286 |                    71 |          2,357 |         1429 |      31.8 |          18 |            4.39s |      1.75s |       6.49s | title-length(2), ...               |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,115 |                   111 |          3,226 |         1427 |      63.2 |          13 |            4.48s |      1.35s |       6.18s |                                    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               616 |                   500 |          1,116 |         1822 |       132 |         5.5 |            4.77s |      0.58s |       5.69s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               765 |                   500 |          1,265 |         2486 |       124 |           6 |            4.83s |      1.44s |       6.62s | repetitive(17:45:40:), ...         |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               779 |                    84 |            863 |          584 |      27.5 |          20 |            4.88s |      2.52s |       7.74s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,716 |                   500 |          2,216 |         4131 |       126 |         5.5 |            5.00s |      0.63s |       5.96s | repetitive(unt), ...               |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,308 |                   100 |          3,408 |         1699 |      39.3 |          16 |            5.03s |      1.72s |       7.11s | metadata-borrowing, ...            |                 |
| `qnguyen3/nanoLLaVA`                                    |  98,266 |               503 |                   500 |          1,003 |         4463 |       112 |         4.6 |            5.07s |      0.53s |       5.95s | repetitive(phrase: "paintin...     |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,716 |                   500 |          2,216 |         4132 |       124 |         5.5 |            5.08s |      0.59s |       6.01s | repetitive(unt), ...               |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   2,401 |             1,497 |                   500 |          1,997 |         1719 |       128 |          18 |            5.37s |      1.95s |       7.66s | missing-sections(title+desc...     |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     395 |             1,497 |                   500 |          1,997 |         1771 |       115 |          22 |            5.79s |      2.16s |       8.30s | repetitive(phrase: "clearly...     |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,617 |                    81 |          2,698 |          755 |        32 |          22 |            6.54s |      2.05s |       8.94s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               774 |                    87 |            861 |          532 |      17.8 |          33 |            6.86s |      3.38s |      10.60s | metadata-borrowing                 |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               773 |                   321 |          1,094 |         1683 |      48.6 |          17 |            7.55s |      2.23s |      10.14s | description-sentences(9), ...      |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      13 |             1,497 |                   500 |          1,997 |         1725 |      79.3 |          37 |            7.85s |      3.25s |      11.45s | degeneration, ...                  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               474 |                   149 |            623 |          297 |      21.6 |          15 |            8.96s |      1.46s |      10.77s | missing-sections(title+desc...     |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,905 |             1,319 |                   500 |          1,819 |         3747 |      56.4 |         9.5 |            9.68s |      0.91s |      10.94s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |      13 |             6,618 |                   500 |          7,118 |         1162 |      71.6 |         8.4 |           13.13s |      1.29s |      14.77s | missing-sections(title+desc...     |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |  36,407 |             4,603 |                   500 |          5,103 |         3901 |      43.8 |         4.6 |           13.35s |      1.22s |      14.93s | repetitive(phrase: "- outpu...     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |  28,686 |             3,399 |                   500 |          3,899 |         1500 |      42.1 |          15 |           14.67s |      1.57s |      16.60s | missing-sections(title+desc...     |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |     374 |             6,618 |                   500 |          7,118 |         1194 |      54.5 |          11 |           15.19s |      1.37s |      16.91s | degeneration, ...                  |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             | 151,645 |            16,720 |                   166 |         16,886 |         1230 |      90.4 |         8.6 |           16.39s |      0.68s |      17.45s | fabrication, title-length(16), ... |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,786 |                   500 |          3,286 |         2423 |        32 |          19 |           17.41s |      1.89s |      19.65s | missing-sections(title+desc...     |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |     290 |             1,818 |                   500 |          2,318 |          286 |      47.9 |          60 |           17.60s |     10.75s |      28.71s | missing-sections(title+desc...     |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |             2,286 |                   500 |          2,786 |         3029 |      33.8 |          18 |           19.12s |      1.72s |      21.19s | repetitive(phrase: "rencont...     |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     448 |            16,722 |                   500 |         17,222 |         1156 |      87.7 |         8.6 |           21.14s |      0.74s |      22.24s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |      12 |            16,731 |                   500 |         17,231 |         1063 |      57.5 |          13 |           25.26s |      1.12s |      26.72s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |   5,086 |             1,531 |                   500 |          2,031 |         3277 |      19.1 |          11 |           27.15s |      1.43s |      28.93s | repetitive(phrase: "the cen...     |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |      13 |             1,690 |                   500 |          2,190 |         88.3 |      52.3 |          41 |           29.49s |      1.15s |      31.00s | repetitive(phrase: "season:...     |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |      11 |             6,618 |                   500 |          7,118 |          384 |      37.4 |          78 |           31.23s |      9.70s |      41.28s | missing-sections(title+desc...     |                 |
| `mlx-community/pixtral-12b-bf16`                        |   1,092 |             3,308 |                   500 |          3,808 |         1966 |      19.8 |          28 |           31.59s |      2.54s |      34.47s | missing-sections(title+desc...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |     369 |            16,746 |                   500 |         17,246 |          345 |       108 |          26 |           54.15s |      2.45s |      56.96s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |      12 |            16,746 |                   500 |         17,246 |          347 |      90.3 |          35 |           54.76s |      3.16s |      58.27s | missing-sections(title+desc...     |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |     198 |            16,746 |                   500 |         17,246 |          338 |      91.5 |          12 |           55.89s |      1.37s |      57.61s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |      13 |             1,690 |                   500 |          2,190 |         88.1 |      30.1 |          48 |           57.68s |      1.83s |      59.87s | repetitive(phrase: "</point...     |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,731 |                     2 |         16,733 |          292 |       370 |         5.1 |           58.15s |      0.49s |      58.99s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |      11 |            16,746 |                   500 |         17,246 |          318 |      65.7 |          76 |           61.31s |     11.54s |      73.20s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-4-31b-bf16`                        | 236,832 |               767 |                   500 |          1,267 |          220 |       7.3 |          64 |           72.53s |      6.35s |      79.24s | missing-sections(title+desc...     |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     262 |            16,746 |                   500 |         17,246 |          247 |      30.3 |          26 |           85.32s |      2.12s |      87.81s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |     264 |            16,746 |                   500 |         17,246 |          244 |      18.2 |          39 |           96.98s |      3.09s |     100.44s | refusal(explicit_refusal), ...     |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |  32,235 |               475 |                   500 |            975 |          254 |      4.99 |          25 |          102.47s |      2.17s |     105.00s | repetitive(phrase: "castle...      |                 |

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

_Report generated on: 2026-04-19 23:34:45 BST_
