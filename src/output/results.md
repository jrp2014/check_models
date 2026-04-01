# Model Performance Results

_Generated on 2026-04-02 00:00:55 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=1/50.
- _Useful now:_ 1 clean A/B model(s) worth first review.
- _Review watchlist:_ 49 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=11, neutral=0, worse=39 (baseline B 78/100).
- _Quality signal frequency:_ missing_sections=33, cutoff=30,
  context_ignored=27, trusted_hint_ignored=26, metadata_borrowing=19,
  repetitive=16.
- _Runtime pattern:_ decode dominates measured phase time (90%; 49/51 measured
  model(s)).
- _Phase totals:_ model load=108.01s, prompt prep=0.12s, decode=972.15s,
  cleanup=4.80s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=50, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (362.9 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.50s)
- **📊 Average TPS:** 84.5 across 50 models

## 📈 Resource Usage

- **Total peak memory:** 976.5 GB
- **Average peak memory:** 19.5 GB
- **Memory efficiency:** 281 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 11 | ✅ B: 11 | 🟡 C: 11 | 🟠 D: 5 | ❌ F: 12

**Average Utility Score:** 55/100

**Existing Metadata Baseline:** ✅ B (78/100)
**Vs Existing Metadata:** Avg Δ -22 | Better: 11, Neutral: 0, Worse: 39

- **Best for cataloging:** `mlx-community/GLM-4.6V-Flash-mxfp4` (🏆 A, 97/100)
- **Worst for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (❌ F, 0/100)

### ⚠️ 17 Models with Low Utility (D/F)

- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (29/100) - Mostly echoes context without adding value
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (46/100) - Mostly echoes context without adding value
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (15/100) - Mostly echoes context without adding value
- `mlx-community/LFM2.5-VL-1.6B-bf16`: 🟠 D (39/100) - Mostly echoes context without adding value
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (46/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (23/100) - Output lacks detail
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (10/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: ❌ F (24/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: 🟠 D (38/100) - Lacks visual description of image
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (17/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (20/100) - Output lacks detail

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (16):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `phrase: "treasured treasured treasured ..."`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "15:52:29 local time, 15:52:29..."`)
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "educational artistic expressio..."`)
  - `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (token: `phrase: ""a stone or bronze..."`)
  - `mlx-community/FastVLM-0.5B-bf16` (token: `phrase: "the image is clear..."`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (token: `phrase: "sculpture, statue, stone, bric..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "these details to create..."`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "university of reading, whitekn..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "art, piece, art, piece,..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "educational artistic expressio..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `phrase: "treasured treasured treasured ..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `15:52:29.`)
  - `mlx-community/nanoLLaVA-1.5-4bit` (token: `phrase: "bricks, bricks, bricks, bricks..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `qnguyen3/nanoLLaVA` (token: `Neon`)
- **👻 Hallucinations (4):**
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit`
  - `mlx-community/Qwen3.5-27B-mxfp8`
  - `mlx-community/Qwen3.5-9B-MLX-4bit`
- **📝 Formatting Issues (5):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 84.5 | Min: 5.06 | Max: 363
- **Peak Memory**: Avg: 20 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 21.73s | Min: 1.41s | Max: 103.77s
- **Generation Time**: Avg: 19.44s | Min: 0.64s | Max: 99.64s
- **Model Load Time**: Avg: 2.11s | Min: 0.50s | Max: 10.39s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 49/51 measured model(s)).
- **Phase totals:** model load=108.01s, prompt prep=0.12s, decode=972.15s, cleanup=4.80s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=50, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 8.59s total (avg 0.17s across 51 model(s)).
- **First-token latency:** Avg 11.21s | Min 0.07s | Max 71.11s across 50 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4)
  (A 97/100 | Gen 72.1 TPS | Peak 8.4 | A 97/100 | ⚠️HARNESS:stop_token;
  Context ignored (missing: 10 Best (structured), Abstract Art, Handrail,
  Modern Art, Objects); ...)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (F 24/100 | Gen 363 TPS | Peak 2.9 | F 24/100 | ⚠️REVIEW:cutoff; Repetitive
  output (phrase: "bricks, bricks, bricks, bricks..."); ...)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (B 72/100 | Gen 339 TPS | Peak 2.1 | B 72/100 | ⚠️REVIEW:cutoff; Repetitive
  output (phrase: "the image is clear..."); ...)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (A 90/100 | Gen 63.1 TPS | Peak 13 | A 90/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
- _🔄 Repetitive Output (16):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  +12 more. Example: token: `phrase: "treasured treasured treasured ..."`.
- _👻 Hallucinations (4):_ [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8),
  [`mlx-community/Qwen3.5-9B-MLX-4bit`](model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit).
- _📝 Formatting Issues (5):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +1 more.
- _Low-utility outputs (17):_ [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  +13 more. Common weakness: Mostly echoes context without adding value.

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

<!-- markdownlint-disable MD028 -->
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
> \- 5-10 words, concrete and factual, limited to clearly visible content.
> \- Output only the title text after the label.
> \- Do not repeat or paraphrase these instructions in the title.
>
> Description:
> \- 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> \- Output only the description text after the label.
>
> Keywords:
> \- 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> \- Output only the keyword list after the label.
>
> Rules:
> \- Include only details that are definitely visible in the image.
> \- Reuse metadata terms only when they are clearly supported by the image.
> \- If metadata and image disagree, follow the image.
> \- Prefer omission to speculation.
> \- Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> \- Do not infer identity, location, event, brand, species, time period, or
> intent unless visually obvious.
> \- Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually
> confirmed):
> \- Description hint: The sculpture 'Maquette for the Spirit of the
> University' by British sculptor Hubert 'Nibs' Dalwood, created in 1961, is
> seen on display at the University of Reading's Whiteknights campus in
> Reading, UK. The bronze piece is a small-scale model for a much larger,
> unrealized sculpture intended to represent the spirit of the university and
> is located outside the Department of Typography &amp; Graphic Communication.
> \- Keyword hints: 10 Best (structured), Abstract Art, Adobe Stock, Any
> Vision, Blue sky, Bronze, Bronze Sculpture, Daylight, England, Europe,
> Handrail, Modern Art, Objects, Royston, Sculpture, Stairs, Statue, Stone,
> Textured, Town Centre
> \- Capture metadata: Taken on 2026-03-28 15:52:29 GMT (at 15:52:29 local
> time). GPS: 51.758450°N, 1.255650°W.
<!-- markdownlint-enable MD028 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1094.52s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.33s |       2.49s |                                    |    model-config |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               851 |                    89 |            940 |         9182 |       329 |         2.9 |            0.64s |      0.61s |       1.41s | missing-sections(title+descrip...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,618 |                    11 |          1,629 |         3335 |      21.1 |          11 |            1.38s |      1.51s |       3.06s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |  49,037 |               596 |                   500 |          1,096 |         7013 |       363 |         2.9 |            1.81s |      0.50s |       2.48s | repetitive(phrase: "bricks, ...    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,205 |                    97 |          3,302 |         3053 |       185 |         7.8 |            1.97s |      1.05s |       3.19s | title-length(4), ...               |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,618 |                    20 |          1,638 |         1241 |      30.6 |          12 |            2.33s |      1.60s |       4.11s | missing-sections(title+descrip...  |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |     323 |               600 |                   500 |          1,100 |         4534 |       339 |         2.1 |            2.45s |      0.72s |       3.35s | repetitive(phrase: "the image...   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |     521 |               661 |                   500 |          1,161 |        10069 |       190 |         3.8 |            2.96s |      0.60s |       3.73s | repetitive(phrase: "university...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,830 |                    10 |          2,840 |         1221 |      67.3 |         9.7 |            2.96s |      1.05s |       4.18s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,618 |                     9 |          1,627 |         1125 |      5.83 |          27 |            3.36s |      2.46s |       5.99s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,206 |                   100 |          3,306 |         1387 |      66.6 |          13 |            4.21s |      1.48s |       5.86s | trusted-hints-ignored, ...         |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,206 |                    97 |          3,303 |         1385 |      63.1 |          13 |            4.24s |      1.49s |       5.90s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               863 |                    87 |            950 |          659 |      31.2 |          19 |            4.42s |      2.39s |       6.98s | trusted-hints-degraded             |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,379 |                    71 |          2,450 |         1262 |      32.5 |          18 |            4.48s |      1.95s |       6.60s | title-length(2), ...               |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               708 |                   500 |          1,208 |         2077 |       131 |         5.5 |            4.65s |      0.71s |       5.53s | missing-sections(title+descrip...  |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,399 |                    95 |          3,494 |         1715 |      39.2 |          16 |            4.77s |      1.83s |       6.77s | description-sentences(3), ...      |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,808 |                   500 |          2,308 |         4277 |       125 |         5.6 |            4.92s |      0.74s |       5.83s | repetitive(phrase: "treasured...   |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               854 |                   500 |          1,354 |         1594 |       123 |           6 |            4.92s |      1.49s |       6.60s | repetitive(15:52:29.), ...         |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,808 |                   500 |          2,308 |         2851 |       125 |         5.6 |            5.10s |      0.77s |       6.03s | repetitive(phrase: "treasured...   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | 134,421 |             1,584 |                   500 |          2,084 |         1705 |       126 |          18 |            5.34s |      2.10s |       7.61s | degeneration, ...                  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |   4,937 |             1,584 |                   500 |          2,084 |         1670 |       115 |          22 |            5.75s |      2.25s |       8.17s | repetitive(phrase: "these deta...  |                 |
| `qnguyen3/nanoLLaVA`                                    |  76,964 |               596 |                   500 |          1,096 |         6502 |       112 |         5.5 |            6.22s |      0.64s |       7.03s | repetitive(Neon), ...              |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,709 |                    75 |          2,784 |          727 |      31.6 |          22 |            6.51s |      2.18s |       8.87s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               863 |                    82 |            945 |          548 |      17.7 |          33 |            6.52s |      3.44s |      10.15s | trusted-hints-degraded, ...        |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               862 |                   294 |          1,156 |         1842 |      48.8 |          17 |            6.80s |      2.32s |       9.30s | fabrication, title-length(11), ... |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | 134,421 |             1,584 |                   500 |          2,084 |         1804 |      78.7 |          37 |            7.76s |      3.36s |      11.29s | degeneration, ...                  |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  29,892 |             1,427 |                   500 |          1,927 |         3924 |      56.6 |         9.6 |            9.49s |      0.90s |      10.56s | repetitive(phrase: "educationa...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,892 |             1,427 |                   500 |          1,927 |         3275 |      55.5 |         9.6 |            9.74s |      1.05s |      10.96s | repetitive(phrase: "educationa...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   5,966 |             4,690 |                   500 |          5,190 |         3909 |      46.6 |         4.6 |           12.52s |      1.16s |      13.86s | repetitive(phrase: "- output o...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |   1,352 |             6,707 |                   500 |          7,207 |         1047 |      72.1 |         8.4 |           13.69s |      1.49s |      15.38s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,044 |             3,490 |                   500 |          3,990 |         1487 |      42.1 |          15 |           14.60s |      1.70s |      16.48s | missing-sections(title), ...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |      11 |             6,707 |                   500 |          7,207 |         1150 |      52.8 |          11 |           15.61s |      1.47s |      17.26s | repetitive(phrase: "sculpture, ... |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,876 |                   500 |          3,376 |         2346 |        32 |          19 |           17.34s |      2.08s |      19.61s | missing-sections(title+descrip...  |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | 151,645 |            16,824 |                   157 |         16,981 |         1051 |      57.4 |          13 |           19.41s |      1.23s |      20.81s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  25,473 |             1,921 |                   500 |          2,421 |          256 |        43 |          60 |           19.90s |     10.39s |      30.49s | repetitive(phrase: ""a stone o...  |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |      17 |            16,813 |                   500 |         17,313 |         1209 |      87.3 |         8.6 |           20.37s |      0.81s |      21.35s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |   5,009 |            16,815 |                   500 |         17,315 |         1151 |      87.2 |         8.6 |           21.16s |      0.84s |      22.20s | keyword-count(19), ...             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,783 |                   294 |          2,077 |          105 |      52.1 |          41 |           23.29s |      1.22s |      24.68s | title-length(11), ...              |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |             2,379 |                   500 |          2,879 |         2969 |      33.9 |          18 |           23.91s |      1.85s |      25.92s | repetitive(phrase: "rencontre...   |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               565 |                   112 |            677 |          274 |      5.06 |          25 |           24.54s |      3.08s |      27.81s | missing-sections(title+descrip...  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |   6,710 |               564 |                   500 |          1,064 |          346 |      20.8 |          15 |           26.01s |      1.56s |      27.74s | repetitive(phrase: "art, ...       |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          | 151,336 |             6,707 |                   438 |          7,145 |          461 |      37.7 |          78 |           26.57s |      6.91s |      33.69s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/pixtral-12b-bf16`                        |  84,714 |             3,399 |                   500 |          3,899 |         2035 |      20.1 |          28 |           31.71s |      2.62s |      34.49s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |  11,791 |            16,839 |                   500 |         17,339 |          342 |       106 |          26 |           54.66s |      2.62s |      57.45s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 109,066 |             1,783 |                   500 |          2,283 |         91.4 |      30.3 |          48 |           56.04s |      1.84s |      58.05s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |   7,099 |            16,839 |                   500 |         17,339 |          330 |      91.1 |          12 |           57.33s |      1.45s |      58.96s | hallucination, ...                 |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |   5,721 |            16,839 |                   500 |         17,339 |          327 |      85.3 |          35 |           58.10s |      3.20s |      61.48s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,824 |                    18 |         16,842 |          285 |       200 |         5.1 |           59.72s |      0.65s |      60.54s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |   9,947 |            16,839 |                   500 |         17,339 |          315 |      64.5 |          76 |           62.13s |     10.14s |      72.44s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     198 |            16,839 |                   500 |         17,339 |          239 |      29.7 |          26 |           88.19s |      2.25s |      90.63s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |   4,568 |            16,839 |                   500 |         17,339 |          237 |        18 |          39 |           99.64s |      3.94s |     103.77s | refusal(explicit_refusal), ...     |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Automated review digest:_ [review.md](review.md)
- _Canonical run log:_ [check_models.log](check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.4.0
- _macOS Version:_ 26.4
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
- `mlx`: `0.31.2.dev20260401+b0748ad8`
- `mlx-vlm`: `0.4.3`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.8.0`
- `transformers`: `5.4.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.1.1`

_Report generated on: 2026-04-02 00:00:55 BST_
