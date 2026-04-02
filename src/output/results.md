# Model Performance Results

_Generated on 2026-04-02 23:22:04 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=1/50.
- _Useful now:_ 2 clean A/B model(s) worth first review.
- _Review watchlist:_ 48 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=11, neutral=0, worse=39 (baseline B 78/100).
- _Quality signal frequency:_ missing_sections=33, cutoff=30,
  context_ignored=22, trusted_hint_ignored=22, metadata_borrowing=19,
  repetitive=17.
- _Runtime pattern:_ decode dominates measured phase time (90%; 49/51 measured
  model(s)).
- _Phase totals:_ model load=107.35s, prompt prep=0.13s, decode=994.90s,
  cleanup=4.77s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=50, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (361.7 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (1.9 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.55s)
- **📊 Average TPS:** 84.4 across 50 models

## 📈 Resource Usage

- **Total peak memory:** 976.3 GB
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
- **🔄 Repetitive Output (17):**
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
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "strtotime => strtotime =>..."`)
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
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 84.4 | Min: 5.05 | Max: 362
- **Peak Memory**: Avg: 20 | Min: 1.9 | Max: 78
- **Total Time**: Avg: 22.17s | Min: 1.67s | Max: 104.63s
- **Generation Time**: Avg: 19.90s | Min: 0.88s | Max: 101.25s
- **Model Load Time**: Avg: 2.10s | Min: 0.55s | Max: 10.01s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 49/51 measured model(s)).
- **Phase totals:** model load=107.35s, prompt prep=0.13s, decode=994.90s, cleanup=4.77s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=50, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 8.53s total (avg 0.17s across 51 model(s)).
- **First-token latency:** Avg 11.62s | Min 0.08s | Max 72.35s across 50 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4)
  (A 97/100 | Gen 71.8 TPS | Peak 8.4 | A 97/100 | ⚠️HARNESS:stop_token;
  Missing sections (title, keywords); ...)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (F 24/100 | Gen 362 TPS | Peak 2.9 | F 24/100 | ⚠️REVIEW:cutoff; Repetitive
  output (phrase: "bricks, bricks, bricks, bricks..."); ...)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (B 72/100 | Gen 342 TPS | Peak 1.9 | B 72/100 | ⚠️REVIEW:cutoff; Repetitive
  output (phrase: "the image is clear..."); ...)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (A 90/100 | Gen 63.2 TPS | Peak 13 | A 90/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
- _🔄 Repetitive Output (17):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  +13 more. Example: token: `phrase: "treasured treasured treasured ..."`.
- _👻 Hallucinations (4):_ [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8),
  [`mlx-community/Qwen3.5-9B-MLX-4bit`](model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit).
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
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
> &#45; Description hint: The sculpture 'Maquette for the Spirit of the
> University' by British sculptor Hubert 'Nibs' Dalwood, created in 1961, is
> seen on display at the University of Reading's Whiteknights campus in
> Reading, UK. The bronze piece is a small-scale model for a much larger,
> unrealized sculpture intended to represent the spirit of the university and
> is located outside the Department of Typography &amp; Graphic Communication.
> &#45; Keyword hints: 10 Best (structured), Abstract Art, Adobe Stock, Any
> Vision, Blue sky, Bronze, Bronze Sculpture, Daylight, England, Europe,
> Handrail, Modern Art, Objects, Royston, Sculpture, Stairs, Statue, Stone,
> Textured, Town Centre
> &#45; Capture metadata: Taken on 2026-03-28 15:52:29 GMT (at 15:52:29 local
> time). GPS: 51.758450°N, 1.255650°W.
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1116.46s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.40s |       2.56s |                                    |    model-config |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               851 |                    89 |            940 |         2393 |       327 |         2.9 |            0.88s |      0.62s |       1.67s | missing-sections(title+descrip...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,618 |                    11 |          1,629 |         3353 |      21.1 |          11 |            1.37s |      1.55s |       3.10s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |  49,037 |               596 |                   500 |          1,096 |         6005 |       362 |         2.9 |            1.83s |      0.55s |       2.55s | repetitive(phrase: "bricks, ...    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,205 |                    97 |          3,302 |         2873 |       185 |         7.8 |            2.04s |      1.11s |       3.32s | title-length(4)                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |     323 |               600 |                   500 |          1,100 |         1748 |       342 |         1.9 |            2.62s |      0.80s |       3.60s | repetitive(phrase: "the image...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,618 |                    20 |          1,638 |          914 |      30.9 |          12 |            2.79s |      1.68s |       4.65s | missing-sections(title+descrip...  |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |     521 |               661 |                   500 |          1,161 |         7908 |       182 |         3.8 |            3.09s |      0.57s |       3.83s | repetitive(phrase: "university...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,830 |                    10 |          2,840 |         1083 |      67.5 |         9.7 |            3.26s |      1.10s |       4.54s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,618 |                     9 |          1,627 |          926 |      5.83 |          27 |            3.66s |      2.43s |       6.27s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,206 |                    97 |          3,303 |         1384 |      63.2 |          13 |            4.24s |      1.46s |       5.87s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,206 |                   100 |          3,306 |         1370 |      66.3 |          13 |            4.25s |      1.50s |       5.93s | trusted-hints-ignored, ...         |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,379 |                    71 |          2,450 |         1277 |      32.3 |          18 |            4.47s |      1.92s |       6.55s | title-length(2), ...               |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               863 |                    87 |            950 |          621 |      31.2 |          19 |            4.49s |      2.42s |       7.09s | trusted-hints-degraded             |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               708 |                   500 |          1,208 |         2071 |       130 |         5.5 |            4.66s |      0.71s |       5.54s | missing-sections(title+descrip...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,808 |                   500 |          2,308 |         4287 |       124 |         5.6 |            4.93s |      0.73s |       5.82s | repetitive(phrase: "treasured...   |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,399 |                    95 |          3,494 |         1451 |      39.2 |          16 |            5.14s |      1.78s |       7.11s | description-sentences(3), ...      |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               854 |                   500 |          1,354 |          637 |       124 |           6 |            5.69s |      1.53s |       7.41s | repetitive(15:52:29.), ...         |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | 134,421 |             1,584 |                   500 |          2,084 |          980 |       127 |          18 |            5.95s |      2.10s |       8.25s | degeneration, ...                  |                 |
| `qnguyen3/nanoLLaVA`                                    |  76,964 |               596 |                   500 |          1,096 |         6500 |       112 |         5.5 |            6.21s |      0.65s |       7.02s | repetitive(Neon), ...              |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,808 |                   500 |          2,308 |          918 |       128 |         5.6 |            6.32s |      0.84s |       7.32s | repetitive(phrase: "treasured...   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |   4,937 |             1,584 |                   500 |          2,084 |          992 |       113 |          22 |            6.44s |      2.26s |       8.87s | repetitive(phrase: "these deta...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               863 |                    82 |            945 |          512 |      17.8 |          33 |            6.62s |      3.65s |      10.44s | trusted-hints-degraded             |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,709 |                    75 |          2,784 |          676 |      31.4 |          22 |            6.81s |      2.20s |       9.20s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               862 |                   294 |          1,156 |         1868 |      48.4 |          17 |            6.86s |      2.35s |       9.39s | fabrication, title-length(11), ... |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | 134,421 |             1,584 |                   500 |          2,084 |         1829 |      78.9 |          37 |            7.72s |      3.37s |      11.27s | degeneration, ...                  |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  29,892 |             1,427 |                   500 |          1,927 |         3883 |      56.5 |         9.6 |            9.50s |      1.27s |      10.94s | repetitive(phrase: "educationa...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,892 |             1,427 |                   500 |          1,927 |         1742 |        56 |         9.6 |           10.05s |      1.03s |      11.24s | repetitive(phrase: "educationa...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   5,966 |             4,690 |                   500 |          5,190 |         3646 |      43.6 |         4.6 |           13.37s |      1.22s |      14.76s | repetitive(phrase: "- output o...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |   1,352 |             6,707 |                   500 |          7,207 |         1038 |      71.8 |         8.4 |           13.76s |      1.50s |      15.45s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,044 |             3,490 |                   500 |          3,990 |         1425 |      41.9 |          15 |           14.77s |      1.83s |      16.77s | missing-sections(title), ...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |      11 |             6,707 |                   500 |          7,207 |         1061 |      54.5 |          11 |           15.82s |      1.47s |      17.46s | repetitive(phrase: "sculpture, ... |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,876 |                   500 |          3,376 |         2351 |      31.7 |          19 |           17.48s |      2.23s |      19.90s | missing-sections(title+descrip...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  25,473 |             1,921 |                   500 |          2,421 |          347 |      43.1 |          60 |           17.77s |      8.46s |      26.41s | repetitive(phrase: ""a stone o...  |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |      17 |            16,813 |                   500 |         17,313 |         1212 |      88.5 |         8.6 |           20.23s |      0.83s |      21.21s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | 151,645 |            16,824 |                   157 |         16,981 |          998 |      57.7 |          13 |           20.23s |      1.23s |      21.63s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |   5,009 |            16,815 |                   500 |         17,315 |         1139 |      87.6 |         8.6 |           21.32s |      0.89s |      22.42s | keyword-count(19), ...             |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |             2,379 |                   500 |          2,879 |         2992 |        34 |          18 |           23.79s |      1.88s |      25.83s | repetitive(phrase: "rencontre...   |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               565 |                   112 |            677 |          248 |      5.05 |          25 |           24.78s |      2.30s |      27.26s | missing-sections(title+descrip...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,783 |                   294 |          2,077 |         94.4 |      52.3 |          41 |           25.19s |      1.31s |      26.67s | title-length(11), ...              |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |   6,710 |               564 |                   500 |          1,064 |          327 |      20.7 |          15 |           26.17s |      1.53s |      27.86s | repetitive(phrase: "art, ...       |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          | 151,336 |             6,707 |                   438 |          7,145 |          402 |      37.5 |          78 |           28.91s |      7.88s |      36.99s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/pixtral-12b-bf16`                        |  84,714 |             3,399 |                   500 |          3,899 |         1761 |        20 |          28 |           31.99s |      2.64s |      34.79s | missing-sections(title+descrip...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |  50,962 |             1,783 |                   500 |          2,283 |          103 |      30.3 |          48 |           54.92s |      1.83s |      56.93s | repetitive(phrase: "strtotime...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |  11,791 |            16,839 |                   500 |         17,339 |          326 |       105 |          26 |           57.20s |      2.61s |      59.98s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |   7,099 |            16,839 |                   500 |         17,339 |          330 |      90.5 |          12 |           57.32s |      1.49s |      58.98s | hallucination, ...                 |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |   5,721 |            16,839 |                   500 |         17,339 |          319 |      89.5 |          35 |           59.08s |      3.25s |      62.51s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,824 |                    18 |         16,842 |          277 |       202 |         5.1 |           61.40s |      0.84s |      62.41s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |   9,947 |            16,839 |                   500 |         17,339 |          292 |      64.6 |          76 |           66.22s |     10.01s |      76.41s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     198 |            16,839 |                   500 |         17,339 |          234 |      29.1 |          26 |           90.03s |      2.32s |      92.53s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |   4,568 |            16,839 |                   500 |         17,339 |          233 |      17.8 |          39 |          101.25s |      3.20s |     104.63s | refusal(explicit_refusal), ...     |                 |

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
- `mlx`: `0.31.2.dev20260402+befe42d3`
- `mlx-vlm`: `0.4.3`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.8.0`
- `transformers`: `5.5.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-02 23:22:04 BST_
