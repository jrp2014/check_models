# Model Performance Results

_Generated on 2026-04-10 17:31:13 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=7, clean outputs=6/52.
- _Useful now:_ 2 clean A/B model(s) worth first review.
- _Review watchlist:_ 50 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=10, neutral=1, worse=41 (baseline B 76/100).
- _Quality signal frequency:_ missing_sections=35, cutoff=30,
  context_ignored=22, trusted_hint_ignored=22, repetitive=10,
  metadata_borrowing=9.
- _Runtime pattern:_ decode dominates measured phase time (91%; 51/53 measured
  model(s)).
- _Phase totals:_ model load=101.63s, prompt prep=0.16s, decode=1054.44s,
  cleanup=5.05s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=52, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (340.1 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 81.8 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1084.3 GB
- **Average peak memory:** 20.9 GB
- **Memory efficiency:** 260 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 9 | ✅ B: 14 | 🟡 C: 14 | 🟠 D: 3 | ❌ F: 12

**Average Utility Score:** 55/100

**Existing Metadata Baseline:** ✅ B (76/100)
**Vs Existing Metadata:** Avg Δ -21 | Better: 10, Neutral: 1, Worse: 41

- **Best for cataloging:** `mlx-community/gemma-4-31b-bf16` (🏆 A, 96/100)
- **Worst for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (❌ F, 0/100)

### ⚠️ 15 Models with Low Utility (D/F)

- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (34/100) - Lacks visual description of image
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (44/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (1/100) - Output too short to be useful
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (35/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: ❌ F (32/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (22/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (22/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (18/100) - Output lacks detail

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (10):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "city bus route, city..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "and they are not..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "birds, flight, scenic, peacefu..."`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `phrase: "the car has a..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `1.`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `qnguyen3/nanoLLaVA` (token: `Neon`)
- **📝 Formatting Issues (5):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 81.8 | Min: 5.04 | Max: 340
- **Peak Memory**: Avg: 21 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 22.37s | Min: 1.20s | Max: 103.18s
- **Generation Time**: Avg: 20.28s | Min: 0.58s | Max: 99.89s
- **Model Load Time**: Avg: 1.91s | Min: 0.45s | Max: 7.52s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (91%; 51/53 measured model(s)).
- **Phase totals:** model load=101.63s, prompt prep=0.16s, decode=1054.44s, cleanup=5.05s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=52, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 9.13s total (avg 0.17s across 53 model(s)).
- **First-token latency:** Avg 10.87s | Min 0.06s | Max 71.03s across 52 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/gemma-4-31b-bf16`](model_gallery.md#model-mlx-community-gemma-4-31b-bf16)
  (A 96/100 | Gen 7.17 TPS | Peak 64 | A 96/100 | hit token cap (500) |
  output/prompt=66.31% | missing sections: title, description, keywords |
  missing terms: flies, low, over, tranquil, style)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (F 32/100 | Gen 340 TPS | Peak 2.7 | F 32/100 | missing sections: keywords |
  context echo=73%)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (F 34/100 | Gen 331 TPS | Peak 2.2 | F 34/100 | missing sections: title,
  description, keywords | missing terms: grey, heron, flies, low, over)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (A 88/100 | Gen 173 TPS | Peak 13 | A 88/100 | nontext prompt burden=90% |
  missing terms: flies, low, style, bird, soaring)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
- _🔄 Repetitive Output (10):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +6 more. Example: token: `unt`.
- _📝 Formatting Issues (5):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +1 more.
- _Low-utility outputs (15):_ [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +11 more. Common weakness: Output too short to be useful.

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
> &#45; Description hint: A grey heron flies low over a tranquil pond in a
> Japanese-style garden. The bird is in mid-flight, soaring above the water's
> surface, with a traditional wooden zigzag bridge and lush green landscape
> visible in the background.
> &#45; Capture metadata: Taken on 2026-04-03 14:23:14 BST (at 14:23:14 local
> time). GPS: 45.518800°N, 122.708000°W.
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1171.20s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.21s |       2.38s |                                   |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               492 |                    68 |            560 |         5910 |       340 |         2.7 |            0.58s |      0.45s |       1.20s | missing-sections(keywords), ...   |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               496 |                    22 |            518 |         5099 |       331 |         2.2 |            0.60s |      0.60s |       1.38s | missing-sections(title+desc...    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               745 |                   120 |            865 |         8894 |       333 |         2.9 |            0.69s |      0.51s |       1.37s | fabrication, ...                  |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               553 |                   125 |            678 |         8897 |       193 |         3.8 |            0.95s |      0.54s |       1.66s | metadata-borrowing                |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,521 |                     8 |          1,529 |         3325 |      21.9 |          11 |            1.13s |      1.45s |       2.76s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,521 |                    25 |          1,546 |         1377 |      31.9 |          12 |            2.20s |      1.65s |       4.03s | missing-sections(title+desc...    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             3,482 |                     8 |          3,490 |         1688 |      68.3 |         9.7 |            2.56s |      0.94s |       3.68s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             4,093 |                   100 |          4,193 |         2231 |       173 |          13 |            2.74s |      0.93s |       3.86s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,521 |                    12 |          1,533 |         1096 |       5.7 |          27 |            3.82s |      2.48s |       6.48s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               765 |                    93 |            858 |          655 |      31.3 |          19 |            4.42s |      2.34s |       6.94s |                                   |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               767 |                    79 |            846 |          599 |      27.2 |          20 |            4.48s |      2.55s |       7.22s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               605 |                   500 |          1,105 |         1455 |       132 |         5.8 |            4.59s |      0.63s |       5.40s | missing-sections(title+desc...    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               756 |                   500 |          1,256 |         2577 |       122 |         6.1 |            4.65s |      1.44s |       6.28s | repetitive(1.), degeneration, ... |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             2,069 |                   500 |          2,569 |         3954 |       126 |         5.8 |            4.84s |      0.67s |       5.67s | repetitive(unt), ...              |                 |
| `qnguyen3/nanoLLaVA`                                    |  76,964 |               492 |                   500 |            992 |         5097 |       112 |         4.5 |            4.89s |      0.55s |       5.61s | repetitive(Neon), ...             |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             2,069 |                   500 |          2,569 |         3971 |       123 |         5.8 |            4.98s |      0.63s |       5.78s | repetitive(unt), ...              |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | 134,421 |             1,493 |                   500 |          1,993 |         1817 |       129 |          18 |            5.06s |      2.00s |       7.24s | degeneration, ...                 |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             3,043 |                    80 |          3,123 |         1230 |      31.9 |          19 |            5.31s |      1.76s |       7.25s | title-length(2), ...              |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             4,641 |                    99 |          4,740 |         1875 |      38.7 |          18 |            5.36s |      1.70s |       7.25s | context-echo(0.54)                |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             4,094 |                   107 |          4,201 |         1223 |      61.7 |          19 |            5.41s |      1.35s |       6.94s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             4,094 |                   109 |          4,203 |         1192 |      64.3 |          18 |            5.46s |      1.33s |       6.98s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     667 |             1,493 |                   500 |          1,993 |         1804 |       115 |          22 |            5.57s |      2.15s |       7.89s | repetitive(phrase: "and the...    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               765 |                    98 |            863 |          570 |      17.6 |          33 |            7.20s |      3.42s |      10.81s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      12 |             1,493 |                   500 |          1,993 |         1791 |      79.5 |          37 |            7.59s |      3.29s |      11.05s | degeneration, ...                 |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               764 |                   373 |          1,137 |         1793 |      48.6 |          17 |            8.39s |      2.26s |      10.83s | fabrication, ...                  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             3,598 |                    82 |          3,680 |          618 |      31.2 |          27 |            8.81s |      2.13s |      11.15s | ⚠️harness(encoding), ...          |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |     713 |             1,312 |                   500 |          1,812 |         3880 |      56.9 |         9.5 |            9.40s |      0.92s |      10.50s | keyword-count(180), ...           |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |     713 |             1,312 |                   500 |          1,812 |         3906 |      56.6 |         9.5 |            9.42s |      0.89s |      10.48s | keyword-count(180), ...           |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |  36,407 |             4,593 |                   500 |          5,093 |         3929 |      47.8 |         4.6 |           12.15s |      1.08s |      13.42s | repetitive(phrase: "- outpu...    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |  34,674 |             6,545 |                   500 |          7,045 |         1138 |      71.8 |         8.4 |           12.98s |      1.29s |      14.45s | missing-sections(title+desc...    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |     274 |             1,789 |                   500 |          2,289 |         1010 |      42.9 |          60 |           13.97s |      4.88s |      19.05s | degeneration, ...                 |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  62,052 |             6,545 |                   500 |          7,045 |         1116 |      54.5 |          11 |           15.31s |      1.45s |      16.94s | missing-sections(title+desc...    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   7,285 |             4,732 |                   500 |          5,232 |         1490 |      40.5 |          15 |           15.83s |      1.61s |      17.63s | missing-sections(title+desc...    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             3,485 |                   500 |          3,985 |         2463 |        32 |          20 |           17.39s |      1.93s |      19.50s | missing-sections(title+desc...    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  44,585 |             3,043 |                   500 |          3,543 |         3045 |      34.1 |          19 |           19.35s |      1.73s |      21.26s | repetitive(phrase: "rencont...    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | 151,645 |            16,752 |                   110 |         16,862 |          987 |      57.5 |          13 |           19.50s |      1.18s |      20.86s | title-length(4)                   |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |   3,283 |            16,741 |                   500 |         17,241 |         1237 |        89 |         8.6 |           19.80s |      0.71s |      20.69s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      13 |            16,743 |                   500 |         17,243 |         1150 |      87.7 |         8.6 |           21.01s |      0.77s |      21.96s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |  10,764 |             6,545 |                   500 |          7,045 |          543 |      36.8 |          78 |           25.91s |      5.50s |      31.60s | missing-sections(title+desc...    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |   2,522 |               461 |                   500 |            961 |          285 |      20.2 |          15 |           26.61s |      1.54s |      28.33s | repetitive(phrase: "birds,...     |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |     869 |             1,671 |                   500 |          2,171 |           89 |      52.9 |          41 |           28.70s |      1.21s |      30.09s | fabrication, ...                  |                 |
| `mlx-community/pixtral-12b-bf16`                        |   1,278 |             4,641 |                   500 |          5,141 |         2056 |      19.8 |          29 |           31.54s |      2.56s |      34.27s | missing-sections(title+desc...    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               462 |                   171 |            633 |          249 |      5.04 |          25 |           36.09s |      2.23s |      38.50s | missing-sections(title+desc...    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,671 |                   377 |          2,048 |         85.7 |      30.3 |          48 |           52.75s |      1.75s |      54.67s | missing-sections(title+desc...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |   2,099 |            16,767 |                   500 |         17,267 |          328 |       103 |          26 |           56.58s |      2.51s |      59.28s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |  20,226 |            16,767 |                   500 |         17,267 |          324 |      87.1 |          35 |           58.23s |      3.17s |      61.59s | missing-sections(title+desc...    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |     321 |            16,767 |                   500 |         17,267 |          320 |      90.6 |          12 |           58.64s |      1.40s |      60.23s | missing-sections(descriptio...    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,752 |                     6 |         16,758 |          288 |       227 |         5.1 |           58.83s |      0.55s |      59.56s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |     353 |            16,767 |                   500 |         17,267 |          313 |      64.7 |          76 |           62.21s |      7.52s |      69.91s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |  21,904 |               754 |                   500 |          1,254 |          318 |      7.17 |          64 |           72.38s |      6.01s |      78.58s | fabrication, ...                  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |  66,565 |            16,767 |                   500 |         17,267 |          240 |      29.2 |          26 |           87.67s |      2.17s |      90.04s | refusal(explicit_refusal), ...    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |      13 |            16,767 |                   500 |         17,267 |          236 |      17.8 |          39 |           99.89s |      3.10s |     103.18s | refusal(explicit_refusal), ...    |                 |

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
- `mlx`: `0.31.2.dev20260410+a33b7916`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.10.1`
- `transformers`: `5.5.3`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-10 17:31:13 BST_
