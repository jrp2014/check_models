# Model Performance Results

_Generated on 2026-04-11 00:42:19 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=5/52.
- _Useful now:_ 2 clean A/B model(s) worth first review.
- _Review watchlist:_ 50 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=7, neutral=5, worse=40 (baseline B 74/100).
- _Quality signal frequency:_ missing_sections=33, cutoff=29,
  context_ignored=21, trusted_hint_ignored=21, description_length=10,
  metadata_borrowing=10.
- _Runtime pattern:_ decode dominates measured phase time (90%; 51/53 measured
  model(s)).
- _Phase totals:_ model load=115.20s, prompt prep=0.16s, decode=1037.80s,
  cleanup=5.25s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=52, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (359.0 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.42s)
- **📊 Average TPS:** 83.9 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1058.1 GB
- **Average peak memory:** 20.3 GB
- **Memory efficiency:** 256 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 6 | ✅ B: 13 | 🟡 C: 14 | 🟠 D: 3 | ❌ F: 16

**Average Utility Score:** 52/100

**Existing Metadata Baseline:** ✅ B (74/100)
**Vs Existing Metadata:** Avg Δ -23 | Better: 7, Neutral: 5, Worse: 40

- **Best for cataloging:** `mlx-community/X-Reasoner-7B-8bit` (🏆 A, 98/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 0/100)

### ⚠️ 19 Models with Low Utility (D/F)

- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (29/100) - Mostly echoes context without adding value
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (25/100) - Lacks visual description of image
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: 🟠 D (40/100) - Mostly echoes context without adding value
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (36/100) - Mostly echoes context without adding value
- `mlx-community/LFM2.5-VL-1.6B-bf16`: ❌ F (34/100) - Mostly echoes context without adding value
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: ❌ F (23/100) - Mostly echoes context without adding value
- `mlx-community/Phi-3.5-vision-instruct-bf16`: ❌ F (33/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: ❌ F (24/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (22/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (17/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (18/100) - Output lacks detail
- `mlx-community/pixtral-12b-8bit`: ❌ F (34/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (10):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `phrase: "treasured treasured treasured ..."`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "garden view, garden atmosphere..."`)
  - `mlx-community/FastVLM-0.5B-bf16` (token: `phrase: "the name of the..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: ". use: a traditional..."`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `phrase: "the windows are all..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `phrase: "treasured treasured treasured ..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "- 12.3815.331200. - 12.3815.33..."`)
  - `mlx-community/gemma-4-31b-bf16` (token: `phrase: "peaceful, quiet, serene, tranq..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `qnguyen3/nanoLLaVA` (token: `Drawing`)
- **👻 Hallucinations (4):**
  - `microsoft/Phi-3.5-vision-instruct`
  - `mlx-community/FastVLM-0.5B-bf16`
  - `mlx-community/Qwen3.5-35B-A3B-bf16`
  - `mlx-community/Qwen3.5-9B-MLX-4bit`
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 83.9 | Min: 5.05 | Max: 359
- **Peak Memory**: Avg: 20 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 22.51s | Min: 1.65s | Max: 100.97s
- **Generation Time**: Avg: 19.96s | Min: 0.85s | Max: 97.41s
- **Model Load Time**: Avg: 2.17s | Min: 0.42s | Max: 11.06s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 51/53 measured model(s)).
- **Phase totals:** model load=115.20s, prompt prep=0.16s, decode=1037.80s, cleanup=5.25s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=52, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 19.70s total (avg 0.37s across 53 model(s)).
- **First-token latency:** Avg 10.61s | Min 0.09s | Max 69.82s across 52 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/X-Reasoner-7B-8bit`](model_gallery.md#model-mlx-community-x-reasoner-7b-8bit)
  (A 98/100 | Gen 57.6 TPS | Peak 13 | A 98/100 | At long prompt length (16738
  tokens), output may stop following prompt/image context. | hit token cap
  (500) | output/prompt=2.99% | nontext prompt burden=97%)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (F 24/100 | Gen 359 TPS | Peak 2.5 | F 24/100 | missing sections: keywords |
  missing terms: wooden, style, azumaya | context echo=98%)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (C 58/100 | Gen 336 TPS | Peak 2.1 | C 58/100 | hit token cap (500) |
  output/prompt=97.28% | missing sections: title, description, keywords |
  missing terms: traditional, wooden, Japanese, style, gazebo)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (B 76/100 | Gen 186 TPS | Peak 7.8 | B 76/100 | nontext prompt burden=86% |
  missing terms: style, azumaya, extends, rainy, day)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
- _🔄 Repetitive Output (10):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +6 more. Example: token: `phrase: "treasured treasured treasured ..."`.
- _👻 Hallucinations (4):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/Qwen3.5-35B-A3B-bf16`](model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16),
  [`mlx-community/Qwen3.5-9B-MLX-4bit`](model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit).
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (19):_ [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  +15 more. Common weakness: Mostly echoes context without adding value.

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
> &#45; Description hint: A traditional wooden Japanese-style gazebo, or azumaya,
> extends over a tranquil pond on a rainy day. Its reflection is visible in
> the dark, rippling water, creating a serene and contemplative scene within
> the landscaped garden. The surrounding area features lush green lawns,
> moss-covered rocks, and early spring foliage.
> &#45; Capture metadata: Taken on 2026-04-03 14:24:53 BST (at 14:24:53 local
> time). GPS: 53.331200°N, 2.381400°W.
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1178.99s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.19s |       2.56s |                                 |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               510 |                    74 |            584 |         5416 |       359 |         2.5 |            0.85s |      0.42s |       1.65s | missing-sections(keywords), ... |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               767 |                   118 |            885 |         7663 |       328 |         2.9 |            0.92s |      0.50s |       1.79s | fabrication, ...                |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               577 |                   142 |            719 |         6748 |       189 |         3.8 |            1.31s |      0.58s |       2.26s | title-length(19), ...           |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,536 |                     8 |          1,544 |         3119 |      22.2 |          11 |            1.43s |      1.48s |       3.29s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,123 |                   123 |          3,246 |         3045 |       186 |         7.8 |            2.28s |      0.99s |       3.64s |                                 |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,536 |                    23 |          1,559 |         1407 |        32 |          12 |            2.40s |      1.56s |       4.33s | missing-sections(title+desc...  |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |     829 |               514 |                   500 |          1,014 |         5082 |       336 |         2.1 |            2.60s |      0.66s |       3.64s | repetitive(phrase: "the nam...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,733 |                     8 |          2,741 |         1321 |        70 |         9.7 |            2.87s |      1.04s |       4.29s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |             1,335 |                   138 |          1,473 |         3871 |      57.4 |         9.5 |            3.24s |      0.85s |       4.45s | context-echo(0.95)              |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,536 |                     9 |          1,545 |          984 |      5.85 |          27 |            3.68s |      2.42s |       6.48s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,124 |                   118 |          3,242 |         1436 |      63.2 |          13 |            4.63s |      1.35s |       6.35s |                                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               780 |                    94 |            874 |          635 |      31.1 |          19 |            4.78s |      2.30s |       7.46s |                                 |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               623 |                   500 |          1,123 |         1851 |       133 |         5.5 |            4.78s |      0.59s |       5.74s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               771 |                   500 |          1,271 |         2464 |       125 |           6 |            4.83s |      1.41s |       6.63s | repetitive(phrase: "- 12.38...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,124 |                   139 |          3,263 |         1424 |      66.3 |          13 |            4.87s |      1.32s |       6.57s | description-sentences(3)        |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,723 |                   500 |          2,223 |         4011 |       127 |         5.5 |            5.02s |      0.62s |       5.99s | repetitive(phrase: "treasur...  |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               786 |                    86 |            872 |          586 |      27.3 |          20 |            5.05s |      2.48s |       7.93s |                                 |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,293 |                    93 |          2,386 |         1423 |      32.3 |          18 |            5.09s |      1.77s |       7.23s | trusted-hints-ignored, ...      |                 |
| `qnguyen3/nanoLLaVA`                                    |  37,759 |               510 |                   500 |          1,010 |         4745 |       113 |         4.8 |            5.10s |      0.57s |       6.03s | repetitive(Drawing), ...        |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,723 |                   500 |          2,223 |         4138 |       125 |         5.5 |            5.11s |      0.65s |       6.14s | repetitive(phrase: "treasur...  |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,317 |                   112 |          3,429 |         1768 |      39.3 |          16 |            5.32s |      1.70s |       7.42s | title-length(4), ...            |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |      13 |             1,500 |                   500 |          2,000 |         1697 |       127 |          18 |            5.44s |      1.95s |       7.76s | missing-sections(title+desc...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |   6,682 |             1,500 |                   500 |          2,000 |         1778 |       115 |          22 |            5.84s |      2.14s |       8.35s | repetitive(phrase: ". use:...   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               481 |                    88 |            569 |          303 |        22 |          15 |            6.10s |      1.48s |       7.95s | missing-sections(keywords), ... |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               780 |                    88 |            868 |          545 |      17.8 |          33 |            6.91s |      3.41s |      10.69s |                                 |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               779 |                   306 |          1,085 |         1716 |        49 |          17 |            7.22s |      2.21s |       9.82s | missing-sections(title+desc...  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,628 |                   103 |          2,731 |          744 |        32 |          22 |            7.34s |      2.16s |       9.89s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      12 |             1,500 |                   500 |          2,000 |         1763 |      79.5 |          37 |            7.87s |      3.23s |      11.49s | degeneration, ...               |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |     736 |             1,335 |                   500 |          1,835 |         3719 |      55.5 |         9.5 |            9.85s |      0.98s |      11.21s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   5,966 |             4,608 |                   500 |          5,108 |         3899 |      43.6 |         4.6 |           13.45s |      1.08s |      14.92s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |   6,821 |             6,625 |                   500 |          7,125 |         1089 |      71.2 |         8.4 |           13.64s |      1.33s |      15.36s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   3,811 |             3,408 |                   500 |          3,908 |         1532 |      42.3 |          15 |           14.60s |      1.62s |      16.60s | missing-sections(title+keyw...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |      11 |             6,625 |                   500 |          7,125 |         1134 |      54.1 |          11 |           15.60s |      1.37s |      17.34s | missing-sections(title+desc...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  19,730 |             1,831 |                   500 |          2,331 |          293 |      49.7 |          60 |           17.24s |     10.85s |      28.49s | missing-sections(title+desc...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,793 |                   500 |          3,293 |         2429 |        32 |          19 |           17.45s |      1.92s |      19.75s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |             2,293 |                   500 |          2,793 |         3045 |      34.1 |          18 |           18.21s |      1.71s |      20.29s | degeneration, ...               |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |   6,243 |            16,727 |                   500 |         17,227 |         1236 |      88.5 |         8.6 |           20.28s |      0.70s |      21.36s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      13 |            16,729 |                   500 |         17,229 |         1152 |      87.4 |         8.6 |           21.33s |      0.73s |      22.46s | ⚠️harness(long_context), ...    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               482 |                   104 |            586 |          256 |      5.05 |          25 |           22.98s |      2.22s |      25.58s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,697 |                   261 |          1,958 |         90.7 |        52 |          41 |           24.61s |      1.16s |      26.14s | title-length(11), ...           |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |   1,095 |            16,738 |                   500 |         17,238 |         1074 |      57.6 |          13 |           25.13s |      1.21s |      26.71s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |      13 |             6,625 |                   500 |          7,125 |          426 |      37.3 |          78 |           29.55s |      8.81s |      38.74s | missing-sections(title+desc...  |                 |
| `mlx-community/pixtral-12b-bf16`                        |   1,278 |             3,317 |                   500 |          3,817 |         1995 |      19.9 |          28 |           38.22s |      2.56s |      41.15s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |  34,393 |            16,753 |                   500 |         17,253 |          344 |       108 |          26 |           54.21s |      2.56s |      57.15s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |      17 |            16,753 |                   500 |         17,253 |          346 |        89 |          35 |           54.93s |      3.11s |      58.42s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |     471 |            16,753 |                   500 |         17,253 |          341 |      91.5 |          12 |           55.49s |      1.47s |      57.35s | hallucination, fabrication, ... |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,738 |                     3 |         16,741 |          287 |       282 |         5.1 |           59.10s |      0.52s |      60.00s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |   1,657 |            16,753 |                   500 |         17,253 |          321 |      65.9 |          76 |           60.68s |     11.06s |      72.12s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |      26 |             1,697 |                   500 |          2,197 |         89.1 |      30.1 |          48 |           70.69s |      1.81s |      72.87s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        | 236,764 |               773 |                   500 |          1,273 |          221 |      7.26 |          64 |           72.96s |      7.00s |      80.35s | repetitive(phrase: "peacefu...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     553 |            16,753 |                   500 |         17,253 |          240 |      30.3 |          26 |           87.31s |      2.25s |      89.95s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |      17 |            16,753 |                   500 |         17,253 |          243 |      18.2 |          39 |           97.41s |      3.16s |     100.97s | missing-sections(title+desc...  |                 |

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
- `mlx`: `0.31.2.dev20260411+a33b7916`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.10.1`
- `transformers`: `5.5.3`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-11 00:42:19 BST_
