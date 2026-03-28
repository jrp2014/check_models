# Model Performance Results

_Generated on 2026-03-27 23:26:59 GMT_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 10 (top owners: transformers=6, mlx-vlm=2,
  model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=8, clean outputs=3/42.
- _Useful now:_ 1 clean A/B model(s) worth first review.
- _Review watchlist:_ 41 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=12, neutral=1, worse=29 (baseline B 79/100).
- _Quality signal frequency:_ missing_sections=27, cutoff=25,
  trusted_hint_ignored=19, context_ignored=18, repetitive=11,
  description_length=8.
- _Runtime pattern:_ decode dominates measured phase time (89%; 42/52 measured
  model(s)).
- _Phase totals:_ model load=101.19s, prompt prep=0.12s, decode=901.87s,
  cleanup=5.23s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=42, exception=10.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/FastVLM-0.5B-bf16` (340.3 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.0 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 74.3 across 42 models

## 📈 Resource Usage

- **Total peak memory:** 874.9 GB
- **Average peak memory:** 20.8 GB
- **Memory efficiency:** 235 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 12 | ✅ B: 9 | 🟡 C: 5 | 🟠 D: 7 | ❌ F: 9

**Average Utility Score:** 57/100

**Existing Metadata Baseline:** ✅ B (79/100)
**Vs Existing Metadata:** Avg Δ -23 | Better: 12, Neutral: 1, Worse: 29

- **Best for cataloging:** `mlx-community/X-Reasoner-7B-8bit` (🏆 A, 98/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 0/100)

### ⚠️ 16 Models with Low Utility (D/F)

- `Qwen/Qwen3-VL-2B-Instruct`: 🟠 D (38/100) - Mostly echoes context without adding value
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (48/100) - Mostly echoes context without adding value
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (36/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: 🟠 D (48/100) - Lacks visual description of image
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/pixtral-12b-8bit`: 🟠 D (39/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (10):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
  - `mlx-community/deepseek-vl2-8bit` (`Processor Error`)
  - `mlx-community/paligemma2-10b-ft-docci-448-6bit` (`API Mismatch`)
  - `mlx-community/paligemma2-10b-ft-docci-448-bf16` (`API Mismatch`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (`API Mismatch`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (`API Mismatch`)
  - `prince-canuma/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (11):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "reflections, white blossoms, t..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "use only if a..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "waterfront living, waterfront ..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "uk. spring. water. boats...."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: ":: ,param( :: ,param(..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/X-Reasoner-7B-8bit` (token: `phrase: "- monochrome stripes -..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "10:18: 16:42: 10:18: 16:42:..."`)
  - `qnguyen3/nanoLLaVA` (token: `phrase: "painting glasses glasses, moto..."`)
- **👻 Hallucinations (4):**
  - `microsoft/Phi-3.5-vision-instruct`
  - `mlx-community/FastVLM-0.5B-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/Qwen3.5-27B-mxfp8`
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 74.3 | Min: 0 | Max: 340
- **Peak Memory**: Avg: 21 | Min: 2.0 | Max: 78
- **Total Time**: Avg: 23.80s | Min: 1.77s | Max: 117.15s
- **Generation Time**: Avg: 21.42s | Min: 0.99s | Max: 113.63s
- **Model Load Time**: Avg: 2.06s | Min: 0.45s | Max: 9.48s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (89%; 42/52 measured model(s)).
- **Phase totals:** model load=101.19s, prompt prep=0.12s, decode=901.87s, cleanup=5.23s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=42, exception=10.

### ⏱ Timing Snapshot

- **Validation overhead:** 15.98s total (avg 0.31s across 52 model(s)).
- **First-token latency:** Avg 10.66s | Min 0.13s | Max 82.56s across 41 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/X-Reasoner-7B-8bit`](model_gallery.md#model-mlx-community-x-reasoner-7b-8bit)
  (A 98/100 | Gen 54.1 TPS | Peak 13 | A 98/100 | ⚠️REVIEW:cutoff;
  ⚠️HARNESS:long_context; ...)
- _Fastest generation:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (A 88/100 | Gen 340 TPS | Peak 2.0 | A 88/100 | Context ignored (missing:
  large, former, warehouse, London, Canal); Missing sections (title,
  description, keywords); ...)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (A 88/100 | Gen 340 TPS | Peak 2.0 | A 88/100 | Context ignored (missing:
  large, former, warehouse, London, Canal); Missing sections (title,
  description, keywords); ...)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (B 69/100 | Gen 63.0 TPS | Peak 13 | B 69/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (10):_ [`microsoft/Florence-2-large-ft`](model_gallery.md#model-microsoft-florence-2-large-ft),
  [`mlx-community/LFM2-VL-1.6B-8bit`](model_gallery.md#model-mlx-community-lfm2-vl-16b-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16),
  +6 more. Example: `Model Error`.
- _🔄 Repetitive Output (11):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +7 more. Example: token: `unt`.
- _👻 Hallucinations (4):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/Molmo-7B-D-0924-bf16`](model_gallery.md#model-mlx-community-molmo-7b-d-0924-bf16),
  [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8).
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (16):_ [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +12 more. Common weakness: Mostly echoes context without adding value.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `transformers` | 6 | API Mismatch, Model Error | `microsoft/Florence-2-large-ft`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`, `prince-canuma/Florence-2-large-ft` |
| `mlx-vlm` | 2 | Model Error | `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `model-config` | 2 | Processor Error | `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/deepseek-vl2-8bit` |

### Actionable Items by Package

#### transformers

- microsoft/Florence-2-large-ft (Model Error)
  - Error: `Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
  - Type: `ValueError`
- mlx-community/paligemma2-10b-ft-docci-448-6bit (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-6bit: Failed to process inputs with error: Imag...`
  - Type: `ValueError`
- mlx-community/paligemma2-10b-ft-docci-448-bf16 (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-bf16: Failed to process inputs with error: Imag...`
  - Type: `ValueError`
- mlx-community/paligemma2-3b-ft-docci-448-bf16 (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-3b-ft-docci-448-bf16: Failed to process inputs with error: Image...`
  - Type: `ValueError`
- mlx-community/paligemma2-3b-pt-896-4bit (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-3b-pt-896-4bit: Failed to process inputs with error: ImagesKwarg...`
  - Type: `ValueError`
- prince-canuma/Florence-2-large-ft (Model Error)
  - Error: `Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
  - Type: `ValueError`

#### mlx-vlm

- mlx-community/LFM2-VL-1.6B-8bit (Model Error)
  - Error: `Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Failed to process inputs with error: object of type 'num...`
  - Type: `ValueError`
- mlx-community/LFM2.5-VL-1.6B-bf16 (Model Error)
  - Error: `Model generation failed for mlx-community/LFM2.5-VL-1.6B-bf16: Failed to process inputs with error: object of type 'n...`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`
- mlx-community/deepseek-vl2-8bit (Processor Error)
  - Error: `Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimo...`
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
> \- Description hint: A large brick former warehouse, the London Canal Museum,
> stands beside the Regent's Canal in King's Cross, London, on a sunny spring
> day. Several narrowboats are moored along the towpath, their reflections
> visible in the calm water. White blossoms in the foreground frame the
> tranquil urban scene.
> \- Capture metadata: Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local
> time). GPS: 51.532500°N, 0.122500°W.
<!-- markdownlint-enable MD028 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1025.29s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |      0.34s |       0.64s |                                    |    transformers |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |         |                   |                       |                |              |           |             |            0.40s |      0.52s |       1.22s | ⚠️harness(stop_token), ...         |         mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |         |                   |                       |                |              |           |             |            0.42s |      0.63s |       1.37s | ⚠️harness(stop_token), ...         |         mlx-vlm |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.26s |       2.57s |                                    |    model-config |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |      2.80s |       3.11s | missing-sections(title+descrip...  |    model-config |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |         |                   |                       |                |              |           |             |            0.31s |      1.64s |       2.26s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |         |                   |                       |                |              |           |             |            0.30s |      2.45s |       3.07s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |         |                   |                       |                |              |           |             |            0.31s |      1.41s |       2.03s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |         |                   |                       |                |              |           |             |            0.30s |      1.38s |       1.99s | fabrication, title-length(29), ... |    transformers |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |      1.03s |       1.35s |                                    |    transformers |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               506 |                    84 |            590 |         2320 |       304 |         2.4 |            0.99s |      0.45s |       1.77s | missing-sections(keywords), ...    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               510 |                   118 |            628 |         1026 |       340 |           2 |            1.75s |      0.63s |       2.68s | hallucination, ...                 |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |              3115 |                   143 |           3258 |         2822 |       186 |         7.8 |            2.46s |      0.97s |       3.78s | description-sentences(3)           |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |              1326 |                   141 |           1467 |         3922 |      58.2 |         9.5 |            3.20s |      0.97s |       4.48s | description-sentences(3), ...      |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |              2726 |                     8 |           2734 |          958 |      67.5 |         9.7 |            3.61s |      0.95s |       4.87s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |              3116 |                    97 |           3213 |         1405 |        66 |          13 |            4.26s |      1.37s |       5.95s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |              2289 |                    71 |           2360 |         1261 |      32.3 |          18 |            4.55s |      1.78s |       6.63s | title-length(2), ...               |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |              3116 |                   113 |           3229 |         1377 |        63 |          13 |            4.62s |      1.42s |       6.39s |                                    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               614 |                   500 |           1114 |         1844 |       126 |         5.5 |            4.91s |      0.62s |       5.84s | missing-sections(title+descrip...  |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               777 |                    94 |            871 |          500 |      29.4 |          19 |            5.21s |      2.34s |       7.87s | keyword-count(19)                  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |              1714 |                   500 |           2214 |         4119 |       119 |         5.5 |            5.23s |      0.64s |       6.18s | repetitive(unt), ...               |                 |
| `qnguyen3/nanoLLaVA`                                    |  98,266 |               506 |                   500 |           1006 |         3833 |       107 |         4.7 |            5.30s |      0.59s |       6.19s | repetitive(phrase: "painting g...  |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,812 |               768 |                   500 |           1268 |          582 |       117 |           6 |            6.06s |      1.51s |       7.89s | repetitive(phrase: "10:18: 16:...  |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |              3309 |                   122 |           3431 |         1389 |      38.3 |          16 |            6.07s |      1.67s |       8.04s | description-sentences(3), ...      |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |  13,780 |              1497 |                   500 |           1997 |          867 |       116 |          22 |            6.62s |      2.16s |       9.09s | repetitive(phrase: "use only i...  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |              1714 |                   500 |           2214 |          802 |       125 |         5.5 |            6.72s |      0.66s |       7.67s | repetitive(unt), ...               |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   1,162 |              1497 |                   500 |           1997 |          525 |       125 |          18 |            7.39s |      1.98s |       9.67s | degeneration, ...                  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               777 |                    92 |            869 |          537 |      16.8 |          34 |            7.41s |      3.42s |      11.15s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |              2618 |                   104 |           2722 |          685 |      31.6 |          22 |            7.67s |      2.11s |      10.13s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |  21,474 |              1497 |                   500 |           1997 |         1736 |      79.8 |          37 |            7.77s |      3.29s |      11.37s | missing-sections(title+descrip...  |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               776 |                   347 |           1123 |         1238 |      46.2 |          17 |            8.59s |      2.33s |      11.26s | description-sentences(6), ...      |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,892 |              1326 |                   500 |           1826 |         1692 |      56.4 |         9.5 |           10.07s |      0.86s |      11.22s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |     330 |              6524 |                   500 |           7024 |         1172 |      71.7 |         8.4 |           12.97s |      1.33s |      14.60s | missing-sections(title+descrip...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   8,223 |              3400 |                   500 |           3900 |         1406 |      41.8 |          15 |           14.89s |      1.62s |      16.82s | missing-sections(title+keyword...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |     304 |              6524 |                   500 |           7024 |         1093 |      54.9 |          11 |           15.51s |      1.37s |      17.19s | degeneration, ...                  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |              2789 |                   500 |           3289 |         2468 |      32.3 |          19 |           17.19s |      1.91s |      19.42s | missing-sections(title+descrip...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |   4,877 |              1822 |                   500 |           2322 |          342 |      42.6 |          60 |           17.79s |      7.83s |      25.94s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     264 |             16725 |                   500 |          17225 |         1310 |      86.7 |         8.3 |           19.54s |      0.79s |      20.64s | ⚠️harness(long_context), ...       |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  15,662 |             16723 |                   500 |          17223 |         1165 |      89.5 |         8.3 |           20.78s |      0.72s |      21.82s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |              2289 |                   500 |           2789 |         3057 |      34.2 |          18 |           21.40s |      1.77s |      23.48s | repetitive(phrase: "rencontre...   |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |  51,366 |              1693 |                   500 |           2193 |          109 |      52.7 |          41 |           25.80s |      1.26s |      27.38s | repetitive(phrase: "uk. spring...  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |      11 |               477 |                   500 |            977 |          283 |      20.9 |          15 |           26.01s |      1.68s |      27.99s | repetitive(phrase: "waterfront...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |       1 |              6524 |                   500 |           7024 |          440 |      37.5 |          78 |           28.78s |      7.19s |      36.28s | missing-sections(title+descrip...  |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |   8,923 |             16734 |                   500 |          17234 |          860 |      54.1 |          13 |           29.51s |      1.18s |      31.00s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/pixtral-12b-bf16`                        |   4,054 |              3309 |                   500 |           3809 |         2029 |      19.6 |          28 |           30.17s |      2.56s |      33.04s | degeneration, ...                  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               478 |                   159 |            637 |          191 |      5.02 |          25 |           34.65s |      2.23s |      37.19s | missing-sections(title+descrip...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 115,207 |              1693 |                   500 |           2193 |           89 |      31.6 |          48 |           55.41s |      1.75s |      57.47s | repetitive(phrase: ":: , ...       |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                 0 |                     0 |              0 |            0 |         0 |         5.1 |           60.97s |      0.61s |      61.89s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |   3,766 |             16749 |                   500 |          17249 |          277 |        63 |          76 |           69.49s |      9.48s |      79.28s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |  29,642 |             16749 |                   500 |          17249 |          264 |      86.9 |          35 |           70.19s |      3.28s |      73.80s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     424 |             16749 |                   500 |          17249 |          221 |      27.7 |          26 |           94.66s |      2.26s |      97.25s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |      13 |             16749 |                   500 |          17249 |          203 |      16.6 |          39 |          113.63s |      3.20s |     117.15s | refusal(explicit_refusal), ...     |                 |

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

- `numpy`: `2.4.3`
- `mlx`: `0.31.2.dev20260327+0ff1115a`
- `mlx-vlm`: `0.4.1`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.8.0`
- `transformers`: `5.4.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.1.1`

_Report generated on: 2026-03-27 23:26:59 GMT_
