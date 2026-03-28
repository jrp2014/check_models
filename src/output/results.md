# Model Performance Results

_Generated on 2026-03-28 11:07:10 GMT_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 8 (top owners: transformers=6,
  model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=8, clean outputs=3/44.
- _Useful now:_ 1 clean A/B model(s) worth first review.
- _Review watchlist:_ 43 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=12, neutral=1, worse=31 (baseline B 79/100).
- _Quality signal frequency:_ missing_sections=28, cutoff=25,
  trusted_hint_ignored=19, context_ignored=19, description_length=9,
  repetitive=9.
- _Runtime pattern:_ decode dominates measured phase time (89%; 44/52 measured
  model(s)).
- _Phase totals:_ model load=98.22s, prompt prep=0.11s, decode=822.04s,
  cleanup=5.05s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=44, exception=8.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (359.0 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.46s)
- **📊 Average TPS:** 90.7 across 44 models

## 📈 Resource Usage

- **Total peak memory:** 881.8 GB
- **Average peak memory:** 20.0 GB
- **Memory efficiency:** 254 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 12 | ✅ B: 9 | 🟡 C: 5 | 🟠 D: 10 | ❌ F: 8

**Average Utility Score:** 56/100

**Existing Metadata Baseline:** ✅ B (79/100)
**Vs Existing Metadata:** Avg Δ -23 | Better: 12, Neutral: 1, Worse: 31

- **Best for cataloging:** `mlx-community/X-Reasoner-7B-8bit` (🏆 A, 98/100)
- **Worst for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (❌ F, 0/100)

### ⚠️ 18 Models with Low Utility (D/F)

- `Qwen/Qwen3-VL-2B-Instruct`: 🟠 D (38/100) - Mostly echoes context without adding value
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (38/100) - Mostly echoes context without adding value
- `mlx-community/LFM2.5-VL-1.6B-bf16`: 🟠 D (38/100) - Mostly echoes context without adding value
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (48/100) - Mostly echoes context without adding value
- `mlx-community/Molmo-7B-D-0924-8bit`: 🟠 D (48/100) - Missing requested structure
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (36/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: 🟠 D (48/100) - Lacks visual description of image
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/pixtral-12b-8bit`: 🟠 D (39/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (8):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
  - `mlx-community/deepseek-vl2-8bit` (`Processor Error`)
  - `mlx-community/paligemma2-10b-ft-docci-448-6bit` (`API Mismatch`)
  - `mlx-community/paligemma2-10b-ft-docci-448-bf16` (`API Mismatch`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (`API Mismatch`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (`API Mismatch`)
  - `prince-canuma/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (9):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "reflections, white blossoms, t..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "use only if a..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "waterfront living, waterfront ..."`)
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

- **Generation Tps**: Avg: 90.7 | Min: 5.02 | Max: 359
- **Peak Memory**: Avg: 20 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 20.89s | Min: 1.57s | Max: 96.23s
- **Generation Time**: Avg: 18.66s | Min: 0.79s | Max: 92.77s
- **Model Load Time**: Avg: 1.93s | Min: 0.46s | Max: 8.66s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (89%; 44/52 measured model(s)).
- **Phase totals:** model load=98.22s, prompt prep=0.11s, decode=822.04s, cleanup=5.05s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=44, exception=8.

### ⏱ Timing Snapshot

- **Validation overhead:** 15.71s total (avg 0.30s across 52 model(s)).
- **First-token latency:** Avg 9.59s | Min 0.08s | Max 64.35s across 44 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/X-Reasoner-7B-8bit`](model_gallery.md#model-mlx-community-x-reasoner-7b-8bit)
  (A 98/100 | Gen 57.4 TPS | Peak 13 | A 98/100 | ⚠️REVIEW:cutoff;
  ⚠️HARNESS:long_context; ...)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (F 28/100 | Gen 359 TPS | Peak 2.7 | F 28/100 | Missing sections (keywords);
  Title length violation (11 words; expected 5-10); ...)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (A 88/100 | Gen 344 TPS | Peak 2.2 | A 88/100 | Context ignored (missing:
  large, former, warehouse, London, Canal); Missing sections (title,
  description, keywords); ...)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (B 69/100 | Gen 62.8 TPS | Peak 13 | B 69/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (8):_ [`microsoft/Florence-2-large-ft`](model_gallery.md#model-microsoft-florence-2-large-ft),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16),
  [`mlx-community/deepseek-vl2-8bit`](model_gallery.md#model-mlx-community-deepseek-vl2-8bit),
  [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](model_gallery.md#model-mlx-community-paligemma2-10b-ft-docci-448-6bit),
  +4 more. Example: `Model Error`.
- _🔄 Repetitive Output (9):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +5 more. Example: token: `unt`.
- _👻 Hallucinations (4):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/Molmo-7B-D-0924-bf16`](model_gallery.md#model-mlx-community-molmo-7b-d-0924-bf16),
  [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8).
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (18):_ [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +14 more. Common weakness: Mostly echoes context without adding value.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `transformers` | 6 | API Mismatch, Model Error | `microsoft/Florence-2-large-ft`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`, `prince-canuma/Florence-2-large-ft` |
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

_Overall runtime:_ 942.23s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |      0.45s |       0.76s |                                    |    transformers |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.34s |       2.65s |                                    |    model-config |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |      2.79s |       3.10s | missing-sections(title+descrip...  |    model-config |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |         |                   |                       |                |              |           |             |            0.31s |      1.59s |       2.20s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |         |                   |                       |                |              |           |             |            0.30s |      2.54s |       3.14s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |         |                   |                       |                |              |           |             |            0.30s |      1.39s |       2.00s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |         |                   |                       |                |              |           |             |            0.29s |      1.09s |       1.69s | fabrication, title-length(29), ... |    transformers |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |      1.09s |       1.40s |                                    |    transformers |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               760 |                    85 |            845 |         5202 |       329 |         2.8 |            0.79s |      0.58s |       1.67s | missing-sections(title+descrip...  |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               506 |                    84 |            590 |         5589 |       359 |         2.7 |            0.81s |      0.46s |       1.57s | missing-sections(keywords), ...    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               570 |                   166 |            736 |         6982 |       189 |         3.7 |            1.34s |      0.53s |       2.18s | title-length(23), ...              |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               510 |                   118 |            628 |         5071 |       344 |         2.2 |            1.34s |      0.66s |       2.31s | hallucination, ...                 |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,115 |                   143 |          3,258 |         3049 |       183 |         7.8 |            2.40s |      1.22s |       3.93s | description-sentences(3)           |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,726 |                     8 |          2,734 |         1327 |      70.3 |         9.7 |            2.81s |      0.94s |       4.05s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |             1,326 |                   141 |          1,467 |         3890 |      56.9 |         9.5 |            3.25s |      0.99s |       4.54s | description-sentences(3), ...      |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,116 |                    97 |          3,213 |         1359 |      65.3 |          13 |            4.34s |      1.47s |       6.13s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,289 |                    71 |          2,360 |         1424 |      32.2 |          18 |            4.35s |      1.82s |       6.49s | title-length(2), ...               |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               777 |                    94 |            871 |          642 |      31.7 |          19 |            4.64s |      2.31s |       7.26s | keyword-count(19)                  |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,812 |               768 |                   500 |          1,268 |         2605 |       128 |           6 |            4.64s |      1.45s |       6.41s | repetitive(phrase: "10:18: 16:...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,116 |                   113 |          3,229 |         1354 |      62.8 |          13 |            4.69s |      1.43s |       6.45s |                                    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               614 |                   500 |          1,114 |         1852 |       130 |         5.5 |            4.79s |      0.76s |       5.85s | missing-sections(title+descrip...  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,714 |                   500 |          2,214 |         4220 |       123 |         5.5 |            5.04s |      0.68s |       6.01s | repetitive(unt), ...               |                 |
| `qnguyen3/nanoLLaVA`                                    |  98,266 |               506 |                   500 |          1,006 |         4468 |       112 |         4.7 |            5.06s |      0.49s |       5.85s | repetitive(phrase: "painting g...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,714 |                   500 |          2,214 |         4159 |       123 |         5.5 |            5.09s |      0.70s |       6.10s | repetitive(unt), ...               |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   1,162 |             1,497 |                   500 |          1,997 |         1749 |       128 |          18 |            5.30s |      1.99s |       7.59s | degeneration, ...                  |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,309 |                   122 |          3,431 |         1811 |      39.2 |          16 |            5.43s |      1.74s |       7.47s | description-sentences(3), ...      |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |  13,780 |             1,497 |                   500 |          1,997 |         1787 |       115 |          22 |            5.74s |      2.43s |       8.48s | repetitive(phrase: "use only i...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               777 |                    92 |            869 |          561 |      17.6 |          34 |            7.06s |      3.40s |      10.77s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,618 |                   104 |          2,722 |          753 |      32.1 |          22 |            7.26s |      2.11s |       9.68s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |  21,474 |             1,497 |                   500 |          1,997 |         1752 |      79.4 |          37 |            7.80s |      3.28s |      11.38s | missing-sections(title+descrip...  |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               776 |                   347 |          1,123 |         1767 |        49 |          17 |            7.97s |      2.24s |      10.52s | description-sentences(6), ...      |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,892 |             1,326 |                   500 |          1,826 |         3803 |      56.5 |         9.5 |            9.62s |      0.90s |      10.82s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |     330 |             6,524 |                   500 |          7,024 |         1097 |      71.4 |         8.4 |           13.39s |      1.41s |      15.12s | missing-sections(title+descrip...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   8,223 |             3,400 |                   500 |          3,900 |         1535 |        42 |          15 |           14.61s |      1.79s |      16.71s | missing-sections(title+keyword...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |   4,877 |             1,822 |                   500 |          2,322 |          814 |      42.5 |          60 |           14.73s |      5.61s |      20.65s | missing-sections(title+descrip...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |     304 |             6,524 |                   500 |          7,024 |         1167 |      54.1 |          11 |           15.27s |      1.46s |      17.04s | degeneration, ...                  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,789 |                   500 |          3,289 |         2354 |      31.8 |          19 |           17.53s |      1.91s |      19.77s | missing-sections(title+descrip...  |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  15,662 |            16,723 |                   500 |         17,223 |         1263 |      88.1 |         8.3 |           19.74s |      0.68s |      20.71s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     264 |            16,725 |                   500 |         17,225 |         1175 |      87.1 |         8.3 |           20.89s |      0.74s |      21.93s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |             2,289 |                   500 |          2,789 |         3032 |        34 |          18 |           21.53s |      1.72s |      23.56s | repetitive(phrase: "rencontre...   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |   8,923 |            16,734 |                   500 |         17,234 |         1123 |      57.4 |          13 |           24.40s |      1.17s |      25.88s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |      11 |               477 |                   500 |            977 |          299 |      20.9 |          15 |           25.93s |      1.58s |      27.82s | repetitive(phrase: "waterfront...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |       1 |             6,524 |                   500 |          7,024 |          464 |      37.4 |          78 |           27.89s |      6.65s |      34.86s | missing-sections(title+descrip...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |      17 |             1,693 |                   500 |          2,193 |         88.9 |      52.4 |          41 |           29.48s |      1.30s |      31.10s | fabrication, ...                   |                 |
| `mlx-community/pixtral-12b-bf16`                        |   4,054 |             3,309 |                   500 |          3,809 |         2074 |      19.8 |          28 |           29.75s |      2.62s |      32.67s | degeneration, ...                  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               478 |                   159 |            637 |          253 |      5.02 |          25 |           34.01s |      2.27s |      36.58s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |  29,642 |            16,749 |                   500 |         17,249 |          366 |      90.6 |          35 |           52.21s |      3.14s |      55.66s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |   3,766 |            16,749 |                   500 |         17,249 |          342 |      66.1 |          76 |           57.55s |      8.66s |      66.52s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |  90,838 |             1,693 |                   500 |          2,193 |         86.2 |      30.4 |          48 |           57.56s |      1.74s |      59.61s | hallucination, ...                 |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,734 |                     4 |         16,738 |          288 |       256 |         5.1 |           58.93s |      0.47s |      59.71s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     424 |            16,749 |                   500 |         17,249 |          263 |      30.1 |          26 |           81.11s |      2.30s |      83.73s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |      13 |            16,749 |                   500 |         17,249 |          260 |      18.2 |          39 |           92.77s |      3.15s |      96.23s | refusal(explicit_refusal), ...     |                 |

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
- `mlx`: `0.31.2.dev20260328+0ff1115a`
- `mlx-vlm`: `0.4.1`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.8.0`
- `transformers`: `5.4.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.1.1`

_Report generated on: 2026-03-28 11:07:10 GMT_
