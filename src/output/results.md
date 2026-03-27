# Model Performance Results

_Generated on 2026-03-27 16:58:36 GMT_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 11 (top owners: transformers=7, mlx-vlm=2,
  model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=8, clean outputs=3/41.
- _Useful now:_ 1 clean A/B model(s) worth first review.
- _Review watchlist:_ 40 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=11, neutral=0, worse=30 (baseline B 79/100).
- _Quality signal frequency:_ missing_sections=26, cutoff=26,
  context_ignored=17, trusted_hint_ignored=17, repetitive=11,
  description_length=8.
- _Runtime pattern:_ decode dominates measured phase time (90%; 41/52 measured
  model(s)).
- _Phase totals:_ model load=94.55s, prompt prep=0.12s, decode=848.22s,
  cleanup=4.82s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=41, exception=11.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (345.2 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (2.7 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 75.7 across 41 models

## 📈 Resource Usage

- **Total peak memory:** 873.0 GB
- **Average peak memory:** 21.3 GB
- **Memory efficiency:** 254 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 11 | ✅ B: 9 | 🟡 C: 5 | 🟠 D: 7 | ❌ F: 9

**Average Utility Score:** 56/100

**Existing Metadata Baseline:** ✅ B (79/100)
**Vs Existing Metadata:** Avg Δ -23 | Better: 11, Neutral: 0, Worse: 30

- **Best for cataloging:** `mlx-community/X-Reasoner-7B-8bit` (🏆 A, 98/100)
- **Worst for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (❌ F, 0/100)

### ⚠️ 16 Models with Low Utility (D/F)

- `Qwen/Qwen3-VL-2B-Instruct`: 🟠 D (38/100) - Mostly echoes context without adding value
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (48/100) - Mostly echoes context without adding value
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (30/100) - Mostly echoes context without adding value
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (36/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: 🟠 D (48/100) - Lacks visual description of image
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (11/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/pixtral-12b-8bit`: 🟠 D (39/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (11):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
  - `mlx-community/FastVLM-0.5B-bf16` (`Model Error`)
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
  - `mlx-community/Qwen3.5-35B-A3B-bf16` (token: `phrase: "no clear text. *..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/X-Reasoner-7B-8bit` (token: `phrase: "- monochrome stripes -..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "10:18: 16:42: 10:18: 16:42:..."`)
  - `qnguyen3/nanoLLaVA` (token: `phrase: "painting glasses glasses, moto..."`)
- **👻 Hallucinations (1):**
  - `microsoft/Phi-3.5-vision-instruct`
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 75.7 | Min: 5.03 | Max: 345
- **Peak Memory**: Avg: 21 | Min: 2.7 | Max: 78
- **Total Time**: Avg: 22.90s | Min: 1.56s | Max: 102.73s
- **Generation Time**: Avg: 20.64s | Min: 0.81s | Max: 99.28s
- **Model Load Time**: Avg: 1.95s | Min: 0.45s | Max: 8.60s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 41/52 measured model(s)).
- **Phase totals:** model load=94.55s, prompt prep=0.12s, decode=848.22s, cleanup=4.82s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=41, exception=11.

### ⏱ Timing Snapshot

- **Validation overhead:** 15.73s total (avg 0.30s across 52 model(s)).
- **First-token latency:** Avg 10.91s | Min 0.09s | Max 71.13s across 41 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/X-Reasoner-7B-8bit`](model_gallery.md#model-mlx-community-x-reasoner-7b-8bit)
  (A 98/100 | Gen 53.7 TPS | Peak 13 | A 98/100 | ⚠️REVIEW:cutoff;
  ⚠️HARNESS:long_context; ...)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (F 28/100 | Gen 345 TPS | Peak 2.7 | F 28/100 | Missing sections (keywords);
  Title length violation (11 words; expected 5-10); ...)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (F 28/100 | Gen 345 TPS | Peak 2.7 | F 28/100 | Missing sections (keywords);
  Title length violation (11 words; expected 5-10); ...)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (B 69/100 | Gen 63.3 TPS | Peak 13 | B 69/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (11):_ [`microsoft/Florence-2-large-ft`](model_gallery.md#model-microsoft-florence-2-large-ft),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/LFM2-VL-1.6B-8bit`](model_gallery.md#model-mlx-community-lfm2-vl-16b-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  +7 more. Example: `Model Error`.
- _🔄 Repetitive Output (11):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +7 more. Example: token: `unt`.
- _👻 Hallucinations (1):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct).
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
| `transformers` | 7 | API Mismatch, Model Error | `microsoft/Florence-2-large-ft`, `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`, `prince-canuma/Florence-2-large-ft` |
| `mlx-vlm` | 2 | Model Error | `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `model-config` | 2 | Processor Error | `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/deepseek-vl2-8bit` |

### Actionable Items by Package

#### transformers

- microsoft/Florence-2-large-ft (Model Error)
  - Error: `Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
  - Type: `ValueError`
- mlx-community/FastVLM-0.5B-bf16 (Model Error)
  - Error: `Model loading failed: No module named 'timm'`
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

_Overall runtime:_ 964.42s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |      0.36s |       0.66s |                                    |    transformers |
| `mlx-community/FastVLM-0.5B-bf16`                       |         |                   |                       |                |              |           |             |                  |      0.33s |       0.64s |                                    |    transformers |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |         |                   |                       |                |              |           |             |            0.40s |      0.51s |       1.22s | ⚠️harness(stop_token), ...         |         mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |         |                   |                       |                |              |           |             |            0.42s |      0.54s |       1.27s | ⚠️harness(stop_token), ...         |         mlx-vlm |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.23s |       2.53s |                                    |    model-config |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |      2.81s |       3.12s | missing-sections(title+descrip...  |    model-config |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |         |                   |                       |                |              |           |             |            0.30s |      1.63s |       2.24s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |         |                   |                       |                |              |           |             |            0.30s |      2.47s |       3.07s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |         |                   |                       |                |              |           |             |            0.29s |      1.47s |       2.07s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |         |                   |                       |                |              |           |             |            0.29s |      1.16s |       1.75s | fabrication, title-length(29), ... |    transformers |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |      1.01s |       1.32s |                                    |    transformers |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               506 |                    84 |            590 |         5603 |       345 |         2.7 |            0.81s |      0.45s |       1.56s | missing-sections(keywords), ...    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,115 |                   143 |          3,258 |         3073 |       186 |         7.8 |            2.29s |      0.95s |       3.55s | description-sentences(3)           |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |             1,326 |                   141 |          1,467 |         3915 |      57.6 |         9.5 |            3.20s |      0.86s |       4.37s | description-sentences(3), ...      |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,726 |                     8 |          2,734 |         1063 |      67.2 |         9.7 |            3.31s |      0.97s |       4.58s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,116 |                    97 |          3,213 |         1384 |      66.2 |          13 |            4.25s |      1.39s |       5.95s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,289 |                    71 |          2,360 |         1439 |      32.3 |          18 |            4.33s |      1.78s |       6.42s | title-length(2), ...               |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,116 |                   113 |          3,229 |         1441 |      63.3 |          13 |            4.46s |      1.38s |       6.15s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               777 |                    94 |            871 |          644 |      31.5 |          19 |            4.64s |      2.26s |       7.22s | keyword-count(19)                  |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,812 |               768 |                   500 |          1,268 |         2606 |       125 |           6 |            4.74s |      1.44s |       6.49s | repetitive(phrase: "10:18: 16:...  |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               614 |                   500 |          1,114 |         1850 |       131 |         5.5 |            4.75s |      0.63s |       5.68s | missing-sections(title+descrip...  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,714 |                   500 |          2,214 |         4240 |       127 |         5.5 |            4.92s |      0.63s |       5.84s | repetitive(unt), ...               |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,714 |                   500 |          2,214 |         4123 |       124 |         5.5 |            5.06s |      0.64s |       6.02s | repetitive(unt), ...               |                 |
| `qnguyen3/nanoLLaVA`                                    |  98,266 |               506 |                   500 |          1,006 |         4435 |       112 |         4.7 |            5.06s |      0.50s |       5.86s | repetitive(phrase: "painting g...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   1,162 |             1,497 |                   500 |          1,997 |         1772 |       128 |          18 |            5.29s |      2.00s |       7.59s | degeneration, ...                  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |  13,780 |             1,497 |                   500 |          1,997 |         1783 |       116 |          22 |            5.72s |      2.14s |       8.16s | repetitive(phrase: "use only i...  |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,309 |                   122 |          3,431 |         1490 |        39 |          16 |            5.84s |      1.73s |       7.86s | description-sentences(3), ...      |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               777 |                    92 |            869 |          470 |      17.3 |          34 |            7.42s |      3.36s |      11.10s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,618 |                   104 |          2,722 |          684 |      31.5 |          22 |            7.70s |      2.16s |      10.21s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |  21,474 |             1,497 |                   500 |          1,997 |         1755 |      79.8 |          37 |            7.76s |      3.28s |      11.35s | missing-sections(title+descrip...  |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               776 |                   347 |          1,123 |         1775 |      48.8 |          17 |            7.99s |      2.25s |      10.55s | description-sentences(6), ...      |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,892 |             1,326 |                   500 |          1,826 |         3833 |      55.9 |         9.5 |            9.72s |      0.86s |      10.88s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |     330 |             6,524 |                   500 |          7,024 |         1158 |      69.8 |         8.4 |           13.25s |      1.35s |      14.92s | missing-sections(title+descrip...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |   4,877 |             1,822 |                   500 |          2,322 |         1114 |      42.4 |          60 |           14.21s |      4.81s |      19.35s | missing-sections(title+descrip...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   8,223 |             3,400 |                   500 |          3,900 |         1455 |      41.6 |          15 |           14.85s |      1.63s |      16.79s | missing-sections(title+keyword...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |     304 |             6,524 |                   500 |          7,024 |         1181 |      54.6 |          11 |           15.12s |      1.38s |      16.81s | degeneration, ...                  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,789 |                   500 |          3,289 |         2479 |      32.2 |          19 |           17.23s |      2.17s |      19.70s | missing-sections(title+descrip...  |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  15,662 |            16,723 |                   500 |         17,223 |         1257 |      89.2 |         8.3 |           19.77s |      0.70s |      20.76s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     264 |            16,725 |                   500 |         17,225 |         1155 |      87.5 |         8.3 |           21.14s |      0.75s |      22.21s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |             2,289 |                   500 |          2,789 |         2932 |      34.2 |          18 |           21.47s |      1.73s |      23.51s | repetitive(phrase: "rencontre...   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |   8,923 |            16,734 |                   500 |         17,234 |          982 |      53.7 |          13 |           27.16s |      1.18s |      28.64s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |      11 |               477 |                   500 |            977 |          283 |      19.5 |          15 |           27.78s |      1.47s |      29.56s | repetitive(phrase: "waterfront...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |     594 |             1,693 |                   500 |          2,193 |         91.3 |      52.7 |          41 |           28.84s |      1.23s |      30.38s | repetitive(phrase: "uk. spring...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |       1 |             6,524 |                   500 |          7,024 |          444 |      34.8 |          78 |           29.54s |      5.60s |      35.45s | missing-sections(title+descrip...  |                 |
| `mlx-community/pixtral-12b-bf16`                        |   4,054 |             3,309 |                   500 |          3,809 |         2067 |        20 |          28 |           29.59s |      2.55s |      32.44s | degeneration, ...                  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               478 |                   159 |            637 |          253 |      5.03 |          25 |           33.93s |      2.24s |      36.48s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |   3,050 |            16,749 |                   500 |         17,249 |          351 |      88.7 |          35 |           53.92s |      3.34s |      57.60s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 121,621 |             1,693 |                   500 |          2,193 |         87.9 |      30.5 |          48 |           56.85s |      1.75s |      58.91s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,734 |                     4 |         16,738 |          290 |       254 |         5.1 |           58.61s |      0.57s |      59.48s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |     328 |            16,749 |                   500 |         17,249 |          295 |      63.5 |          76 |           65.42s |      8.60s |      74.36s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     424 |            16,749 |                   500 |         17,249 |          235 |      28.8 |          26 |           89.02s |      2.17s |      91.51s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |  10,151 |            16,749 |                   500 |         17,249 |          238 |      17.5 |          39 |           99.28s |      3.14s |     102.73s | refusal(explicit_refusal), ...     |                 |

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
- `mlx`: `0.31.2.dev20260322+38ad2570`
- `mlx-vlm`: `0.4.1`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.8.0`
- `transformers`: `5.4.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.1.1`

_Report generated on: 2026-03-27 16:58:36 GMT_
