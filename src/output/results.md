# Model Performance Results

_Generated on 2026-03-27 14:43:58 GMT_

## 🎯 Action Snapshot

- **Framework/runtime failures:** 11 (top owners: transformers=7, mlx-vlm=2, model-config=2).
- **Next action:** review failure ownership below and use diagnostics.md for filing.
- **Maintainer signals:** harness-risk successes=8, clean outputs=3/41.
- **Useful now:** 1 clean A/B model(s) worth first review.
- **Review watchlist:** 40 model(s) with breaking or lower-value output.
- **Preflight compatibility:** 1 informational warning(s); do not treat these alone as run failures.
- **Escalate only if:** they line up with unexpected TF/Flax/JAX imports, startup hangs, or backend/runtime crashes.
- **Vs existing metadata:** better=11, neutral=0, worse=30 (baseline B 79/100).
- **Quality signal frequency:** missing_sections=26, cutoff=26, context_ignored=18, trusted_hint_ignored=18, repetitive=10, description_length=8.
- **Runtime pattern:** decode dominates measured phase time (90%; 41/52 measured model(s)).
- **Phase totals:** model load=93.28s, prompt prep=0.12s, decode=851.73s, cleanup=4.68s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=41, exception=11.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (343.1 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (2.7 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 76.0 across 41 models

## 📈 Resource Usage

- **Total peak memory:** 873.1 GB
- **Average peak memory:** 21.3 GB
- **Memory efficiency:** 254 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 11 | ✅ B: 9 | 🟡 C: 5 | 🟠 D: 8 | ❌ F: 8

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
- `mlx-community/Molmo-7B-D-0924-8bit`: 🟠 D (41/100) - Lacks visual description of image
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
- **🔄 Repetitive Output (10):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "reflections, white blossoms, t..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "use only if a..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "waterfront living, waterfront ..."`)
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

- **Generation Tps**: Avg: 76.0 | Min: 5.12 | Max: 343
- **Peak Memory**: Avg: 21 | Min: 2.7 | Max: 78
- **Total Time**: Avg: 22.96s | Min: 1.56s | Max: 102.42s
- **Generation Time**: Avg: 20.72s | Min: 0.81s | Max: 98.97s
- **Model Load Time**: Avg: 1.93s | Min: 0.45s | Max: 7.68s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 41/52 measured model(s)).
- **Phase totals:** model load=93.28s, prompt prep=0.12s, decode=851.73s, cleanup=4.68s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=41, exception=11.

### ⏱ Timing Snapshot

- **Validation overhead:** 15.64s total (avg 0.30s across 52 model(s)).
- **First-token latency:** Avg 11.05s | Min 0.09s | Max 70.86s across 41 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- **Best cataloging quality:** [`mlx-community/X-Reasoner-7B-8bit`](model_gallery.md#model-mlx-community-x-reasoner-7b-8bit) (A 98/100 | Gen 54.6 TPS | Peak 13 | A 98/100 | ⚠️REVIEW:cutoff; ⚠️HARNESS:long_context; ...)
- **Fastest generation:** [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit) (F 28/100 | Gen 343 TPS | Peak 2.7 | F 28/100 | Missing sections (keywords); Title length violation (11 words; expected 5-10); ...)
- **Lowest memory footprint:** [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit) (F 28/100 | Gen 343 TPS | Peak 2.7 | F 28/100 | Missing sections (keywords); Title length violation (11 words; expected 5-10); ...)
- **Best balance:** [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4) (B 69/100 | Gen 63.4 TPS | Peak 13 | B 69/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery when available.

- **❌ Failed Models (11):** [`microsoft/Florence-2-large-ft`](model_gallery.md#model-microsoft-florence-2-large-ft), [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16), [`mlx-community/LFM2-VL-1.6B-8bit`](model_gallery.md#model-mlx-community-lfm2-vl-16b-8bit), [`mlx-community/LFM2.5-VL-1.6B-bf16`](model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16), +7 more. Example: `Model Error`.
- **🔄 Repetitive Output (10):** [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct), [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct), [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16), [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit), +6 more. Example: token: `unt`.
- **👻 Hallucinations (1):** [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct).
- **📝 Formatting Issues (6):** [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit), [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4), [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4), [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16), +2 more.
- **Low-utility outputs (16):** [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct), [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit), [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit), [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16), +12 more. Common weakness: Mostly echoes context without adding value.

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `transformers` | 7 | API Mismatch, Model Error | `microsoft/Florence-2-large-ft`, `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`, `prince-canuma/Florence-2-large-ft` |
| `mlx-vlm` | 2 | Model Error | `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `model-config` | 2 | Processor Error | `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/deepseek-vl2-8bit` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### transformers

- **microsoft/Florence-2-large-ft** (Model Error)
  - Error: `Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
  - Type: `ValueError`
- **mlx-community/FastVLM-0.5B-bf16** (Model Error)
  - Error: `Model loading failed: No module named 'timm'`
  - Type: `ValueError`
- **mlx-community/paligemma2-10b-ft-docci-448-6bit** (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-6bit: Failed to process inputs with error: Imag...`
  - Type: `ValueError`
- **mlx-community/paligemma2-10b-ft-docci-448-bf16** (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-10b-ft-docci-448-bf16: Failed to process inputs with error: Imag...`
  - Type: `ValueError`
- **mlx-community/paligemma2-3b-ft-docci-448-bf16** (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-3b-ft-docci-448-bf16: Failed to process inputs with error: Image...`
  - Type: `ValueError`
- **mlx-community/paligemma2-3b-pt-896-4bit** (API Mismatch)
  - Error: `Model generation failed for mlx-community/paligemma2-3b-pt-896-4bit: Failed to process inputs with error: ImagesKwarg...`
  - Type: `ValueError`
- **prince-canuma/Florence-2-large-ft** (Model Error)
  - Error: `Model loading failed: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
  - Type: `ValueError`

#### mlx-vlm

- **mlx-community/LFM2-VL-1.6B-8bit** (Model Error)
  - Error: `Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Failed to process inputs with error: object of type 'num...`
  - Type: `ValueError`
- **mlx-community/LFM2.5-VL-1.6B-bf16** (Model Error)
  - Error: `Model generation failed for mlx-community/LFM2.5-VL-1.6B-bf16: Failed to process inputs with error: object of type 'n...`
  - Type: `ValueError`

#### model-config

- **mlx-community/MolmoPoint-8B-fp16** (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`
- **mlx-community/deepseek-vl2-8bit** (Processor Error)
  - Error: `Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimo...`
  - Type: `ValueError`

**Prompt used:**

<!-- markdownlint-disable MD028 MD049 -->
>
> Analyze this image for cataloguing metadata, using British English.
>
> Use only details that are clearly and definitely visible in the image. If a detail is
> uncertain, ambiguous, partially obscured, too small to verify, or not directly visible,
> leave it out. Do not guess.
>
> Treat the metadata hints below as a draft catalog record. Keep only details that are
> clearly confirmed by the image, correct anything contradicted by the image, and add
> important visible details that are definitely present.
>
> Return exactly these three sections, and nothing else:
>
> Title:
> \- 5-10 words, concrete and factual, limited to clearly visible content.
> \- Output only the title text after the label.
> \- Do not repeat or paraphrase these instructions in the title.
>
> Description:
> \- 1-2 factual sentences describing the main visible subject, setting, lighting, action,
> and other distinctive visible details. Omit anything uncertain or inferred.
> \- Output only the description text after the label.
>
> Keywords:
> \- 10-18 unique comma-separated terms based only on clearly visible subjects, setting,
> colors, composition, and style. Omit uncertain tags rather than guessing.
> \- Output only the keyword list after the label.
>
> Rules:
> \- Include only details that are definitely visible in the image.
> \- Reuse metadata terms only when they are clearly supported by the image.
> \- If metadata and image disagree, follow the image.
> \- Prefer omission to speculation.
> \- Do not copy prompt instructions into the Title, Description, or Keywords fields.
> \- Do not infer identity, location, event, brand, species, time period, or intent unless
> visually obvious.
> \- Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually confirmed):
> \- Description hint: A large brick former warehouse, the London Canal Museum, stands
> beside the Regent's Canal in King's Cross, London, on a sunny spring day. Several
> narrowboats are moored along the towpath, their reflections visible in the calm water.
> White blossoms in the foreground frame the tranquil urban scene.
> \- Capture metadata: Taken on 2026-03-21 16:42:44 GMT (at 16:42:44 local time). GPS:
> 51.532500°N, 0.122500°W.
<!-- markdownlint-enable MD028 MD049 -->

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 967.01s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |      0.37s |       0.67s |                                    |    transformers |
| `mlx-community/FastVLM-0.5B-bf16`                       |         |                   |                       |                |              |           |             |                  |      0.33s |       0.63s |                                    |    transformers |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |         |                   |                       |                |              |           |             |            0.40s |      0.51s |       1.22s | ⚠️harness(stop_token), ...         |         mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |         |                   |                       |                |              |           |             |            0.42s |      0.54s |       1.27s | ⚠️harness(stop_token), ...         |         mlx-vlm |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.24s |       2.54s |                                    |    model-config |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |      2.80s |       3.12s | missing-sections(title+descrip...  |    model-config |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |         |                   |                       |                |              |           |             |            0.30s |      1.58s |       2.19s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |         |                   |                       |                |              |           |             |            0.29s |      2.46s |       3.05s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |         |                   |                       |                |              |           |             |            0.30s |      1.37s |       1.97s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |         |                   |                       |                |              |           |             |            0.29s |      1.07s |       1.65s | fabrication, title-length(29), ... |    transformers |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |      0.98s |       1.29s |                                    |    transformers |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               506 |                    84 |            590 |         5627 |       343 |         2.7 |            0.81s |      0.45s |       1.56s | missing-sections(keywords), ...    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,115 |                   143 |          3,258 |         3072 |       186 |         7.8 |            2.30s |      0.93s |       3.54s | description-sentences(3)           |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,726 |                     8 |          2,734 |         1108 |      66.9 |         9.7 |            3.20s |      0.95s |       4.45s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |             1,326 |                   141 |          1,467 |         3933 |      56.6 |         9.5 |            3.25s |      0.86s |       4.40s | description-sentences(3), ...      |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,116 |                    97 |          3,213 |         1385 |      65.8 |          13 |            4.24s |      1.33s |       5.87s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,116 |                   113 |          3,229 |         1424 |      63.4 |          13 |            4.49s |      1.41s |       6.21s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,289 |                    71 |          2,360 |         1201 |      32.1 |          18 |            4.65s |      1.76s |       6.72s | title-length(2), ...               |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,812 |               768 |                   500 |          1,268 |         2608 |       126 |           6 |            4.69s |      1.45s |       6.46s | repetitive(phrase: "10:18: 16:...  |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               614 |                   500 |          1,114 |         1853 |       132 |         5.5 |            4.73s |      0.63s |       5.66s | missing-sections(title+descrip...  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,714 |                   500 |          2,214 |         4196 |       130 |         5.5 |            4.83s |      0.67s |       5.78s | repetitive(unt), ...               |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               777 |                    94 |            871 |          567 |      30.6 |          19 |            4.89s |      2.28s |       7.49s | keyword-count(19)                  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,714 |                   500 |          2,214 |         4212 |       124 |         5.5 |            5.05s |      0.64s |       5.99s | repetitive(unt), ...               |                 |
| `qnguyen3/nanoLLaVA`                                    |  98,266 |               506 |                   500 |          1,006 |         4532 |       112 |         4.7 |            5.05s |      0.49s |       5.84s | repetitive(phrase: "painting g...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   1,162 |             1,497 |                   500 |          1,997 |         1784 |       129 |          18 |            5.22s |      1.95s |       7.48s | degeneration, ...                  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |  13,780 |             1,497 |                   500 |          1,997 |         1805 |       116 |          22 |            5.70s |      2.14s |       8.15s | repetitive(phrase: "use only i...  |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,309 |                   122 |          3,431 |         1555 |      38.9 |          16 |            5.76s |      1.62s |       7.68s | description-sentences(3), ...      |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               777 |                    92 |            869 |          563 |      17.8 |          34 |            7.01s |      3.53s |      10.85s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,618 |                   104 |          2,722 |          735 |        32 |          22 |            7.34s |      2.08s |       9.75s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |  21,474 |             1,497 |                   500 |          1,997 |         1768 |      79.9 |          37 |            7.72s |      3.28s |      11.31s | missing-sections(title+descrip...  |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               776 |                   347 |          1,123 |         1786 |      48.7 |          17 |            8.00s |      2.26s |      10.58s | description-sentences(6), ...      |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,892 |             1,326 |                   500 |          1,826 |         4030 |      57.4 |         9.5 |            9.46s |      1.33s |      11.09s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |     330 |             6,524 |                   500 |          7,024 |         1093 |      71.7 |         8.4 |           13.38s |      1.33s |      15.03s | missing-sections(title+descrip...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |   4,877 |             1,822 |                   500 |          2,322 |         1099 |      43.2 |          60 |           13.99s |      4.96s |      19.27s | missing-sections(title+descrip...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   8,223 |             3,400 |                   500 |          3,900 |         1507 |      42.2 |          15 |           14.60s |      1.62s |      16.53s | missing-sections(title+keyword...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |     304 |             6,524 |                   500 |          7,024 |         1156 |      54.4 |          11 |           15.28s |      1.41s |      16.99s | degeneration, ...                  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,789 |                   500 |          3,289 |         2427 |      31.9 |          19 |           17.40s |      1.96s |      19.67s | missing-sections(title+descrip...  |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  15,662 |            16,723 |                   500 |         17,223 |         1277 |      89.3 |         8.3 |           19.55s |      0.71s |      20.54s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     264 |            16,725 |                   500 |         17,225 |         1157 |      87.5 |         8.3 |           21.11s |      0.76s |      22.19s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |             2,289 |                   500 |          2,789 |         3062 |      34.1 |          18 |           21.43s |      1.74s |      23.48s | repetitive(phrase: "rencontre...   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |       1 |             6,524 |                   500 |          7,024 |          522 |      37.5 |          78 |           26.29s |      5.47s |      32.08s | missing-sections(title+descrip...  |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |   8,923 |            16,734 |                   500 |         17,234 |          956 |      54.6 |          13 |           27.45s |      1.16s |      28.91s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |      11 |               477 |                   500 |            977 |          286 |      19.7 |          15 |           27.47s |      1.51s |      29.28s | repetitive(phrase: "waterfront...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |      15 |             1,693 |                   500 |          2,193 |         89.8 |        53 |          41 |           29.07s |      1.19s |      30.57s | missing-sections(title+descrip...  |                 |
| `mlx-community/pixtral-12b-bf16`                        |   4,054 |             3,309 |                   500 |          3,809 |         2069 |        20 |          28 |           29.47s |      2.55s |      32.33s | degeneration, ...                  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               478 |                   159 |            637 |          263 |      5.12 |          25 |           33.33s |      2.25s |      35.89s | missing-sections(title+descrip...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |  26,238 |             1,693 |                   500 |          2,193 |         87.7 |      30.2 |          48 |           57.24s |      1.76s |      59.31s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,734 |                     4 |         16,738 |          287 |       256 |         5.1 |           59.07s |      0.49s |      59.87s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |   3,050 |            16,749 |                   500 |         17,249 |          306 |      86.6 |          35 |           61.11s |      3.17s |      64.61s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |     328 |            16,749 |                   500 |         17,249 |          289 |      64.9 |          76 |           66.81s |      7.68s |      74.81s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     424 |            16,749 |                   500 |         17,249 |          236 |      29.6 |          26 |           88.30s |      2.15s |      90.78s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |  10,151 |            16,749 |                   500 |         17,249 |          241 |      17.3 |          39 |           98.97s |      3.12s |     102.42s | refusal(explicit_refusal), ...     |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

**Review artifacts:**

- Standalone output gallery: [model_gallery.md](model_gallery.md)
- Automated review digest: [review.md](review.md)
- Canonical run log: [check_models.log](check_models.log)

---

## System/Hardware Information

- **OS**: Darwin 25.4.0
- **macOS Version**: 26.4
- **SDK Version**: 26.4
- **Xcode Version**: 26.4
- **Xcode Build**: 17E192
- **Metal SDK**: MacOSX.sdk
- **Python Version**: 3.13.12
- **Architecture**: arm64
- **GPU/Chip**: Apple M5 Max
- **GPU Cores**: 40
- **Metal Support**: Metal 4
- **RAM**: 128.0 GB
- **CPU Cores (Physical)**: 18
- **CPU Cores (Logical)**: 18

## Library Versions

- `numpy`: `2.4.3`
- `mlx`: `0.31.2.dev20260322+38ad2570`
- `mlx-vlm`: `0.4.1`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.8.0`
- `transformers`: `5.4.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.1.1`

_Report generated on: 2026-03-27 14:43:58 GMT_
