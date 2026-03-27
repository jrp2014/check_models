# Model Performance Results

_Generated on 2026-03-27 13:06:07 GMT_

## 🎯 Action Snapshot

- **Framework/runtime failures:** 11 (top owners: transformers=7, mlx-vlm=2, model-config=2).
- **Next action:** review failure ownership below and use diagnostics.md for filing.
- **Maintainer signals:** harness-risk successes=8, clean outputs=3/41.
- **Useful now:** 1 clean A/B model(s) worth first review.
- **Review watchlist:** 40 model(s) with breaking or lower-value output.
- **Preflight compatibility:** 1 informational warning(s); do not treat these alone as run failures.
- **Escalate only if:** they line up with unexpected TF/Flax/JAX imports, startup hangs, or backend/runtime crashes.
- **Vs existing metadata:** better=11, neutral=0, worse=30 (baseline B 79/100).
- **Quality signal frequency:** missing_sections=26, cutoff=26, trusted_hint_ignored=18, context_ignored=17, repetitive=11, description_length=8.
- **Runtime pattern:** decode dominates measured phase time (89%; 41/52 measured model(s)).
- **Phase totals:** model load=98.18s, prompt prep=0.12s, decode=858.44s, cleanup=4.76s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=41, exception=11.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (353.6 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (2.7 GB)
- **⚡ Fastest load:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (0.52s)
- **📊 Average TPS:** 69.9 across 41 models

## 📈 Resource Usage

- **Total peak memory:** 872.9 GB
- **Average peak memory:** 21.3 GB
- **Memory efficiency:** 235 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 11 | ✅ B: 9 | 🟡 C: 5 | 🟠 D: 8 | ❌ F: 8

**Average Utility Score:** 56/100

**Existing Metadata Baseline:** ✅ B (79/100)
**Vs Existing Metadata:** Avg Δ -23 | Better: 11, Neutral: 0, Worse: 30

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
- `mlx-community/Molmo-7B-D-0924-8bit`: 🟠 D (41/100) - Lacks visual description of image
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (36/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) - Empty or minimal output
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
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "minute */ minute */..."`)
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

- **Generation Tps**: Avg: 69.9 | Min: 0 | Max: 354
- **Peak Memory**: Avg: 21 | Min: 2.7 | Max: 78
- **Total Time**: Avg: 23.24s | Min: 1.68s | Max: 103.24s
- **Generation Time**: Avg: 20.89s | Min: 0.84s | Max: 99.82s
- **Model Load Time**: Avg: 2.04s | Min: 0.52s | Max: 8.79s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (89%; 41/52 measured model(s)).
- **Phase totals:** model load=98.18s, prompt prep=0.12s, decode=858.44s, cleanup=4.76s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=41, exception=11.

### ⏱ Timing Snapshot

- **Validation overhead:** 15.74s total (avg 0.30s across 52 model(s)).
- **First-token latency:** Avg 10.04s | Min 0.12s | Max 71.24s across 40 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- **Best cataloging quality:** [`mlx-community/X-Reasoner-7B-8bit`](model_gallery.md#model-mlx-community-x-reasoner-7b-8bit) (A 98/100 | Gen 56.6 TPS | Peak 13 | A 98/100 | ⚠️REVIEW:cutoff; ⚠️HARNESS:long_context; ...)
- **Fastest generation:** [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit) (F 28/100 | Gen 354 TPS | Peak 2.7 | F 28/100 | Missing sections (keywords); Title length violation (11 words; expected 5-10); ...)
- **Lowest memory footprint:** [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit) (F 28/100 | Gen 354 TPS | Peak 2.7 | F 28/100 | Missing sections (keywords); Title length violation (11 words; expected 5-10); ...)
- **Best balance:** [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4) (B 69/100 | Gen 63.0 TPS | Peak 13 | B 69/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery when available.

- **❌ Failed Models (11):** [`microsoft/Florence-2-large-ft`](model_gallery.md#model-microsoft-florence-2-large-ft), [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16), [`mlx-community/LFM2-VL-1.6B-8bit`](model_gallery.md#model-mlx-community-lfm2-vl-16b-8bit), [`mlx-community/LFM2.5-VL-1.6B-bf16`](model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16), +7 more. Example: `Model Error`.
- **🔄 Repetitive Output (11):** [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct), [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct), [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16), [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit), +7 more. Example: token: `unt`.
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

**Overall runtime:** 978.27s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |      0.35s |       0.65s |                                    |    transformers |
| `mlx-community/FastVLM-0.5B-bf16`                       |         |                   |                       |                |              |           |             |                  |      0.32s |       0.62s |                                    |    transformers |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |         |                   |                       |                |              |           |             |            0.40s |      0.50s |       1.22s | ⚠️harness(stop_token), ...         |         mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |         |                   |                       |                |              |           |             |            0.43s |      0.53s |       1.27s | ⚠️harness(stop_token), ...         |         mlx-vlm |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.22s |       2.53s |                                    |    model-config |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |      2.78s |       3.09s | missing-sections(title+descrip...  |    model-config |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |         |                   |                       |                |              |           |             |            0.30s |      1.64s |       2.25s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |         |                   |                       |                |              |           |             |            0.30s |      2.47s |       3.06s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |         |                   |                       |                |              |           |             |            0.30s |      1.46s |       2.06s | fabrication, title-length(29), ... |    transformers |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |         |                   |                       |                |              |           |             |            0.29s |      1.12s |       1.72s | fabrication, title-length(29), ... |    transformers |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |      1.30s |       1.61s |                                    |    transformers |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               506 |                    84 |            590 |         4068 |       354 |         2.7 |            0.84s |      0.52s |       1.68s | missing-sections(keywords), ...    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |              3115 |                   143 |           3258 |         2848 |       186 |         7.8 |            2.39s |      1.62s |       4.32s | description-sentences(3)           |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |              1326 |                   141 |           1467 |         3837 |      56.2 |         9.5 |            3.27s |      0.86s |       4.43s | description-sentences(3), ...      |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |              2726 |                     8 |           2734 |         1076 |      69.8 |         9.7 |            3.27s |      0.94s |       4.52s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |              3116 |                    97 |           3213 |         1392 |      66.3 |          13 |            4.25s |      1.38s |       5.95s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |              3116 |                   113 |           3229 |         1414 |        63 |          13 |            4.52s |      1.35s |       6.18s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |              2289 |                    71 |           2360 |         1175 |      32.2 |          18 |            4.69s |      1.82s |       6.84s | title-length(2), ...               |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               614 |                   500 |           1114 |         1851 |       130 |         5.5 |            4.78s |      0.62s |       5.70s | missing-sections(title+descrip...  |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               777 |                    94 |            871 |          489 |      31.2 |          19 |            5.07s |      2.32s |       7.71s | keyword-count(19)                  |                 |
| `qnguyen3/nanoLLaVA`                                    |  98,266 |               506 |                   500 |           1006 |         4302 |       111 |         4.5 |            5.10s |      0.55s |       5.95s | repetitive(phrase: "painting g...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |              1714 |                   500 |           2214 |         4126 |       123 |         5.5 |            5.11s |      0.62s |       6.05s | repetitive(unt), ...               |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |              3309 |                   122 |           3431 |         1532 |      38.7 |          16 |            5.82s |      1.64s |       7.77s | description-sentences(3), ...      |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,812 |               768 |                   500 |           1268 |          480 |       125 |           6 |            6.07s |      1.43s |       7.82s | repetitive(phrase: "10:18: 16:...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   1,162 |              1497 |                   500 |           1997 |          706 |       129 |          18 |            6.54s |      2.06s |       8.91s | degeneration, ...                  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |              1714 |                   500 |           2214 |          785 |       129 |         5.5 |            6.63s |      0.65s |       7.57s | repetitive(unt), ...               |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |  13,780 |              1497 |                   500 |           1997 |          792 |       113 |          22 |            6.86s |      2.18s |       9.34s | repetitive(phrase: "use only i...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               777 |                    92 |            869 |          552 |      17.6 |          34 |            7.09s |      3.41s |      10.81s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |              2618 |                   104 |           2722 |          679 |      31.7 |          22 |            7.69s |      2.19s |      10.21s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |  21,474 |              1497 |                   500 |           1997 |         1742 |      79.3 |          37 |            7.80s |      3.30s |      11.41s | missing-sections(title+descrip...  |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               776 |                   347 |           1123 |         1181 |      48.6 |          17 |            8.24s |      2.26s |      10.82s | description-sentences(6), ...      |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,892 |              1326 |                   500 |           1826 |         1573 |      56.7 |         9.5 |           10.08s |      0.87s |      11.26s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |     330 |              6524 |                   500 |           7024 |         1162 |      72.1 |         8.4 |           12.99s |      1.35s |      14.67s | missing-sections(title+descrip...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   8,223 |              3400 |                   500 |           3900 |         1405 |      41.7 |          15 |           14.90s |      1.64s |      16.85s | missing-sections(title+keyword...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |     304 |              6524 |                   500 |           7024 |         1069 |      54.8 |          11 |           15.66s |      1.37s |      17.34s | degeneration, ...                  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |   4,877 |              1822 |                   500 |           2322 |          540 |      42.5 |          60 |           15.90s |      6.28s |      22.50s | missing-sections(title+descrip...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |              2789 |                   500 |           3289 |         2442 |      31.9 |          19 |           17.41s |      1.92s |      19.64s | missing-sections(title+descrip...  |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  15,662 |             16723 |                   500 |          17223 |         1184 |      89.7 |         8.3 |           20.52s |      0.70s |      21.51s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     264 |             16725 |                   500 |          17225 |         1163 |      87.6 |         8.3 |           21.02s |      0.79s |      22.14s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |              2289 |                   500 |           2789 |         3056 |      33.9 |          18 |           21.57s |      1.75s |      23.62s | repetitive(phrase: "rencontre...   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |      11 |               477 |                   500 |            977 |          281 |      20.7 |          15 |           26.32s |      1.47s |      28.09s | repetitive(phrase: "waterfront...  |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |   8,923 |             16734 |                   500 |          17234 |          996 |      56.6 |          13 |           26.42s |      1.16s |      27.89s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |       1 |              6524 |                   500 |           7024 |          452 |      37.4 |          78 |           28.32s |      6.82s |      35.46s | missing-sections(title+descrip...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |      15 |              1693 |                   500 |           2193 |         89.1 |      52.8 |          41 |           29.26s |      1.30s |      30.87s | missing-sections(title+descrip...  |                 |
| `mlx-community/pixtral-12b-bf16`                        |   4,054 |              3309 |                   500 |           3809 |         1951 |      19.8 |          28 |           30.08s |      2.59s |      32.97s | degeneration, ...                  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               478 |                   159 |            637 |          218 |      5.07 |          25 |           33.98s |      2.22s |      36.51s | missing-sections(title+descrip...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |  53,751 |              1693 |                   500 |           2193 |         87.5 |      29.8 |          48 |           57.22s |      1.76s |      59.29s | repetitive(phrase: "minute */...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |   3,050 |             16749 |                   500 |          17249 |          327 |      86.9 |          35 |           57.41s |      3.17s |      60.89s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                 0 |                     0 |              0 |            0 |         0 |         5.1 |           59.92s |      0.52s |      60.74s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |     328 |             16749 |                   500 |          17249 |          309 |      64.4 |          76 |           62.63s |      8.79s |      71.73s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     424 |             16749 |                   500 |          17249 |          235 |      29.1 |          26 |           88.96s |      2.15s |      91.43s | refusal(explicit_refusal), ...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |  10,151 |             16749 |                   500 |          17249 |          235 |      17.8 |          39 |           99.82s |      3.10s |     103.24s | refusal(explicit_refusal), ...     |                 |

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

_Report generated on: 2026-03-27 13:06:07 GMT_
