# Model Performance Results

_Generated on 2026-03-22 21:43:07 GMT_

## 🎯 Action Snapshot

- **Framework/runtime failures:** 8 (top owners: transformers=5, mlx-vlm=2, model-config=1).
- **Next action:** review failure ownership below and use diagnostics.md for filing.
- **Maintainer signals:** harness-risk successes=8, clean outputs=3/43.
- **Useful now:** 1 clean A/B model(s) worth first review.
- **Review watchlist:** 42 model(s) with breaking or lower-value output.
- **Preflight compatibility:** 1 informational warning(s); do not treat these alone as run failures.
- **Escalate only if:** they line up with unexpected TF/Flax/JAX imports, startup hangs, or backend/runtime crashes.
- **Vs existing metadata:** better=8, neutral=1, worse=34 (baseline A 80/100).
- **Quality signal frequency:** missing_sections=26, cutoff=20, trusted_hint_ignored=19, context_ignored=18, metadata_borrowing=13, title_length=8.
- **Runtime pattern:** decode dominates measured phase time (82%; 46/51 measured model(s)).
- **Phase totals:** model load=102.75s, prompt prep=0.12s, decode=500.08s, cleanup=6.14s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=43, exception=8.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (366.9 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.55s)
- **📊 Average TPS:** 88.5 across 43 models

## 📈 Resource Usage

- **Total peak memory:** 699.4 GB
- **Average peak memory:** 16.3 GB
- **Memory efficiency:** 210 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 9 | ✅ B: 7 | 🟡 C: 11 | 🟠 D: 3 | ❌ F: 13

**Average Utility Score:** 51/100

**Existing Metadata Baseline:** 🏆 A (80/100)
**Vs Existing Metadata:** Avg Δ -29 | Better: 8, Neutral: 1, Worse: 34

- **Best for cataloging:** `mlx-community/gemma-3n-E4B-it-bf16` (🏆 A, 95/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 0/100)

### ⚠️ 16 Models with Low Utility (D/F)

- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (25/100) - Lacks visual description of image
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: 🟠 D (47/100) - Mostly echoes context without adding value
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (26/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (40/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (17/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (17/100) - Output lacks detail
- `mlx-community/pixtral-12b-8bit`: ❌ F (32/100) - Mostly echoes context without adding value
- `mlx-community/pixtral-12b-bf16`: ❌ F (35/100) - Mostly echoes context without adding value
- `prince-canuma/Florence-2-large-ft`: ❌ F (0/100) - Output too short to be useful

## ⚠️ Quality Issues

- **❌ Failed Models (8):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
  - `mlx-community/InternVL3-8B-bf16` (`Model Error`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (`Model Error`)
  - `mlx-community/Qwen3.5-27B-4bit` (`Model Error`)
  - `mlx-community/Qwen3.5-27B-mxfp8` (`Model Error`)
  - `mlx-community/Qwen3.5-35B-A3B-6bit` (`Model Error`)
  - `mlx-community/Qwen3.5-35B-A3B-bf16` (`Model Error`)
  - `mlx-community/deepseek-vl2-8bit` (`Processor Error`)
- **🔄 Repetitive Output (8):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (token: `因此，所以可打印表图1.`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "use only details that..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "waterfront scene, waterfront a..."`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `phrase: "the image is of..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `qnguyen3/nanoLLaVA` (token: `phrase: "painting glasses, guitar paint..."`)
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`
  - `prince-canuma/Florence-2-large-ft`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 88.5 | Min: 0 | Max: 367
- **Peak Memory**: Avg: 16 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 13.26s | Min: 1.65s | Max: 63.23s
- **Generation Time**: Avg: 11.19s | Min: 0.82s | Max: 62.40s
- **Model Load Time**: Avg: 1.80s | Min: 0.55s | Max: 10.46s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (82%; 46/51 measured model(s)).
- **Phase totals:** model load=102.75s, prompt prep=0.12s, decode=500.08s, cleanup=6.14s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=43, exception=8.

### ⏱ Timing Snapshot

- **Validation overhead:** 13.41s total (avg 0.26s across 51 model(s)).
- **First-token latency:** Avg 2.97s | Min 0.09s | Max 16.19s across 42 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- **Best cataloging quality:** [`mlx-community/gemma-3n-E4B-it-bf16`](model_gallery.md#model-mlx-community-gemma-3n-e4b-it-bf16) (A 95/100 | Gen 48.3 TPS | Peak 17 | A 95/100 | Excessive verbosity; Missing sections (title, description, keywords))
- **Fastest generation:** [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit) (D 40/100 | Gen 367 TPS | Peak 2.7 | D 40/100 | Title length violation (11 words; expected 5-10); Description sentence violation (3; expected 1-2); ...)
- **Lowest memory footprint:** [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16) (D 46/100 | Gen 331 TPS | Peak 2.2 | D 46/100 | Context ignored (missing: Pedestrians, cross, footbridge, over, canal); Missing sections (title, description, keywords); ...)
- **Best balance:** [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit) (A 90/100 | Gen 183 TPS | Peak 7.8 | A 90/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery when available.

- **❌ Failed Models (8):** [`microsoft/Florence-2-large-ft`](model_gallery.md#model-microsoft-florence-2-large-ft), [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16), [`mlx-community/Molmo-7B-D-0924-bf16`](model_gallery.md#model-mlx-community-molmo-7b-d-0924-bf16), [`mlx-community/Qwen3.5-27B-4bit`](model_gallery.md#model-mlx-community-qwen35-27b-4bit), +4 more. Example: `Model Error`.
- **🔄 Repetitive Output (8):** [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct), [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit), [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit), [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](model_gallery.md#model-mlx-community-llama-32-11b-vision-instruct-8bit), +4 more. Example: token: `unt`.
- **📝 Formatting Issues (6):** [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit), [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4), [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4), [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16), +2 more.
- **Low-utility outputs (16):** [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit), [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct), [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit), [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16), +12 more. Common weakness: Lacks visual description of image.

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `transformers` | 5 | Model Error | `microsoft/Florence-2-large-ft`, `mlx-community/Qwen3.5-27B-4bit`, `mlx-community/Qwen3.5-27B-mxfp8`, `mlx-community/Qwen3.5-35B-A3B-6bit`, `mlx-community/Qwen3.5-35B-A3B-bf16` |
| `mlx-vlm` | 2 | Model Error | `mlx-community/InternVL3-8B-bf16`, `mlx-community/Molmo-7B-D-0924-bf16` |
| `model-config` | 1 | Processor Error | `mlx-community/deepseek-vl2-8bit` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### transformers

- **microsoft/Florence-2-large-ft** (Model Error)
  - Error: `Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate ...`
  - Type: `ValueError`
- **mlx-community/Qwen3.5-27B-4bit** (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen3.5-27B-4bit: Failed to process inputs with error: Only returning PyTor...`
  - Type: `ValueError`
- **mlx-community/Qwen3.5-27B-mxfp8** (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen3.5-27B-mxfp8: Failed to process inputs with error: Only returning PyTo...`
  - Type: `ValueError`
- **mlx-community/Qwen3.5-35B-A3B-6bit** (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen3.5-35B-A3B-6bit: Failed to process inputs with error: Only returning P...`
  - Type: `ValueError`
- **mlx-community/Qwen3.5-35B-A3B-bf16** (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen3.5-35B-A3B-bf16: Failed to process inputs with error: Only returning P...`
  - Type: `ValueError`

#### mlx-vlm

- **mlx-community/InternVL3-8B-bf16** (Model Error)
  - Error: `Model generation failed for mlx-community/InternVL3-8B-bf16: 'utf-8' codec can't decode byte 0xab in position 10: inv...`
  - Type: `ValueError`
- **mlx-community/Molmo-7B-D-0924-bf16** (Model Error)
  - Error: `Model generation failed for mlx-community/Molmo-7B-D-0924-bf16: 'utf-8' codec can't decode byte 0xa1 in position 0: i...`
  - Type: `ValueError`

#### model-config

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
> \- Description hint: Pedestrians cross a footbridge over a canal at dusk in a vibrant
> urban waterside area. A modern glass building reflects the golden light of the setting
> sun against a purple twilight sky, while people walk along the towpath, relax on the
> bank, and socialize at a nearby restaurant. Moored boats line the canal, completing the
> lively evening scene as people go about their daily lives, commuting or enjoying leisure
> time.
> \- Capture metadata: Taken on 2026-03-21 18:22:22 GMT (at 18:22:22 local time). GPS:
> 51.536500°N, 0.126500°W.
<!-- markdownlint-enable MD028 MD049 -->

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 623.38s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                          |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |            0.51s |      0.44s |       1.25s | fabrication, title-length(29), ...      |    transformers |
| `mlx-community/InternVL3-8B-bf16`                       |         |                   |                       |                |              |           |             |            1.85s |      1.70s |       3.81s | fabrication, title-length(29), ...      |         mlx-vlm |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |         |                   |                       |                |              |           |             |           15.55s |      1.68s |      17.49s | fabrication, title-length(29), ...      |         mlx-vlm |
| `mlx-community/Qwen3.5-27B-4bit`                        |         |                   |                       |                |              |           |             |            0.27s |      2.35s |       2.88s | ⚠️harness(stop_token), ...              |    transformers |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |         |                   |                       |                |              |           |             |            0.26s |      3.25s |       3.77s | ⚠️harness(stop_token), ...              |    transformers |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |         |                   |                       |                |              |           |             |            0.26s |      3.38s |       3.91s | ⚠️harness(stop_token), ...              |    transformers |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |         |                   |                       |                |              |           |             |            0.27s |      9.72s |      10.25s | ⚠️harness(stop_token), ...              |    transformers |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |      2.75s |       3.02s | missing-sections(title+descrip...       |    model-config |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               784 |                    93 |            877 |        7,548 |       326 |         2.8 |            0.82s |      0.58s |       1.65s | missing-sections(title+descrip...       |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |  151645 |               533 |                    28 |            561 |        5,162 |       331 |         2.2 |            1.05s |      0.60s |       1.91s | missing-sections(title+descrip...       |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |     624 |               529 |                   500 |           1029 |        5,986 |       367 |         2.7 |            1.88s |      0.55s |       2.69s | title-length(11), ...                   |                 |
| `prince-canuma/Florence-2-large-ft`                     |       0 |              1099 |                   500 |           1599 |        9,518 |       353 |         5.1 |            1.98s |      0.70s |       2.94s | ⚠️harness(stop_token), ...              |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |              1556 |                    17 |           1573 |        1,365 |      29.6 |          12 |            2.15s |      1.87s |       4.27s | missing-sections(title+descrip...       |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |              3123 |                   142 |           3265 |        2,984 |       183 |         7.8 |            2.29s |      0.94s |       3.50s |                                         |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |   32007 |              1350 |                   123 |           1473 |        3,811 |      55.4 |         9.5 |            2.95s |      0.85s |       4.06s | metadata-borrowing                      |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |              1556 |                    42 |           1598 |        3,263 |      19.1 |          11 |            3.11s |      1.72s |       5.10s | missing-sections(title+descrip...       |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |   12123 |               784 |                   500 |           1284 |        7,420 |       187 |         3.9 |            3.17s |      0.57s |       4.00s | title-length(15), ...                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |              1556 |                     9 |           1565 |        1,050 |      5.54 |          27 |            3.54s |      2.71s |       6.52s | ⚠️harness(prompt_template), ...         |                 |
| `mlx-community/InternVL3-14B-8bit`                      |  151645 |              2312 |                    64 |           2376 |        1,421 |      31.5 |          18 |            4.12s |      1.74s |       6.11s | title-length(2), ...                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               800 |                    87 |            887 |          654 |      31.1 |          19 |            4.42s |      2.55s |       7.24s |                                         |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |              3314 |                    97 |           3411 |        1,802 |      38.6 |          16 |            4.80s |      1.77s |       6.84s | title-length(4)                         |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |    1597 |              1736 |                   500 |           2236 |        4,328 |       128 |         5.5 |            4.82s |      0.66s |       5.72s | repetitive(unt), ...                    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |   49153 |               637 |                   500 |           1137 |        1,872 |       127 |         5.5 |            4.83s |      0.60s |       5.68s | missing-sections(title+descrip...       |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |  236761 |               791 |                   500 |           1291 |        2,489 |       121 |           6 |            4.84s |      1.69s |       6.80s | missing-sections(title+descrip...       |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |              3124 |                   130 |           3254 |        1,390 |      56.3 |          13 |            5.03s |      1.35s |       6.65s |                                         |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |    1597 |              1736 |                   500 |           2236 |        4,108 |       122 |         5.5 |            5.06s |      0.59s |       5.91s | repetitive(unt), ...                    |                 |
| `qnguyen3/nanoLLaVA`                                    |   98266 |               529 |                   500 |           1029 |        4,762 |       110 |         4.5 |            5.06s |      0.57s |       5.89s | repetitive(phrase: "painting g...       |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |              3124 |                   128 |           3252 |        1,380 |      54.4 |          13 |            5.08s |      1.33s |       6.68s | description-sentences(3)                |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |    1609 |              1522 |                   500 |           2022 |        1,793 |       114 |          22 |            5.76s |      2.12s |       8.15s | repetitive(phrase: "use only d...       |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |              2732 |                   182 |           2914 |        1,210 |      61.2 |         9.7 |            5.78s |      0.90s |       6.95s | missing-sections(title+descrip...       |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |      16 |              1522 |                   500 |           2022 |        1,702 |       110 |          18 |            5.93s |      2.07s |       8.37s | repetitive(因此，所以可打印表图1.), ... |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               800 |                    89 |            889 |          565 |      17.5 |          34 |            6.92s |      3.63s |      10.82s | trusted-hints-degraded                  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      13 |              1522 |                   500 |           2022 |        1,772 |      78.6 |          37 |            7.81s |      3.23s |      11.30s | degeneration, ...                       |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |              2626 |                    93 |           2719 |          755 |      23.6 |          22 |            7.88s |      2.10s |      10.26s | ⚠️harness(encoding), ...                |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               799 |                   355 |           1154 |        1,780 |      48.3 |          17 |            8.19s |      2.47s |      10.94s | missing-sections(title+descrip...       |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |      13 |              1350 |                   500 |           1850 |        3,796 |      52.9 |         9.5 |           10.21s |      0.96s |      11.48s | ⚠️harness(stop_token), ...              |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |    5966 |              4628 |                   500 |           5128 |        3,881 |      43.4 |         4.5 |           13.32s |      1.41s |      15.00s | repetitive(phrase: "- output o...       |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |    3884 |              6643 |                   500 |           7143 |        1,172 |      65.6 |         8.4 |           13.85s |      1.28s |      15.39s | missing-sections(title+descrip...       |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |       2 |              1845 |                   464 |           2309 |          309 |      58.6 |          60 |           14.67s |     10.46s |      25.41s | ⚠️harness(stop_token), ...              |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |      12 |              6643 |                   500 |           7143 |        1,192 |      53.4 |          11 |           15.49s |      1.38s |      17.14s | missing-sections(title+descrip...       |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |  128256 |              2811 |                   500 |           3311 |        2,423 |      31.6 |          19 |           17.50s |      1.90s |      19.68s | missing-sections(title+descrip...       |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |    1044 |              3405 |                   500 |           3905 |        1,410 |      30.5 |          15 |           19.29s |      1.66s |      21.25s | missing-sections(title+keyword...       |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |   40208 |             16746 |                   500 |          17246 |        1,240 |      86.5 |         8.3 |           20.06s |      0.74s |      21.06s | title-length(50), ...                   |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |   11030 |             16748 |                   500 |          17248 |        1,260 |      68.5 |         8.3 |           21.44s |      0.81s |      22.51s | ⚠️harness(long_context), ...            |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |  151643 |              1708 |                   380 |           2088 |          128 |      51.3 |          41 |           21.54s |      1.16s |      22.96s | title-length(15), ...                   |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |  128009 |               501 |                   100 |            601 |          258 |      5.07 |          25 |           22.14s |      2.25s |      24.68s | missing-sections(title+descrip...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |    7217 |             16757 |                   500 |          17257 |        1,035 |      55.8 |          13 |           25.91s |      1.20s |      27.38s | ⚠️harness(long_context), ...            |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |      11 |               500 |                   500 |           1000 |          308 |      20.3 |          15 |           26.65s |      1.46s |      28.39s | repetitive(phrase: "waterfront...       |                 |
| `mlx-community/pixtral-12b-bf16`                        |   20760 |              3314 |                   500 |           3814 |        1,887 |      19.7 |          28 |           27.61s |      2.51s |      30.39s | missing-sections(title+descrip...       |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |      11 |              6643 |                   500 |           7143 |          456 |      34.5 |          78 |           29.75s |      7.29s |      37.33s | missing-sections(title+descrip...       |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                 0 |                     0 |              0 |            0 |         0 |         5.1 |           62.40s |      0.57s |      63.23s | ⚠️harness(prompt_template), ...         |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

**Review artifacts:**

- Standalone output gallery: [model_gallery.md](model_gallery.md)
- Automated review digest: [review.md](review.md)
- Canonical run log: [check_models.log](check_models.log)

---

## System/Hardware Information

- **OS**: Darwin 25.3.0
- **macOS Version**: 26.3.1
- **SDK Version**: 26.2
- **Xcode Version**: 26.3
- **Xcode Build**: 17C529
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
- `huggingface-hub`: `1.7.2`
- `transformers`: `5.3.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.1.1`

_Report generated on: 2026-03-22 21:43:07 GMT_
