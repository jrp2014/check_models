# Model Performance Results

_Generated on 2026-03-22 02:17:59 GMT_

## 🎯 Action Snapshot

- **Framework/runtime failures:** 8 (top owners: transformers=5, mlx-vlm=2, model-config=1).
- **Next action:** review failure ownership below and use diagnostics.md for filing.
- **Maintainer signals:** harness-risk successes=8, clean outputs=3/43.
- **Useful now:** 1 clean A/B model(s) worth first review.
- **Review watchlist:** 42 model(s) with breaking or lower-value output.
- **Preflight compatibility:** 1 informational warning(s); do not treat these alone as run failures.
- **Escalate only if:** they line up with unexpected TF/Flax/JAX imports, startup hangs, or backend/runtime crashes.
- **Vs existing metadata:** better=7, neutral=1, worse=35 (baseline A 80/100).
- **Quality signal frequency:** missing_sections=27, cutoff=21, trusted_hint_ignored=19, context_ignored=18, metadata_borrowing=12, repetitive=9.
- **Runtime pattern:** decode dominates measured phase time (83%; 45/51 measured model(s)).
- **Phase totals:** model load=98.63s, prompt prep=0.12s, decode=535.10s, cleanup=7.26s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=43, exception=8.

## 🏆 Performance Highlights

- **Fastest:** `prince-canuma/Florence-2-large-ft` (357.3 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/LFM2-VL-1.6B-8bit` (0.53s)
- **📊 Average TPS:** 88.1 across 43 models

## 📈 Resource Usage

- **Total peak memory:** 699.4 GB
- **Average peak memory:** 16.3 GB
- **Memory efficiency:** 210 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 8 | ✅ B: 8 | 🟡 C: 11 | 🟠 D: 3 | ❌ F: 13

**Average Utility Score:** 50/100

**Existing Metadata Baseline:** 🏆 A (80/100)
**Vs Existing Metadata:** Avg Δ -30 | Better: 7, Neutral: 1, Worse: 35

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
- **🔄 Repetitive Output (9):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (token: `因此，所以可打印表图1.`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "use only details that..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "waterfront scene, waterfront a..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "waterway, water body, water..."`)
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

- **Generation Tps**: Avg: 88.1 | Min: 0 | Max: 357
- **Peak Memory**: Avg: 16 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 13.85s | Min: 1.62s | Max: 66.27s
- **Generation Time**: Avg: 11.91s | Min: 0.82s | Max: 65.41s
- **Model Load Time**: Avg: 1.66s | Min: 0.53s | Max: 5.55s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (83%; 45/51 measured model(s)).
- **Phase totals:** model load=98.63s, prompt prep=0.12s, decode=535.10s, cleanup=7.26s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=43, exception=8.

### ⏱ Timing Snapshot

- **Validation overhead:** 13.91s total (avg 0.27s across 51 model(s)).
- **First-token latency:** Avg 3.28s | Min 0.09s | Max 18.54s across 42 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- **Best cataloging quality:** [`mlx-community/gemma-3n-E4B-it-bf16`](model_gallery.md#model-mlx-community-gemma-3n-e4b-it-bf16) (A 95/100 | Gen 48.0 TPS | Peak 17 | A 95/100 | Excessive verbosity; Missing sections (title, description, keywords))
- **Fastest generation:** [`prince-canuma/Florence-2-large-ft`](model_gallery.md#model-prince-canuma-florence-2-large-ft) (F 0/100 | Gen 357 TPS | Peak 5.1 | F 0/100 | ⚠️HARNESS:stop_token; Context ignored (missing: Pedestrians, cross, footbridge, over, canal); ...)
- **Lowest memory footprint:** [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16) (D 46/100 | Gen 329 TPS | Peak 2.1 | D 46/100 | Context ignored (missing: Pedestrians, cross, footbridge, over, canal); Missing sections (title, description, keywords); ...)
- **Best balance:** [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit) (A 90/100 | Gen 175 TPS | Peak 7.8 | A 90/100 | No quality issues detected)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery when available.

- **❌ Failed Models (8):** [`microsoft/Florence-2-large-ft`](model_gallery.md#model-microsoft-florence-2-large-ft), [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16), [`mlx-community/Molmo-7B-D-0924-bf16`](model_gallery.md#model-mlx-community-molmo-7b-d-0924-bf16), [`mlx-community/Qwen3.5-27B-4bit`](model_gallery.md#model-mlx-community-qwen35-27b-4bit), +4 more. Example: `Model Error`.
- **🔄 Repetitive Output (9):** [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct), [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit), [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit), [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](model_gallery.md#model-mlx-community-llama-32-11b-vision-instruct-8bit), +5 more. Example: token: `unt`.
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

**Overall runtime:** 656.08s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                          |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |            0.43s |      0.44s |       1.14s | fabrication, title-length(29), ...      |    transformers |
| `mlx-community/InternVL3-8B-bf16`                       |         |                   |                       |                |              |           |             |            2.08s |      1.70s |       4.13s | fabrication, title-length(29), ...      |         mlx-vlm |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |         |                   |                       |                |              |           |             |           19.34s |      1.69s |      21.29s | fabrication, title-length(29), ...      |         mlx-vlm |
| `mlx-community/Qwen3.5-27B-4bit`                        |         |                   |                       |                |              |           |             |            0.26s |      2.37s |       2.95s | ⚠️harness(stop_token), ...              |    transformers |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |         |                   |                       |                |              |           |             |            0.26s |      3.35s |       3.87s | ⚠️harness(stop_token), ...              |    transformers |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |         |                   |                       |                |              |           |             |            0.26s |      3.71s |       4.23s | ⚠️harness(stop_token), ...              |    transformers |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |         |                   |                       |                |              |           |             |            0.27s |     11.35s |      11.89s | ⚠️harness(stop_token), ...              |    transformers |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |      2.78s |       3.04s | missing-sections(title+descrip...       |    model-config |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               784 |                    93 |            877 |         7884 |       322 |         2.8 |            0.82s |      0.53s |       1.62s | missing-sections(title+descrip...       |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |  151645 |               533 |                    28 |            561 |         5419 |       329 |         2.1 |            0.95s |      0.60s |       1.95s | missing-sections(title+descrip...       |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |     624 |               529 |                   500 |           1029 |         5586 |       352 |         2.5 |            1.94s |      0.57s |       2.77s | title-length(11), ...                   |                 |
| `prince-canuma/Florence-2-large-ft`                     |       0 |              1099 |                   500 |           1599 |         9384 |       357 |         5.1 |            1.97s |      0.67s |       2.90s | ⚠️harness(stop_token), ...              |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |              1556 |                    17 |           1573 |         1364 |      30.8 |          12 |            2.12s |      1.98s |       4.37s | missing-sections(title+descrip...       |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |              3123 |                   142 |           3265 |         2829 |       175 |         7.8 |            2.38s |      0.89s |       3.54s |                                         |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |   32007 |              1350 |                   123 |           1473 |         3741 |      56.2 |         9.5 |            2.92s |      0.84s |       4.02s | metadata-borrowing                      |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |              1556 |                    42 |           1598 |         3236 |      19.4 |          11 |            3.07s |      1.73s |       5.07s | missing-sections(title+descrip...       |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |   12123 |               784 |                   500 |           1284 |         7227 |       188 |           4 |            3.18s |      0.59s |       4.04s | title-length(15), ...                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |              1556 |                     9 |           1565 |         1054 |      5.67 |          27 |            3.50s |      2.79s |       6.54s | ⚠️harness(prompt_template), ...         |                 |
| `mlx-community/InternVL3-14B-8bit`                      |  151645 |              2312 |                    64 |           2376 |         1246 |      30.9 |          18 |            4.39s |      1.75s |       6.62s | title-length(2), ...                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               800 |                    87 |            887 |          612 |        31 |          19 |            4.53s |      2.57s |       7.36s |                                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |   49153 |               637 |                   500 |           1137 |         1893 |       129 |         5.5 |            4.74s |      0.63s |       5.63s | missing-sections(title+descrip...       |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |              3314 |                    97 |           3411 |         1813 |      38.9 |          16 |            4.78s |      1.72s |       6.78s | title-length(4)                         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |  236761 |               791 |                   500 |           1291 |         2452 |       121 |           6 |            4.83s |      1.70s |       6.80s | missing-sections(title+descrip...       |                 |
| `qnguyen3/nanoLLaVA`                                    |   98266 |               529 |                   500 |           1029 |         4894 |       111 |         4.6 |            5.02s |      0.79s |       6.06s | repetitive(phrase: "painting g...       |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |    1597 |              1736 |                   500 |           2236 |         4068 |       122 |         5.5 |            5.05s |      0.59s |       5.91s | repetitive(unt), ...                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |              3124 |                   128 |           3252 |         1185 |      64.5 |          13 |            5.09s |      1.34s |       6.70s | description-sentences(3)                |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |    1597 |              1736 |                   500 |           2236 |         4038 |       121 |         5.5 |            5.10s |      0.63s |       5.97s | repetitive(unt), ...                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |              3124 |                   130 |           3254 |         1136 |      62.1 |          13 |            5.32s |      1.37s |       6.96s |                                         |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |      16 |              1522 |                   500 |           2022 |         1769 |       124 |          18 |            5.41s |      1.97s |       7.65s | repetitive(因此，所以可打印表图1.), ... |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |              2732 |                   182 |           2914 |         1133 |      61.5 |         9.7 |            5.93s |      0.89s |       7.08s | missing-sections(title+descrip...       |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |    1609 |              1522 |                   500 |           2022 |         1722 |       108 |          22 |            6.06s |      2.18s |       8.63s | repetitive(phrase: "use only d...       |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               800 |                    89 |            889 |          544 |      16.8 |          34 |            7.17s |      3.78s |      11.23s | trusted-hints-degraded                  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |              2626 |                    93 |           2719 |          763 |      21.7 |          22 |            8.19s |      2.27s |      10.74s | ⚠️harness(encoding), ...                |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               799 |                   355 |           1154 |         1707 |        48 |          17 |            8.24s |      2.52s |      11.03s | missing-sections(title+descrip...       |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      13 |              1522 |                   500 |           2022 |         1523 |      76.1 |          37 |            8.26s |      3.40s |      11.96s | degeneration, ...                       |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |      13 |              1350 |                   500 |           1850 |         3720 |      55.3 |         9.5 |            9.78s |      0.88s |      10.92s | ⚠️harness(stop_token), ...              |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |       2 |              1845 |                   464 |           2309 |         1030 |      52.4 |          60 |           11.35s |      4.80s |      16.42s | ⚠️harness(stop_token), ...              |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |    5966 |              4628 |                   500 |           5128 |         3933 |      40.2 |         4.5 |           14.25s |      1.43s |      15.95s | repetitive(phrase: "- output o...       |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |    3884 |              6643 |                   500 |           7143 |          881 |      63.8 |         8.4 |           15.93s |      1.32s |      17.54s | missing-sections(title+descrip...       |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |  128256 |              2811 |                   500 |           3311 |         2297 |      30.6 |          19 |           18.11s |      1.98s |      20.40s | missing-sections(title+descrip...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |      12 |              6643 |                   500 |           7143 |          891 |      45.9 |          11 |           18.90s |      1.44s |      20.61s | missing-sections(title+descrip...       |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |    1044 |              3405 |                   500 |           3905 |         1441 |      30.7 |          15 |           19.11s |      1.64s |      21.01s | missing-sections(title+keyword...       |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |   40208 |             16746 |                   500 |          17246 |         1230 |      86.8 |         8.3 |           20.21s |      0.76s |      21.24s | title-length(50), ...                   |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |  128009 |               501 |                   100 |            601 |          264 |      4.71 |          25 |           23.55s |      2.23s |      26.06s | missing-sections(title+descrip...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |   11030 |             16748 |                   500 |          17248 |          947 |      77.8 |         8.3 |           24.97s |      0.81s |      26.05s | ⚠️harness(long_context), ...            |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |    7217 |             16757 |                   500 |          17257 |          970 |      56.4 |          13 |           26.89s |      1.22s |      28.36s | ⚠️harness(long_context), ...            |                 |
| `mlx-community/pixtral-12b-bf16`                        |   20760 |              3314 |                   500 |           3814 |         1908 |      20.1 |          28 |           27.05s |      2.53s |      29.84s | missing-sections(title+descrip...       |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |    3015 |              1708 |                   500 |           2208 |         92.1 |      51.1 |          41 |           29.06s |      1.16s |      30.48s | repetitive(phrase: "waterway, ...       |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |      11 |              6643 |                   500 |           7143 |          424 |      31.7 |          78 |           32.01s |      5.55s |      37.84s | missing-sections(title+descrip...       |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |      11 |               500 |                   500 |           1000 |          287 |      15.4 |          15 |           34.68s |      1.60s |      36.54s | repetitive(phrase: "waterfront...       |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                 0 |                     0 |              0 |            0 |         0 |         5.1 |           65.41s |      0.60s |      66.27s | ⚠️harness(prompt_template), ...         |                 |

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

_Report generated on: 2026-03-22 02:17:59 GMT_
