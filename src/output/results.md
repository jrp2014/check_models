# Model Performance Results

_Generated on 2026-04-17 13:13:03 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 5 (top owners: mlx-vlm=4, model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, clean outputs=0/48.
- _Useful now:_ none (no clean A/B shortlist for this run).
- _Review watchlist:_ 48 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=42, neutral=0, worse=6 (baseline F 16/100).
- _Quality signal frequency:_ missing_sections=37, cutoff=31,
  context_ignored=28, trusted_hint_ignored=28, metadata_borrowing=18,
  repetitive=12.
- _Runtime pattern:_ decode dominates measured phase time (90%; 51/53 measured
  model(s)).
- _Phase totals:_ model load=115.13s, prompt prep=0.17s, decode=1064.80s,
  cleanup=5.19s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=48, exception=5.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (350.8 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (1.8 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.52s)
- **📊 Average TPS:** 78.5 across 48 models

## 📈 Resource Usage

- **Total peak memory:** 1021.4 GB
- **Average peak memory:** 21.3 GB
- **Memory efficiency:** 197 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 10 | ✅ B: 16 | 🟡 C: 6 | 🟠 D: 7 | ❌ F: 9

**Average Utility Score:** 60/100

**Existing Metadata Baseline:** ❌ F (16/100)
**Vs Existing Metadata:** Avg Δ +44 | Better: 42, Neutral: 0, Worse: 6

- **Best for cataloging:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (🏆 A, 100/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (93/100)
- **Worst for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (❌ F, 0/100)

### ⚠️ 16 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-bf16`: 🟠 D (50/100) - Lacks visual description of image
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (2/100) - Output too short to be useful
- `mlx-community/gemma-4-31b-bf16`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/nanoLLaVA-1.5-4bit`: ❌ F (30/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (19/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) - Output lacks detail
- `qnguyen3/nanoLLaVA`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (5):**
  - `Qwen/Qwen3-VL-2B-Instruct` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`Model Error`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (`Model Error`)
  - `mlx-community/X-Reasoner-7B-8bit` (`Model Error`)
- **🔄 Repetitive Output (12):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (token: `phrase: "stone church with clock..."`)
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "creative visual art design,..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `prize.th`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "easily visible, easily visible..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "not visible. man's left..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "creative visual art design,..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/gemma-4-31b-bf16` (token: `phrase: "church clock face, church..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "- 100% real-time. -..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `qnguyen3/nanoLLaVA` (token: `Painting`)
- **👻 Hallucinations (2):**
  - `mlx-community/Qwen3.5-27B-mxfp8`
  - `mlx-community/Qwen3.5-35B-A3B-6bit`
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 78.5 | Min: 4.9 | Max: 351
- **Peak Memory**: Avg: 21 | Min: 1.8 | Max: 78
- **Total Time**: Avg: 24.35s | Min: 1.43s | Max: 109.36s
- **Generation Time**: Avg: 21.78s | Min: 0.63s | Max: 105.94s
- **Model Load Time**: Avg: 2.29s | Min: 0.52s | Max: 11.20s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 51/53 measured model(s)).
- **Phase totals:** model load=115.13s, prompt prep=0.17s, decode=1064.80s, cleanup=5.19s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=48, exception=5.

### ⏱ Timing Snapshot

- **Validation overhead:** 14.39s total (avg 0.27s across 53 model(s)).
- **First-token latency:** Avg 9.89s | Min 0.07s | Max 76.72s across 48 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 100/100 | Desc 93 | Keywords 93 | Gen 65.2 TPS | Peak 13 | A 100/100 |
  nontext prompt burden=87% | missing terms: Town, Centre, Alton, United,
  Kingdom)
- _Best descriptions:_ [`mlx-community/gemma-4-31b-it-4bit`](model_gallery.md#model-mlx-community-gemma-4-31b-it-4bit)
  (A 98/100 | Desc 100 | Keywords 84 | Gen 27.3 TPS | Peak 20 | A 98/100 |
  missing terms: Centre, United, Kingdom | nonvisual metadata reused)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 100/100 | Desc 93 | Keywords 93 | Gen 65.2 TPS | Peak 13 | A 100/100 |
  nontext prompt burden=87% | missing terms: Town, Centre, Alton, United,
  Kingdom)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (F 30/100 | Desc 52 | Keywords 0 | Gen 351 TPS | Peak 2.5 | F 30/100 |
  missing sections: keywords | nonvisual metadata reused)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (B 72/100 | Desc 77 | Keywords 0 | Gen 340 TPS | Peak 1.8 | B 72/100 |
  missing sections: title, description, keywords | missing terms: Town,
  Centre, Alton, United, Kingdom)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 100/100 | Desc 93 | Keywords 93 | Gen 65.2 TPS | Peak 13 | A 100/100 |
  nontext prompt burden=87% | missing terms: Town, Centre, Alton, United,
  Kingdom)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (5):_ [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](model_gallery.md#model-mlx-community-qwen3-vl-2b-thinking-bf16),
  +1 more. Example: `Model Error`.
- _🔄 Repetitive Output (12):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +8 more. Example: token: `unt`.
- _👻 Hallucinations (2):_ [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8),
  [`mlx-community/Qwen3.5-35B-A3B-6bit`](model_gallery.md#model-mlx-community-qwen35-35b-a3b-6bit).
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (16):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +12 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `mlx-vlm` | 4 | Model Error | `Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/X-Reasoner-7B-8bit` |
| `model-config` | 1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16` |

### Actionable Items by Package

#### mlx-vlm

- Qwen/Qwen3-VL-2B-Instruct (Model Error)
  - Error: `Model generation failed for Qwen/Qwen3-VL-2B-Instruct: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be...`
  - Type: `ValueError`
- mlx-community/Qwen2-VL-2B-Instruct-4bit (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen2-VL-2B-Instruct-4bit: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16...`
  - Type: `ValueError`
- mlx-community/Qwen3-VL-2B-Thinking-bf16 (Model Error)
  - Error: `Model generation failed for mlx-community/Qwen3-VL-2B-Thinking-bf16: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16...`
  - Type: `ValueError`
- mlx-community/X-Reasoner-7B-8bit (Model Error)
  - Error: `Model generation failed for mlx-community/X-Reasoner-7B-8bit: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16239) ca...`
  - Type: `ValueError`

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
> &#8203;Return exactly these three sections, and nothing else:
>
> &#8203;Title:
> &#45; 5-10 words, concrete and factual, limited to clearly visible content.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> &#8203;Description:
> &#45; 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> &#45; Output only the description text after the label.
>
> &#8203;Keywords:
> &#45; 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> &#45; Output only the keyword list after the label.
>
> &#8203;Rules:
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
> &#8203;confirmed):
> &#45; Description hint: , Town Centre, Alton, England, United Kingdom, UK
> &#45; Capture metadata: Taken on 2026-04-11 17:53:12 BST (at 17:53:12 local
> time). GPS: 51.145067°N, 0.980317°W.
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1200.62s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `Qwen/Qwen3-VL-2B-Instruct`                             |         |                   |                       |                |              |           |             |            8.93s |      0.70s |       9.91s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.24s |       2.51s |                                 |    model-config |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                   |                       |                |              |           |             |            0.77s |      0.53s |       1.58s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |         |                   |                       |                |              |           |             |            8.73s |      0.71s |       9.72s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/X-Reasoner-7B-8bit`                      |         |                   |                       |                |              |           |             |            0.89s |      1.14s |       2.30s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               458 |                    38 |            496 |        4,886 |       351 |         2.5 |            0.63s |      0.52s |       1.43s | missing-sections(keywords), ... |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               711 |                    49 |            760 |        2,050 |       323 |         2.9 |            0.85s |      0.54s |       1.65s | missing-sections(title+desc...  |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               462 |                    34 |            496 |        1,510 |       340 |         1.8 |            1.20s |      0.63s |       2.11s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,067 |                   131 |          3,198 |        2,912 |       180 |         7.8 |            2.25s |      0.92s |       3.45s | description-sentences(3), ...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,484 |                    23 |          1,507 |          975 |      32.8 |          12 |            2.67s |      1.55s |       4.51s | missing-sections(title+desc...  |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |   6,081 |               521 |                   500 |          1,021 |        7,905 |       179 |         3.8 |            3.23s |      0.54s |       4.04s | repetitive(phrase: "easily...   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,484 |                    12 |          1,496 |          876 |      5.72 |          27 |            4.25s |      2.51s |       7.04s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,241 |                    70 |          2,311 |        1,372 |      31.3 |          18 |            4.34s |      1.77s |       6.39s | title-length(2), ...            |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               728 |                    88 |            816 |          563 |      31.2 |          19 |            4.52s |      2.28s |       7.08s | metadata-borrowing              |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,668 |                   109 |          2,777 |        1,201 |      62.2 |         9.7 |            4.53s |      0.96s |       5.76s | fabrication, ...                |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,068 |                   109 |          3,177 |        1,257 |      62.2 |          13 |            4.66s |      1.38s |       6.32s | title-length(12), ...           |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               567 |                   500 |          1,067 |        1,690 |       131 |         5.5 |            4.70s |      0.61s |       5.58s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,068 |                   126 |          3,194 |        1,232 |      65.2 |          13 |            4.89s |      1.35s |       6.52s | trusted-hints-ignored, ...      |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               733 |                    87 |            820 |          572 |      27.3 |          20 |            4.90s |      3.11s |       8.30s | metadata-borrowing              |                 |
| `qnguyen3/nanoLLaVA`                                    |  54,043 |               458 |                   500 |            958 |        4,212 |       113 |         4.7 |            4.96s |      0.58s |       5.82s | repetitive(Painting), ...       |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,667 |                   500 |          2,167 |        4,021 |       123 |         5.5 |            5.03s |      0.76s |       6.06s | repetitive(unt), ...            |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,261 |                   106 |          3,367 |        1,601 |      39.1 |          16 |            5.21s |      1.71s |       7.20s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               719 |                   500 |          1,219 |          608 |       124 |           6 |            5.60s |      1.48s |       7.35s | fabrication, ...                |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   4,599 |             1,451 |                   500 |          1,951 |          986 |       128 |          18 |            5.88s |      2.04s |       8.20s | missing-sections(title+desc...  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,569 |                    77 |          2,646 |          724 |      31.5 |          22 |            6.46s |      2.06s |       8.80s | ⚠️harness(encoding), ...        |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,667 |                   500 |          2,167 |          856 |       122 |         5.5 |            6.55s |      0.71s |       7.52s | repetitive(unt), ...            |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               429 |                    90 |            519 |          224 |      18.6 |          15 |            7.16s |      1.57s |       9.00s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               728 |                    99 |            827 |          543 |      17.8 |          33 |            7.32s |      3.41s |      11.00s | metadata-borrowing              |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               727 |                   328 |          1,055 |        1,702 |      48.7 |          17 |            7.56s |      2.22s |      10.06s | missing-sections(title+desc...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     417 |             1,451 |                   500 |          1,951 |        1,727 |       114 |          23 |            7.77s |      2.17s |      10.22s | repetitive(prize.th), ...       |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      15 |             1,451 |                   500 |          1,951 |        1,669 |      77.9 |          37 |            7.89s |      3.30s |      11.48s | degeneration, ...               |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |     907 |             1,265 |                   500 |          1,765 |        3,888 |      55.7 |         9.4 |            9.69s |      0.86s |      10.82s | repetitive(phrase: "creativ...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |     907 |             1,265 |                   500 |          1,765 |        1,723 |      55.4 |         9.4 |           10.15s |      0.93s |      11.36s | repetitive(phrase: "creativ...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   5,966 |             4,556 |                   500 |          5,056 |        3,654 |        45 |         4.6 |           13.04s |      1.16s |      14.50s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,261 |             3,352 |                   500 |          3,852 |        1,425 |      41.8 |          15 |           14.75s |      1.61s |      16.65s | degeneration, ...               |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |  19,998 |             6,574 |                   500 |          7,074 |          923 |      67.9 |         8.4 |           14.85s |      1.29s |      16.42s | missing-sections(title+desc...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,741 |                   500 |          3,241 |        2,399 |      31.7 |          19 |           17.46s |      1.91s |      19.65s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  96,300 |             2,241 |                   500 |          2,741 |        2,938 |      33.3 |          18 |           17.64s |      1.73s |      19.65s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  26,061 |             6,574 |                   500 |          7,074 |          926 |      48.3 |          11 |           17.83s |      1.39s |      19.50s | missing-sections(title+desc...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  93,937 |             1,769 |                   500 |          2,269 |          252 |      42.4 |          60 |           19.50s |     10.22s |      30.01s | fabrication, ...                |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |      13 |             1,645 |                   500 |          2,145 |          117 |      51.3 |          41 |           24.53s |      1.18s |      25.99s | repetitive(phrase: "not vis...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | 235,276 |             1,484 |                   500 |          1,984 |        3,258 |      19.2 |          11 |           27.02s |      1.47s |      28.76s | repetitive(phrase: "- 100%...   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |     374 |             6,574 |                   500 |          7,074 |          359 |      34.6 |          78 |           33.27s |      8.74s |      42.30s | degeneration, ...               |                 |
| `mlx-community/pixtral-12b-bf16`                        |   1,278 |             3,261 |                   500 |          3,761 |        1,984 |      19.5 |          28 |           40.63s |      2.56s |      43.46s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |  84,025 |             1,645 |                   500 |          2,145 |          114 |      29.6 |          48 |           48.22s |      1.74s |      50.24s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |     524 |            16,700 |                   500 |         17,200 |          328 |       106 |          26 |           56.47s |      2.53s |      59.28s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |     279 |            16,700 |                   500 |         17,200 |          325 |      91.3 |          12 |           57.75s |      1.39s |      59.42s | fabrication, ...                |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |      12 |            16,700 |                   500 |         17,200 |          324 |      89.4 |          35 |           58.00s |      3.16s |      61.43s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |  13,570 |            16,700 |                   500 |         17,200 |          308 |      65.4 |          76 |           62.78s |     11.20s |      74.26s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |  11,072 |               721 |                   500 |          1,221 |          238 |      7.16 |          64 |           73.33s |      6.55s |      80.17s | repetitive(phrase: "church...   |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |       9 |            16,700 |                   500 |         17,200 |          224 |      28.6 |          26 |           92.86s |      2.17s |      95.33s | refusal(explicit_refusal), ...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |   9,042 |               430 |                   500 |            930 |          184 |       4.9 |          25 |          104.74s |      2.22s |     107.24s | repetitive(phrase: "stone c...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |     348 |            16,700 |                   500 |         17,200 |          218 |      17.6 |          39 |          105.94s |      3.12s |     109.36s | refusal(explicit_refusal), ...  |                 |

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
- `mlx`: `0.31.2.dev20260417+d142de6a`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.11.0`
- `transformers`: `5.5.4`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-17 13:13:03 BST_
