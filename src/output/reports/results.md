# Model Performance Results

_Generated on 2026-04-18 00:45:49 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 6 (top owners: mlx-vlm=4, model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, clean outputs=0/48.
- _Useful now:_ 5 clean A/B model(s) worth first review.
- _Review watchlist:_ 43 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=42, neutral=1, worse=5 (baseline F 5/100).
- _Quality signal frequency:_ missing_sections=37, trusted_hint_ignored=31,
  cutoff=31, metadata_borrowing=24, repetitive=13, reasoning_leak=11.
- _Runtime pattern:_ decode dominates measured phase time (93%; 51/54 measured
  model(s)).
- _Phase totals:_ model load=127.00s, prompt prep=0.18s, decode=1751.41s,
  cleanup=7.96s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=48, exception=6.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/FastVLM-0.5B-bf16` (242.9 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (1.8 GB)
- **⚡ Fastest load:** `mlx-community/Phi-3.5-vision-instruct-bf16` (0.47s)
- **📊 Average TPS:** 40.5 across 48 models

## 📈 Resource Usage

- **Total peak memory:** 1021.9 GB
- **Average peak memory:** 21.3 GB
- **Memory efficiency:** 197 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 10 | ✅ B: 15 | 🟡 C: 8 | 🟠 D: 7 | ❌ F: 8

**Average Utility Score:** 60/100

**Existing Metadata Baseline:** ❌ F (5/100)
**Vs Existing Metadata:** Avg Δ +55 | Better: 42, Neutral: 1, Worse: 5

- **Best for cataloging:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (🏆 A, 100/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (93/100)
- **Worst for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (❌ F, 0/100)

### ⚠️ 15 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (2/100) - Output too short to be useful
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (42/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (19/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) - Output lacks detail
- `qnguyen3/nanoLLaVA`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (6):**
  - `Qwen/Qwen3-VL-2B-Instruct` (`Model Error`)
  - `ggml-org/gemma-3-1b-it-GGUF` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`Model Error`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (`Model Error`)
  - `mlx-community/X-Reasoner-7B-8bit` (`Model Error`)
- **🔄 Repetitive Output (13):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (token: `phrase: "stone church with clock..."`)
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "creative visual art design,..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `prize.th`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "easily visible, easily visible..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "uk - capture metadata:..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: ",… ,… ,… ,…..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "creative visual art design,..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/gemma-4-31b-bf16` (token: `phrase: "church clock face, church..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "- 100% real-time. -..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
  - `qnguyen3/nanoLLaVA` (token: `Painting`)
- **👻 Hallucinations (2):**
  - `mlx-community/Qwen3.5-27B-mxfp8`
  - `mlx-community/Qwen3.5-35B-A3B-6bit`
- **📝 Formatting Issues (5):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 40.5 | Min: 3.46 | Max: 243
- **Peak Memory**: Avg: 21 | Min: 1.8 | Max: 78
- **Total Time**: Avg: 38.81s | Min: 1.65s | Max: 207.80s
- **Generation Time**: Avg: 35.98s | Min: 0.86s | Max: 204.94s
- **Model Load Time**: Avg: 2.53s | Min: 0.47s | Max: 9.18s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (93%; 51/54 measured model(s)).
- **Phase totals:** model load=127.00s, prompt prep=0.18s, decode=1751.41s, cleanup=7.96s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=48, exception=6.

### ⏱ Timing Snapshot

- **Validation overhead:** 15.71s total (avg 0.29s across 54 model(s)).
- **First-token latency:** Avg 14.43s | Min 0.11s | Max 139.53s across 48 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 100/100 | Desc 93 | Keywords 93 | Gen 12.1 TPS | Peak 13 | A 100/100 |
  nontext prompt burden=87% | missing terms: Alton, United, Kingdom)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (A 94/100 | Desc 100 | Keywords 92 | Gen 48.5 TPS | Peak 13 | A 94/100 |
  nontext prompt burden=87% | missing terms: Alton, United, Kingdom)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 100/100 | Desc 93 | Keywords 93 | Gen 12.1 TPS | Peak 13 | A 100/100 |
  nontext prompt burden=87% | missing terms: Alton, United, Kingdom)
- _Fastest generation:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (B 72/100 | Desc 77 | Keywords 0 | Gen 243 TPS | Peak 1.8 | B 72/100 |
  missing sections: title, description, keywords | missing terms: Alton,
  United, Kingdom)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (B 72/100 | Desc 77 | Keywords 0 | Gen 243 TPS | Peak 1.8 | B 72/100 |
  missing sections: title, description, keywords | missing terms: Alton,
  United, Kingdom)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 100/100 | Desc 93 | Keywords 93 | Gen 12.1 TPS | Peak 13 | A 100/100 |
  nontext prompt burden=87% | missing terms: Alton, United, Kingdom)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (6):_ [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`ggml-org/gemma-3-1b-it-GGUF`](model_gallery.md#model-ggml-org-gemma-3-1b-it-gguf),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  +2 more. Example: `Model Error`.
- _🔄 Repetitive Output (13):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +9 more. Example: token: `unt`.
- _👻 Hallucinations (2):_ [`mlx-community/Qwen3.5-27B-mxfp8`](model_gallery.md#model-mlx-community-qwen35-27b-mxfp8),
  [`mlx-community/Qwen3.5-35B-A3B-6bit`](model_gallery.md#model-mlx-community-qwen35-35b-a3b-6bit).
- _📝 Formatting Issues (5):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +1 more.
- _Low-utility outputs (15):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +11 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `mlx-vlm` | 4 | Model Error | `Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/X-Reasoner-7B-8bit` |
| `model-config` | 2 | Model Error, Processor Error | `ggml-org/gemma-3-1b-it-GGUF`, `mlx-community/MolmoPoint-8B-fp16` |

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

- ggml-org/gemma-3-1b-it-GGUF (Model Error)
  - Error: `Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snap...`
  - Type: `ValueError`
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

_Overall runtime:_ 1903.41s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `Qwen/Qwen3-VL-2B-Instruct`                             |         |                   |                       |                |              |           |             |           11.20s |      0.82s |      12.33s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `ggml-org/gemma-3-1b-it-GGUF`                           |         |                   |                       |                |              |           |             |                  |      0.15s |       0.45s |                                 |    model-config |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.28s |       2.56s |                                 |    model-config |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                   |                       |                |              |           |             |            0.95s |      0.61s |       1.87s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |         |                   |                       |                |              |           |             |           10.98s |      0.64s |      11.94s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/X-Reasoner-7B-8bit`                      |         |                   |                       |                |              |           |             |            1.01s |      1.23s |       2.54s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               458 |                    38 |            496 |        2,046 |       189 |         2.3 |            0.86s |      0.51s |       1.65s | missing-sections(keywords), ... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               462 |                    34 |            496 |          960 |       243 |         1.8 |            2.12s |      1.95s |       4.40s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,484 |                    23 |          1,507 |          815 |      31.6 |          12 |            3.00s |      1.70s |       4.98s | missing-sections(title+desc...  |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               711 |                    49 |            760 |          610 |      20.9 |         2.9 |            3.91s |      0.53s |       4.73s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,484 |                    12 |          1,496 |          909 |      5.71 |          27 |            4.20s |      2.54s |       7.01s | ⚠️harness(prompt_template), ... |                 |
| `qnguyen3/nanoLLaVA`                                    |  54,043 |               458 |                   500 |            958 |        4,283 |       111 |         4.8 |            5.04s |      0.55s |       5.87s | repetitive(Painting), ...       |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,261 |                   106 |          3,367 |        1,703 |      39.1 |          16 |            5.07s |      1.85s |       7.20s | description-sentences(3), ...   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,668 |                   109 |          2,777 |          969 |      61.7 |         9.6 |            5.16s |      1.01s |       6.46s | fabrication, ...                |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,667 |                   500 |          2,167 |        3,937 |      98.4 |         5.5 |            6.07s |      0.65s |       7.00s | repetitive(unt), ...            |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               728 |                    88 |            816 |          275 |      27.9 |          19 |            6.42s |      5.48s |      12.20s | metadata-borrowing              |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,241 |                    70 |          2,311 |          930 |      19.5 |          18 |            6.51s |      1.95s |       8.77s | title-length(2), ...            |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               733 |                    87 |            820 |          500 |      15.9 |          20 |            7.36s |      2.87s |      10.58s | metadata-borrowing              |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               567 |                   500 |          1,067 |        1,204 |      79.3 |         5.5 |            7.39s |      0.63s |       8.32s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,068 |                   109 |          3,177 |          550 |      48.5 |          13 |            8.30s |      1.53s |      10.11s | title-length(12), ...           |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               429 |                    90 |            519 |          203 |      14.3 |          15 |            8.91s |      1.94s |      11.15s | missing-sections(title+desc...  |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |   6,081 |               521 |                   500 |          1,021 |          800 |        54 |         3.8 |           10.26s |      0.56s |      11.10s | repetitive(phrase: "easily...   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     417 |             1,451 |                   500 |          1,951 |        1,718 |      76.8 |          23 |           10.94s |      2.43s |      13.65s | repetitive(prize.th), ...       |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   5,966 |             4,556 |                   500 |          5,056 |        3,414 |      49.3 |         4.6 |           12.14s |      1.22s |      13.63s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,569 |                    77 |          2,646 |          314 |      21.4 |          22 |           12.28s |      2.61s |      15.22s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      15 |             1,451 |                   500 |          1,951 |          953 |      49.2 |          37 |           12.41s |      3.69s |      16.38s | degeneration, ...               |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |     907 |             1,265 |                   500 |          1,765 |        1,483 |      44.2 |         9.4 |           12.56s |      1.11s |      14.00s | repetitive(phrase: "creativ...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,067 |                   131 |          3,198 |          645 |      16.4 |         7.8 |           13.24s |      1.10s |      14.63s | description-sentences(3), ...   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,068 |                   126 |          3,194 |        1,274 |      12.1 |          13 |           13.34s |      1.65s |      15.30s | trusted-hints-ignored           |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               727 |                   328 |          1,055 |          605 |      22.8 |          18 |           16.02s |      2.77s |      19.10s | missing-sections(title+desc...  |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |     907 |             1,265 |                   500 |          1,765 |        1,202 |      33.4 |         9.4 |           16.44s |      0.47s |      17.20s | repetitive(phrase: "creativ...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |  19,998 |             6,574 |                   500 |          7,074 |          947 |      49.2 |         8.4 |           17.55s |      1.86s |      19.74s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,261 |             3,352 |                   500 |          3,852 |        1,221 |      33.1 |          15 |           18.37s |      1.98s |      20.66s | degeneration, ...               |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,667 |                   500 |          2,167 |          555 |      30.7 |         5.5 |           19.91s |      0.67s |      20.87s | repetitive(unt), ...            |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  96,300 |             2,241 |                   500 |          2,741 |        2,632 |      28.4 |          18 |           20.55s |      1.96s |      22.82s | missing-sections(title+desc...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,741 |                   500 |          3,241 |        1,632 |        26 |          19 |           21.45s |      2.13s |      23.87s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               728 |                    99 |            827 |          134 |      6.06 |          33 |           22.28s |      4.99s |      27.56s | metadata-borrowing              |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  93,937 |             1,769 |                   500 |          2,269 |          216 |        29 |          60 |           26.36s |      8.89s |      35.56s | fabrication, ...                |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | 235,276 |             1,484 |                   500 |          1,984 |        3,411 |      19.2 |          11 |           26.90s |      1.50s |      28.68s | repetitive(phrase: "- 100%...   |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  26,061 |             6,574 |                   500 |          7,074 |          726 |      22.9 |          11 |           31.30s |      2.46s |      34.07s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,770 |               719 |                   500 |          1,219 |          341 |      17.2 |           6 |           31.59s |      1.96s |      33.85s | fabrication, ...                |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   4,599 |             1,451 |                   500 |          1,951 |          339 |      17.2 |          18 |           34.00s |      2.59s |      36.88s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |      11 |             1,645 |                   500 |          2,145 |          114 |      26.7 |          41 |           34.05s |      1.47s |      35.83s | repetitive(phrase: "uk - ca...  |                 |
| `mlx-community/pixtral-12b-bf16`                        |   1,278 |             3,261 |                   500 |          3,761 |        2,056 |      20.2 |          28 |           39.31s |      2.84s |      42.42s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |     374 |             6,574 |                   500 |          7,074 |          299 |      17.8 |          78 |           50.55s |      9.18s |      60.04s | degeneration, ...               |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |     279 |            16,700 |                   500 |         17,200 |          292 |      78.2 |          12 |           64.66s |      1.80s |      66.75s | fabrication, ...                |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |  13,570 |            16,700 |                   500 |         17,200 |          277 |      51.6 |          76 |           72.89s |      8.14s |      81.33s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |  11,072 |               721 |                   500 |          1,221 |          223 |      5.55 |          64 |           93.71s |      7.00s |     101.02s | repetitive(phrase: "church...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |     524 |            16,700 |                   500 |         17,200 |          316 |       8.7 |          26 |          111.34s |      3.08s |     114.73s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |      12 |            16,700 |                   500 |         17,200 |          157 |      41.4 |          35 |          120.03s |      4.32s |     124.67s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |     271 |             1,645 |                   500 |          2,145 |          105 |      5.76 |          48 |          126.14s |      1.33s |     127.75s | repetitive(phrase: ",… ,… ,...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |   9,042 |               430 |                   500 |            930 |          111 |      3.46 |          25 |          148.95s |      3.26s |     152.55s | repetitive(phrase: "stone c...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |       9 |            16,700 |                   500 |         17,200 |          120 |      12.6 |          26 |          180.49s |      2.04s |     182.86s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |     348 |            16,700 |                   500 |         17,200 |          136 |      6.18 |          39 |          204.94s |      2.54s |     207.80s | refusal(explicit_refusal), ...  |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Automated review digest:_ [review.md](review.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.4.0
- _macOS Version:_ 26.4.1
- _SDK Version:_ 26.4
- _Xcode Version:_ 26.4.1
- _Xcode Build:_ 17E202
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

_Report generated on: 2026-04-18 00:45:49 BST_
