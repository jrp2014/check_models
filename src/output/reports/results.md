# Model Performance Results

_Generated on 2026-04-19 02:03:59 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 6 (top owners: mlx-vlm=4, model-config=2).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=4, clean outputs=1/48.
- _Useful now:_ 1 clean A/B model(s) worth first review.
- _Review watchlist:_ 47 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.

### Quality & Metadata

- _Vs existing metadata:_ better=7, neutral=10, worse=31 (baseline B 74/100).
- _Quality signal frequency:_ missing_sections=36, cutoff=28,
  metadata_borrowing=27, trusted_hint_ignored=19, context_ignored=19,
  reasoning_leak=10.
- _Termination reasons:_ completed=48, exception=6.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (91%; 50/54 measured
  model(s)).
- _Phase totals:_ model load=123.19s, prompt prep=0.20s, decode=1319.96s,
  cleanup=8.20s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=48, exception=6.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (319.0 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.47s)
- **📊 Average TPS:** 60.1 across 48 models

## 📈 Resource Usage

- **Total peak memory:** 1023.3 GB
- **Average peak memory:** 21.3 GB
- **Memory efficiency:** 200 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 5 | ✅ B: 16 | 🟡 C: 7 | 🟠 D: 11 | ❌ F: 9

**Average Utility Score:** 53/100

**Existing Metadata Baseline:** ✅ B (74/100)
**Vs Existing Metadata:** Avg Δ -21 | Better: 7, Neutral: 10, Worse: 31

- **Best for cataloging:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (🏆 A, 89/100)
- **Best descriptions:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (100/100)
- **Best keywording:** `mlx-community/GLM-4.6V-Flash-mxfp4` (89/100)
- **Worst for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (❌ F, 0/100)

### ⚠️ 20 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: 🟠 D (39/100) - Keywords are not specific or diverse enough
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (1/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (2/100) - Output too short to be useful
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (12/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (19/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-bf16`: 🟠 D (36/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (50/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (6):**
  - `Qwen/Qwen3-VL-2B-Instruct` (`Model Error`)
  - `ggml-org/gemma-3-1b-it-GGUF` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`Model Error`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (`Model Error`)
  - `mlx-community/X-Reasoner-7B-8bit` (`Model Error`)
- **🔄 Repetitive Output (8):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `treasured`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "1st image 1st image..."`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "sunset, architecture, warm col..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "white balance: correct. white..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `treasured`)
  - `mlx-community/paligemma2-10b-ft-docci-448-6bit` (token: `phrase: "the pub has a..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "after the label. -..."`)
  - `qnguyen3/nanoLLaVA` (token: `phrase: "motorcycle painting glasses, m..."`)
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 60.1 | Min: 3.84 | Max: 319
- **Peak Memory**: Avg: 21 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 29.44s | Min: 1.67s | Max: 143.54s
- **Generation Time**: Avg: 26.47s | Min: 0.82s | Max: 140.44s
- **Model Load Time**: Avg: 2.45s | Min: 0.47s | Max: 12.41s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (91%; 50/54 measured model(s)).
- **Phase totals:** model load=123.19s, prompt prep=0.20s, decode=1319.96s, cleanup=8.20s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=48, exception=6.

### ⏱ Timing Snapshot

- **Validation overhead:** 26.93s total (avg 0.50s across 54 model(s)).
- **First-token latency:** Avg 13.21s | Min 0.08s | Max 119.62s across 48 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 89/100 | Desc 93 | Keywords 77 | Gen 54.1 TPS | Peak 13 | A 89/100 |
  nontext prompt burden=86% | missing terms: Activities, Berkshire, Couple,
  Door, Fortress)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 89/100 | Desc 93 | Keywords 77 | Gen 54.1 TPS | Peak 13 | A 89/100 |
  nontext prompt burden=86% | missing terms: Activities, Berkshire, Couple,
  Door, Fortress)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 89/100 | Desc 93 | Keywords 77 | Gen 54.1 TPS | Peak 13 | A 89/100 |
  nontext prompt burden=86% | missing terms: Activities, Berkshire, Couple,
  Door, Fortress)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (D 46/100 | Desc 82 | Keywords 0 | Gen 319 TPS | Peak 2.5 | D 46/100 |
  missing sections: keywords | missing terms: Activities, Berkshire, Door,
  Fortress, Kissing)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (D 39/100 | Desc 53 | Keywords 0 | Gen 57.2 TPS | Peak 2.2 | D 39/100 |
  missing sections: title, description, keywords | missing terms: Activities,
  Berkshire, Couple, Door, Fortress)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (A 89/100 | Desc 93 | Keywords 77 | Gen 54.1 TPS | Peak 13 | A 89/100 |
  nontext prompt burden=86% | missing terms: Activities, Berkshire, Couple,
  Door, Fortress)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (6):_ [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`ggml-org/gemma-3-1b-it-GGUF`](model_gallery.md#model-ggml-org-gemma-3-1b-it-gguf),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  +2 more. Example: `Model Error`.
- _🔄 Repetitive Output (8):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/Molmo-7B-D-0924-8bit`](model_gallery.md#model-mlx-community-molmo-7b-d-0924-8bit),
  +4 more. Example: token: `treasured`.
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (20):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  +16 more. Common weakness: Keywords are not specific or diverse enough.

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
> &#45; Description hint: Windsor Castle is illuminated at night, towering over a
> street scene in Windsor, England. Below, people stand on the pavement near
> The Royal Windsor pub, with a couple embracing.
> &#45; Keyword hints: Activities, Adobe Stock, Any Vision, Berkshire, Castle,
> Couple, Door, England, Europe, Fortress, Kissing, Man, Pedestrians, People,
> Round Tower, Sign, Standing, Street Scene, Town, Tree
> &#45; Capture metadata: Taken on 2026-04-18 21:36:24 BST (at 21:36:24 local
> time). GPS: 51.483900°N, 0.604400°W.
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1479.62s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `Qwen/Qwen3-VL-2B-Instruct`                             |         |                   |                       |                |              |           |             |           29.80s |      0.76s |      30.96s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `ggml-org/gemma-3-1b-it-GGUF`                           |         |                   |                       |                |              |           |             |                  |      0.13s |       0.50s |                                 |    model-config |
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.31s |       3.03s |                                 |    model-config |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                   |                       |                |              |           |             |            0.92s |      0.48s |       1.79s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |         |                   |                       |                |              |           |             |           17.44s |      0.78s |      18.61s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/X-Reasoner-7B-8bit`                      |         |                   |                       |                |              |           |             |            1.04s |      1.11s |       2.66s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               530 |                    46 |            576 |         5510 |       319 |         2.5 |            0.82s |      0.47s |       1.67s | missing-sections(keywords), ... |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               790 |                   184 |            974 |         7519 |       297 |         2.9 |            1.28s |      0.55s |       2.44s | fabrication, ...                |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,558 |                     6 |          1,564 |         3192 |      22.5 |          11 |            1.33s |      1.41s |       3.16s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               534 |                    32 |            566 |          991 |      57.2 |         2.2 |            2.22s |      0.66s |       3.30s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,143 |                   117 |          3,260 |         2279 |       145 |         7.8 |            2.80s |      0.93s |       4.19s | metadata-borrowing, ...         |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |   5,690 |               600 |                   500 |          1,100 |         7575 |       163 |         3.7 |            3.67s |      0.55s |       4.64s | repetitive(phrase: "sunset,...  |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |             1,363 |                   167 |          1,530 |         3796 |        58 |         9.5 |            3.75s |      0.85s |       4.99s | title-length(4), ...            |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,558 |                    12 |          1,570 |         1093 |      5.75 |          27 |            4.12s |      2.46s |       7.05s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,761 |                    17 |          2,778 |          859 |      57.5 |         9.7 |            4.24s |      0.92s |       5.62s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |               645 |                   500 |          1,145 |         1725 |       125 |         5.5 |            5.06s |      0.57s |       6.11s | missing-sections(title+desc...  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  43,227 |             1,745 |                   500 |          2,245 |         4191 |       125 |         5.5 |            5.08s |      0.68s |       6.12s | repetitive(treasured), ...      |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,812 |               793 |                   500 |          1,293 |         2483 |       116 |           6 |            5.17s |      1.40s |       6.96s | fabrication, ...                |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  43,227 |             1,745 |                   500 |          2,245 |         3711 |       106 |         5.5 |            5.92s |      0.59s |       6.90s | repetitive(treasured), ...      |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               802 |                    91 |            893 |          654 |      21.6 |          19 |            5.99s |      2.30s |       8.68s | metadata-borrowing              |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |      59 |             1,525 |                   500 |          2,025 |         1748 |      98.8 |          18 |            6.55s |      1.94s |       8.87s | degeneration, ...               |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |     106 |               807 |                    82 |            889 |          556 |      17.1 |          20 |            6.81s |      2.53s |       9.94s | metadata-borrowing              |                 |
| `qnguyen3/nanoLLaVA`                                    |  98,266 |               530 |                   500 |          1,030 |         4694 |      82.8 |         4.7 |            6.93s |      0.65s |       8.34s | repetitive(phrase: "motorcy...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,144 |                   105 |          3,249 |          703 |      54.1 |          13 |            7.06s |      1.45s |       9.02s |                                 |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,313 |                    81 |          2,394 |          906 |      18.2 |          18 |            7.65s |      1.92s |      10.60s | title-length(2), ...            |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,144 |                   122 |          3,266 |          692 |      41.3 |          13 |            8.14s |      1.38s |       9.91s | metadata-borrowing, ...         |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,337 |                   106 |          3,443 |         1078 |      23.6 |          16 |            8.19s |      1.69s |      10.44s | keyword-count(21), ...          |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               801 |                   348 |          1,149 |         1786 |      42.6 |          17 |            9.15s |      2.20s |      11.83s | fabrication, ...                |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     220 |             1,525 |                   500 |          2,025 |         1502 |      67.1 |          22 |            9.33s |      2.37s |      12.50s | repetitive(phrase: "1st ima...  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,646 |                    82 |          2,728 |          508 |      20.6 |          22 |            9.84s |      2.15s |      12.58s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |  11,280 |             1,525 |                   500 |          2,025 |         1786 |        58 |          37 |           10.22s |      3.23s |      14.31s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               802 |                    96 |            898 |          515 |      11.6 |          33 |           10.38s |      3.34s |      14.13s | metadata-borrowing              |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |     270 |             1,363 |                   500 |          1,863 |         3822 |      50.2 |         9.5 |           10.83s |      0.89s |      12.32s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |   2,793 |             4,630 |                   500 |          5,130 |         3603 |      40.6 |         4.6 |           14.44s |      1.08s |      15.92s | repetitive(phrase: "after t...  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               501 |                   126 |            627 |          248 |      10.5 |          15 |           14.56s |      1.80s |      16.76s | missing-sections(title+desc...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,813 |                   500 |          3,313 |         1467 |      24.8 |          19 |           22.98s |      2.15s |      25.87s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   8,446 |             3,428 |                   500 |          3,928 |         1088 |      23.2 |          15 |           25.28s |      1.60s |      27.42s | missing-sections(descriptio...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |      13 |             6,645 |                   500 |          7,145 |          475 |      40.5 |         8.4 |           26.98s |      1.54s |      29.30s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 106,897 |             2,313 |                   500 |          2,813 |         1413 |      25.2 |          18 |           27.11s |      2.04s |      29.96s | degeneration, ...               |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |   5,807 |             1,717 |                   500 |          2,217 |          118 |      39.9 |          41 |           28.03s |      1.22s |      29.65s | repetitive(phrase: "white b...  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |   2,731 |             1,558 |                   500 |          2,058 |         1316 |      17.5 |          12 |           30.58s |      1.54s |      32.52s | repetitive(phrase: "the pub...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |   4,022 |             1,852 |                   500 |          2,352 |          286 |      20.6 |          60 |           31.66s |     10.54s |      42.58s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  49,135 |             6,645 |                   500 |          7,145 |          463 |      29.1 |          11 |           32.50s |      1.92s |      34.84s | missing-sections(title+desc...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               502 |                   167 |            669 |          234 |      3.84 |          25 |           46.16s |      2.16s |      48.70s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |   4,778 |             6,645 |                   500 |          7,145 |          245 |      20.5 |          78 |           52.07s |      8.65s |      61.50s | missing-sections(title+desc...  |                 |
| `mlx-community/pixtral-12b-bf16`                        |   6,780 |             3,337 |                   500 |          3,837 |          868 |      14.4 |          28 |           56.59s |      2.54s |      59.58s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |  22,157 |            16,773 |                   500 |         17,273 |          322 |       105 |          26 |           57.84s |      2.45s |      60.69s | missing-sections(keywords), ... |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 135,210 |             1,717 |                   500 |          2,217 |          110 |      22.6 |          48 |           57.97s |      1.89s |      60.33s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |     264 |            16,773 |                   500 |         17,273 |          260 |      75.5 |          35 |           72.08s |      3.09s |      75.56s | degeneration, ...               |                 |
| `mlx-community/gemma-4-31b-bf16`                        |   7,628 |               795 |                   500 |          1,295 |         97.9 |      7.26 |          65 |           77.64s |     11.09s |      89.33s | missing-sections(title), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |   6,466 |            16,773 |                   500 |         17,273 |          225 |      51.8 |          76 |           85.29s |     12.41s |      98.12s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |      13 |            16,773 |                   500 |         17,273 |          185 |      65.2 |          12 |           99.84s |      1.74s |     102.28s | missing-sections(descriptio...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |   3,559 |            16,773 |                   500 |         17,273 |          214 |      17.5 |          39 |          108.20s |      3.43s |     112.05s | missing-sections(descriptio...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     303 |            16,773 |                   500 |         17,273 |          140 |      25.3 |          26 |          140.44s |      2.66s |     143.54s | degeneration, ...               |                 |

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
- `mlx`: `0.31.2.dev20260419+fa4320d5`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.11.0`
- `transformers`: `5.5.4`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-19 02:03:59 BST_
