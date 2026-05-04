# Model Performance Results

_Generated on 2026-05-04 22:51:32 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 4 (top owners: mlx=2, mlx-vlm=1,
  model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, clean outputs=7/51.
- _Useful now:_ 8 clean A/B model(s) worth first review.
- _Review watchlist:_ 43 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=20, neutral=3, worse=28 (baseline C 62/100).
- _Quality signal frequency:_ missing_sections=32, cutoff=19, context_echo=13,
  description_length=10, keyword_count=10, metadata_borrowing=10.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (44%; 16/55 measured model(s)).
- _Phase totals:_ model load=339.14s, local prompt prep=0.16s, upstream
  prefill / first-token=638.53s, post-prefill decode=475.09s, cleanup=5.98s.
- _Generation total:_ 1113.62s across 51 model(s); upstream prefill /
  first-token split available for 51/51 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=51, exception=4.
- _Validation overhead:_ 11.79s total (avg 0.21s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 12.52s | Min 0.08s | Max
  79.82s across 51 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (342.7 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.36s)
- **📊 Average TPS:** 75.4 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1078.3 GB
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 244 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 9 | ✅ B: 11 | 🟡 C: 11 | 🟠 D: 13 | ❌ F: 7

**Average Utility Score:** 57/100

**Existing Metadata Baseline:** 🟡 C (62/100)
**Vs Existing Metadata:** Avg Δ -5 | Better: 20, Neutral: 3, Worse: 28

- **Best for cataloging:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (🏆 A, 87/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (93/100)
- **Best keywording:** `mlx-community/Qwen3.6-27B-mxfp8` (97/100)
- **Worst for cataloging:** `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (❌ F, 4/100)

### ⚠️ 20 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (38/100) - Keywords are not specific or diverse enough
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (18/100) - Output lacks detail
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-nvfp4`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (42/100) - Missing requested structure
- `mlx-community/LFM2.5-VL-1.6B-bf16`: 🟠 D (36/100) - Lacks visual description of image
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (33/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (38/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (30/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (16/100) - Output lacks detail
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (39/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (20/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (20/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (4):**
  - `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (`Model Error`)
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (7):**
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "flags, flags, flags, flags,..."`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (token: `phrase: "mudflats, flags, boat, water,..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "flags, flags, flags, flags,..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "boat anchor boat anchor..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "18:33:45 18:33:45 18:33:45 18:..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the image is in..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 75.4 | Min: 4.66 | Max: 343
- **Peak Memory**: Avg: 21 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 24.16s | Min: 1.32s | Max: 112.25s
- **Generation Time**: Avg: 21.84s | Min: 0.68s | Max: 108.92s
- **Model Load Time**: Avg: 2.10s | Min: 0.36s | Max: 9.96s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 87/100 | Description 83 | Keywords 83 | Speed 177 TPS | Memory 13
  | Caveat nontext prompt burden=89%; missing terms: classic, style, during,
  exposing, vast)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (Utility A 86/100 | Description 93 | Keywords 86 | Speed 59.1 TPS | Memory
  19 | Caveat nontext prompt burden=89%; missing terms: style, during,
  receded, exposing, vast)
- _Best keywording:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit)
  (Utility A 86/100 | Description 84 | Keywords 92 | Speed 17.3 TPS | Memory
  33 | Caveat missing terms: classic, style, wooden, estuary, receded)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility D 39/100 | Description 78 | Keywords 0 | Speed 343 TPS | Memory 2.5
  | Caveat missing sections: keywords; context echo=98%)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility D 45/100 | Description 70 | Keywords 0 | Speed 336 TPS | Memory 2.1
  | Caveat missing sections: keywords; context echo=44%; nonvisual metadata
  reused)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 87/100 | Description 83 | Keywords 83 | Speed 177 TPS | Memory 13
  | Caveat nontext prompt burden=89%; missing terms: classic, style, during,
  exposing, vast)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (4):_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`facebook/pe-av-large`](model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Model Error`.
- _🔄 Repetitive Output (7):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/Phi-3.5-vision-instruct-bf16`](model_gallery.md#model-mlx-community-phi-35-vision-instruct-bf16),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  +3 more. Example: token: `phrase: "flags, flags, flags, flags,..."`.
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (20):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  +16 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types     | Affected Models                                                               |
|----------------|------------|-----------------|-------------------------------------------------------------------------------|
| `mlx`          |          2 | Model Error     | `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit` |
| `mlx-vlm`      |          1 | Model Error     | `facebook/pe-av-large`                                                        |
| `model-config` |          1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16`                                            |

### Actionable Items by Package

#### mlx

- LiquidAI/LFM2.5-VL-450M-MLX-bf16 (Model Error)
  - Error: `Model loading failed: Received 2 parameters not in model: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_project...`
  - Type: `ValueError`
- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
  - Type: `ValueError`

#### mlx-vlm

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.p...`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
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
> &#45; Description hint: A classic-style sailboat with a dark hull and wooden
> mast is moored in a calm estuary during low tide. The water has receded,
> exposing a vast expanse of green, algae-covered mudflats behind the vessel.
> The boat, adorned with a string of small flags, floats peacefully, waiting
> for the tide to rise again.
> &#45; Capture metadata: Taken on 2026-05-02 18:33:45 BST (at 18:33:45 local
> time). GPS: 52.089294°N, 1.317741°E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1475.66s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                   |                       |                |              |           |             |                  |      0.13s |       1.14s |                                 |             mlx |
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |    229.48s |     230.49s |                                 |         mlx-vlm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.17s |       1.15s |                                 |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.23s |       3.57s |                                 |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               513 |                    90 |            603 |        6,066 |       343 |         2.5 |            0.68s |      0.44s |       1.32s | missing-sections(keywords), ... |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               767 |                   110 |            877 |        7,191 |       322 |           3 |            0.71s |      0.45s |       1.37s | title-length(3), ...            |                 |
| `qnguyen3/nanoLLaVA`                                    |               513 |                    64 |            577 |        4,395 |       109 |         4.5 |            1.05s |      0.54s |       1.82s | missing-sections(keywords)      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               517 |                   181 |            698 |        4,905 |       336 |         2.1 |            1.21s |      0.65s |       2.09s | fabrication, ...                |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             2,087 |                    70 |          2,157 |        4,029 |       120 |         5.8 |            1.50s |      0.36s |       2.06s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             2,087 |                    70 |          2,157 |        3,974 |       123 |         5.8 |            1.52s |      0.59s |       2.33s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               784 |                    89 |            873 |        1,561 |       110 |          17 |            1.65s |      2.34s |       4.22s |                                 |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               623 |                   137 |            760 |        1,507 |       132 |         5.8 |            1.89s |      0.56s |       2.65s | fabrication, ...                |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,538 |                    31 |          1,569 |        1,352 |        31 |          12 |            2.48s |      1.64s |       4.34s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             4,114 |                   127 |          4,241 |        2,183 |       177 |          13 |            3.03s |      0.99s |       4.25s |                                 |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             3,503 |                    20 |          3,523 |        1,522 |      63.9 |         9.7 |            3.04s |      0.89s |       4.14s | missing-sections(title+desc...  |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               575 |                   500 |          1,075 |        7,594 |       181 |         3.7 |            3.09s |      0.52s |       3.82s | repetitive(phrase: "mudflat...  |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,064 |                    87 |          3,151 |        2,830 |      34.1 |          18 |            4.00s |      1.68s |       5.90s |                                 |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               774 |                   500 |          1,274 |        2,456 |       121 |           6 |            4.76s |      1.43s |       6.40s | repetitive(phrase: "18:33:4...  |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               784 |                    88 |            872 |          513 |      26.7 |          20 |            5.16s |      2.55s |       7.93s |                                 |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               484 |                    17 |            501 |          264 |       5.3 |          25 |            5.35s |      2.17s |       7.74s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             4,115 |                   114 |          4,229 |        1,235 |      64.4 |          18 |            5.47s |      1.30s |       7.00s |                                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               783 |                   113 |            896 |          579 |      29.5 |          19 |            5.51s |      2.43s |       8.18s | keyword-count(19)               |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             3,507 |                   117 |          3,624 |        2,376 |      32.3 |          20 |            5.57s |      1.97s |       7.75s | description-sentences(3), ...   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             4,115 |                   118 |          4,233 |        1,215 |      59.1 |          19 |            5.78s |      1.37s |       7.37s |                                 |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               782 |                   266 |          1,048 |        1,789 |      46.6 |          17 |            6.45s |      2.23s |       8.90s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,064 |                   120 |          3,184 |        1,342 |      31.8 |          19 |            6.45s |      1.78s |       8.46s | description-sentences(3), ...   |                 |
| `mlx-community/pixtral-12b-8bit`                        |             4,662 |                   135 |          4,797 |        1,620 |      38.2 |          18 |            6.78s |      1.65s |       8.65s | description-sentences(3), ...   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               483 |                   119 |            602 |          295 |      21.7 |          15 |            7.43s |      1.50s |       9.13s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,538 |                    31 |          1,569 |        1,058 |      5.41 |          27 |            7.54s |      2.44s |      10.19s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               783 |                   104 |            887 |          533 |      17.3 |          33 |            7.80s |      3.49s |      11.54s |                                 |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,516 |                   490 |          2,006 |        1,597 |      67.7 |          18 |            8.60s |      2.07s |      10.87s | missing-sections(title), ...    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             4,662 |                   138 |          4,800 |        1,908 |        21 |          28 |            9.38s |      2.55s |      12.15s | description-sentences(3), ...   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             3,619 |                   108 |          3,727 |          663 |      30.1 |          27 |            9.44s |      2.11s |      11.82s | ⚠️harness(encoding), ...        |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,337 |                   500 |          1,837 |        3,814 |      55.9 |         9.5 |            9.58s |      0.88s |      10.67s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,337 |                   500 |          1,837 |        3,868 |      55.2 |         9.5 |            9.69s |      0.84s |      10.75s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,810 |                   500 |          2,310 |          687 |        51 |          60 |           13.05s |      5.92s |      19.40s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,610 |                   500 |          5,110 |        3,505 |        42 |         4.6 |           13.79s |      1.26s |      15.28s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,570 |                   500 |          7,070 |        1,053 |      68.1 |         8.4 |           13.90s |      1.32s |      15.44s | missing-sections(title+keyw...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,468 |                   190 |          1,658 |          152 |      50.6 |          29 |           13.92s |      1.46s |      15.61s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,570 |                   500 |          7,070 |        1,111 |      53.2 |          11 |           15.63s |      1.45s |      17.30s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             4,753 |                   500 |          5,253 |        1,550 |      39.9 |          15 |           15.95s |      1.56s |      17.74s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,468 |                   163 |          1,631 |          143 |      30.6 |          36 |           16.16s |      1.81s |      18.20s | missing-sections(title+desc...  |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,901 |                   500 |         17,401 |          954 |      53.7 |          14 |           27.59s |      1.29s |      29.09s | description-sentences(3), ...   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,538 |                   500 |          2,038 |        3,163 |      18.6 |          11 |           27.81s |      1.49s |      29.52s | repetitive(phrase: "the ima...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,570 |                   500 |          7,070 |          465 |      36.1 |          78 |           28.30s |      7.10s |      35.62s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,916 |                   500 |         17,416 |          320 |       101 |          26 |           58.49s |      2.46s |      61.17s | missing-sections(descriptio...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,916 |                   500 |         17,416 |          290 |      79.5 |          35 |           65.40s |      3.13s |      68.75s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,901 |                   500 |         17,401 |          268 |       192 |         5.1 |           66.38s |      0.55s |      67.18s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,916 |                   500 |         17,416 |          279 |      84.4 |          12 |           67.21s |      1.44s |      68.87s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               772 |                   500 |          1,272 |          186 |       7.6 |          64 |           70.36s |      6.95s |      77.53s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,916 |                   500 |         17,416 |          271 |        60 |          76 |           71.52s |      9.96s |      81.70s | missing-sections(title+desc...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,516 |                   351 |          1,867 |        1,054 |      4.66 |          39 |           77.22s |      3.22s |      80.65s | missing-sections(title), ...    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,916 |                   500 |         17,416 |          246 |      28.9 |          26 |           86.83s |      2.19s |      89.25s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,916 |                   500 |         17,416 |          230 |      17.6 |          39 |          102.60s |      3.08s |     105.96s | keyword-count(20), ...          |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,916 |                   500 |         17,416 |          212 |      17.6 |          39 |          108.92s |      3.07s |     112.25s | title-length(4), ...            |                 |

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
- `mlx`: `0.32.0.dev20260504+e8ebdebe`
- `mlx-vlm`: `0.4.5`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.13.0`
- `transformers`: `5.8.0.dev0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-04 22:51:32 BST_
