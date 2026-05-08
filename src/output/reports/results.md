# Model Performance Results

_Generated on 2026-05-08 14:04:39 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 7 (top owners: mlx-vlm=4, mlx=2,
  model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, clean outputs=7/48.
- _Useful now:_ 8 clean A/B model(s) worth first review.
- _Review watchlist:_ 40 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=22, neutral=3, worse=23 (baseline C 62/100).
- _Quality signal frequency:_ missing_sections=30, cutoff=16, context_echo=12,
  description_length=9, metadata_borrowing=9, keyword_count=8.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (52%; 16/55 measured model(s)).
- _Phase totals:_ model load=104.15s, local prompt prep=0.15s, upstream
  prefill / first-token=603.55s, post-prefill decode=453.63s, generation total
  (unsplit)=3.71s, cleanup=5.75s.
- _Generation total:_ 1060.89s across 51 model(s); upstream prefill /
  first-token split available for 48/51 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=48, exception=7.
- _Validation overhead:_ 11.29s total (avg 0.21s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 12.57s | Min 0.09s | Max
  73.49s across 48 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (348.6 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 70.7 across 48 models

## 📈 Resource Usage

- **Total peak memory:** 1011.9 GB
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 255 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 9 | ✅ B: 13 | 🟡 C: 8 | 🟠 D: 11 | ❌ F: 7

**Average Utility Score:** 58/100

**Existing Metadata Baseline:** 🟡 C (62/100)
**Vs Existing Metadata:** Avg Δ -5 | Better: 22, Neutral: 3, Worse: 23

- **Best for cataloging:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (🏆 A, 87/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (93/100)
- **Best keywording:** `mlx-community/Qwen3.6-27B-mxfp8` (97/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 1/100)

### ⚠️ 18 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (38/100) - Keywords are not specific or diverse enough
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (18/100) - Output lacks detail
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-nvfp4`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (1/100) - Output lacks detail
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

- **❌ Failed Models (7):**
  - `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (`Model Error`)
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (`Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (`Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (5):**
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "flags, flags, flags, flags,..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "flags, flags, flags, flags,..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "18:33:45 18:33:45 18:33:45 18:..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the image is in..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- output only the..."`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 70.7 | Min: 4.73 | Max: 349
- **Peak Memory**: Avg: 21 | Min: 2.2 | Max: 78
- **Total Time**: Avg: 24.23s | Min: 1.38s | Max: 105.76s
- **Generation Time**: Avg: 22.02s | Min: 0.71s | Max: 102.44s
- **Model Load Time**: Avg: 1.99s | Min: 0.45s | Max: 7.21s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 87/100 | Description 83 | Keywords 83 | Speed 183 TPS | Memory 13
  | Caveat nontext prompt burden=89%; missing terms: classic, style, during,
  exposing, vast)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (Utility A 86/100 | Description 93 | Keywords 86 | Speed 62.9 TPS | Memory
  19 | Caveat nontext prompt burden=89%; missing terms: style, during,
  receded, exposing, vast)
- _Best keywording:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit)
  (Utility A 86/100 | Description 84 | Keywords 92 | Speed 17.7 TPS | Memory
  33 | Caveat missing terms: classic, style, wooden, estuary, receded)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility D 39/100 | Description 78 | Keywords 0 | Speed 349 TPS | Memory 2.6
  | Caveat missing sections: keywords; context echo=98%)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility D 45/100 | Description 70 | Keywords 0 | Speed 348 TPS | Memory 2.2
  | Caveat missing sections: keywords; context echo=44%; nonvisual metadata
  reused)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 87/100 | Description 83 | Keywords 83 | Speed 183 TPS | Memory 13
  | Caveat nontext prompt burden=89%; missing terms: classic, style, during,
  exposing, vast)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (7):_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`facebook/pe-av-large`](model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +3 more. Example: `Model Error`.
- _🔄 Repetitive Output (5):_ [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Phi-3.5-vision-instruct-bf16`](model_gallery.md#model-mlx-community-phi-35-vision-instruct-bf16),
  [`mlx-community/gemma-3n-E2B-4bit`](model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](model_gallery.md#model-mlx-community-paligemma2-3b-ft-docci-448-bf16),
  +1 more. Example: token: `phrase: "flags, flags, flags, flags,..."`.
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (18):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  +14 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types        | Affected Models                                                                                                                                    |
|----------------|------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-vlm`      |          4 | Error, Model Error | `facebook/pe-av-large`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `mlx`          |          2 | Model Error        | `LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`                                                                      |
| `model-config` |          1 | Processor Error    | `mlx-community/MolmoPoint-8B-fp16`                                                                                                                 |

### Actionable Items by Package

#### mlx-vlm

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.p...`
  - Type: `ValueError`
- mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 (Error)
  - Error: `Model runtime error during generation for mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16: property 'text' of 'Naive...`
  - Type: `ValueError`
- mlx-community/LFM2-VL-1.6B-8bit (Error)
  - Error: `Model runtime error during generation for mlx-community/LFM2-VL-1.6B-8bit: property 'text' of 'NaiveStreamingDetokeni...`
  - Type: `ValueError`
- mlx-community/LFM2.5-VL-1.6B-bf16 (Error)
  - Error: `Model runtime error during generation for mlx-community/LFM2.5-VL-1.6B-bf16: property 'text' of 'NaiveStreamingDetoke...`
  - Type: `ValueError`

#### mlx

- LiquidAI/LFM2.5-VL-450M-MLX-bf16 (Model Error)
  - Error: `Model loading failed: Received 2 parameters not in model: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_project...`
  - Type: `ValueError`
- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
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

_Overall runtime:_ 1187.03s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                   |                       |                |              |           |             |                  |      0.14s |       1.08s |                                 |             mlx |
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.12s |       1.06s |                                 |         mlx-vlm |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |                   |                       |                |              |           |             |            1.50s |      4.89s |       6.77s | hallucination, fabrication, ... |         mlx-vlm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.22s |       1.21s |                                 |             mlx |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |                   |                       |                |              |           |             |            1.11s |      0.46s |       1.77s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |            1.10s |      0.53s |       1.83s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.20s |       3.51s |                                 |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               513 |                    90 |            603 |        5,811 |       349 |         2.6 |            0.71s |      0.45s |       1.38s | missing-sections(keywords), ... |                 |
| `qnguyen3/nanoLLaVA`                                    |               513 |                    64 |            577 |        4,689 |       113 |         4.8 |            1.00s |      0.54s |       1.75s | missing-sections(keywords)      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               517 |                   181 |            698 |        5,018 |       348 |         2.2 |            1.08s |      0.65s |       1.94s | fabrication, ...                |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             2,087 |                    70 |          2,157 |        3,994 |       127 |         5.8 |            1.47s |      0.66s |       2.33s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             2,087 |                    70 |          2,157 |        4,036 |       126 |         5.8 |            1.50s |      0.63s |       2.34s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               784 |                    89 |            873 |        1,551 |       113 |          17 |            1.62s |      2.35s |       4.21s |                                 |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               623 |                   137 |            760 |        1,511 |       135 |         5.8 |            1.86s |      0.59s |       2.66s | fabrication, ...                |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,538 |                    31 |          1,569 |        1,359 |      31.4 |          12 |            2.51s |      1.59s |       4.33s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             4,114 |                   127 |          4,241 |        2,281 |       183 |          13 |            2.86s |      0.93s |       4.00s |                                 |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             3,503 |                    20 |          3,523 |        1,592 |      62.4 |         9.7 |            2.95s |      0.94s |       4.11s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,064 |                    87 |          3,151 |        2,833 |      34.1 |          18 |            4.00s |      1.72s |       5.93s |                                 |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               774 |                   500 |          1,274 |        2,508 |       122 |           6 |            4.70s |      1.44s |       6.35s | repetitive(phrase: "18:33:4...  |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               784 |                    88 |            872 |          543 |      26.8 |          20 |            5.06s |      2.55s |       7.82s |                                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               783 |                   113 |            896 |          631 |      30.4 |          19 |            5.27s |      2.33s |       7.81s | keyword-count(19)               |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               484 |                    17 |            501 |          264 |      5.24 |          25 |            5.38s |      2.31s |       7.89s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             4,115 |                   114 |          4,229 |        1,226 |      65.9 |          18 |            5.45s |      1.32s |       6.98s |                                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             4,115 |                   118 |          4,233 |        1,252 |      62.9 |          19 |            5.53s |      1.35s |       7.09s |                                 |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             3,507 |                   117 |          3,624 |        2,167 |      32.3 |          20 |            5.66s |      1.90s |       7.78s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               782 |                   266 |          1,048 |        1,812 |      48.6 |          17 |            6.21s |      2.25s |       8.67s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,064 |                   120 |          3,184 |        1,441 |      31.7 |          19 |            6.28s |      1.77s |       8.26s | description-sentences(3), ...   |                 |
| `mlx-community/pixtral-12b-8bit`                        |             4,662 |                   135 |          4,797 |        1,629 |      37.5 |          18 |            6.81s |      1.70s |       8.72s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               783 |                   104 |            887 |          557 |      17.7 |          33 |            7.61s |      3.44s |      11.26s |                                 |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,538 |                    31 |          1,569 |        1,056 |      5.26 |          27 |            7.73s |      2.48s |      10.44s | missing-sections(title+desc...  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               483 |                   119 |            602 |          286 |      20.7 |          15 |            7.75s |      1.45s |       9.40s | missing-sections(title+desc...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,516 |                   490 |          2,006 |        1,643 |        70 |          18 |            8.30s |      2.74s |      11.25s | missing-sections(title), ...    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             3,619 |                   108 |          3,727 |          696 |      31.6 |          27 |            9.00s |      2.10s |      11.32s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,337 |                   500 |          1,837 |        3,943 |      56.9 |         9.5 |            9.41s |      0.85s |      10.46s | repetitive(phrase: "flags,...   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,337 |                   500 |          1,837 |        3,886 |      56.5 |         9.5 |            9.47s |      0.96s |      10.64s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/pixtral-12b-bf16`                        |             4,662 |                   138 |          4,800 |        1,693 |      18.6 |          28 |           10.52s |      2.58s |      13.31s | description-sentences(3), ...   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,610 |                   500 |          5,110 |        3,883 |      48.4 |         4.6 |           12.06s |      1.17s |      13.45s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,570 |                   500 |          7,070 |        1,029 |      71.7 |         8.4 |           13.68s |      1.33s |      15.23s | missing-sections(title+keyw...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,468 |                   160 |          1,628 |          137 |      52.6 |          29 |           14.27s |      1.18s |      15.67s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,570 |                   500 |          7,070 |        1,130 |      54.1 |          11 |           15.35s |      1.39s |      16.95s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             4,753 |                   500 |          5,253 |        1,579 |      41.3 |          15 |           15.44s |      1.63s |      17.29s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,468 |                   210 |          1,678 |          133 |      30.4 |          36 |           18.46s |      1.73s |      20.39s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,570 |                   500 |          7,070 |          535 |      37.5 |          78 |           25.92s |      5.59s |      31.74s | missing-sections(title+desc...  |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,901 |                   500 |         17,401 |          976 |        54 |          14 |           27.12s |      1.09s |      28.42s | description-sentences(3), ...   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,538 |                   500 |          2,038 |        3,241 |      18.9 |          11 |           27.33s |      1.50s |      29.06s | repetitive(phrase: "the ima...  |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,916 |                   500 |         17,416 |          323 |      90.6 |          12 |           58.50s |      1.37s |      60.09s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,901 |                    11 |         16,912 |          291 |       196 |         5.1 |           58.71s |      0.55s |      59.47s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,916 |                   500 |         17,416 |          312 |       105 |          26 |           59.58s |      2.48s |      62.28s | missing-sections(descriptio...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,916 |                   500 |         17,416 |          315 |      88.2 |          35 |           59.92s |      3.12s |      63.25s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,916 |                   500 |         17,416 |          297 |      64.8 |          76 |           65.32s |      7.21s |      72.74s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               772 |                   500 |          1,272 |          264 |      7.27 |          64 |           72.04s |      5.95s |      78.21s | missing-sections(title+desc...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,516 |                   351 |          1,867 |        1,057 |      4.73 |          39 |           76.14s |      3.27s |      79.62s | missing-sections(title), ...    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,916 |                   500 |         17,416 |          248 |      29.9 |          26 |           85.52s |      2.19s |      87.94s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,916 |                   500 |         17,416 |          232 |      17.8 |          39 |          101.70s |      3.17s |     105.10s | title-length(4), ...            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,916 |                   500 |         17,416 |          230 |      17.6 |          39 |          102.44s |      3.09s |     105.76s | keyword-count(20), ...          |                 |

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
- `mlx`: `0.32.0.dev20260508+a1c0b6f9`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.14.0`
- `transformers`: `5.8.0.dev0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-08 14:04:39 BST_
