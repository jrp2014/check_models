# Model Performance Results

_Generated on 2026-05-09 23:43:04 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 7 (top owners: mlx-vlm=4, mlx=2,
  model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, clean outputs=0/48.
- _Useful now:_ 1 clean A/B model(s) worth first review.
- _Review watchlist:_ 47 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=21, neutral=3, worse=24 (baseline C 62/100).
- _Quality signal frequency:_ missing_sections=30, cutoff=17,
  context_ignored=13, context_echo=12, description_length=9,
  metadata_borrowing=9.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (52%; 16/55 measured model(s)).
- _Phase totals:_ model load=107.42s, local prompt prep=0.16s, upstream
  prefill / first-token=641.95s, post-prefill decode=481.53s, generation total
  (unsplit)=3.70s, cleanup=6.02s.
- _Generation total:_ 1127.18s across 51 model(s); upstream prefill /
  first-token split available for 48/51 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=29, exception=7, max_tokens=19.
- _Validation overhead:_ 11.28s total (avg 0.21s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 13.37s | Min 0.08s | Max
  104.99s across 48 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (352.8 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (0.51s)
- **📊 Average TPS:** 70.0 across 48 models

## 📈 Resource Usage

- **Total peak memory:** 1011.8 GB
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 256 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 9 | ✅ B: 12 | 🟡 C: 9 | 🟠 D: 11 | ❌ F: 7

**Average Utility Score:** 57/100

**Existing Metadata Baseline:** 🟡 C (62/100)
**Vs Existing Metadata:** Avg Δ -5 | Better: 21, Neutral: 3, Worse: 24

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
  - `microsoft/Phi-3.5-vision-instruct` (token: `repetitive(phrase: "flags, flags, flags, flags,...")`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `repetitive(phrase: "flags, flags, flags, flags,...")`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `repetitive(phrase: "18:33:45 18:33:45 18:33:45 18:...")`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `repetitive(phrase: "the image is in...")`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `repetitive(phrase: "- output only the...")`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 70.0 | Min: 4.75 | Max: 353
- **Peak Memory**: Avg: 21 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 25.67s | Min: 1.45s | Max: 137.03s
- **Generation Time**: Avg: 23.41s | Min: 0.67s | Max: 133.15s
- **Model Load Time**: Avg: 2.05s | Min: 0.51s | Max: 9.30s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 82/100 | Description 87 | Keywords 84 | Speed 31.2 TPS | Memory
  19 | Caveat missing terms: classic, style, wooden, during, receded;
  keywords=19)
- _Best descriptions:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 82/100 | Description 87 | Keywords 84 | Speed 31.2 TPS | Memory
  19 | Caveat missing terms: classic, style, wooden, during, receded;
  keywords=19)
- _Best keywording:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 82/100 | Description 87 | Keywords 84 | Speed 31.2 TPS | Memory
  19 | Caveat missing terms: classic, style, wooden, during, receded;
  keywords=19)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility D 39/100 | Description 78 | Keywords 0 | Speed 353 TPS | Memory 2.6
  | Caveat missing sections: keywords)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility D 45/100 | Description 70 | Keywords 0 | Speed 347 TPS | Memory 2.1
  | Caveat missing sections: keywords; nonvisual metadata reused)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 82/100 | Description 87 | Keywords 84 | Speed 31.2 TPS | Memory
  19 | Caveat missing terms: classic, style, wooden, during, receded;
  keywords=19)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (7):_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  +3 more. Example: `Model Error`.
- _🔄 Repetitive Output (5):_ [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Phi-3.5-vision-instruct-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-phi-35-vision-instruct-bf16),
  [`mlx-community/gemma-3n-E2B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-paligemma2-3b-ft-docci-448-bf16),
  +1 more. Example: token: `repetitive(phrase: "flags, flags, flags,
  flags,...")`.
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (18):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
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

_Overall runtime:_ 1249.93s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                  |      0.19s |       1.17s |                                 |             mlx |
| `facebook/pe-av-large`                                  |                  |      0.14s |       1.11s |                                 |         mlx-vlm |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |            1.51s |      5.00s |       6.91s | hallucination, fabrication, ... |         mlx-vlm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                  |      0.32s |       1.30s |                                 |             mlx |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |            1.10s |      0.44s |       1.75s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |            1.09s |      0.69s |       1.99s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/MolmoPoint-8B-fp16`                      |                  |      2.22s |       3.52s |                                 |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |            0.67s |      0.58s |       1.45s | missing-sections(keywords), ... |                 |
| `qnguyen3/nanoLLaVA`                                    |            1.01s |      0.58s |       1.79s | missing-sections(keywords)      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |            1.08s |      0.63s |       1.92s | fabrication, ...                |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |            1.50s |      0.61s |       2.31s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |            1.50s |      0.61s |       2.32s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |            1.62s |      2.34s |       4.19s | context-ignored                 |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |            1.87s |      0.57s |       2.65s | fabrication, ...                |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |            2.44s |      1.55s |       4.20s | missing-sections(title+desc...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |            2.79s |      0.93s |       3.93s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |            3.03s |      1.13s |       4.41s | context-ignored                 |                 |
| `mlx-community/InternVL3-8B-bf16`                       |            3.98s |      1.73s |       5.92s | context-ignored                 |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |            4.70s |      1.49s |       6.40s | repetitive(phrase: "18:33:4...  |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |            4.87s |      2.55s |       7.63s | context-ignored                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |            5.17s |      2.57s |       7.95s | keyword-count(19)               |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |            5.32s |      2.20s |       7.73s | missing-sections(title+desc...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |            5.50s |      1.93s |       7.64s | description-sentences(3), ...   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |            5.55s |      1.37s |       7.13s | context-ignored                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |            5.78s |      1.44s |       7.43s | context-ignored                 |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |            6.23s |      2.55s |       9.00s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-14B-8bit`                      |            6.29s |      1.77s |       8.27s | description-sentences(3), ...   |                 |
| `mlx-community/pixtral-12b-8bit`                        |            6.48s |      1.72s |       8.42s | description-sentences(3), ...   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |            7.30s |      1.90s |       9.40s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |            7.52s |      2.43s |      10.17s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |            7.56s |      3.40s |      11.17s | context-ignored                 |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |            8.59s |      1.87s |      10.67s | missing-sections(title), ...    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |            8.80s |      2.08s |      11.11s | ⚠️harness(encoding), ...        |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |            9.42s |      0.90s |      10.52s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |            9.43s |      0.94s |      10.58s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/pixtral-12b-bf16`                        |            9.68s |      2.60s |      12.49s | description-sentences(3), ...   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |           12.46s |      1.15s |      13.83s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |           13.35s |      1.31s |      14.88s | missing-sections(title+keyw...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |           15.37s |      1.56s |      17.14s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |           15.55s |      1.40s |      17.16s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |           18.96s |      1.73s |      20.90s | refusal(insufficient_info), ... |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |           20.90s |      1.17s |      22.29s | refusal(uncertainty), ...       |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |           25.66s |      5.50s |      31.37s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |           27.12s |      1.52s |      28.86s | repetitive(phrase: "the ima...  |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |           29.00s |      1.11s |      30.32s | description-sentences(3), ...   |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |           55.75s |      2.49s |      58.44s | missing-sections(descriptio...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |           57.77s |      3.13s |      61.11s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |           59.18s |      0.51s |      59.91s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |           63.68s |      9.30s |      73.21s | missing-sections(title+desc...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |           75.78s |      3.36s |      79.35s | missing-sections(title), ...    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |           81.33s |      1.33s |      82.87s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |           87.12s |      6.02s |      93.35s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |           87.87s |      2.15s |      90.25s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |           97.79s |      3.10s |     101.12s | keyword-count(20), ...          |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |          133.15s |      3.63s |     137.03s | title-length(4), ...            |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

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
- `mlx`: `0.32.0.dev20260509+84961223`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.14.0`
- `transformers`: `5.8.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-09 23:43:04 BST_
