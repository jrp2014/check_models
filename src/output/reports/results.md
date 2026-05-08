# Model Performance Results

_Generated on 2026-05-08 23:54:43 BST_

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
- _Quality signal frequency:_ missing_sections=28, cutoff=16, context_echo=12,
  description_length=10, metadata_borrowing=9, reasoning_leak=9.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (51%; 16/55 measured model(s)).
- _Phase totals:_ model load=106.62s, local prompt prep=0.16s, upstream
  prefill / first-token=602.82s, post-prefill decode=456.09s, generation total
  (unsplit)=3.72s, cleanup=5.94s.
- _Generation total:_ 1062.63s across 51 model(s); upstream prefill /
  first-token split available for 48/51 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=30, exception=7, max_tokens=18.
- _Validation overhead:_ 11.34s total (avg 0.21s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 12.56s | Min 0.09s | Max
  74.15s across 48 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/FastVLM-0.5B-bf16` (346.7 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `qnguyen3/nanoLLaVA` (0.54s)
- **📊 Average TPS:** 70.9 across 48 models

## 📈 Resource Usage

- **Total peak memory:** 1011.9 GB
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 256 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 9 | ✅ B: 13 | 🟡 C: 8 | 🟠 D: 11 | ❌ F: 7

**Average Utility Score:** 58/100

**Existing Metadata Baseline:** 🟡 C (62/100)
**Vs Existing Metadata:** Avg Δ -5 | Better: 22, Neutral: 3, Worse: 23

- **Best for cataloging:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (🏆 A, 87/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (93/100)
- **Best keywording:** `mlx-community/Qwen3.6-27B-mxfp8` (97/100)
- **Worst for cataloging:** `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (❌ F, 4/100)

### ⚠️ 18 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (38/100) - Keywords are not specific or diverse enough
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (18/100) - Output lacks detail
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-nvfp4`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (46/100) - Lacks visual description of image
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (4/100) - Output too short to be useful
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

- **Generation Tps**: Avg: 70.9 | Min: 4.7 | Max: 347
- **Peak Memory**: Avg: 21 | Min: 2.1 | Max: 78
- **Total Time**: Avg: 24.32s | Min: 1.52s | Max: 106.40s
- **Generation Time**: Avg: 22.06s | Min: 0.68s | Max: 103.01s
- **Model Load Time**: Avg: 2.04s | Min: 0.54s | Max: 8.84s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 87/100 | Description 83 | Keywords 83 | Speed 183 TPS | Memory 13
  | Caveat nontext prompt burden=89%; missing terms: classic, style, during,
  exposing, vast)
- _Best descriptions:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
  (Utility A 86/100 | Description 93 | Keywords 86 | Speed 63.1 TPS | Memory
  19 | Caveat nontext prompt burden=89%; missing terms: style, during,
  receded, exposing, vast)
- _Best keywording:_ [`mlx-community/gemma-3-27b-it-qat-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit)
  (Utility A 86/100 | Description 84 | Keywords 92 | Speed 17.7 TPS | Memory
  33 | Caveat missing terms: classic, style, wooden, estuary, receded)
- _Fastest generation:_ [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility D 45/100 | Description 70 | Keywords 0 | Speed 347 TPS | Memory 2.1
  | Caveat missing sections: keywords; context echo=44%; nonvisual metadata
  reused)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility D 45/100 | Description 70 | Keywords 0 | Speed 347 TPS | Memory 2.1
  | Caveat missing sections: keywords; context echo=44%; nonvisual metadata
  reused)
- _Best balance:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility A 87/100 | Description 83 | Keywords 83 | Speed 183 TPS | Memory 13
  | Caveat nontext prompt burden=89%; missing terms: classic, style, during,
  exposing, vast)

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
  +1 more. Example: token: `phrase: "flags, flags, flags, flags,..."`.
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

_Overall runtime:_ 1191.28s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                   |                       |                |              |           |             |                  |      0.13s |       1.11s |                                 |             mlx |
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.13s |       1.10s |                                 |         mlx-vlm |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |                   |                       |                |              |           |             |            1.48s |      4.94s |       6.81s | hallucination, fabrication, ... |         mlx-vlm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.22s |       1.21s |                                 |             mlx |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |                   |                       |                |              |           |             |            1.11s |      0.45s |       1.78s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |            1.12s |      0.55s |       1.89s | ⚠️harness(stop_token), ...      |         mlx-vlm |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.23s |       3.56s |                                 |    model-config |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               513 |                    90 |            603 |        5,498 |       345 |         2.6 |            0.68s |      0.63s |       1.52s | missing-sections(keywords), ... |                 |
| `qnguyen3/nanoLLaVA`                                    |               513 |                    64 |            577 |        4,652 |       113 |         4.7 |            1.01s |      0.54s |       1.77s | missing-sections(keywords)      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               517 |                   181 |            698 |        5,089 |       347 |         2.1 |            1.10s |      0.62s |       1.92s | fabrication, ...                |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             2,087 |                    70 |          2,157 |        4,034 |       128 |         5.8 |            1.47s |      0.68s |       2.34s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             2,087 |                    70 |          2,157 |        4,020 |       127 |         5.8 |            1.52s |      0.64s |       2.37s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               784 |                    89 |            873 |        1,564 |       108 |          17 |            1.66s |      2.41s |       4.30s |                                 |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               623 |                   137 |            760 |        1,508 |       132 |         5.8 |            1.90s |      0.57s |       2.67s | fabrication, ...                |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,538 |                    31 |          1,569 |        1,366 |      32.3 |          12 |            2.44s |      1.76s |       4.43s | missing-sections(title+desc...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             3,503 |                    20 |          3,523 |        1,649 |      64.6 |         9.7 |            2.85s |      0.94s |       3.99s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             4,114 |                   127 |          4,241 |        2,283 |       183 |          13 |            2.87s |      0.93s |       4.01s |                                 |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,064 |                    87 |          3,151 |        2,844 |      34.7 |          18 |            3.97s |      1.73s |       5.91s |                                 |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               774 |                   500 |          1,274 |        2,495 |       121 |           6 |            4.75s |      1.45s |       6.42s | repetitive(phrase: "18:33:4...  |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               784 |                    88 |            872 |          561 |      27.1 |          20 |            4.98s |      2.58s |       7.77s |                                 |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               783 |                   113 |            896 |          637 |      31.3 |          19 |            5.16s |      2.35s |       7.73s | keyword-count(19)               |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               484 |                    17 |            501 |          265 |      5.29 |          25 |            5.35s |      2.25s |       7.82s | missing-sections(title+desc...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             4,115 |                   114 |          4,229 |        1,249 |      66.3 |          18 |            5.40s |      1.34s |       6.95s |                                 |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             4,115 |                   118 |          4,233 |        1,257 |      63.1 |          19 |            5.52s |      1.35s |       7.09s |                                 |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             3,507 |                   117 |          3,624 |        2,288 |      32.6 |          20 |            5.55s |      1.91s |       7.67s | description-sentences(3), ...   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               782 |                   266 |          1,048 |        1,804 |      48.6 |          17 |            6.21s |      2.26s |       8.69s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,064 |                   120 |          3,184 |        1,438 |      31.9 |          19 |            6.27s |      1.75s |       8.22s | description-sentences(3), ...   |                 |
| `mlx-community/pixtral-12b-8bit`                        |             4,662 |                   135 |          4,797 |        1,879 |      38.7 |          18 |            6.32s |      1.74s |       8.27s | description-sentences(3), ...   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               483 |                   119 |            602 |          302 |      22.1 |          15 |            7.29s |      1.48s |       8.98s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,538 |                    31 |          1,569 |        1,077 |      5.44 |          27 |            7.48s |      2.59s |      10.29s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               783 |                   104 |            887 |          556 |      17.7 |          33 |            7.59s |      3.41s |      11.22s |                                 |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,516 |                   490 |          2,006 |        1,626 |      68.1 |          18 |            8.54s |      2.01s |      10.76s | missing-sections(title), ...    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             3,619 |                   108 |          3,727 |          684 |      31.6 |          27 |            9.09s |      2.07s |      11.39s | ⚠️harness(encoding), ...        |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,337 |                   500 |          1,837 |        3,908 |      56.8 |         9.5 |            9.42s |      0.92s |      10.55s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,337 |                   500 |          1,837 |        3,914 |      56.6 |         9.5 |            9.47s |      0.84s |      10.52s | repetitive(phrase: "flags,...   |                 |
| `mlx-community/pixtral-12b-bf16`                        |             4,662 |                   138 |          4,800 |        1,932 |      20.5 |          28 |            9.50s |      2.58s |      12.29s | description-sentences(3), ...   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,610 |                   500 |          5,110 |        3,883 |      46.9 |         4.6 |           12.40s |      1.15s |      13.77s | repetitive(phrase: "- outpu...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,570 |                   500 |          7,070 |        1,048 |      71.7 |         8.4 |           13.56s |      1.38s |      15.16s | missing-sections(title+keyw...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,570 |                   500 |          7,070 |        1,142 |        54 |          11 |           15.31s |      1.41s |      16.93s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             4,753 |                   500 |          5,253 |        1,563 |      41.1 |          15 |           15.58s |      1.69s |      17.50s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,468 |                   292 |          1,760 |          146 |      52.1 |          29 |           16.20s |      1.31s |      17.72s | title-length(4), ...            |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,468 |                   243 |          1,711 |          144 |      30.8 |          36 |           18.63s |      1.74s |      20.57s | missing-sections(title+desc...  |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,901 |                   500 |         17,401 |        1,007 |      55.1 |          14 |           26.42s |      1.10s |      27.74s | description-sentences(3), ...   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,570 |                   500 |          7,070 |          510 |      37.5 |          78 |           26.53s |      6.02s |      32.79s | missing-sections(title+desc...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,538 |                   500 |          2,038 |        3,297 |      19.1 |          11 |           27.01s |      1.43s |      28.67s | repetitive(phrase: "the ima...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,916 |                   500 |         17,416 |          318 |       103 |          26 |           58.60s |      2.53s |      61.35s | missing-sections(descriptio...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,901 |                     6 |         16,907 |          290 |       219 |         5.1 |           58.88s |      0.59s |      59.69s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,916 |                   500 |         17,416 |          319 |      88.6 |          12 |           59.39s |      1.36s |      60.96s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,916 |                   500 |         17,416 |          307 |      86.7 |          35 |           61.53s |      3.13s |      64.88s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,916 |                   500 |         17,416 |          302 |        64 |          76 |           64.60s |      8.84s |      73.64s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               772 |                   500 |          1,272 |          229 |      7.37 |          64 |           71.63s |      6.22s |      78.07s | missing-sections(title+desc...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,516 |                   351 |          1,867 |        1,058 |       4.7 |          39 |           76.63s |      3.28s |      80.12s | missing-sections(title), ...    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,916 |                   500 |         17,416 |          249 |      30.1 |          26 |           85.14s |      2.16s |      87.54s | refusal(explicit_refusal), ...  |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,916 |                   500 |         17,416 |          230 |      17.7 |          39 |          102.49s |      3.14s |     105.85s | title-length(4), ...            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,916 |                   500 |         17,416 |          228 |      17.7 |          39 |          103.01s |      3.16s |     106.40s | keyword-count(20), ...          |                 |

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
- `mlx`: `0.32.0.dev20260508+a1c0b6f9`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.14.0`
- `transformers`: `5.8.0.dev0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-08 23:54:43 BST_
