# Model Performance Results

_Generated on 2026-05-16 00:16:45 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, mlx-lm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=13, clean outputs=0/52.
- _Useful now:_ none (no clean A/B shortlist for this run).
- _Review watchlist:_ 52 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=11, neutral=4, worse=37 (baseline D 49/100).
- _Quality signal frequency:_ trusted_hint_ignored=48, context_ignored=48,
  missing_sections=45, cutoff=38, repetitive=24, harness=13.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (52%;
  36/55 measured model(s)).
- _Phase totals:_ model load=117.36s, local prompt prep=0.18s, upstream
  prefill / first-token=697.13s, post-prefill decode=873.98s, cleanup=6.02s.
- _Generation total:_ 1571.11s across 52 model(s); upstream prefill /
  first-token split available for 52/52 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=13, exception=3, max_tokens=39.
- _Validation overhead:_ 12.03s total (avg 0.22s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 13.41s | Min 0.04s | Max
  104.42s across 52 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (492.9 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.5 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.48s)
- **📊 Average TPS:** 78.6 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1105.7 GB
- **Average peak memory:** 21.3 GB
- **Memory efficiency:** 245 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 2 | ✅ B: 4 | 🟡 C: 5 | 🟠 D: 23 | ❌ F: 18

**Average Utility Score:** 39/100

**Existing Metadata Baseline:** 🟠 D (49/100)
**Vs Existing Metadata:** Avg Δ -10 | Better: 11, Neutral: 4, Worse: 37

- **Best for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (🏆 A, 96/100)
- **Best descriptions:** `mlx-community/GLM-4.6V-nvfp4` (23/100)
- **Best keywording:** `mlx-community/GLM-4.6V-nvfp4` (0/100)
- **Worst for cataloging:** `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (❌ F, 0/100)

### ⚠️ 41 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (5/100) - Output too short to be useful
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (0/100) - Output too short to be useful
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: ❌ F (12/100) - Output lacks detail
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/GLM-4.6V-Flash-mxfp4`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/GLM-4.6V-nvfp4`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-14B-8bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/InternVL3-8B-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-27B-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-27B-mxfp8`: ❌ F (25/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-6bit`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-9B-MLX-4bit`: ❌ F (25/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.6-27B-mxfp8`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (33/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🟠 D (48/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E4B-it-bf16`: ❌ F (25/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-26b-a4b-it-4bit`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-31b-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-31b-it-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (42/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (19/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (20/100) - Output lacks detail
- `mlx-community/pixtral-12b-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-bf16`: 🟠 D (44/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
- **🔄 Repetitive Output (24):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `phrase: "the, the, the, the,..."`)
  - `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (token: `phrase: "(z_0 z. (z_0 z...."`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (token: `phrase: "(p) (p) (p) (p)..."`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (token: `phrase: "201-oto 201-oto 201-oto 201-ot..."`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (token: `"`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` (token: `phrase: "a a a a..."`)
  - `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (token: `phrase: "e. e. e. e...."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "the image of a..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "the image is cut..."`)
  - `mlx-community/Qwen3.5-27B-mxfp8` (token: `觉`)
  - `mlx-community/Qwen3.5-35B-A3B-4bit` (token: ```)
  - `mlx-community/Qwen3.5-35B-A3B-6bit` (token: `1`)
  - `mlx-community/Qwen3.5-35B-A3B-bf16` (token: `was`)
  - `mlx-community/Qwen3.5-9B-MLX-4bit` (token: `phrase: "2 日 2 日..."`)
  - `mlx-community/Qwen3.6-27B-mxfp8` (token: `phrase: "3 \, y, y..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `phrase: "the, the, the, the,..."`)
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (token: `phrase: "outbre outbre outbre outbre..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `they,`)
  - `mlx-community/gemma-3n-E4B-it-bf16` (token: `1)`)
  - `mlx-community/llava-v1.6-mistral-7b-8bit` (token: `phrase: "the best the best..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "of the frame. the..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- do not output..."`)
  - `mlx-community/pixtral-12b-8bit` (token: `phrase: "a年発売 a年発売 a年発売 a年発売..."`)
  - `mlx-community/pixtral-12b-bf16` (token: `phrase: "b. b. b. b...."`)
- **📝 Formatting Issues (3):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Qwen3.6-27B-mxfp8`
  - `mlx-community/gemma-4-26b-a4b-it-4bit`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 78.6 | Min: 4.73 | Max: 493
- **Peak Memory**: Avg: 21 | Min: 1.5 | Max: 78
- **Total Time**: Avg: 32.69s | Min: 1.28s | Max: 191.82s
- **Generation Time**: Avg: 30.21s | Min: 0.56s | Max: 188.39s
- **Model Load Time**: Avg: 2.25s | Min: 0.48s | Max: 10.99s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4)
  (Utility F 6/100 | Description 23 | Keywords 0 | Speed 39.4 TPS | Memory 78
  | Caveat output/prompt=0.29%; nontext prompt burden=94%; missing sections:
  title, description, keywords; missing terms: Rochester, Castle, turns, Red,
  celebrate)
- _Best descriptions:_ [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4)
  (Utility F 6/100 | Description 23 | Keywords 0 | Speed 39.4 TPS | Memory 78
  | Caveat output/prompt=0.29%; nontext prompt burden=94%; missing sections:
  title, description, keywords; missing terms: Rochester, Castle, turns, Red,
  celebrate)
- _Best keywording:_ [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4)
  (Utility F 6/100 | Description 23 | Keywords 0 | Speed 39.4 TPS | Memory 78
  | Caveat output/prompt=0.29%; nontext prompt burden=94%; missing sections:
  title, description, keywords; missing terms: Rochester, Castle, turns, Red,
  celebrate)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 5/100 | Description 43 | Keywords 0 | Speed 493 TPS | Memory 1.5
  | Caveat hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate; nonvisual
  metadata reused)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 5/100 | Description 43 | Keywords 0 | Speed 493 TPS | Memory 1.5
  | Caveat hit token cap (500); missing sections: title, description,
  keywords; missing terms: Rochester, Castle, turns, Red, celebrate; nonvisual
  metadata reused)
- _Best balance:_ [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4)
  (Utility F 6/100 | Description 23 | Keywords 0 | Speed 39.4 TPS | Memory 78
  | Caveat output/prompt=0.29%; nontext prompt burden=94%; missing sections:
  title, description, keywords; missing terms: Rochester, Castle, turns, Red,
  celebrate)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (24):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-apriel-15-15b-thinker-6bit-mlx),
  [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +20 more. Example: token: `phrase: "the, the, the, the,..."`.
- _📝 Formatting Issues (3):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/Qwen3.6-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen36-27b-mxfp8),
  [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-4-26b-a4b-it-4bit).
- _Low-utility outputs (41):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  +37 more. Common weakness: Keywords are not specific or diverse enough.

## 🚨 Failures by Package (Actionable)

| Package   |   Failures | Error Types                  | Affected Models                                                                |
|-----------|------------|------------------------------|--------------------------------------------------------------------------------|
| `mlx`     |          2 | Model Error, Weight Mismatch | `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `mlx-lm`  |          1 | Model Error                  | `facebook/pe-av-large`                                                         |

### Actionable Items by Package

#### mlx

- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
  - Type: `ValueError`
- mlx-community/LFM2.5-VL-1.6B-bf16 (Weight Mismatch)
  - Error: `Model loading failed: Missing 2 parameters: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm....`
  - Type: `ValueError`

#### mlx-lm

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: Model type pe_audio_video not supported.`
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
> &#45; Description hint: Rochester Castle turns Red to celebrate Medway winning
> its bid to the European Footballing body UEFA to become the UK's first ever
> completely 100 per cent carbon neutral city
> &#45; Capture metadata: Taken on 2026-05-15 21:17:14 BST (at 21:17:14 local
> time). GPS: 51.396828°N, 0.501581°E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1710.22s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.16s |       1.20s |                                    |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.30s |       1.31s |                                    |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                  |      0.15s |       1.16s |                                    |             mlx |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               480 |                    38 |            518 |        5,499 |       304 |         2.5 |            0.56s |      0.50s |       1.28s | missing-sections(keywords), ...    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               484 |                     4 |            488 |        4,851 |       245 |         2.1 |            0.59s |      1.09s |       1.90s | ⚠️harness(prompt_template), ...    |                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               727 |                   500 |          1,227 |       18,306 |       493 |         1.5 |            1.32s |      0.48s |       2.01s | degeneration, ...                  |                 |
| `qnguyen3/nanoLLaVA`                                    |               480 |                   105 |            585 |        4,591 |       113 |         4.6 |            1.40s |      0.70s |       2.32s | missing-sections(keywords), ...    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,031 |                     3 |          3,034 |        2,242 |      49.6 |          18 |            1.79s |      1.86s |       3.87s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               727 |                   500 |          1,227 |        6,066 |       326 |           3 |            1.92s |      0.54s |       2.67s | repetitive("), ...                 |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,508 |                    24 |          1,532 |        1,364 |      31.8 |          12 |            2.23s |      1.68s |       4.13s | missing-sections(title+desc...     |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,770 |                    64 |          1,834 |        1,188 |      60.6 |          60 |            3.13s |      5.15s |       8.66s | missing-sections(title+desc...     |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,031 |                     2 |          3,033 |          904 |      56.4 |          19 |            3.78s |      2.09s |       6.10s | ⚠️harness(long_context), ...       |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             2,055 |                   500 |          2,555 |        3,480 |       125 |         5.8 |            5.03s |      0.72s |       5.96s | repetitive(phrase: "the, th...     |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               743 |                   500 |          1,243 |        2,225 |       113 |         6.1 |            5.10s |      1.64s |       6.97s | repetitive(they,), ...             |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             2,055 |                   500 |          2,555 |        3,430 |       120 |         5.8 |            5.22s |      0.64s |       6.08s | repetitive(phrase: "the, th...     |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               753 |                   500 |          1,253 |        1,229 |       107 |          17 |            5.62s |      2.53s |       8.40s | degeneration, ...                  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,480 |                   299 |          1,779 |        1,151 |      71.8 |          18 |            5.86s |      2.13s |       8.22s | reasoning-leak, ...                |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             4,079 |                   500 |          4,579 |        1,527 |       172 |          13 |            5.99s |      1.06s |       7.29s | repetitive(phrase: "e. e. e...     |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             3,581 |                     2 |          3,583 |          577 |      33.3 |          27 |            6.68s |      2.18s |       9.09s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               591 |                   500 |          1,091 |        1,337 |       128 |           6 |            8.49s |      0.61s |       9.32s | repetitive(phrase: "outbre...      |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,508 |                    39 |          1,547 |        1,107 |      5.49 |          27 |            8.84s |      2.52s |      11.58s | missing-sections(title+desc...     |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,290 |                   500 |          1,790 |        3,232 |      57.7 |         9.5 |            9.36s |      0.98s |      10.55s | missing-sections(title+desc...     |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,290 |                   500 |          1,790 |        3,224 |      56.6 |         9.6 |            9.54s |      0.93s |      10.69s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               751 |                   500 |          1,251 |        1,669 |      47.3 |          17 |           11.35s |      2.34s |      13.93s | repetitive(1)), ...                |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             3,460 |                   500 |          3,960 |        1,247 |      60.2 |         9.7 |           11.55s |      0.91s |      12.68s | repetitive(phrase: "the bes...     |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,580 |                   500 |          5,080 |        3,787 |      45.4 |         4.6 |           12.79s |      1.25s |      14.27s | repetitive(phrase: "- do no...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             4,080 |                   500 |          4,580 |          915 |        62 |          18 |           12.98s |      1.57s |      14.79s | missing-sections(title+desc...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             4,080 |                   500 |          4,580 |          905 |      59.7 |          19 |           13.29s |      1.48s |      15.00s | repetitive(phrase: "a a a a...     |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,535 |                   500 |          7,035 |          784 |      68.4 |         8.4 |           15.99s |      1.60s |      17.82s | degeneration, ...                  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             4,718 |                   500 |          5,218 |        1,209 |      41.1 |          15 |           16.43s |      1.65s |      18.30s | repetitive(phrase: "(z_0 z....     |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,535 |                   500 |          7,035 |          821 |      54.6 |          11 |           17.43s |      1.75s |      19.40s | repetitive(phrase: "(p) (p)...     |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             3,472 |                   500 |          3,972 |        1,984 |      31.6 |          19 |           17.99s |      2.00s |      20.22s | repetitive(phrase: "201-oto...     |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,535 |                    19 |          6,554 |          370 |      39.4 |          78 |           18.48s |      6.63s |      25.34s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               752 |                   500 |          1,252 |          517 |      29.9 |          19 |           18.49s |      2.48s |      21.20s | missing-sections(title+desc...     |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,435 |                   500 |          1,935 |          146 |      51.9 |          29 |           20.12s |      1.46s |      21.80s | repetitive(phrase: "the ima...     |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               753 |                   500 |          1,253 |          481 |      26.4 |          20 |           20.84s |      2.76s |      23.83s | missing-sections(title+desc...     |                 |
| `mlx-community/pixtral-12b-8bit`                        |             4,627 |                   500 |          5,127 |        1,483 |      37.8 |          19 |           23.49s |      1.75s |      25.47s | repetitive(phrase: "a年発売 a年... |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               448 |                   500 |            948 |          260 |      20.9 |          15 |           26.01s |      1.49s |      27.72s | degeneration, ...                  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,435 |                   500 |          1,935 |          139 |      31.6 |          36 |           26.74s |      1.81s |      28.77s | repetitive(phrase: "the ima...     |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,508 |                   500 |          2,008 |        3,290 |      18.9 |          11 |           27.29s |      1.58s |      29.09s | repetitive(phrase: "of the...      |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,868 |                   500 |         17,368 |          861 |      55.7 |          14 |           29.15s |      1.37s |      30.75s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               752 |                   500 |          1,252 |          440 |      16.9 |          33 |           31.63s |      3.56s |      35.42s | missing-sections(title+desc...     |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             2,478 |                   196 |          2,674 |        1,270 |      6.42 |          25 |           33.02s |      2.85s |      36.10s | fabrication, title-length(11), ... |                 |
| `mlx-community/pixtral-12b-bf16`                        |             4,627 |                   500 |          5,127 |        1,680 |      21.8 |          29 |           36.58s |      2.77s |      39.57s | repetitive(phrase: "b. b. b...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,882 |                   500 |         17,382 |          306 |      79.7 |          35 |           62.19s |      4.45s |      66.87s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,882 |                   500 |         17,382 |          297 |      96.2 |          26 |           62.72s |      2.62s |      65.56s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               741 |                   500 |          1,241 |          255 |      8.24 |          64 |           64.04s |      6.71s |      70.98s | missing-sections(title+desc...     |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,480 |                   300 |          1,780 |          984 |      4.73 |          39 |           65.49s |      3.36s |      69.07s | title-length(13), ...              |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,882 |                   500 |         17,382 |          288 |      61.1 |          76 |           67.58s |     10.99s |      78.81s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,868 |                   500 |         17,368 |          261 |       181 |         5.1 |           68.01s |      0.59s |      68.83s | ⚠️harness(long_context), ...       |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               449 |                   500 |            949 |          246 |      5.04 |          25 |          101.35s |      2.72s |     104.30s | degeneration, ...                  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,882 |                   500 |         17,382 |          200 |      17.7 |          39 |          113.39s |      3.27s |     116.97s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,882 |                   500 |         17,382 |          295 |      88.1 |          13 |          121.40s |      1.48s |     123.11s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,882 |                   500 |         17,382 |          162 |      28.8 |          26 |          122.47s |      2.30s |     125.02s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,882 |                   500 |         17,382 |          213 |      17.7 |          41 |          188.39s |      3.19s |     191.82s | ⚠️harness(long_context), ...       |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5
- _SDK Version:_ 26.5
- _Xcode Version:_ 26.5
- _Xcode Build:_ 17F42
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
- `mlx`: `0.32.0.dev20260515+7b7c1240`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.15.0`
- `transformers`: `5.8.0.dev0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-16 00:16:45 BST_
