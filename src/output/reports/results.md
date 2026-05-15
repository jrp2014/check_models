# Model Performance Results

_Generated on 2026-05-15 12:25:02 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, mlx-lm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=13, clean outputs=0/52.
- _Useful now:_ none (no clean A/B shortlist for this run).
- _Review watchlist:_ 52 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=0, neutral=2, worse=50 (baseline B 75/100).
- _Quality signal frequency:_ missing_sections=48, trusted_hint_ignored=46,
  context_ignored=46, cutoff=40, repetitive=27, degeneration=14.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (44%;
  34/55 measured model(s)).
- _Phase totals:_ model load=198.78s, local prompt prep=0.17s, upstream
  prefill / first-token=712.64s, post-prefill decode=731.66s, cleanup=5.63s.
- _Generation total:_ 1444.30s across 52 model(s); upstream prefill /
  first-token split available for 52/52 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=10, exception=3, max_tokens=42.
- _Validation overhead:_ 9.67s total (avg 0.18s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 13.70s | Min 0.12s | Max
  104.90s across 52 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (485.0 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.5 GB)
- **⚡ Fastest load:** `mlx-community/LFM2-VL-1.6B-8bit` (0.58s)
- **📊 Average TPS:** 78.9 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1110.0 GB
- **Average peak memory:** 21.3 GB
- **Memory efficiency:** 243 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 4 | 🟡 C: 3 | 🟠 D: 29 | ❌ F: 16

**Average Utility Score:** 38/100

**Existing Metadata Baseline:** ✅ B (75/100)
**Vs Existing Metadata:** Avg Δ -37 | Better: 0, Neutral: 2, Worse: 50

- **Best for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (✅ B, 77/100)
- **Best descriptions:** `mlx-community/MolmoPoint-8B-fp16` (93/100)
- **Best keywording:** `mlx-community/MolmoPoint-8B-fp16` (57/100)
- **Worst for cataloging:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (❌ F, 0/100)

### ⚠️ 45 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (0/100) - Output too short to be useful
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (0/100) - Output too short to be useful
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/GLM-4.6V-Flash-6bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-Flash-mxfp4`: 🟠 D (40/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-nvfp4`: ❌ F (16/100) - Output lacks detail
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (26/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-14B-8bit`: 🟠 D (41/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🟠 D (42/100) - Keywords are not specific or diverse enough
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/MolmoPoint-8B-fp16`: 🟠 D (48/100) - Limited novel information
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (47/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-27B-4bit`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-4bit`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-6bit`: 🟠 D (48/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-bf16`: 🟠 D (48/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-9B-MLX-4bit`: ❌ F (33/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.6-27B-mxfp8`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (33/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E4B-it-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-26b-a4b-it-4bit`: 🟠 D (39/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-31b-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-31b-it-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (35/100) - Keywords are not specific or diverse enough
- `mlx-community/nanoLLaVA-1.5-4bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (16/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (16/100) - Output lacks detail
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-8bit`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (41/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
- **🔄 Repetitive Output (27):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `phrase: "th, th, th, th,..."`)
  - `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (token: `phrase: "the sum the sum..."`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (token: `phrase: "the value, the value,..."`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (token: `and,`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (token: `phrase: "and<fake_token_around_image> a..."`)
  - `mlx-community/InternVL3-14B-8bit` (token: `phrase: "the given text. the..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "answer. answer. answer. answer..."`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (token: `phrase: "and fortunately and fortunatel..."`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` (token: `phrase: "_____ is _____ is..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "the image shows a..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "a building. the image..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "and the world, and..."`)
  - `mlx-community/Qwen3.5-27B-4bit` (token: `1`)
  - `mlx-community/Qwen3.5-27B-mxfp8` (token: `phrase: "- 伊 - 德..."`)
  - `mlx-community/Qwen3.5-35B-A3B-4bit` (token: `作为`)
  - `mlx-community/Qwen3.5-35B-A3B-6bit` (token: `phrase: "100% biodegradable, 100% recyc..."`)
  - `mlx-community/Qwen3.5-35B-A3B-bf16` (token: `phrase: "9780470474335_ch01_p001-016.qx..."`)
  - `mlx-community/Qwen3.5-9B-MLX-4bit` (token: `phrase: "and 0+ 1. of..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `phrase: "th, th, th, th,..."`)
  - `mlx-community/X-Reasoner-7B-8bit` (token: `phrase: "1.<|endoftext|>the 1.<|endofte..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "they, they, they, they,..."`)
  - `mlx-community/gemma-3n-E4B-it-bf16` (token: `•`)
  - `mlx-community/gemma-4-26b-a4b-it-4bit` (token: `phrase: "3- 3- 3- 3-..."`)
  - `mlx-community/llava-v1.6-mistral-7b-8bit` (token: `phrase: "the best the best..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "- do not output..."`)
  - `mlx-community/pixtral-12b-8bit` (token: `phrase: "b. b. b. b...."`)
  - `mlx-community/pixtral-12b-bf16` (token: `phrase: "essage the essage the..."`)
- **📝 Formatting Issues (2):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 78.9 | Min: 4.56 | Max: 485
- **Peak Memory**: Avg: 21 | Min: 1.5 | Max: 78
- **Total Time**: Avg: 31.70s | Min: 1.58s | Max: 136.94s
- **Generation Time**: Avg: 27.77s | Min: 0.69s | Max: 133.43s
- **Model Load Time**: Avg: 3.74s | Min: 0.58s | Max: 11.55s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility D 48/100 | Description 93 | Keywords 57 | Speed 5.91 TPS | Memory
  27 | Caveat output/prompt=4.14%; nontext prompt burden=86%; missing terms:
  Bench, rises; keywords=19)
- _Best descriptions:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility D 48/100 | Description 93 | Keywords 57 | Speed 5.91 TPS | Memory
  27 | Caveat output/prompt=4.14%; nontext prompt burden=86%; missing terms:
  Bench, rises; keywords=19)
- _Best keywording:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility D 48/100 | Description 93 | Keywords 57 | Speed 5.91 TPS | Memory
  27 | Caveat output/prompt=4.14%; nontext prompt burden=86%; missing terms:
  Bench, rises; keywords=19)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 41 | Keywords 0 | Speed 485 TPS | Memory 1.5
  | Caveat hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=character_loop: 'orm' repeated)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 41 | Keywords 0 | Speed 485 TPS | Memory 1.5
  | Caveat hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=character_loop: 'orm' repeated)
- _Best balance:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility D 48/100 | Description 93 | Keywords 57 | Speed 5.91 TPS | Memory
  27 | Caveat output/prompt=4.14%; nontext prompt burden=86%; missing terms:
  Bench, rises; keywords=19)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (27):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-apriel-15-15b-thinker-6bit-mlx),
  [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  +23 more. Example: token: `phrase: "th, th, th, th,..."`.
- _📝 Formatting Issues (2):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-smolvlm2-22b-instruct-mlx).
- _Low-utility outputs (45):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  +41 more. Common weakness: Keywords are not specific or diverse enough.

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
> &#45; Description hint: The tall spire of St John the Evangelist's Church in
> Upper St Leonards, Dorking, England, rises against a blue sky with wispy
> clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil
> green churchyard with mature trees, and a bird is captured in flight near
> the steeple.
> &#45; Keyword hints: Architecture, Bench, Bird, Building, Bush, Church,
> Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic,
> Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn,
> Outdoors
> &#45; Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local
> time). GPS: 51.413600°N, 0.081900°W.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1662.64s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.70s |       1.68s |                                 |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      2.58s |       3.56s |                                 |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                  |      1.14s |       2.12s |                                 |             mlx |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               567 |                    67 |            634 |        2,458 |       346 |         2.4 |            0.69s |      0.71s |       1.58s | missing-sections(keywords), ... |                 |
| `qnguyen3/nanoLLaVA`                                    |               567 |                    82 |            649 |        4,889 |       112 |         4.9 |            1.11s |      2.48s |       3.76s | missing-sections(keywords), ... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               571 |                     9 |            580 |          749 |       305 |         2.2 |            1.23s |      0.85s |       2.26s | ⚠️harness(prompt_template), ... |                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               828 |                   500 |          1,328 |        1,533 |       485 |         1.5 |            1.78s |      2.30s |       4.24s | degeneration, ...               |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,584 |                     8 |          1,592 |        1,066 |      33.4 |          12 |            2.00s |      1.63s |       3.81s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               828 |                   500 |          1,328 |        1,988 |       319 |           3 |            2.20s |      0.58s |       2.96s | degeneration, ...               |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,584 |                     8 |          1,592 |        1,034 |      5.88 |          27 |            3.18s |      2.53s |       5.89s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,396 |                   500 |          3,896 |        2,103 |       177 |         8.7 |            4.73s |      2.55s |       7.47s | missing-sections(title+desc...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,772 |                   500 |          2,272 |        3,672 |       121 |         5.7 |            4.99s |      2.71s |       7.88s | repetitive(phrase: "th, th,...  |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               843 |                   500 |          1,343 |          884 |       111 |          17 |            5.73s |      3.83s |       9.76s | repetitive(phrase: "3- 3- 3...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,584 |                   100 |          1,684 |        3,324 |      19.2 |          11 |            5.97s |      1.78s |       7.93s | missing-sections(title+keyw...  |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               823 |                   500 |          1,323 |          484 |       119 |           6 |            6.14s |      2.89s |       9.21s | repetitive(phrase: "they, t...  |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               672 |                   500 |          1,172 |        1,885 |       129 |         5.7 |            6.43s |      2.08s |       8.69s | missing-sections(title+desc...  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,772 |                   500 |          2,272 |          648 |       124 |         5.6 |            7.09s |      3.86s |      11.12s | repetitive(phrase: "th, th,...  |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,394 |                   500 |          1,894 |        3,366 |      54.5 |         9.5 |            9.83s |      2.57s |      12.59s | missing-sections(title+desc...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,542 |                   447 |          1,989 |          478 |      68.8 |          18 |           10.09s |      3.53s |      13.78s | missing-sections(title), ...    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,394 |                   500 |          1,894 |        1,068 |      55.5 |         9.5 |           10.58s |      1.52s |      12.28s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               831 |                   500 |          1,331 |        1,543 |      47.8 |          17 |           11.25s |      3.98s |      15.42s | repetitive(•), ...              |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,397 |                   500 |          3,897 |        1,013 |      63.1 |          14 |           11.58s |      2.80s |      14.57s | repetitive(phrase: "and for...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,992 |                   500 |          3,492 |          964 |      60.6 |         9.7 |           11.70s |      2.36s |      14.24s | repetitive(phrase: "the bes...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,397 |                   500 |          3,897 |        1,002 |        59 |          14 |           12.17s |      3.81s |      16.16s | repetitive(phrase: "_____ i...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,656 |                   500 |          5,156 |        3,686 |      45.7 |         4.6 |           12.67s |      3.08s |      15.93s | repetitive(phrase: "- do no...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,872 |                   500 |          2,372 |          764 |      41.6 |          60 |           14.99s |      7.02s |      22.35s | degeneration, ...               |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,568 |                   500 |          7,068 |          796 |      68.8 |         8.4 |           15.77s |      5.27s |      21.23s | repetitive(and,), ...           |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,886 |                   500 |          4,386 |        2,183 |      33.6 |          19 |           16.96s |      3.36s |      20.51s | repetitive(phrase: "answer....  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,850 |                   500 |          3,350 |        1,935 |      31.4 |          19 |           17.73s |      5.47s |      23.40s | repetitive(phrase: "and<fak...  |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,568 |                    28 |          6,596 |          375 |      38.2 |          78 |           18.52s |      7.75s |      26.46s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               832 |                   500 |          1,332 |          470 |      29.8 |          19 |           18.83s |      3.76s |      22.78s | missing-sections(title+desc...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,568 |                   500 |          7,068 |          714 |      53.3 |          11 |           18.83s |      2.47s |      21.50s | repetitive(phrase: "the val...  |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,690 |                   500 |          4,190 |        1,394 |      37.9 |          17 |           19.58s |      3.38s |      23.15s | repetitive(phrase: "b. b. b...  |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               843 |                   500 |          1,343 |          475 |      26.5 |          20 |           20.88s |      5.03s |      26.09s | degeneration, ...               |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,886 |                   500 |          4,386 |          935 |      30.1 |          19 |           21.10s |      3.32s |      24.60s | repetitive(phrase: "the giv...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,781 |                   500 |          4,281 |        1,085 |      39.7 |          16 |           22.14s |      4.05s |      26.39s | repetitive(phrase: "the sum...  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,899 |                   500 |          3,399 |          544 |      29.9 |          23 |           22.35s |      4.71s |      27.27s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,542 |                    98 |          1,640 |          943 |      4.56 |          39 |           23.61s |      5.49s |      29.28s | missing-sections(title), ...    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,754 |                   500 |          2,254 |          122 |      50.3 |          41 |           24.77s |      2.85s |      27.80s | repetitive(phrase: "the ima...  |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               538 |                   500 |          1,038 |          281 |      20.4 |          15 |           26.73s |      2.15s |      29.06s | degeneration, ...               |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,851 |                   500 |         17,351 |          921 |      55.9 |          14 |           27.73s |      3.08s |      30.99s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             3,382 |                   140 |          3,522 |          758 |      5.91 |          27 |           28.62s |      4.38s |      33.19s | title-length(14), ...           |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,690 |                   500 |          4,190 |        1,652 |      19.4 |          29 |           31.54s |      4.07s |      35.78s | repetitive(phrase: "essage...   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               832 |                   500 |          1,332 |          483 |      16.9 |          33 |           31.59s |      5.37s |      37.14s | missing-sections(title+desc...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,754 |                   500 |          2,254 |          115 |      29.5 |          48 |           32.69s |      4.44s |      37.30s | repetitive(phrase: "a build...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,866 |                   500 |         17,366 |          315 |       105 |          26 |           58.86s |      5.06s |      64.10s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,866 |                   500 |         17,366 |          318 |      88.7 |          35 |           59.20s |      5.69s |      65.08s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,851 |                   500 |         17,351 |          266 |       195 |         5.1 |           66.54s |      3.24s |      69.95s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,866 |                   500 |         17,366 |          259 |      75.1 |          12 |           72.33s |      1.68s |      74.19s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,866 |                   500 |         17,366 |          284 |      37.6 |          76 |           73.30s |     11.55s |      85.03s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               831 |                   500 |          1,331 |          300 |      7.08 |          65 |           73.73s |      7.82s |      81.73s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,866 |                   500 |         17,366 |          215 |      30.3 |          26 |           95.69s |      4.26s |     100.15s | ⚠️harness(long_context), ...    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               539 |                   500 |          1,039 |          155 |      4.89 |          25 |          106.09s |      4.37s |     110.65s | degeneration, ...               |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,866 |                   500 |         17,366 |          215 |      17.8 |          39 |          107.04s |      5.56s |     112.80s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,866 |                   500 |         17,366 |          161 |      17.9 |          39 |          133.43s |      3.31s |     136.94s | ⚠️harness(long_context), ...    |                 |

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
- `mlx`: `0.32.0.dev20260515+7b7c12407`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.14.0`
- `transformers`: `5.8.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-15 12:25:02 BST_
