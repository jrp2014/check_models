# Model Performance Results

_Generated on 2026-05-15 19:22:25 BST_

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
  context_ignored=46, cutoff=39, repetitive=26, degeneration=14.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (48%;
  35/55 measured model(s)).
- _Phase totals:_ model load=103.82s, local prompt prep=0.16s, upstream
  prefill / first-token=632.71s, post-prefill decode=698.00s, cleanup=5.71s.
- _Generation total:_ 1330.71s across 52 model(s); upstream prefill /
  first-token split available for 52/52 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=11, exception=3, max_tokens=41.
- _Validation overhead:_ 9.58s total (avg 0.17s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 12.17s | Min 0.04s | Max
  76.86s across 52 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (494.0 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.6 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.46s)
- **📊 Average TPS:** 80.8 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1110.5 GB
- **Average peak memory:** 21.4 GB
- **Memory efficiency:** 243 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 3 | 🟡 C: 3 | 🟠 D: 29 | ❌ F: 17

**Average Utility Score:** 37/100

**Existing Metadata Baseline:** ✅ B (75/100)
**Vs Existing Metadata:** Avg Δ -38 | Better: 0, Neutral: 2, Worse: 50

- **Best for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (✅ B, 77/100)
- **Best descriptions:** `mlx-community/MolmoPoint-8B-fp16` (93/100)
- **Best keywording:** `mlx-community/MolmoPoint-8B-fp16` (57/100)
- **Worst for cataloging:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (❌ F, 0/100)

### ⚠️ 46 Models with Low Utility (D/F)

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
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (32/100) - Keywords are not specific or diverse enough
- `mlx-community/MolmoPoint-8B-fp16`: 🟠 D (48/100) - Limited novel information
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (41/100) - Keywords are not specific or diverse enough
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
- **🔄 Repetitive Output (26):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `phrase: "th, th, th, th,..."`)
  - `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (token: `phrase: "the sum the sum..."`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (token: `phrase: "the value, the value,..."`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (token: `and,`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (token: `phrase: "and<fake_token_around_image> a..."`)
  - `mlx-community/InternVL3-14B-8bit` (token: `phrase: "the given text. the..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "answer. answer. answer. answer..."`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (token: `phrase: "and fortunately and fortunatel..."`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` (token: `phrase: "_____ is _____ is..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "the image shows a..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "and a, and a,..."`)
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

- **Generation Tps**: Avg: 80.8 | Min: 4.84 | Max: 494
- **Peak Memory**: Avg: 21 | Min: 1.6 | Max: 78
- **Total Time**: Avg: 27.76s | Min: 1.19s | Max: 105.94s
- **Generation Time**: Avg: 25.59s | Min: 0.55s | Max: 102.67s
- **Model Load Time**: Avg: 1.99s | Min: 0.46s | Max: 7.73s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility D 48/100 | Description 93 | Keywords 57 | Speed 6.17 TPS | Memory
  27 | Caveat output/prompt=4.14%; nontext prompt burden=86%; missing terms:
  Bench, rises; keywords=19)
- _Best descriptions:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility D 48/100 | Description 93 | Keywords 57 | Speed 6.17 TPS | Memory
  27 | Caveat output/prompt=4.14%; nontext prompt burden=86%; missing terms:
  Bench, rises; keywords=19)
- _Best keywording:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility D 48/100 | Description 93 | Keywords 57 | Speed 6.17 TPS | Memory
  27 | Caveat output/prompt=4.14%; nontext prompt burden=86%; missing terms:
  Bench, rises; keywords=19)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 41 | Keywords 0 | Speed 494 TPS | Memory 1.6
  | Caveat hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=character_loop: 'orm' repeated)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 41 | Keywords 0 | Speed 494 TPS | Memory 1.6
  | Caveat hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Bench, Bird, Building, Bush;
  degeneration=character_loop: 'orm' repeated)
- _Best balance:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility D 48/100 | Description 93 | Keywords 57 | Speed 6.17 TPS | Memory
  27 | Caveat output/prompt=4.14%; nontext prompt burden=86%; missing terms:
  Bench, rises; keywords=19)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (26):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-apriel-15-15b-thinker-6bit-mlx),
  [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  +22 more. Example: token: `phrase: "th, th, th, th,..."`.
- _📝 Formatting Issues (2):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-smolvlm2-22b-instruct-mlx).
- _Low-utility outputs (46):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  +42 more. Common weakness: Keywords are not specific or diverse enough.

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

_Overall runtime:_ 1453.75s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                  |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:--------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.12s |       1.13s |                                 |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.19s |       1.15s |                                 |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                  |      0.19s |       1.16s |                                 |             mlx |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               567 |                    67 |            634 |         6369 |       355 |         2.8 |            0.55s |      0.46s |       1.19s | missing-sections(keywords), ... |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               571 |                     9 |            580 |         5458 |       295 |         2.2 |            0.56s |      0.59s |       1.33s | ⚠️harness(prompt_template), ... |                 |
| `qnguyen3/nanoLLaVA`                                    |               567 |                    82 |            649 |         4976 |       112 |         4.9 |            1.11s |      0.53s |       1.81s | missing-sections(keywords), ... |                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               828 |                   500 |          1,328 |        19301 |       494 |         1.6 |            1.26s |      0.47s |       1.92s | degeneration, ...               |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,584 |                     8 |          1,592 |         1371 |      33.8 |          12 |            1.66s |      1.58s |       3.42s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               828 |                   500 |          1,328 |         6515 |       330 |           3 |            1.86s |      0.51s |       2.55s | degeneration, ...               |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,584 |                     8 |          1,592 |         1103 |         6 |          27 |            3.05s |      2.51s |       5.74s | ⚠️harness(prompt_template), ... |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,396 |                   500 |          3,896 |         2296 |       183 |         8.7 |            4.51s |      0.91s |       5.61s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               823 |                   500 |          1,323 |         2430 |       121 |           6 |            4.72s |      1.51s |       6.41s | repetitive(phrase: "they, t...  |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,772 |                   500 |          2,272 |         3758 |       127 |         5.6 |            4.73s |      0.67s |       5.57s | repetitive(phrase: "th, th,...  |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,772 |                   500 |          2,272 |         3744 |       126 |         5.7 |            4.80s |      0.61s |       5.58s | repetitive(phrase: "th, th,...  |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               843 |                   500 |          1,343 |         1429 |       109 |          17 |            5.44s |      2.49s |       8.13s | repetitive(phrase: "3- 3- 3...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,584 |                   100 |          1,684 |         3327 |      19.3 |          11 |            5.93s |      1.42s |       7.53s | missing-sections(title+keyw...  |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               672 |                   500 |          1,172 |         1901 |       132 |         5.7 |            6.19s |      0.58s |       6.95s | missing-sections(title+desc...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,542 |                   447 |          1,989 |         1630 |      69.3 |          18 |            7.74s |      2.00s |       9.92s | missing-sections(title), ...    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,394 |                   500 |          1,894 |         3329 |      57.3 |         9.5 |            9.40s |      0.94s |      10.52s | missing-sections(title+desc...  |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,394 |                   500 |          1,894 |         3362 |      56.1 |         9.5 |            9.59s |      1.01s |      10.78s | missing-sections(title+desc...  |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               831 |                   500 |          1,331 |         1772 |        49 |          17 |           10.93s |      2.30s |      13.42s | repetitive(•), ...              |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,397 |                   500 |          3,897 |         1060 |      64.8 |          14 |           11.22s |      1.32s |      12.73s | repetitive(phrase: "and for...  |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,992 |                   500 |          3,492 |         1027 |      61.4 |         9.7 |           11.39s |      0.92s |      12.49s | repetitive(phrase: "the bes...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,397 |                   500 |          3,897 |         1066 |      62.7 |          14 |           11.46s |      1.34s |      13.00s | repetitive(phrase: "_____ i...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,656 |                   500 |          5,156 |         3833 |      48.1 |         4.6 |           12.11s |      1.13s |      13.43s | repetitive(phrase: "- do no...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,872 |                   500 |          2,372 |         1202 |        43 |          60 |           13.70s |      4.96s |      19.00s | degeneration, ...               |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,568 |                   500 |          7,068 |          850 |      71.4 |         8.4 |           14.99s |      1.29s |      16.46s | repetitive(and,), ...           |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,568 |                    28 |          6,596 |          443 |      39.2 |          78 |           15.78s |      5.47s |      21.44s | missing-sections(title+desc...  |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,886 |                   500 |          4,386 |         2269 |      33.6 |          19 |           16.89s |      1.69s |      18.76s | repetitive(phrase: "answer....  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,568 |                   500 |          7,068 |          835 |      55.1 |          11 |           17.18s |      1.42s |      18.78s | repetitive(phrase: "the val...  |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,850 |                   500 |          3,350 |         2001 |        32 |          19 |           17.40s |      1.88s |      19.46s | repetitive(phrase: "and<fak...  |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               832 |                   500 |          1,332 |          524 |      30.3 |          19 |           18.32s |      2.31s |      20.82s | missing-sections(title+desc...  |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,690 |                   500 |          4,190 |         1399 |      38.6 |          17 |           19.12s |      1.67s |      20.97s | repetitive(phrase: "b. b. b...  |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,886 |                   500 |          4,386 |         1101 |      31.3 |          19 |           19.83s |      1.83s |      21.85s | repetitive(phrase: "the giv...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,754 |                    36 |          1,790 |         93.4 |        48 |          41 |           19.98s |      1.21s |      21.37s | missing-sections(title+desc...  |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,781 |                   500 |          4,281 |         1122 |      40.7 |          16 |           21.29s |      1.60s |      23.07s | repetitive(phrase: "the sum...  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,899 |                   500 |          3,399 |          580 |      31.2 |          23 |           21.31s |      2.07s |      23.58s | ⚠️harness(encoding), ...        |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,542 |                    98 |          1,640 |         1070 |      4.84 |          39 |           22.16s |      3.24s |      25.59s | missing-sections(title), ...    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               843 |                   500 |          1,343 |          409 |      25.1 |          20 |           22.26s |      2.57s |      25.02s | degeneration, ...               |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               538 |                   500 |          1,038 |          333 |        21 |          15 |           25.66s |      1.54s |      27.37s | degeneration, ...               |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             3,382 |                   140 |          3,522 |         1282 |      6.17 |          27 |           25.77s |      2.22s |      28.17s | title-length(14), ...           |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,851 |                   500 |         17,351 |          919 |      57.3 |          14 |           27.52s |      1.12s |      28.81s | ⚠️harness(stop_token), ...      |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               832 |                   500 |          1,332 |          471 |      17.5 |          33 |           30.66s |      3.37s |      34.21s | missing-sections(title+desc...  |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,690 |                   500 |          4,190 |         1626 |      19.8 |          29 |           31.05s |      2.58s |      33.80s | repetitive(phrase: "essage...   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,754 |                   500 |          2,254 |         92.9 |      30.3 |          48 |           35.78s |      1.79s |      37.76s | repetitive(phrase: "the ima...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,866 |                   500 |         17,366 |          335 |       100 |          26 |           55.88s |      2.60s |      58.66s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,866 |                   500 |         17,366 |          337 |      84.9 |          35 |           56.46s |      3.11s |      59.75s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,866 |                   500 |         17,366 |          325 |      90.8 |          12 |           57.85s |      1.33s |      59.36s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,866 |                   500 |         17,366 |          326 |      65.1 |          76 |           59.95s |      7.73s |      67.87s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,851 |                   500 |         17,351 |          280 |       192 |         5.1 |           63.24s |      0.53s |      63.94s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               831 |                   500 |          1,331 |          333 |      7.29 |          65 |           71.39s |      5.86s |      77.44s | missing-sections(title+desc...  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,866 |                   500 |         17,366 |          219 |      30.3 |          26 |           93.89s |      2.15s |      96.24s | ⚠️harness(long_context), ...    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               539 |                   500 |          1,039 |          281 |      5.06 |          25 |          100.97s |      2.20s |     103.36s | degeneration, ...               |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,866 |                   500 |         17,366 |          229 |      18.2 |          39 |          101.54s |      3.10s |     104.84s | ⚠️harness(long_context), ...    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,866 |                   500 |         17,366 |          226 |      18.2 |          39 |          102.67s |      3.08s |     105.94s | ⚠️harness(long_context), ...    |                 |

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
- `transformers`: `5.8.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-15 19:22:25 BST_
