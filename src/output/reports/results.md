# Model Performance Results

_Generated on 2026-05-16 23:30:18 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, mlx-lm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=14, clean outputs=0/52.
- _Useful now:_ none (no clean A/B shortlist for this run).
- _Review watchlist:_ 52 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Vs existing metadata:_ better=4, neutral=1, worse=47 (baseline B 67/100).
- _Quality signal frequency:_ missing_sections=48, context_ignored=47,
  trusted_hint_ignored=47, cutoff=39, repetitive=23, degeneration=16.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (49%;
  37/55 measured model(s)).
- _Phase totals:_ model load=121.49s, local prompt prep=0.17s, upstream
  prefill / first-token=686.51s, post-prefill decode=775.37s, cleanup=6.06s.
- _Generation total:_ 1461.87s across 52 model(s); upstream prefill /
  first-token split available for 52/52 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=10, exception=3, max_tokens=42.
- _Validation overhead:_ 18.76s total (avg 0.34s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 13.20s | Min 0.05s | Max
  88.87s across 52 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (496.7 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.5 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.35s)
- **📊 Average TPS:** 78.1 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1102.1 GB
- **Average peak memory:** 21.2 GB
- **Memory efficiency:** 236 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 1 | ✅ B: 4 | 🟡 C: 2 | 🟠 D: 27 | ❌ F: 18

**Average Utility Score:** 35/100

**Existing Metadata Baseline:** ✅ B (67/100)
**Vs Existing Metadata:** Avg Δ -32 | Better: 4, Neutral: 1, Worse: 47

- **Best for cataloging:** `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (🏆 A, 81/100)
- **Best descriptions:** `mlx-community/MolmoPoint-8B-fp16` (77/100)
- **Best keywording:** `mlx-community/MolmoPoint-8B-fp16` (79/100)
- **Worst for cataloging:** `meta-llama/Llama-3.2-11B-Vision-Instruct` (❌ F, 0/100)

### ⚠️ 45 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (5/100) - Output too short to be useful
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (0/100) - Output too short to be useful
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (39/100) - Keywords are not specific or diverse enough
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/GLM-4.6V-Flash-6bit`: 🟠 D (40/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-Flash-mxfp4`: 🟠 D (39/100) - Lacks visual description of image
- `mlx-community/GLM-4.6V-nvfp4`: 🟠 D (48/100) - Keywords are not specific or diverse enough
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (18/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-14B-8bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🟠 D (48/100) - Keywords are not specific or diverse enough
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🟠 D (48/100) - Keywords are not specific or diverse enough
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-27B-4bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen3.5-27B-mxfp8`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen3.5-35B-A3B-4bit`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-6bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen3.5-35B-A3B-bf16`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/Qwen3.5-9B-MLX-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/X-Reasoner-7B-8bit`: ❌ F (33/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (43/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E4B-it-bf16`: 🟠 D (40/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-26b-a4b-it-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/gemma-4-31b-bf16`: 🟠 D (47/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-31b-it-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (37/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (11/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (33/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (16/100) - Output lacks detail
- `mlx-community/paligemma2-3b-pt-896-4bit`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-8bit`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-bf16`: 🟠 D (46/100) - Keywords are not specific or diverse enough
- `qnguyen3/nanoLLaVA`: 🟠 D (40/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
- **🔄 Repetitive Output (23):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `phrase: "th, th, th, th,..."`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (token: `phrase: "(i:1,000,000) (i:1,000,000) (i..."`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (token: `phrase: "(b) (b) (b) (b)..."`)
  - `mlx-community/GLM-4.6V-nvfp4` (token: `phrase: "q q q q..."`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (token: `phrase: "and<fake_token_around_image> a..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "the problem, the problem,..."`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` (token: `phrase: "$a is $a is..."`)
  - `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (token: `phrase: ". . . ...."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "a house. the image..."`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (token: `phrase: "a house. the image..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "it is. it is...."`)
  - `mlx-community/Qwen3.5-35B-A3B-4bit` (token: `24`)
  - `mlx-community/Qwen3.5-9B-MLX-4bit` (token: `phrase: "american, t. 0the american,..."`)
  - `mlx-community/Qwen3.6-27B-mxfp8` (token: `phrase: "``` 2500000000 ``` ```..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `phrase: "th, th, th, th,..."`)
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (token: `phrase: "neurotransmit outbre neurotran..."`)
  - `mlx-community/X-Reasoner-7B-8bit` (token: `phrase: "1.<|endoftext|>3 1.<|endoftext..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "the learning the learning..."`)
  - `mlx-community/gemma-3n-E4B-it-bf16` (token: `phrase: "aesthetic patterns, and the..."`)
  - `mlx-community/llava-v1.6-mistral-7b-8bit` (token: `phrase: "the best the best..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "text after the label...."`)
  - `mlx-community/pixtral-12b-8bit` (token: `phrase: "k年発売 k年発売 k年発売 k年発売..."`)
  - `mlx-community/pixtral-12b-bf16` (token: `phrase: "k年発売 k年発売 k年発売 k年発売..."`)
- **👻 Hallucinations (1):**
  - `mlx-community/gemma-3n-E2B-4bit`
- **📝 Formatting Issues (3):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Qwen3.6-27B-mxfp8`
  - `mlx-community/gemma-3-27b-it-qat-8bit`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 78.1 | Min: 4.74 | Max: 497
- **Peak Memory**: Avg: 21 | Min: 1.5 | Max: 79
- **Total Time**: Avg: 30.79s | Min: 1.42s | Max: 122.46s
- **Generation Time**: Avg: 28.11s | Min: 0.56s | Max: 118.70s
- **Model Load Time**: Avg: 2.32s | Min: 0.35s | Max: 12.68s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility B 66/100 | Description 77 | Keywords 79 | Speed 5.98 TPS | Memory
  26 | Caveat output/prompt=6.70%; nontext prompt burden=87%; nonvisual
  metadata reused)
- _Best descriptions:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility B 66/100 | Description 77 | Keywords 79 | Speed 5.98 TPS | Memory
  26 | Caveat output/prompt=6.70%; nontext prompt burden=87%; nonvisual
  metadata reused)
- _Best keywording:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility B 66/100 | Description 77 | Keywords 79 | Speed 5.98 TPS | Memory
  26 | Caveat output/prompt=6.70%; nontext prompt burden=87%; nonvisual
  metadata reused)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 5/100 | Description 41 | Keywords 0 | Speed 497 TPS | Memory 1.5
  | Caveat hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=character_loop: 'ore' repeated)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 5/100 | Description 41 | Keywords 0 | Speed 497 TPS | Memory 1.5
  | Caveat hit token cap (500); missing sections: title, description,
  keywords; missing terms: scenic, view, looking, through, open;
  degeneration=character_loop: 'ore' repeated)
- _Best balance:_ [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16)
  (Utility B 66/100 | Description 77 | Keywords 79 | Speed 5.98 TPS | Memory
  26 | Caveat output/prompt=6.70%; nontext prompt burden=87%; nonvisual
  metadata reused)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (23):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  +19 more. Example: token: `phrase: "th, th, th, th,..."`.
- _👻 Hallucinations (1):_ [`mlx-community/gemma-3n-E2B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit).
- _📝 Formatting Issues (3):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/Qwen3.6-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen36-27b-mxfp8),
  [`mlx-community/gemma-3-27b-it-qat-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-8bit).
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
> &#45; Description hint: A scenic view looking through open wrought-iron gates
> down a paved driveway lined with wooden fences, lush green trees, and
> blooming flowers, leading to the grand entrance of a historic gothic-style
> stone abbey.
> &#45; Capture metadata: Taken on 2026-05-16 14:37:59 BST (at 14:37:59 local
> time). GPS: 50.811559°N, 1.777085°W.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1612.18s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.20s |       1.32s |                                    |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.38s |       1.51s |                                    |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                  |      0.14s |       1.28s |                                    |             mlx |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               745 |                     6 |            751 |         6258 |       302 |           3 |            0.56s |      0.54s |       1.42s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |               487 |                   128 |            615 |         5071 |       354 |         2.5 |            0.97s |      0.48s |       1.79s | fabrication, ...                   |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |               491 |                     3 |            494 |         4228 |       200 |         2.1 |            1.23s |      0.67s |       2.27s | ⚠️harness(prompt_template), ...    |                 |
| `qnguyen3/nanoLLaVA`                                    |               487 |                    81 |            568 |         4445 |       113 |         4.7 |            1.34s |      0.52s |       2.19s | missing-sections(keywords), ...    |                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               745 |                   500 |          1,245 |        15555 |       497 |         1.5 |            1.44s |      0.35s |       2.11s | degeneration, ...                  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,513 |                    11 |          1,524 |         3277 |      21.1 |          11 |            1.51s |      1.41s |       3.26s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,513 |                    15 |          1,528 |         1368 |        33 |          12 |            2.08s |      1.63s |       4.05s | missing-sections(title+desc...     |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,270 |                     6 |          2,276 |          969 |      37.3 |          18 |            3.08s |      1.74s |       5.17s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             3,082 |                   500 |          3,582 |         2325 |       185 |         7.8 |            4.61s |      0.94s |       5.92s | repetitive(phrase: ". . . ....     |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               748 |                   500 |          1,248 |         2292 |       119 |         6.1 |            5.00s |      1.42s |       6.76s | repetitive(phrase: "the lea...     |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,695 |                   500 |          2,195 |         3527 |       124 |         5.6 |            5.11s |      0.72s |       6.15s | repetitive(phrase: "th, th,...     |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,695 |                   500 |          2,195 |         3441 |       122 |         5.6 |            5.26s |      0.59s |       6.20s | repetitive(phrase: "th, th,...     |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               762 |                   500 |          1,262 |         1319 |       112 |          17 |            5.57s |      2.34s |       8.27s | degeneration, ...                  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,513 |                    21 |          1,534 |         1069 |      5.46 |          27 |            5.79s |      2.40s |       8.54s | missing-sections(title+desc...     |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |               596 |                   500 |          1,096 |         1691 |       131 |         5.7 |            5.90s |      0.57s |       6.81s | repetitive(phrase: "neurotr...     |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,479 |                   500 |          1,979 |         1558 |      70.5 |          18 |            8.61s |      2.04s |      10.96s | missing-sections(title+desc...     |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |             1,305 |                   500 |          1,805 |         3251 |      56.1 |         9.5 |            9.77s |      0.89s |      10.99s | missing-sections(title+desc...     |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |             1,305 |                   500 |          1,805 |         3041 |        54 |         9.5 |           10.15s |      0.91s |      11.40s | missing-sections(title+desc...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             3,083 |                   500 |          3,583 |         1081 |      66.5 |          13 |           10.93s |      1.31s |      12.59s | degeneration, ...                  |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               756 |                   500 |          1,256 |         1676 |      48.4 |          17 |           11.25s |      2.24s |      13.84s | repetitive(phrase: "aesthet...     |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,690 |                   500 |          3,190 |          974 |      61.7 |         9.7 |           11.51s |      0.92s |      12.77s | repetitive(phrase: "the bes...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             3,083 |                   500 |          3,583 |          953 |      62.2 |          13 |           11.84s |      1.38s |      13.56s | repetitive(phrase: "$a is $...     |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,585 |                   500 |          5,085 |         3806 |      42.4 |         4.6 |           13.75s |      1.15s |      15.26s | repetitive(phrase: "text af...     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             3,364 |                   500 |          3,864 |         1138 |      40.4 |          15 |           15.87s |      1.70s |      17.93s | missing-sections(title+desc...     |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,270 |                   500 |          2,770 |         2178 |      34.3 |          18 |           16.16s |      1.70s |      18.20s | repetitive(phrase: "the pro...     |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,540 |                   500 |          7,040 |          791 |      67.4 |         8.4 |           16.16s |      1.34s |      17.86s | repetitive(phrase: "(b) (b)...     |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,769 |                   500 |          3,269 |         1938 |      32.2 |          19 |           17.62s |      1.97s |      19.96s | repetitive(phrase: "and<fak...     |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,540 |                   500 |          7,040 |          706 |        53 |          11 |           19.18s |      1.42s |      20.99s | repetitive(phrase: "(i:1,00...     |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,808 |                   500 |          2,308 |          293 |      40.6 |          60 |           19.35s |     11.14s |      31.05s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               757 |                   500 |          1,257 |          515 |      28.7 |          19 |           19.37s |      2.31s |      22.03s | missing-sections(title+desc...     |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               762 |                   500 |          1,262 |          486 |      26.9 |          20 |           20.66s |      2.53s |      23.54s | degeneration, ...                  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,586 |                   500 |          3,086 |          613 |      30.1 |          22 |           21.37s |      2.06s |      23.80s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/pixtral-12b-8bit`                        |             3,273 |                   500 |          3,773 |         1433 |      38.6 |          16 |           23.23s |      1.68s |      25.27s | repetitive(phrase: "k年発売 k年... |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |               458 |                   500 |            958 |          283 |      20.3 |          15 |           26.70s |      1.57s |      28.61s | degeneration, ...                  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,666 |                   500 |          2,166 |          100 |      52.1 |          41 |           27.07s |      1.22s |      28.63s | repetitive(phrase: "a house...     |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,479 |                   122 |          1,601 |         1015 |      4.74 |          39 |           27.89s |      3.26s |      31.49s | missing-sections(title), ...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,715 |                   500 |         17,215 |          871 |      54.7 |          13 |           29.09s |      1.11s |      30.54s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               757 |                   500 |          1,257 |          466 |        17 |          33 |           31.53s |      3.39s |      35.26s | missing-sections(title+desc...     |                 |
| `mlx-community/pixtral-12b-bf16`                        |             3,273 |                   500 |          3,773 |         1664 |      20.1 |          28 |           33.46s |      2.52s |      36.32s | repetitive(phrase: "k年発売 k年... |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,666 |                   500 |          2,166 |         98.2 |        30 |          48 |           34.46s |      1.72s |      36.53s | repetitive(phrase: "a house...     |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             3,283 |                   220 |          3,503 |         1184 |      5.98 |          26 |           40.31s |      2.20s |      42.85s | fabrication, title-length(13), ... |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,540 |                   500 |          7,040 |          458 |      35.1 |          79 |           53.90s |      9.97s |      64.21s | repetitive(phrase: "q q q q...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,730 |                   500 |         17,230 |          313 |       103 |          26 |           59.08s |      2.50s |      61.94s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,730 |                   500 |         17,230 |          307 |      86.9 |          35 |           61.12s |      3.14s |      64.60s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,715 |                   500 |         17,215 |          287 |       201 |         5.1 |           61.42s |      0.52s |      62.28s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,730 |                   500 |         17,230 |          274 |      82.8 |          12 |           67.95s |      1.35s |      69.65s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,730 |                   500 |         17,230 |          271 |      64.3 |          76 |           70.26s |     12.68s |      83.29s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               750 |                   500 |          1,250 |          167 |       7.2 |          64 |           74.47s |      7.89s |      82.70s | degeneration, ...                  |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,730 |                   500 |         17,230 |          208 |      26.8 |          26 |           99.91s |      2.12s |     102.41s | ⚠️harness(long_context), ...       |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |               459 |                   500 |            959 |          216 |       4.9 |          25 |          104.56s |      2.22s |     107.12s | degeneration, ...                  |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,730 |                   500 |         17,230 |          197 |      17.8 |          39 |          113.76s |      3.10s |     117.24s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,730 |                   500 |         17,230 |          188 |      17.3 |          39 |          118.70s |      3.30s |     122.46s | ⚠️harness(stop_token), ...         |                 |

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

- `numpy`: `2.4.5`
- `mlx`: `0.32.0.dev20260516+7b7c1240`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.15.0`
- `transformers`: `5.8.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-16 23:30:18 BST_
