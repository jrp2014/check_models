# Model Selection Brief

Generated on: 2026-07-19 02:41:57 BST

- Evaluation lane: assisted
- Metadata exposed to prompt: yes
- Semantic rankings: grounded (metadata-assisted visual verification)
- Policy: reliability-gated assisted enrichment
- Evidence scope: 1 image, 1 current run
- Primary use cases: brief captions; structured title/description/keywords
- Scope: ranked shortlist, not the complete run; complete per-model outputs and diagnostics are in `model_gallery.md`.

## Evidence Links

- _Output evidence:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Maintainer diagnostics:_ [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)

### Reliability-gated Current-run View

Policy: reliability-gated; crashes and integration warnings remain visible but
cannot be named as usable winners.

| Model                                                 | Task outcome            | Compatibility       | Recommendation eligibility                    | Prompt burden   | Gen TPS   | Peak GB                                          |
|-------------------------------------------------------|-------------------------|---------------------|-----------------------------------------------|-----------------|-----------|--------------------------------------------------|
| mlx-community/Step-3.7-Flash-oQ2e                     | Task outcome: crashed   | crashed             | excluded: current run crashed                 | unknown         | -         | -                                                |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 508       | 1.5 GB (1.3% of 108 GB recommended working set)  |
| mlx-community/nanoLLaVA-1.5-4bit                      | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 354       | 2.8 GB (2.4% of 108 GB recommended working set)  |
| mlx-community/LFM2-VL-1.6B-8bit                       | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 328       | 3.0 GB (2.58% of 108 GB recommended working set) |
| mlx-community/MiniCPM-V-4.6-8bit                      | Task outcome: completed | integration-warning | excluded: integration warning requires review | normal          | 274       | 4.1 GB (3.58% of 108 GB recommended working set) |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 124       | 17 GB (14.3% of 108 GB recommended working set)  |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 187       | 4.1 GB (3.56% of 108 GB recommended working set) |
| qnguyen3/nanoLLaVA                                    | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 112       | 4.7 GB (4.1% of 108 GB recommended working set)  |
| mlx-community/FastVLM-0.5B-bf16                       | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 344       | 2.2 GB (1.88% of 108 GB recommended working set) |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 134       | 5.5 GB (4.75% of 108 GB recommended working set) |
| HuggingFaceTB/SmolVLM-Instruct                        | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 128       | 5.7 GB (4.95% of 108 GB recommended working set) |
| mlx-community/SmolVLM-Instruct-bf16                   | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 126       | 5.7 GB (4.97% of 108 GB recommended working set) |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | Task outcome: completed | clean               | excluded: current review says avoid           | mixed           | 185       | 7.8 GB (6.76% of 108 GB recommended working set) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 75.7      | 28 GB (24.6% of 108 GB recommended working set)  |
| mlx-community/paligemma2-10b-ft-docci-448-6bit        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 30.9      | 12 GB (10.5% of 108 GB recommended working set)  |
| mlx-community/diffusiongemma-26B-A4B-it-8bit          | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 60.8      | 29 GB (25.3% of 108 GB recommended working set)  |
| mlx-community/Idefics3-8B-Llama3-bf16                 | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 33.1      | 19 GB (16.6% of 108 GB recommended working set)  |
| microsoft/Phi-3.5-vision-instruct                     | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 56.2      | 9.6 GB (8.28% of 108 GB recommended working set) |
| mlx-community/Phi-3.5-vision-instruct-bf16            | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 56.1      | 9.6 GB (8.29% of 108 GB recommended working set) |
| mlx-community/InternVL3-8B-bf16                       | Task outcome: completed | clean               | eligible                                      | normal          | 34.2      | 18 GB (15.3% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | Task outcome: completed | clean               | excluded: current review says avoid           | mixed           | 66.6      | 13 GB (11.3% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b-8bit                        | Task outcome: completed | clean               | excluded: current review says avoid           | mixed           | 39.4      | 16 GB (13.6% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | Task outcome: completed | clean               | excluded: current review says avoid           | mixed           | 63.1      | 13 GB (11.7% of 108 GB recommended working set)  |
| mlx-community/gemma-3n-E2B-4bit                       | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 125       | 6.1 GB (5.25% of 108 GB recommended working set) |
| mlx-community/llava-v1.6-mistral-7b-8bit              | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 62.0      | 9.7 GB (8.41% of 108 GB recommended working set) |
| mlx-community/InternVL3-14B-8bit                      | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 32.3      | 18 GB (15.7% of 108 GB recommended working set)  |
| mlx-community/paligemma2-10b-ft-docci-448-bf16        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 5.56      | 27 GB (23.3% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it-qat-4bit                 | Task outcome: completed | clean               | eligible                                      | normal          | 31.3      | 18 GB (15.7% of 108 GB recommended working set)  |
| mlx-community/gemma-4-31b-it-4bit                     | Task outcome: completed | clean               | eligible                                      | normal          | 26.5      | 20 GB (17.6% of 108 GB recommended working set)  |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit                 | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 71.5      | 18 GB (15.6% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b-bf16                        | Task outcome: completed | clean               | excluded: current review says avoid           | mixed           | 20.0      | 28 GB (23.9% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it-qat-8bit                 | Task outcome: completed | clean               | eligible                                      | normal          | 17.8      | 32 GB (27.4% of 108 GB recommended working set)  |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | Task outcome: completed | clean               | eligible                                      | normal          | 30.1      | 23 GB (19.9% of 108 GB recommended working set)  |
| mlx-community/GLM-4.6V-Flash-6bit                     | Task outcome: completed | clean               | excluded: current review says caveat          | visual_input    | 57.9      | 11 GB (9.58% of 108 GB recommended working set)  |
| mlx-community/gemma-3n-E4B-it-bf16                    | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 48.5      | 17 GB (14.9% of 108 GB recommended working set)  |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 21.9      | 15 GB (13.1% of 108 GB recommended working set)  |
| mlx-community/Kimi-VL-A3B-Thinking-8bit               | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 66.2      | 22 GB (19% of 108 GB recommended working set)    |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | Task outcome: completed | clean               | excluded: current review says avoid           | visual_input    | 80.1      | 8.4 GB (7.29% of 108 GB recommended working set) |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 5.08      | 25 GB (21.9% of 108 GB recommended working set)  |
| mlx-community/paligemma2-3b-pt-896-4bit               | Task outcome: completed | clean               | excluded: current review says avoid           | visual_input    | 43.4      | 4.6 GB (3.96% of 108 GB recommended working set) |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | Task outcome: completed | clean               | excluded: current review says avoid           | mixed           | 42.5      | 15 GB (13% of 108 GB recommended working set)    |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 46.5      | 60 GB (52.1% of 108 GB recommended working set)  |
| mlx-community/gemma-4-31b-bf16                        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 7.39      | 65 GB (56% of 108 GB recommended working set)    |
| mlx-community/GLM-4.6V-nvfp4                          | Task outcome: completed | clean               | excluded: current review says caveat          | visual_input    | 43.1      | 78 GB (67.3% of 108 GB recommended working set)  |
| Qwen/Qwen3-VL-2B-Instruct                             | Task outcome: completed | integration-warning | excluded: integration warning requires review | mixed           | 92.9      | 8.6 GB (7.43% of 108 GB recommended working set) |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review | mixed           | 92.8      | 8.6 GB (7.44% of 108 GB recommended working set) |
| mlx-community/MolmoPoint-8B-fp16                      | Task outcome: completed | clean               | excluded: current review says caveat          | visual_input    | 5.96      | 27 GB (23.1% of 108 GB recommended working set)  |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review | mixed           | 92.7      | 8.6 GB (7.44% of 108 GB recommended working set) |
| mlx-community/Molmo-7B-D-0924-8bit                    | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 51.3      | 41 GB (35.2% of 108 GB recommended working set)  |
| mlx-community/Molmo-7B-D-0924-bf16                    | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 29.5      | 48 GB (41.4% of 108 GB recommended working set)  |
| mlx-community/X-Reasoner-7B-8bit                      | Task outcome: completed | integration-warning | excluded: integration warning requires review | mixed           | 56.1      | 13 GB (11.7% of 108 GB recommended working set)  |
| mlx-community/paligemma2-3b-ft-docci-448-bf16         | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 19.0      | 11 GB (9.69% of 108 GB recommended working set)  |
| mlx-community/Ornith-1.0-35B-bf16                     | Task outcome: completed | clean               | excluded: current review says avoid           | mixed           | 63.3      | 76 GB (65.8% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | Task outcome: completed | clean               | eligible                                      | mixed           | 92.7      | 11 GB (9.94% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | Task outcome: completed | clean               | excluded: current review says avoid           | mixed           | 65.0      | 76 GB (65.8% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | Task outcome: completed | clean               | eligible                                      | mixed           | 90.1      | 26 GB (22.6% of 108 GB recommended working set)  |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | Task outcome: completed | integration-warning | excluded: integration warning requires review | mixed           | 49,588    | 5.1 GB (4.39% of 108 GB recommended working set) |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | Task outcome: completed | clean               | excluded: current review says avoid           | mixed           | 93.6      | 35 GB (30.1% of 108 GB recommended working set)  |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16          | Task outcome: completed | clean               | excluded: current review says caveat          | normal          | 4.68      | 40 GB (34.6% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B-4bit                        | Task outcome: completed | clean               | eligible                                      | mixed           | 30.7      | 26 GB (22.4% of 108 GB recommended working set)  |
| mlx-community/Qwen3.6-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                      | mixed           | 17.8      | 38 GB (33.3% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                      | mixed           | 17.4      | 38 GB (33.3% of 108 GB recommended working set)  |

## Quick Chooser

Practical current-run buckets for model users. These are triage signals, not grounded visual-quality claims.

### Best under 4 GB

Policy: memory-aware (reliability-gated assisted enrichment; budget 4 GB).
Evidence scope: 1 image, 1 current run.

- No clean current-run candidates fit under this memory budget.

### Best under 8 GB

Policy: memory-aware (reliability-gated assisted enrichment; budget 8 GB).
Evidence scope: 1 image, 1 current run.

- No clean current-run candidates fit under this memory budget.

### Fastest usable

Policy: efficiency-aware Pareto frontier (reliability-gated).
Evidence scope: 1 image, 1 current run.

| Model                                   | Peak GB                                         |   Gen TPS |   Usefulness | Status        | Evidence                                                                                         |
|-----------------------------------------|-------------------------------------------------|-----------|--------------|---------------|--------------------------------------------------------------------------------------------------|
| `mlx-community/InternVL3-8B-bf16`       | 18 GB (15.3% of 108 GB recommended working set) |      34.2 |           85 | `recommended` | [low-draft-improvement] Title: The Walkie-Talkie Building, London, England ...                   |
| `mlx-community/gemma-3-27b-it-qat-4bit` | 18 GB (15.7% of 108 GB recommended working set) |      31.3 |           93 | `recommended` | [low-draft-improvement] Title: The Fenchurch Building illuminated at night ...                   |
| `mlx-community/gemma-4-31b-it-4bit`     | 20 GB (17.6% of 108 GB recommended working set) |      26.5 |           87 | `recommended` | [low-draft-improvement] Title: The Fenchurch Building at night in London, England Description... |

### Quality if memory allows

Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                | Peak GB                                         |   Gen TPS |   Usefulness | Status        | Evidence                                                                                         |
|--------------------------------------|-------------------------------------------------|-----------|--------------|---------------|--------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-31b-it-4bit`  | 20 GB (17.6% of 108 GB recommended working set) |      26.5 |           87 | `recommended` | [low-draft-improvement] Title: The Fenchurch Building at night in London, England Description... |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | 26 GB (22.6% of 108 GB recommended working set) |      90.1 |           93 | `recommended` | Title: The Fenchurch Building at night Description: A low-angle view captures the distinctive... |
| `mlx-community/Qwen3.6-27B-mxfp8`    | 38 GB (33.3% of 108 GB recommended working set) |      17.8 |          100 | `recommended` | Title: Night view of 20 Fenchurch Street, London ...                                             |

### Current failures / avoid

Policy: reliability-gated exclusion evidence.
Evidence scope: 1 image, 1 current run.

| Model                               | Peak GB                                          | Gen TPS   | Usefulness   | Status   | Evidence                                                                                                                                                                                                                                |
|-------------------------------------|--------------------------------------------------|-----------|--------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Step-3.7-Flash-oQ2e` | -                                                | -         | -            | `avoid`  | processor error \| model config processor load processor \| Task outcome: crashed \| ValueError: Loaded processor has no image_processor; expected multimodal processor. \| current run crashed                                         |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`  | 1.5 GB (1.3% of 108 GB recommended working set)  | 508       | 80           | `caveat` | missing terms: Cars, Cityscape, Commuting, Fenchurch Street, Nightscape \| current review says caveat                                                                                                                                   |
| `mlx-community/nanoLLaVA-1.5-4bit`  | 2.8 GB (2.4% of 108 GB recommended working set)  | 354       | 76           | `caveat` | missing terms: Urban, Urban landscape \| low-draft-improvement \| current review says caveat                                                                                                                                            |
| `mlx-community/LFM2-VL-1.6B-8bit`   | 3.0 GB (2.58% of 108 GB recommended working set) | 328       | 80           | `caveat` | missing terms: formally \| keywords=20 \| low-draft-improvement \| current review says caveat                                                                                                                                           |
| `mlx-community/MiniCPM-V-4.6-8bit`  | 4.1 GB (3.58% of 108 GB recommended working set) | 274       | 87           | `avoid`  | Special control token &lt;/think&gt; appeared in generated text. \| missing terms: Cars, Commuting, Fenchurch Street, London, Nightscape \| reasoning leak \| text-sanity=gibberish(token_noise) \| integration warning requires review |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.
Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                                   |   Hygiene |   Usefulness |   Gen TPS | Peak GB                                         | Verdict       | Caption Preview                                                                                                                                                                      | Caveat                                                                                                                        |
|---------------------------------------------------------|-----------|--------------|-----------|-------------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-31b-it-4bit`                     |       100 |           87 |      26.5 | 20 GB (17.6% of 108 GB recommended working set) | `recommended` | [low-draft-improvement] Title: The Fenchurch Building at night in London, England Description: A tall, curved glass skyscraper towers over a city street at night. The scene feat... | missing terms: Cars, Commuting, Street signs, The Fenchurch Building (The Walki..., GBR \| low-draft-improvement              |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |       100 |           93 |      90.1 | 26 GB (22.6% of 108 GB recommended working set) | `recommended` | Title: The Fenchurch Building at night Description: A low-angle view captures the distinctive curved glass facade of ... [tail] cle headlights reflect on the wet road surface. ...  | missing terms: Cars, City, Commuting, Fenchurch Street, Nightscape                                                            |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |        85 |          100 |      17.8 | 38 GB (33.3% of 108 GB recommended working set) | `recommended` | Title: Night view of 20 Fenchurch Street, London ...                                                                                                                                 | missing terms: Cars, City, Commuting, Nightscape, Street signs                                                                |
| `mlx-community/Qwen3.5-27B-4bit`                        |       100 |          100 |      30.7 | 26 GB (22.4% of 108 GB recommended working set) | `recommended` | Title: 20 Fenchurch Street at Night, London ...                                                                                                                                      | missing terms: Cars, Cityscape, Commuting, Modern, Nightscape                                                                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |       100 |          100 |      17.4 | 38 GB (33.3% of 108 GB recommended working set) | `recommended` | Title: Night view of 20 Fenchurch Street, London ...                                                                                                                                 | missing terms: Cars, Commuting, The Fenchurch Building (The Walki..., Walkie Talkie building, GBR                             |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |       100 |           93 |      31.3 | 18 GB (15.7% of 108 GB recommended working set) | `recommended` | [low-draft-improvement] Title: The Fenchurch Building illuminated at night ...                                                                                                       | missing terms: Cars, Commuting, Fenchurch Street, Street signs, The Fenchurch Building (The Walki... \| low-draft-improvement |
| `mlx-community/InternVL3-8B-bf16`                       |       100 |           85 |      34.2 | 18 GB (15.3% of 108 GB recommended working set) | `recommended` | [low-draft-improvement] Title: The Walkie-Talkie Building, London, England ...                                                                                                       | missing terms: Nightscape, The Fenchurch Building (The Walki..., GBR, known, formally \| low-draft-improvement                |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |       100 |           87 |      92.7 | 11 GB (9.94% of 108 GB recommended working set) | `recommended` | Title: The Fenchurch Building at Night, London ...                                                                                                                                   | missing terms: Cars, Commuting, Fenchurch Street, The Fenchurch Building (The Walki..., GBR                                   |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       100 |           87 |      30.1 | 23 GB (19.9% of 108 GB recommended working set) | `recommended` | [low-draft-improvement] Title: The Fenchurch Building at night ...                                                                                                                   | missing terms: Cityscape, Commuting, London, Nightscape, The Fenchurch Building (The Walki... \| low-draft-improvement        |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |       100 |           87 |      17.8 | 32 GB (27.4% of 108 GB recommended working set) | `recommended` | [low-draft-improvement] Title: The Fenchurch Building at Night, London ...                                                                                                           | missing terms: Cars, Commuting, Fenchurch Street, Street signs, The Fenchurch Building (The Walki... \| low-draft-improvement |

## Structured Metadata Candidates

Top 10 ranked candidates for structured title/description/keywords. Use the gallery for complete per-model evidence.
Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                                   |   Metadata agreement | Verdict   | Output Preview                                                                                                                                                                       |
|---------------------------------------------------------|----------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-31b-it-4bit`                     |                   34 | `clean`   | [low-draft-improvement] Title: The Fenchurch Building at night in London, England Description: A tall, curved glass skyscraper towers over a city street at night. The scene feat... |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |                   25 | `clean`   | Title: The Fenchurch Building at night Description: A low-angle view captures the distinctive curved glass facade of ... [tail] cle headlights reflect on the wet road surface. ...  |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |                   28 | `clean`   | Title: Night view of 20 Fenchurch Street, London ...                                                                                                                                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |                   26 | `clean`   | Title: 20 Fenchurch Street at Night, London ...                                                                                                                                      |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |                   32 | `clean`   | Title: Night view of 20 Fenchurch Street, London ...                                                                                                                                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |                   30 | `clean`   | [low-draft-improvement] Title: The Fenchurch Building illuminated at night ...                                                                                                       |
| `mlx-community/InternVL3-8B-bf16`                       |                   39 | `clean`   | [low-draft-improvement] Title: The Walkie-Talkie Building, London, England ...                                                                                                       |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |                   35 | `clean`   | Title: The Fenchurch Building at Night, London ...                                                                                                                                   |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |                   31 | `clean`   | [low-draft-improvement] Title: The Fenchurch Building at night ...                                                                                                                   |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |                   34 | `clean`   | [low-draft-improvement] Title: The Fenchurch Building at Night, London ...                                                                                                           |

## Repository Variant Comparisons

Policy: quality-first (reliability-gated assisted enrichment) within matching repository families.
Evidence scope: 1 image, 1 current run per variant; scores and histories are not merged.

| Family                                      | Variant                                             | Eligible   |   Quality | Total   | Peak GB                                          |
|---------------------------------------------|-----------------------------------------------------|------------|-----------|---------|--------------------------------------------------|
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-6bit`                 | no         |        73 | 9.72s   | 11 GB (9.58% of 108 GB recommended working set)  |
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-mxfp4`                | no         |        46 | 13.75s  | 8.4 GB (7.29% of 108 GB recommended working set) |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | no         |        70 | 6.39s   | 13 GB (11.7% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | no         |        65 | 5.93s   | 13 GB (11.3% of 108 GB recommended working set)  |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-8bit`                | no         |        69 | 24.48s  | 41 GB (35.2% of 108 GB recommended working set)  |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-bf16`                | no         |        69 | 27.46s  | 48 GB (41.4% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-4bit`                    | yes        |        64 | 80.47s  | 26 GB (22.4% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-mxfp8`                   | yes        |        64 | 85.77s  | 38 GB (33.3% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-4bit`                | yes        |        72 | 58.52s  | 26 GB (22.6% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-bf16`                | no         |        77 | 64.64s  | 76 GB (65.8% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-6bit`                | no         |        77 | 64.58s  | 35 GB (30.1% of 108 GB recommended working set)  |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`     | no         |        72 | 5.84s   | 28 GB (24.6% of 108 GB recommended working set)  |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-8bit`      | no         |        54 | 6.17s   | 29 GB (25.3% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-4bit`             | yes        |        60 | 7.87s   | 18 GB (15.7% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-8bit`             | yes        |        57 | 10.98s  | 32 GB (27.4% of 108 GB recommended working set)  |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | no         |        22 | 4.21s   | 12 GB (10.5% of 108 GB recommended working set)  |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | no         |        21 | 7.94s   | 27 GB (23.3% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-bf16`                    | no         |        56 | 9.25s   | 28 GB (23.9% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-8bit`                    | no         |        50 | 6.45s   | 16 GB (13.6% of 108 GB recommended working set)  |
