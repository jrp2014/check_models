# Model Selection Brief

Generated on: 2026-07-19 20:37:23 BST

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

| Model                                                 | Task outcome            | Compatibility       | Recommendation eligibility                                   | Prompt burden   | Gen TPS   | Peak GB                                          |
|-------------------------------------------------------|-------------------------|---------------------|--------------------------------------------------------------|-----------------|-----------|--------------------------------------------------|
| mlx-community/Molmo-7B-D-0924-8bit                    | Task outcome: crashed   | indeterminate       | excluded: model input was unreachable; outcome indeterminate | unknown         | -         | -                                                |
| mlx-community/Step-3.7-Flash-oQ2e                     | Task outcome: crashed   | crashed             | excluded: current run crashed                                | unknown         | -         | -                                                |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 509       | 1.5 GB (1.31% of 108 GB recommended working set) |
| mlx-community/nanoLLaVA-1.5-4bit                      | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 346       | 2.8 GB (2.42% of 108 GB recommended working set) |
| mlx-community/LFM2-VL-1.6B-8bit                       | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 331       | 3.0 GB (2.58% of 108 GB recommended working set) |
| HuggingFaceTB/SmolVLM-Instruct                        | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 131       | 5.7 GB (4.96% of 108 GB recommended working set) |
| mlx-community/SmolVLM-Instruct-bf16                   | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 124       | 5.7 GB (4.97% of 108 GB recommended working set) |
| qnguyen3/nanoLLaVA                                    | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 113       | 4.7 GB (4.08% of 108 GB recommended working set) |
| mlx-community/FastVLM-0.5B-bf16                       | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 343       | 2.2 GB (1.88% of 108 GB recommended working set) |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | Task outcome: completed | clean               | eligible                                                     | normal          | 123       | 17 GB (14.3% of 108 GB recommended working set)  |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 132       | 5.5 GB (4.75% of 108 GB recommended working set) |
| mlx-community/paligemma2-10b-ft-docci-448-6bit        | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 31.0      | 12 GB (10.5% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | Task outcome: completed | clean               | excluded: current review says caveat                         | mixed           | 185       | 7.8 GB (6.76% of 108 GB recommended working set) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit          | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 66.5      | 29 GB (25.4% of 108 GB recommended working set)  |
| mlx-community/Idefics3-8B-Llama3-bf16                 | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 33.4      | 19 GB (16.6% of 108 GB recommended working set)  |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 62.7      | 28 GB (24.6% of 108 GB recommended working set)  |
| mlx-community/InternVL3-8B-bf16                       | Task outcome: completed | clean               | eligible                                                     | normal          | 34.2      | 18 GB (15.3% of 108 GB recommended working set)  |
| mlx-community/gemma-3n-E2B-4bit                       | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 123       | 6.0 GB (5.21% of 108 GB recommended working set) |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 5.39      | 25 GB (21.9% of 108 GB recommended working set)  |
| mlx-community/paligemma2-10b-ft-docci-448-bf16        | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 5.54      | 27 GB (23.3% of 108 GB recommended working set)  |
| mlx-community/llava-v1.6-mistral-7b-8bit              | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 62.1      | 9.7 GB (8.41% of 108 GB recommended working set) |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | Task outcome: completed | clean               | eligible                                                     | mixed           | 62.9      | 13 GB (11.7% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b-8bit                        | Task outcome: completed | clean               | excluded: current review says caveat                         | mixed           | 39.2      | 16 GB (13.6% of 108 GB recommended working set)  |
| mlx-community/InternVL3-14B-8bit                      | Task outcome: completed | clean               | eligible                                                     | normal          | 32.1      | 18 GB (15.7% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it-qat-4bit                 | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 30.9      | 18 GB (15.7% of 108 GB recommended working set)  |
| mlx-community/gemma-4-31b-it-4bit                     | Task outcome: completed | clean               | eligible                                                     | normal          | 26.4      | 20 GB (17.7% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b-bf16                        | Task outcome: completed | clean               | excluded: current review says caveat                         | mixed           | 20.1      | 28 GB (23.9% of 108 GB recommended working set)  |
| mlx-community/gemma-3n-E4B-it-bf16                    | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 48.2      | 17 GB (14.9% of 108 GB recommended working set)  |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | Task outcome: completed | clean               | eligible                                                     | visual_input    | 80.0      | 8.6 GB (7.42% of 108 GB recommended working set) |
| mlx-community/MiniCPM-V-4.6-8bit                      | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 24.7      | 4.1 GB (3.53% of 108 GB recommended working set) |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit                 | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 71.0      | 18 GB (15.6% of 108 GB recommended working set)  |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 30.3      | 23 GB (19.9% of 108 GB recommended working set)  |
| mlx-community/GLM-4.6V-Flash-6bit                     | Task outcome: completed | clean               | excluded: current review says caveat                         | visual_input    | 57.7      | 11 GB (9.58% of 108 GB recommended working set)  |
| mlx-community/Kimi-VL-A3B-Thinking-8bit               | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 62.6      | 22 GB (19% of 108 GB recommended working set)    |
| microsoft/Phi-3.5-vision-instruct                     | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 57.2      | 9.6 GB (8.3% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it-qat-8bit                 | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 17.4      | 32 GB (27.4% of 108 GB recommended working set)  |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 59.6      | 60 GB (52.1% of 108 GB recommended working set)  |
| mlx-community/paligemma2-3b-pt-896-4bit               | Task outcome: completed | clean               | excluded: current review says avoid                          | visual_input    | 42.4      | 4.6 GB (3.96% of 108 GB recommended working set) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review                | mixed           | 103       | 8.6 GB (7.44% of 108 GB recommended working set) |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | Task outcome: completed | clean               | excluded: current review says avoid                          | mixed           | 42.7      | 15 GB (13% of 108 GB recommended working set)    |
| mlx-community/GLM-4.6V-nvfp4                          | Task outcome: completed | clean               | excluded: current review says caveat                         | visual_input    | 43.0      | 78 GB (67.3% of 108 GB recommended working set)  |
| mlx-community/gemma-4-31b-bf16                        | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 7.5       | 65 GB (56% of 108 GB recommended working set)    |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 18.0      | 4.1 GB (3.56% of 108 GB recommended working set) |
| Qwen/Qwen3-VL-2B-Instruct                             | Task outcome: completed | integration-warning | excluded: integration warning requires review                | mixed           | 93.3      | 8.6 GB (7.43% of 108 GB recommended working set) |
| mlx-community/X-Reasoner-7B-8bit                      | Task outcome: completed | integration-warning | excluded: integration warning requires review                | mixed           | 64.1      | 13 GB (11.7% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | Task outcome: completed | clean               | eligible                                                     | mixed           | 58.0      | 13 GB (11.3% of 108 GB recommended working set)  |
| mlx-community/paligemma2-3b-ft-docci-448-bf16         | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 19.0      | 11 GB (9.71% of 108 GB recommended working set)  |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 11.5      | 15 GB (13% of 108 GB recommended working set)    |
| mlx-community/Phi-3.5-vision-instruct-bf16            | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 10.8      | 9.6 GB (8.31% of 108 GB recommended working set) |
| mlx-community/Molmo-7B-D-0924-bf16                    | Task outcome: completed | clean               | excluded: current review says caveat                         | normal          | 14.2      | 48 GB (41.4% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | Task outcome: completed | clean               | eligible                                                     | mixed           | 92.1      | 11 GB (9.94% of 108 GB recommended working set)  |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review                | mixed           | 92.3      | 8.6 GB (7.44% of 108 GB recommended working set) |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | Task outcome: completed | clean               | eligible                                                     | mixed           | 107       | 26 GB (22.6% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | Task outcome: completed | clean               | eligible                                                     | mixed           | 91.2      | 35 GB (30.1% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | Task outcome: completed | clean               | eligible                                                     | mixed           | 65.8      | 76 GB (65.8% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B-4bit                        | Task outcome: completed | clean               | eligible                                                     | mixed           | 30.7      | 26 GB (22.4% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                                     | mixed           | 18.3      | 38 GB (33.3% of 108 GB recommended working set)  |
| mlx-community/Qwen3.6-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                                     | mixed           | 17.7      | 38 GB (33.3% of 108 GB recommended working set)  |
| mlx-community/MolmoPoint-8B-fp16                      | Task outcome: completed | clean               | excluded: current review says caveat                         | visual_input    | 1.83      | 27 GB (23.2% of 108 GB recommended working set)  |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16          | Task outcome: completed | clean               | excluded: current review says avoid                          | normal          | 4.69      | 40 GB (34.7% of 108 GB recommended working set)  |
| mlx-community/Ornith-1.0-35B-bf16                     | Task outcome: completed | clean               | eligible                                                     | mixed           | 57.3      | 76 GB (65.8% of 108 GB recommended working set)  |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | Task outcome: completed | integration-warning | excluded: integration warning requires review                | mixed           | 225       | 5.1 GB (4.39% of 108 GB recommended working set) |

## Quick Chooser

Practical current-run buckets for model users. In assisted mode, quality rankings combine visual usefulness with correct use of authoritative metadata; runtime and contract findings remain diagnostic triage signals.

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

| Model                                               | Peak GB                                          |   Gen TPS |   Usefulness | Status        | Evidence                                                                   |
|-----------------------------------------------------|--------------------------------------------------|-----------|--------------|---------------|----------------------------------------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit`             | 17 GB (14.3% of 108 GB recommended working set)  |     123   |           93 | `recommended` | Title: The Fenchurch Building in London at night ...                       |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | 13 GB (11.7% of 108 GB recommended working set)  |      62.9 |          100 | `recommended` | **Title:** *20 Fenchurch Street at Night, London, England, UK* ...         |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                | 8.6 GB (7.42% of 108 GB recommended working set) |      80   |           87 | `recommended` | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK ... |

### Quality if memory allows

Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                               | Peak GB                                         |   Gen TPS |   Usefulness | Status        | Evidence                                                           |
|-----------------------------------------------------|-------------------------------------------------|-----------|--------------|---------------|--------------------------------------------------------------------|
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | 13 GB (11.7% of 108 GB recommended working set) |      62.9 |          100 | `recommended` | **Title:** *20 Fenchurch Street at Night, London, England, UK* ... |
| `mlx-community/Qwen3.5-27B-4bit`                    | 26 GB (22.4% of 108 GB recommended working set) |      30.7 |          100 | `recommended` | Title: 20 Fenchurch Street, London, England, UK, Europe ...        |
| `mlx-community/Qwen3.5-27B-mxfp8`                   | 38 GB (33.3% of 108 GB recommended working set) |      18.3 |          100 | `recommended` | Title: The Walkie-Talkie Building, London, England, UK ...         |

### Current failures / avoid

Policy: reliability-gated exclusion evidence.
Evidence scope: 1 image, 1 current run.

| Model                                | Peak GB                                          | Gen TPS   | Usefulness   | Status          | Evidence                                                                                                                                                                                                    |
|--------------------------------------|--------------------------------------------------|-----------|--------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Molmo-7B-D-0924-8bit` | -                                                | -         | -            | `not_evaluated` | network error \| unknown model load network error \| external connectivity \| Task outcome: crashed \| ReadError: [Errno 54] Connection reset by peer \| model input was unreachable; outcome indeterminate |
| `mlx-community/Step-3.7-Flash-oQ2e`  | -                                                | -         | -            | `avoid`         | processor error \| model config processor load processor \| Task outcome: crashed \| ValueError: Loaded processor has no image_processor; expected multimodal processor. \| current run crashed             |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`   | 1.5 GB (1.31% of 108 GB recommended working set) | 509       | 77           | `caveat`        | missing terms: Cars, Commuting \| keywords=23 \| low-draft-improvement \| current review says caveat                                                                                                        |
| `mlx-community/nanoLLaVA-1.5-4bit`   | 2.8 GB (2.42% of 108 GB recommended working set) | 346       | 76           | `caveat`        | keywords=20 \| context echo=76% \| low-draft-improvement \| current review says caveat                                                                                                                      |
| `mlx-community/LFM2-VL-1.6B-8bit`    | 3.0 GB (2.58% of 108 GB recommended working set) | 331       | 77           | `caveat`        | missing terms: formally \| keywords=20 \| low-draft-improvement \| current review says caveat                                                                                                               |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.
Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                               |   Hygiene |   Usefulness |   Gen TPS | Peak GB                                         | Verdict       | Caption Preview                                                    | Caveat                                                                                         |
|-----------------------------------------------------|-----------|--------------|-----------|-------------------------------------------------|---------------|--------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` |        85 |          100 |      62.9 | 13 GB (11.7% of 108 GB recommended working set) | `recommended` | **Title:** *20 Fenchurch Street at Night, London, England, UK* ... | missing terms: Cars, City, Commuting, Nightscape, Street signs                                 |
| `mlx-community/Qwen3.5-27B-4bit`                    |       100 |          100 |      30.7 | 26 GB (22.4% of 108 GB recommended working set) | `recommended` | Title: 20 Fenchurch Street, London, England, UK, Europe ...        | missing terms: Building, Buildings, Cars, City, Commuting                                      |
| `mlx-community/Qwen3.5-27B-mxfp8`                   |        85 |          100 |      18.3 | 38 GB (33.3% of 108 GB recommended working set) | `recommended` | Title: The Walkie-Talkie Building, London, England, UK ...         | missing terms: Cars, City, Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie) |
| `mlx-community/Qwen3.6-27B-mxfp8`                   |       100 |          100 |      17.7 | 38 GB (33.3% of 108 GB recommended working set) | `recommended` | Title: 20 Fenchurch Street, London, England, UK, Europe ...        | missing terms: Cars, City, Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie) |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                |       100 |           93 |      65.8 | 76 GB (65.8% of 108 GB recommended working set) | `recommended` | Title: The Fenchurch Building at night in London ...               | missing terms: Cars, City, Commuting, Modern, The Fenchurch Building (The Walkie-Talkie)       |
| `mlx-community/Ornith-1.0-35B-bf16`                 |       100 |           93 |      57.3 | 76 GB (65.8% of 108 GB recommended working set) | `recommended` | Title: - The Fenchurch Street skyscraper at night ...              | missing terms: Building, Buildings, Commuting, Modern, Nightscape                              |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                |       100 |           93 |      91.2 | 35 GB (30.1% of 108 GB recommended working set) | `recommended` | Title: The Fenchurch Building at night in London ...               | missing terms: Cars, Cityscape, Commuting, Modern, The Fenchurch Building (The Walkie-Talkie)  |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                |        85 |           93 |     107   | 26 GB (22.6% of 108 GB recommended working set) | `recommended` | Title: The Fenchurch Building at night in London ...               | missing terms: Cars, City, Commuting, Fenchurch Street, Nightscape                             |
| `mlx-community/gemma-4-26b-a4b-it-4bit`             |       100 |           93 |     123   | 17 GB (14.3% of 108 GB recommended working set) | `recommended` | Title: The Fenchurch Building in London at night ...               | missing terms: Cars, City, Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie) |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 |       100 |           93 |      92.1 | 11 GB (9.94% of 108 GB recommended working set) | `recommended` | Title: The Fenchurch Building, London, UK, Night ...               | missing terms: Cars, Commuting, The Fenchurch Building (The Walkie-Talkie), GBR, formally      |

## Structured Metadata Candidates

Top 10 ranked candidates for structured title/description/keywords. Use the gallery for complete per-model evidence.
Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                               |   Metadata agreement | Verdict   | Output Preview                                                     |
|-----------------------------------------------------|----------------------|-----------|--------------------------------------------------------------------|
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` |                   21 | `clean`   | **Title:** *20 Fenchurch Street at Night, London, England, UK* ... |
| `mlx-community/Qwen3.5-27B-4bit`                    |                   30 | `clean`   | Title: 20 Fenchurch Street, London, England, UK, Europe ...        |
| `mlx-community/Qwen3.5-27B-mxfp8`                   |                   39 | `clean`   | Title: The Walkie-Talkie Building, London, England, UK ...         |
| `mlx-community/Qwen3.6-27B-mxfp8`                   |                   31 | `clean`   | Title: 20 Fenchurch Street, London, England, UK, Europe ...        |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                |                   28 | `clean`   | Title: The Fenchurch Building at night in London ...               |
| `mlx-community/Ornith-1.0-35B-bf16`                 |                   25 | `clean`   | Title: - The Fenchurch Street skyscraper at night ...              |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                |                   30 | `clean`   | Title: The Fenchurch Building at night in London ...               |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                |                   29 | `clean`   | Title: The Fenchurch Building at night in London ...               |
| `mlx-community/gemma-4-26b-a4b-it-4bit`             |                   31 | `clean`   | Title: The Fenchurch Building in London at night ...               |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                 |                   41 | `clean`   | Title: The Fenchurch Building, London, UK, Night ...               |

## Repository Variant Comparisons

Policy: quality-first (reliability-gated assisted enrichment) within matching repository families.
Evidence scope: 1 image, 1 current run per variant; scores and histories are not merged.

| Family                                      | Variant                                             | Eligible   |   Quality | Total   | Peak GB                                          |
|---------------------------------------------|-----------------------------------------------------|------------|-----------|---------|--------------------------------------------------|
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-mxfp4`                | yes        |        74 | 9.67s   | 8.6 GB (7.42% of 108 GB recommended working set) |
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-6bit`                 | no         |        56 | 10.46s  | 11 GB (9.58% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | yes        |        84 | 7.10s   | 13 GB (11.7% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | yes        |        63 | 28.24s  | 13 GB (11.3% of 108 GB recommended working set)  |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-bf16`                | no         |        70 | 53.58s  | 48 GB (41.4% of 108 GB recommended working set)  |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-8bit`                | no         |         0 | 0.29s   | -                                                |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-4bit`                    | yes        |        84 | 75.00s  | 26 GB (22.4% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-mxfp8`                   | yes        |        84 | 80.20s  | 38 GB (33.3% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-bf16`                | yes        |        83 | 78.47s  | 76 GB (65.8% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-6bit`                | yes        |        82 | 66.04s  | 35 GB (30.1% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-4bit`                | yes        |        81 | 63.41s  | 26 GB (22.6% of 108 GB recommended working set)  |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-8bit`      | no         |        72 | 6.65s   | 29 GB (25.4% of 108 GB recommended working set)  |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`     | no         |        70 | 6.56s   | 28 GB (24.6% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-4bit`             | no         |        59 | 9.54s   | 18 GB (15.7% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-8bit`             | no         |        56 | 14.22s  | 32 GB (27.4% of 108 GB recommended working set)  |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | no         |        21 | 4.32s   | 12 GB (10.5% of 108 GB recommended working set)  |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | no         |        21 | 8.13s   | 27 GB (23.3% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-bf16`                    | no         |        53 | 11.01s  | 28 GB (23.9% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-8bit`                    | no         |        48 | 7.46s   | 16 GB (13.6% of 108 GB recommended working set)  |
