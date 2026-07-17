# Model Selection Brief

Generated on: 2026-07-17 13:44:05 BST

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

| Model                                                 | Task outcome            | Compatibility       | Recommendation eligibility                                 | Prompt burden   | Gen TPS   | Peak GB   |
|-------------------------------------------------------|-------------------------|---------------------|------------------------------------------------------------|-----------------|-----------|-----------|
| mlx-community/gemma-4-31b-bf16                        | Task outcome: crashed   | crashed             | excluded: current run crashed                              | unknown         | -         | -         |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | Task outcome: completed | clean               | eligible                                                   | normal          | 504       | 1.4       |
| mlx-community/LFM2-VL-1.6B-8bit                       | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 331       | 3.0       |
| qnguyen3/nanoLLaVA                                    | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 113       | 5.1       |
| mlx-community/nanoLLaVA-1.5-4bit                      | Task outcome: completed | clean               | eligible                                                   | normal          | 192       | 2.8       |
| HuggingFaceTB/SmolVLM-Instruct                        | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 127       | 5.6       |
| mlx-community/MiniCPM-V-4.6-8bit                      | Task outcome: completed | integration-warning | excluded: integration warning requires review              | normal          | 181       | 4.0       |
| mlx-community/FastVLM-0.5B-bf16                       | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 292       | 2.2       |
| mlx-community/paligemma2-10b-ft-docci-448-6bit        | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 30.5      | 12        |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 56.7      | 18        |
| microsoft/Phi-3.5-vision-instruct                     | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 57.6      | 9.5       |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 175       | 4.1       |
| mlx-community/InternVL3-8B-bf16                       | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 35.3      | 18        |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 66.1      | 13        |
| mlx-community/pixtral-12b-8bit                        | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 39.4      | 16        |
| mlx-community/paligemma2-10b-ft-docci-448-bf16        | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 5.63      | 26        |
| mlx-community/Idefics3-8B-Llama3-bf16                 | Task outcome: completed | clean               | excluded: output is below the configured chooser threshold | normal          | 33.3      | 19        |
| mlx-community/diffusiongemma-26B-A4B-it-8bit          | Task outcome: completed | clean               | eligible                                                   | normal          | 20.6      | 29        |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 64.9      | 7.8       |
| mlx-community/SmolVLM-Instruct-bf16                   | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 38.6      | 5.7       |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 120       | 5.5       |
| mlx-community/pixtral-12b-bf16                        | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 20.7      | 28        |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 26.5      | 13        |
| mlx-community/gemma-4-31b-it-4bit                     | Task outcome: completed | clean               | eligible                                                   | normal          | 19.6      | 22        |
| mlx-community/InternVL3-14B-8bit                      | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 23.6      | 18        |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 30.4      | 23        |
| mlx-community/gemma-3n-E2B-4bit                       | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 77.6      | 6.0       |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 15.0      | 28        |
| mlx-community/Phi-3.5-vision-instruct-bf16            | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 19.1      | 9.5       |
| mlx-community/gemma-3n-E4B-it-bf16                    | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 46.8      | 17        |
| mlx-community/gemma-3-27b-it-qat-8bit                 | Task outcome: completed | clean               | eligible                                                   | normal          | 17.7      | 32        |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit                 | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 66.7      | 18        |
| mlx-community/Kimi-VL-A3B-Thinking-8bit               | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 61.5      | 22        |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | Task outcome: completed | clean               | eligible                                                   | normal          | 19.0      | 15        |
| mlx-community/gemma-3-27b-it-qat-4bit                 | Task outcome: completed | clean               | eligible                                                   | normal          | 18.9      | 18        |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 57.4      | 60        |
| mlx-community/llava-v1.6-mistral-7b-8bit              | Task outcome: completed | clean               | eligible                                                   | normal          | 59.9      | 9.7       |
| mlx-community/paligemma2-3b-pt-896-4bit               | Task outcome: completed | clean               | excluded: current review says avoid                        | visual_input    | 50.0      | 4.6       |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | Task outcome: completed | clean               | excluded: current review says avoid                        | visual_input    | 80.0      | 8.4       |
| mlx-community/GLM-4.6V-Flash-6bit                     | Task outcome: completed | clean               | eligible                                                   | visual_input    | 15.2      | 11        |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 38.8      | 15        |
| mlx-community/GLM-4.6V-nvfp4                          | Task outcome: completed | clean               | eligible                                                   | visual_input    | 32.7      | 78        |
| mlx-community/Molmo-7B-D-0924-bf16                    | Task outcome: completed | clean               | eligible                                                   | normal          | 30.3      | 48        |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review              | mixed           | 91.9      | 8.6       |
| mlx-community/MolmoPoint-8B-fp16                      | Task outcome: completed | clean               | eligible                                                   | visual_input    | 5.95      | 27        |
| mlx-community/Molmo-7B-D-0924-8bit                    | Task outcome: completed | clean               | eligible                                                   | normal          | 50.9      | 41        |
| Qwen/Qwen3-VL-2B-Instruct                             | Task outcome: completed | integration-warning | excluded: integration warning requires review              | mixed           | 92.5      | 8.6       |
| mlx-community/paligemma2-3b-ft-docci-448-bf16         | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 18.9      | 11        |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 3.35      | 25        |
| mlx-community/X-Reasoner-7B-8bit                      | Task outcome: completed | integration-warning | excluded: integration warning requires review              | mixed           | 46.4      | 13        |
| mlx-community/Ornith-1.0-35B-bf16                     | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 56.1      | 76        |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 68.7      | 35        |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review              | mixed           | 27.9      | 8.6       |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 91.4      | 11        |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 59.7      | 76        |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 68.2      | 26        |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | Task outcome: completed | integration-warning | excluded: integration warning requires review              | mixed           | 207       | 5.1       |
| mlx-community/Qwen3.5-27B-mxfp8                       | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 18.3      | 38        |
| mlx-community/Qwen3.5-27B-4bit                        | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           | 23.9      | 26        |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16          | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          | 4.37      | 39        |
| mlx-community/Qwen3.6-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                                   | mixed           | 6.35      | 38        |

## Quick Chooser

Practical current-run buckets for model users. These are triage signals, not grounded visual-quality claims.

### Best under 4 GB

Policy: memory-aware (reliability-gated assisted enrichment; budget 4 GB).
Evidence scope: 1 image, 1 current run.

| Model                              |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                         |
|------------------------------------|-----------|-----------|--------------|----------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` |       1.4 |       504 |           90 | `caveat` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, Boating, Sailboat, River, Riverba... |
| `mlx-community/nanoLLaVA-1.5-4bit` |       2.8 |       192 |           75 | `caveat` | [low-draft-improvement] Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Descriptio... |

### Best under 8 GB

Policy: memory-aware (reliability-gated assisted enrichment; budget 8 GB).
Evidence scope: 1 image, 1 current run.

| Model                              |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                         |
|------------------------------------|-----------|-----------|--------------|----------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` |       1.4 |       504 |           90 | `caveat` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, Boating, Sailboat, River, Riverba... |
| `mlx-community/nanoLLaVA-1.5-4bit` |       2.8 |       192 |           75 | `caveat` | [low-draft-improvement] Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Descriptio... |

### Fastest usable

Policy: efficiency-aware Pareto frontier (reliability-gated).
Evidence scope: 1 image, 1 current run.

| Model                              |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                         |
|------------------------------------|-----------|-----------|--------------|----------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` |       1.4 |       504 |           90 | `caveat` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, Boating, Sailboat, River, Riverba... |

### Quality if memory allows

Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                      |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                         |
|--------------------------------------------|-----------|-----------|--------------|----------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`         |       1.4 |     504   |           90 | `caveat` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, Boating, Sailboat, River, Riverba... |
| `mlx-community/llava-v1.6-mistral-7b-8bit` |       9.7 |      59.9 |           75 | `caveat` | [unverified-context-copy; low-draft-improvement; metadata-borrowing] Title: Deben Estuary, Wo... |
| `mlx-community/nanoLLaVA-1.5-4bit`         |       2.8 |     192   |           75 | `caveat` | [low-draft-improvement] Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Descriptio... |

### Current failures / avoid

Policy: reliability-gated exclusion evidence.
Evidence scope: 1 image, 1 current run.

| Model                              | Peak GB   | Gen TPS   | Usefulness   | Status   | Evidence                                                                                                                                                                                                                |
|------------------------------------|-----------|-----------|--------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-31b-bf16`   | -         | -         | -            | `avoid`  | model error \| mlx vlm decode model \| Task outcome: crashed \| IndexError: list index out of range \| current run crashed                                                                                              |
| `mlx-community/LFM2-VL-1.6B-8bit`  | 3.0       | 331       | 83           | `avoid`  | keywords=20 \| unverified-context-copy \| low-draft-improvement \| current review says avoid                                                                                                                            |
| `qnguyen3/nanoLLaVA`               | 5.1       | 113       | 75           | `avoid`  | missing sections: keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| low-draft-improvement \| current review says avoid                                                                                   |
| `HuggingFaceTB/SmolVLM-Instruct`   | 5.6       | 127       | 47           | `avoid`  | missing sections: title, description, keywords \| missing terms: Bird, Boat, Boating, Buoy, Bushes \| low-draft-improvement \| current review says avoid                                                                |
| `mlx-community/MiniCPM-V-4.6-8bit` | 4.0       | 181       | 90           | `avoid`  | Special control token &lt;/think&gt; appeared in generated text. \| missing terms: Bird, Boating, Bushes, Coast, Estuary \| reasoning leak \| text-sanity=gibberish(token_noise) \| integration warning requires review |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.
Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                          |   Hygiene |   Usefulness |   Gen TPS |   Peak GB | Verdict       | Caption Preview                                                                                                                                                                      | Caveat                                                                                                                         |
|------------------------------------------------|-----------|--------------|-----------|-----------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`             |        60 |           90 |    504    |       1.4 | `caveat`      | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, Boating, Sailboat, River, Riverbank, Nature, Outdoors, Pe ... [tail] The overall atmosphere is peaceful and natural. ... | missing terms: Bird, Buoy, Mudflat, behind, bank                                                                               |
| `mlx-community/llava-v1.6-mistral-7b-8bit`     |        60 |           75 |     59.9  |       9.7 | `caveat`      | [unverified-context-copy; low-draft-improvement; metadata-borrowing] Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river...  | hit token cap (500) \| keywords=27 \| low-draft-improvement                                                                    |
| `mlx-community/nanoLLaVA-1.5-4bit`             |        75 |           75 |    192    |       2.8 | `caveat`      | [low-draft-improvement] Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywords: Bird,... | keywords=21 \| low-draft-improvement                                                                                           |
| `mlx-community/gemma-4-31b-it-4bit`            |       100 |           93 |     19.6  |      22   | `recommended` | [low-draft-improvement] Title: Sailing boats moored on the Deben Estuary, England Description: Two sailing boats with wooden masts are moored on a river in front of a dense gree... | missing terms: Bird, Boating, Bushes, Coast, Mudflat \| low-draft-improvement                                                  |
| `mlx-community/Qwen3.6-27B-mxfp8`              |       100 |           93 |      6.35 |      38   | `recommended` | Title: Two sailing boats moored on a river ...                                                                                                                                       | missing terms: Bird, Boating, Bushes, Coast, Forest                                                                            |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit` |       100 |           87 |     20.6  |      29   | `recommended` | [low-draft-improvement] Title: - Two sailing boats moored on the Deben Estuary ...                                                                                                   | missing terms: Bird, Coast, Mudflat, Peaceful, Woodbridge \| low-draft-improvement                                             |
| `mlx-community/GLM-4.6V-Flash-6bit`            |        75 |           93 |     15.2  |      11   | `caveat`      | [context-budget; missing-sections] <\|begin_of_box\|>Title: Two sailboats moored on water with trees in background ...                                                               | output/prompt=1.46% \| visual input burden=93% \| missing sections: title \| missing terms: Bird, Boating, Buoy, Bushes, Coast |
| `mlx-community/Molmo-7B-D-0924-bf16`           |        75 |           73 |     30.3  |      48   | `caveat`      | [low-draft-improvement] Remove uncertain tags. Title: Sailing boats on the River Deben near Woodbridge, England ...                                                                  | missing terms: GBR, bank \| keywords=20 \| low-draft-improvement                                                               |
| `mlx-community/Molmo-7B-D-0924-8bit`           |        75 |           73 |     50.9  |      41   | `caveat`      | [low-draft-improvement] Remove uncertain tags. Title: Sailing boats on the River Deben near Woodbridge, England ...                                                                  | missing terms: GBR, bank \| keywords=20 \| low-draft-improvement                                                               |
| `mlx-community/gemma-3-27b-it-qat-4bit`        |        75 |           92 |     18.9  |      18   | `caveat`      | [low-draft-improvement] Title: Two sailing boats moored on a riverbank ...                                                                                                           | missing terms: Bird, Bushes, Forest, Peaceful, Rigging \| keywords=21 \| low-draft-improvement                                 |

## Structured Metadata Candidates

Top 10 ranked candidates for structured title/description/keywords. Use the gallery for complete per-model evidence.
Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                          |   Metadata agreement | Verdict          | Output Preview                                                                                                                                                                       |
|------------------------------------------------|----------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`             |                   43 | `clean`          | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, Boating, Sailboat, River, Riverbank, Nature, Outdoors, Pe ... [tail] The overall atmosphere is peaceful and natural. ... |
| `mlx-community/llava-v1.6-mistral-7b-8bit`     |                   26 | `token_cap`      | [unverified-context-copy; low-draft-improvement; metadata-borrowing] Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river...  |
| `mlx-community/nanoLLaVA-1.5-4bit`             |                   47 | `clean`          | [low-draft-improvement] Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywords: Bird,... |
| `mlx-community/gemma-4-31b-it-4bit`            |                   25 | `clean`          | [low-draft-improvement] Title: Sailing boats moored on the Deben Estuary, England Description: Two sailing boats with wooden masts are moored on a river in front of a dense gree... |
| `mlx-community/Qwen3.6-27B-mxfp8`              |                   27 | `clean`          | Title: Two sailing boats moored on a river ...                                                                                                                                       |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit` |                   25 | `clean`          | [low-draft-improvement] Title: - Two sailing boats moored on the Deben Estuary ...                                                                                                   |
| `mlx-community/GLM-4.6V-Flash-6bit`            |                   18 | `context_budget` | [context-budget; missing-sections] <\|begin_of_box\|>Title: Two sailboats moored on water with trees in background ...                                                               |
| `mlx-community/Molmo-7B-D-0924-bf16`           |                   29 | `clean`          | [low-draft-improvement] Remove uncertain tags. Title: Sailing boats on the River Deben near Woodbridge, England ...                                                                  |
| `mlx-community/Molmo-7B-D-0924-8bit`           |                   29 | `clean`          | [low-draft-improvement] Remove uncertain tags. Title: Sailing boats on the River Deben near Woodbridge, England ...                                                                  |
| `mlx-community/gemma-3-27b-it-qat-4bit`        |                   28 | `clean`          | [low-draft-improvement] Title: Two sailing boats moored on a riverbank ...                                                                                                           |

## Repository Variant Comparisons

Policy: quality-first (reliability-gated assisted enrichment) within matching repository families.
Evidence scope: 1 image, 1 current run per variant; scores and histories are not merged.

| Family                                      | Variant                                             | Eligible   |   Quality | Total   |   Peak GB |
|---------------------------------------------|-----------------------------------------------------|------------|-----------|---------|-----------|
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-6bit`                 | yes        |        59 | 17.55s  |      11   |
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-mxfp4`                | no         |        52 | 21.21s  |       8.4 |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | no         |        73 | 8.86s   |      13   |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | no         |        57 | 6.26s   |      13   |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-8bit`                | yes        |        58 | 28.52s  |      41   |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-bf16`                | yes        |        58 | 21.87s  |      48   |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-mxfp8`                   | no         |        66 | 103.01s |      38   |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-4bit`                    | no         |        66 | 106.12s |      26   |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-4bit`                | no         |        83 | 76.73s  |      26   |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-6bit`                | no         |        82 | 62.95s  |      35   |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-bf16`                | no         |        63 | 75.08s  |      76   |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-8bit`      | yes        |        62 | 9.35s   |      29   |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`     | no         |        60 | 11.37s  |      28   |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-4bit`             | yes        |        56 | 13.13s  |      18   |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-8bit`             | yes        |        56 | 11.81s  |      32   |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | no         |        22 | 4.25s   |      12   |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | no         |        22 | 7.83s   |      26   |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-bf16`                    | no         |        58 | 9.67s   |      28   |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-8bit`                    | no         |        58 | 7.02s   |      16   |
