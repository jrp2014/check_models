# Model Selection Brief

Generated on: 2026-07-17 23:16:01 BST

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

| Model                                                 | Task outcome            | Compatibility       | Recommendation eligibility                                 | Prompt burden   |   Gen TPS |   Peak GB |
|-------------------------------------------------------|-------------------------|---------------------|------------------------------------------------------------|-----------------|-----------|-----------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | Task outcome: completed | clean               | eligible                                                   | normal          |    510    |       1.5 |
| mlx-community/nanoLLaVA-1.5-4bit                      | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |    291    |       2.9 |
| mlx-community/LFM2-VL-1.6B-8bit                       | Task outcome: completed | clean               | eligible                                                   | normal          |    308    |       3   |
| HuggingFaceTB/SmolVLM-Instruct                        | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |    130    |       5.7 |
| mlx-community/SmolVLM-Instruct-bf16                   | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |    110    |       5.7 |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | Task outcome: completed | clean               | eligible                                                   | normal          |    185    |       4.1 |
| mlx-community/MiniCPM-V-4.6-8bit                      | Task outcome: completed | integration-warning | excluded: integration warning requires review              | normal          |    209    |       4   |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | Task outcome: completed | clean               | eligible                                                   | normal          |    104    |      17   |
| mlx-community/FastVLM-0.5B-bf16                       | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |    307    |       2.2 |
| mlx-community/paligemma2-10b-ft-docci-448-6bit        | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     29.6  |      12   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |    119    |       6.4 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     58.9  |      28   |
| mlx-community/diffusiongemma-26B-A4B-it-8bit          | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     59.7  |      29   |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |    130    |       5.5 |
| microsoft/Phi-3.5-vision-instruct                     | Task outcome: completed | clean               | eligible                                                   | normal          |     55.6  |       9.6 |
| mlx-community/Phi-3.5-vision-instruct-bf16            | Task outcome: completed | clean               | eligible                                                   | normal          |     52.8  |       9.6 |
| mlx-community/InternVL3-8B-bf16                       | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     34.3  |      18   |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     62.8  |      12   |
| mlx-community/pixtral-12b-8bit                        | Task outcome: completed | clean               | eligible                                                   | normal          |     38.9  |      15   |
| mlx-community/gemma-4-31b-bf16                        | Task outcome: completed | integration-warning | excluded: integration warning requires review              | normal          |      9.1  |      65   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     58.6  |      12   |
| mlx-community/llava-v1.6-mistral-7b-8bit              | Task outcome: completed | clean               | eligible                                                   | normal          |     58.9  |       9.7 |
| mlx-community/Idefics3-8B-Llama3-bf16                 | Task outcome: completed | clean               | excluded: output is below the configured chooser threshold | normal          |     32.2  |      19   |
| mlx-community/gemma-3n-E2B-4bit                       | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |    121    |       6.1 |
| mlx-community/gemma-3-27b-it-qat-4bit                 | Task outcome: completed | clean               | eligible                                                   | normal          |     30.8  |      18   |
| qnguyen3/nanoLLaVA                                    | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |    111    |       4.9 |
| mlx-community/InternVL3-14B-8bit                      | Task outcome: completed | clean               | eligible                                                   | normal          |     31.5  |      19   |
| mlx-community/paligemma2-10b-ft-docci-448-bf16        | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |      5.09 |      27   |
| mlx-community/gemma-4-31b-it-4bit                     | Task outcome: completed | clean               | eligible                                                   | normal          |     23.9  |      20   |
| mlx-community/pixtral-12b-bf16                        | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     20.4  |      27   |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | Task outcome: completed | clean               | eligible                                                   | normal          |     29.2  |      22   |
| mlx-community/GLM-4.6V-Flash-6bit                     | Task outcome: completed | clean               | eligible                                                   | visual_input    |     56.1  |      11   |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     19.4  |      15   |
| mlx-community/gemma-3-27b-it-qat-8bit                 | Task outcome: completed | clean               | eligible                                                   | normal          |     17.4  |      32   |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit                 | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     71    |      18   |
| mlx-community/Kimi-VL-A3B-Thinking-8bit               | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     60.5  |      22   |
| mlx-community/gemma-3n-E4B-it-bf16                    | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     41.4  |      17   |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | Task outcome: completed | clean               | excluded: current review says avoid                        | visual_input    |     77.1  |       8.4 |
| mlx-community/paligemma2-3b-pt-896-4bit               | Task outcome: completed | clean               | excluded: current review says avoid                        | visual_input    |     47.1  |       4.7 |
| Qwen/Qwen3-VL-2B-Instruct                             | Task outcome: completed | integration-warning | excluded: integration warning requires review              | mixed           |    179    |       8.6 |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | Task outcome: completed | clean               | excluded: current review says avoid                        | mixed           |     40.7  |      15   |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     44.9  |      60   |
| mlx-community/GLM-4.6V-nvfp4                          | Task outcome: completed | clean               | eligible                                                   | visual_input    |     40.8  |      78   |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |      5.09 |      25   |
| mlx-community/Molmo-7B-D-0924-8bit                    | Task outcome: completed | clean               | eligible                                                   | normal          |     46.5  |      41   |
| mlx-community/X-Reasoner-7B-8bit                      | Task outcome: completed | integration-warning | excluded: integration warning requires review              | mixed           |     68.5  |      14   |
| mlx-community/Molmo-7B-D-0924-bf16                    | Task outcome: completed | clean               | eligible                                                   | normal          |     29.2  |      48   |
| mlx-community/MolmoPoint-8B-fp16                      | Task outcome: completed | clean               | excluded: current review says avoid                        | visual_input    |      6.04 |      27   |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review              | mixed           |    139    |       8.6 |
| mlx-community/paligemma2-3b-ft-docci-448-bf16         | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |     18.6  |      11   |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review              | mixed           |     73.6  |       8.6 |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | Task outcome: completed | clean               | eligible                                                   | mixed           |     93.1  |      11   |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | Task outcome: completed | clean               | eligible                                                   | mixed           |     94.5  |      26   |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | Task outcome: completed | clean               | eligible                                                   | mixed           |     89.3  |      35   |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | Task outcome: completed | clean               | eligible                                                   | mixed           |     65.7  |      76   |
| mlx-community/Qwen3.6-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                                   | mixed           |     17.5  |      38   |
| mlx-community/Ornith-1.0-35B-bf16                     | Task outcome: completed | clean               | eligible                                                   | mixed           |     51.4  |      76   |
| mlx-community/Qwen3.5-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                                   | mixed           |     17.5  |      38   |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | Task outcome: completed | integration-warning | excluded: integration warning requires review              | mixed           |    110    |       5.1 |
| mlx-community/Qwen3.5-27B-4bit                        | Task outcome: completed | clean               | eligible                                                   | mixed           |     29    |      26   |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16          | Task outcome: completed | clean               | excluded: current review says avoid                        | normal          |      4.26 |      40   |

## Quick Chooser

Practical current-run buckets for model users. These are triage signals, not grounded visual-quality claims.

### Best under 4 GB

Policy: memory-aware (reliability-gated assisted enrichment; budget 4 GB).
Evidence scope: 1 image, 1 current run.

| Model                              |   Peak GB |   Gen TPS |   Usefulness | Status        | Evidence                                                                                         |
|------------------------------------|-----------|-----------|--------------|---------------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` |       1.5 |       510 |           60 | `caveat`      | [low-draft-improvement] Title: Deal Seafront, England, UK, GBR, Europe Description: A coastal... |
| `mlx-community/LFM2-VL-1.6B-8bit`  |       3   |       308 |           60 | `recommended` | [low-draft-improvement] Title: Deal Beach, Deal, Kent, UK, GBR, Europe Description: A coastal... |

### Best under 8 GB

Policy: memory-aware (reliability-gated assisted enrichment; budget 8 GB).
Evidence scope: 1 image, 1 current run.

| Model                               |   Peak GB |   Gen TPS |   Usefulness | Status        | Evidence                                                                                         |
|-------------------------------------|-----------|-----------|--------------|---------------|--------------------------------------------------------------------------------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16` |       4.1 |       185 |           60 | `caveat`      | [low-draft-improvement] Title: Deal, Kent, England, UK, GBR, Europe, Seaside, Beach, Shingle,... |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`  |       1.5 |       510 |           60 | `caveat`      | [low-draft-improvement] Title: Deal Seafront, England, UK, GBR, Europe Description: A coastal... |
| `mlx-community/LFM2-VL-1.6B-8bit`   |       3   |       308 |           60 | `recommended` | [low-draft-improvement] Title: Deal Beach, Deal, Kent, UK, GBR, Europe Description: A coastal... |

### Fastest usable

Policy: efficiency-aware Pareto frontier (reliability-gated).
Evidence scope: 1 image, 1 current run.

| Model                               |   Peak GB |   Gen TPS |   Usefulness | Status        | Evidence                                                                                         |
|-------------------------------------|-----------|-----------|--------------|---------------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`  |       1.5 |     510   |           60 | `caveat`      | [low-draft-improvement] Title: Deal Seafront, England, UK, GBR, Europe Description: A coastal... |
| `mlx-community/LFM2.5-VL-1.6B-bf16` |       4.1 |     185   |           60 | `caveat`      | [low-draft-improvement] Title: Deal, Kent, England, UK, GBR, Europe, Seaside, Beach, Shingle,... |
| `mlx-community/gemma-4-31b-it-4bit` |      20   |      23.9 |           87 | `recommended` | [low-draft-improvement] Title: Seafront buildings and shingle beach in Deal, England Descript... |

### Quality if memory allows

Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status        | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------|--------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-35B-A3B-bf16` |        76 |      65.7 |           93 | `recommended` | Title: Shingle beach, seafront buildings, and sea at Deal ...                                    |
| `mlx-community/Ornith-1.0-35B-bf16`  |        76 |      51.4 |          100 | `caveat`      | Title: Seafront buildings and shingle beach at low tide Description: A row of multi-storey se... |
| `mlx-community/Qwen3.6-27B-mxfp8`    |        38 |      17.5 |           93 | `caveat`      | Title: Shingle beach and seafront buildings in Deal, England ...                                 |

### Current failures / avoid

Policy: reliability-gated exclusion evidence.
Evidence scope: 1 image, 1 current run.

| Model                                 |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                                                                                                                                               |
|---------------------------------------|-----------|-----------|--------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/nanoLLaVA-1.5-4bit`    |       2.9 |       291 |           60 | `avoid`  | missing sections: keywords \| missing terms: Cars, Coastline, Horizon, Landscape, People \| low-draft-improvement \| current review says avoid                                                                         |
| `HuggingFaceTB/SmolVLM-Instruct`      |       5.7 |       130 |           46 | `avoid`  | missing sections: title, description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Horizon \| low-draft-improvement \| current review says avoid                                                      |
| `mlx-community/SmolVLM-Instruct-bf16` |       5.7 |       110 |           46 | `avoid`  | missing sections: title, description, keywords \| missing terms: Beach, Buildings, Cars, Coastline, Horizon \| low-draft-improvement \| current review says avoid                                                      |
| `mlx-community/MiniCPM-V-4.6-8bit`    |       4   |       209 |           93 | `avoid`  | Special control token &lt;/think&gt; appeared in generated text. \| missing terms: Cars, Coastline, Deal, Horizon, Kent \| reasoning leak \| text-sanity=gibberish(token_noise) \| integration warning requires review |
| `mlx-community/FastVLM-0.5B-bf16`     |       2.2 |       307 |           66 | `avoid`  | missing sections: keywords \| missing terms: Horizon, Landscape, Promenade, Sitting, Swimming \| low-draft-improvement \| current review says avoid                                                                    |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.
Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                   |   Hygiene |   Usefulness |   Gen TPS |   Peak GB | Verdict       | Caption Preview                                                                                                                                                                      | Caveat                                                                                     |
|-----------------------------------------|-----------|--------------|-----------|-----------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-35B-A3B-bf16`    |       100 |           93 |      65.7 |        76 | `recommended` | Title: Shingle beach, seafront buildings, and sea at Deal ...                                                                                                                        | missing terms: Horizon, Kent, Sitting, Swimming, Walking                                   |
| `mlx-community/Ornith-1.0-35B-bf16`     |        75 |          100 |      51.4 |        76 | `caveat`      | Title: Seafront buildings and shingle beach at low tide Description: A row of multi-storey seafront buildings lines a ... [tail] th white cliffs visible on the distant horizon. ... | missing terms: Cars, Deal, Swimming, architecture, GBR \| keywords=20                      |
| `mlx-community/Qwen3.6-27B-mxfp8`       |        75 |           93 |      17.5 |        38 | `caveat`      | Title: Shingle beach and seafront buildings in Deal, England ...                                                                                                                     | missing terms: Kent, Promenade, Seaside, Swimming, Walking \| keywords=19                  |
| `mlx-community/Qwen3.5-35B-A3B-4bit`    |        60 |           93 |      94.5 |        26 | `caveat`      | Title: Shingle beach and seafront buildings under cloudy sky Description: A wide shingle beach stretches along the co ... [tail] s rolling onto the shore under an overcast sky. ... | missing terms: Deal, Horizon, Kent, Swimming, Walking                                      |
| `mlx-community/Qwen3.5-27B-4bit`        |       100 |           99 |      29   |        26 | `recommended` | Title: Shingle Beach and Seafront Buildings in Deal, Kent ...                                                                                                                        | missing terms: Seaside, Shore, Walking, Water, architecture                                |
| `mlx-community/Qwen3.5-35B-A3B-6bit`    |       100 |           93 |      89.3 |        35 | `recommended` | Title: Shingle beach and seafront buildings at Deal ...                                                                                                                              | missing terms: Kent, Sitting, Swimming, Walking, GBR                                       |
| `mlx-community/Molmo-7B-D-0924-bf16`    |        60 |           87 |      29.2 |        48 | `caveat`      | [low-draft-improvement] Remove uncertain tags. Title: Seafront, Deal, Kent, UK, GBR, Europe ...                                                                                      | missing terms: Horizon \| keywords=23 \| low-draft-improvement                             |
| `mlx-community/Qwen3.5-27B-mxfp8`       |        75 |          100 |      17.5 |        38 | `caveat`      | [low-draft-improvement] Title: Shingle Beach and Seafront Buildings, Deal, England ...                                                                                               | missing terms: Kent, Sitting, Walking, GBR, view \| keywords=20 \| low-draft-improvement   |
| `mlx-community/gemma-4-31b-it-4bit`     |       100 |           87 |      23.9 |        20 | `recommended` | [low-draft-improvement] Title: Seafront buildings and shingle beach in Deal, England Description: A coastal view shows a row of multi-storey buildings facing a shingle beach and... | missing terms: Sitting, Swimming, Walking, GBR, showing \| low-draft-improvement           |
| `mlx-community/gemma-3-27b-it-qat-8bit` |        75 |           93 |      17.4 |        32 | `caveat`      | [low-draft-improvement] Title: Seafront buildings, shingle beach, and waves ...                                                                                                      | missing terms: Cars, Deal, Kent, Sitting, Swimming \| keywords=19 \| low-draft-improvement |

## Structured Metadata Candidates

Top 10 ranked candidates for structured title/description/keywords. Use the gallery for complete per-model evidence.
Policy: quality-first (reliability-gated assisted enrichment).
Evidence scope: 1 image, 1 current run.

| Model                                   |   Metadata agreement | Verdict   | Output Preview                                                                                                                                                                       |
|-----------------------------------------|----------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-35B-A3B-bf16`    |                   36 | `clean`   | Title: Shingle beach, seafront buildings, and sea at Deal ...                                                                                                                        |
| `mlx-community/Ornith-1.0-35B-bf16`     |                   30 | `clean`   | Title: Seafront buildings and shingle beach at low tide Description: A row of multi-storey seafront buildings lines a ... [tail] th white cliffs visible on the distant horizon. ... |
| `mlx-community/Qwen3.6-27B-mxfp8`       |                   37 | `clean`   | Title: Shingle beach and seafront buildings in Deal, England ...                                                                                                                     |
| `mlx-community/Qwen3.5-35B-A3B-4bit`    |                   31 | `clean`   | Title: Shingle beach and seafront buildings under cloudy sky Description: A wide shingle beach stretches along the co ... [tail] s rolling onto the shore under an overcast sky. ... |
| `mlx-community/Qwen3.5-27B-4bit`        |                   37 | `clean`   | Title: Shingle Beach and Seafront Buildings in Deal, Kent ...                                                                                                                        |
| `mlx-community/Qwen3.5-35B-A3B-6bit`    |                   37 | `clean`   | Title: Shingle beach and seafront buildings at Deal ...                                                                                                                              |
| `mlx-community/Molmo-7B-D-0924-bf16`    |                   56 | `clean`   | [low-draft-improvement] Remove uncertain tags. Title: Seafront, Deal, Kent, UK, GBR, Europe ...                                                                                      |
| `mlx-community/Qwen3.5-27B-mxfp8`       |                   38 | `clean`   | [low-draft-improvement] Title: Shingle Beach and Seafront Buildings, Deal, England ...                                                                                               |
| `mlx-community/gemma-4-31b-it-4bit`     |                   39 | `clean`   | [low-draft-improvement] Title: Seafront buildings and shingle beach in Deal, England Description: A coastal view shows a row of multi-storey buildings facing a shingle beach and... |
| `mlx-community/gemma-3-27b-it-qat-8bit` |                   28 | `clean`   | [low-draft-improvement] Title: Seafront buildings, shingle beach, and waves ...                                                                                                      |

## Repository Variant Comparisons

Policy: quality-first (reliability-gated assisted enrichment) within matching repository families.
Evidence scope: 1 image, 1 current run per variant; scores and histories are not merged.

| Family                                      | Variant                                             | Eligible   |   Quality | Total   |   Peak GB |
|---------------------------------------------|-----------------------------------------------------|------------|-----------|---------|-----------|
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-6bit`                 | yes        |        46 | 9.33s   |      11   |
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-mxfp4`                | no         |        78 | 14.25s  |       8.4 |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | no         |        70 | 6.44s   |      12   |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | no         |        68 | 6.16s   |      12   |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-bf16`                | yes        |        70 | 23.69s  |      48   |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-8bit`                | yes        |        62 | 18.30s  |      41   |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-4bit`                    | yes        |        75 | 103.19s |      26   |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-mxfp8`                   | yes        |        66 | 93.21s  |      38   |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-bf16`                | yes        |        80 | 82.87s  |      76   |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-4bit`                | yes        |        76 | 61.40s  |      26   |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-6bit`                | yes        |        74 | 64.70s  |      35   |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-8bit`      | no         |        60 | 6.40s   |      29   |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`     | no         |        59 | 6.17s   |      28   |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-8bit`             | yes        |        63 | 11.67s  |      32   |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-4bit`             | yes        |        51 | 7.70s   |      18   |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | no         |        21 | 4.33s   |      12   |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | no         |        21 | 8.81s   |      27   |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-8bit`                    | yes        |        48 | 6.60s   |      15   |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-bf16`                    | no         |        49 | 9.46s   |      27   |
