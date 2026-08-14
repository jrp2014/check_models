# Model Output Gallery

Generated on: 2026-08-14 18:45:05 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                                                                     |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | 1.47s   | 515 tok/s  | 0.06            | 1.2     | 355        | 76      | none                                                                                                                                             |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | 2.87s   | 156 tok/s  | 0.12            | 4.1     | 555        | 122     | none                                                                                                                                             |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | 13.46s  | 11.7 tok/s | 2.13            | 15      | 275        | 101     | none                                                                                                                                             |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 8.19s   | 45.6 tok/s | 3.49            | 12      | 2,730      | 120     | none                                                                                                                                             |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 4.29s   | 144 tok/s  | 1.15            | 7.1     | 2,729      | 154     | none                                                                                                                                             |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | 93.40s  | 60.1 tok/s | 77.26           | 74      | 16,548     | 122     | none                                                                                                                                             |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 87.25s  | 108 tok/s  | 82.39           | 24      | 16,548     | 121     | none                                                                                                                                             |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 75.86s  | 87.7 tok/s | 72.11           | 10.0    | 16,548     | 120     | none                                                                                                                                             |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.14s   | 129 tok/s  | 0.42            | 16      | 587        | 94      | none                                                                                                                                             |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 7.48s   | 27.2 tok/s | 1.05            | 20      | 587        | 89      | none                                                                                                                                             |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | 17.08s  | 94.2 tok/s | 13.96           | 8.4     | 16,533     | 147     | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | 9.83s   | 29.6 tok/s | 3.21            | 22      | 2,197      | 112     | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | 10.58s  | 74.5 tok/s | 7.67            | 8.4     | 6,281      | 73      | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | 26.92s  | 37.5 tok/s | 16.81           | 77      | 6,281      | 78      | control tokens visible; title/keyword constraints failed                                                                                         |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | 8.22s   | 30.3 tok/s | 1.27            | 18      | 2,586      | 126     | role tokens visible                                                                                                                              |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable_with_caveats` | 6.06s   | 34.1 tok/s | 0.96            | 17      | 2,087      | 93      | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable_with_caveats` | 2.61s   | 238 tok/s  | 0.27            | 3.8     | 895        | 75      | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | 10.20s  | 41.2 tok/s | 3.52            | 13      | 2,730      | 175     | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | 6.24s   | 38.8 tok/s | 0.91            | 11      | 1,487      | 107     | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable_with_caveats` | 3.31s   | 56.3 tok/s | 0.30            | 9.6     | 1,094      | 84      | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | 32.29s  | 83.5 tok/s | 27.29           | 8.4     | 16,533     | 147     | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable_with_caveats` | 114.12s | 14.4 tok/s | 97.68           | 35      | 16,548     | 121     | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable_with_caveats` | 39.57s  | 39.9 tok/s | 22.11           | 70      | 3,468      | 116     | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | 5.96s   | 55.3 tok/s | 0.43            | 29      | 583        | 82      | control tokens visible                                                                                                                           |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | 5.91s   | 53.0 tok/s | 0.43            | 28      | 583        | 81      | control tokens visible; title/keyword constraints failed                                                                                         |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | 8.98s   | 26.6 tok/s | 0.99            | 18      | 569        | 136     | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | 6.59s   | 39.6 tok/s | 1.88            | 15      | 2,828      | 96      | title/keyword constraints failed                                                                                                                 |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 28.00s  | 41.9 tok/s | 1.86            | 14      | 2,919      | 1,000   | missing required fields; echoes instructions; cut off at token limit                                                                             |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 32.00s  | 43.7 tok/s | 2.21            | 60      | 1,569      | 1,000   | repeated text; missing required fields; cut off at token limit; incomplete thinking block                                                        |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 2.45s   | 338 tok/s  | 0.09            | 2.1     | 308        | 120     | missing required fields; echoes instructions                                                                                                     |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 34.74s  | 40.6 tok/s | 7.77            | 13      | 6,277      | 1,000   | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | 220.02s | 4.66 tok/s | 1.41            | 40      | 1,269      | 1,000   | repeated text; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | 40.08s  | 5.64 tok/s | 4.15            | 24      | 3,100      | 183     | missing required fields                                                                                                                          |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `unusable`            | 100.13s | 144 tok/s  | 98.08           | 5.1     | 16,544     | 98      | missing required fields                                                                                                                          |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | 44.58s  | 68.1 tok/s | 27.41           | 8.4     | 16,535     | 1,000   | missing required fields; cut off at token limit; incomplete thinking block                                                                       |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | 44.52s  | 46.8 tok/s | 21.09           | 13      | 16,544     | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                                                          |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 7.46s   | 47.4 tok/s | 0.39            | 17      | 568        | 199     | missing required fields                                                                                                                          |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | 5.48s   | 60.6 tok/s | 2.90            | 9.7     | 2,386      | 53      | missing required fields                                                                                                                          |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 1.56s   | 351 tok/s  | 0.08            | 2.4     | 304        | 130     | missing required fields; echoes instructions                                                                                                     |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 24.82s  | 46.0 tok/s | 1.09            | 4.4     | 4,397      | 1,000   | repeated text; missing required fields; cut off at token limit                                                                                   |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `not_evaluated`       | 2.03s   | -          | -               | -       | -          | -       | none                                                                                                                                             |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 515 tok/s

Average clean-completion throughput: 128 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.2 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | missing required fields; echoes instructions; cut off at token limit                                                                             |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable`      | repeated text; missing required fields; cut off at token limit; incomplete thinking block                                                        |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable`      | missing required fields; echoes instructions                                                                                                     |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable`      | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | `unusable`      | repeated text; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                | `unusable`      | missing required fields                                                                                                                          |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                  | `unusable`      | missing required fields                                                                                                                          |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable`      | missing required fields; cut off at token limit; incomplete thinking block                                                                       |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                                          |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                            | `unusable`      | missing required fields                                                                                                                          |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable`      | missing required fields                                                                                                                          |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable`      | missing required fields; echoes instructions                                                                                                     |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable`      | repeated text; missing required fields; cut off at token limit                                                                                   |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                 | `not_evaluated` | none                                                                                                                                             |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                                    |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | Title:<br>Seafront, Felixstowe, England, UK, GBR, Europe<br><br>Description:<br>A concrete seawall stretches along the coastline of Felixstowe, England, with people enjoying the beach and the sea.<br><br>Keywords:<br>Seafront, Felixstowe, England, UK, GBR, Europe, beach, sea, walkway, coastal scenery                     |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | Title: Seafront at Felixstowe, England, UK, GBR<br><br>Description: A vibrant seaside scene in Felixstowe, England, UK, GBR, captures the lively atmosphere of a popular beachfront. Visitors enjoy the sun, sand, and sea, with the backdrop of historic buildings and lush greenery. The...                                     |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | Title: Summer Afternoon at the Seaside in Felixstove, Suffolk, England<br><br>Description: A serene summer afternoon scene unfolds at the seaside in Felixstove, Suffolk, England, with people enjoying the sun, sea, and sand, surrounded by the picturesque town's buildings and lush gre...                                    |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**Felixstowe Seafront Rocks and Beachgoers, August**<br><br>Description:<br>A sunny August afternoon at Felixstowe’s seafront reveals people swimming, wading, and relaxing among the distinctive concrete and rock formations along the tide line, with the town’s green hills and his...                              |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>Felixstowe Seafront Rocks, Golden Hour<br><br>Description:<br>A lively coastal scene at Felixstowe’s seafront on 2026-08-13, featuring stepped concrete breakwaters teeming with swimmers and waders, backed by a hilly residential area under clear blue skies. The warm lighting high...                              |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | Title: Sunny day at Felixstowe seafront with swimmers and rock pools<br>Description: People swim and play in the calm sea and rock pools along the Felixstowe seafront on a bright summer afternoon. The scene features tiered concrete sea defences, a rocky breakwater, and the town's...                                       |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Sunny Seafront at Felixstowe with Rock Pools<br><br>Description: A sunny afternoon scene at Felixstowe, Suffolk, showing people swimming and playing in the calm blue sea and rock pools. The foreground features weathered concrete groynes and a sandy beach, while the background...                                    |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | Title:<br>Felixstowe Seafront, August 2026, UK<br><br>Description:<br>A sunny day at Felixstowe seafront with people enjoying the water and rocky shoreline under clear skies. The scene includes coastal buildings, a pebble beach, and scattered seagulls near the tide pools.<br><br>Keywords:<br>Felix...                     |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: Seafront with stone groynes in Felixstowe, England<br>Description: People enjoy the seaside at a beach in Felixstowe, England, where stone groynes extend into the blue sea under clear daylight. The coastal town features buildings and greenery overlooking the coastline.<br>Key...                                    |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: Seafront and coastal defences at Felixstowe, Suffolk<br>Description: People enjoy the beach and sea along the seafront in Felixstowe, Suffolk, featuring stepped concrete sea walls and rock armour under a clear late afternoon sky.<br>Keywords: Felixstowe, Suffolk, East Suffolk...                                    |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | Title:<br>Felixstowe Seafront, England<br><br>Description:<br>A bustling seafront in Felixstowe, England, on a sunny day, with people enjoying the beach, swimming, and playing near the rocky breakwaters. The scene is set against a backdrop of historic buildings and lush greenery, with the...                              |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | Title:<br>Felixstowe Seafront, England, UK<br><br>Description:<br>A lively seafront scene in Felixstowe, England, on a sunny day, with people enjoying the beach, swimming in the sea, and walking along the rocky shore, backed by a hillside of houses and greenery.<br><br>Keywords:<br>Felixstowe, Eng...                     |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | Title: Felixstowe Seafront<br>Description: A sunny scene at Felixstowe's seafront, with people enjoying the beach and water, backed by coastal buildings and greenery.<br>Keywords: Felixstowe seafront, England UK, beach scene, coastal buildings, sunny weather, people swimming, East S...                                    |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Felixstowe Seafront, England<br>Description: A sunny seafront scene in Felixstowe, England, with people enjoying the beach, rocky shoreline, and coastal buildings under clear skies.<br>Keywords: Felixstowe, seafront, England, UK, East Suffolk, Suffolk, Europe,...                                  |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | Title: Felixstowe Seafront in Suffolk, England, UK.<br>Description: The seafront in Felixstowe, Suffolk, England, UK, is bustling with activity as people enjoy the sunny day. The sandy beach is lined with large rocks, and the clear blue water of the sea laps gently against the sh...                                       |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable_with_caveats` | Title: Felixstowe Seafront at Sunset<br>Description: A bustling Felixstowe seafront on 2026-08-13, with people enjoying the beach and sea under clear skies, featuring historic buildings and a rocky breakwater.<br>Keywords: Felixstowe, seafront, Suffolk, England, UK, East Suffolk, be...                                    |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable_with_caveats` | &lt;think&gt;<br><br>&lt;/think&gt;<br><br>Title:<br>Felixstowe Seafront Beach Activity<br><br>Description:<br>A lively seafront scene with people enjoying the water and sand under bright sunlight.<br><br>Keywords: Felixstowe, seafront, beach, activity, sunlight, water, people, rocks, greenery, buildings, Europe, UK,... |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | **Title:**<br>*Felixstowe Seafront Beach, UK – Summer Activity*<br><br>**Description:**<br>A lively seaside scene at Felixstowe’s seafront beach on **August 13, 2026 (17:23 UTC+1)**, featuring swimmers, waders, and sunbathers along rocky tide pools and sandy shores. Behind them, a bustlin...                              |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | Title: Felixstowe Seaside Scene, England<br><br>Description: A bustling Felixstowe beach with people enjoying the water and sand, backed by a rocky jetty and framed by trees and buildings under a clear blue sky.<br><br>Keywords: Seafront, Felixstowe, England, UK, GBR, East Suffolk, Englan...                              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable_with_caveats` | Title: Felixstowe Seafront, England<br><br>Description: A serene scene at Felixstowe seafront on August 13, 2026, with people enjoying the beach and the sea, captured on a sunny day.<br><br>Keywords: Felixstowe, seafront, England, UK, GBR, beach, people, sea, sky, rocks, coastline                                         |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | Title:<br>Felixstowe Seafront, England<br><br>Description:<br>A bustling seafront in Felixstowe, England, on a sunny day, with people enjoying the beach, swimming, and playing near the rocky breakwaters. The scene is set against a backdrop of historic buildings and lush greenery, with the...                              |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable_with_caveats` | Title:<br>Seafront at Felixstowe, Suffolk, England<br><br>Description:<br>A sunny late afternoon view of the Felixstowe seafront in Suffolk, England, showing people swimming and playing on the stepped concrete groynes and pebble beach. The background features a hillside lined with residen...                              |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable_with_caveats` | Title:<br>Felixstowe Seafront, Suffolk, England<br><br>Description:<br>A sunny day at the Felixstowe seafront in Suffolk, England, with people swimming, sunbathing, and playing on the beach alongside a rocky breakwater, set against a backdrop of coastal buildings and greenery under a clea...                              |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Sunny seafront at Felixstowe in Suffolk, England<br>Description: People enjoy a sunny day at the Felixstowe seafront, where stone groynes meet the sea under a clear blue sky.<br>Keywords: Felixstowe, Suffolk, England, UK, East Suffolk, seafront, be...                               |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Sunny day at the seafront in Felixstowe, England<br>Description: People enjoy a sunny day at the stone-tiered sea beach in Felixstowe, Suffolk, under a clear blue sky.<br>Keywords: Felixstowe, Suffolk, East Suffolk, England, UK, seafront, beach, se...                               |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | Title: Felixstowe Seafront, Suffolk: Beach and Promenade View<br>Description: Captured on 13th August 2026 at 17:23, this image shows the busy seafront at Felixstowe, Suffolk, with people enjoying the beach, sea and distinctive groynes under a clear blue sky. The promenade buildi...                                       |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | Title:<br>People Enjoying Seafront at Felixstowe, Suffolk, UK<br><br>Description:<br>On a sunny day, people relax and play by the seafront in Felixstowe, Suffolk, UK, with clear skies and calm waters.<br><br>Keywords:<br>Felixstowe, Suffolk, UK, seafront, beach, people, sea, sunny, clear skies, ca...                     |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS. The descriptive hints are given. We need to treat capture date/time and GPS as authoritative facts, but not c...                                       |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. The hints are pretty clear: "Seafront, Felixstowe, England, UK, GBR, Europe." I need something concise, so I'll go with "Seafront at Felixstowe, England, UK." Tha...                                          |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A concrete 5-10-word title: "Felixstowe, UK - Seafront, GBR, East Suffolk, UK, gbr, seafront"<br>A 1-2-sentence factual description: "A vibrant seaside scene with a concrete 5-10-word title, featuring a concrete 5-10-word title, a 1-2-sentence factual description combining releva...                                       |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image, it's a seafront with people, buildings, and the coastal area. So maybe "Felixstowe Seafront Beach Scene" or something similar. Wait, need to be concrete. Let's go with "Felixstowe...                                     |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | ◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.<br><br>The image shows a seafront in Felixstowe, England, UK. The capture date is 2026-08-13...                                     |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | Felixstowe Seafront, England: Rocky Shore with People and Buildings<br><br>A bustling Felixstowe seafront scene captured on August 13, 2026, at 17:23 UTC+01:00. The image shows a rocky shoreline with people enjoying the water, a prominent clock tower, and a mix of residential and co...                                    |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `unusable`            | Title:<br>Seafront, Felixstowe, England, UK, GBR, Europe<br><br>Description:<br>The image depicts a lively seafront scene in Felixstowe, England. The beach is bustling with people enjoying the sunny weather, and the coastline is lined with rocks and greenery. The background features a mix...                              |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | Got it, let's tackle this step by step. First, the title. The context says Felixstowe, England, UK, so we need a concise title. The image shows a seafront with people, so maybe "Felixstowe Seafront, Suffolk, England". Wait, the title hint was Seafront, Felixstowe, England, UK,...                                          |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | Title:<br>Felixstowe Seafront, East Suffolk, England<br><br>Description:<br>A vibrant seafront scene in Felixstowe, East Suffolk, England, with people enjoying the sunny day at the beach and rocky promenade, set against a backdrop of historic buildings and lush greenery. The image capture...                              |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | A wide shot captures a sunny day at Seafront in Felixstowe, England. The scene is bustling with people enjoying the coastal area. The foreground features a series of low stone walls leading down to the water, where numerous people are wading, swimming, and relaxing. Some indiv...                                          |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | Seafront, Felixstowe, England, UK, GBR, Europe, East Suffolk, England, UK, gbr, seafront, Adobe Stock, Any Vision, Felixstowe, Suffolk, UK, gbr,                                                                                                                                                                                  |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | Title: Seafront, Felixstowe, England, UK, GBR, Europe<br>Description: A 5-10-word title that describes the image and includes relevant context, such as the location, the photographer's watermark, and the image's metadata. The description also includes the main visible subject, se...                                       |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | - Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>- Authoritative context:<br>-...          |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `not_evaluated`       | Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: not enough values to unpack (expected 3, got 2)                                                                                                                                                                                                             |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.14`
- `mlx`: `0.32.1.dev20260814+3d23f7d87`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.15.0`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.27.0`
- *Python Version:* 3.13.14
- *OS:* Darwin 25.6.0
- *macOS Version:* 26.6.1
- *GPU/Chip:* Apple M5 Max
- *MLX Device:* Apple M5 Max
- *GPU Architecture:* applegpu_g17s
- *RAM:* 128.0 GB
- *Recommended Working Set:* 108 GB
- *Fused Attention:* Available

## Image Metadata

- *Title:* , Seafront, Felixstowe, England, UK, GBR, Europe
- *Description:* , Seafront, Felixstowe, England, UK, GBR
- *Keywords:* Adobe Stock, Any Vision, East Suffolk, England, Europe,
  Felixstowe, Suffolk, UK, gbr, seafront
- *Date:* 2026-08-13 17:23:18 UTC+01:00
- *Time:* 17:23:18
- *GPS:* 51.957967°N, 1.346900°E

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Create British-English catalogue metadata from the image and supplied
> context.
>
> Treat any capture date/time and GPS as authoritative facts, but do not claim
> they are visible. Descriptive hints may be incomplete or wrong: retain
> details supported by the image, correct conflicts, and add important visible
> details. Prefer image evidence when a hint conflicts, and omit uncertain
> details.
>
> Context: Authoritative context:
> &#45; Capture date/time: 2026-08-13 17:23:18 UTC+01:00
> &#45; GPS: 51.957967°N, 1.346900°E
>
> &#8203;Descriptive hints:
> &#45; Title hint: Seafront, Felixstowe, England, UK, GBR, Europe
> &#45; Description hint: Seafront, Felixstowe, England, UK, GBR
> &#45; Keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe,
> Felixstowe, Suffolk, UK, gbr, seafront
>
> &#8203;Write:
> &#45; a concrete 5-10-word title;
> &#45; a 1-2-sentence factual description combining relevant context with the
> main visible subject, setting, action, lighting, and distinctive details;
> &#45; 10-18 unique, comma-separated keywords covering relevant context and
> visible details.
>
> &#8203;Return exactly these three sections and nothing else:
> &#8203;Title:
> &#8203;Description:
> &#8203;Keywords:
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Complete Per-model Evidence

Complete generated or crash evidence for every attempted model.

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

<details>
<summary>Complete evidence: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.46s
- *Generation time:* 0.64s
- *Total time:* 1.47s
- *Input validation time:* 0.368
- *Prompt preparation time:* 0.00603
- *First-token latency:* 0.0608
- *Cleanup time:* 0.0595
- *Prompt tokens:* 355
- *Generation tokens:* 76
- *Total tokens:* 431
- *Prompt throughput (raw):* 5,835 tok/s
- *Generation throughput (raw):* 515 tok/s
- *Peak memory:* 1.2
- *Active memory:* 0.90
- *Cache memory:* 0.02
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1183
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* ed71acdae0799c98cb450787d049d1343b65cd52
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-450M-MLX-bf16/snapshots/ed71acdae0799c98cb450787d049d1343b65cd52
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Seafront, Felixstowe, England, UK, GBR, Europe

Description:
A concrete seawall stretches along the coastline of Felixstowe, England, with people enjoying the beach and the sea.

Keywords:
Seafront, Felixstowe, England, UK, GBR, Europe, beach, sea, walkway, coastal scenery
</pre>

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### mlx-community/LFM2.5-VL-1.6B-bf16

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 1.09s
- *Generation time:* 1.77s
- *Total time:* 2.87s
- *Input validation time:* 9.39e-05
- *Prompt preparation time:* 0.00735
- *First-token latency:* 0.123
- *Cleanup time:* 0.132
- *Prompt tokens:* 555
- *Generation tokens:* 122
- *Total tokens:* 677
- *Prompt throughput (raw):* 4,523 tok/s
- *Generation throughput (raw):* 156 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.02
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.00379
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1183
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-1.6B-bf16/snapshots/16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Seafront at Felixstowe, England, UK, GBR

Description: A vibrant seaside scene in Felixstowe, England, UK, GBR, captures the lively atmosphere of a popular beachfront. Visitors enjoy the sun, sand, and sea, with the backdrop of historic buildings and lush greenery. The clear blue sky and gentle waves add to the picturesque setting.

Keywords: Felixstowe, England, UK, GBR, Seafront, Beach, Sun, Sand, Sea, Historic Buildings, Greenery, Clear Sky, Waves
</pre>

</details>

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

<details>
<summary>Complete evidence: mlx-community/Llama-3.2-11B-Vision-Instruct-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 1.90s
- *Generation time:* 11.55s
- *Total time:* 13.46s
- *Input validation time:* 6.26e-05
- *Prompt preparation time:* 0.00175
- *First-token latency:* 2.13
- *Cleanup time:* 0.155
- *Prompt tokens:* 275
- *Generation tokens:* 101
- *Total tokens:* 376
- *Prompt throughput (raw):* 129 tok/s
- *Generation throughput (raw):* 11.7 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.59
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00431
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1236
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-11B-Vision-Instruct-8bit/snapshots/8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|eot_id\|>

*Readable output:*

<pre class="model-output-readable">
Title: Summer Afternoon at the Seaside in Felixstove, Suffolk, England

Description: A serene summer afternoon scene unfolds at the seaside in Felixstove, Suffolk, England, with people enjoying the sun, sea, and sand, surrounded by the picturesque town's buildings and lush greenery, under a clear blue sky.

Keywords: seaside, summer, afternoon, Felixstove, Suffolk, England, people, sun, sea, sand, buildings, greenery, blue sky.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Summer Afternoon at the Seaside in Felixstove, Suffolk, England

Description: A serene summer afternoon scene unfolds at the seaside in Felixstove, Suffolk, England, with people enjoying the sun, sea, and sand, surrounded by the picturesque town's buildings and lush greenery, under a clear blue sky. 

Keywords: seaside, summer, afternoon, Felixstove, Suffolk, England, people, sun, sea, sand, buildings, greenery, blue sky.
```

</details>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-mxfp4</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.46s
- *Generation time:* 6.71s
- *Total time:* 8.19s
- *Input validation time:* 5.92e-05
- *Prompt preparation time:* 0.00733
- *First-token latency:* 3.49
- *Cleanup time:* 0.143
- *Prompt tokens:* 2,730
- *Generation tokens:* 120
- *Total tokens:* 2,850
- *Prompt throughput (raw):* 782 tok/s
- *Generation throughput (raw):* 45.6 tok/s
- *Peak memory:* 12
- *Active memory:* 8.0
- *Cache memory:* 0.56
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.0051
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3570
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 7c992876448fc5426841a85c6eb951b056fe38d2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-mxfp4/snapshots/7c992876448fc5426841a85c6eb951b056fe38d2
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
**Felixstowe Seafront Rocks and Beachgoers, August**

Description:
A sunny August afternoon at Felixstowe’s seafront reveals people swimming, wading, and relaxing among the distinctive concrete and rock formations along the tide line, with the town’s green hills and historic buildings rising in the background.

Keywords:
Felixstowe, Suffolk, England, UK, GBR, Europe, seafront, coastal rocks, beachgoers, tide pools, summer swimming, historic seaside town, East Anglia, clear skies, daylight, natural and man-made landscape
</pre>

</details>

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### mlx-community/Ministral-3-3B-Instruct-2512-4bit

<details>
<summary>Complete evidence: mlx-community/Ministral-3-3B-Instruct-2512-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.47s
- *Generation time:* 2.80s
- *Total time:* 4.29s
- *Input validation time:* 6.78e-05
- *Prompt preparation time:* 0.00786
- *First-token latency:* 1.15
- *Cleanup time:* 0.13
- *Prompt tokens:* 2,729
- *Generation tokens:* 154
- *Total tokens:* 2,883
- *Prompt throughput (raw):* 2,364 tok/s
- *Generation throughput (raw):* 144 tok/s
- *Peak memory:* 7.1
- *Active memory:* 2.8
- *Cache memory:* 0.38
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00562
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3569
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* a962dcb09eee4169c890e544c9eb938f1113fdee
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-3B-Instruct-2512-4bit/snapshots/a962dcb09eee4169c890e544c9eb938f1113fdee
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Felixstowe Seafront Rocks, Golden Hour

Description:
A lively coastal scene at Felixstowe’s seafront on 2026-08-13, featuring stepped concrete breakwaters teeming with swimmers and waders, backed by a hilly residential area under clear blue skies. The warm lighting highlights activity along the shore and rocky edges, with a mix of leisure and nature.

Keywords:
Felixstowe, Suffolk, UK, GBR, seafront, coastal breakwaters, stepped concrete, beachgoers, wading birds, residential hillside, golden hour lighting, East Anglian seaside, historic stone buildings, rocky shore, summer day, Atlantic Ocean, leisure activities.
</pre>

</details>

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### mlx-community/Ornith-1.0-35B-bf16

<details>
<summary>Complete evidence: mlx-community/Ornith-1.0-35B-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 13.07s
- *Generation time:* 80.30s
- *Total time:* 93.40s
- *Input validation time:* 0.00013
- *Prompt preparation time:* 0.0139
- *First-token latency:* 77.3
- *Cleanup time:* 0.279
- *Prompt tokens:* 16,548
- *Generation tokens:* 122
- *Total tokens:* 16,670
- *Prompt throughput (raw):* 214 tok/s
- *Generation throughput (raw):* 60.1 tok/s
- *Peak memory:* 74
- *Active memory:* 70
- *Cache memory:* 0.89
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1223
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-bf16/snapshots/9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sunny day at Felixstowe seafront with swimmers and rock pools
Description: People swim and play in the calm sea and rock pools along the Felixstowe seafront on a bright summer afternoon. The scene features tiered concrete sea defences, a rocky breakwater, and the town's buildings rising on the hillside under a clear blue sky.
Keywords: Felixstowe, seafront, England, UK, swimmers, rock pools, beach, summer, sunny, breakwater, promenade, houses, cliff, gulls, leisure, coastal, East Suffolk, Europe
</pre>

</details>

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### mlx-community/Qwen3.5-35B-A3B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-35B-A3B-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 2.81s
- *Generation time:* 84.42s
- *Total time:* 87.25s
- *Input validation time:* 0.000111
- *Prompt preparation time:* 0.0143
- *First-token latency:* 82.4
- *Cleanup time:* 0.118
- *Prompt tokens:* 16,548
- *Generation tokens:* 121
- *Total tokens:* 16,669
- *Prompt throughput (raw):* 201 tok/s
- *Generation throughput (raw):* 108 tok/s
- *Peak memory:* 24
- *Active memory:* 21
- *Cache memory:* 0.89
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00857
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1223
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 1e20fd8d42056f870933bf98ca6211024744f7ec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sunny Seafront at Felixstowe with Rock Pools

Description: A sunny afternoon scene at Felixstowe, Suffolk, showing people swimming and playing in the calm blue sea and rock pools. The foreground features weathered concrete groynes and a sandy beach, while the background displays a town with historic buildings and a clock tower on a green hill.

Keywords: Felixstowe, seafront, Suffolk, England, UK, summer, beach, rock pools, groynes, swimming, children, seagulls, town, clock tower, blue sky, coastal, leisure, East Suffolk
</pre>

</details>

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### mlx-community/Qwen3.5-9B-MLX-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 1.48s
- *Generation time:* 74.38s
- *Total time:* 75.86s
- *Input validation time:* 5.28e-05
- *Prompt preparation time:* 0.000304
- *First-token latency:* 72.1
- *Cleanup time:* 0.0851
- *Prompt tokens:* 16,548
- *Generation tokens:* 120
- *Total tokens:* 16,668
- *Prompt throughput (raw):* 229 tok/s
- *Generation throughput (raw):* 87.7 tok/s
- *Peak memory:* 10.0
- *Active memory:* 6.6
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00909
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1223
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Felixstowe Seafront, August 2026, UK

Description:
A sunny day at Felixstowe seafront with people enjoying the water and rocky shoreline under clear skies. The scene includes coastal buildings, a pebble beach, and scattered seagulls near the tide pools.

Keywords:
Felixstowe, seafront, Suffolk, England, UK, GBR, Europe, August 2026, rocky shoreline, tide pools, seagulls, coastal buildings, pebble beach, clear skies, people swimming, family activity
</pre>

</details>

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### mlx-community/gemma-4-26b-a4b-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-26b-a4b-it-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Model load time:* 2.38s
- *Generation time:* 1.73s
- *Total time:* 4.14s
- *Input validation time:* 4.25e-05
- *Prompt preparation time:* 0.0216
- *First-token latency:* 0.415
- *Cleanup time:* 0.121
- *Prompt tokens:* 587
- *Generation tokens:* 94
- *Total tokens:* 681
- *Prompt throughput (raw):* 1,413 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.20
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0118
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1186
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Seafront with stone groynes in Felixstowe, England
Description: People enjoy the seaside at a beach in Felixstowe, England, where stone groynes extend into the blue sea under clear daylight. The coastal town features buildings and greenery overlooking the coastline.
Keywords: Felixstowe, Suffolk, England, UK, seafront, beach, coast, sea, groynes, coastline, summer, tourism, Europe, East Suffolk, water, outdoors
</pre>

</details>

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### mlx-community/gemma-4-31b-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-31b-it-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Model load time:* 2.55s
- *Generation time:* 4.92s
- *Total time:* 7.48s
- *Input validation time:* 5.13e-05
- *Prompt preparation time:* 0.000303
- *First-token latency:* 1.05
- *Cleanup time:* 0.133
- *Prompt tokens:* 587
- *Generation tokens:* 89
- *Total tokens:* 676
- *Prompt throughput (raw):* 561 tok/s
- *Generation throughput (raw):* 27.2 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.77
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0123
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1186
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 696d436c404745a59f30e4939a658162b0a9e57f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/696d436c404745a59f30e4939a658162b0a9e57f
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Seafront and coastal defences at Felixstowe, Suffolk
Description: People enjoy the beach and sea along the seafront in Felixstowe, Suffolk, featuring stepped concrete sea walls and rock armour under a clear late afternoon sky.
Keywords: Felixstowe, Suffolk, East Suffolk, England, UK, seafront, beach, coastal defences, sea wall, rock armour, ocean, summer, leisure, tourism, Europe, GBR
</pre>

</details>

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### Qwen/Qwen3-VL-2B-Instruct

<details>
<summary>Complete evidence: Qwen/Qwen3-VL-2B-Instruct</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 3 words (requested 5-10); Keyword list has 28
  terms (requested 10-18); Duplicate keywords: felixstowe, england, seafront,
  uk, suffolk, gbr
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.68s
- *Generation time:* 16.38s
- *Total time:* 17.08s
- *Input validation time:* 4.12e-05
- *Prompt preparation time:* 0.00891
- *First-token latency:* 14
- *Cleanup time:* 0.0832
- *Prompt tokens:* 16,533
- *Generation tokens:* 147
- *Total tokens:* 16,680
- *Prompt throughput (raw):* 1,184 tok/s
- *Generation throughput (raw):* 94.2 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1204
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Felixstowe Seafront, England

Description:
A bustling seafront in Felixstowe, England, on a sunny day, with people enjoying the beach, swimming, and playing near the rocky breakwaters. The scene is set against a backdrop of historic buildings and lush greenery, with the clear blue sea and sky reflecting the bright sunlight.

Keywords: Felixstowe, England, seafront, beach, people, swimming, rocky breakwaters, historic buildings, greenery, sea, sky, sunlight, beachgoers, coastal, UK, Europe, Suffolk, gbr, seafront, England, UK, gbr, seafront, Felixstowe, Suffolk, UK, gbr, seafront
</pre>

</details>

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

<details>
<summary>Complete evidence: mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.23s
- *Generation time:* 7.59s
- *Total time:* 9.83s
- *Input validation time:* 5.5e-05
- *Prompt preparation time:* 0.00761
- *First-token latency:* 3.21
- *Cleanup time:* 0.12
- *Prompt tokens:* 2,197
- *Generation tokens:* 112
- *Total tokens:* 2,309
- *Prompt throughput (raw):* 685 tok/s
- *Generation throughput (raw):* 29.6 tok/s
- *Peak memory:* 22
- *Active memory:* 18
- *Cache memory:* 0.46
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.000968
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1132
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Devstral-Small-2-24B-Instruct-2512-5bit/snapshots/0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Felixstowe Seafront, England, UK

Description:
A lively seafront scene in Felixstowe, England, on a sunny day, with people enjoying the beach, swimming in the sea, and walking along the rocky shore, backed by a hillside of houses and greenery.

Keywords:
Felixstowe, England, UK, seafront, beach, sea, swimming, people, rocky shore, hillside, houses, greenery, sunny day, coastal town, East Suffolk, Suffolk, Europe, GBR
</pre>

</details>

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### mlx-community/GLM-4.6V-Flash-mxfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-mxfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 2 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.42s
- *Generation time:* 9.16s
- *Total time:* 10.58s
- *Input validation time:* 4.95e-05
- *Prompt preparation time:* 0.00674
- *First-token latency:* 7.67
- *Cleanup time:* 0.0814
- *Prompt tokens:* 6,281
- *Generation tokens:* 73
- *Total tokens:* 6,354
- *Prompt throughput (raw):* 819 tok/s
- *Generation throughput (raw):* 74.5 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.45
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1213
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-Flash-mxfp4/snapshots/773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Felixstowe Seafront
Description: A sunny scene at Felixstowe's seafront, with people enjoying the beach and water, backed by coastal buildings and greenery.
Keywords: Felixstowe seafront, England UK, beach scene, coastal buildings, sunny weather, people swimming, East Suffolk, Europe, gbr, seafront landscape.
</pre>

</details>

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### mlx-community/GLM-4.6V-nvfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible; Title has
  3 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 7.47s
- *Generation time:* 19.43s
- *Total time:* 26.92s
- *Input validation time:* 3.85e-05
- *Prompt preparation time:* 0.00761
- *First-token latency:* 16.8
- *Cleanup time:* 0.268
- *Prompt tokens:* 6,281
- *Generation tokens:* 78
- *Total tokens:* 6,359
- *Prompt throughput (raw):* 374 tok/s
- *Generation throughput (raw):* 37.5 tok/s
- *Peak memory:* 77
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1213
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-nvfp4/snapshots/2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;|begin_of_box|&gt;Title: Felixstowe Seafront, England
Description: A sunny seafront scene in Felixstowe, England, with people enjoying the beach, rocky shoreline, and coastal buildings under clear skies.
Keywords: Felixstowe, seafront, England, UK, East Suffolk, Suffolk, Europe, beach, coastal, people, rocky shoreline, buildings, sunny, clear skies&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Felixstowe Seafront, England
Description: A sunny seafront scene in Felixstowe, England, with people enjoying the beach, rocky shoreline, and coastal buildings under clear skies.
Keywords: Felixstowe, seafront, England, UK, East Suffolk, Suffolk, Europe, beach, coastal, people, rocky shoreline, buildings, sunny, clear skies<|end_of_box|>
```

</details>

</details>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Conversation-role control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 2.03s
- *Generation time:* 6.18s
- *Total time:* 8.22s
- *Input validation time:* 7.45e-05
- *Prompt preparation time:* 0.00145
- *First-token latency:* 1.27
- *Cleanup time:* 0.187
- *Prompt tokens:* 2,586
- *Generation tokens:* 126
- *Total tokens:* 2,712
- *Prompt throughput (raw):* 2,033 tok/s
- *Generation throughput (raw):* 30.3 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.00269
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1169
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Idefics3-8B-Llama3-bf16/snapshots/8c2a30c48864f3251701b7bde40f601d25535098
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Felixstowe Seafront in Suffolk, England, UK.
Description: The seafront in Felixstowe, Suffolk, England, UK, is bustling with activity as people enjoy the sunny day. The sandy beach is lined with large rocks, and the clear blue water of the sea laps gently against the shore. In the background, there are buildings, trees, and a clock tower. The sky is a bright blue, indicating good weather.
Keywords: Felixstowe, Suffolk, England, UK, seafront, beach, rocks, water, buildings, trees, clock tower, blue sky, sunny day.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Title: Felixstowe Seafront in Suffolk, England, UK.
Description: The seafront in Felixstowe, Suffolk, England, UK, is bustling with activity as people enjoy the sunny day. The sandy beach is lined with large rocks, and the clear blue water of the sea laps gently against the shore. In the background, there are buildings, trees, and a clock tower. The sky is a bright blue, indicating good weather.
Keywords: Felixstowe, Suffolk, England, UK, seafront, beach, rocks, water, buildings, trees, clock tower, blue sky, sunny day.<end_of_utterance>
```

</details>

</details>

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### mlx-community/InternVL3-8B-bf16

<details>
<summary>Complete evidence: mlx-community/InternVL3-8B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type internvl_chat)
- *Model load time:* 1.72s
- *Generation time:* 4.32s
- *Total time:* 6.06s
- *Input validation time:* 6.18e-05
- *Prompt preparation time:* 0.00188
- *First-token latency:* 0.957
- *Cleanup time:* 0.181
- *Prompt tokens:* 2,087
- *Generation tokens:* 93
- *Total tokens:* 2,180
- *Prompt throughput (raw):* 2,181 tok/s
- *Generation throughput (raw):* 34.1 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.17
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.003
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1137
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.internvl_chat.processor.InternVLChatProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3-8B-bf16/snapshots/e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Felixstowe Seafront at Sunset
Description: A bustling Felixstowe seafront on 2026-08-13, with people enjoying the beach and sea under clear skies, featuring historic buildings and a rocky breakwater.
Keywords: Felixstowe, seafront, Suffolk, England, UK, East Suffolk, beach, breakwater, historic buildings, clear skies, summer, recreation, shoreline, waterfront, coastal, Europe
</pre>

</details>

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### mlx-community/MiniCPM-V-4.6-8bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.6-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Model load time:* 1.17s
- *Generation time:* 1.43s
- *Total time:* 2.61s
- *Input validation time:* 7.89e-05
- *Prompt preparation time:* 0.00799
- *First-token latency:* 0.266
- *Cleanup time:* 0.104
- *Prompt tokens:* 895
- *Generation tokens:* 75
- *Total tokens:* 970
- *Prompt throughput (raw):* 3,363 tok/s
- *Generation throughput (raw):* 238 tok/s
- *Peak memory:* 3.8
- *Active memory:* 2.3
- *Cache memory:* 0.05
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00483
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1200
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.6-8bit/snapshots/03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;

&lt;/think&gt;

Title:
Felixstowe Seafront Beach Activity

Description:
A lively seafront scene with people enjoying the water and sand under bright sunlight.

Keywords: Felixstowe, seafront, beach, activity, sunlight, water, people, rocks, greenery, buildings, Europe, UK, GBR, August, tide, recreation
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>

</think>

Title:
Felixstowe Seafront Beach Activity

Description:
A lively seafront scene with people enjoying the water and sand under bright sunlight.

Keywords: Felixstowe, seafront, beach, activity, sunlight, water, people, rocks, greenery, buildings, Europe, UK, GBR, August, tide, recreation
```

</details>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.64s
- *Generation time:* 8.55s
- *Total time:* 10.20s
- *Input validation time:* 6.27e-05
- *Prompt preparation time:* 0.000283
- *First-token latency:* 3.52
- *Cleanup time:* 0.149
- *Prompt tokens:* 2,730
- *Generation tokens:* 175
- *Total tokens:* 2,905
- *Prompt throughput (raw):* 775 tok/s
- *Generation throughput (raw):* 41.2 tok/s
- *Peak memory:* 13
- *Active memory:* 8.4
- *Cache memory:* 0.56
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00536
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3570
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 28777b889d841a86369c736175cb77258c8134b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-nvfp4/snapshots/28777b889d841a86369c736175cb77258c8134b2
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
**Title:**
*Felixstowe Seafront Beach, UK – Summer Activity*

**Description:**
A lively seaside scene at Felixstowe’s seafront beach on **August 13, 2026 (17:23 UTC+1)**, featuring swimmers, waders, and sunbathers along rocky tide pools and sandy shores. Behind them, a bustling promenade lined with historic buildings, green hills, and clear blue skies frames the vibrant coastal atmosphere.

**Keywords:**
Felixstowe, East Suffolk, UK, GBR, Europe, seafront beach, rocky tide pools, sandy shore, summer activity, coastal promenade, historic buildings, clear blue skies, seaside swimming, tide pools, seaside holiday, Suffolk coast, UK seaside, summer beach, seaside setting, seaside lighting
</pre>

</details>

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### mlx-community/Molmo-7B-D-0924-8bit

<details>
<summary>Complete evidence: mlx-community/Molmo-7B-D-0924-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10); Keyword list has 23
  terms (requested 10-18); Duplicate keywords: felixstowe, england, uk, gbr
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.44s
- *Generation time:* 4.78s
- *Total time:* 6.24s
- *Input validation time:* 5.53e-05
- *Prompt preparation time:* 0.00191
- *First-token latency:* 0.913
- *Cleanup time:* 0.219
- *Prompt tokens:* 1,487
- *Generation tokens:* 107
- *Total tokens:* 1,594
- *Prompt throughput (raw):* 1,630 tok/s
- *Generation throughput (raw):* 38.8 tok/s
- *Peak memory:* 11
- *Active memory:* 9.0
- *Cache memory:* 0.14
- *Model-load active memory:* 9.04
- *Post-cleanup active memory:* 0.00593
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1128
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Molmo-7B-D-0924-8bit/snapshots/90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Felixstowe Seaside Scene, England

Description: A bustling Felixstowe beach with people enjoying the water and sand, backed by a rocky jetty and framed by trees and buildings under a clear blue sky.

Keywords: Seafront, Felixstowe, England, UK, GBR, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, beach, people, water, sand, rocky jetty, trees, buildings, blue sky, summer, beachgoers, holidaymakers
</pre>

</details>

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### mlx-community/Phi-3.5-vision-instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Phi-3.5-vision-instruct-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 3 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type phi3_v)
- *Model load time:* 1.00s
- *Generation time:* 2.31s
- *Total time:* 3.31s
- *Input validation time:* 5.59e-05
- *Prompt preparation time:* 0.000887
- *First-token latency:* 0.301
- *Cleanup time:* 0.0766
- *Prompt tokens:* 1,094
- *Generation tokens:* 84
- *Total tokens:* 1,178
- *Prompt throughput (raw):* 3,638 tok/s
- *Generation throughput (raw):* 56.3 tok/s
- *Peak memory:* 9.6
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00713
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1153
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Felixstowe Seafront, England

Description: A serene scene at Felixstowe seafront on August 13, 2026, with people enjoying the beach and the sea, captured on a sunny day.

Keywords: Felixstowe, seafront, England, UK, GBR, beach, people, sea, sky, rocks, coastline
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### mlx-community/Qwen3-VL-2B-Instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 3 words (requested 5-10); Keyword list has 28
  terms (requested 10-18); Duplicate keywords: felixstowe, england, seafront,
  uk, suffolk, gbr
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 1.63s
- *Generation time:* 30.65s
- *Total time:* 32.29s
- *Input validation time:* 0.00015
- *Prompt preparation time:* 0.00061
- *First-token latency:* 27.3
- *Cleanup time:* 0.134
- *Prompt tokens:* 16,533
- *Generation tokens:* 147
- *Total tokens:* 16,680
- *Prompt throughput (raw):* 606 tok/s
- *Generation throughput (raw):* 83.5 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00775
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1204
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Instruct-bf16/snapshots/c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Felixstowe Seafront, England

Description:
A bustling seafront in Felixstowe, England, on a sunny day, with people enjoying the beach, swimming, and playing near the rocky breakwaters. The scene is set against a backdrop of historic buildings and lush greenery, with the clear blue sea and sky reflecting the bright sunlight.

Keywords: Felixstowe, England, seafront, beach, people, swimming, rocky breakwaters, historic buildings, greenery, sea, sky, sunlight, beachgoers, coastal, UK, Europe, Suffolk, gbr, seafront, England, UK, gbr, seafront, Felixstowe, Suffolk, UK, gbr, seafront
</pre>

</details>

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### mlx-community/Qwen3.6-27B-mxfp8

<details>
<summary>Complete evidence: mlx-community/Qwen3.6-27B-mxfp8</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 7.02s
- *Generation time:* 107.07s
- *Total time:* 114.12s
- *Input validation time:* 4.15e-05
- *Prompt preparation time:* 0.0145
- *First-token latency:* 97.7
- *Cleanup time:* 0.183
- *Prompt tokens:* 16,548
- *Generation tokens:* 121
- *Total tokens:* 16,669
- *Prompt throughput (raw):* 169 tok/s
- *Generation throughput (raw):* 14.4 tok/s
- *Peak memory:* 35
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.0096
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1223
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-mxfp8/snapshots/5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Seafront at Felixstowe, Suffolk, England

Description:
A sunny late afternoon view of the Felixstowe seafront in Suffolk, England, showing people swimming and playing on the stepped concrete groynes and pebble beach. The background features a hillside lined with residential buildings and the Spa Pavilion under a clear blue sky.

Keywords:
Felixstowe, Suffolk, England, UK, seafront, beach, groynes, swimming, summer, people, coast, East Suffolk, ocean, concrete steps, Spa Pavilion, seaside, recreation, blue sky, Europe
</pre>

</details>

---

<a id="model-mlx-community-step-37-flash-oq2e"></a>

### mlx-community/Step-3.7-Flash-oQ2e

<details>
<summary>Complete evidence: mlx-community/Step-3.7-Flash-oQ2e</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 13.09s
- *Generation time:* 26.39s
- *Total time:* 39.57s
- *Input validation time:* 4.62e-05
- *Prompt preparation time:* 0.0799
- *First-token latency:* 22.1
- *Cleanup time:* 0.305
- *Prompt tokens:* 3,468
- *Generation tokens:* 116
- *Total tokens:* 3,584
- *Prompt throughput (raw):* 157 tok/s
- *Generation throughput (raw):* 39.9 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.39
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.00986
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1210
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.step3p7.processing_step3p7.Step3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Step-3.7-Flash-oQ2e/snapshots/3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Felixstowe Seafront, Suffolk, England

Description:
A sunny day at the Felixstowe seafront in Suffolk, England, with people swimming, sunbathing, and playing on the beach alongside a rocky breakwater, set against a backdrop of coastal buildings and greenery under a clear blue sky.

Keywords:
Felixstowe, Suffolk, England, UK, seafront, beach, coastline, swimming, sunbathing, breakwater, coastal buildings, blue sky, sunny day, East Suffolk, Europe, GBR, Adobe Stock
</pre>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.28s
- *Generation time:* 2.65s
- *Total time:* 5.96s
- *Input validation time:* 4.47e-05
- *Prompt preparation time:* 0.0257
- *First-token latency:* 0.432
- *Cleanup time:* 0.146
- *Prompt tokens:* 583
- *Generation tokens:* 82
- *Total tokens:* 665
- *Prompt throughput (raw):* 1,348 tok/s
- *Generation throughput (raw):* 55.3 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1158
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-8bit/snapshots/7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
&lt;channel|&gt;Title: Sunny seafront at Felixstowe in Suffolk, England
Description: People enjoy a sunny day at the Felixstowe seafront, where stone groynes meet the sea under a clear blue sky.
Keywords: Felixstowe, Suffolk, England, UK, East Suffolk, seafront, beach, ocean, coastal, summer, recreation, tourism, seaside, leisure, water, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Sunny seafront at Felixstowe in Suffolk, England
Description: People enjoy a sunny day at the Felixstowe seafront, where stone groynes meet the sea under a clear blue sky.
Keywords: Felixstowe, Suffolk, England, UK, East Suffolk, seafront, beach, ocean, coastal, summer, recreation, tourism, seaside, leisure, water, Europe
```

</details>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-mxfp8</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible; Duplicate
  keywords: coastal
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.21s
- *Generation time:* 2.69s
- *Total time:* 5.91s
- *Input validation time:* 4.78e-05
- *Prompt preparation time:* 0.000318
- *First-token latency:* 0.428
- *Cleanup time:* 0.136
- *Prompt tokens:* 583
- *Generation tokens:* 81
- *Total tokens:* 664
- *Prompt throughput (raw):* 1,362 tok/s
- *Generation throughput (raw):* 53.0 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1158
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-mxfp8/snapshots/ded389e478f86d498ad9e7f47666e83b166a28f1
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
&lt;channel|&gt;Title: Sunny day at the seafront in Felixstowe, England
Description: People enjoy a sunny day at the stone-tiered sea beach in Felixstowe, Suffolk, under a clear blue sky.
Keywords: Felixstowe, Suffolk, East Suffolk, England, UK, seafront, beach, sea, summer, coastal, leisure, outdoors, tourism, coastal, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Sunny day at the seafront in Felixstowe, England
Description: People enjoy a sunny day at the stone-tiered sea beach in Felixstowe, Suffolk, under a clear blue sky.
Keywords: Felixstowe, Suffolk, East Suffolk, England, UK, seafront, beach, sea, summer, coastal, leisure, outdoors, tourism, coastal, Europe
```

</details>

</details>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### mlx-community/gemma-3-27b-it-qat-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3-27b-it-qat-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 21 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 2.31s
- *Generation time:* 6.65s
- *Total time:* 8.98s
- *Input validation time:* 4.74e-05
- *Prompt preparation time:* 0.00281
- *First-token latency:* 0.987
- *Cleanup time:* 0.125
- *Prompt tokens:* 569
- *Generation tokens:* 136
- *Total tokens:* 705
- *Prompt throughput (raw):* 576 tok/s
- *Generation throughput (raw):* 26.6 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.43
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0107
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1187
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3.processing_gemma3.Gemma3Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3-27b-it-qat-4bit/snapshots/fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Felixstowe Seafront, Suffolk: Beach and Promenade View
Description: Captured on 13th August 2026 at 17:23, this image shows the busy seafront at Felixstowe, Suffolk, with people enjoying the beach, sea and distinctive groynes under a clear blue sky. The promenade buildings and trees form a backdrop to the coastal scene.
Keywords: Felixstowe, Suffolk, England, UK, GBR, East Suffolk, seafront, beach, promenade, groynes, coastline, sea, summer, people, coastal defence, clear sky, 2026, August, daytime, buildings, trees
</pre>

</details>

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### mlx-community/pixtral-12b-8bit

<details>
<summary>Complete evidence: mlx-community/pixtral-12b-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.73s
- *Generation time:* 4.85s
- *Total time:* 6.59s
- *Input validation time:* 4.66e-05
- *Prompt preparation time:* 0.00231
- *First-token latency:* 1.88
- *Cleanup time:* 0.0889
- *Prompt tokens:* 2,828
- *Generation tokens:* 96
- *Total tokens:* 2,924
- *Prompt throughput (raw):* 1,506 tok/s
- *Generation throughput (raw):* 39.6 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.54
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0143
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1132
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--pixtral-12b-8bit/snapshots/79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
People Enjoying Seafront at Felixstowe, Suffolk, UK

Description:
On a sunny day, people relax and play by the seafront in Felixstowe, Suffolk, UK, with clear skies and calm waters.

Keywords:
Felixstowe, Suffolk, UK, seafront, beach, people, sea, sunny, clear skies, calm waters, rocks, buildings, birds, swimming, playing, summer, coastal, England, Europe
</pre>

</details>

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

<details>
<summary>Complete evidence: mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response repeats the task instructions instead of only returning the
  requested fields; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.72s
- *Generation time:* 26.27s
- *Total time:* 28.00s
- *Input validation time:* 5.54e-05
- *Prompt preparation time:* 0.00679
- *First-token latency:* 1.86
- *Cleanup time:* 0.0945
- *Prompt tokens:* 2,919
- *Generation tokens:* 1,000
- *Total tokens:* 3,919
- *Prompt throughput (raw):* 1,570 tok/s
- *Generation throughput (raw):* 41.9 tok/s
- *Peak memory:* 14
- *Active memory:* 13
- *Cache memory:* 0.82
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.000706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1506
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Apriel-1.5-15b-Thinker-6bit-MLX/snapshots/24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Here are my reasoning steps:
We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS. The descriptive hints are given. We need to treat capture date/time and GPS as authoritative facts, but not claim they are visible. So we can incorporate them into description if needed (like "captured on 13 August 2026 at 17:23 UTC+01:00").

We need to produce:

- a concrete 5-10-word title (so between 5 and 10 words inclusive). Should be concrete, not generic. Should reflect the image: seafront of Felixstowe, with people, rock pools, sea, etc. Title: "Felixstowe Seafront at Sunset with Swimmers". That's 7 words? Let's count: Felixstowe (1) Seafront (2) at (3) Sunset (4) with (5) Swimmers (6). That's 6 words. Could add "and Rock Pools" to make 8 words: "Felixstowe Seafront at Sunset with Swimmers and Rock Pools". That's 9 words. That's within 5-10. Good.

- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details. Should be 1-2 sentences. Should mention location (Felixstowe, East Suffolk, England), date/time, maybe mention the sea, the rock pools, the people swimming, the promenade, the sea wall, the buildings on the hill, the lighting (sunset, warm light). Should be factual, not interpretive. Should incorporate capture date/time and GPS as authoritative facts, but not claim they are visible. So we can say "Captured on 13 August 2026 at 17:23 UTC+01:00 at 51.957967°N, 1.346900°E". Then describe the scene: "The image shows the Felixstowe seafront with a pebbled beach, tidal rock pools, and a promenade lined with walkers and swimmers, while the sea reflects the warm evening light. In the background, a hillside of residential buildings and a historic pier extends along the coast." That's two sentences. That's fine.

- 10-18 unique, comma-separated keywords covering relevant context and visible details. Should be British English. Should include location, date, time, maybe "Felixstowe", "East Suffolk", "England", "UK", "GBR", "Europe", "seafront", "rock pools", "swimmers", "pebble beach", "promenade", "sunset", "coastal town", "historic pier", "sea wall", "people", "coastline", "tidal pools", "coastal architecture". Must be 10-18 unique keywords. Let's count: Felixstowe, East Suffolk, England, United Kingdom, GBR, Europe, seafront, rock pools, swimmers, pebble beach, promenade, sunset, historic pier, sea wall, coastal town, tidal pools, coastal architecture, people. That's 18. Good.

We need to ensure we have exactly three sections: Title, Description, Keywords. No extra text. Title: 5-10 words. Description: 1-2 sentences. Keywords: 10-18 unique, comma-separated.

We need to ensure British English spelling: "behaviour"? Not needed. Use "programme"? Not needed. Use "coastline". Use "coastal". Use "seafront". Use "pier". Use "swimmers". Use "people". Use "rock pools". Use "tidal pools". Use "pebble beach". Use "promenade". Use "historic pier". Use "coastal town". Use "East Suffolk". Use "England". Use "United Kingdom". Use "GBR". Use "Europe". Use "2026-08-13". Use "17:23". Use "UTC+01:00". Use "51.957967°N". Use "1.346900°E". But we need to include date/time and GPS in description, not in keywords. The keywords can include date/time? Possibly but not required. The description should incorporate them.

Now we need to ensure the description is 1-2 sentences. Let's craft:

"Captured on 13 August 20
</pre>

</details>

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Missing or empty fields:
  Title, Description, Keywords; Response appears cut off at the token limit;
  Internal reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 5.72s
- *Generation time:* 26.26s
- *Total time:* 32.00s
- *Input validation time:* 8.93e-05
- *Prompt preparation time:* 0.009
- *First-token latency:* 2.21
- *Cleanup time:* 0.198
- *Prompt tokens:* 1,569
- *Generation tokens:* 1,000
- *Total tokens:* 2,569
- *Prompt throughput (raw):* 710 tok/s
- *Generation throughput (raw):* 43.7 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.17
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00118
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1322
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--ERNIE-4.5-VL-28B-A3B-Thinking-bf16/snapshots/32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. The hints are pretty clear: "Seafront, Felixstowe, England, UK, GBR, Europe." I need something concise, so I'll go with "Seafront at Felixstowe, England, UK." That's five words, right in the sweet spot.

Now for the description. I'll need to weave in the context – the date, time, and location – with the visual elements. The image is a wide shot of a beach, so I'll start there. I see a bustling beach scene with people enjoying the water and the sand. The lighting is bright, suggesting a sunny day. I'll mention the location and the time of day. I'll also include the key details like the seawall and the distant town. I'll make sure to mention the people and the activity. I'll refine it to be a single, concise sentence. Okay, here it is: "A wide-angle, sunlit photograph of a bustling beach at Felixstowe, England, UK, taken on August 13, 2026, at 5:23 PM, showing people on the sand and in the water with a town and seafront in the background."

Finally, the keywords. I need to cover the location, the subject matter, the time of day, and the general atmosphere. I'll start with the location: Felixstowe, England, UK, GBR, Europe, Suffolk, East Suffolk. Then the subject: beach, seashore, seafront, coast, shoreline, waterfront, shore, sand, sandy beach, shoreline, seawall, cliffs, rocks, rocks, rocks, rocks, rocks. People and activity: people, crowd, family, couple, group, children, kids, teenagers, adults, swimming, bathing, wading, playing, relaxing, strolling, walking, sunbathing, tourists, holidaymakers, beachgoers, lifeguard, lifeguard tower, lifeguard tower. Nature and environment: sea, ocean, water, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves,
</pre>

</details>

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### mlx-community/FastVLM-0.5B-bf16

<details>
<summary>Complete evidence: mlx-community/FastVLM-0.5B-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response repeats the task instructions instead of only returning the
  requested fields
- *Arch supported by installed mlx-vlm:* yes (model_type llava_qwen2 via
  fastvlm)
- *Model load time:* 0.84s
- *Generation time:* 1.60s
- *Total time:* 2.45s
- *Input validation time:* 3.86e-05
- *Prompt preparation time:* 0.00171
- *First-token latency:* 0.0943
- *Cleanup time:* 0.0972
- *Prompt tokens:* 308
- *Generation tokens:* 120
- *Total tokens:* 428
- *Prompt throughput (raw):* 3,268 tok/s
- *Generation throughput (raw):* 338 tok/s
- *Peak memory:* 2.1
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00149
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1227
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--FastVLM-0.5B-bf16/snapshots/81ffe929046666c43de53691147b1669ba0f3a4c
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
A concrete 5-10-word title: "Felixstowe, UK - Seafront, GBR, East Suffolk, UK, gbr, seafront"
A 1-2-sentence factual description: "A vibrant seaside scene with a concrete 5-10-word title, featuring a concrete 5-10-word title, a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details, and 10-18 unique, comma-separated keywords covering relevant context and visible details."
</pre>

</details>

---

<a id="model-mlx-community-glm-41v-9b-thinking-8bit"></a>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

<details>
<summary>Complete evidence: mlx-community/GLM-4.1V-9B-Thinking-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the task instructions instead of only
  returning the requested fields; Extra text appears before the Title field;
  Response appears cut off at the token limit; Internal reasoning block
  appears incomplete; Title has 4 words (requested 5-10); Keyword list has 69
  terms (requested 10-18); Duplicate keywords: felixstowe, suffolk, uk,
  seafront, coastal steps, beach, people, swimming, historic buildings,
  daylight, europe, east suffolk, england, gbr, seagulls, seaside, wait, with
  people swimming and walking near distinctive coastal steps, under clear
  daylight with historic buildings in the background
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.80s
- *Generation time:* 32.93s
- *Total time:* 34.74s
- *Input validation time:* 4.26e-05
- *Prompt preparation time:* 0.0017
- *First-token latency:* 7.77
- *Cleanup time:* 0.121
- *Prompt tokens:* 6,277
- *Generation tokens:* 1,000
- *Total tokens:* 7,277
- *Prompt throughput (raw):* 808 tok/s
- *Generation throughput (raw):* 40.6 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.43
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1189
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.1V-9B-Thinking-8bit/snapshots/9677807f106500eb7690391c27645d59f6855cfb
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image, it's a seafront with people, buildings, and the coastal area. So maybe "Felixstowe Seafront Beach Scene" or something similar. Wait, need to be concrete. Let's go with "Felixstowe Coastal Promenade" maybe, but let's check the hints. The title hint says Seafront, Felixstowe... so maybe "Felixstowe Seafront with Coastal Steps".

Then the description: need to combine context (date/time, location) with main subject (seafront, people, setting, action, lighting, distinctive details). The image shows people swimming, walking on the beach, coastal steps, buildings, clear sky. So something like "Seafront scene in Felixstowe, England, showing people enjoying the beach with distinctive coastal steps and historic buildings under clear daylight."

Wait, let's make sure. The description should be 1-2 sentences. Let's structure it: "Seafront in Felixstowe, England, with people swimming and walking near distinctive coastal steps, under clear daylight with historic buildings in the background."

Then keywords: need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (but maybe not, since the image might not be from those, but the hints include them? Wait, the hints say keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Also visible details: people swimming, coastal steps, buildings, beach, sea, seagulls, etc. So keywords could be: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade.

Wait, let's check the hints again. The keyword hints include: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. So need to include those, plus visible details. So keywords: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade.

Now, let's put it all together.

Title: "Felixstowe Coastal Steps Seafront"

Description: "Seafront scene in Felixstowe, England, showing people swimming and enjoying the beach near distinctive coastal steps, with historic buildings and clear daylight in the background."

Keywords: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade

Wait, let's check the word count. Title should be 5-10 words. "Felixstowe Coastal Steps Seafront" is 6 words. Description is 1-2 sentences, let's make it 2 sentences? Wait, the user said 1-2 sentence. Let's make it one sentence. "Seafront scene in Felixstowe, England, with people swimming and walking near distinctive coastal steps, historic buildings, and clear daylight." That's one sentence. Or two: "Seafront scene in Felixstowe, England, shows people enjoying the beach near distinctive coastal steps. Historic buildings and clear daylight are visible in the background." But maybe better to combine. Let's do: "Seafront scene in Felixstowe, England, with people swimming and walking near distinctive coastal steps, under clear daylight with historic buildings in the background."

Keywords need to be unique, comma-separated. Let's list them: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade. That's 17 keywords, which is within 10-18. Let's check if all are unique. Yes. Now, make sure to include the authoritative context: capture date/time and GPS, but in the description, maybe mention the date/time? Wait, the description should combine relevant context with main visible subject, setting, action, lighting, distinctive details. So maybe: "Seafront scene in Felixstowe, England, captured on 2026-08-13 at 17:23:18 UTC+01:00, showing people swimming and walking near distinctive coastal steps, under clear daylight with historic buildings in the background."

Wait, the description hint says "Seafront, Felixstowe, England, UK, GBR" so maybe include the date. Let's adjust description: "Seafront scene in Felixstowe, England, captured on
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image, it's a seafront with people, buildings, and the coastal area. So maybe "Felixstowe Seafront Beach Scene" or something similar. Wait, need to be concrete. Let's go with "Felixstowe Coastal Promenade" maybe, but let's check the hints. The title hint says Seafront, Felixstowe... so maybe "Felixstowe Seafront with Coastal Steps".

Then the description: need to combine context (date/time, location) with main subject (seafront, people, setting, action, lighting, distinctive details). The image shows people swimming, walking on the beach, coastal steps, buildings, clear sky. So something like "Seafront scene in Felixstowe, England, showing people enjoying the beach with distinctive coastal steps and historic buildings under clear daylight."

Wait, let's make sure. The description should be 1-2 sentences. Let's structure it: "Seafront in Felixstowe, England, with people swimming and walking near distinctive coastal steps, under clear daylight with historic buildings in the background."

Then keywords: need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (but maybe not, since the image might not be from those, but the hints include them? Wait, the hints say keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Also visible details: people swimming, coastal steps, buildings, beach, sea, seagulls, etc. So keywords could be: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade.

Wait, let's check the hints again. The keyword hints include: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. So need to include those, plus visible details. So keywords: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade.

Now, let's put it all together.

Title: "Felixstowe Coastal Steps Seafront"

Description: "Seafront scene in Felixstowe, England, showing people swimming and enjoying the beach near distinctive coastal steps, with historic buildings and clear daylight in the background."

Keywords: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade

Wait, let's check the word count. Title should be 5-10 words. "Felixstowe Coastal Steps Seafront" is 6 words. Description is 1-2 sentences, let's make it 2 sentences? Wait, the user said 1-2 sentence. Let's make it one sentence. "Seafront scene in Felixstowe, England, with people swimming and walking near distinctive coastal steps, historic buildings, and clear daylight." That's one sentence. Or two: "Seafront scene in Felixstowe, England, shows people enjoying the beach near distinctive coastal steps. Historic buildings and clear daylight are visible in the background." But maybe better to combine. Let's do: "Seafront scene in Felixstowe, England, with people swimming and walking near distinctive coastal steps, under clear daylight with historic buildings in the background."

Keywords need to be unique, comma-separated. Let's list them: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade. That's 17 keywords, which is within 10-18. Let's check if all are unique. Yes. Now, make sure to include the authoritative context: capture date/time and GPS, but in the description, maybe mention the date/time? Wait, the description should combine relevant context with main visible subject, setting, action, lighting, distinctive details. So maybe: "Seafront scene in Felixstowe, England, captured on 2026-08-13 at 17:23:18 UTC+01:00, showing people swimming and walking near distinctive coastal steps, under clear daylight with historic buildings in the background."

Wait, the description hint says "Seafront, Felixstowe, England, UK, GBR" so maybe include the date. Let's adjust description: "Seafront scene in Felixstowe, England, captured on 
```

</details>

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-2506-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response repeats the task
  instructions instead of only returning the requested fields; Extra text
  appears before the Title field; Response appears cut off at the token limit;
  Internal reasoning block appears incomplete; Title has 11 words (requested
  5-10); Keyword list has 85 terms (requested 10-18); Duplicate keywords:
  seafront, england, uk, gbr, europe, rocky shoreline, people, coastal
  buildings, clear sky, calm sea, birds, beachgoers, utc 01 00, any vision,
  2026 08 13, adobe stock
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 3.29s
- *Generation time:* 216.72s
- *Total time:* 220.02s
- *Input validation time:* 5.19e-05
- *Prompt preparation time:* 0.00179
- *First-token latency:* 1.41
- *Cleanup time:* 0.343
- *Prompt tokens:* 1,269
- *Generation tokens:* 1,000
- *Total tokens:* 2,269
- *Prompt throughput (raw):* 901 tok/s
- *Generation throughput (raw):* 4.66 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.8
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00365
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1319
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Kimi-VL-A3B-Thinking-2506-bf16/snapshots/fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* [EOS]

*Readable output:*

<pre class="model-output-readable">
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.

The image shows a seafront in Felixstowe, England, UK. The capture date is 2026-08-13 17:23:18 UTC+01:00, GPS is 51.957967°N, 1.346900°E. The hints mention seafront, Felixstowe, England, UK, GBR, Europe, Adobe Stock, Any Vision, East Suffolk, Suffolk, UK. The description hints are similar.

First, the title needs to be a concrete 5-10 words. Let's see the main elements: seafront, people, rocky shore, buildings, sea, clear sky. Maybe "Felixstowe seafront with people on rocky shore and coastal buildings." Wait, that's 9 words. Maybe "Felixstowe seafront scene with rocky shoreline and coastal buildings." That's 9 words.

Next, the description: 1-2 sentences. It should combine context (date/time, GPS) with main subject, setting, action, lighting, details. So: "A sunny day at Felixstowe's seafront on August 13, 2026, showing people enjoying the rocky shoreline, coastal buildings, and calm sea under clear skies." Wait, the capture date is 2026-08-13 17:23:18 UTC+01:00, so maybe include that. Also, GPS is 51.957967°N, 1.346900°E. So: "On August 13, 2026, at 17:23 UTC+01:00, the sunny Felixstowe seafront at 51.957967°N, 1.346900°E features people on a rocky shore, coastal buildings, and calm waters." That's a bit long but includes the date, time, location, and elements.

Keywords: Need 10-18 unique, comma-separated. Let's list elements: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, birds, beachgoers, UTC+01:00, 2026-08-13. Wait, but the hints mention Adobe Stock, Any Vision, East Suffolk. Maybe include those? Wait, the hints are hints, but the image might not have those. The image shows Adobe Stock watermark? Wait, the image is from Adobe Stock, so maybe include that. Also, East Suffolk is a county. So keywords: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk. That's 15 keywords. Let me check: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk. That's 15. Maybe add birds? The image has birds in the water. So include birds. So 16. Let's adjust: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk, birds. That's 16.

Wait, the user said "unique, comma-separated keywords covering relevant context and visible details". So visible details: people, rocky shore, buildings, sea, sky, birds, beachgoers, etc. Context: date, time, location. So include those. Let's finalize:

Title: Felixstowe seafront scene with people on rocky shore and coastal buildings.

Description: A sunny day at Felixstowe's seafront on August 13, 2026, at 17:23 UTC+01:00, featuring people enjoying the rocky shoreline, coastal buildings, and calm sea under clear skies.

Keywords: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk, birds.

Wait, the description should be 1-2 sentences. Let me check the example. The user's example might have a more concise version. Let me adjust the description to be 1-2 sentences. Maybe: "Sunny Felixstowe seafront at 51.
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.

The image shows a seafront in Felixstowe, England, UK. The capture date is 2026-08-13 17:23:18 UTC+01:00, GPS is 51.957967°N, 1.346900°E. The hints mention seafront, Felixstowe, England, UK, GBR, Europe, Adobe Stock, Any Vision, East Suffolk, Suffolk, UK. The description hints are similar. 

First, the title needs to be a concrete 5-10 words. Let's see the main elements: seafront, people, rocky shore, buildings, sea, clear sky. Maybe "Felixstowe seafront with people on rocky shore and coastal buildings." Wait, that's 9 words. Maybe "Felixstowe seafront scene with rocky shoreline and coastal buildings." That's 9 words.

Next, the description: 1-2 sentences. It should combine context (date/time, GPS) with main subject, setting, action, lighting, details. So: "A sunny day at Felixstowe's seafront on August 13, 2026, showing people enjoying the rocky shoreline, coastal buildings, and calm sea under clear skies." Wait, the capture date is 2026-08-13 17:23:18 UTC+01:00, so maybe include that. Also, GPS is 51.957967°N, 1.346900°E. So: "On August 13, 2026, at 17:23 UTC+01:00, the sunny Felixstowe seafront at 51.957967°N, 1.346900°E features people on a rocky shore, coastal buildings, and calm waters." That's a bit long but includes the date, time, location, and elements.

Keywords: Need 10-18 unique, comma-separated. Let's list elements: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, birds, beachgoers, UTC+01:00, 2026-08-13. Wait, but the hints mention Adobe Stock, Any Vision, East Suffolk. Maybe include those? Wait, the hints are hints, but the image might not have those. The image shows Adobe Stock watermark? Wait, the image is from Adobe Stock, so maybe include that. Also, East Suffolk is a county. So keywords: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk. That's 15 keywords. Let me check: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk. That's 15. Maybe add birds? The image has birds in the water. So include birds. So 16. Let's adjust: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk, birds. That's 16.

Wait, the user said "unique, comma-separated keywords covering relevant context and visible details". So visible details: people, rocky shore, buildings, sea, sky, birds, beachgoers, etc. Context: date, time, location. So include those. Let's finalize:

Title: Felixstowe seafront scene with people on rocky shore and coastal buildings.

Description: A sunny day at Felixstowe's seafront on August 13, 2026, at 17:23 UTC+01:00, featuring people enjoying the rocky shoreline, coastal buildings, and calm sea under clear skies.

Keywords: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk, birds.

Wait, the description should be 1-2 sentences. Let me check the example. The user's example might have a more concise version. Let me adjust the description to be 1-2 sentences. Maybe: "Sunny Felixstowe seafront at 51.
```

</details>

</details>

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### mlx-community/MolmoPoint-8B-fp16

<details>
<summary>Complete evidence: mlx-community/MolmoPoint-8B-fp16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type molmo_point)
- *Model load time:* 2.58s
- *Generation time:* 37.47s
- *Total time:* 40.08s
- *Input validation time:* 8.27e-05
- *Prompt preparation time:* 0.00507
- *First-token latency:* 4.15
- *Cleanup time:* 0.397
- *Prompt tokens:* 3,100
- *Generation tokens:* 183
- *Total tokens:* 3,283
- *Prompt throughput (raw):* 747 tok/s
- *Generation throughput (raw):* 5.64 tok/s
- *Peak memory:* 24
- *Active memory:* 19
- *Cache memory:* 5.0
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.00655
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1170
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo_point.processing_molmo_point.MolmoPointProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MolmoPoint-8B-fp16/snapshots/0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Felixstowe Seafront, England: Rocky Shore with People and Buildings

A bustling Felixstowe seafront scene captured on August 13, 2026, at 17:23 UTC+01:00. The image shows a rocky shoreline with people enjoying the water, a prominent clock tower, and a mix of residential and commercial buildings in the background. The sky is clear and blue, indicating a pleasant summer day.

Felixstowe, England, UK, GBR, East Suffolk, seafront, rocky shore, clock tower, residential buildings, commercial buildings, people swimming, people walking, clear blue sky, summer day, 2026, 17:23, UTC+01:00, 51.957967°N, 1.346900°E
</pre>

</details>

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.52s
- *Generation time:* 99.60s
- *Total time:* 100.13s
- *Input validation time:* 4.54e-05
- *Prompt preparation time:* 0.00392
- *First-token latency:* 98.1
- *Cleanup time:* 0.183
- *Prompt tokens:* 16,544
- *Generation tokens:* 98
- *Total tokens:* 16,642
- *Prompt throughput (raw):* 169 tok/s
- *Generation throughput (raw):* 144 tok/s
- *Peak memory:* 5.1
- *Active memory:* 1.3
- *Cache memory:* 0.86
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.00744
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1262
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 01af461cdb9574acc09084a0ef94e216e142b085
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen2-VL-2B-Instruct-4bit/snapshots/01af461cdb9574acc09084a0ef94e216e142b085
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Seafront, Felixstowe, England, UK, GBR, Europe

Description:
The image depicts a lively seafront scene in Felixstowe, England. The beach is bustling with people enjoying the sunny weather, and the coastline is lined with rocks and greenery. The background features a mix of residential buildings and a clock tower, adding to the picturesque charm of the location. The clear blue sky and the presence of seagulls add to the overall serene atmosphere.
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response appears cut off at the token limit; Internal reasoning block
  appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.99s
- *Generation time:* 43.55s
- *Total time:* 44.58s
- *Input validation time:* 5.85e-05
- *Prompt preparation time:* 0.0198
- *First-token latency:* 27.4
- *Cleanup time:* 0.238
- *Prompt tokens:* 16,535
- *Generation tokens:* 1,000
- *Total tokens:* 17,535
- *Prompt throughput (raw):* 603 tok/s
- *Generation throughput (raw):* 68.1 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.4
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00806
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1212
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Thinking-bf16/snapshots/c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Got it, let's tackle this step by step. First, the title. The context says Felixstowe, England, UK, so we need a concise title. The image shows a seafront with people, so maybe "Felixstowe Seafront, Suffolk, England". Wait, the title hint was Seafront, Felixstowe, England, UK, GBR, Europe. So need to make it 5-10 words. Let's check: "Felixstowe Seafront, Suffolk, England, UK" – that's 10 words? Wait, no, "Felixstowe Seafront, Suffolk, England, UK" is 8 words. Maybe "Felixstowe Seafront, Suffolk, England, UK" is okay. Wait, the title should be concrete. Let's see: "Felixstowe Seafront, Suffolk, England" – that's 7 words. Maybe "Felixstowe Seafront, Suffolk, England, UK" is better. Wait, the hint says "Seafront, Felixstowe, England, UK, GBR, Europe" so the title should capture that. Let's go with "Felixstowe Seafront, Suffolk, England, UK".

Next, the description. Need 1-2 sentences. Context has capture date/time, GPS, but we don't claim them visible. The main visible subject is the seafront with people, setting is a coastal area with rocky breakwaters, people swimming, walking, beach. Lighting is sunny, clear blue sky. Distinctive details: the rocky breakwaters, the buildings in the background (like the Spa Pavilion), people in the water, seagulls. So: "A sunny day at Felixstowe seafront, England, featuring rocky breakwaters, people swimming and walking along the shore, with historic buildings and seagulls visible in the background under a clear blue sky."

Wait, check the context: capture date/time is 2026-08-13 17:23:18 UTC+01:00, but we don't claim that's visible. The description should combine relevant context (Felixstowe, England, UK) with main visible subject (seafront), setting (coastal area with breakwaters), action (people swimming, walking), lighting (sunny, clear sky), distinctive details (rocky breakwaters, historic buildings like Spa Pavilion, seagulls). So the description: "Sunny day at Felixstowe seafront, England, with rocky breakwaters, people swimming and wading in the water, historic buildings and seagulls visible under a clear blue sky."

Now keywords. Need 10-18 unique, comma-separated. Context: East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Visible details: seafront, Felixstowe, Suffolk, England, UK, gbr, seafront, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Wait, the keyword hints include Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. So need to pick unique ones. Let's list: Felixstowe, Suffolk, England, UK, gbr, seafront, East Suffolk, Europe, England, UK, gbr, seafront. Wait, but need 10-18. Let's check the image: the buildings have a Spa Pavilion, so maybe "Spa Pavilion" is a detail. Also, the rocky breakwaters, people in the water, seagulls. So keywords: Felixstowe, Suffolk, England, UK, gbr, seafront, East Suffolk, Europe, England, UK, gbr, seafront, Spa Pavilion, seagulls, beach, people, water, sunny, clear sky. Wait, but need to be unique. Let's count: Felixstowe, Suffolk, England, UK, gbr, seafront, East Suffolk, Europe, England, UK, gbr, seafront, Spa Pavilion, seagulls, beach, people, water, sunny, clear sky. Wait, but the context says "East Suffolk" is a hint, so include that. Also, "Suffolk" is in the keyword hints. Let's make sure they are unique. Let's list: Felixstowe, Suffolk, England, UK, gbr, seafront, East Suffolk, Europe, England, UK, gbr, seafront, Spa Pavilion, seagulls, beach, people, water, sunny, clear sky. Wait, but the keyword hints include "Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront". So we
</pre>

</details>

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### mlx-community/X-Reasoner-7B-8bit

<details>
<summary>Complete evidence: mlx-community/X-Reasoner-7B-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Keyword list has 179 terms (requested 10-18); Duplicate
  keywords: promenade, east suffolk coastline, east suffolk beach, felixstowe
  historic buildings, felixstowe historic architecture, felixstowe greenery,
  felixstowe historic structures, felixstowe historic town, east suffolk
  beachgoers, east suffolk beach buildings, uk seafront greenery, uk beach
  greenery
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.20s
- *Generation time:* 43.32s
- *Total time:* 44.52s
- *Input validation time:* 5.63e-05
- *Prompt preparation time:* 0.000257
- *First-token latency:* 21.1
- *Cleanup time:* 0.133
- *Prompt tokens:* 16,544
- *Generation tokens:* 1,000
- *Total tokens:* 17,544
- *Prompt throughput (raw):* 784 tok/s
- *Generation throughput (raw):* 46.8 tok/s
- *Peak memory:* 13
- *Active memory:* 9.5
- *Cache memory:* 1.3
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1262
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--X-Reasoner-7B-8bit/snapshots/21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Felixstowe Seafront, East Suffolk, England

Description:
A vibrant seafront scene in Felixstowe, East Suffolk, England, with people enjoying the sunny day at the beach and rocky promenade, set against a backdrop of historic buildings and lush greenery. The image captures the lively atmosphere of a typical summer day at the seaside town.

Keywords: Felixstowe, East Suffolk, England, UK, GBR, seafront, beach, promenade, historic buildings, greenery, sunny day, people, summer, historic architecture, coastal town, East Anglia, United Kingdom, Europe, seaside, promenade, beachgoers, historic structures, waterfront, East Suffolk coastline, British seaside, Felixstowe beach, East Anglia coast, UK seafront, historic Felixstowe, East Suffolk beach, UK coastal town, Felixstowe promenade, East Anglia seafront, UK beach scene, Felixstowe historic buildings, East Suffolk coastline, UK seaside town, Felixstowe beachgoers, East Suffolk promenade, UK seafront scene, Felixstowe historic architecture, East Suffolk beach, UK coastal promenade, Felixstowe greenery, East Suffolk buildings, UK seafront buildings, Felixstowe historic structures, East Suffolk waterfront, UK beach buildings, Felixstowe promenade buildings, East Suffolk seafront, UK beach promenade, Felixstowe historic town, East Suffolk beachgoers, UK seafront town, Felixstowe historic buildings, East Suffolk promenade buildings, UK beach town, Felixstowe greenery, East Suffolk beach buildings, UK seafront greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic town, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures
</pre>

</details>

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### mlx-community/gemma-3n-E4B-it-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E4B-it-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 2.32s
- *Generation time:* 5.13s
- *Total time:* 7.46s
- *Input validation time:* 4.89e-05
- *Prompt preparation time:* 0.00331
- *First-token latency:* 0.386
- *Cleanup time:* 0.119
- *Prompt tokens:* 568
- *Generation tokens:* 199
- *Total tokens:* 767
- *Prompt throughput (raw):* 1,472 tok/s
- *Generation throughput (raw):* 47.4 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0113
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1189
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3n-E4B-it-bf16/snapshots/d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
A wide shot captures a sunny day at Seafront in Felixstowe, England. The scene is bustling with people enjoying the coastal area. The foreground features a series of low stone walls leading down to the water, where numerous people are wading, swimming, and relaxing. Some individuals are standing in the shallow water, while others are further out. Several seagulls are scattered around, some near the people and others perched on the stone walls.

The middle ground shows the beach area, with people sunbathing and walking along the sand. A long row of rocky breakwaters extends into the sea, providing a barrier against the waves. Beyond the breakwaters, the calm, blue sea stretches out to the horizon under a clear sky.

The background is dominated by a collection of buildings, mostly residential, with varying architectural styles. Lush green trees are interspersed among the buildings, adding a touch of nature to the urban landscape. The overall atmosphere is one of leisure and relaxation on a pleasant summer day.
</pre>

</details>

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### mlx-community/llava-v1.6-mistral-7b-8bit

<details>
<summary>Complete evidence: mlx-community/llava-v1.6-mistral-7b-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava_next)
- *Model load time:* 0.97s
- *Generation time:* 4.51s
- *Total time:* 5.48s
- *Input validation time:* 4.56e-05
- *Prompt preparation time:* 0.00155
- *First-token latency:* 2.9
- *Cleanup time:* 0.0848
- *Prompt tokens:* 2,386
- *Generation tokens:* 53
- *Total tokens:* 2,439
- *Prompt throughput (raw):* 824 tok/s
- *Generation throughput (raw):* 60.6 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.90
- *Model-load active memory:* 8.05
- *Post-cleanup active memory:* 0.0124
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1134
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--llava-v1.6-mistral-7b-8bit/snapshots/b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Seafront, Felixstowe, England, UK, GBR, Europe, East Suffolk, England, UK, gbr, seafront, Adobe Stock, Any Vision, Felixstowe, Suffolk, UK, gbr,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Seafront, Felixstowe, England, UK, GBR, Europe, East Suffolk, England, UK, gbr, seafront, Adobe Stock, Any Vision, Felixstowe, Suffolk, UK, gbr, 
```

</details>

</details>

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### mlx-community/nanoLLaVA-1.5-4bit

<details>
<summary>Complete evidence: mlx-community/nanoLLaVA-1.5-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Keywords; Response repeats the task
  instructions instead of only returning the requested fields
- *Arch supported by installed mlx-vlm:* yes (model_type llava-qwen2 via
  llava_bunny)
- *Model load time:* 0.52s
- *Generation time:* 1.03s
- *Total time:* 1.56s
- *Input validation time:* 4.5e-05
- *Prompt preparation time:* 0.00401
- *First-token latency:* 0.0781
- *Cleanup time:* 0.0799
- *Prompt tokens:* 304
- *Generation tokens:* 130
- *Total tokens:* 434
- *Prompt throughput (raw):* 3,893 tok/s
- *Generation throughput (raw):* 351 tok/s
- *Peak memory:* 2.4
- *Active memory:* 0.61
- *Cache memory:* 0.30
- *Model-load active memory:* 0.611
- *Post-cleanup active memory:* 0.013
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1218
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--nanoLLaVA-1.5-4bit/snapshots/5240204744963d72823e5de933c528c4aa82dfca
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Seafront, Felixstowe, England, UK, GBR, Europe
Description: A 5-10-word title that describes the image and includes relevant context, such as the location, the photographer's watermark, and the image's metadata. The description also includes the main visible subject, setting, action, lighting, and distinctive details, such as the seafront, the location, the photographer's watermark, and the metadata. The keywords are carefully chosen to cover the most important details, such as the location, the photographer's watermark, and the metadata, while also including the main visible subject, setting, and action.
</pre>

</details>

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### mlx-community/paligemma2-3b-pt-896-4bit

<details>
<summary>Complete evidence: mlx-community/paligemma2-3b-pt-896-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Missing or empty fields:
  Title, Description, Keywords; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Model load time:* 1.13s
- *Generation time:* 23.67s
- *Total time:* 24.82s
- *Input validation time:* 3.41e-05
- *Prompt preparation time:* 0.000169
- *First-token latency:* 1.09
- *Cleanup time:* 0.124
- *Prompt tokens:* 4,397
- *Generation tokens:* 1,000
- *Total tokens:* 5,397
- *Prompt throughput (raw):* 4,019 tok/s
- *Generation throughput (raw):* 46.0 tok/s
- *Peak memory:* 4.4
- *Active memory:* 1.7
- *Cache memory:* 2.1
- *Model-load active memory:* 1.73
- *Post-cleanup active memory:* 0.014
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1118
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--paligemma2-3b-pt-896-4bit/snapshots/a26bac48c7a661dfdafe1799c90177f818e79925
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context
</pre>

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

<details>
<summary>Complete evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* decode
- *Error stage:* Model Error
- *Error code:* MLX_VLM_DECODE_MODEL
- *Error type:* ValueError
- *Error package:* mlx-vlm
- *Error message:* Model generation failed for
  mlx-community/SmolVLM2-2.2B-Instruct-mlx: not enough values to unpack
  (expected 3, got 2)
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* not enough values to unpack (expected 3, got 2)
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.70s
- *Generation time:* 1.33s
- *Total time:* 2.03s
- *Input validation time:* 4.82e-05
- *Prompt preparation time:* 0.00196
- *First-token latency:* -
- *Cleanup time:* 0.0764
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.0096
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1164
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--SmolVLM2-2.2B-Instruct-mlx/snapshots/844516024a1c4400d34489b89ee067d794e432ed
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <end_of_utterance>

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11827, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12175, in _generate_once
    return strict_generate(
        model=model,
    ...<3 lines>...
        **generate_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1158, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 977, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 393, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/idefics3/idefics3.py", line 160, in get_input_embeddings
    final_inputs_embeds = self._prepare_inputs_for_multimodal(
        image_features, inputs_embeds, input_ids
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/smolvlm/smolvlm.py", line 13, in _prepare_inputs_for_multimodal
    N, S, D_img = image_features.shape
    ^^^^^^^^^^^
ValueError: not enough values to unpack (expected 3, got 2)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12536, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12188, in _run_model_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11834, in _run_generation_guarded
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: not enough values to unpack (expected 3, got 2)

```

#### Captured upstream output

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '8', '1', '3', '-', '1', '7', '2', '3', '1', '8', '_', 'D', 'S', 'C', '0', '1', '5', '2', '5', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>User:<image>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-13 17:23:18 UTC+01:00
- GPS: 51.957967°N, 1.346900°E

Descriptive hints:
- Title hint: Seafront, Felixstowe, England, UK, GBR, Europe
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:<end_of_utterance>
Assistant:

=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 12 files:   0%|          | 0/12 [00:00<?, ?it/s]
Fetching 12 files: 100%|##########| 12/12 [00:00<00:00, 1898.38it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[18:42:19] ERROR    Generation error for mlx-community/SmolVLM2-2.2B-Instruct-mlx
                    ValueError: not enough values to unpack (expected 3, got 2)
```

</details>

---
