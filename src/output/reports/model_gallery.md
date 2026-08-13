# Model Output Gallery

Generated on: 2026-08-13 00:08:03 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                                             |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|------------|---------|--------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | 1.28s   | 517 tok/s  | 0.04            | 1.2     | 396        | 90      | none                                                                                                                     |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`              | 8.63s   | 81.0 tok/s | 5.66            | 8.4     | 6,348      | 97      | none                                                                                                                     |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 6.23s   | 34.4 tok/s | 1.25            | 17      | 3,662      | 96      | none                                                                                                                     |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | 9.76s   | 19.0 tok/s | 1.65            | 15      | 314        | 116     | none                                                                                                                     |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 6.10s   | 67.0 tok/s | 2.32            | 14      | 3,229      | 130     | none                                                                                                                     |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 3.36s   | 182 tok/s  | 1.19            | 9.0     | 3,228      | 132     | none                                                                                                                     |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | 77.33s  | 63.6 tok/s | 63.40           | 74      | 16,521     | 119     | none                                                                                                                     |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | 4.20s   | 55.6 tok/s | 0.29            | 9.4     | 1,142      | 138     | none                                                                                                                     |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 64.74s  | 110 tok/s  | 60.43           | 24      | 16,521     | 105     | none                                                                                                                     |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 65.16s  | 90.5 tok/s | 61.86           | 10.0    | 16,521     | 99      | none                                                                                                                     |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 91.91s  | 16.9 tok/s | 80.98           | 35      | 16,521     | 118     | none                                                                                                                     |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable`              | 2.39s   | 129 tok/s  | 0.33            | 5.5     | 439        | 98      | none                                                                                                                     |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.13s   | 129 tok/s  | 0.41            | 16      | 617        | 104     | none                                                                                                                     |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 7.83s   | 26.7 tok/s | 1.06            | 20      | 617        | 99      | none                                                                                                                     |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | 17.39s  | 94.8 tok/s | 13.98           | 8.4     | 16,506     | 184     | title/keyword constraints failed                                                                                         |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | 10.13s  | 30.5 tok/s | 3.60            | 24      | 2,696      | 119     | title/keyword constraints failed                                                                                         |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | 29.98s  | 38.1 tok/s | 18.04           | 78      | 6,348      | 122     | control tokens visible; title/keyword constraints failed                                                                 |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable_with_caveats` | 1.77s   | 186 tok/s  | 0.08            | 4.1     | 599        | 125     | title/keyword constraints failed                                                                                         |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | 8.43s   | 61.2 tok/s | 2.49            | 15      | 3,229      | 248     | title/keyword constraints failed                                                                                         |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | 5.62s   | 53.4 tok/s | 0.65            | 11      | 1,534      | 153     | title/keyword constraints failed                                                                                         |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | 22.10s  | 93.1 tok/s | 18.55           | 8.4     | 16,506     | 184     | title/keyword constraints failed                                                                                         |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable_with_caveats` | 29.58s  | 45.5 tok/s | 18.66           | 70      | 3,503      | 115     | title/keyword constraints failed                                                                                         |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | 5.90s   | 57.5 tok/s | 0.44            | 29      | 613        | 87      | control tokens visible                                                                                                   |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | 5.51s   | 65.8 tok/s | 0.44            | 28      | 613        | 82      | control tokens visible                                                                                                   |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | 8.41s   | 31.5 tok/s | 1.04            | 18      | 606        | 146     | title/keyword constraints failed                                                                                         |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | 6.69s   | 39.4 tok/s | 1.87            | 16      | 3,467      | 103     | title/keyword constraints failed                                                                                         |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 16.05s  | 42.7 tok/s | 2.21            | 15      | 3,558      | 500     | missing required fields; echoes instructions; cut off at token limit                                                     |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 24.46s  | 46.7 tok/s | 5.02            | 60      | 1,626      | 500     | repeated text; missing required fields; cut off at token limit; incomplete thinking block                                |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 1.87s   | 344 tok/s  | 0.10            | 2.1     | 347        | 53      | missing required fields                                                                                                  |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 18.35s  | 47.7 tok/s | 5.66            | 13      | 6,344      | 500     | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block                      |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `unusable`            | 4.16s   | 34.2 tok/s | 1.11            | 18      | 2,626      | 16      | missing required fields; role tokens visible                                                                             |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | 112.35s | 4.67 tok/s | 1.37            | 40      | 1,296      | 500     | missing required fields; cut off at token limit; incomplete thinking block                                               |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 3.55s   | 276 tok/s  | 0.26            | 3.9     | 1,003      | 500     | missing required fields; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | 27.99s  | 5.97 tok/s | 2.47            | 24      | 3,158      | 134     | missing required fields                                                                                                  |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `unusable`            | 72.59s  | 216 tok/s  | 70.70           | 5.1     | 16,517     | 124     | missing required fields                                                                                                  |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | 26.19s  | 88.8 tok/s | 18.96           | 8.4     | 16,508     | 500     | missing required fields; cut off at token limit; incomplete thinking block                                               |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | 30.87s  | 53.3 tok/s | 19.53           | 13      | 16,517     | 500     | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 5.32s   | 48.8 tok/s | 0.37            | 17      | 605        | 90      | missing required fields                                                                                                  |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | 5.01s   | 62.6 tok/s | 2.49            | 9.7     | 2,739      | 56      | missing required fields                                                                                                  |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 1.34s   | 341 tok/s  | 0.08            | 2.4     | 343        | 72      | missing required fields                                                                                                  |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 14.02s  | 45.4 tok/s | 1.10            | 4.3     | 4,433      | 500     | repeated text; missing required fields; echoes instructions; cut off at token limit                                      |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                                   | `not_evaluated`       | 0.20s   | -          | -               | -       | -          | -       | none                                                                                                                     |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 517 tok/s

Average clean-completion throughput: 109 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.2 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                                                                             |
|--------------------------------------------------------------------------------------------------------------|-----------------|--------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | missing required fields; echoes instructions; cut off at token limit                                                     |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable`      | repeated text; missing required fields; cut off at token limit; incomplete thinking block                                |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable`      | missing required fields                                                                                                  |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable`      | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block                      |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                      | `unusable`      | missing required fields; role tokens visible                                                                             |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | `unusable`      | missing required fields; cut off at token limit; incomplete thinking block                                               |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                 | `unusable`      | missing required fields; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                | `unusable`      | missing required fields                                                                                                  |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                  | `unusable`      | missing required fields                                                                                                  |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable`      | missing required fields; cut off at token limit; incomplete thinking block                                               |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                            | `unusable`      | missing required fields                                                                                                  |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable`      | missing required fields                                                                                                  |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable`      | missing required fields                                                                                                  |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable`      | repeated text; missing required fields; echoes instructions; cut off at token limit                                      |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                        | `not_evaluated` | none                                                                                                                     |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | Title:<br>The Thekla anchored in Mud Dock, Bristol Harbour<br><br>Description:<br>A black boat with a white mast and red stripe is moored in Mud Dock, Bristol Harbour, with the steeple of St Mary Redcliffe in the background.<br><br>Keywords:<br>Boat, Mud Dock, Bristol Harbour, St Mary Redcliffe, S... |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable`              | Title: Thekla Ship in Bristol Harbour<br>Description: The black ship "Thekla" is moored at Mud Dock in Bristol Harbour, with the steeple of St Mary Redcliffe Church visible in the background under a clear blue sky.<br>Keywords: Thekla ship, Bristol Harbour, St Mary Redcliffe Church,...                |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | Title: Thekla Moored in Bristol Harbour<br><br>Description: The Thekla, a ship venue, is moored in Mud Dock with St Mary Redcliffe's church spire visible in the background, under a clear sky.<br><br>Keywords: Thekla, ship venue, Mud Dock, Bristol Harbour, St Mary Redcliffe, church spire,...           |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | Title: Thekla Ship Moored in Bristol Harbour<br><br>Description: The Thekla, a ship venue, is anchored in the tranquil waters of Bristol Harbour on a sunny day, with the historic church of St Mary Redcliffe's steeple visible in the background. The ship's black hull and white mast st...                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**Thekla Ship Venue Moored in Bristol Harbour**<br><br>Description:<br>The historic vessel *Thekla*, a repurposed ship venue, is docked in Mud Dock, Bristol Harbour, under clear daylight skies. The church steeple of St Mary Redcliffe is visible in the background, with mooring li...          |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>Thekla in Mud Dock, Bristol Harbour<br><br>Description:<br>The historic sailing vessel *Thekla* is moored in Mud Dock of Bristol Harbour, with the iconic spire of St Mary Redcliffe Church visible under clear skies. Its tall mast and traditional hull, accented with bold red and w...          |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | Title: The Thekla ship moored in Bristol Harbour<br>Description: The Thekla, a former ship turned venue, is moored at the dockside in Bristol Harbour under a clear blue sky. The vessel's black hull and white mast stand in the foreground, with the distinctive spire of St Mary Redc...                   |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | Title: Thekla Ship Anchored in Bristol Harbour<br><br>Description: Thekla, a large ship, is moored in Bristol Harbour's Mud Dock, with the church of St Mary Redcliffe's spire visible in the background. The scene captures a clear day in the city centre of Bristol, England, with the s...                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: The Thekla Ship Moored in Bristol Harbour<br><br>Description: The black-hulled vessel Thekla is moored in Mud Dock, Bristol Harbour, under a clear blue sky, with the stone spire of St Mary Redcliffe church visible in the background.<br><br>Keywords: Thekla, Bristol Harbour, Moored,...          |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | Title: Thekla Ship Moored in Bristol Harbour<br><br>Description: The Thekla, a historic ship venue, is moored in Mud Dock, Bristol Harbour, with the spire of St Mary Redcliffe Church visible in the background under a clear sky.<br><br>Keywords: Thekla, Ship, Moored, Bristol Harbour, Mud D...          |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | Title:<br>The Thekla ship moored in Bristol Harbour<br><br>Description:<br>The black hull of the historic ship The Thekla is moored in Mud Dock, Bristol, under a clear blue sky. In the background, the distinctive stone spire of St Mary Redcliffe church rises above the trees on the riverba...          |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable`              | Title: Thekla, Bristol, England, UK, GBR, Europe<br>Description: The Thekla (a ship venue) anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background.<br>Keywords: Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire,...                |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: Thekla ship moored in Bristol Harbour<br><br>Description: The Thekla ship is moored in Mud Dock, Bristol Harbour, with the spire of St Mary Redcliffe church visible in the background under a clear sky. The vessel's black hull and tall white mast are captured in bright dayligh...                |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: The Thekla ship venue in Bristol Harbour<br>Description: The Thekla, a black ship venue, is moored in Mud Dock in Bristol, England, with the spire of St Mary Redcliffe church visible in the background under a clear blue sky.<br>Keywords: Architecture, Boat, Bristol, Building,...                |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | Title:<br>Thekla Ship in Bristol Harbour<br><br>Description:<br>The Thekla, a historic ship venue moored in Mud Dock, Bristol, England, is pictured under a clear blue sky with the spire of St Mary Redcliffe Church visible in the background. The vessel's black hull, red stripe, and white m...          |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | Title:<br>Thekla Ship Venue in Bristol Harbour<br><br>Description:<br>The Thekla, a moored ship venue, is anchored in Mud Dock, Bristol Harbour, with the distinctive steeple of St Mary Redcliffe Church visible in the background under a clear blue sky.<br><br>Keywords:<br>Architecture, Boat, Bristo... |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: The Thekla Ship at Mud Dock, Bristol<br>Description: A black ship named Thekla is moored at Mud Dock in Bristol Harbour, with the steeple of St Mary Redcliffe church visible in the background under a clear blue sky.<br>Keywords: Thekla, Ship, Mud Dock, Bristol...              |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable_with_caveats` | Title: Thekla at Bristol Harbour<br><br>Description: The Thekla, a historic ship venue, is moored in the Mud Dock of Bristol Harbour. The vessel is anchored with mooring lines, and the backdrop features the steeple of St Mary Redcliffe Church, adding a touch of architectural eleganc...                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | **Title:**<br>*Thekla Ship Venue Moored in Bristol’s Mud Dock<br><br>**Description:**<br>On **12 August 2026** at **17:52 UTC+01:00** (GPS: **51.4485°N, 2.593°W**), the historic **Bristol-built wooden ship *Thekla***—a converted venue vessel—is anchored in **Mud Dock, Bristol Harbour**. A...          |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | Title: The Thekla, Bristol's historic ship venue in Mud Dock<br><br>Description: A black ship named Thekla is moored in Bristol Harbour's Mud Dock, with the steeple of St Mary Redcliffe church visible in the background. The scene captures a clear day, showcasing the city's blend of...                 |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | Title:<br>Thekla Ship in Bristol Harbour<br><br>Description:<br>The Thekla, a historic ship venue moored in Mud Dock, Bristol, England, is pictured under a clear blue sky with the spire of St Mary Redcliffe Church visible in the background. The vessel's black hull, red stripe, and white m...          |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable_with_caveats` | Title:<br>Thekla ship moored in Bristol Harbour with St Mary Redcliffe spire<br><br>Description:<br>The vessel Thekla, a ship venue, is moored in Mud Dock within Bristol Harbour, with the steeple of St Mary Redcliffe church visible in the background under a clear blue sky.<br><br>Keywords:<br>Bris... |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: The Thekla ship venue moored in Bristol Harbour England<br>Description: The Thekla ship venue is moored in Mud Dock within Bristol Harbour with the spire of St Mary Redcliffe Church visible in the background under a clear sky.<br>Keywords: Architec...           |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: The Thekla Ship Moored in Bristol Harbour<br>Description: The Thekla ship venue is moored in Bristol Harbour under a clear sky with the steeple of St Mary Redcliffe church visible in the background.<br>Keywords: Bristol, Harbour, Ship, Boat, Church...           |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | Title: The Thekla, Mud Dock, Bristol Harbour, August 2026<br>Description: Captured on 12 August 2026, The Thekla, a black ship venue, is moored in Mud Dock, Bristol Harbour, with the spire of St Mary Redcliffe church visible in the background under a clear blue sky. The ship disp...                   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | Title:<br>Thekla Moored in Bristol Harbour<br><br>Description:<br>Thekla, a ship venue, is moored in Mud Dock, Bristol Harbour, with the steeple of St Mary Redcliffe church in the background under a clear sky.<br><br>Keywords:<br>Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire,... |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS coordinates. The descriptive hints: Title hint: City Centre, Bristol, England, UK, GBR, Europe. Description hi...                   |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata for this. First, the title. It's got to be concise, descriptive, and accurate. "The Thekla in Bristol Harbour" is a good start, but I can do better. "The Thekla, Bristol Harbour, St...                      |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | Thekla, Bristol Harbour, City Centre, England, UK, GBR, architecture, boat, bridge, church, church spire, clear sky, dock, England, Europe, flag, harbor, mooring, mooring lines, nautical.                                                                                                                   |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image shows Thekla (a ship venue) in Bristol Harbour, so maybe "Thekla Ship in Bristol Harbour" or something similar. Wait, the description hint says "The Thekla (a ship...                 |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `unusable`            | Thekla in Bristol Harbour with St Mary Redcliffe in the background.<end_of_utterance>                                                                                                                                                                                                                         |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | ◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords. Let's start with the title. The main subject is the Thekl...                      |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | &lt;think&gt;<br>We need to create a concrete 5-10 word title, then a 1-2 sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details. Then 10-18 unique, comma-separated keywords covering relevant context and...             |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | Thekla Ship in Mud Dock, Bristol Harbour<br><br>The Thekla, a large black ship with a white mast, is moored in Mud Dock, Bristol Harbour. The steeple of St Mary Redcliffe Church is visible in the background, with a clear blue sky overhead. The ship's name is prominently displayed on...                |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `unusable`            | Title:<br>City Centre, Bristol, England, UK, GBR, Europe<br><br>Description:<br>The Thekla, a historic ship venue, is moored in Mud Dock in Bristol Harbour, with the steeple of St Mary Redcliffe church in the background. The clear sky and calm waters create a serene atmosphere, highlighti...          |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | Got it, let's tackle this step by step. First, the title. The image shows a ship named Thekla anchored in Bristol Harbour. The context mentions Bristol, so the title should be specific. Let's see: "Thekla, Bristol Harbour, Moored" – that's 5 words. Wait, maybe "Thekla Ship, Br...                      |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | Title:<br>The Thekla in Mud Dock, Bristol<br><br>Description:<br>The historic ship Thekla, now a venue, is moored in Mud Dock, Bristol Harbour, with the steeple of St Mary Redcliffe Church visible in the background under a clear blue sky. The ship's black hull, white mast, and mooring lin...          |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | The Thekla, a historic ship, is moored in the Mud Dock of Bristol Harbour on a clear day. The ship's black hull features a red stripe and the name "THEKLA" in white lettering. It is secured to the dock with mooring lines. In the background, the tall spire of St Mary Redcliffe...                       |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical                                                                                                                        |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | Title: Thekla in Bristol, England, UK, GBR, Europe<br>Description: A ship named Thekla anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background. The image captures the ship's mooring lines and the surrounding area, with a clear...                   |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | - Title hint: Thekla (a ship venue) anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background.<br>- Description hint: Thekla (a ship venue) anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcli...                   |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                                   | `not_evaluated`       | Model loading failed: Received 362 parameters not in model:<br>audio_tower.encoder.biases,<br>audio_tower.encoder.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.biases,<br>language_model.model.layers.10.mlp.experts.down_proj.scales,<br>language_model.model.layers.10.mlp.expe...       |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.13`
- `mlx`: `0.32.1.dev20260812+52960f80f`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.15.0`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.27.0`
- *Python Version:* 3.13.13
- *OS:* Darwin 25.6.0
- *macOS Version:* 26.6.1
- *GPU/Chip:* Apple M5 Max
- *MLX Device:* Apple M5 Max
- *GPU Architecture:* applegpu_g17s
- *RAM:* 128.0 GB
- *Recommended Working Set:* 108 GB
- *Fused Attention:* Available

## Image Metadata

- *Title:* , City Centre, Bristol, England, UK, GBR, Europe
- *Description:* The Thekla (a ship venue) anchored in Mud Dock in Bristol
  Harbour with the steeple of the church of St Mary Redcliffe in the
  background.
- *Keywords:* Architecture, Boat, Bridge, Bristol, Building, Church, Church
  Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor,
  Mast, Moored, Mooring, Mooring lines, Nautical, Rigging, Ship, Sky,
  Somerset, Spire, Trees, UK, Water, anchor, black hull, blue sky, bow,
  british, calm, canal, cityscape, day, destination, docked, english,
  exterior, floating harbour, harbour, historic, marine, maritime, mud dock,
  outdoors, port, reflection, river, ropes, scenic, skyline, st mary redcliffe
  church, summer, sunny, thekla, tourism, transport, transportation, travel,
  united kingdom, urban, vessel, watercraft, waterfront
- *Date:* 2026-08-12 17:52:14 UTC+01:00
- *Time:* 17:52:14
- *GPS:* 51.448500°N, 2.593000°W

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
> &#45; Capture date/time: 2026-08-12 17:52:14 UTC+01:00
> &#45; GPS: 51.448500°N, 2.593000°W
>
> &#8203;Descriptive hints:
> &#45; Title hint: City Centre, Bristol, England, UK, GBR, Europe
> &#45; Description hint: The Thekla (a ship venue) anchored in Mud Dock in
> Bristol Harbour with the steeple of the church of St Mary Redcliffe in the
> background.
> &#45; Keyword hints: Architecture, Boat, Bridge, Bristol, Building, Church,
> Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag,
> Harbor, Mast, Moored, Mooring, Mooring lines, Nautical
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
- *Model load time:* 0.39s
- *Generation time:* 0.59s
- *Total time:* 1.28s
- *Input validation time:* 0.302
- *Prompt preparation time:* 0.00606
- *First-token latency:* 0.0369
- *Cleanup time:* 0.0577
- *Prompt tokens:* 396
- *Generation tokens:* 90
- *Total tokens:* 486
- *Prompt throughput (raw):* 10,741 tok/s
- *Generation throughput (raw):* 517 tok/s
- *Peak memory:* 1.2
- *Active memory:* 0.90
- *Cache memory:* 0.02
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1369
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* ed71acdae0799c98cb450787d049d1343b65cd52
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-450M-MLX-bf16/snapshots/ed71acdae0799c98cb450787d049d1343b65cd52
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
The Thekla anchored in Mud Dock, Bristol Harbour

Description:
A black boat with a white mast and red stripe is moored in Mud Dock, Bristol Harbour, with the steeple of St Mary Redcliffe in the background.

Keywords:
Boat, Mud Dock, Bristol Harbour, St Mary Redcliffe, Ship, Masts, Shipyard, City, City Centre, Clear Sky, Dock, England, Europe
</pre>

</details>

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### mlx-community/GLM-4.6V-Flash-mxfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-mxfp4</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.30s
- *Generation time:* 7.32s
- *Total time:* 8.63s
- *Input validation time:* 4.27e-05
- *Prompt preparation time:* 0.00712
- *First-token latency:* 5.66
- *Cleanup time:* 0.0807
- *Prompt tokens:* 6,348
- *Generation tokens:* 97
- *Total tokens:* 6,445
- *Prompt throughput (raw):* 1,121 tok/s
- *Generation throughput (raw):* 81.0 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.46
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1399
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-Flash-mxfp4/snapshots/773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Thekla Ship in Bristol Harbour
Description: The black ship "Thekla" is moored at Mud Dock in Bristol Harbour, with the steeple of St Mary Redcliffe Church visible in the background under a clear blue sky.
Keywords: Thekla ship, Bristol Harbour, St Mary Redcliffe Church, Mud Dock, Bristol, England, UK, moored ship, black hull, clear sky, Bristol Harbour water, church steeple, dockside setting.
</pre>

</details>

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### mlx-community/InternVL3-8B-bf16

<details>
<summary>Complete evidence: mlx-community/InternVL3-8B-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type internvl_chat)
- *Model load time:* 1.58s
- *Generation time:* 4.64s
- *Total time:* 6.23s
- *Input validation time:* 3.01e-05
- *Prompt preparation time:* 0.00171
- *First-token latency:* 1.25
- *Cleanup time:* 0.0982
- *Prompt tokens:* 3,662
- *Generation tokens:* 96
- *Total tokens:* 3,758
- *Prompt throughput (raw):* 2,918 tok/s
- *Generation throughput (raw):* 34.4 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.28
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.003
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1323
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.internvl_chat.processor.InternVLChatProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3-8B-bf16/snapshots/e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Thekla Moored in Bristol Harbour

Description: The Thekla, a ship venue, is moored in Mud Dock with St Mary Redcliffe's church spire visible in the background, under a clear sky.

Keywords: Thekla, ship venue, Mud Dock, Bristol Harbour, St Mary Redcliffe, church spire, moored, mooring lines, clear sky, architecture, dock, England, Europe, flag, mast, nautical
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
- *Model load time:* 1.50s
- *Generation time:* 8.25s
- *Total time:* 9.76s
- *Input validation time:* 3.83e-05
- *Prompt preparation time:* 0.0013
- *First-token latency:* 1.65
- *Cleanup time:* 0.0896
- *Prompt tokens:* 314
- *Generation tokens:* 116
- *Total tokens:* 430
- *Prompt throughput (raw):* 191 tok/s
- *Generation throughput (raw):* 19.0 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.59
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00431
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1422
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-11B-Vision-Instruct-8bit/snapshots/8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|eot_id\|>

*Readable output:*

<pre class="model-output-readable">
Title: Thekla Ship Moored in Bristol Harbour

Description: The Thekla, a ship venue, is anchored in the tranquil waters of Bristol Harbour on a sunny day, with the historic church of St Mary Redcliffe's steeple visible in the background. The ship's black hull and white mast stand out against the clear blue sky.

Keywords: Thekla, ship, Bristol Harbour, St Mary Redcliffe, church, harbour, boat, nautical, moored, mooring lines, mast, black hull, white mast, clear blue sky, sunny day.
</pre>

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
- *Model load time:* 1.34s
- *Generation time:* 4.75s
- *Total time:* 6.10s
- *Input validation time:* 3.48e-05
- *Prompt preparation time:* 0.00701
- *First-token latency:* 2.32
- *Cleanup time:* 0.0817
- *Prompt tokens:* 3,229
- *Generation tokens:* 130
- *Total tokens:* 3,359
- *Prompt throughput (raw):* 1,394 tok/s
- *Generation throughput (raw):* 67.0 tok/s
- *Peak memory:* 14
- *Active memory:* 8.0
- *Cache memory:* 0.65
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.0051
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3756
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 7c992876448fc5426841a85c6eb951b056fe38d2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-mxfp4/snapshots/7c992876448fc5426841a85c6eb951b056fe38d2
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
**Thekla Ship Venue Moored in Bristol Harbour**

Description:
The historic vessel *Thekla*, a repurposed ship venue, is docked in Mud Dock, Bristol Harbour, under clear daylight skies. The church steeple of St Mary Redcliffe is visible in the background, with mooring lines and nautical details prominently displayed.

Keywords:
Boat, Bristol Harbour, Clear Sky, Docked Ship, England, European Heritage, Historic Vessel, Mooring Lines, Mud Dock, Nautical Architecture, Ship Venue, St Mary Redcliffe Steeple, Thekla, UK Landmark, Waterfront
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
- *Model load time:* 0.94s
- *Generation time:* 2.41s
- *Total time:* 3.36s
- *Input validation time:* 4.95e-05
- *Prompt preparation time:* 0.00684
- *First-token latency:* 1.19
- *Cleanup time:* 0.0721
- *Prompt tokens:* 3,228
- *Generation tokens:* 132
- *Total tokens:* 3,360
- *Prompt throughput (raw):* 2,711 tok/s
- *Generation throughput (raw):* 182 tok/s
- *Peak memory:* 9.0
- *Active memory:* 2.8
- *Cache memory:* 0.44
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00562
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3755
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* a962dcb09eee4169c890e544c9eb938f1113fdee
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-3B-Instruct-2512-4bit/snapshots/a962dcb09eee4169c890e544c9eb938f1113fdee
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Thekla in Mud Dock, Bristol Harbour

Description:
The historic sailing vessel *Thekla* is moored in Mud Dock of Bristol Harbour, with the iconic spire of St Mary Redcliffe Church visible under clear skies. Its tall mast and traditional hull, accented with bold red and white stripes, dominate the scene alongside a serene waterfront and lush greenery.

Keywords:
Architecture, Boat, Bristol, Church, Church Spire, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring lines, Nautical, Redcliffe Church, Steeple, Traditional Ship
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
- *Model load time:* 10.84s
- *Generation time:* 66.46s
- *Total time:* 77.33s
- *Input validation time:* 5.25e-05
- *Prompt preparation time:* 0.014
- *First-token latency:* 63.4
- *Cleanup time:* 0.216
- *Prompt tokens:* 16,521
- *Generation tokens:* 119
- *Total tokens:* 16,640
- *Prompt throughput (raw):* 261 tok/s
- *Generation throughput (raw):* 63.6 tok/s
- *Peak memory:* 74
- *Active memory:* 70
- *Cache memory:* 0.88
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1409
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-bf16/snapshots/9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: The Thekla ship moored in Bristol Harbour
Description: The Thekla, a former ship turned venue, is moored at the dockside in Bristol Harbour under a clear blue sky. The vessel's black hull and white mast stand in the foreground, with the distinctive spire of St Mary Redcliffe Church visible in the background.
Keywords: Thekla, Bristol, Harbour, Ship, Moored, Mast, Hull, Church, Spire, St Mary Redcliffe, Dock, Nautical, Blue Sky, Architecture, England, UK, River Avon, Landmark
</pre>

</details>

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### mlx-community/Phi-3.5-vision-instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Phi-3.5-vision-instruct-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type phi3_v)
- *Model load time:* 0.96s
- *Generation time:* 3.23s
- *Total time:* 4.20s
- *Input validation time:* 4.87e-05
- *Prompt preparation time:* 0.000886
- *First-token latency:* 0.291
- *Cleanup time:* 0.0781
- *Prompt tokens:* 1,142
- *Generation tokens:* 138
- *Total tokens:* 1,280
- *Prompt throughput (raw):* 3,919 tok/s
- *Generation throughput (raw):* 55.6 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00713
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1339
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Thekla Ship Anchored in Bristol Harbour

Description: Thekla, a large ship, is moored in Bristol Harbour's Mud Dock, with the church of St Mary Redcliffe's spire visible in the background. The scene captures a clear day in the city centre of Bristol, England, with the ship's masts and rigging prominently displayed.

Keywords: Thekla, ship, moored, Mud Dock, Bristol Harbour, St Mary Redcliffe, church spire, city centre, clear sky, nautical, architecture
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
- *Model load time:* 2.51s
- *Generation time:* 62.21s
- *Total time:* 64.74s
- *Input validation time:* 5.67e-05
- *Prompt preparation time:* 0.0145
- *First-token latency:* 60.4
- *Cleanup time:* 0.114
- *Prompt tokens:* 16,521
- *Generation tokens:* 105
- *Total tokens:* 16,626
- *Prompt throughput (raw):* 273 tok/s
- *Generation throughput (raw):* 110 tok/s
- *Peak memory:* 24
- *Active memory:* 21
- *Cache memory:* 0.88
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00857
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1409
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 1e20fd8d42056f870933bf98ca6211024744f7ec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: The Thekla Ship Moored in Bristol Harbour

Description: The black-hulled vessel Thekla is moored in Mud Dock, Bristol Harbour, under a clear blue sky, with the stone spire of St Mary Redcliffe church visible in the background.

Keywords: Thekla, Bristol Harbour, Moored, Church Spire, St Mary Redcliffe, Nautical, Boat, Dock, Clear Sky, England, UK, Europe, Architecture, Building, City Centre, Mooring lines, Mast, Flag
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
- *Model load time:* 1.38s
- *Generation time:* 63.77s
- *Total time:* 65.16s
- *Input validation time:* 5.24e-05
- *Prompt preparation time:* 0.00031
- *First-token latency:* 61.9
- *Cleanup time:* 0.0824
- *Prompt tokens:* 16,521
- *Generation tokens:* 99
- *Total tokens:* 16,620
- *Prompt throughput (raw):* 267 tok/s
- *Generation throughput (raw):* 90.5 tok/s
- *Peak memory:* 10.0
- *Active memory:* 6.6
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00909
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1409
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Thekla Ship Moored in Bristol Harbour

Description: The Thekla, a historic ship venue, is moored in Mud Dock, Bristol Harbour, with the spire of St Mary Redcliffe Church visible in the background under a clear sky.

Keywords: Thekla, Ship, Moored, Bristol Harbour, Mud Dock, St Mary Redcliffe, Church Spire, Clear Sky, England, Europe, Architecture, Nautical, Dock, Mooring Lines, Flag, Mast
</pre>

</details>

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### mlx-community/Qwen3.6-27B-mxfp8

<details>
<summary>Complete evidence: mlx-community/Qwen3.6-27B-mxfp8</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 3.11s
- *Generation time:* 88.77s
- *Total time:* 91.91s
- *Input validation time:* 4.5e-05
- *Prompt preparation time:* 0.0151
- *First-token latency:* 81
- *Cleanup time:* 0.129
- *Prompt tokens:* 16,521
- *Generation tokens:* 118
- *Total tokens:* 16,639
- *Prompt throughput (raw):* 204 tok/s
- *Generation throughput (raw):* 16.9 tok/s
- *Peak memory:* 35
- *Active memory:* 29
- *Cache memory:* 1.8
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.0096
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1409
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-mxfp8/snapshots/5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
The Thekla ship moored in Bristol Harbour

Description:
The black hull of the historic ship The Thekla is moored in Mud Dock, Bristol, under a clear blue sky. In the background, the distinctive stone spire of St Mary Redcliffe church rises above the trees on the riverbank.

Keywords:
Bristol, The Thekla, Ship, Mud Dock, St Mary Redcliffe, Church Spire, Harbour, Moored, Mast, Nautical, River, England, UK, Clear Sky, Water, Architecture, Historic, Black Hull
</pre>

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

<details>
<summary>Complete evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.66s
- *Generation time:* 1.73s
- *Total time:* 2.39s
- *Input validation time:* 5.3e-05
- *Prompt preparation time:* 0.00184
- *First-token latency:* 0.331
- *Cleanup time:* 0.0729
- *Prompt tokens:* 439
- *Generation tokens:* 98
- *Total tokens:* 537
- *Prompt throughput (raw):* 1,326 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.19
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.00972
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1350
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--SmolVLM2-2.2B-Instruct-mlx/snapshots/844516024a1c4400d34489b89ee067d794e432ed
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <end_of_utterance>

*Readable output:*

<pre class="model-output-readable">
 Title: Thekla, Bristol, England, UK, GBR, Europe
Description: The Thekla (a ship venue) anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background.
Keywords: Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mooring, Mooring lines, Nautical
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
- *Model load time:* 2.36s
- *Generation time:* 1.74s
- *Total time:* 4.13s
- *Input validation time:* 4.19e-05
- *Prompt preparation time:* 0.0201
- *First-token latency:* 0.411
- *Cleanup time:* 0.115
- *Prompt tokens:* 617
- *Generation tokens:* 104
- *Total tokens:* 721
- *Prompt throughput (raw):* 1,503 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.21
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0119
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1372
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Thekla ship moored in Bristol Harbour

Description: The Thekla ship is moored in Mud Dock, Bristol Harbour, with the spire of St Mary Redcliffe church visible in the background under a clear sky. The vessel's black hull and tall white mast are captured in bright daylight.

Keywords: Architecture, Boat, Bristol, Church, Church Spire, City, Clear Sky, Dock, England, Europe, Harbor, Mast, Moored, Mooring lines, Nautical, Ship, Vessel, Water
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
- *Model load time:* 2.52s
- *Generation time:* 5.30s
- *Total time:* 7.83s
- *Input validation time:* 4.92e-05
- *Prompt preparation time:* 0.0003
- *First-token latency:* 1.06
- *Cleanup time:* 0.126
- *Prompt tokens:* 617
- *Generation tokens:* 99
- *Total tokens:* 716
- *Prompt throughput (raw):* 581 tok/s
- *Generation throughput (raw):* 26.7 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.80
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0124
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1372
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 696d436c404745a59f30e4939a658162b0a9e57f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/696d436c404745a59f30e4939a658162b0a9e57f
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: The Thekla ship venue in Bristol Harbour
Description: The Thekla, a black ship venue, is moored in Mud Dock in Bristol, England, with the spire of St Mary Redcliffe church visible in the background under a clear blue sky.
Keywords: Architecture, Boat, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Harbor, Mast, Moored, Mooring lines, Nautical, The Thekla
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
- *Observations:* Keyword list has 30 terms (requested 10-18); Duplicate
  keywords: england, europe, mooring, mooring lines, nautical, flag
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.65s
- *Generation time:* 16.72s
- *Total time:* 17.39s
- *Input validation time:* 4.62e-05
- *Prompt preparation time:* 0.00897
- *First-token latency:* 14
- *Cleanup time:* 0.0839
- *Prompt tokens:* 16,506
- *Generation tokens:* 184
- *Total tokens:* 16,690
- *Prompt throughput (raw):* 1,180 tok/s
- *Generation throughput (raw):* 94.8 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1390
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Thekla Ship in Bristol Harbour

Description:
The Thekla, a historic ship venue moored in Mud Dock, Bristol, England, is pictured under a clear blue sky with the spire of St Mary Redcliffe Church visible in the background. The vessel's black hull, red stripe, and white mast are prominent, with mooring lines securing it to the dock. The scene captures a serene moment in the city centre, with green trees and other boats in the surrounding water.

Keywords: Thekla, ship venue, Mud Dock, Bristol Harbour, England, UK, Europe, City Centre, St Mary Redcliffe Church, Church Spire, Dock, Moored, Mooring, Mooring lines, Nautical, Boat, Mast, Clear Sky, Harbor, Flag, Building, Architecture, Bristol, City, England, Europe, Flag, Mooring, Mooring lines, Nautical
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
- *Observations:* Keyword list has 22 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.11s
- *Generation time:* 8.01s
- *Total time:* 10.13s
- *Input validation time:* 4.57e-05
- *Prompt preparation time:* 0.00667
- *First-token latency:* 3.6
- *Cleanup time:* 0.0942
- *Prompt tokens:* 2,696
- *Generation tokens:* 119
- *Total tokens:* 2,815
- *Prompt throughput (raw):* 749 tok/s
- *Generation throughput (raw):* 30.5 tok/s
- *Peak memory:* 24
- *Active memory:* 18
- *Cache memory:* 0.52
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.000968
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1318
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Devstral-Small-2-24B-Instruct-2512-5bit/snapshots/0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Thekla Ship Venue in Bristol Harbour

Description:
The Thekla, a moored ship venue, is anchored in Mud Dock, Bristol Harbour, with the distinctive steeple of St Mary Redcliffe Church visible in the background under a clear blue sky.

Keywords:
Architecture, Boat, Bristol, Building, Church, Church Spire, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical, Ship Venue, St Mary Redcliffe, Thekla, Waterfront
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
- *Observations:* Unrecognised model control tokens remain visible; Keyword
  list has 25 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 8.11s
- *Generation time:* 21.85s
- *Total time:* 29.98s
- *Input validation time:* 5.9e-05
- *Prompt preparation time:* 0.00683
- *First-token latency:* 18
- *Cleanup time:* 0.239
- *Prompt tokens:* 6,348
- *Generation tokens:* 122
- *Total tokens:* 6,470
- *Prompt throughput (raw):* 352 tok/s
- *Generation throughput (raw):* 38.1 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1399
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-nvfp4/snapshots/2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;|begin_of_box|&gt;Title: The Thekla Ship at Mud Dock, Bristol
Description: A black ship named Thekla is moored at Mud Dock in Bristol Harbour, with the steeple of St Mary Redcliffe church visible in the background under a clear blue sky.
Keywords: Thekla, Ship, Mud Dock, Bristol Harbour, St Mary Redcliffe, Church Spire, Moored, Mooring Lines, Black Hull, Red Stripe, White Mast, Clear Sky, England, UK, Europe, Nautical, Dock, Harbor, City Centre, Architecture, Building, Boat, Mast, Flag, City&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: The Thekla Ship at Mud Dock, Bristol
Description: A black ship named Thekla is moored at Mud Dock in Bristol Harbour, with the steeple of St Mary Redcliffe church visible in the background under a clear blue sky.
Keywords: Thekla, Ship, Mud Dock, Bristol Harbour, St Mary Redcliffe, Church Spire, Moored, Mooring Lines, Black Hull, Red Stripe, White Mast, Clear Sky, England, UK, Europe, Nautical, Dock, Harbor, City Centre, Architecture, Building, Boat, Mast, Flag, City<|end_of_box|>
```

</details>

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### mlx-community/LFM2.5-VL-1.6B-bf16

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.60s
- *Generation time:* 1.16s
- *Total time:* 1.77s
- *Input validation time:* 5.52e-05
- *Prompt preparation time:* 0.00342
- *First-token latency:* 0.0806
- *Cleanup time:* 0.0684
- *Prompt tokens:* 599
- *Generation tokens:* 125
- *Total tokens:* 724
- *Prompt throughput (raw):* 7,436 tok/s
- *Generation throughput (raw):* 186 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.02
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.00379
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1369
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-1.6B-bf16/snapshots/16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Thekla at Bristol Harbour

Description: The Thekla, a historic ship venue, is moored in the Mud Dock of Bristol Harbour. The vessel is anchored with mooring lines, and the backdrop features the steeple of St Mary Redcliffe Church, adding a touch of architectural elegance to the scene. The clear sky and calm waters highlight the serene atmosphere of the harbor.

Keywords: Thekla, Bristol Harbour, moored, mooring lines, ship venue, historic, architecture, church spire, city, harbor, clear sky, water, dock, England, Europe
</pre>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 22 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.39s
- *Generation time:* 7.04s
- *Total time:* 8.43s
- *Input validation time:* 4.06e-05
- *Prompt preparation time:* 0.000233
- *First-token latency:* 2.49
- *Cleanup time:* 0.0843
- *Prompt tokens:* 3,229
- *Generation tokens:* 248
- *Total tokens:* 3,477
- *Prompt throughput (raw):* 1,298 tok/s
- *Generation throughput (raw):* 61.2 tok/s
- *Peak memory:* 15
- *Active memory:* 8.4
- *Cache memory:* 0.65
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00536
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3756
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 28777b889d841a86369c736175cb77258c8134b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-nvfp4/snapshots/28777b889d841a86369c736175cb77258c8134b2
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
**Title:**
*Thekla Ship Venue Moored in Bristol’s Mud Dock

**Description:**
On **12 August 2026** at **17:52 UTC+01:00** (GPS: **51.4485°N, 2.593°W**), the historic **Bristol-built wooden ship *Thekla***—a converted venue vessel—is anchored in **Mud Dock, Bristol Harbour**. Against a clear blue sky, its black hull and white mast stand out, with **St Mary Redcliffe’s church spire** visible in the background, mooring lines taut along the dockside.

**Keywords:**
*Thekla, Bristol, Mud Dock, Bristol Harbour, wooden ship, venue vessel, St Mary Redcliffe, church spire, moored vessel, mooring lines, historic ship, clear sky, dockside, UK architecture, nautical heritage, 2026-08-12, 17:52 UTC+01:00, 51.4485°N 2.593°W, black hull, white mast, clear blue sky, background landmark*
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
- *Observations:* Keyword list has 30 terms (requested 10-18); Duplicate
  keywords: church spire, church, city centre, clear sky, dock, england,
  europe, flag, harbor, mast, moored, mooring lines, nautical
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.20s
- *Generation time:* 4.42s
- *Total time:* 5.62s
- *Input validation time:* 3.67e-05
- *Prompt preparation time:* 0.00134
- *First-token latency:* 0.652
- *Cleanup time:* 0.0775
- *Prompt tokens:* 1,534
- *Generation tokens:* 153
- *Total tokens:* 1,687
- *Prompt throughput (raw):* 2,354 tok/s
- *Generation throughput (raw):* 53.4 tok/s
- *Peak memory:* 11
- *Active memory:* 9.0
- *Cache memory:* 0.14
- *Model-load active memory:* 9.04
- *Post-cleanup active memory:* 0.00593
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1314
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Molmo-7B-D-0924-8bit/snapshots/90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
 Title: The Thekla, Bristol's historic ship venue in Mud Dock

Description: A black ship named Thekla is moored in Bristol Harbour's Mud Dock, with the steeple of St Mary Redcliffe church visible in the background. The scene captures a clear day, showcasing the city's blend of maritime history and urban architecture.

Keywords: Bristol, Church Spire, Church, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring lines, Nautical, Thekla, Church, St Mary Redcliffe, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring lines, Nautical
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
- *Observations:* Keyword list has 30 terms (requested 10-18); Duplicate
  keywords: england, europe, mooring, mooring lines, nautical, flag
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.75s
- *Generation time:* 21.35s
- *Total time:* 22.10s
- *Input validation time:* 4e-05
- *Prompt preparation time:* 0.000211
- *First-token latency:* 18.6
- *Cleanup time:* 0.083
- *Prompt tokens:* 16,506
- *Generation tokens:* 184
- *Total tokens:* 16,690
- *Prompt throughput (raw):* 890 tok/s
- *Generation throughput (raw):* 93.1 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00775
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1390
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Instruct-bf16/snapshots/c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Thekla Ship in Bristol Harbour

Description:
The Thekla, a historic ship venue moored in Mud Dock, Bristol, England, is pictured under a clear blue sky with the spire of St Mary Redcliffe Church visible in the background. The vessel's black hull, red stripe, and white mast are prominent, with mooring lines securing it to the dock. The scene captures a serene moment in the city centre, with green trees and other boats in the surrounding water.

Keywords: Thekla, ship venue, Mud Dock, Bristol Harbour, England, UK, Europe, City Centre, St Mary Redcliffe Church, Church Spire, Dock, Moored, Mooring, Mooring lines, Nautical, Boat, Mast, Clear Sky, Harbor, Flag, Building, Architecture, Bristol, City, England, Europe, Flag, Mooring, Mooring lines, Nautical
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
- *Observations:* Title has 11 words (requested 5-10); Keyword list has 19
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 7.56s
- *Generation time:* 22.00s
- *Total time:* 29.58s
- *Input validation time:* 4.03e-05
- *Prompt preparation time:* 0.0141
- *First-token latency:* 18.7
- *Cleanup time:* 0.204
- *Prompt tokens:* 3,503
- *Generation tokens:* 115
- *Total tokens:* 3,618
- *Prompt throughput (raw):* 188 tok/s
- *Generation throughput (raw):* 45.5 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.41
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.00998
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1396
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.step3p7.processing_step3p7.Step3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Step-3.7-Flash-oQ2e/snapshots/3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Thekla ship moored in Bristol Harbour with St Mary Redcliffe spire

Description:
The vessel Thekla, a ship venue, is moored in Mud Dock within Bristol Harbour, with the steeple of St Mary Redcliffe church visible in the background under a clear blue sky.

Keywords:
Bristol, Thekla, ship venue, moored vessel, Mud Dock, Bristol Harbour, St Mary Redcliffe, church spire, clear sky, harbour, dock, nautical, architecture, boat, mooring lines, mast, flag, England, UK
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
- *Model load time:* 3.26s
- *Generation time:* 2.61s
- *Total time:* 5.90s
- *Input validation time:* 4.72e-05
- *Prompt preparation time:* 0.0201
- *First-token latency:* 0.441
- *Cleanup time:* 0.144
- *Prompt tokens:* 613
- *Generation tokens:* 87
- *Total tokens:* 700
- *Prompt throughput (raw):* 1,389 tok/s
- *Generation throughput (raw):* 57.5 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1344
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-8bit/snapshots/7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
&lt;channel|&gt;Title: The Thekla ship venue moored in Bristol Harbour England
Description: The Thekla ship venue is moored in Mud Dock within Bristol Harbour with the spire of St Mary Redcliffe Church visible in the background under a clear sky.
Keywords: Architecture, Boat, Bristol, Building, Church, Church Spire, Dock, England, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: The Thekla ship venue moored in Bristol Harbour England
Description: The Thekla ship venue is moored in Mud Dock within Bristol Harbour with the spire of St Mary Redcliffe Church visible in the background under a clear sky.
Keywords: Architecture, Boat, Bristol, Building, Church, Church Spire, Dock, England, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical
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
- *Observations:* Unrecognised model control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.15s
- *Generation time:* 2.36s
- *Total time:* 5.51s
- *Input validation time:* 4.33e-05
- *Prompt preparation time:* 0.000295
- *First-token latency:* 0.438
- *Cleanup time:* 0.13
- *Prompt tokens:* 613
- *Generation tokens:* 82
- *Total tokens:* 695
- *Prompt throughput (raw):* 1,401 tok/s
- *Generation throughput (raw):* 65.8 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1344
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-mxfp8/snapshots/ded389e478f86d498ad9e7f47666e83b166a28f1
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
&lt;channel|&gt;Title: The Thekla Ship Moored in Bristol Harbour
Description: The Thekla ship venue is moored in Bristol Harbour under a clear sky with the steeple of St Mary Redcliffe church visible in the background.
Keywords: Bristol, Harbour, Ship, Boat, Church Spire, Mast, Moored, Mooring lines, Nautical, England, City Centre, Clear Sky, Architecture
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: The Thekla Ship Moored in Bristol Harbour
Description: The Thekla ship venue is moored in Bristol Harbour under a clear sky with the steeple of St Mary Redcliffe church visible in the background.
Keywords: Bristol, Harbour, Ship, Boat, Church Spire, Mast, Moored, Mooring lines, Nautical, England, City Centre, Clear Sky, Architecture
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 2.23s
- *Generation time:* 6.17s
- *Total time:* 8.41s
- *Input validation time:* 4.68e-05
- *Prompt preparation time:* 0.00275
- *First-token latency:* 1.04
- *Cleanup time:* 0.128
- *Prompt tokens:* 606
- *Generation tokens:* 146
- *Total tokens:* 752
- *Prompt throughput (raw):* 585 tok/s
- *Generation throughput (raw):* 31.5 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.45
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0108
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1373
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3.processing_gemma3.Gemma3Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3-27b-it-qat-4bit/snapshots/fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: The Thekla, Mud Dock, Bristol Harbour, August 2026
Description: Captured on 12 August 2026, The Thekla, a black ship venue, is moored in Mud Dock, Bristol Harbour, with the spire of St Mary Redcliffe church visible in the background under a clear blue sky. The ship displays prominent branding and mooring lines, reflecting its function and location.
Keywords: The Thekla, Mud Dock, Bristol, Bristol Harbour, St Mary Redcliffe, Church Spire, Ship, Boat, Venue, Moored, Mooring, Architecture, City Centre, England, Clear Sky, Water, Flag, Mast, Branding, August 2026
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.71s
- *Generation time:* 4.97s
- *Total time:* 6.69s
- *Input validation time:* 8.53e-05
- *Prompt preparation time:* 0.00262
- *First-token latency:* 1.87
- *Cleanup time:* 0.0917
- *Prompt tokens:* 3,467
- *Generation tokens:* 103
- *Total tokens:* 3,570
- *Prompt throughput (raw):* 1,853 tok/s
- *Generation throughput (raw):* 39.4 tok/s
- *Peak memory:* 16
- *Active memory:* 13
- *Cache memory:* 0.64
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0144
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1318
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--pixtral-12b-8bit/snapshots/79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Thekla Moored in Bristol Harbour

Description:
Thekla, a ship venue, is moored in Mud Dock, Bristol Harbour, with the steeple of St Mary Redcliffe church in the background under a clear sky.

Keywords:
Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical
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
- *Model load time:* 1.62s
- *Generation time:* 14.41s
- *Total time:* 16.05s
- *Input validation time:* 5.52e-05
- *Prompt preparation time:* 0.00704
- *First-token latency:* 2.21
- *Cleanup time:* 0.0811
- *Prompt tokens:* 3,558
- *Generation tokens:* 500
- *Total tokens:* 4,058
- *Prompt throughput (raw):* 1,612 tok/s
- *Generation throughput (raw):* 42.7 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.82
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.000706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1692
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Apriel-1.5-15b-Thinker-6bit-MLX/snapshots/24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Here are my reasoning steps:
We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS coordinates. The descriptive hints: Title hint: City Centre, Bristol, England, UK, GBR, Europe. Description hint: The Thekla (a ship venue) anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background. Keyword hints: Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical.

We need to produce:

- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

We must treat capture date/time and GPS as authoritative facts, but not claim they are visible. So we can incorporate them in description if needed? The description should combine relevant context with main visible subject, setting, action, lighting, and distinctive details. We can mention the date/time? Usually description includes date/time? The instruction: "Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible." So we can incorporate them as metadata? The description can mention the date/time? Possibly not needed. But we can incorporate the location (Bristol, Mud Dock). The description should be 1-2 sentences. So we can say: "The Thekla, a repurposed ship venue, is moored in Bristol's Mud Dock under a clear blue sky, with St Mary Redcliffe's iconic spire visible in the background." That's one sentence. Could add "The photograph was taken on 12 August 2026 at 17:52 UTC+01:00, with the ship's black hull and red and white stripes prominent." But that might be too many details. The description should combine relevant context with main visible subject, setting, action, lighting, and distinctive details. So we can mention the clear sky, the ship's black hull, the red and white stripes, the mooring lines, the mast, the church spire. So maybe
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
- *Model load time:* 8.12s
- *Generation time:* 16.33s
- *Total time:* 24.46s
- *Input validation time:* 5.23e-05
- *Prompt preparation time:* 0.00752
- *First-token latency:* 5.02
- *Cleanup time:* 0.212
- *Prompt tokens:* 1,626
- *Generation tokens:* 500
- *Total tokens:* 2,126
- *Prompt throughput (raw):* 324 tok/s
- *Generation throughput (raw):* 46.7 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.14
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00118
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1508
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--ERNIE-4.5-VL-28B-A3B-Thinking-bf16/snapshots/32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata for this. First, the title. It's got to be concise, descriptive, and accurate. "The Thekla in Bristol Harbour" is a good start, but I can do better. "The Thekla, Bristol Harbour, St Mary Redcliffe" is even better, but I'll keep it simple. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla, Bristol Harbour, St Mary Redcliffe" is a bit long. "The Thekla
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
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava_qwen2 via
  fastvlm)
- *Model load time:* 0.60s
- *Generation time:* 1.26s
- *Total time:* 1.87s
- *Input validation time:* 9.23e-05
- *Prompt preparation time:* 0.00156
- *First-token latency:* 0.0977
- *Cleanup time:* 0.071
- *Prompt tokens:* 347
- *Generation tokens:* 53
- *Total tokens:* 400
- *Prompt throughput (raw):* 3,553 tok/s
- *Generation throughput (raw):* 344 tok/s
- *Peak memory:* 2.1
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00149
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1413
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--FastVLM-0.5B-bf16/snapshots/81ffe929046666c43de53691147b1669ba0f3a4c
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Thekla, Bristol Harbour, City Centre, England, UK, GBR, architecture, boat, bridge, church, church spire, clear sky, dock, England, Europe, flag, harbor, mooring, mooring lines, nautical.
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
- *Observations:* Missing or empty fields: Keywords; Extra text appears before
  the Title field; Response appears cut off at the token limit; Internal
  reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.75s
- *Generation time:* 16.60s
- *Total time:* 18.35s
- *Input validation time:* 4.09e-05
- *Prompt preparation time:* 0.00166
- *First-token latency:* 5.66
- *Cleanup time:* 0.0936
- *Prompt tokens:* 6,344
- *Generation tokens:* 500
- *Total tokens:* 6,844
- *Prompt throughput (raw):* 1,120 tok/s
- *Generation throughput (raw):* 47.7 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.41
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1375
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.1V-9B-Thinking-8bit/snapshots/9677807f106500eb7690391c27645d59f6855cfb
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image shows Thekla (a ship venue) in Bristol Harbour, so maybe "Thekla Ship in Bristol Harbour" or something similar. Wait, the description hint says "The Thekla (a ship venue) anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background." So title could be "Thekla Ship at Bristol Harbour" or "Bristol Harbour Ship Scene".

Then the description: need to combine context (date/time, location) with main subject, setting, action, lighting, distinctive details. The ship is moored, clear sky, Bristol Harbour, church steeple. So something like "Thekla ship moored in Bristol Harbour on 2026-08-12, with St Mary Redcliffe's steeple visible under clear blue skies."

Keywords need to be 10-18 unique, comma-separated. Let's list the hints and add visible details. Hints include Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical. Also visible details: ship (Thekla), church steeple, mooring lines, clear sky, Bristol Harbour.

Now, let's structure each part.

Title: Need 5-10 words. "Thekla Ship in Bristol Harbour" is about 4 words, maybe "Thekla Moored in Bristol Harbour" – 5 words. Or "Bristol Harbour Ship Scene" – 3 words, but need 5-10. Let's go with "Thekla Ship at Bristol Harbour" (5 words).

Description: Combine context (date/time, location) with subject, setting, action, lighting, details. So: "Thekla ship moored in Bristol Harbour on 2026-08-12, with St Mary Redcliffe's church steeple visible under clear blue skies." That's two sentences? Wait, need 1-2 sentences. So maybe "Thekla, a ship venue, is moored in Bristol Harbour on 2026-08-12, with the steeple of St Mary Redcliffe church visible under clear blue skies." That's one sentence, covers context, subject, setting
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image shows Thekla (a ship venue) in Bristol Harbour, so maybe "Thekla Ship in Bristol Harbour" or something similar. Wait, the description hint says "The Thekla (a ship venue) anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background." So title could be "Thekla Ship at Bristol Harbour" or "Bristol Harbour Ship Scene".

Then the description: need to combine context (date/time, location) with main subject, setting, action, lighting, distinctive details. The ship is moored, clear sky, Bristol Harbour, church steeple. So something like "Thekla ship moored in Bristol Harbour on 2026-08-12, with St Mary Redcliffe's steeple visible under clear blue skies."

Keywords need to be 10-18 unique, comma-separated. Let's list the hints and add visible details. Hints include Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical. Also visible details: ship (Thekla), church steeple, mooring lines, clear sky, Bristol Harbour.

Now, let's structure each part.

Title: Need 5-10 words. "Thekla Ship in Bristol Harbour" is about 4 words, maybe "Thekla Moored in Bristol Harbour" – 5 words. Or "Bristol Harbour Ship Scene" – 3 words, but need 5-10. Let's go with "Thekla Ship at Bristol Harbour" (5 words).

Description: Combine context (date/time, location) with subject, setting, action, lighting, details. So: "Thekla ship moored in Bristol Harbour on 2026-08-12, with St Mary Redcliffe's church steeple visible under clear blue skies." That's two sentences? Wait, need 1-2 sentences. So maybe "Thekla, a ship venue, is moored in Bristol Harbour on 2026-08-12, with the steeple of St Mary Redcliffe church visible under clear blue skies." That's one sentence, covers context, subject, setting
```

</details>

</details>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Conversation-role control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 1.92s
- *Generation time:* 2.24s
- *Total time:* 4.16s
- *Input validation time:* 5.53e-05
- *Prompt preparation time:* 0.00127
- *First-token latency:* 1.11
- *Cleanup time:* 0.0938
- *Prompt tokens:* 2,626
- *Generation tokens:* 16
- *Total tokens:* 2,642
- *Prompt throughput (raw):* 2,362 tok/s
- *Generation throughput (raw):* 34.2 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.00269
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1355
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Idefics3-8B-Llama3-bf16/snapshots/8c2a30c48864f3251701b7bde40f601d25535098
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
 Thekla in Bristol Harbour with St Mary Redcliffe in the background.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Thekla in Bristol Harbour with St Mary Redcliffe in the background.<end_of_utterance>
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
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response appears cut off at the token limit; Internal reasoning block
  appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 3.24s
- *Generation time:* 109.09s
- *Total time:* 112.35s
- *Input validation time:* 4.57e-05
- *Prompt preparation time:* 0.00184
- *First-token latency:* 1.37
- *Cleanup time:* 0.174
- *Prompt tokens:* 1,296
- *Generation tokens:* 500
- *Total tokens:* 1,796
- *Prompt throughput (raw):* 944 tok/s
- *Generation throughput (raw):* 4.67 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.6
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00365
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1505
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Kimi-VL-A3B-Thinking-2506-bf16/snapshots/fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* [EOS]

*Readable output:*

<pre class="model-output-readable">
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords. Let's start with the title. The main subject is the Thekla ship in Bristol Harbour, with the church steeple in the background. So maybe "Thekla ship moored in Bristol Harbour with St Mary Redcliffe spire visible." That's concise and includes key elements.

Next, the description. It should mention the ship, location, time, and key details. The image shows the Thekla in Mud Dock, with the church steeple (St Mary Redcliffe) in the background, clear sky, and the ship's details like the mast and mooring lines. So the description could be: "The Thekla, a historic ship moored in Mud Dock, Bristol Harbour, with the steeple of St Mary Redcliffe visible against a clear blue sky." That captures the main elements.

For keywords, list unique terms from the image and context. Thekla, Bristol Harbour, Mud Dock, St Mary Redcliffe, ship, mast, mooring lines, England, Europe, clear sky, bridge (maybe the bridge in the background?), building (like the dark building), trees, water, etc. Let me check the hints: Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring lines, Nautical. So include those. Maybe: Thekla, Bristol Harbour, Mud Dock, St Mary Redcliffe, ship, mast, mooring lines, clear sky, bridge, building, England, Europe, nautical vessel, city centre, church spire, water, trees. That's 14 keywords. Need to make sure they are unique and relevant. Let me count again. Thekla, Bristol Harbour, Mud Dock, St Mary Redcliffe, ship, mast, mooring lines, clear sky, bridge, building, England, Europe, nautical vessel, city centre, church spire, water, trees. That's 15. Maybe remove "nautical vessel" as it's redundant with ship. So adjust to: Thekla, Bristol Harbour, Mud Dock, St Mary Redcliffe, ship,
</pre>

</details>

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### mlx-community/MiniCPM-V-4.6-8bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.6-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Keywords; Response repeats
  the task instructions instead of only returning the requested fields; Extra
  text appears before the Title field; Response appears cut off at the token
  limit; Internal reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Model load time:* 0.85s
- *Generation time:* 2.68s
- *Total time:* 3.55s
- *Input validation time:* 4.12e-05
- *Prompt preparation time:* 0.00882
- *First-token latency:* 0.262
- *Cleanup time:* 0.0797
- *Prompt tokens:* 1,003
- *Generation tokens:* 500
- *Total tokens:* 1,503
- *Prompt throughput (raw):* 3,834 tok/s
- *Generation throughput (raw):* 276 tok/s
- *Peak memory:* 3.9
- *Active memory:* 2.3
- *Cache memory:* 0.06
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00483
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1386
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.6-8bit/snapshots/03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;
We need to create a concrete 5-10 word title, then a 1-2 sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details. Then 10-18 unique, comma-separated keywords covering relevant context and visible details.

First, title: should be concrete, short, around 5-10 words. Based on the image: a black ship moored in Bristol harbor with church spire. So title: Black Ship Moored at Bristol Harbor with Church Spire.

But need 5-10 words. Let's count: "Black Ship Moored at Bristol Harbor with Church Spire" is 6 words. Could be "Boat Moored in Bristol Harbor, Featuring Church Spire". But better to include key elements: ship, harbor, church, clear sky.

Description: The image shows a large black ship moored in Bristol Harbor, with a church steeple visible in the background under clear blue skies.

Now, keywords: need 10-18 unique, covering relevant context and visible details. Let's list:

- ship, boat, harbor, Bristol, church, church spire, mast, moored, dock, England, Europe, clear sky, nautical, architecture, masts, rigging, flags, dockside, green trees, clear weather, harbor water, anchor, S, Thekla, mooring lines, city center, England, UK, GBR, clear, sunny, architecture, building, steeple, church, harbor, dock, England, Europe, nautical, boat, ship, masts, rigging, moored, dockside, trees, clear sky, architecture, building, church spire, harbor, mooring, nautical, England, UK, GBR, clear, sunny, dock, city center, Europe, nautical, ship, harbor, architecture, building, steeple, church, moored, dockside, trees, clear sky, England, UK, GBR, clear, sunny, mooring lines, anchor, masts, rigging, flags, dock, city center, Europe, nautical, ship, harbor, architecture, building, steeple, church, moored, dockside, trees, clear sky, England, UK, GBR, clear, sunny, mooring, nautical, ship, harbor,
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>
We need to create a concrete 5-10 word title, then a 1-2 sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details. Then 10-18 unique, comma-separated keywords covering relevant context and visible details.

First, title: should be concrete, short, around 5-10 words. Based on the image: a black ship moored in Bristol harbor with church spire. So title: Black Ship Moored at Bristol Harbor with Church Spire.

But need 5-10 words. Let's count: "Black Ship Moored at Bristol Harbor with Church Spire" is 6 words. Could be "Boat Moored in Bristol Harbor, Featuring Church Spire". But better to include key elements: ship, harbor, church, clear sky.

Description: The image shows a large black ship moored in Bristol Harbor, with a church steeple visible in the background under clear blue skies.

Now, keywords: need 10-18 unique, covering relevant context and visible details. Let's list:

- ship, boat, harbor, Bristol, church, church spire, mast, moored, dock, England, Europe, clear sky, nautical, architecture, masts, rigging, flags, dockside, green trees, clear weather, harbor water, anchor, S, Thekla, mooring lines, city center, England, UK, GBR, clear, sunny, architecture, building, steeple, church, harbor, dock, England, Europe, nautical, boat, ship, masts, rigging, moored, dockside, trees, clear sky, architecture, building, church spire, harbor, mooring, nautical, England, UK, GBR, clear, sunny, dock, city center, Europe, nautical, ship, harbor, architecture, building, steeple, church, moored, dockside, trees, clear sky, England, UK, GBR, clear, sunny, mooring lines, anchor, masts, rigging, flags, dock, city center, Europe, nautical, ship, harbor, architecture, building, steeple, church, moored, dockside, trees, clear sky, England, UK, GBR, clear, sunny, mooring, nautical, ship, harbor,
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
- *Model load time:* 2.27s
- *Generation time:* 25.70s
- *Total time:* 27.99s
- *Input validation time:* 4.51e-05
- *Prompt preparation time:* 0.00481
- *First-token latency:* 2.47
- *Cleanup time:* 0.1
- *Prompt tokens:* 3,158
- *Generation tokens:* 134
- *Total tokens:* 3,292
- *Prompt throughput (raw):* 1,278 tok/s
- *Generation throughput (raw):* 5.97 tok/s
- *Peak memory:* 24
- *Active memory:* 19
- *Cache memory:* 5.0
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.00655
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1356
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo_point.processing_molmo_point.MolmoPointProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MolmoPoint-8B-fp16/snapshots/0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Thekla Ship in Mud Dock, Bristol Harbour

The Thekla, a large black ship with a white mast, is moored in Mud Dock, Bristol Harbour. The steeple of St Mary Redcliffe Church is visible in the background, with a clear blue sky overhead. The ship's name is prominently displayed on its side, and it is secured with mooring lines to the dock.

Bristol, England, Europe, Thekla, Mud Dock, Ship, Harbour, Church, St Mary Redcliffe, Mooring, Mooring lines, Clear Sky, Nautical, Boat, Dock, Architecture, Building, Mast, Anchor, Ship venue
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
- *Model load time:* 0.50s
- *Generation time:* 72.08s
- *Total time:* 72.59s
- *Input validation time:* 4.71e-05
- *Prompt preparation time:* 0.00377
- *First-token latency:* 70.7
- *Cleanup time:* 0.079
- *Prompt tokens:* 16,517
- *Generation tokens:* 124
- *Total tokens:* 16,641
- *Prompt throughput (raw):* 234 tok/s
- *Generation throughput (raw):* 216 tok/s
- *Peak memory:* 5.1
- *Active memory:* 1.3
- *Cache memory:* 1.00
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.00744
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1448
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 01af461cdb9574acc09084a0ef94e216e142b085
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen2-VL-2B-Instruct-4bit/snapshots/01af461cdb9574acc09084a0ef94e216e142b085
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
City Centre, Bristol, England, UK, GBR, Europe

Description:
The Thekla, a historic ship venue, is moored in Mud Dock in Bristol Harbour, with the steeple of St Mary Redcliffe church in the background. The clear sky and calm waters create a serene atmosphere, highlighting the black hull and red stripe of the ship. The boat is moored with ropes, and the mast is visible, with a black and white flag on top. The scene is set against a backdrop of green trees and a blue sky, creating a picturesque view of the city's harbor.
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
- *Model load time:* 0.75s
- *Generation time:* 25.42s
- *Total time:* 26.19s
- *Input validation time:* 4.75e-05
- *Prompt preparation time:* 0.0115
- *First-token latency:* 19
- *Cleanup time:* 0.14
- *Prompt tokens:* 16,508
- *Generation tokens:* 500
- *Total tokens:* 17,008
- *Prompt throughput (raw):* 871 tok/s
- *Generation throughput (raw):* 88.8 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.4
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00806
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1398
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Thinking-bf16/snapshots/c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Got it, let's tackle this step by step. First, the title. The image shows a ship named Thekla anchored in Bristol Harbour. The context mentions Bristol, so the title should be specific. Let's see: "Thekla, Bristol Harbour, Moored" – that's 5 words. Wait, maybe "Thekla Ship, Bristol Harbour, Moored" but need to be concise. The hint says "City Centre, Bristol, England, UK, GBR, Europe" so title should include Thekla and location.

Next, the description. Need to combine context with visible subject, setting, action, lighting, details. The ship is Thekla, anchored in Mud Dock (from context), background has St Mary Redcliffe church spire. The sky is clear blue, so lighting is bright. Visible details: black hull, red stripe, mast, mooring lines, trees, church spire. Action: moored. So: "Thekla ship, black hull with red stripe, moored in Bristol Harbour Mud Dock, clear blue sky, with St Mary Redcliffe church spire in background."

Then keywords. Need 10-18 unique, comma-separated. From context: Architecture, Boat, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical. Let's check image: ship is moored, so Moored, Mooring lines. Church spire is visible. The ship has a mast. The hull is black with red stripe. So keywords: Architecture, Boat, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring lines, Nautical. Wait, need to pick 10-18. Let's list: Architecture, Boat, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring lines, Nautical. That's 17. Let's check if any are missing. The ship has a mast, so Mast. The context says "Thekla (a ship venue)" so maybe include Ship, but the hint says "Ship venue" but the image is a ship. Wait,
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
  the token limit; Keyword list has 199 terms (requested 10-18); Duplicate
  keywords: england, europe, flag, building, bridge, dock
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.16s
- *Generation time:* 29.70s
- *Total time:* 30.87s
- *Input validation time:* 5.36e-05
- *Prompt preparation time:* 0.000264
- *First-token latency:* 19.5
- *Cleanup time:* 0.0945
- *Prompt tokens:* 16,517
- *Generation tokens:* 500
- *Total tokens:* 17,017
- *Prompt throughput (raw):* 846 tok/s
- *Generation throughput (raw):* 53.3 tok/s
- *Peak memory:* 13
- *Active memory:* 9.5
- *Cache memory:* 1.3
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1448
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--X-Reasoner-7B-8bit/snapshots/21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
The Thekla in Mud Dock, Bristol

Description:
The historic ship Thekla, now a venue, is moored in Mud Dock, Bristol Harbour, with the steeple of St Mary Redcliffe Church visible in the background under a clear blue sky. The ship's black hull, white mast, and mooring lines are prominent, while the surrounding greenery and urban structures frame the scene.

Keywords: Thekla, Mud Dock, Bristol, St Mary Redcliffe, Church Spire, Clear Sky, Harbor, Black Hull, White Mast, Moored, Mooring Lines, Nautical, Architecture, Boat, City Centre, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building, Bridge, Dock, England, Europe, Flag, Building,
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
- *Model load time:* 2.62s
- *Generation time:* 2.69s
- *Total time:* 5.32s
- *Input validation time:* 5.45e-05
- *Prompt preparation time:* 0.00308
- *First-token latency:* 0.373
- *Cleanup time:* 0.119
- *Prompt tokens:* 605
- *Generation tokens:* 90
- *Total tokens:* 695
- *Prompt throughput (raw):* 1,624 tok/s
- *Generation throughput (raw):* 48.8 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0114
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1375
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3n-E4B-it-bf16/snapshots/d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
The Thekla, a historic ship, is moored in the Mud Dock of Bristol Harbour on a clear day. The ship's black hull features a red stripe and the name "THEKLA" in white lettering. It is secured to the dock with mooring lines. In the background, the tall spire of St Mary Redcliffe Church rises above the surrounding trees. The sky is a bright blue, and the water in the harbor is calm.
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
- *Model load time:* 0.94s
- *Generation time:* 4.06s
- *Total time:* 5.01s
- *Input validation time:* 4.4e-05
- *Prompt preparation time:* 0.00157
- *First-token latency:* 2.49
- *Cleanup time:* 0.0869
- *Prompt tokens:* 2,739
- *Generation tokens:* 56
- *Total tokens:* 2,795
- *Prompt throughput (raw):* 1,098 tok/s
- *Generation throughput (raw):* 62.6 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.90
- *Model-load active memory:* 8.05
- *Post-cleanup active memory:* 0.0125
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1320
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--llava-v1.6-mistral-7b-8bit/snapshots/b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical
</pre>

<details>
<summary>Exact raw output</summary>

```text
Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical 
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
- *Observations:* Missing or empty fields: Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava-qwen2 via
  llava_bunny)
- *Model load time:* 0.51s
- *Generation time:* 0.83s
- *Total time:* 1.34s
- *Input validation time:* 4.23e-05
- *Prompt preparation time:* 0.00133
- *First-token latency:* 0.0759
- *Cleanup time:* 0.0809
- *Prompt tokens:* 343
- *Generation tokens:* 72
- *Total tokens:* 415
- *Prompt throughput (raw):* 4,521 tok/s
- *Generation throughput (raw):* 341 tok/s
- *Peak memory:* 2.4
- *Active memory:* 0.61
- *Cache memory:* 0.30
- *Model-load active memory:* 0.611
- *Post-cleanup active memory:* 0.0131
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1404
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--nanoLLaVA-1.5-4bit/snapshots/5240204744963d72823e5de933c528c4aa82dfca
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Thekla in Bristol, England, UK, GBR, Europe
Description: A ship named Thekla anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background. The image captures the ship's mooring lines and the surrounding area, with a clear sky in the background.
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
  Title, Description, Keywords; Response repeats the task instructions instead
  of only returning the requested fields; Response appears cut off at the
  token limit
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Model load time:* 1.12s
- *Generation time:* 12.89s
- *Total time:* 14.02s
- *Input validation time:* 4.19e-05
- *Prompt preparation time:* 0.000166
- *First-token latency:* 1.1
- *Cleanup time:* 0.172
- *Prompt tokens:* 4,433
- *Generation tokens:* 500
- *Total tokens:* 4,933
- *Prompt throughput (raw):* 4,041 tok/s
- *Generation throughput (raw):* 45.4 tok/s
- *Peak memory:* 4.3
- *Active memory:* 1.7
- *Cache memory:* 1.9
- *Model-load active memory:* 1.73
- *Post-cleanup active memory:* 0.0141
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1304
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--paligemma2-3b-pt-896-4bit/snapshots/a26bac48c7a661dfdafe1799c90177f818e79925
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
- Title hint: Thekla (a ship venue) anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background.
- Description hint: Thekla (a ship venue) anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background.
- Keyword hints: Architecture, Building, Church, Flag, Harbor, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island, Island,
</pre>

</details>

---

<a id="model-mlx-community-inkling-small-mlx-4bit"></a>

### mlx-community/Inkling-Small-mlx-4bit

<details>
<summary>Complete evidence: mlx-community/Inkling-Small-mlx-4bit</summary>

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* model_load
- *Error stage:* Model Error
- *Error code:* MLX_VLM_MODEL_LOAD_MODEL
- *Error type:* ValueError
- *Error package:* mlx-vlm
- *Error message:* Model loading failed: Received 362 parameters not in model:
  <br>audio_tower.encoder.biases,<br>audio_tower.encoder.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.biases,<br>language_model.model.layers.10.mlp.experts.down_proj.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.weight,<br>language_model.model.layers.10.mlp.experts.gate_proj.biases,<br>language_model.model.layers.10.mlp.experts.gate_proj.scales,<br>language_model.model.layers.10.mlp.experts.gate_proj.weight,<br>language_model.model.layers.10.mlp.experts.up_proj.biases,<br>language_model.model.layers.10.mlp.experts.up_proj.scales,<br>language_model.model.layers.10.mlp.experts.up_proj.weight,<br>language_model.model.layers.11.mlp.experts.down_proj.biases,<br>language_model.model.layers.11.mlp.experts.down_proj.scales,<br>language_model.model.layers.11.mlp.experts.down_proj.weight,<br>language_model.model.layers.11.mlp.experts.gate_proj.biases,<br>language_model.model.layers.11.mlp.experts.gate_proj.scales,<br>language_model.model.layers.11.mlp.experts.gate_proj.weight,<br>language_model.model.layers.11.mlp.experts.up_proj.biases,<br>language_model.model.layers.11.mlp.experts.up_proj.scales,<br>language_model.model.layers.11.mlp.experts.up_proj.weight,<br>language_model.model.layers.12.mlp.experts.down_proj.biases,<br>language_model.model.layers.12.mlp.experts.down_proj.scales,<br>language_model.model.layers.12.mlp.experts.down_proj.weight,<br>language_model.model.layers.12.mlp.experts.gate_proj.biases,<br>language_model.model.layers.12.mlp.experts.gate_proj.scales,<br>language_model.model.layers.12.mlp.experts.gate_proj.weight,<br>language_model.model.layers.12.mlp.experts.up_proj.biases,<br>language_model.model.layers.12.mlp.experts.up_proj.scales,<br>language_model.model.layers.12.mlp.experts.up_proj.weight,<br>language_model.model.layers.13.mlp.experts.down_proj.biases,<br>language_model.model.layers.13.mlp.experts.down_proj.scales,<br>language_model.model.layers.13.mlp.experts.down_proj.weight,<br>language_model.model.layers.13.mlp.experts.gate_proj.biases,<br>language_model.model.layers.13.mlp.experts.gate_proj.scales,<br>language_model.model.layers.13.mlp.experts.gate_proj.weight,<br>language_model.model.layers.13.mlp.experts.up_proj.biases,<br>language_model.model.layers.13.mlp.experts.up_proj.scales,<br>language_model.model.layers.13.mlp.experts.up_proj.weight,<br>language_model.model.layers.14.mlp.experts.down_proj.biases,<br>language_model.model.layers.14.mlp.experts.down_proj.scales,<br>language_model.model.layers.14.mlp.experts.down_proj.weight,<br>language_model.model.layers.14.mlp.experts.gate_proj.biases,<br>language_model.model.layers.14.mlp.experts.gate_proj.scales,<br>language_model.model.layers.14.mlp.experts.gate_proj.weight,<br>language_model.model.layers.14.mlp.experts.up_proj.biases,<br>language_model.model.layers.14.mlp.experts.up_proj.scales,<br>language_model.model.layers.14.mlp.experts.up_proj.weight,<br>language_model.model.layers.15.mlp.experts.down_proj.biases,<br>language_model.model.layers.15.mlp.experts.down_proj.scales,<br>language_model.model.layers.15.mlp.experts.down_proj.weight,<br>language_model.model.layers.15.mlp.experts.gate_proj.biases,<br>language_model.model.layers.15.mlp.experts.gate_proj.scales,<br>language_model.model.layers.15.mlp.experts.gate_proj.weight,<br>language_model.model.layers.15.mlp.experts.up_proj.biases,<br>language_model.model.layers.15.mlp.experts.up_proj.scales,<br>language_model.model.layers.15.mlp.experts.up_proj.weight,<br>language_model.model.layers.16.mlp.experts.down_proj.biases,<br>language_model.model.layers.16.mlp.experts.down_proj.scales,<br>language_model.model.layers.16.mlp.experts.down_proj.weight,<br>language_model.model.layers.16.mlp.experts.gate_proj.biases,<br>language_model.model.layers.16.mlp.experts.gate_proj.scales,<br>language_model.model.layers.16.mlp.experts.gate_proj.weight,<br>language_model.model.layers.16.mlp.experts.up_proj.biases,<br>language_model.model.layers.16.mlp.experts.up_proj.scales,<br>language_model.model.layers.16.mlp.experts.up_proj.weight,<br>language_model.model.layers.17.mlp.experts.down_proj.biases,<br>language_model.model.layers.17.mlp.experts.down_proj.scales,<br>language_model.model.layers.17.mlp.experts.down_proj.weight,<br>language_model.model.layers.17.mlp.experts.gate_proj.biases,<br>language_model.model.layers.17.mlp.experts.gate_proj.scales,<br>language_model.model.layers.17.mlp.experts.gate_proj.weight,<br>language_model.model.layers.17.mlp.experts.up_proj.biases,<br>language_model.model.layers.17.mlp.experts.up_proj.scales,<br>language_model.model.layers.17.mlp.experts.up_proj.weight,<br>language_model.model.layers.18.mlp.experts.down_proj.biases,<br>language_model.model.layers.18.mlp.experts.down_proj.scales,<br>language_model.model.layers.18.mlp.experts.down_proj.weight,<br>language_model.model.layers.18.mlp.experts.gate_proj.biases,<br>language_model.model.layers.18.mlp.experts.gate_proj.scales,<br>language_model.model.layers.18.mlp.experts.gate_proj.weight,<br>language_model.model.layers.18.mlp.experts.up_proj.biases,<br>language_model.model.layers.18.mlp.experts.up_proj.scales,<br>language_model.model.layers.18.mlp.experts.up_proj.weight,<br>language_model.model.layers.19.mlp.experts.down_proj.biases,<br>language_model.model.layers.19.mlp.experts.down_proj.scales,<br>language_model.model.layers.19.mlp.experts.down_proj.weight,<br>language_model.model.layers.19.mlp.experts.gate_proj.biases,<br>language_model.model.layers.19.mlp.experts.gate_proj.scales,<br>language_model.model.layers.19.mlp.experts.gate_proj.weight,<br>language_model.model.layers.19.mlp.experts.up_proj.biases,<br>language_model.model.layers.19.mlp.experts.up_proj.scales,<br>language_model.model.layers.19.mlp.experts.up_proj.weight,<br>language_model.model.layers.2.mlp.experts.down_proj.biases,<br>language_model.model.layers.2.mlp.experts.down_proj.scales,<br>language_model.model.layers.2.mlp.experts.down_proj.weight,<br>language_model.model.layers.2.mlp.experts.gate_proj.biases,<br>language_model.model.layers.2.mlp.experts.gate_proj.scales,<br>language_model.model.layers.2.mlp.experts.gate_proj.weight,<br>language_model.model.layers.2.mlp.experts.up_proj.biases,<br>language_model.model.layers.2.mlp.experts.up_proj.scales,<br>language_model.model.layers.2.mlp.experts.up_proj.weight,<br>language_model.model.layers.20.mlp.experts.down_proj.biases,<br>language_model.model.layers.20.mlp.experts.down_proj.scales,<br>language_model.model.layers.20.mlp.experts.down_proj.weight,<br>language_model.model.layers.20.mlp.experts.gate_proj.biases,<br>language_model.model.layers.20.mlp.experts.gate_proj.scales,<br>language_model.model.layers.20.mlp.experts.gate_proj.weight,<br>language_model.model.layers.20.mlp.experts.up_proj.biases,<br>language_model.model.layers.20.mlp.experts.up_proj.scales,<br>language_model.model.layers.20.mlp.experts.up_proj.weight,<br>language_model.model.layers.21.mlp.experts.down_proj.biases,<br>language_model.model.layers.21.mlp.experts.down_proj.scales,<br>language_model.model.layers.21.mlp.experts.down_proj.weight,<br>language_model.model.layers.21.mlp.experts.gate_proj.biases,<br>language_model.model.layers.21.mlp.experts.gate_proj.scales,<br>language_model.model.layers.21.mlp.experts.gate_proj.weight,<br>language_model.model.layers.21.mlp.experts.up_proj.biases,<br>language_model.model.layers.21.mlp.experts.up_proj.scales,<br>language_model.model.layers.21.mlp.experts.up_proj.weight,<br>language_model.model.layers.22.mlp.experts.down_proj.biases,<br>language_model.model.layers.22.mlp.experts.down_proj.scales,<br>language_model.model.layers.22.mlp.experts.down_proj.weight,<br>language_model.model.layers.22.mlp.experts.gate_proj.biases,<br>language_model.model.layers.22.mlp.experts.gate_proj.scales,<br>language_model.model.layers.22.mlp.experts.gate_proj.weight,<br>language_model.model.layers.22.mlp.experts.up_proj.biases,<br>language_model.model.layers.22.mlp.experts.up_proj.scales,<br>language_model.model.layers.22.mlp.experts.up_proj.weight,<br>language_model.model.layers.23.mlp.experts.down_proj.biases,<br>language_model.model.layers.23.mlp.experts.down_proj.scales,<br>language_model.model.layers.23.mlp.experts.down_proj.weight,<br>language_model.model.layers.23.mlp.experts.gate_proj.biases,<br>language_model.model.layers.23.mlp.experts.gate_proj.scales,<br>language_model.model.layers.23.mlp.experts.gate_proj.weight,<br>language_model.model.layers.23.mlp.experts.up_proj.biases,<br>language_model.model.layers.23.mlp.experts.up_proj.scales,<br>language_model.model.layers.23.mlp.experts.up_proj.weight,<br>language_model.model.layers.24.mlp.experts.down_proj.biases,<br>language_model.model.layers.24.mlp.experts.down_proj.scales,<br>language_model.model.layers.24.mlp.experts.down_proj.weight,<br>language_model.model.layers.24.mlp.experts.gate_proj.biases,<br>language_model.model.layers.24.mlp.experts.gate_proj.scales,<br>language_model.model.layers.24.mlp.experts.gate_proj.weight,<br>language_model.model.layers.24.mlp.experts.up_proj.biases,<br>language_model.model.layers.24.mlp.experts.up_proj.scales,<br>language_model.model.layers.24.mlp.experts.up_proj.weight,<br>language_model.model.layers.25.mlp.experts.down_proj.biases,<br>language_model.model.layers.25.mlp.experts.down_proj.scales,<br>language_model.model.layers.25.mlp.experts.down_proj.weight,<br>language_model.model.layers.25.mlp.experts.gate_proj.biases,<br>language_model.model.layers.25.mlp.experts.gate_proj.scales,<br>language_model.model.layers.25.mlp.experts.gate_proj.weight,<br>language_model.model.layers.25.mlp.experts.up_proj.biases,<br>language_model.model.layers.25.mlp.experts.up_proj.scales,<br>language_model.model.layers.25.mlp.experts.up_proj.weight,<br>language_model.model.layers.26.mlp.experts.down_proj.biases,<br>language_model.model.layers.26.mlp.experts.down_proj.scales,<br>language_model.model.layers.26.mlp.experts.down_proj.weight,<br>language_model.model.layers.26.mlp.experts.gate_proj.biases,<br>language_model.model.layers.26.mlp.experts.gate_proj.scales,<br>language_model.model.layers.26.mlp.experts.gate_proj.weight,<br>language_model.model.layers.26.mlp.experts.up_proj.biases,<br>language_model.model.layers.26.mlp.experts.up_proj.scales,<br>language_model.model.layers.26.mlp.experts.up_proj.weight,<br>language_model.model.layers.27.mlp.experts.down_proj.biases,<br>language_model.model.layers.27.mlp.experts.down_proj.scales,<br>language_model.model.layers.27.mlp.experts.down_proj.weight,<br>language_model.model.layers.27.mlp.experts.gate_proj.biases,<br>language_model.model.layers.27.mlp.experts.gate_proj.scales,<br>language_model.model.layers.27.mlp.experts.gate_proj.weight,<br>language_model.model.layers.27.mlp.experts.up_proj.biases,<br>language_model.model.layers.27.mlp.experts.up_proj.scales,<br>language_model.model.layers.27.mlp.experts.up_proj.weight,<br>language_model.model.layers.28.mlp.experts.down_proj.biases,<br>language_model.model.layers.28.mlp.experts.down_proj.scales,<br>language_model.model.layers.28.mlp.experts.down_proj.weight,<br>language_model.model.layers.28.mlp.experts.gate_proj.biases,<br>language_model.model.layers.28.mlp.experts.gate_proj.scales,<br>language_model.model.layers.28.mlp.experts.gate_proj.weight,<br>language_model.model.layers.28.mlp.experts.up_proj.biases,<br>language_model.model.layers.28.mlp.experts.up_proj.scales,<br>language_model.model.layers.28.mlp.experts.up_proj.weight,<br>language_model.model.layers.29.mlp.experts.down_proj.biases,<br>language_model.model.layers.29.mlp.experts.down_proj.scales,<br>language_model.model.layers.29.mlp.experts.down_proj.weight,<br>language_model.model.layers.29.mlp.experts.gate_proj.biases,<br>language_model.model.layers.29.mlp.experts.gate_proj.scales,<br>language_model.model.layers.29.mlp.experts.gate_proj.weight,<br>language_model.model.layers.29.mlp.experts.up_proj.biases,<br>language_model.model.layers.29.mlp.experts.up_proj.scales,<br>language_model.model.layers.29.mlp.experts.up_proj.weight,<br>language_model.model.layers.3.mlp.experts.down_proj.biases,<br>language_model.model.layers.3.mlp.experts.down_proj.scales,<br>language_model.model.layers.3.mlp.experts.down_proj.weight,<br>language_model.model.layers.3.mlp.experts.gate_proj.biases,<br>language_model.model.layers.3.mlp.experts.gate_proj.scales,<br>language_model.model.layers.3.mlp.experts.gate_proj.weight,<br>language_model.model.layers.3.mlp.experts.up_proj.biases,<br>language_model.model.layers.3.mlp.experts.up_proj.scales,<br>language_model.model.layers.3.mlp.experts.up_proj.weight,<br>language_model.model.layers.30.mlp.experts.down_proj.biases,<br>language_model.model.layers.30.mlp.experts.down_proj.scales,<br>language_model.model.layers.30.mlp.experts.down_proj.weight,<br>language_model.model.layers.30.mlp.experts.gate_proj.biases,<br>language_model.model.layers.30.mlp.experts.gate_proj.scales,<br>language_model.model.layers.30.mlp.experts.gate_proj.weight,<br>language_model.model.layers.30.mlp.experts.up_proj.biases,<br>language_model.model.layers.30.mlp.experts.up_proj.scales,<br>language_model.model.layers.30.mlp.experts.up_proj.weight,<br>language_model.model.layers.31.mlp.experts.down_proj.biases,<br>language_model.model.layers.31.mlp.experts.down_proj.scales,<br>language_model.model.layers.31.mlp.experts.down_proj.weight,<br>language_model.model.layers.31.mlp.experts.gate_proj.biases,<br>language_model.model.layers.31.mlp.experts.gate_proj.scales,<br>language_model.model.layers.31.mlp.experts.gate_proj.weight,<br>language_model.model.layers.31.mlp.experts.up_proj.biases,<br>language_model.model.layers.31.mlp.experts.up_proj.scales,<br>language_model.model.layers.31.mlp.experts.up_proj.weight,<br>language_model.model.layers.32.mlp.experts.down_proj.biases,<br>language_model.model.layers.32.mlp.experts.down_proj.scales,<br>language_model.model.layers.32.mlp.experts.down_proj.weight,<br>language_model.model.layers.32.mlp.experts.gate_proj.biases,<br>language_model.model.layers.32.mlp.experts.gate_proj.scales,<br>language_model.model.layers.32.mlp.experts.gate_proj.weight,<br>language_model.model.layers.32.mlp.experts.up_proj.biases,<br>language_model.model.layers.32.mlp.experts.up_proj.scales,<br>language_model.model.layers.32.mlp.experts.up_proj.weight,<br>language_model.model.layers.33.mlp.experts.down_proj.biases,<br>language_model.model.layers.33.mlp.experts.down_proj.scales,<br>language_model.model.layers.33.mlp.experts.down_proj.weight,<br>language_model.model.layers.33.mlp.experts.gate_proj.biases,<br>language_model.model.layers.33.mlp.experts.gate_proj.scales,<br>language_model.model.layers.33.mlp.experts.gate_proj.weight,<br>language_model.model.layers.33.mlp.experts.up_proj.biases,<br>language_model.model.layers.33.mlp.experts.up_proj.scales,<br>language_model.model.layers.33.mlp.experts.up_proj.weight,<br>language_model.model.layers.34.mlp.experts.down_proj.biases,<br>language_model.model.layers.34.mlp.experts.down_proj.scales,<br>language_model.model.layers.34.mlp.experts.down_proj.weight,<br>language_model.model.layers.34.mlp.experts.gate_proj.biases,<br>language_model.model.layers.34.mlp.experts.gate_proj.scales,<br>language_model.model.layers.34.mlp.experts.gate_proj.weight,<br>language_model.model.layers.34.mlp.experts.up_proj.biases,<br>language_model.model.layers.34.mlp.experts.up_proj.scales,<br>language_model.model.layers.34.mlp.experts.up_proj.weight,<br>language_model.model.layers.35.mlp.experts.down_proj.biases,<br>language_model.model.layers.35.mlp.experts.down_proj.scales,<br>language_model.model.layers.35.mlp.experts.down_proj.weight,<br>language_model.model.layers.35.mlp.experts.gate_proj.biases,<br>language_model.model.layers.35.mlp.experts.gate_proj.scales,<br>language_model.model.layers.35.mlp.experts.gate_proj.weight,<br>language_model.model.layers.35.mlp.experts.up_proj.biases,<br>language_model.model.layers.35.mlp.experts.up_proj.scales,<br>language_model.model.layers.35.mlp.experts.up_proj.weight,<br>language_model.model.layers.36.mlp.experts.down_proj.biases,<br>language_model.model.layers.36.mlp.experts.down_proj.scales,<br>language_model.model.layers.36.mlp.experts.down_proj.weight,<br>language_model.model.layers.36.mlp.experts.gate_proj.biases,<br>language_model.model.layers.36.mlp.experts.gate_proj.scales,<br>language_model.model.layers.36.mlp.experts.gate_proj.weight,<br>language_model.model.layers.36.mlp.experts.up_proj.biases,<br>language_model.model.layers.36.mlp.experts.up_proj.scales,<br>language_model.model.layers.36.mlp.experts.up_proj.weight,<br>language_model.model.layers.37.mlp.experts.down_proj.biases,<br>language_model.model.layers.37.mlp.experts.down_proj.scales,<br>language_model.model.layers.37.mlp.experts.down_proj.weight,<br>language_model.model.layers.37.mlp.experts.gate_proj.biases,<br>language_model.model.layers.37.mlp.experts.gate_proj.scales,<br>language_model.model.layers.37.mlp.experts.gate_proj.weight,<br>language_model.model.layers.37.mlp.experts.up_proj.biases,<br>language_model.model.layers.37.mlp.experts.up_proj.scales,<br>language_model.model.layers.37.mlp.experts.up_proj.weight,<br>language_model.model.layers.38.mlp.experts.down_proj.biases,<br>language_model.model.layers.38.mlp.experts.down_proj.scales,<br>language_model.model.layers.38.mlp.experts.down_proj.weight,<br>language_model.model.layers.38.mlp.experts.gate_proj.biases,<br>language_model.model.layers.38.mlp.experts.gate_proj.scales,<br>language_model.model.layers.38.mlp.experts.gate_proj.weight,<br>language_model.model.layers.38.mlp.experts.up_proj.biases,<br>language_model.model.layers.38.mlp.experts.up_proj.scales,<br>language_model.model.layers.38.mlp.experts.up_proj.weight,<br>language_model.model.layers.39.mlp.experts.down_proj.biases,<br>language_model.model.layers.39.mlp.experts.down_proj.scales,<br>language_model.model.layers.39.mlp.experts.down_proj.weight,<br>language_model.model.layers.39.mlp.experts.gate_proj.biases,<br>language_model.model.layers.39.mlp.experts.gate_proj.scales,<br>language_model.model.layers.39.mlp.experts.gate_proj.weight,<br>language_model.model.layers.39.mlp.experts.up_proj.biases,<br>language_model.model.layers.39.mlp.experts.up_proj.scales,<br>language_model.model.layers.39.mlp.experts.up_proj.weight,<br>language_model.model.layers.4.mlp.experts.down_proj.biases,<br>language_model.model.layers.4.mlp.experts.down_proj.scales,<br>language_model.model.layers.4.mlp.experts.down_proj.weight,<br>language_model.model.layers.4.mlp.experts.gate_proj.biases,<br>language_model.model.layers.4.mlp.experts.gate_proj.scales,<br>language_model.model.layers.4.mlp.experts.gate_proj.weight,<br>language_model.model.layers.4.mlp.experts.up_proj.biases,<br>language_model.model.layers.4.mlp.experts.up_proj.scales,<br>language_model.model.layers.4.mlp.experts.up_proj.weight,<br>language_model.model.layers.40.mlp.experts.down_proj.biases,<br>language_model.model.layers.40.mlp.experts.down_proj.scales,<br>language_model.model.layers.40.mlp.experts.down_proj.weight,<br>language_model.model.layers.40.mlp.experts.gate_proj.biases,<br>language_model.model.layers.40.mlp.experts.gate_proj.scales,<br>language_model.model.layers.40.mlp.experts.gate_proj.weight,<br>language_model.model.layers.40.mlp.experts.up_proj.biases,<br>language_model.model.layers.40.mlp.experts.up_proj.scales,<br>language_model.model.layers.40.mlp.experts.up_proj.weight,<br>language_model.model.layers.41.mlp.experts.down_proj.biases,<br>language_model.model.layers.41.mlp.experts.down_proj.scales,<br>language_model.model.layers.41.mlp.experts.down_proj.weight,<br>language_model.model.layers.41.mlp.experts.gate_proj.biases,<br>language_model.model.layers.41.mlp.experts.gate_proj.scales,<br>language_model.model.layers.41.mlp.experts.gate_proj.weight,<br>language_model.model.layers.41.mlp.experts.up_proj.biases,<br>language_model.model.layers.41.mlp.experts.up_proj.scales,<br>language_model.model.layers.41.mlp.experts.up_proj.weight,<br>language_model.model.layers.5.mlp.experts.down_proj.biases,<br>language_model.model.layers.5.mlp.experts.down_proj.scales,<br>language_model.model.layers.5.mlp.experts.down_proj.weight,<br>language_model.model.layers.5.mlp.experts.gate_proj.biases,<br>language_model.model.layers.5.mlp.experts.gate_proj.scales,<br>language_model.model.layers.5.mlp.experts.gate_proj.weight,<br>language_model.model.layers.5.mlp.experts.up_proj.biases,<br>language_model.model.layers.5.mlp.experts.up_proj.scales,<br>language_model.model.layers.5.mlp.experts.up_proj.weight,<br>language_model.model.layers.6.mlp.experts.down_proj.biases,<br>language_model.model.layers.6.mlp.experts.down_proj.scales,<br>language_model.model.layers.6.mlp.experts.down_proj.weight,<br>language_model.model.layers.6.mlp.experts.gate_proj.biases,<br>language_model.model.layers.6.mlp.experts.gate_proj.scales,<br>language_model.model.layers.6.mlp.experts.gate_proj.weight,<br>language_model.model.layers.6.mlp.experts.up_proj.biases,<br>language_model.model.layers.6.mlp.experts.up_proj.scales,<br>language_model.model.layers.6.mlp.experts.up_proj.weight,<br>language_model.model.layers.7.mlp.experts.down_proj.biases,<br>language_model.model.layers.7.mlp.experts.down_proj.scales,<br>language_model.model.layers.7.mlp.experts.down_proj.weight,<br>language_model.model.layers.7.mlp.experts.gate_proj.biases,<br>language_model.model.layers.7.mlp.experts.gate_proj.scales,<br>language_model.model.layers.7.mlp.experts.gate_proj.weight,<br>language_model.model.layers.7.mlp.experts.up_proj.biases,<br>language_model.model.layers.7.mlp.experts.up_proj.scales,<br>language_model.model.layers.7.mlp.experts.up_proj.weight,<br>language_model.model.layers.8.mlp.experts.down_proj.biases,<br>language_model.model.layers.8.mlp.experts.down_proj.scales,<br>language_model.model.layers.8.mlp.experts.down_proj.weight,<br>language_model.model.layers.8.mlp.experts.gate_proj.biases,<br>language_model.model.layers.8.mlp.experts.gate_proj.scales,<br>language_model.model.layers.8.mlp.experts.gate_proj.weight,<br>language_model.model.layers.8.mlp.experts.up_proj.biases,<br>language_model.model.layers.8.mlp.experts.up_proj.scales,<br>language_model.model.layers.8.mlp.experts.up_proj.weight,<br>language_model.model.layers.9.mlp.experts.down_proj.biases,<br>language_model.model.layers.9.mlp.experts.down_proj.scales,<br>language_model.model.layers.9.mlp.experts.down_proj.weight,<br>language_model.model.layers.9.mlp.experts.gate_proj.biases,<br>language_model.model.layers.9.mlp.experts.gate_proj.scales,<br>language_model.model.layers.9.mlp.experts.gate_proj.weight,<br>language_model.model.layers.9.mlp.experts.up_proj.biases,<br>language_model.model.layers.9.mlp.experts.up_proj.scales,<br>language_model.model.layers.9.mlp.experts.up_proj.weight.
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* Received 362 parameters not in model:
  <br>audio_tower.encoder.biases,<br>audio_tower.encoder.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.biases,<br>language_model.model.layers.10.mlp.experts.down_proj.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.weight,<br>language_model.model.layers.10.mlp.experts.gate_proj.biases,<br>language_model.model.layers.10.mlp.experts.gate_proj.scales,<br>language_model.model.layers.10.mlp.experts.gate_proj.weight,<br>language_model.model.layers.10.mlp.experts.up_proj.biases,<br>language_model.model.layers.10.mlp.experts.up_proj.scales,<br>language_model.model.layers.10.mlp.experts.up_proj.weight,<br>language_model.model.layers.11.mlp.experts.down_proj.biases,<br>language_model.model.layers.11.mlp.experts.down_proj.scales,<br>language_model.model.layers.11.mlp.experts.down_proj.weight,<br>language_model.model.layers.11.mlp.experts.gate_proj.biases,<br>language_model.model.layers.11.mlp.experts.gate_proj.scales,<br>language_model.model.layers.11.mlp.experts.gate_proj.weight,<br>language_model.model.layers.11.mlp.experts.up_proj.biases,<br>language_model.model.layers.11.mlp.experts.up_proj.scales,<br>language_model.model.layers.11.mlp.experts.up_proj.weight,<br>language_model.model.layers.12.mlp.experts.down_proj.biases,<br>language_model.model.layers.12.mlp.experts.down_proj.scales,<br>language_model.model.layers.12.mlp.experts.down_proj.weight,<br>language_model.model.layers.12.mlp.experts.gate_proj.biases,<br>language_model.model.layers.12.mlp.experts.gate_proj.scales,<br>language_model.model.layers.12.mlp.experts.gate_proj.weight,<br>language_model.model.layers.12.mlp.experts.up_proj.biases,<br>language_model.model.layers.12.mlp.experts.up_proj.scales,<br>language_model.model.layers.12.mlp.experts.up_proj.weight,<br>language_model.model.layers.13.mlp.experts.down_proj.biases,<br>language_model.model.layers.13.mlp.experts.down_proj.scales,<br>language_model.model.layers.13.mlp.experts.down_proj.weight,<br>language_model.model.layers.13.mlp.experts.gate_proj.biases,<br>language_model.model.layers.13.mlp.experts.gate_proj.scales,<br>language_model.model.layers.13.mlp.experts.gate_proj.weight,<br>language_model.model.layers.13.mlp.experts.up_proj.biases,<br>language_model.model.layers.13.mlp.experts.up_proj.scales,<br>language_model.model.layers.13.mlp.experts.up_proj.weight,<br>language_model.model.layers.14.mlp.experts.down_proj.biases,<br>language_model.model.layers.14.mlp.experts.down_proj.scales,<br>language_model.model.layers.14.mlp.experts.down_proj.weight,<br>language_model.model.layers.14.mlp.experts.gate_proj.biases,<br>language_model.model.layers.14.mlp.experts.gate_proj.scales,<br>language_model.model.layers.14.mlp.experts.gate_proj.weight,<br>language_model.model.layers.14.mlp.experts.up_proj.biases,<br>language_model.model.layers.14.mlp.experts.up_proj.scales,<br>language_model.model.layers.14.mlp.experts.up_proj.weight,<br>language_model.model.layers.15.mlp.experts.down_proj.biases,<br>language_model.model.layers.15.mlp.experts.down_proj.scales,<br>language_model.model.layers.15.mlp.experts.down_proj.weight,<br>language_model.model.layers.15.mlp.experts.gate_proj.biases,<br>language_model.model.layers.15.mlp.experts.gate_proj.scales,<br>language_model.model.layers.15.mlp.experts.gate_proj.weight,<br>language_model.model.layers.15.mlp.experts.up_proj.biases,<br>language_model.model.layers.15.mlp.experts.up_proj.scales,<br>language_model.model.layers.15.mlp.experts.up_proj.weight,<br>language_model.model.layers.16.mlp.experts.down_proj.biases,<br>language_model.model.layers.16.mlp.experts.down_proj.scales,<br>language_model.model.layers.16.mlp.experts.down_proj.weight,<br>language_model.model.layers.16.mlp.experts.gate_proj.biases,<br>language_model.model.layers.16.mlp.experts.gate_proj.scales,<br>language_model.model.layers.16.mlp.experts.gate_proj.weight,<br>language_model.model.layers.16.mlp.experts.up_proj.biases,<br>language_model.model.layers.16.mlp.experts.up_proj.scales,<br>language_model.model.layers.16.mlp.experts.up_proj.weight,<br>language_model.model.layers.17.mlp.experts.down_proj.biases,<br>language_model.model.layers.17.mlp.experts.down_proj.scales,<br>language_model.model.layers.17.mlp.experts.down_proj.weight,<br>language_model.model.layers.17.mlp.experts.gate_proj.biases,<br>language_model.model.layers.17.mlp.experts.gate_proj.scales,<br>language_model.model.layers.17.mlp.experts.gate_proj.weight,<br>language_model.model.layers.17.mlp.experts.up_proj.biases,<br>language_model.model.layers.17.mlp.experts.up_proj.scales,<br>language_model.model.layers.17.mlp.experts.up_proj.weight,<br>language_model.model.layers.18.mlp.experts.down_proj.biases,<br>language_model.model.layers.18.mlp.experts.down_proj.scales,<br>language_model.model.layers.18.mlp.experts.down_proj.weight,<br>language_model.model.layers.18.mlp.experts.gate_proj.biases,<br>language_model.model.layers.18.mlp.experts.gate_proj.scales,<br>language_model.model.layers.18.mlp.experts.gate_proj.weight,<br>language_model.model.layers.18.mlp.experts.up_proj.biases,<br>language_model.model.layers.18.mlp.experts.up_proj.scales,<br>language_model.model.layers.18.mlp.experts.up_proj.weight,<br>language_model.model.layers.19.mlp.experts.down_proj.biases,<br>language_model.model.layers.19.mlp.experts.down_proj.scales,<br>language_model.model.layers.19.mlp.experts.down_proj.weight,<br>language_model.model.layers.19.mlp.experts.gate_proj.biases,<br>language_model.model.layers.19.mlp.experts.gate_proj.scales,<br>language_model.model.layers.19.mlp.experts.gate_proj.weight,<br>language_model.model.layers.19.mlp.experts.up_proj.biases,<br>language_model.model.layers.19.mlp.experts.up_proj.scales,<br>language_model.model.layers.19.mlp.experts.up_proj.weight,<br>language_model.model.layers.2.mlp.experts.down_proj.biases,<br>language_model.model.layers.2.mlp.experts.down_proj.scales,<br>language_model.model.layers.2.mlp.experts.down_proj.weight,<br>language_model.model.layers.2.mlp.experts.gate_proj.biases,<br>language_model.model.layers.2.mlp.experts.gate_proj.scales,<br>language_model.model.layers.2.mlp.experts.gate_proj.weight,<br>language_model.model.layers.2.mlp.experts.up_proj.biases,<br>language_model.model.layers.2.mlp.experts.up_proj.scales,<br>language_model.model.layers.2.mlp.experts.up_proj.weight,<br>language_model.model.layers.20.mlp.experts.down_proj.biases,<br>language_model.model.layers.20.mlp.experts.down_proj.scales,<br>language_model.model.layers.20.mlp.experts.down_proj.weight,<br>language_model.model.layers.20.mlp.experts.gate_proj.biases,<br>language_model.model.layers.20.mlp.experts.gate_proj.scales,<br>language_model.model.layers.20.mlp.experts.gate_proj.weight,<br>language_model.model.layers.20.mlp.experts.up_proj.biases,<br>language_model.model.layers.20.mlp.experts.up_proj.scales,<br>language_model.model.layers.20.mlp.experts.up_proj.weight,<br>language_model.model.layers.21.mlp.experts.down_proj.biases,<br>language_model.model.layers.21.mlp.experts.down_proj.scales,<br>language_model.model.layers.21.mlp.experts.down_proj.weight,<br>language_model.model.layers.21.mlp.experts.gate_proj.biases,<br>language_model.model.layers.21.mlp.experts.gate_proj.scales,<br>language_model.model.layers.21.mlp.experts.gate_proj.weight,<br>language_model.model.layers.21.mlp.experts.up_proj.biases,<br>language_model.model.layers.21.mlp.experts.up_proj.scales,<br>language_model.model.layers.21.mlp.experts.up_proj.weight,<br>language_model.model.layers.22.mlp.experts.down_proj.biases,<br>language_model.model.layers.22.mlp.experts.down_proj.scales,<br>language_model.model.layers.22.mlp.experts.down_proj.weight,<br>language_model.model.layers.22.mlp.experts.gate_proj.biases,<br>language_model.model.layers.22.mlp.experts.gate_proj.scales,<br>language_model.model.layers.22.mlp.experts.gate_proj.weight,<br>language_model.model.layers.22.mlp.experts.up_proj.biases,<br>language_model.model.layers.22.mlp.experts.up_proj.scales,<br>language_model.model.layers.22.mlp.experts.up_proj.weight,<br>language_model.model.layers.23.mlp.experts.down_proj.biases,<br>language_model.model.layers.23.mlp.experts.down_proj.scales,<br>language_model.model.layers.23.mlp.experts.down_proj.weight,<br>language_model.model.layers.23.mlp.experts.gate_proj.biases,<br>language_model.model.layers.23.mlp.experts.gate_proj.scales,<br>language_model.model.layers.23.mlp.experts.gate_proj.weight,<br>language_model.model.layers.23.mlp.experts.up_proj.biases,<br>language_model.model.layers.23.mlp.experts.up_proj.scales,<br>language_model.model.layers.23.mlp.experts.up_proj.weight,<br>language_model.model.layers.24.mlp.experts.down_proj.biases,<br>language_model.model.layers.24.mlp.experts.down_proj.scales,<br>language_model.model.layers.24.mlp.experts.down_proj.weight,<br>language_model.model.layers.24.mlp.experts.gate_proj.biases,<br>language_model.model.layers.24.mlp.experts.gate_proj.scales,<br>language_model.model.layers.24.mlp.experts.gate_proj.weight,<br>language_model.model.layers.24.mlp.experts.up_proj.biases,<br>language_model.model.layers.24.mlp.experts.up_proj.scales,<br>language_model.model.layers.24.mlp.experts.up_proj.weight,<br>language_model.model.layers.25.mlp.experts.down_proj.biases,<br>language_model.model.layers.25.mlp.experts.down_proj.scales,<br>language_model.model.layers.25.mlp.experts.down_proj.weight,<br>language_model.model.layers.25.mlp.experts.gate_proj.biases,<br>language_model.model.layers.25.mlp.experts.gate_proj.scales,<br>language_model.model.layers.25.mlp.experts.gate_proj.weight,<br>language_model.model.layers.25.mlp.experts.up_proj.biases,<br>language_model.model.layers.25.mlp.experts.up_proj.scales,<br>language_model.model.layers.25.mlp.experts.up_proj.weight,<br>language_model.model.layers.26.mlp.experts.down_proj.biases,<br>language_model.model.layers.26.mlp.experts.down_proj.scales,<br>language_model.model.layers.26.mlp.experts.down_proj.weight,<br>language_model.model.layers.26.mlp.experts.gate_proj.biases,<br>language_model.model.layers.26.mlp.experts.gate_proj.scales,<br>language_model.model.layers.26.mlp.experts.gate_proj.weight,<br>language_model.model.layers.26.mlp.experts.up_proj.biases,<br>language_model.model.layers.26.mlp.experts.up_proj.scales,<br>language_model.model.layers.26.mlp.experts.up_proj.weight,<br>language_model.model.layers.27.mlp.experts.down_proj.biases,<br>language_model.model.layers.27.mlp.experts.down_proj.scales,<br>language_model.model.layers.27.mlp.experts.down_proj.weight,<br>language_model.model.layers.27.mlp.experts.gate_proj.biases,<br>language_model.model.layers.27.mlp.experts.gate_proj.scales,<br>language_model.model.layers.27.mlp.experts.gate_proj.weight,<br>language_model.model.layers.27.mlp.experts.up_proj.biases,<br>language_model.model.layers.27.mlp.experts.up_proj.scales,<br>language_model.model.layers.27.mlp.experts.up_proj.weight,<br>language_model.model.layers.28.mlp.experts.down_proj.biases,<br>language_model.model.layers.28.mlp.experts.down_proj.scales,<br>language_model.model.layers.28.mlp.experts.down_proj.weight,<br>language_model.model.layers.28.mlp.experts.gate_proj.biases,<br>language_model.model.layers.28.mlp.experts.gate_proj.scales,<br>language_model.model.layers.28.mlp.experts.gate_proj.weight,<br>language_model.model.layers.28.mlp.experts.up_proj.biases,<br>language_model.model.layers.28.mlp.experts.up_proj.scales,<br>language_model.model.layers.28.mlp.experts.up_proj.weight,<br>language_model.model.layers.29.mlp.experts.down_proj.biases,<br>language_model.model.layers.29.mlp.experts.down_proj.scales,<br>language_model.model.layers.29.mlp.experts.down_proj.weight,<br>language_model.model.layers.29.mlp.experts.gate_proj.biases,<br>language_model.model.layers.29.mlp.experts.gate_proj.scales,<br>language_model.model.layers.29.mlp.experts.gate_proj.weight,<br>language_model.model.layers.29.mlp.experts.up_proj.biases,<br>language_model.model.layers.29.mlp.experts.up_proj.scales,<br>language_model.model.layers.29.mlp.experts.up_proj.weight,<br>language_model.model.layers.3.mlp.experts.down_proj.biases,<br>language_model.model.layers.3.mlp.experts.down_proj.scales,<br>language_model.model.layers.3.mlp.experts.down_proj.weight,<br>language_model.model.layers.3.mlp.experts.gate_proj.biases,<br>language_model.model.layers.3.mlp.experts.gate_proj.scales,<br>language_model.model.layers.3.mlp.experts.gate_proj.weight,<br>language_model.model.layers.3.mlp.experts.up_proj.biases,<br>language_model.model.layers.3.mlp.experts.up_proj.scales,<br>language_model.model.layers.3.mlp.experts.up_proj.weight,<br>language_model.model.layers.30.mlp.experts.down_proj.biases,<br>language_model.model.layers.30.mlp.experts.down_proj.scales,<br>language_model.model.layers.30.mlp.experts.down_proj.weight,<br>language_model.model.layers.30.mlp.experts.gate_proj.biases,<br>language_model.model.layers.30.mlp.experts.gate_proj.scales,<br>language_model.model.layers.30.mlp.experts.gate_proj.weight,<br>language_model.model.layers.30.mlp.experts.up_proj.biases,<br>language_model.model.layers.30.mlp.experts.up_proj.scales,<br>language_model.model.layers.30.mlp.experts.up_proj.weight,<br>language_model.model.layers.31.mlp.experts.down_proj.biases,<br>language_model.model.layers.31.mlp.experts.down_proj.scales,<br>language_model.model.layers.31.mlp.experts.down_proj.weight,<br>language_model.model.layers.31.mlp.experts.gate_proj.biases,<br>language_model.model.layers.31.mlp.experts.gate_proj.scales,<br>language_model.model.layers.31.mlp.experts.gate_proj.weight,<br>language_model.model.layers.31.mlp.experts.up_proj.biases,<br>language_model.model.layers.31.mlp.experts.up_proj.scales,<br>language_model.model.layers.31.mlp.experts.up_proj.weight,<br>language_model.model.layers.32.mlp.experts.down_proj.biases,<br>language_model.model.layers.32.mlp.experts.down_proj.scales,<br>language_model.model.layers.32.mlp.experts.down_proj.weight,<br>language_model.model.layers.32.mlp.experts.gate_proj.biases,<br>language_model.model.layers.32.mlp.experts.gate_proj.scales,<br>language_model.model.layers.32.mlp.experts.gate_proj.weight,<br>language_model.model.layers.32.mlp.experts.up_proj.biases,<br>language_model.model.layers.32.mlp.experts.up_proj.scales,<br>language_model.model.layers.32.mlp.experts.up_proj.weight,<br>language_model.model.layers.33.mlp.experts.down_proj.biases,<br>language_model.model.layers.33.mlp.experts.down_proj.scales,<br>language_model.model.layers.33.mlp.experts.down_proj.weight,<br>language_model.model.layers.33.mlp.experts.gate_proj.biases,<br>language_model.model.layers.33.mlp.experts.gate_proj.scales,<br>language_model.model.layers.33.mlp.experts.gate_proj.weight,<br>language_model.model.layers.33.mlp.experts.up_proj.biases,<br>language_model.model.layers.33.mlp.experts.up_proj.scales,<br>language_model.model.layers.33.mlp.experts.up_proj.weight,<br>language_model.model.layers.34.mlp.experts.down_proj.biases,<br>language_model.model.layers.34.mlp.experts.down_proj.scales,<br>language_model.model.layers.34.mlp.experts.down_proj.weight,<br>language_model.model.layers.34.mlp.experts.gate_proj.biases,<br>language_model.model.layers.34.mlp.experts.gate_proj.scales,<br>language_model.model.layers.34.mlp.experts.gate_proj.weight,<br>language_model.model.layers.34.mlp.experts.up_proj.biases,<br>language_model.model.layers.34.mlp.experts.up_proj.scales,<br>language_model.model.layers.34.mlp.experts.up_proj.weight,<br>language_model.model.layers.35.mlp.experts.down_proj.biases,<br>language_model.model.layers.35.mlp.experts.down_proj.scales,<br>language_model.model.layers.35.mlp.experts.down_proj.weight,<br>language_model.model.layers.35.mlp.experts.gate_proj.biases,<br>language_model.model.layers.35.mlp.experts.gate_proj.scales,<br>language_model.model.layers.35.mlp.experts.gate_proj.weight,<br>language_model.model.layers.35.mlp.experts.up_proj.biases,<br>language_model.model.layers.35.mlp.experts.up_proj.scales,<br>language_model.model.layers.35.mlp.experts.up_proj.weight,<br>language_model.model.layers.36.mlp.experts.down_proj.biases,<br>language_model.model.layers.36.mlp.experts.down_proj.scales,<br>language_model.model.layers.36.mlp.experts.down_proj.weight,<br>language_model.model.layers.36.mlp.experts.gate_proj.biases,<br>language_model.model.layers.36.mlp.experts.gate_proj.scales,<br>language_model.model.layers.36.mlp.experts.gate_proj.weight,<br>language_model.model.layers.36.mlp.experts.up_proj.biases,<br>language_model.model.layers.36.mlp.experts.up_proj.scales,<br>language_model.model.layers.36.mlp.experts.up_proj.weight,<br>language_model.model.layers.37.mlp.experts.down_proj.biases,<br>language_model.model.layers.37.mlp.experts.down_proj.scales,<br>language_model.model.layers.37.mlp.experts.down_proj.weight,<br>language_model.model.layers.37.mlp.experts.gate_proj.biases,<br>language_model.model.layers.37.mlp.experts.gate_proj.scales,<br>language_model.model.layers.37.mlp.experts.gate_proj.weight,<br>language_model.model.layers.37.mlp.experts.up_proj.biases,<br>language_model.model.layers.37.mlp.experts.up_proj.scales,<br>language_model.model.layers.37.mlp.experts.up_proj.weight,<br>language_model.model.layers.38.mlp.experts.down_proj.biases,<br>language_model.model.layers.38.mlp.experts.down_proj.scales,<br>language_model.model.layers.38.mlp.experts.down_proj.weight,<br>language_model.model.layers.38.mlp.experts.gate_proj.biases,<br>language_model.model.layers.38.mlp.experts.gate_proj.scales,<br>language_model.model.layers.38.mlp.experts.gate_proj.weight,<br>language_model.model.layers.38.mlp.experts.up_proj.biases,<br>language_model.model.layers.38.mlp.experts.up_proj.scales,<br>language_model.model.layers.38.mlp.experts.up_proj.weight,<br>language_model.model.layers.39.mlp.experts.down_proj.biases,<br>language_model.model.layers.39.mlp.experts.down_proj.scales,<br>language_model.model.layers.39.mlp.experts.down_proj.weight,<br>language_model.model.layers.39.mlp.experts.gate_proj.biases,<br>language_model.model.layers.39.mlp.experts.gate_proj.scales,<br>language_model.model.layers.39.mlp.experts.gate_proj.weight,<br>language_model.model.layers.39.mlp.experts.up_proj.biases,<br>language_model.model.layers.39.mlp.experts.up_proj.scales,<br>language_model.model.layers.39.mlp.experts.up_proj.weight,<br>language_model.model.layers.4.mlp.experts.down_proj.biases,<br>language_model.model.layers.4.mlp.experts.down_proj.scales,<br>language_model.model.layers.4.mlp.experts.down_proj.weight,<br>language_model.model.layers.4.mlp.experts.gate_proj.biases,<br>language_model.model.layers.4.mlp.experts.gate_proj.scales,<br>language_model.model.layers.4.mlp.experts.gate_proj.weight,<br>language_model.model.layers.4.mlp.experts.up_proj.biases,<br>language_model.model.layers.4.mlp.experts.up_proj.scales,<br>language_model.model.layers.4.mlp.experts.up_proj.weight,<br>language_model.model.layers.40.mlp.experts.down_proj.biases,<br>language_model.model.layers.40.mlp.experts.down_proj.scales,<br>language_model.model.layers.40.mlp.experts.down_proj.weight,<br>language_model.model.layers.40.mlp.experts.gate_proj.biases,<br>language_model.model.layers.40.mlp.experts.gate_proj.scales,<br>language_model.model.layers.40.mlp.experts.gate_proj.weight,<br>language_model.model.layers.40.mlp.experts.up_proj.biases,<br>language_model.model.layers.40.mlp.experts.up_proj.scales,<br>language_model.model.layers.40.mlp.experts.up_proj.weight,<br>language_model.model.layers.41.mlp.experts.down_proj.biases,<br>language_model.model.layers.41.mlp.experts.down_proj.scales,<br>language_model.model.layers.41.mlp.experts.down_proj.weight,<br>language_model.model.layers.41.mlp.experts.gate_proj.biases,<br>language_model.model.layers.41.mlp.experts.gate_proj.scales,<br>language_model.model.layers.41.mlp.experts.gate_proj.weight,<br>language_model.model.layers.41.mlp.experts.up_proj.biases,<br>language_model.model.layers.41.mlp.experts.up_proj.scales,<br>language_model.model.layers.41.mlp.experts.up_proj.weight,<br>language_model.model.layers.5.mlp.experts.down_proj.biases,<br>language_model.model.layers.5.mlp.experts.down_proj.scales,<br>language_model.model.layers.5.mlp.experts.down_proj.weight,<br>language_model.model.layers.5.mlp.experts.gate_proj.biases,<br>language_model.model.layers.5.mlp.experts.gate_proj.scales,<br>language_model.model.layers.5.mlp.experts.gate_proj.weight,<br>language_model.model.layers.5.mlp.experts.up_proj.biases,<br>language_model.model.layers.5.mlp.experts.up_proj.scales,<br>language_model.model.layers.5.mlp.experts.up_proj.weight,<br>language_model.model.layers.6.mlp.experts.down_proj.biases,<br>language_model.model.layers.6.mlp.experts.down_proj.scales,<br>language_model.model.layers.6.mlp.experts.down_proj.weight,<br>language_model.model.layers.6.mlp.experts.gate_proj.biases,<br>language_model.model.layers.6.mlp.experts.gate_proj.scales,<br>language_model.model.layers.6.mlp.experts.gate_proj.weight,<br>language_model.model.layers.6.mlp.experts.up_proj.biases,<br>language_model.model.layers.6.mlp.experts.up_proj.scales,<br>language_model.model.layers.6.mlp.experts.up_proj.weight,<br>language_model.model.layers.7.mlp.experts.down_proj.biases,<br>language_model.model.layers.7.mlp.experts.down_proj.scales,<br>language_model.model.layers.7.mlp.experts.down_proj.weight,<br>language_model.model.layers.7.mlp.experts.gate_proj.biases,<br>language_model.model.layers.7.mlp.experts.gate_proj.scales,<br>language_model.model.layers.7.mlp.experts.gate_proj.weight,<br>language_model.model.layers.7.mlp.experts.up_proj.biases,<br>language_model.model.layers.7.mlp.experts.up_proj.scales,<br>language_model.model.layers.7.mlp.experts.up_proj.weight,<br>language_model.model.layers.8.mlp.experts.down_proj.biases,<br>language_model.model.layers.8.mlp.experts.down_proj.scales,<br>language_model.model.layers.8.mlp.experts.down_proj.weight,<br>language_model.model.layers.8.mlp.experts.gate_proj.biases,<br>language_model.model.layers.8.mlp.experts.gate_proj.scales,<br>language_model.model.layers.8.mlp.experts.gate_proj.weight,<br>language_model.model.layers.8.mlp.experts.up_proj.biases,<br>language_model.model.layers.8.mlp.experts.up_proj.scales,<br>language_model.model.layers.8.mlp.experts.up_proj.weight,<br>language_model.model.layers.9.mlp.experts.down_proj.biases,<br>language_model.model.layers.9.mlp.experts.down_proj.scales,<br>language_model.model.layers.9.mlp.experts.down_proj.weight,<br>language_model.model.layers.9.mlp.experts.gate_proj.biases,<br>language_model.model.layers.9.mlp.experts.gate_proj.scales,<br>language_model.model.layers.9.mlp.experts.gate_proj.weight,<br>language_model.model.layers.9.mlp.experts.up_proj.biases,<br>language_model.model.layers.9.mlp.experts.up_proj.scales,<br>language_model.model.layers.9.mlp.experts.up_proj.weight.
- *Arch supported by installed mlx-vlm:* yes (model_type inkling_mm_model via
  inkling)
- *Model load time:* 0.19s
- *Generation time:* -
- *Total time:* 0.20s
- *Input validation time:* 4.56e-05
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0637
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.00269
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* not captured
- *Image placeholders:* not captured
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-4bit/snapshots/f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612
- *Generation settings:* not captured
- *EOS token:* not captured

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12108, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11587, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 786, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 966, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 842, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 362 parameters not in model: 
audio_tower.encoder.biases,
audio_tower.encoder.scales,
language_model.model.layers.10.mlp.experts.down_proj.biases,
language_model.model.layers.10.mlp.experts.down_proj.scales,
language_model.model.layers.10.mlp.experts.down_proj.weight,
language_model.model.layers.10.mlp.experts.gate_proj.biases,
language_model.model.layers.10.mlp.experts.gate_proj.scales,
language_model.model.layers.10.mlp.experts.gate_proj.weight,
language_model.model.layers.10.mlp.experts.up_proj.biases,
language_model.model.layers.10.mlp.experts.up_proj.scales,
language_model.model.layers.10.mlp.experts.up_proj.weight,
language_model.model.layers.11.mlp.experts.down_proj.biases,
language_model.model.layers.11.mlp.experts.down_proj.scales,
language_model.model.layers.11.mlp.experts.down_proj.weight,
language_model.model.layers.11.mlp.experts.gate_proj.biases,
language_model.model.layers.11.mlp.experts.gate_proj.scales,
language_model.model.layers.11.mlp.experts.gate_proj.weight,
language_model.model.layers.11.mlp.experts.up_proj.biases,
language_model.model.layers.11.mlp.experts.up_proj.scales,
language_model.model.layers.11.mlp.experts.up_proj.weight,
language_model.model.layers.12.mlp.experts.down_proj.biases,
language_model.model.layers.12.mlp.experts.down_proj.scales,
language_model.model.layers.12.mlp.experts.down_proj.weight,
language_model.model.layers.12.mlp.experts.gate_proj.biases,
language_model.model.layers.12.mlp.experts.gate_proj.scales,
language_model.model.layers.12.mlp.experts.gate_proj.weight,
language_model.model.layers.12.mlp.experts.up_proj.biases,
language_model.model.layers.12.mlp.experts.up_proj.scales,
language_model.model.layers.12.mlp.experts.up_proj.weight,
language_model.model.layers.13.mlp.experts.down_proj.biases,
language_model.model.layers.13.mlp.experts.down_proj.scales,
language_model.model.layers.13.mlp.experts.down_proj.weight,
language_model.model.layers.13.mlp.experts.gate_proj.biases,
language_model.model.layers.13.mlp.experts.gate_proj.scales,
language_model.model.layers.13.mlp.experts.gate_proj.weight,
language_model.model.layers.13.mlp.experts.up_proj.biases,
language_model.model.layers.13.mlp.experts.up_proj.scales,
language_model.model.layers.13.mlp.experts.up_proj.weight,
language_model.model.layers.14.mlp.experts.down_proj.biases,
language_model.model.layers.14.mlp.experts.down_proj.scales,
language_model.model.layers.14.mlp.experts.down_proj.weight,
language_model.model.layers.14.mlp.experts.gate_proj.biases,
language_model.model.layers.14.mlp.experts.gate_proj.scales,
language_model.model.layers.14.mlp.experts.gate_proj.weight,
language_model.model.layers.14.mlp.experts.up_proj.biases,
language_model.model.layers.14.mlp.experts.up_proj.scales,
language_model.model.layers.14.mlp.experts.up_proj.weight,
language_model.model.layers.15.mlp.experts.down_proj.biases,
language_model.model.layers.15.mlp.experts.down_proj.scales,
language_model.model.layers.15.mlp.experts.down_proj.weight,
language_model.model.layers.15.mlp.experts.gate_proj.biases,
language_model.model.layers.15.mlp.experts.gate_proj.scales,
language_model.model.layers.15.mlp.experts.gate_proj.weight,
language_model.model.layers.15.mlp.experts.up_proj.biases,
language_model.model.layers.15.mlp.experts.up_proj.scales,
language_model.model.layers.15.mlp.experts.up_proj.weight,
language_model.model.layers.16.mlp.experts.down_proj.biases,
language_model.model.layers.16.mlp.experts.down_proj.scales,
language_model.model.layers.16.mlp.experts.down_proj.weight,
language_model.model.layers.16.mlp.experts.gate_proj.biases,
language_model.model.layers.16.mlp.experts.gate_proj.scales,
language_model.model.layers.16.mlp.experts.gate_proj.weight,
language_model.model.layers.16.mlp.experts.up_proj.biases,
language_model.model.layers.16.mlp.experts.up_proj.scales,
language_model.model.layers.16.mlp.experts.up_proj.weight,
language_model.model.layers.17.mlp.experts.down_proj.biases,
language_model.model.layers.17.mlp.experts.down_proj.scales,
language_model.model.layers.17.mlp.experts.down_proj.weight,
language_model.model.layers.17.mlp.experts.gate_proj.biases,
language_model.model.layers.17.mlp.experts.gate_proj.scales,
language_model.model.layers.17.mlp.experts.gate_proj.weight,
language_model.model.layers.17.mlp.experts.up_proj.biases,
language_model.model.layers.17.mlp.experts.up_proj.scales,
language_model.model.layers.17.mlp.experts.up_proj.weight,
language_model.model.layers.18.mlp.experts.down_proj.biases,
language_model.model.layers.18.mlp.experts.down_proj.scales,
language_model.model.layers.18.mlp.experts.down_proj.weight,
language_model.model.layers.18.mlp.experts.gate_proj.biases,
language_model.model.layers.18.mlp.experts.gate_proj.scales,
language_model.model.layers.18.mlp.experts.gate_proj.weight,
language_model.model.layers.18.mlp.experts.up_proj.biases,
language_model.model.layers.18.mlp.experts.up_proj.scales,
language_model.model.layers.18.mlp.experts.up_proj.weight,
language_model.model.layers.19.mlp.experts.down_proj.biases,
language_model.model.layers.19.mlp.experts.down_proj.scales,
language_model.model.layers.19.mlp.experts.down_proj.weight,
language_model.model.layers.19.mlp.experts.gate_proj.biases,
language_model.model.layers.19.mlp.experts.gate_proj.scales,
language_model.model.layers.19.mlp.experts.gate_proj.weight,
language_model.model.layers.19.mlp.experts.up_proj.biases,
language_model.model.layers.19.mlp.experts.up_proj.scales,
language_model.model.layers.19.mlp.experts.up_proj.weight,
language_model.model.layers.2.mlp.experts.down_proj.biases,
language_model.model.layers.2.mlp.experts.down_proj.scales,
language_model.model.layers.2.mlp.experts.down_proj.weight,
language_model.model.layers.2.mlp.experts.gate_proj.biases,
language_model.model.layers.2.mlp.experts.gate_proj.scales,
language_model.model.layers.2.mlp.experts.gate_proj.weight,
language_model.model.layers.2.mlp.experts.up_proj.biases,
language_model.model.layers.2.mlp.experts.up_proj.scales,
language_model.model.layers.2.mlp.experts.up_proj.weight,
language_model.model.layers.20.mlp.experts.down_proj.biases,
language_model.model.layers.20.mlp.experts.down_proj.scales,
language_model.model.layers.20.mlp.experts.down_proj.weight,
language_model.model.layers.20.mlp.experts.gate_proj.biases,
language_model.model.layers.20.mlp.experts.gate_proj.scales,
language_model.model.layers.20.mlp.experts.gate_proj.weight,
language_model.model.layers.20.mlp.experts.up_proj.biases,
language_model.model.layers.20.mlp.experts.up_proj.scales,
language_model.model.layers.20.mlp.experts.up_proj.weight,
language_model.model.layers.21.mlp.experts.down_proj.biases,
language_model.model.layers.21.mlp.experts.down_proj.scales,
language_model.model.layers.21.mlp.experts.down_proj.weight,
language_model.model.layers.21.mlp.experts.gate_proj.biases,
language_model.model.layers.21.mlp.experts.gate_proj.scales,
language_model.model.layers.21.mlp.experts.gate_proj.weight,
language_model.model.layers.21.mlp.experts.up_proj.biases,
language_model.model.layers.21.mlp.experts.up_proj.scales,
language_model.model.layers.21.mlp.experts.up_proj.weight,
language_model.model.layers.22.mlp.experts.down_proj.biases,
language_model.model.layers.22.mlp.experts.down_proj.scales,
language_model.model.layers.22.mlp.experts.down_proj.weight,
language_model.model.layers.22.mlp.experts.gate_proj.biases,
language_model.model.layers.22.mlp.experts.gate_proj.scales,
language_model.model.layers.22.mlp.experts.gate_proj.weight,
language_model.model.layers.22.mlp.experts.up_proj.biases,
language_model.model.layers.22.mlp.experts.up_proj.scales,
language_model.model.layers.22.mlp.experts.up_proj.weight,
language_model.model.layers.23.mlp.experts.down_proj.biases,
language_model.model.layers.23.mlp.experts.down_proj.scales,
language_model.model.layers.23.mlp.experts.down_proj.weight,
language_model.model.layers.23.mlp.experts.gate_proj.biases,
language_model.model.layers.23.mlp.experts.gate_proj.scales,
language_model.model.layers.23.mlp.experts.gate_proj.weight,
language_model.model.layers.23.mlp.experts.up_proj.biases,
language_model.model.layers.23.mlp.experts.up_proj.scales,
language_model.model.layers.23.mlp.experts.up_proj.weight,
language_model.model.layers.24.mlp.experts.down_proj.biases,
language_model.model.layers.24.mlp.experts.down_proj.scales,
language_model.model.layers.24.mlp.experts.down_proj.weight,
language_model.model.layers.24.mlp.experts.gate_proj.biases,
language_model.model.layers.24.mlp.experts.gate_proj.scales,
language_model.model.layers.24.mlp.experts.gate_proj.weight,
language_model.model.layers.24.mlp.experts.up_proj.biases,
language_model.model.layers.24.mlp.experts.up_proj.scales,
language_model.model.layers.24.mlp.experts.up_proj.weight,
language_model.model.layers.25.mlp.experts.down_proj.biases,
language_model.model.layers.25.mlp.experts.down_proj.scales,
language_model.model.layers.25.mlp.experts.down_proj.weight,
language_model.model.layers.25.mlp.experts.gate_proj.biases,
language_model.model.layers.25.mlp.experts.gate_proj.scales,
language_model.model.layers.25.mlp.experts.gate_proj.weight,
language_model.model.layers.25.mlp.experts.up_proj.biases,
language_model.model.layers.25.mlp.experts.up_proj.scales,
language_model.model.layers.25.mlp.experts.up_proj.weight,
language_model.model.layers.26.mlp.experts.down_proj.biases,
language_model.model.layers.26.mlp.experts.down_proj.scales,
language_model.model.layers.26.mlp.experts.down_proj.weight,
language_model.model.layers.26.mlp.experts.gate_proj.biases,
language_model.model.layers.26.mlp.experts.gate_proj.scales,
language_model.model.layers.26.mlp.experts.gate_proj.weight,
language_model.model.layers.26.mlp.experts.up_proj.biases,
language_model.model.layers.26.mlp.experts.up_proj.scales,
language_model.model.layers.26.mlp.experts.up_proj.weight,
language_model.model.layers.27.mlp.experts.down_proj.biases,
language_model.model.layers.27.mlp.experts.down_proj.scales,
language_model.model.layers.27.mlp.experts.down_proj.weight,
language_model.model.layers.27.mlp.experts.gate_proj.biases,
language_model.model.layers.27.mlp.experts.gate_proj.scales,
language_model.model.layers.27.mlp.experts.gate_proj.weight,
language_model.model.layers.27.mlp.experts.up_proj.biases,
language_model.model.layers.27.mlp.experts.up_proj.scales,
language_model.model.layers.27.mlp.experts.up_proj.weight,
language_model.model.layers.28.mlp.experts.down_proj.biases,
language_model.model.layers.28.mlp.experts.down_proj.scales,
language_model.model.layers.28.mlp.experts.down_proj.weight,
language_model.model.layers.28.mlp.experts.gate_proj.biases,
language_model.model.layers.28.mlp.experts.gate_proj.scales,
language_model.model.layers.28.mlp.experts.gate_proj.weight,
language_model.model.layers.28.mlp.experts.up_proj.biases,
language_model.model.layers.28.mlp.experts.up_proj.scales,
language_model.model.layers.28.mlp.experts.up_proj.weight,
language_model.model.layers.29.mlp.experts.down_proj.biases,
language_model.model.layers.29.mlp.experts.down_proj.scales,
language_model.model.layers.29.mlp.experts.down_proj.weight,
language_model.model.layers.29.mlp.experts.gate_proj.biases,
language_model.model.layers.29.mlp.experts.gate_proj.scales,
language_model.model.layers.29.mlp.experts.gate_proj.weight,
language_model.model.layers.29.mlp.experts.up_proj.biases,
language_model.model.layers.29.mlp.experts.up_proj.scales,
language_model.model.layers.29.mlp.experts.up_proj.weight,
language_model.model.layers.3.mlp.experts.down_proj.biases,
language_model.model.layers.3.mlp.experts.down_proj.scales,
language_model.model.layers.3.mlp.experts.down_proj.weight,
language_model.model.layers.3.mlp.experts.gate_proj.biases,
language_model.model.layers.3.mlp.experts.gate_proj.scales,
language_model.model.layers.3.mlp.experts.gate_proj.weight,
language_model.model.layers.3.mlp.experts.up_proj.biases,
language_model.model.layers.3.mlp.experts.up_proj.scales,
language_model.model.layers.3.mlp.experts.up_proj.weight,
language_model.model.layers.30.mlp.experts.down_proj.biases,
language_model.model.layers.30.mlp.experts.down_proj.scales,
language_model.model.layers.30.mlp.experts.down_proj.weight,
language_model.model.layers.30.mlp.experts.gate_proj.biases,
language_model.model.layers.30.mlp.experts.gate_proj.scales,
language_model.model.layers.30.mlp.experts.gate_proj.weight,
language_model.model.layers.30.mlp.experts.up_proj.biases,
language_model.model.layers.30.mlp.experts.up_proj.scales,
language_model.model.layers.30.mlp.experts.up_proj.weight,
language_model.model.layers.31.mlp.experts.down_proj.biases,
language_model.model.layers.31.mlp.experts.down_proj.scales,
language_model.model.layers.31.mlp.experts.down_proj.weight,
language_model.model.layers.31.mlp.experts.gate_proj.biases,
language_model.model.layers.31.mlp.experts.gate_proj.scales,
language_model.model.layers.31.mlp.experts.gate_proj.weight,
language_model.model.layers.31.mlp.experts.up_proj.biases,
language_model.model.layers.31.mlp.experts.up_proj.scales,
language_model.model.layers.31.mlp.experts.up_proj.weight,
language_model.model.layers.32.mlp.experts.down_proj.biases,
language_model.model.layers.32.mlp.experts.down_proj.scales,
language_model.model.layers.32.mlp.experts.down_proj.weight,
language_model.model.layers.32.mlp.experts.gate_proj.biases,
language_model.model.layers.32.mlp.experts.gate_proj.scales,
language_model.model.layers.32.mlp.experts.gate_proj.weight,
language_model.model.layers.32.mlp.experts.up_proj.biases,
language_model.model.layers.32.mlp.experts.up_proj.scales,
language_model.model.layers.32.mlp.experts.up_proj.weight,
language_model.model.layers.33.mlp.experts.down_proj.biases,
language_model.model.layers.33.mlp.experts.down_proj.scales,
language_model.model.layers.33.mlp.experts.down_proj.weight,
language_model.model.layers.33.mlp.experts.gate_proj.biases,
language_model.model.layers.33.mlp.experts.gate_proj.scales,
language_model.model.layers.33.mlp.experts.gate_proj.weight,
language_model.model.layers.33.mlp.experts.up_proj.biases,
language_model.model.layers.33.mlp.experts.up_proj.scales,
language_model.model.layers.33.mlp.experts.up_proj.weight,
language_model.model.layers.34.mlp.experts.down_proj.biases,
language_model.model.layers.34.mlp.experts.down_proj.scales,
language_model.model.layers.34.mlp.experts.down_proj.weight,
language_model.model.layers.34.mlp.experts.gate_proj.biases,
language_model.model.layers.34.mlp.experts.gate_proj.scales,
language_model.model.layers.34.mlp.experts.gate_proj.weight,
language_model.model.layers.34.mlp.experts.up_proj.biases,
language_model.model.layers.34.mlp.experts.up_proj.scales,
language_model.model.layers.34.mlp.experts.up_proj.weight,
language_model.model.layers.35.mlp.experts.down_proj.biases,
language_model.model.layers.35.mlp.experts.down_proj.scales,
language_model.model.layers.35.mlp.experts.down_proj.weight,
language_model.model.layers.35.mlp.experts.gate_proj.biases,
language_model.model.layers.35.mlp.experts.gate_proj.scales,
language_model.model.layers.35.mlp.experts.gate_proj.weight,
language_model.model.layers.35.mlp.experts.up_proj.biases,
language_model.model.layers.35.mlp.experts.up_proj.scales,
language_model.model.layers.35.mlp.experts.up_proj.weight,
language_model.model.layers.36.mlp.experts.down_proj.biases,
language_model.model.layers.36.mlp.experts.down_proj.scales,
language_model.model.layers.36.mlp.experts.down_proj.weight,
language_model.model.layers.36.mlp.experts.gate_proj.biases,
language_model.model.layers.36.mlp.experts.gate_proj.scales,
language_model.model.layers.36.mlp.experts.gate_proj.weight,
language_model.model.layers.36.mlp.experts.up_proj.biases,
language_model.model.layers.36.mlp.experts.up_proj.scales,
language_model.model.layers.36.mlp.experts.up_proj.weight,
language_model.model.layers.37.mlp.experts.down_proj.biases,
language_model.model.layers.37.mlp.experts.down_proj.scales,
language_model.model.layers.37.mlp.experts.down_proj.weight,
language_model.model.layers.37.mlp.experts.gate_proj.biases,
language_model.model.layers.37.mlp.experts.gate_proj.scales,
language_model.model.layers.37.mlp.experts.gate_proj.weight,
language_model.model.layers.37.mlp.experts.up_proj.biases,
language_model.model.layers.37.mlp.experts.up_proj.scales,
language_model.model.layers.37.mlp.experts.up_proj.weight,
language_model.model.layers.38.mlp.experts.down_proj.biases,
language_model.model.layers.38.mlp.experts.down_proj.scales,
language_model.model.layers.38.mlp.experts.down_proj.weight,
language_model.model.layers.38.mlp.experts.gate_proj.biases,
language_model.model.layers.38.mlp.experts.gate_proj.scales,
language_model.model.layers.38.mlp.experts.gate_proj.weight,
language_model.model.layers.38.mlp.experts.up_proj.biases,
language_model.model.layers.38.mlp.experts.up_proj.scales,
language_model.model.layers.38.mlp.experts.up_proj.weight,
language_model.model.layers.39.mlp.experts.down_proj.biases,
language_model.model.layers.39.mlp.experts.down_proj.scales,
language_model.model.layers.39.mlp.experts.down_proj.weight,
language_model.model.layers.39.mlp.experts.gate_proj.biases,
language_model.model.layers.39.mlp.experts.gate_proj.scales,
language_model.model.layers.39.mlp.experts.gate_proj.weight,
language_model.model.layers.39.mlp.experts.up_proj.biases,
language_model.model.layers.39.mlp.experts.up_proj.scales,
language_model.model.layers.39.mlp.experts.up_proj.weight,
language_model.model.layers.4.mlp.experts.down_proj.biases,
language_model.model.layers.4.mlp.experts.down_proj.scales,
language_model.model.layers.4.mlp.experts.down_proj.weight,
language_model.model.layers.4.mlp.experts.gate_proj.biases,
language_model.model.layers.4.mlp.experts.gate_proj.scales,
language_model.model.layers.4.mlp.experts.gate_proj.weight,
language_model.model.layers.4.mlp.experts.up_proj.biases,
language_model.model.layers.4.mlp.experts.up_proj.scales,
language_model.model.layers.4.mlp.experts.up_proj.weight,
language_model.model.layers.40.mlp.experts.down_proj.biases,
language_model.model.layers.40.mlp.experts.down_proj.scales,
language_model.model.layers.40.mlp.experts.down_proj.weight,
language_model.model.layers.40.mlp.experts.gate_proj.biases,
language_model.model.layers.40.mlp.experts.gate_proj.scales,
language_model.model.layers.40.mlp.experts.gate_proj.weight,
language_model.model.layers.40.mlp.experts.up_proj.biases,
language_model.model.layers.40.mlp.experts.up_proj.scales,
language_model.model.layers.40.mlp.experts.up_proj.weight,
language_model.model.layers.41.mlp.experts.down_proj.biases,
language_model.model.layers.41.mlp.experts.down_proj.scales,
language_model.model.layers.41.mlp.experts.down_proj.weight,
language_model.model.layers.41.mlp.experts.gate_proj.biases,
language_model.model.layers.41.mlp.experts.gate_proj.scales,
language_model.model.layers.41.mlp.experts.gate_proj.weight,
language_model.model.layers.41.mlp.experts.up_proj.biases,
language_model.model.layers.41.mlp.experts.up_proj.scales,
language_model.model.layers.41.mlp.experts.up_proj.weight,
language_model.model.layers.5.mlp.experts.down_proj.biases,
language_model.model.layers.5.mlp.experts.down_proj.scales,
language_model.model.layers.5.mlp.experts.down_proj.weight,
language_model.model.layers.5.mlp.experts.gate_proj.biases,
language_model.model.layers.5.mlp.experts.gate_proj.scales,
language_model.model.layers.5.mlp.experts.gate_proj.weight,
language_model.model.layers.5.mlp.experts.up_proj.biases,
language_model.model.layers.5.mlp.experts.up_proj.scales,
language_model.model.layers.5.mlp.experts.up_proj.weight,
language_model.model.layers.6.mlp.experts.down_proj.biases,
language_model.model.layers.6.mlp.experts.down_proj.scales,
language_model.model.layers.6.mlp.experts.down_proj.weight,
language_model.model.layers.6.mlp.experts.gate_proj.biases,
language_model.model.layers.6.mlp.experts.gate_proj.scales,
language_model.model.layers.6.mlp.experts.gate_proj.weight,
language_model.model.layers.6.mlp.experts.up_proj.biases,
language_model.model.layers.6.mlp.experts.up_proj.scales,
language_model.model.layers.6.mlp.experts.up_proj.weight,
language_model.model.layers.7.mlp.experts.down_proj.biases,
language_model.model.layers.7.mlp.experts.down_proj.scales,
language_model.model.layers.7.mlp.experts.down_proj.weight,
language_model.model.layers.7.mlp.experts.gate_proj.biases,
language_model.model.layers.7.mlp.experts.gate_proj.scales,
language_model.model.layers.7.mlp.experts.gate_proj.weight,
language_model.model.layers.7.mlp.experts.up_proj.biases,
language_model.model.layers.7.mlp.experts.up_proj.scales,
language_model.model.layers.7.mlp.experts.up_proj.weight,
language_model.model.layers.8.mlp.experts.down_proj.biases,
language_model.model.layers.8.mlp.experts.down_proj.scales,
language_model.model.layers.8.mlp.experts.down_proj.weight,
language_model.model.layers.8.mlp.experts.gate_proj.biases,
language_model.model.layers.8.mlp.experts.gate_proj.scales,
language_model.model.layers.8.mlp.experts.gate_proj.weight,
language_model.model.layers.8.mlp.experts.up_proj.biases,
language_model.model.layers.8.mlp.experts.up_proj.scales,
language_model.model.layers.8.mlp.experts.up_proj.weight,
language_model.model.layers.9.mlp.experts.down_proj.biases,
language_model.model.layers.9.mlp.experts.down_proj.scales,
language_model.model.layers.9.mlp.experts.down_proj.weight,
language_model.model.layers.9.mlp.experts.gate_proj.biases,
language_model.model.layers.9.mlp.experts.gate_proj.scales,
language_model.model.layers.9.mlp.experts.gate_proj.weight,
language_model.model.layers.9.mlp.experts.up_proj.biases,
language_model.model.layers.9.mlp.experts.up_proj.scales,
language_model.model.layers.9.mlp.experts.up_proj.weight.

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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12123, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 362 parameters not in model: 
audio_tower.encoder.biases,
audio_tower.encoder.scales,
language_model.model.layers.10.mlp.experts.down_proj.biases,
language_model.model.layers.10.mlp.experts.down_proj.scales,
language_model.model.layers.10.mlp.experts.down_proj.weight,
language_model.model.layers.10.mlp.experts.gate_proj.biases,
language_model.model.layers.10.mlp.experts.gate_proj.scales,
language_model.model.layers.10.mlp.experts.gate_proj.weight,
language_model.model.layers.10.mlp.experts.up_proj.biases,
language_model.model.layers.10.mlp.experts.up_proj.scales,
language_model.model.layers.10.mlp.experts.up_proj.weight,
language_model.model.layers.11.mlp.experts.down_proj.biases,
language_model.model.layers.11.mlp.experts.down_proj.scales,
language_model.model.layers.11.mlp.experts.down_proj.weight,
language_model.model.layers.11.mlp.experts.gate_proj.biases,
language_model.model.layers.11.mlp.experts.gate_proj.scales,
language_model.model.layers.11.mlp.experts.gate_proj.weight,
language_model.model.layers.11.mlp.experts.up_proj.biases,
language_model.model.layers.11.mlp.experts.up_proj.scales,
language_model.model.layers.11.mlp.experts.up_proj.weight,
language_model.model.layers.12.mlp.experts.down_proj.biases,
language_model.model.layers.12.mlp.experts.down_proj.scales,
language_model.model.layers.12.mlp.experts.down_proj.weight,
language_model.model.layers.12.mlp.experts.gate_proj.biases,
language_model.model.layers.12.mlp.experts.gate_proj.scales,
language_model.model.layers.12.mlp.experts.gate_proj.weight,
language_model.model.layers.12.mlp.experts.up_proj.biases,
language_model.model.layers.12.mlp.experts.up_proj.scales,
language_model.model.layers.12.mlp.experts.up_proj.weight,
language_model.model.layers.13.mlp.experts.down_proj.biases,
language_model.model.layers.13.mlp.experts.down_proj.scales,
language_model.model.layers.13.mlp.experts.down_proj.weight,
language_model.model.layers.13.mlp.experts.gate_proj.biases,
language_model.model.layers.13.mlp.experts.gate_proj.scales,
language_model.model.layers.13.mlp.experts.gate_proj.weight,
language_model.model.layers.13.mlp.experts.up_proj.biases,
language_model.model.layers.13.mlp.experts.up_proj.scales,
language_model.model.layers.13.mlp.experts.up_proj.weight,
language_model.model.layers.14.mlp.experts.down_proj.biases,
language_model.model.layers.14.mlp.experts.down_proj.scales,
language_model.model.layers.14.mlp.experts.down_proj.weight,
language_model.model.layers.14.mlp.experts.gate_proj.biases,
language_model.model.layers.14.mlp.experts.gate_proj.scales,
language_model.model.layers.14.mlp.experts.gate_proj.weight,
language_model.model.layers.14.mlp.experts.up_proj.biases,
language_model.model.layers.14.mlp.experts.up_proj.scales,
language_model.model.layers.14.mlp.experts.up_proj.weight,
language_model.model.layers.15.mlp.experts.down_proj.biases,
language_model.model.layers.15.mlp.experts.down_proj.scales,
language_model.model.layers.15.mlp.experts.down_proj.weight,
language_model.model.layers.15.mlp.experts.gate_proj.biases,
language_model.model.layers.15.mlp.experts.gate_proj.scales,
language_model.model.layers.15.mlp.experts.gate_proj.weight,
language_model.model.layers.15.mlp.experts.up_proj.biases,
language_model.model.layers.15.mlp.experts.up_proj.scales,
language_model.model.layers.15.mlp.experts.up_proj.weight,
language_model.model.layers.16.mlp.experts.down_proj.biases,
language_model.model.layers.16.mlp.experts.down_proj.scales,
language_model.model.layers.16.mlp.experts.down_proj.weight,
language_model.model.layers.16.mlp.experts.gate_proj.biases,
language_model.model.layers.16.mlp.experts.gate_proj.scales,
language_model.model.layers.16.mlp.experts.gate_proj.weight,
language_model.model.layers.16.mlp.experts.up_proj.biases,
language_model.model.layers.16.mlp.experts.up_proj.scales,
language_model.model.layers.16.mlp.experts.up_proj.weight,
language_model.model.layers.17.mlp.experts.down_proj.biases,
language_model.model.layers.17.mlp.experts.down_proj.scales,
language_model.model.layers.17.mlp.experts.down_proj.weight,
language_model.model.layers.17.mlp.experts.gate_proj.biases,
language_model.model.layers.17.mlp.experts.gate_proj.scales,
language_model.model.layers.17.mlp.experts.gate_proj.weight,
language_model.model.layers.17.mlp.experts.up_proj.biases,
language_model.model.layers.17.mlp.experts.up_proj.scales,
language_model.model.layers.17.mlp.experts.up_proj.weight,
language_model.model.layers.18.mlp.experts.down_proj.biases,
language_model.model.layers.18.mlp.experts.down_proj.scales,
language_model.model.layers.18.mlp.experts.down_proj.weight,
language_model.model.layers.18.mlp.experts.gate_proj.biases,
language_model.model.layers.18.mlp.experts.gate_proj.scales,
language_model.model.layers.18.mlp.experts.gate_proj.weight,
language_model.model.layers.18.mlp.experts.up_proj.biases,
language_model.model.layers.18.mlp.experts.up_proj.scales,
language_model.model.layers.18.mlp.experts.up_proj.weight,
language_model.model.layers.19.mlp.experts.down_proj.biases,
language_model.model.layers.19.mlp.experts.down_proj.scales,
language_model.model.layers.19.mlp.experts.down_proj.weight,
language_model.model.layers.19.mlp.experts.gate_proj.biases,
language_model.model.layers.19.mlp.experts.gate_proj.scales,
language_model.model.layers.19.mlp.experts.gate_proj.weight,
language_model.model.layers.19.mlp.experts.up_proj.biases,
language_model.model.layers.19.mlp.experts.up_proj.scales,
language_model.model.layers.19.mlp.experts.up_proj.weight,
language_model.model.layers.2.mlp.experts.down_proj.biases,
language_model.model.layers.2.mlp.experts.down_proj.scales,
language_model.model.layers.2.mlp.experts.down_proj.weight,
language_model.model.layers.2.mlp.experts.gate_proj.biases,
language_model.model.layers.2.mlp.experts.gate_proj.scales,
language_model.model.layers.2.mlp.experts.gate_proj.weight,
language_model.model.layers.2.mlp.experts.up_proj.biases,
language_model.model.layers.2.mlp.experts.up_proj.scales,
language_model.model.layers.2.mlp.experts.up_proj.weight,
language_model.model.layers.20.mlp.experts.down_proj.biases,
language_model.model.layers.20.mlp.experts.down_proj.scales,
language_model.model.layers.20.mlp.experts.down_proj.weight,
language_model.model.layers.20.mlp.experts.gate_proj.biases,
language_model.model.layers.20.mlp.experts.gate_proj.scales,
language_model.model.layers.20.mlp.experts.gate_proj.weight,
language_model.model.layers.20.mlp.experts.up_proj.biases,
language_model.model.layers.20.mlp.experts.up_proj.scales,
language_model.model.layers.20.mlp.experts.up_proj.weight,
language_model.model.layers.21.mlp.experts.down_proj.biases,
language_model.model.layers.21.mlp.experts.down_proj.scales,
language_model.model.layers.21.mlp.experts.down_proj.weight,
language_model.model.layers.21.mlp.experts.gate_proj.biases,
language_model.model.layers.21.mlp.experts.gate_proj.scales,
language_model.model.layers.21.mlp.experts.gate_proj.weight,
language_model.model.layers.21.mlp.experts.up_proj.biases,
language_model.model.layers.21.mlp.experts.up_proj.scales,
language_model.model.layers.21.mlp.experts.up_proj.weight,
language_model.model.layers.22.mlp.experts.down_proj.biases,
language_model.model.layers.22.mlp.experts.down_proj.scales,
language_model.model.layers.22.mlp.experts.down_proj.weight,
language_model.model.layers.22.mlp.experts.gate_proj.biases,
language_model.model.layers.22.mlp.experts.gate_proj.scales,
language_model.model.layers.22.mlp.experts.gate_proj.weight,
language_model.model.layers.22.mlp.experts.up_proj.biases,
language_model.model.layers.22.mlp.experts.up_proj.scales,
language_model.model.layers.22.mlp.experts.up_proj.weight,
language_model.model.layers.23.mlp.experts.down_proj.biases,
language_model.model.layers.23.mlp.experts.down_proj.scales,
language_model.model.layers.23.mlp.experts.down_proj.weight,
language_model.model.layers.23.mlp.experts.gate_proj.biases,
language_model.model.layers.23.mlp.experts.gate_proj.scales,
language_model.model.layers.23.mlp.experts.gate_proj.weight,
language_model.model.layers.23.mlp.experts.up_proj.biases,
language_model.model.layers.23.mlp.experts.up_proj.scales,
language_model.model.layers.23.mlp.experts.up_proj.weight,
language_model.model.layers.24.mlp.experts.down_proj.biases,
language_model.model.layers.24.mlp.experts.down_proj.scales,
language_model.model.layers.24.mlp.experts.down_proj.weight,
language_model.model.layers.24.mlp.experts.gate_proj.biases,
language_model.model.layers.24.mlp.experts.gate_proj.scales,
language_model.model.layers.24.mlp.experts.gate_proj.weight,
language_model.model.layers.24.mlp.experts.up_proj.biases,
language_model.model.layers.24.mlp.experts.up_proj.scales,
language_model.model.layers.24.mlp.experts.up_proj.weight,
language_model.model.layers.25.mlp.experts.down_proj.biases,
language_model.model.layers.25.mlp.experts.down_proj.scales,
language_model.model.layers.25.mlp.experts.down_proj.weight,
language_model.model.layers.25.mlp.experts.gate_proj.biases,
language_model.model.layers.25.mlp.experts.gate_proj.scales,
language_model.model.layers.25.mlp.experts.gate_proj.weight,
language_model.model.layers.25.mlp.experts.up_proj.biases,
language_model.model.layers.25.mlp.experts.up_proj.scales,
language_model.model.layers.25.mlp.experts.up_proj.weight,
language_model.model.layers.26.mlp.experts.down_proj.biases,
language_model.model.layers.26.mlp.experts.down_proj.scales,
language_model.model.layers.26.mlp.experts.down_proj.weight,
language_model.model.layers.26.mlp.experts.gate_proj.biases,
language_model.model.layers.26.mlp.experts.gate_proj.scales,
language_model.model.layers.26.mlp.experts.gate_proj.weight,
language_model.model.layers.26.mlp.experts.up_proj.biases,
language_model.model.layers.26.mlp.experts.up_proj.scales,
language_model.model.layers.26.mlp.experts.up_proj.weight,
language_model.model.layers.27.mlp.experts.down_proj.biases,
language_model.model.layers.27.mlp.experts.down_proj.scales,
language_model.model.layers.27.mlp.experts.down_proj.weight,
language_model.model.layers.27.mlp.experts.gate_proj.biases,
language_model.model.layers.27.mlp.experts.gate_proj.scales,
language_model.model.layers.27.mlp.experts.gate_proj.weight,
language_model.model.layers.27.mlp.experts.up_proj.biases,
language_model.model.layers.27.mlp.experts.up_proj.scales,
language_model.model.layers.27.mlp.experts.up_proj.weight,
language_model.model.layers.28.mlp.experts.down_proj.biases,
language_model.model.layers.28.mlp.experts.down_proj.scales,
language_model.model.layers.28.mlp.experts.down_proj.weight,
language_model.model.layers.28.mlp.experts.gate_proj.biases,
language_model.model.layers.28.mlp.experts.gate_proj.scales,
language_model.model.layers.28.mlp.experts.gate_proj.weight,
language_model.model.layers.28.mlp.experts.up_proj.biases,
language_model.model.layers.28.mlp.experts.up_proj.scales,
language_model.model.layers.28.mlp.experts.up_proj.weight,
language_model.model.layers.29.mlp.experts.down_proj.biases,
language_model.model.layers.29.mlp.experts.down_proj.scales,
language_model.model.layers.29.mlp.experts.down_proj.weight,
language_model.model.layers.29.mlp.experts.gate_proj.biases,
language_model.model.layers.29.mlp.experts.gate_proj.scales,
language_model.model.layers.29.mlp.experts.gate_proj.weight,
language_model.model.layers.29.mlp.experts.up_proj.biases,
language_model.model.layers.29.mlp.experts.up_proj.scales,
language_model.model.layers.29.mlp.experts.up_proj.weight,
language_model.model.layers.3.mlp.experts.down_proj.biases,
language_model.model.layers.3.mlp.experts.down_proj.scales,
language_model.model.layers.3.mlp.experts.down_proj.weight,
language_model.model.layers.3.mlp.experts.gate_proj.biases,
language_model.model.layers.3.mlp.experts.gate_proj.scales,
language_model.model.layers.3.mlp.experts.gate_proj.weight,
language_model.model.layers.3.mlp.experts.up_proj.biases,
language_model.model.layers.3.mlp.experts.up_proj.scales,
language_model.model.layers.3.mlp.experts.up_proj.weight,
language_model.model.layers.30.mlp.experts.down_proj.biases,
language_model.model.layers.30.mlp.experts.down_proj.scales,
language_model.model.layers.30.mlp.experts.down_proj.weight,
language_model.model.layers.30.mlp.experts.gate_proj.biases,
language_model.model.layers.30.mlp.experts.gate_proj.scales,
language_model.model.layers.30.mlp.experts.gate_proj.weight,
language_model.model.layers.30.mlp.experts.up_proj.biases,
language_model.model.layers.30.mlp.experts.up_proj.scales,
language_model.model.layers.30.mlp.experts.up_proj.weight,
language_model.model.layers.31.mlp.experts.down_proj.biases,
language_model.model.layers.31.mlp.experts.down_proj.scales,
language_model.model.layers.31.mlp.experts.down_proj.weight,
language_model.model.layers.31.mlp.experts.gate_proj.biases,
language_model.model.layers.31.mlp.experts.gate_proj.scales,
language_model.model.layers.31.mlp.experts.gate_proj.weight,
language_model.model.layers.31.mlp.experts.up_proj.biases,
language_model.model.layers.31.mlp.experts.up_proj.scales,
language_model.model.layers.31.mlp.experts.up_proj.weight,
language_model.model.layers.32.mlp.experts.down_proj.biases,
language_model.model.layers.32.mlp.experts.down_proj.scales,
language_model.model.layers.32.mlp.experts.down_proj.weight,
language_model.model.layers.32.mlp.experts.gate_proj.biases,
language_model.model.layers.32.mlp.experts.gate_proj.scales,
language_model.model.layers.32.mlp.experts.gate_proj.weight,
language_model.model.layers.32.mlp.experts.up_proj.biases,
language_model.model.layers.32.mlp.experts.up_proj.scales,
language_model.model.layers.32.mlp.experts.up_proj.weight,
language_model.model.layers.33.mlp.experts.down_proj.biases,
language_model.model.layers.33.mlp.experts.down_proj.scales,
language_model.model.layers.33.mlp.experts.down_proj.weight,
language_model.model.layers.33.mlp.experts.gate_proj.biases,
language_model.model.layers.33.mlp.experts.gate_proj.scales,
language_model.model.layers.33.mlp.experts.gate_proj.weight,
language_model.model.layers.33.mlp.experts.up_proj.biases,
language_model.model.layers.33.mlp.experts.up_proj.scales,
language_model.model.layers.33.mlp.experts.up_proj.weight,
language_model.model.layers.34.mlp.experts.down_proj.biases,
language_model.model.layers.34.mlp.experts.down_proj.scales,
language_model.model.layers.34.mlp.experts.down_proj.weight,
language_model.model.layers.34.mlp.experts.gate_proj.biases,
language_model.model.layers.34.mlp.experts.gate_proj.scales,
language_model.model.layers.34.mlp.experts.gate_proj.weight,
language_model.model.layers.34.mlp.experts.up_proj.biases,
language_model.model.layers.34.mlp.experts.up_proj.scales,
language_model.model.layers.34.mlp.experts.up_proj.weight,
language_model.model.layers.35.mlp.experts.down_proj.biases,
language_model.model.layers.35.mlp.experts.down_proj.scales,
language_model.model.layers.35.mlp.experts.down_proj.weight,
language_model.model.layers.35.mlp.experts.gate_proj.biases,
language_model.model.layers.35.mlp.experts.gate_proj.scales,
language_model.model.layers.35.mlp.experts.gate_proj.weight,
language_model.model.layers.35.mlp.experts.up_proj.biases,
language_model.model.layers.35.mlp.experts.up_proj.scales,
language_model.model.layers.35.mlp.experts.up_proj.weight,
language_model.model.layers.36.mlp.experts.down_proj.biases,
language_model.model.layers.36.mlp.experts.down_proj.scales,
language_model.model.layers.36.mlp.experts.down_proj.weight,
language_model.model.layers.36.mlp.experts.gate_proj.biases,
language_model.model.layers.36.mlp.experts.gate_proj.scales,
language_model.model.layers.36.mlp.experts.gate_proj.weight,
language_model.model.layers.36.mlp.experts.up_proj.biases,
language_model.model.layers.36.mlp.experts.up_proj.scales,
language_model.model.layers.36.mlp.experts.up_proj.weight,
language_model.model.layers.37.mlp.experts.down_proj.biases,
language_model.model.layers.37.mlp.experts.down_proj.scales,
language_model.model.layers.37.mlp.experts.down_proj.weight,
language_model.model.layers.37.mlp.experts.gate_proj.biases,
language_model.model.layers.37.mlp.experts.gate_proj.scales,
language_model.model.layers.37.mlp.experts.gate_proj.weight,
language_model.model.layers.37.mlp.experts.up_proj.biases,
language_model.model.layers.37.mlp.experts.up_proj.scales,
language_model.model.layers.37.mlp.experts.up_proj.weight,
language_model.model.layers.38.mlp.experts.down_proj.biases,
language_model.model.layers.38.mlp.experts.down_proj.scales,
language_model.model.layers.38.mlp.experts.down_proj.weight,
language_model.model.layers.38.mlp.experts.gate_proj.biases,
language_model.model.layers.38.mlp.experts.gate_proj.scales,
language_model.model.layers.38.mlp.experts.gate_proj.weight,
language_model.model.layers.38.mlp.experts.up_proj.biases,
language_model.model.layers.38.mlp.experts.up_proj.scales,
language_model.model.layers.38.mlp.experts.up_proj.weight,
language_model.model.layers.39.mlp.experts.down_proj.biases,
language_model.model.layers.39.mlp.experts.down_proj.scales,
language_model.model.layers.39.mlp.experts.down_proj.weight,
language_model.model.layers.39.mlp.experts.gate_proj.biases,
language_model.model.layers.39.mlp.experts.gate_proj.scales,
language_model.model.layers.39.mlp.experts.gate_proj.weight,
language_model.model.layers.39.mlp.experts.up_proj.biases,
language_model.model.layers.39.mlp.experts.up_proj.scales,
language_model.model.layers.39.mlp.experts.up_proj.weight,
language_model.model.layers.4.mlp.experts.down_proj.biases,
language_model.model.layers.4.mlp.experts.down_proj.scales,
language_model.model.layers.4.mlp.experts.down_proj.weight,
language_model.model.layers.4.mlp.experts.gate_proj.biases,
language_model.model.layers.4.mlp.experts.gate_proj.scales,
language_model.model.layers.4.mlp.experts.gate_proj.weight,
language_model.model.layers.4.mlp.experts.up_proj.biases,
language_model.model.layers.4.mlp.experts.up_proj.scales,
language_model.model.layers.4.mlp.experts.up_proj.weight,
language_model.model.layers.40.mlp.experts.down_proj.biases,
language_model.model.layers.40.mlp.experts.down_proj.scales,
language_model.model.layers.40.mlp.experts.down_proj.weight,
language_model.model.layers.40.mlp.experts.gate_proj.biases,
language_model.model.layers.40.mlp.experts.gate_proj.scales,
language_model.model.layers.40.mlp.experts.gate_proj.weight,
language_model.model.layers.40.mlp.experts.up_proj.biases,
language_model.model.layers.40.mlp.experts.up_proj.scales,
language_model.model.layers.40.mlp.experts.up_proj.weight,
language_model.model.layers.41.mlp.experts.down_proj.biases,
language_model.model.layers.41.mlp.experts.down_proj.scales,
language_model.model.layers.41.mlp.experts.down_proj.weight,
language_model.model.layers.41.mlp.experts.gate_proj.biases,
language_model.model.layers.41.mlp.experts.gate_proj.scales,
language_model.model.layers.41.mlp.experts.gate_proj.weight,
language_model.model.layers.41.mlp.experts.up_proj.biases,
language_model.model.layers.41.mlp.experts.up_proj.scales,
language_model.model.layers.41.mlp.experts.up_proj.weight,
language_model.model.layers.5.mlp.experts.down_proj.biases,
language_model.model.layers.5.mlp.experts.down_proj.scales,
language_model.model.layers.5.mlp.experts.down_proj.weight,
language_model.model.layers.5.mlp.experts.gate_proj.biases,
language_model.model.layers.5.mlp.experts.gate_proj.scales,
language_model.model.layers.5.mlp.experts.gate_proj.weight,
language_model.model.layers.5.mlp.experts.up_proj.biases,
language_model.model.layers.5.mlp.experts.up_proj.scales,
language_model.model.layers.5.mlp.experts.up_proj.weight,
language_model.model.layers.6.mlp.experts.down_proj.biases,
language_model.model.layers.6.mlp.experts.down_proj.scales,
language_model.model.layers.6.mlp.experts.down_proj.weight,
language_model.model.layers.6.mlp.experts.gate_proj.biases,
language_model.model.layers.6.mlp.experts.gate_proj.scales,
language_model.model.layers.6.mlp.experts.gate_proj.weight,
language_model.model.layers.6.mlp.experts.up_proj.biases,
language_model.model.layers.6.mlp.experts.up_proj.scales,
language_model.model.layers.6.mlp.experts.up_proj.weight,
language_model.model.layers.7.mlp.experts.down_proj.biases,
language_model.model.layers.7.mlp.experts.down_proj.scales,
language_model.model.layers.7.mlp.experts.down_proj.weight,
language_model.model.layers.7.mlp.experts.gate_proj.biases,
language_model.model.layers.7.mlp.experts.gate_proj.scales,
language_model.model.layers.7.mlp.experts.gate_proj.weight,
language_model.model.layers.7.mlp.experts.up_proj.biases,
language_model.model.layers.7.mlp.experts.up_proj.scales,
language_model.model.layers.7.mlp.experts.up_proj.weight,
language_model.model.layers.8.mlp.experts.down_proj.biases,
language_model.model.layers.8.mlp.experts.down_proj.scales,
language_model.model.layers.8.mlp.experts.down_proj.weight,
language_model.model.layers.8.mlp.experts.gate_proj.biases,
language_model.model.layers.8.mlp.experts.gate_proj.scales,
language_model.model.layers.8.mlp.experts.gate_proj.weight,
language_model.model.layers.8.mlp.experts.up_proj.biases,
language_model.model.layers.8.mlp.experts.up_proj.scales,
language_model.model.layers.8.mlp.experts.up_proj.weight,
language_model.model.layers.9.mlp.experts.down_proj.biases,
language_model.model.layers.9.mlp.experts.down_proj.scales,
language_model.model.layers.9.mlp.experts.down_proj.weight,
language_model.model.layers.9.mlp.experts.gate_proj.biases,
language_model.model.layers.9.mlp.experts.gate_proj.scales,
language_model.model.layers.9.mlp.experts.gate_proj.weight,
language_model.model.layers.9.mlp.experts.up_proj.biases,
language_model.model.layers.9.mlp.experts.up_proj.scales,
language_model.model.layers.9.mlp.experts.up_proj.weight.

```

#### Captured upstream output

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 54 files:   0%|          | 0/54 [00:00<?, ?it/s]
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 4705.94it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[23:55:42] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
```

</details>

---
