# Model Output Gallery

Generated on: 2026-08-18 19:44:23 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                           |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|------------|---------|--------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | 1.56s   | 483 tok/s  | 0.11            | 1.9     | 2,072      | 132     | none                                                                                                   |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | 11.08s  | 29.3 tok/s | 3.72            | 24      | 2,658      | 131     | none                                                                                                   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`              | 10.06s  | 32.3 tok/s | 1.12            | 18      | 2,587      | 202     | none                                                                                                   |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | 2.33s   | 187 tok/s  | 0.37            | 4.0     | 2,072      | 140     | none                                                                                                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 6.83s   | 67.0 tok/s | 2.34            | 14      | 3,191      | 162     | none                                                                                                   |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 3.33s   | 189 tok/s  | 1.13            | 9.0     | 3,190      | 123     | none                                                                                                   |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 63.24s  | 110 tok/s  | 58.86           | 24      | 16,482     | 95      | none                                                                                                   |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 64.36s  | 93.5 tok/s | 60.77           | 10.0    | 16,482     | 113     | none                                                                                                   |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable`              | 87.78s  | 29.5 tok/s | 80.34           | 22      | 16,482     | 127     | none                                                                                                   |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable`              | 2.77s   | 126 tok/s  | 0.38            | 5.5     | 1,400      | 89      | none                                                                                                   |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | 24.64s  | 44.8 tok/s | 15.46           | 70      | 3,468      | 114     | none                                                                                                   |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.15s   | 129 tok/s  | 0.40            | 16      | 580        | 98      | none                                                                                                   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 8.76s   | 25.1 tok/s | 1.21            | 20      | 580        | 111     | none                                                                                                   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | 6.60s   | 39.5 tok/s | 1.81            | 16      | 3,429      | 99      | none                                                                                                   |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | 23.70s  | 43.3 tok/s | 14.61           | 78      | 6,311      | 96      | control tokens visible; title/keyword constraints failed                                               |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable_with_caveats` | 6.20s   | 34.3 tok/s | 1.38            | 17      | 3,623      | 85      | title/keyword constraints failed                                                                       |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | 135.74s | 4.68 tok/s | 1.31            | 40      | 1,257      | 610     | role tokens visible                                                                                    |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable_with_caveats` | 2.16s   | 272 tok/s  | 0.26            | 3.8     | 964        | 73      | title/keyword constraints failed                                                                       |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | 6.94s   | 64.5 tok/s | 2.25            | 15      | 3,191      | 174     | title/keyword constraints failed                                                                       |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | 5.13s   | 52.9 tok/s | 0.65            | 11      | 1,495      | 122     | title/keyword constraints failed                                                                       |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable_with_caveats` | 64.19s  | 63.1 tok/s | 53.00           | 74      | 16,482     | 111     | title/keyword constraints failed                                                                       |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable_with_caveats` | 3.23s   | 56.7 tok/s | 0.23            | 9.6     | 1,094      | 81      | title/keyword constraints failed                                                                       |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable_with_caveats` | 29.24s  | 90.6 tok/s | 17.53           | 8.4     | 16,469     | 903     | title/keyword constraints failed                                                                       |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable_with_caveats` | 89.55s  | 17.5 tok/s | 78.70           | 35      | 16,482     | 117     | title/keyword constraints failed                                                                       |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable_with_caveats` | 25.10s  | 54.8 tok/s | 18.63           | 13      | 16,478     | 241     | title/keyword constraints failed                                                                       |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | 5.99s   | 54.6 tok/s | 0.39            | 29      | 576        | 82      | control tokens visible; title/keyword constraints failed                                               |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | 5.69s   | 56.6 tok/s | 0.33            | 28      | 576        | 79      | control tokens visible                                                                                 |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | 8.90s   | 28.3 tok/s | 1.11            | 17      | 569        | 139     | title/keyword constraints failed                                                                       |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `unusable`            | 25.93s  | 93.1 tok/s | 13.73           | 8.4     | 16,467     | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 29.52s  | 41.4 tok/s | 2.19            | 15      | 3,520      | 1,000   | echoes instructions; extra text before Title; cut off at token limit; title/keyword constraints failed |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 24.43s  | 59.1 tok/s | 1.60            | 60      | 1,584      | 1,000   | cut off at token limit; title/keyword constraints failed                                               |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 2.27s   | 347 tok/s  | 0.12            | 2.0     | 308        | 147     | missing required fields; echoes instructions; extra text before Title                                  |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 28.47s  | 47.8 tok/s | 5.50            | 13      | 6,307      | 994     | missing required fields; extra text before Title                                                       |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `unusable`            | 19.95s  | 78.2 tok/s | 5.24            | 8.4     | 6,311      | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `unusable`            | 51.71s  | 20.7 tok/s | 1.43            | 15      | 275        | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | 29.27s  | 6.01 tok/s | 2.43            | 24      | 3,119      | 142     | missing required fields                                                                                |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `unusable`            | 68.07s  | 223 tok/s  | 62.16           | 5.1     | 16,478     | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `unusable`            | 28.94s  | 91.0 tok/s | 16.19           | 8.4     | 16,467     | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 6.64s   | 48.5 tok/s | 0.38            | 17      | 568        | 162     | missing required fields                                                                                |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | 4.32s   | 64.4 tok/s | 2.32            | 9.7     | 2,692      | 18      | missing required fields                                                                                |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 1.67s   | 369 tok/s  | 0.07            | 2.4     | 304        | 151     | missing required fields; echoes instructions                                                           |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 23.95s  | 48.0 tok/s | 1.13            | 4.4     | 4,397      | 1,000   | repeated text; missing required fields; echoes instructions; cut off at token limit                    |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 483 tok/s

Average clean-completion throughput: 113 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.9 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Usability  | Observations                                                                                           |
|--------------------------------------------------------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------------|
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                              | `unusable` | repeated text; cut off at token limit; title/keyword constraints failed                                |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable` | echoes instructions; extra text before Title; cut off at token limit; title/keyword constraints failed |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable` | cut off at token limit; title/keyword constraints failed                                               |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable` | missing required fields; echoes instructions; extra text before Title                                  |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable` | missing required fields; extra text before Title                                                       |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                             | `unusable` | repeated text; cut off at token limit; title/keyword constraints failed                                |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit) | `unusable` | repeated text; cut off at token limit; title/keyword constraints failed                                |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                | `unusable` | missing required fields                                                                                |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                  | `unusable` | repeated text; cut off at token limit; title/keyword constraints failed                                |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                  | `unusable` | repeated text; cut off at token limit; title/keyword constraints failed                                |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                            | `unusable` | missing required fields                                                                                |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable` | missing required fields                                                                                |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable` | missing required fields; echoes instructions                                                           |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable` | repeated text; missing required fields; echoes instructions; cut off at token limit                    |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                                     |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | Title:<br>Seafront, Felixstowe, England, UK, GBR, Europe<br><br>Description:<br>A tall, white column stands prominently on a coastal promenade, overlooking the sea. The column is adorned with a statue of an eagle perched atop, and a string of colorful lights runs along its base. The scene...                               |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | Title:<br>War Memorial by the Sea<br><br>Description:<br>A tall stone war memorial column topped with a winged figure stands on a seafront promenade in Felixstowe, England, overlooking the North Sea. The monument is flanked by a lamp post with barbed wire decorations and surrounded by roc...                               |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`              | Title: Felixstowe Seafront Monument with Ocean View.<br>Description: The image depicts a monument situated on the seafront in Felixstowe, England, UK. The monument is a tall, white column with a Doric column style, topped with a statue of an eagle. The column is surrounded by a c...                                        |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | Title: Seafront War Memorial, Felixstowe, England, UK, GBR<br><br>Description: A war memorial stands prominently on a seaside promenade in Felixstowe, England, overlooking the ocean. The monument features a tall column topped with an eagle statue, surrounded by a decorative light fi...                                     |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**Felixstowe War Memorial and Seafront View**<br><br>Description:<br>A tall classical war memorial column stands prominently on Felixstowe’s seafront, overlooking the ocean under clear blue skies. The monument features engraved plaques and a bronze eagle finial, with two modern...                                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>War Memorial Column at Felixstowe Seafront<br><br>Description:<br>A tall, classical column stands prominently at the seafront of Felixstowe, England, dedicated to honoring fallen soldiers. It features inscriptions, a sculpted eagle atop, and is surrounded by coastal rocks under...                                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: War Memorial Column on Felixstowe Seafront<br><br>Description: A tall stone column topped with a bronze eagle stands on a plinth inscribed with names, situated on a paved promenade overlooking the sea under a clear blue sky.<br><br>Keywords: Felixstowe, seafront, war memorial, colu...                               |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | Title: Felixstowe Seafront Memorial Column with Eagle<br><br>Description: A tall stone column topped with an eagle statue stands on a seafront promenade in Felixstowe, England, under clear blue skies, with the sea and festive string lights visible in the background.<br><br>Keywords: Felix...                               |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable`              | Title:<br>War Memorial Column on the Felixstowe Seafront<br><br>Description:<br>A tall, fluted stone column topped with a bronze eagle stands on a stepped plinth inscribed with names, set against a calm blue sea and clear sky. The monument is situated on a paved promenade with a blue lamp...                               |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable`              | Title:<br>Seafront, Felixstowe, England, UK, GBR, Europe<br>Description:<br>A tall, white column stands on a rocky beach, overlooking the ocean. The column is topped with a statue of an eagle, and there are several plaques on the column.<br>Keywords:<br>Felixstowe, England, UK, GBR, Europe,...                             |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | Title:<br>Column monument on Felixstowe seafront<br><br>Description:<br>A tall column monument topped with a bird statue stands on a paved seafront area in Felixstowe, Suffolk, England, overlooking the ocean under a clear blue sky, with two men walking nearby and decorative streetlights a...                               |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: War Memorial Column on Felixstowe Seafront<br>Description: A tall stone war memorial column topped with a bird sculpture stands against a clear blue sky on the Felixstowe seafront. The monument features inscribed plaques and is set against a backdrop of the sea and rocky c...                                        |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: War Memorial on the Seafront in Felixstowe, Suffolk<br>Description: A tall, fluted Corinthian column topped with an eagle sculpture stands on the seafront in Felixstowe, England, under a clear blue sky. The monument features inscribed plaques at its base, overlooking the c...                                        |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | Title:<br>Seafront Monument in Felixstowe, England<br><br>Description:<br>A tall column monument with an eagle statue at the top stands prominently by the seaside in Felixstowe, England, with a clear blue sky and calm sea in the background.<br><br>Keywords:<br>Felixstowe, England, UK, seafront, mo...                      |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Seafront monument, Felixstowe, England<br>Description: A tall stone column with an eagle statue atop stands on a paved plaza by the sea, with rocky shores, the ocean, and two men walking in the background under a clear blue sky.<br>Keywords: seafront, Felixsto...                                   |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable_with_caveats` | Title: Felixstowe Seafront Monument<br>Description: A tall column with an eagle statue stands on a seafront promenade in Felixstowe, England, under clear blue skies, with people walking nearby and festive lights strung overhead.<br>Keywords: Felixstowe, seafront, monument, eagle sta...                                     |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | ◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image. The image shows a tall column monument by the sea, likely a war memorial, with an eagle statue on top...                                           |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `usable_with_caveats` | &lt;think&gt;<br><br>&lt;/think&gt;<br><br>Title:<br>War Memorial by Seafront<br><br>Description:<br>A tall war memorial stands by the sea, under clear skies, with people walking nearby.<br><br>Keywords: war memorial, seafront, Felixstowe, England, UK, gbr, ocean, rocks, pedestrians, clear sky, decorative lights, ston... |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | **Title:**<br>*Felixstowe War Memorial, Seafront, Suffolk, UK*<br><br>**Description:**<br>A classical stone column monument dedicated to WWI and WWII servicemen stands on Felixstowe’s seafront, overlooking the North Sea under clear daylight. Decorated with inscriptions, plaques, and toppe...                               |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | Title: Felixstowe Seafront, England, UK, GBR<br><br>Description: A man in a blue shirt walks along a concrete path near a tall white column with a bird statue, surrounded by rocks and water, with a blue pole and string lights in the background. The scene is bathed in warm sunlight,...                                      |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable_with_caveats` | Title: War memorial column on Felixstowe seafront<br>Description: A tall fluted stone column topped with a bronze eagle stands on a stepped plinth inscribed with names, set against the blue sea and clear sky at Felixstowe seafront. Festive bunting and a modern communications mast...                                        |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable_with_caveats` | Title: Felixstowe Seafront Monument<br><br>Description: A prominent white column with a bird sculpture at the top stands at the seafront in Felixstowe, England, UK, with a clear blue sky and the sea in the background.<br><br>Keywords: Felixstowe, seafront, monument, column, bird sculpture...                               |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable_with_caveats` | Got it, let's tackle this step by step. First, the title. The context says Felixstowe, England, UK, so we need a concise title. The main subject is the memorial column. Let's see: "Felixstowe Seafront Memorial Column" – that's 10 words, which is good.<br><br>Next, the description. N...                                     |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable_with_caveats` | Title:<br>Felixstowe War Memorial Column by the Seafront<br><br>Description:<br>A tall stone column topped with a bronze bird statue stands on a paved promenade overlooking the sea. The monument features inscribed plaques and is situated near a blue lamppost with festive lights, under a c...                               |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable_with_caveats` | Title:<br>Felixstowe War Memorial, Seafront<br><br>Description:<br>A tall, ornate war memorial stands on the seafront at Felixstowe, East Suffolk, UK, with a bronze eagle perched atop, set against a clear blue sky and the calm sea. The structure bears inscriptions and plaques, marking its...                               |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: War Memorial Column on the Felixstowe Seafront<br>Description: A tall, fluted stone column topped with an eagle statue stands on the coastal promenade overlooking the sea under a clear blue sky.<br>Keywords: Felixstowe, Suffolk, England, UK, coasta...                                |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: War Memorial Column on the Felixstowe Seafront<br>Description: A tall fluted stone column topped with a bronze eagle statue stands along the English coast under a clear blue sky.<br>Keywords: Felixstowe, Suffolk, East Suffolk, England, UK, war memo...                                |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | Title: Felixstowe Seafront War Memorial, Suffolk<br><br>Description: Captured on 13th August 2026, this photograph shows the War Memorial on Felixstowe seafront, Suffolk, bathed in bright sunlight with the North Sea visible beyond. The memorial stands prominently on a stone plinth,...                                      |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `unusable`            | Title:<br>Felixstowe Seafront Memorial<br><br>Description:<br>A stone memorial column stands at the seafront in Felixstowe, England, topped with a bronze eagle, commemorating the 1939-1945 war, with plaques listing names and a blue lamppost nearby under a clear blue sky.<br><br>Keywords: Felixs...                         |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>The user wants British-English catalogue metadata from the image and supplied context. They want:<br><br>- Title: a concrete 5-10-word title.<br>- Description: 1-2 sentences factual description combining relevant context with main visible subject, setting,...                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | Alright, let's get to work. I've got an image of a seaside scene, and I need to generate some metadata for it. First things first, I need to break down the request. The core task is to create a title, description, and keywords for this image, all in British English. The image...                                            |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A concrete 5-10-word title: "Felixstowe, UK, Seafront, GBR, East Suffolk, England, Europe, GBR, Seafront"<br>Description: The image features a prominent 5-10-word title describing a concrete 5-10-word description of a seafront in Felixstowe, UK, with a GBR (Great Britain) locatio...                                        |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. The image shows a monument by the sea, so maybe "Felixstowe Seafront Monument". Then the description should combine context (date, location) with the main subject (monument), setting (seafront), action...                                      |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `unusable`            | Title: Felixstowe War Memorial<br>Description: The Felixstowe War Memorial, a tall stone column topped with an eagle, stands by the seafront in Felixstowe, England.<br>Keywords: Felixstowe War Memorial, seafront, Felixstowe, Suffolk, England, UK, gbr, Europe, war memorial, stone col...                                     |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `unusable`            | Title:<br>Felixstowe Seafront Memorial Column<br><br>Description:<br>The image features a prominent memorial column situated on the seafront in Felixstowe, England, with a columnar structure and a bird statue at the top, set against a clear blue sky and the calm sea in the background.<br><br>Ke...                         |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | Felixstowe War Memorial, East Suffolk Coast<br><br>The image shows a war memorial monument on a rocky seafront in Felixstowe, England. A tall white column with an eagle statue stands prominently, surrounded by a stone base with plaques. The memorial overlooks the ocean, with a clear...                                     |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `unusable`            | Title:<br>Seafront, Felixstowe, England, UK, GBR, Europe<br><br>Description:<br>The Seafront of Felixstowe, England, UK, GBR, Europe, features a prominent seafront monument with a stone column and a bird statue on top. The monument is situated near the sea, with a clear blue sky and a cal...                               |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `unusable`            | Title:<br>Felixstowe Seafront Memorial<br><br>Description:<br>A stone memorial column stands at the seafront in Felixstowe, England, topped with a bronze eagle, commemorating the 1939-1945 war, with plaques listing names and a blue lamppost nearby under a clear blue sky.<br><br>Keywords: Felixs...                         |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | A tall, light-colored stone monument stands prominently on a rocky coastline under a clear blue sky. The monument features a fluted column topped with a bronze eagle with outstretched wings. The base of the monument is multi-tiered and adorned with several dark plaques bearing...                                           |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | Seafront, Felixstowe, England, UK, GBR, Europe,                                                                                                                                                                                                                                                                                    |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | Title: Seafront, Felixstowe, England, UK, GBR, Europe<br>Description: A concrete 5-10-word title, followed by a 1-2-sentence factual description that combines relevant context with the main visible subject, setting, action, lighting, and distinctive details. The description inclu...                                        |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | - Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, GBR,<br>- Description hint: Seafront, Felixstowe, England, UK, GBR,<br>- Description hint: Seafront, Felixstowe, England, UK, GBR,<br>- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Euro...                                  |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.14`
- `mlx`: `0.32.2.dev20260818+7f062ddcb`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.15.0`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.28.0`
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
- *Date:* 2026-08-13 17:14:49 UTC+01:00
- *Time:* 17:14:49
- *GPS:* 51.959333°N, 1.349050°E

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
> &#45; Capture date/time: 2026-08-13 17:14:49 UTC+01:00
> &#45; GPS: 51.959333°N, 1.349050°E
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
- *Model load time:* 0.54s
- *Generation time:* 0.76s
- *Total time:* 1.56s
- *Input validation time:* 0.247
- *Prompt preparation time:* 0.00604
- *First-token latency:* 0.107
- *Cleanup time:* 0.0586
- *Prompt tokens:* 2,072
- *Generation tokens:* 132
- *Total tokens:* 2,204
- *Prompt throughput (raw):* 19,434 tok/s
- *Generation throughput (raw):* 483 tok/s
- *Peak memory:* 1.9
- *Active memory:* 0.90
- *Cache memory:* 0.08
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
A tall, white column stands prominently on a coastal promenade, overlooking the sea. The column is adorned with a statue of an eagle perched atop, and a string of colorful lights runs along its base. The scene is set against a clear blue sky, with a few people walking along the promenade, and the ocean stretches out in the background.

Keywords:
Seafront, Felixstowe, England, UK, GBR, Europe, statue, eagle, coastal promenade, lights, sky, sea
</pre>

</details>

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

<details>
<summary>Complete evidence: mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.39s
- *Generation time:* 8.67s
- *Total time:* 11.08s
- *Input validation time:* 9.78e-05
- *Prompt preparation time:* 0.0071
- *First-token latency:* 3.72
- *Cleanup time:* 0.103
- *Prompt tokens:* 2,658
- *Generation tokens:* 131
- *Total tokens:* 2,789
- *Prompt throughput (raw):* 715 tok/s
- *Generation throughput (raw):* 29.3 tok/s
- *Peak memory:* 24
- *Active memory:* 18
- *Cache memory:* 0.52
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
War Memorial by the Sea

Description:
A tall stone war memorial column topped with a winged figure stands on a seafront promenade in Felixstowe, England, overlooking the North Sea. The monument is flanked by a lamp post with barbed wire decorations and surrounded by rocky outcrops, with people walking nearby under a clear blue sky.

Keywords:
war memorial, seafront, Felixstowe, England, UK, North Sea, stone column, winged figure, barbed wire, lamp post, rocky outcrops, promenade, clear sky, people walking, historical monument, coastal view, Suffolk, Europe
</pre>

</details>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 2.08s
- *Generation time:* 7.97s
- *Total time:* 10.06s
- *Input validation time:* 7.23e-05
- *Prompt preparation time:* 0.00151
- *First-token latency:* 1.12
- *Cleanup time:* 0.102
- *Prompt tokens:* 2,587
- *Generation tokens:* 202
- *Total tokens:* 2,789
- *Prompt throughput (raw):* 2,310 tok/s
- *Generation throughput (raw):* 32.3 tok/s
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
 Title: Felixstowe Seafront Monument with Ocean View.
Description: The image depicts a monument situated on the seafront in Felixstowe, England, UK. The monument is a tall, white column with a Doric column style, topped with a statue of an eagle. The column is surrounded by a circular base and has a plaque with inscriptions. In the background, the vast expanse of the ocean stretches towards the horizon, meeting the clear blue sky. To the left of the monument, there are two people standing on a stone embankment, overlooking the sea. The lighting suggests it is either early morning or late afternoon, with the sun casting soft shadows. The scene is serene and picturesque, capturing the beauty of the coastal landscape.
Keywords: Felixstowe, seafront, England, UK, monument, Doric column, eagle statue, ocean, plaque, inscriptions, stone embankment, coastal landscape, serene, picturesque, early morning, late afternoon, soft shadows.
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
- *Model load time:* 0.77s
- *Generation time:* 1.55s
- *Total time:* 2.33s
- *Input validation time:* 9.2e-05
- *Prompt preparation time:* 0.00371
- *First-token latency:* 0.37
- *Cleanup time:* 0.0734
- *Prompt tokens:* 2,072
- *Generation tokens:* 140
- *Total tokens:* 2,212
- *Prompt throughput (raw):* 5,596 tok/s
- *Generation throughput (raw):* 187 tok/s
- *Peak memory:* 4.0
- *Active memory:* 3.2
- *Cache memory:* 0.06
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
Title: Seafront War Memorial, Felixstowe, England, UK, GBR

Description: A war memorial stands prominently on a seaside promenade in Felixstowe, England, overlooking the ocean. The monument features a tall column topped with an eagle statue, surrounded by a decorative light fixture and a plaque listing names of the fallen. The scene is bathed in bright daylight, with clear blue skies and a few people strolling along the path.

Keywords: Seafront, Felixstowe, England, UK, GBR, War Memorial, Eagle Statue, Light Fixture, Plaque, Ocean View, Promenade, Daylight, People
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
- *Model load time:* 1.61s
- *Generation time:* 5.21s
- *Total time:* 6.83s
- *Input validation time:* 7.2e-05
- *Prompt preparation time:* 0.00677
- *First-token latency:* 2.34
- *Cleanup time:* 0.09
- *Prompt tokens:* 3,191
- *Generation tokens:* 162
- *Total tokens:* 3,353
- *Prompt throughput (raw):* 1,364 tok/s
- *Generation throughput (raw):* 67.0 tok/s
- *Peak memory:* 14
- *Active memory:* 8.0
- *Cache memory:* 0.65
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
**Felixstowe War Memorial and Seafront View**

Description:
A tall classical war memorial column stands prominently on Felixstowe’s seafront, overlooking the ocean under clear blue skies. The monument features engraved plaques and a bronze eagle finial, with two modern lamp posts and decorative string lights nearby. Two individuals are seen walking along the paved path beside the rocky shore.

Keywords:
Felixstowe, Suffolk, UK, GBR, Europe, seafront, war memorial, ocean view, classical column, engraved plaques, bronze eagle, lamp posts, string lights, rocky shore, coastal path, 2026-08-13, 51.959333°N, 1.349050°E
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
- *Model load time:* 1.09s
- *Generation time:* 2.24s
- *Total time:* 3.33s
- *Input validation time:* 5.72e-05
- *Prompt preparation time:* 0.00695
- *First-token latency:* 1.13
- *Cleanup time:* 0.0765
- *Prompt tokens:* 3,190
- *Generation tokens:* 123
- *Total tokens:* 3,313
- *Prompt throughput (raw):* 2,816 tok/s
- *Generation throughput (raw):* 189 tok/s
- *Peak memory:* 9.0
- *Active memory:* 2.8
- *Cache memory:* 0.41
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
War Memorial Column at Felixstowe Seafront

Description:
A tall, classical column stands prominently at the seafront of Felixstowe, England, dedicated to honoring fallen soldiers. It features inscriptions, a sculpted eagle atop, and is surrounded by coastal rocks under clear blue skies with decorative string lights strung nearby.

Keywords:
war memorial, Felixstowe, seafront, Suffolk, UK, coastal column, commemorative plaque, eagle sculpture, maritime setting, string lights, historic monument, East Anglian coast, UK GBR, coastal defense, memorial inscriptions, coastal lighting.
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
- *Model load time:* 2.65s
- *Generation time:* 60.57s
- *Total time:* 63.24s
- *Input validation time:* 8.49e-05
- *Prompt preparation time:* 0.0147
- *First-token latency:* 58.9
- *Cleanup time:* 0.119
- *Prompt tokens:* 16,482
- *Generation tokens:* 95
- *Total tokens:* 16,577
- *Prompt throughput (raw):* 280 tok/s
- *Generation throughput (raw):* 110 tok/s
- *Peak memory:* 24
- *Active memory:* 21
- *Cache memory:* 0.88
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
Title: War Memorial Column on Felixstowe Seafront

Description: A tall stone column topped with a bronze eagle stands on a plinth inscribed with names, situated on a paved promenade overlooking the sea under a clear blue sky.

Keywords: Felixstowe, seafront, war memorial, column, eagle, monument, England, UK, Suffolk, East Suffolk, promenade, ocean, blue sky, stone, sculpture, commemoration, coastal, architecture
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
- *Model load time:* 1.53s
- *Generation time:* 62.82s
- *Total time:* 64.36s
- *Input validation time:* 8.1e-05
- *Prompt preparation time:* 0.0006
- *First-token latency:* 60.8
- *Cleanup time:* 0.0882
- *Prompt tokens:* 16,482
- *Generation tokens:* 113
- *Total tokens:* 16,595
- *Prompt throughput (raw):* 271 tok/s
- *Generation throughput (raw):* 93.5 tok/s
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
Title: Felixstowe Seafront Memorial Column with Eagle

Description: A tall stone column topped with an eagle statue stands on a seafront promenade in Felixstowe, England, under clear blue skies, with the sea and festive string lights visible in the background.

Keywords: Felixstowe, seafront, memorial column, eagle statue, stone pillar, coastal view, blue sky, string lights, rock breakwater, public space, East Suffolk, England, UK, GBR, Europe, 2026-08-13
</pre>

</details>

---

<a id="model-mlx-community-qwen38-27b-4bit"></a>

### mlx-community/Qwen3.8-27B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.8-27B-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 2.25s
- *Generation time:* 85.50s
- *Total time:* 87.78s
- *Input validation time:* 9.6e-05
- *Prompt preparation time:* 0.0155
- *First-token latency:* 80.3
- *Cleanup time:* 0.139
- *Prompt tokens:* 16,482
- *Generation tokens:* 127
- *Total tokens:* 16,609
- *Prompt throughput (raw):* 205 tok/s
- *Generation throughput (raw):* 29.5 tok/s
- *Peak memory:* 22
- *Active memory:* 18
- *Cache memory:* 1.8
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0101
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
- *Resolved model revision:* 3e6447f082e89cc7f0bc6e5441afd38dfce760ff
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
War Memorial Column on the Felixstowe Seafront

Description:
A tall, fluted stone column topped with a bronze eagle stands on a stepped plinth inscribed with names, set against a calm blue sea and clear sky. The monument is situated on a paved promenade with a blue lamp post and string lights nearby, while pedestrians walk along the rocky shoreline in the background.

Keywords:
Felixstowe, seafront, war memorial, column, eagle, bronze, stone, Suffolk, England, UK, promenade, ocean, sky, monument, street lamp, pedestrians, coastal, architecture
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
- *Model load time:* 1.05s
- *Generation time:* 1.71s
- *Total time:* 2.77s
- *Input validation time:* 9.22e-05
- *Prompt preparation time:* 0.0022
- *First-token latency:* 0.378
- *Cleanup time:* 0.0809
- *Prompt tokens:* 1,400
- *Generation tokens:* 89
- *Total tokens:* 1,489
- *Prompt throughput (raw):* 3,704 tok/s
- *Generation throughput (raw):* 126 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.35
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
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

*Readable output:*

<pre class="model-output-readable">
 Title:
Seafront, Felixstowe, England, UK, GBR, Europe
Description:
A tall, white column stands on a rocky beach, overlooking the ocean. The column is topped with a statue of an eagle, and there are several plaques on the column.
Keywords:
Felixstowe, England, UK, GBR, Europe, Column, Eagle, Plaques, Beach, Ocean
</pre>

</details>

---

<a id="model-mlx-community-step-37-flash-oq2e"></a>

### mlx-community/Step-3.7-Flash-oQ2e

<details>
<summary>Complete evidence: mlx-community/Step-3.7-Flash-oQ2e</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 5.98s
- *Generation time:* 18.64s
- *Total time:* 24.64s
- *Input validation time:* 8.18e-05
- *Prompt preparation time:* 0.0165
- *First-token latency:* 15.5
- *Cleanup time:* 0.208
- *Prompt tokens:* 3,468
- *Generation tokens:* 114
- *Total tokens:* 3,582
- *Prompt throughput (raw):* 224 tok/s
- *Generation throughput (raw):* 44.8 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.39
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.0105
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
Column monument on Felixstowe seafront

Description:
A tall column monument topped with a bird statue stands on a paved seafront area in Felixstowe, Suffolk, England, overlooking the ocean under a clear blue sky, with two men walking nearby and decorative streetlights and a WiFi sign present.

Keywords:
Felixstowe, Suffolk, England, UK, seafront, column monument, bird statue, ocean view, blue sky, streetlights, WiFi sign, coastal promenade, war memorial, public monument, East Suffolk, Europe
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
- *Model load time:* 2.48s
- *Generation time:* 1.65s
- *Total time:* 4.15s
- *Input validation time:* 8.15e-05
- *Prompt preparation time:* 0.0205
- *First-token latency:* 0.4
- *Cleanup time:* 0.118
- *Prompt tokens:* 580
- *Generation tokens:* 98
- *Total tokens:* 678
- *Prompt throughput (raw):* 1,449 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.20
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0124
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
Title: War Memorial Column on Felixstowe Seafront
Description: A tall stone war memorial column topped with a bird sculpture stands against a clear blue sky on the Felixstowe seafront. The monument features inscribed plaques and is set against a backdrop of the sea and rocky coastline.
Keywords: Felixstowe, Suffolk, England, UK, war memorial, column, monument, seafront, coast, sea, blue sky, stone, sculpture, memorial, East Suffolk, maritime, daylight
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
- *Model load time:* 2.64s
- *Generation time:* 6.11s
- *Total time:* 8.76s
- *Input validation time:* 7.57e-05
- *Prompt preparation time:* 0.00037
- *First-token latency:* 1.21
- *Cleanup time:* 0.128
- *Prompt tokens:* 580
- *Generation tokens:* 111
- *Total tokens:* 691
- *Prompt throughput (raw):* 480 tok/s
- *Generation throughput (raw):* 25.1 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.76
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0129
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
Title: War Memorial on the Seafront in Felixstowe, Suffolk
Description: A tall, fluted Corinthian column topped with an eagle sculpture stands on the seafront in Felixstowe, England, under a clear blue sky. The monument features inscribed plaques at its base, overlooking the coast and sea.
Keywords: Felixstowe, Suffolk, East Suffolk, England, UK, seafront, war memorial, monument, Corinthian column, eagle sculpture, coast, seaside, public art, commemorative, blue sky, outdoor, Europe, gbr
</pre>

</details>

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### mlx-community/pixtral-12b-8bit

<details>
<summary>Complete evidence: mlx-community/pixtral-12b-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.85s
- *Generation time:* 4.75s
- *Total time:* 6.60s
- *Input validation time:* 5.97e-05
- *Prompt preparation time:* 0.00264
- *First-token latency:* 1.81
- *Cleanup time:* 0.104
- *Prompt tokens:* 3,429
- *Generation tokens:* 99
- *Total tokens:* 3,528
- *Prompt throughput (raw):* 1,893 tok/s
- *Generation throughput (raw):* 39.5 tok/s
- *Peak memory:* 16
- *Active memory:* 13
- *Cache memory:* 0.64
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0149
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
Seafront Monument in Felixstowe, England

Description:
A tall column monument with an eagle statue at the top stands prominently by the seaside in Felixstowe, England, with a clear blue sky and calm sea in the background.

Keywords:
Felixstowe, England, UK, seafront, monument, column, eagle statue, seaside, blue sky, calm sea, coastal, memorial, stone, plaque, commemorative, seascape, East Suffolk
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
  4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 6.42s
- *Generation time:* 17.27s
- *Total time:* 23.70s
- *Input validation time:* 8.53e-05
- *Prompt preparation time:* 0.0073
- *First-token latency:* 14.6
- *Cleanup time:* 0.211
- *Prompt tokens:* 6,311
- *Generation tokens:* 96
- *Total tokens:* 6,407
- *Prompt throughput (raw):* 432 tok/s
- *Generation throughput (raw):* 43.3 tok/s
- *Peak memory:* 78
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
&lt;|begin_of_box|&gt;Title: Seafront monument, Felixstowe, England
Description: A tall stone column with an eagle statue atop stands on a paved plaza by the sea, with rocky shores, the ocean, and two men walking in the background under a clear blue sky.
Keywords: seafront, Felixstowe, England, UK, GBR, Europe, monument, column, eagle statue, stone, plaza, ocean, rocky shores, clear sky, East Suffolk, Suffolk&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Seafront monument, Felixstowe, England
Description: A tall stone column with an eagle statue atop stands on a paved plaza by the sea, with rocky shores, the ocean, and two men walking in the background under a clear blue sky.
Keywords: seafront, Felixstowe, England, UK, GBR, Europe, monument, column, eagle statue, stone, plaza, ocean, rocky shores, clear sky, East Suffolk, Suffolk<|end_of_box|>
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
- *Observations:* Title has 3 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type internvl_chat)
- *Model load time:* 1.79s
- *Generation time:* 4.40s
- *Total time:* 6.20s
- *Input validation time:* 8.5e-05
- *Prompt preparation time:* 0.00202
- *First-token latency:* 1.38
- *Cleanup time:* 0.113
- *Prompt tokens:* 3,623
- *Generation tokens:* 85
- *Total tokens:* 3,708
- *Prompt throughput (raw):* 2,634 tok/s
- *Generation throughput (raw):* 34.3 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.28
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
Title: Felixstowe Seafront Monument
Description: A tall column with an eagle statue stands on a seafront promenade in Felixstowe, England, under clear blue skies, with people walking nearby and festive lights strung overhead.
Keywords: Felixstowe, seafront, monument, eagle statue, column, promenade, England, Suffolk, UK, Europe, clear skies, festive lights, people, walking
</pre>

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-2506-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Conversation-role control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 3.40s
- *Generation time:* 132.32s
- *Total time:* 135.74s
- *Input validation time:* 7.13e-05
- *Prompt preparation time:* 0.00211
- *First-token latency:* 1.31
- *Cleanup time:* 0.159
- *Prompt tokens:* 1,257
- *Generation tokens:* 610
- *Total tokens:* 1,867
- *Prompt throughput (raw):* 961 tok/s
- *Generation throughput (raw):* 4.68 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.6
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00365
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
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
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image. The image shows a tall column monument by the sea, likely a war memorial, with an eagle statue on top. The background has the sea, rocks, a clear blue sky, and some people walking. The location is Felixstowe, England, UK, as per the context.

First, the title: needs to be a 5-10 word concrete title. Maybe " Felixstowe Seafront Memorial Column by the Coast". That's 7 words.

Description: 1-2 sentences. It should combine context (Felixstowe, England, UK), main subject (memorial column), setting (seafront, sea, rocks), action (people walking), lighting (bright sunny day), distinctive details (column with eagle, plaques, string lights). So: "A tall stone memorial column with an eagle statue stands on the seafront of Felixstowe, England, UK, near the coast with people walking nearby under a clear blue sky."

Keywords: Need 10-18 unique keywords. Let's list: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear sky, people walking, plaques, string lights, blue sky. That's 10 keywords.

Wait, check if all context and visible details are covered. The context mentions East Suffolk, UK, Europe, but the image doesn't show East Suffolk specifically, but Felixstowe is in Suffolk. The keywords should include Felixstowe, England, UK, Europe. Also, the eagle statue, the column, the sea, rocks, people, string lights, plaques, clear sky. Let's adjust: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear blue sky, people walking, plaques, string lights. That's 10. Maybe add "war memorial" but the hint says to omit uncertain details. The plaques might be part of the memorial, so include plaques. Let's finalize:

Title: Felixstowe Seafront Memorial Column by the Coast

Description: A tall stone memorial column topped with an eagle statue stands on the seafront of Felixstowe, England, UK, near coastal rocks with people walking nearby under a clear blue sky.

Keywords: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear blue sky, people walking, plaques, string lights◁/think▷Title: Felixstowe Seafront Memorial Column by the Coast

Description: A tall stone memorial column topped with an eagle statue stands on the seafront of Felixstowe, England, UK, near coastal rocks with people walking nearby under a clear blue sky.

Keywords: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear blue sky, people walking, plaques, string lights&lt;|im_assistant|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image. The image shows a tall column monument by the sea, likely a war memorial, with an eagle statue on top. The background has the sea, rocks, a clear blue sky, and some people walking. The location is Felixstowe, England, UK, as per the context. 

First, the title: needs to be a 5-10 word concrete title. Maybe " Felixstowe Seafront Memorial Column by the Coast". That's 7 words.

Description: 1-2 sentences. It should combine context (Felixstowe, England, UK), main subject (memorial column), setting (seafront, sea, rocks), action (people walking), lighting (bright sunny day), distinctive details (column with eagle, plaques, string lights). So: "A tall stone memorial column with an eagle statue stands on the seafront of Felixstowe, England, UK, near the coast with people walking nearby under a clear blue sky."

Keywords: Need 10-18 unique keywords. Let's list: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear sky, people walking, plaques, string lights, blue sky. That's 10 keywords.

Wait, check if all context and visible details are covered. The context mentions East Suffolk, UK, Europe, but the image doesn't show East Suffolk specifically, but Felixstowe is in Suffolk. The keywords should include Felixstowe, England, UK, Europe. Also, the eagle statue, the column, the sea, rocks, people, string lights, plaques, clear sky. Let's adjust: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear blue sky, people walking, plaques, string lights. That's 10. Maybe add "war memorial" but the hint says to omit uncertain details. The plaques might be part of the memorial, so include plaques. Let's finalize:

Title: Felixstowe Seafront Memorial Column by the Coast

Description: A tall stone memorial column topped with an eagle statue stands on the seafront of Felixstowe, England, UK, near coastal rocks with people walking nearby under a clear blue sky.

Keywords: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear blue sky, people walking, plaques, string lights◁/think▷Title: Felixstowe Seafront Memorial Column by the Coast

Description: A tall stone memorial column topped with an eagle statue stands on the seafront of Felixstowe, England, UK, near coastal rocks with people walking nearby under a clear blue sky.

Keywords: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear blue sky, people walking, plaques, string lights<|im_assistant|>
```

</details>

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
- *Model load time:* 1.05s
- *Generation time:* 1.10s
- *Total time:* 2.16s
- *Input validation time:* 5.83e-05
- *Prompt preparation time:* 0.00857
- *First-token latency:* 0.256
- *Cleanup time:* 0.084
- *Prompt tokens:* 964
- *Generation tokens:* 73
- *Total tokens:* 1,037
- *Prompt throughput (raw):* 3,763 tok/s
- *Generation throughput (raw):* 272 tok/s
- *Peak memory:* 3.8
- *Active memory:* 2.3
- *Cache memory:* 0.06
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
War Memorial by Seafront

Description:
A tall war memorial stands by the sea, under clear skies, with people walking nearby.

Keywords: war memorial, seafront, Felixstowe, England, UK, gbr, ocean, rocks, pedestrians, clear sky, decorative lights, stone, commemoration, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>

</think>

Title:
War Memorial by Seafront

Description:
A tall war memorial stands by the sea, under clear skies, with people walking nearby.

Keywords: war memorial, seafront, Felixstowe, England, UK, gbr, ocean, rocks, pedestrians, clear sky, decorative lights, stone, commemoration, Europe
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
- *Model load time:* 1.53s
- *Generation time:* 5.40s
- *Total time:* 6.94s
- *Input validation time:* 6.91e-05
- *Prompt preparation time:* 0.000349
- *First-token latency:* 2.25
- *Cleanup time:* 0.0905
- *Prompt tokens:* 3,191
- *Generation tokens:* 174
- *Total tokens:* 3,365
- *Prompt throughput (raw):* 1,420 tok/s
- *Generation throughput (raw):* 64.5 tok/s
- *Peak memory:* 15
- *Active memory:* 8.4
- *Cache memory:* 0.65
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
*Felixstowe War Memorial, Seafront, Suffolk, UK*

**Description:**
A classical stone column monument dedicated to WWI and WWII servicemen stands on Felixstowe’s seafront, overlooking the North Sea under clear daylight. Decorated with inscriptions, plaques, and topped with a winged figure, it sits beside a modern lamp post adorned with festive lights, surrounded by rocky breakwaters and visited by pedestrians.

**Keywords:**
Felixstowe, Suffolk, United Kingdom, seafront, war memorial, WWI, WWII, stone column, winged figure, rocky breakwaters, North Sea, classical monument, inscriptions, festive lights, daylight, pedestrians, East Anglia, commemorative site, authoritative context, 2026-08-13
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
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.28s
- *Generation time:* 3.83s
- *Total time:* 5.13s
- *Input validation time:* 7.1e-05
- *Prompt preparation time:* 0.00159
- *First-token latency:* 0.652
- *Cleanup time:* 0.0821
- *Prompt tokens:* 1,495
- *Generation tokens:* 122
- *Total tokens:* 1,617
- *Prompt throughput (raw):* 2,293 tok/s
- *Generation throughput (raw):* 52.9 tok/s
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
 Title: Felixstowe Seafront, England, UK, GBR

Description: A man in a blue shirt walks along a concrete path near a tall white column with a bird statue, surrounded by rocks and water, with a blue pole and string lights in the background. The scene is bathed in warm sunlight, creating a picturesque coastal atmosphere.

Keywords: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront, concrete, column, bird statue, man in blue shirt, walk, water, rocks, sun, warm lighting
</pre>

</details>

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### mlx-community/Ornith-1.0-35B-bf16

<details>
<summary>Complete evidence: mlx-community/Ornith-1.0-35B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Duplicate keywords: england
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 8.16s
- *Generation time:* 56.00s
- *Total time:* 64.19s
- *Input validation time:* 7.15e-05
- *Prompt preparation time:* 0.0142
- *First-token latency:* 53
- *Cleanup time:* 0.223
- *Prompt tokens:* 16,482
- *Generation tokens:* 111
- *Total tokens:* 16,593
- *Prompt throughput (raw):* 311 tok/s
- *Generation throughput (raw):* 63.1 tok/s
- *Peak memory:* 74
- *Active memory:* 70
- *Cache memory:* 0.88
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
Title: War memorial column on Felixstowe seafront
Description: A tall fluted stone column topped with a bronze eagle stands on a stepped plinth inscribed with names, set against the blue sea and clear sky at Felixstowe seafront. Festive bunting and a modern communications mast are visible nearby.
Keywords: Felixstowe, seafront, war memorial, column, eagle, Suffolk, England, UK, sea, promenade, bunting, blue sky, stone, monument, East Suffolk, coastal, England, Europe
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
- *Model load time:* 1.10s
- *Generation time:* 2.12s
- *Total time:* 3.23s
- *Input validation time:* 9.57e-05
- *Prompt preparation time:* 0.00113
- *First-token latency:* 0.231
- *Cleanup time:* 0.083
- *Prompt tokens:* 1,094
- *Generation tokens:* 81
- *Total tokens:* 1,175
- *Prompt throughput (raw):* 4,735 tok/s
- *Generation throughput (raw):* 56.7 tok/s
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
Title: Felixstowe Seafront Monument

Description: A prominent white column with a bird sculpture at the top stands at the seafront in Felixstowe, England, UK, with a clear blue sky and the sea in the background.

Keywords: Felixstowe, seafront, monument, column, bird sculpture, blue sky, sea, rocks, people, coast
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.88s
- *Generation time:* 28.34s
- *Total time:* 29.24s
- *Input validation time:* 6.26e-05
- *Prompt preparation time:* 0.0114
- *First-token latency:* 17.5
- *Cleanup time:* 0.106
- *Prompt tokens:* 16,469
- *Generation tokens:* 903
- *Total tokens:* 17,372
- *Prompt throughput (raw):* 940 tok/s
- *Generation throughput (raw):* 90.6 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.7
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00806
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
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
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 800,
  "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Got it, let's tackle this step by step. First, the title. The context says Felixstowe, England, UK, so we need a concise title. The main subject is the memorial column. Let's see: "Felixstowe Seafront Memorial Column" – that's 10 words, which is good.

Next, the description. Need to combine context with visible subject, setting, action, lighting, details. The image shows a seafront with a tall column monument, blue sky, people walking, rocks, a blue pole with lights. The capture date is 2026-08-13, but we don't claim that. The main subject is the memorial. So: "A tall stone memorial column with an eagle statue stands at Felixstowe seafront, overlooking the ocean under a clear blue sky, with people walking along the rocky shore and colorful string lights."

Wait, check the context: "Seafront, Felixstowe, England, UK, GBR, Europe" – so the description should include that. Also, the monument has inscriptions, the blue pole with lights. Let's make sure it's factual. The lighting is sunny, clear sky. The people are walking, rocks in the background. The monument is the central feature.

Now keywords. Need 10-18 unique, comma-separated. From context: Felixstowe, England, UK, GBR, Europe, Seafront, Suffolk, UK, gbr. Visible details: memorial column, eagle statue, blue sky, ocean, rocky shore, string lights, people walking, stone monument, inscriptions. Let's list: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument, Inscriptions, Blue Sky, Sunny Day.

Wait, need to check if they are unique. Let's pick 10-18. Let's see: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument, Inscriptions, Blue Sky, Sunny Day. That's 15. Let's check if any are redundant. Maybe "Suffolk" is in the context, but the image doesn't show Suffolk specifically, but the context has "East Suffolk" as a keyword hint. Wait, the context says "East Suffolk, England, UK, Europe" – but the image is Felixstowe, which is in Suffolk. So "East Suffolk" might be part of the context. Wait, the descriptive hints have "East Suffolk" as a keyword hint. So include that. Let's see: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument, Inscriptions, Blue Sky, Suffolk, East Suffolk. Wait, but the context says "East Suffolk" is a keyword hint, so include that. Let's count: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument, Inscriptions, Blue Sky, Suffolk, East Suffolk. That's 16. Let's check if all are visible or context. The context has "East Suffolk" as a keyword hint, so include it. The image shows Felixstowe, so "Felixstowe" is key. The memorial column is the main subject. The eagle statue is on top. The ocean, rocky shore, string lights, people walking, blue sky. So the keywords should be: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument,
&lt;/think&gt;

Title: Felixstowe Seafront Memorial Column
Description: A tall stone memorial column with an eagle statue stands at Felixstowe seafront, overlooking the ocean under a clear blue sky, with people walking along the rocky shore and colorful string lights.
Keywords: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument, Inscriptions, Blue Sky, Suffolk, East Suffolk
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this step by step. First, the title. The context says Felixstowe, England, UK, so we need a concise title. The main subject is the memorial column. Let's see: "Felixstowe Seafront Memorial Column" – that's 10 words, which is good.

Next, the description. Need to combine context with visible subject, setting, action, lighting, details. The image shows a seafront with a tall column monument, blue sky, people walking, rocks, a blue pole with lights. The capture date is 2026-08-13, but we don't claim that. The main subject is the memorial. So: "A tall stone memorial column with an eagle statue stands at Felixstowe seafront, overlooking the ocean under a clear blue sky, with people walking along the rocky shore and colorful string lights."

Wait, check the context: "Seafront, Felixstowe, England, UK, GBR, Europe" – so the description should include that. Also, the monument has inscriptions, the blue pole with lights. Let's make sure it's factual. The lighting is sunny, clear sky. The people are walking, rocks in the background. The monument is the central feature.

Now keywords. Need 10-18 unique, comma-separated. From context: Felixstowe, England, UK, GBR, Europe, Seafront, Suffolk, UK, gbr. Visible details: memorial column, eagle statue, blue sky, ocean, rocky shore, string lights, people walking, stone monument, inscriptions. Let's list: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument, Inscriptions, Blue Sky, Sunny Day.

Wait, need to check if they are unique. Let's pick 10-18. Let's see: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument, Inscriptions, Blue Sky, Sunny Day. That's 15. Let's check if any are redundant. Maybe "Suffolk" is in the context, but the image doesn't show Suffolk specifically, but the context has "East Suffolk" as a keyword hint. Wait, the context says "East Suffolk, England, UK, Europe" – but the image is Felixstowe, which is in Suffolk. So "East Suffolk" might be part of the context. Wait, the descriptive hints have "East Suffolk" as a keyword hint. So include that. Let's see: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument, Inscriptions, Blue Sky, Suffolk, East Suffolk. Wait, but the context says "East Suffolk" is a keyword hint, so include that. Let's count: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument, Inscriptions, Blue Sky, Suffolk, East Suffolk. That's 16. Let's check if all are visible or context. The context has "East Suffolk" as a keyword hint, so include it. The image shows Felixstowe, so "Felixstowe" is key. The memorial column is the main subject. The eagle statue is on top. The ocean, rocky shore, string lights, people walking, blue sky. So the keywords should be: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument,
</think>

Title: Felixstowe Seafront Memorial Column
Description: A tall stone memorial column with an eagle statue stands at Felixstowe seafront, overlooking the ocean under a clear blue sky, with people walking along the rocky shore and colorful string lights.
Keywords: Felixstowe, Seafront, England, UK, GBR, Europe, Memorial Column, Eagle Statue, Ocean, Rocky Shore, String Lights, People Walking, Stone Monument, Inscriptions, Blue Sky, Suffolk, East Suffolk
```

</details>

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
- *Model load time:* 3.29s
- *Generation time:* 86.24s
- *Total time:* 89.55s
- *Input validation time:* 6.7e-05
- *Prompt preparation time:* 0.015
- *First-token latency:* 78.7
- *Cleanup time:* 0.137
- *Prompt tokens:* 16,482
- *Generation tokens:* 117
- *Total tokens:* 16,599
- *Prompt throughput (raw):* 209 tok/s
- *Generation throughput (raw):* 17.5 tok/s
- *Peak memory:* 35
- *Active memory:* 29
- *Cache memory:* 1.8
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
Felixstowe War Memorial Column by the Seafront

Description:
A tall stone column topped with a bronze bird statue stands on a paved promenade overlooking the sea. The monument features inscribed plaques and is situated near a blue lamppost with festive lights, under a clear blue sky.

Keywords:
Felixstowe, Suffolk, England, UK, Europe, seafront, war memorial, monument, column, statue, bird, ocean, coast, promenade, lamppost, blue sky, stone, plaques, East Suffolk
</pre>

</details>

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### mlx-community/X-Reasoner-7B-8bit

<details>
<summary>Complete evidence: mlx-community/X-Reasoner-7B-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10); Keyword list has 37
  terms (requested 10-18); Duplicate keywords: felixstowe, east suffolk, uk,
  europe
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.28s
- *Generation time:* 23.81s
- *Total time:* 25.10s
- *Input validation time:* 8.68e-05
- *Prompt preparation time:* 0.000466
- *First-token latency:* 18.6
- *Cleanup time:* 0.12
- *Prompt tokens:* 16,478
- *Generation tokens:* 241
- *Total tokens:* 16,719
- *Prompt throughput (raw):* 885 tok/s
- *Generation throughput (raw):* 54.8 tok/s
- *Peak memory:* 13
- *Active memory:* 9.5
- *Cache memory:* 1.4
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0108
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
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
Felixstowe War Memorial, Seafront

Description:
A tall, ornate war memorial stands on the seafront at Felixstowe, East Suffolk, UK, with a bronze eagle perched atop, set against a clear blue sky and the calm sea. The structure bears inscriptions and plaques, marking its historical significance.

Keywords: Felixstowe, East Suffolk, England, UK, Europe, war memorial, seafront, bronze eagle, clear sky, calm sea, plaques, historical monument, East Suffolk, Felixstowe, Suffolk, UK, GBR, Europe, sea, blue sky, public space, memorial, 1939-1945, 1939-1945 plaques, publicWiFi, streetlight, decorative lights, stone steps, concrete base, sunny day, blue sea, clear horizon, publicWiFi sign, GPS coordinates, 2026-08-13 17:14:49 UTC+01:00, 51.959333°N, 1.349050°E.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Felixstowe War Memorial, Seafront

Description:
A tall, ornate war memorial stands on the seafront at Felixstowe, East Suffolk, UK, with a bronze eagle perched atop, set against a clear blue sky and the calm sea. The structure bears inscriptions and plaques, marking its historical significance. 

Keywords: Felixstowe, East Suffolk, England, UK, Europe, war memorial, seafront, bronze eagle, clear sky, calm sea, plaques, historical monument, East Suffolk, Felixstowe, Suffolk, UK, GBR, Europe, sea, blue sky, public space, memorial, 1939-1945, 1939-1945 plaques, publicWiFi, streetlight, decorative lights, stone steps, concrete base, sunny day, blue sea, clear horizon, publicWiFi sign, GPS coordinates, 2026-08-13 17:14:49 UTC+01:00, 51.959333°N, 1.349050°E.
```

</details>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible; Duplicate
  keywords: tourism
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.45s
- *Generation time:* 2.52s
- *Total time:* 5.99s
- *Input validation time:* 6.65e-05
- *Prompt preparation time:* 0.0208
- *First-token latency:* 0.389
- *Cleanup time:* 0.124
- *Prompt tokens:* 576
- *Generation tokens:* 82
- *Total tokens:* 658
- *Prompt throughput (raw):* 1,482 tok/s
- *Generation throughput (raw):* 54.6 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0108
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
&lt;channel|&gt;Title: War Memorial Column on the Felixstowe Seafront
Description: A tall, fluted stone column topped with an eagle statue stands on the coastal promenade overlooking the sea under a clear blue sky.
Keywords: Felixstowe, Suffolk, England, UK, coastal, seafront, memorial, column, eagle, ocean, horizon, monument, stone, sculpture, tourism, tourism
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: War Memorial Column on the Felixstowe Seafront
Description: A tall, fluted stone column topped with an eagle statue stands on the coastal promenade overlooking the sea under a clear blue sky.
Keywords: Felixstowe, Suffolk, England, UK, coastal, seafront, memorial, column, eagle, ocean, horizon, monument, stone, sculpture, tourism, tourism
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
- *Model load time:* 3.31s
- *Generation time:* 2.37s
- *Total time:* 5.69s
- *Input validation time:* 7.35e-05
- *Prompt preparation time:* 0.00035
- *First-token latency:* 0.329
- *Cleanup time:* 0.128
- *Prompt tokens:* 576
- *Generation tokens:* 79
- *Total tokens:* 655
- *Prompt throughput (raw):* 1,749 tok/s
- *Generation throughput (raw):* 56.6 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0108
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
&lt;channel|&gt;Title: War Memorial Column on the Felixstowe Seafront
Description: A tall fluted stone column topped with a bronze eagle statue stands along the English coast under a clear blue sky.
Keywords: Felixstowe, Suffolk, East Suffolk, England, UK, war memorial, column, eagle, seafront, ocean, stone, monument, memorial, coastal, sculpture
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: War Memorial Column on the Felixstowe Seafront
Description: A tall fluted stone column topped with a bronze eagle statue stands along the English coast under a clear blue sky.
Keywords: Felixstowe, Suffolk, East Suffolk, England, UK, war memorial, column, eagle, seafront, ocean, stone, monument, memorial, coastal, sculpture
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
- *Model load time:* 2.40s
- *Generation time:* 6.49s
- *Total time:* 8.90s
- *Input validation time:* 8.26e-05
- *Prompt preparation time:* 0.0027
- *First-token latency:* 1.11
- *Cleanup time:* 0.119
- *Prompt tokens:* 569
- *Generation tokens:* 139
- *Total tokens:* 708
- *Prompt throughput (raw):* 514 tok/s
- *Generation throughput (raw):* 28.3 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.43
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0113
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
Title: Felixstowe Seafront War Memorial, Suffolk

Description: Captured on 13th August 2026, this photograph shows the War Memorial on Felixstowe seafront, Suffolk, bathed in bright sunlight with the North Sea visible beyond. The memorial stands prominently on a stone plinth, flanked by flagpoles and string lights, with a sea wall and promenade in the foreground.

Keywords: Felixstowe, Suffolk, England, UK, GBR, Europe, seafront, war memorial, North Sea, coastal defence, stone plinth, flagpoles, string lights, promenade, East Suffolk, monument, inscription, 2026, August, daylight.
</pre>

</details>

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### Qwen/Qwen3-VL-2B-Instruct

<details>
<summary>Complete evidence: Qwen/Qwen3-VL-2B-Instruct</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Title has 3 words (requested 5-10); Keyword list has 330
  terms (requested 10-18); Duplicate keywords: seafront, memorial, sea,
  england, uk, europe, 1939 1945, war, commemoration, plaques, lamppost, blue,
  sky, stone, column, bronze, eagle
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.72s
- *Generation time:* 25.20s
- *Total time:* 25.93s
- *Input validation time:* 5.8e-05
- *Prompt preparation time:* 0.0091
- *First-token latency:* 13.7
- *Cleanup time:* 0.09
- *Prompt tokens:* 16,467
- *Generation tokens:* 1,000
- *Total tokens:* 17,467
- *Prompt throughput (raw):* 1,199 tok/s
- *Generation throughput (raw):* 93.1 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.4
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
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
Felixstowe Seafront Memorial

Description:
A stone memorial column stands at the seafront in Felixstowe, England, topped with a bronze eagle, commemorating the 1939-1945 war, with plaques listing names and a blue lamppost nearby under a clear blue sky.

Keywords: Felixstowe, seafront, memorial, war memorial, stone column, bronze eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, war, England, UK, Europe, sea, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue
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
- *Observations:* Response repeats the task instructions instead of only
  returning the requested fields; Extra text appears before the Title field;
  Response appears cut off at the token limit; Title has 4 words (requested
  5-10); Keyword list has 40 terms (requested 10-18); Duplicate keywords: east
  suffolk
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 2.72s
- *Generation time:* 26.80s
- *Total time:* 29.52s
- *Input validation time:* 0.000112
- *Prompt preparation time:* 0.00684
- *First-token latency:* 2.19
- *Cleanup time:* 0.097
- *Prompt tokens:* 3,520
- *Generation tokens:* 1,000
- *Total tokens:* 4,520
- *Prompt throughput (raw):* 1,604 tok/s
- *Generation throughput (raw):* 41.4 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.92
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
The user wants British-English catalogue metadata from the image and supplied context. They want:

- Title: a concrete 5-10-word title.
- Description: 1-2 sentences factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details.
- Keywords: 10-18 unique, comma-separated keywords covering relevant context and visible details.

We have the image: a tall column (a war memorial?) with a statue of an eagle at the top, on a seafront. There's a blue pole with decorative lights (string lights) overhead. The sea is visible, clear blue sky. There's a rocky breakwater. Two men are walking on a path near the column. The column is on a stone base with plaques. The column is a classical column with Corinthian capitals? Actually it's a tall column with a statue of an eagle on top, reminiscent of a war memorial. The base has inscriptions (likely names). The column is white, maybe marble. The location is Felixstowe, East Suffolk, England. The capture date/time is 2026-08-13 17:14:49 UTC+01:00. GPS: 51.959333°N, 1.349050°E.

We need to produce metadata in British English. Title: 5-10 words. Something like "Felixstowe Seafront War Memorial at Sunset". That's 6 words: Felixstowe (1) Seafront (2) War (3) Memorial (4) at (5) Sunset (6). Or "Felixstowe Seafront Column with Eagle". That's 5 words: Felixstowe (1) Seafront (2) Column (3) with (4) Eagle (5). But maybe include "Sunset" to capture lighting. The lighting appears to be late afternoon, maybe golden hour. The sky is clear blue, not deep orange. The sun is low? The shadows are long? The column casts a shadow on the ground. The lighting is bright, clear. So "Felixstowe Seafront Column at Dusk" (5 words). Or "Felixstowe Seafront War Memorial, Eagle Top". That's 6 words. Let's think.

The description: 1-2 sentences. Must combine relevant context (Felixstowe, East Suffolk, England, UK) with main visible subject (the column, the eagle statue, the seafront, the sea, the sky, the string lights, the two men). Setting: seafront, rocky breakwater, clear sky. Action: two men walking, maybe a tourist. Lighting: bright, clear, late afternoon. Distinctive details: column with Corinthian capital, eagle statue, plaques, string lights overhead, blue pole, rocky shoreline.

We need to be factual, not speculation. So we can say: "The image shows the Felixstowe seafront war memorial, a tall white column topped by a bronze eagle, standing on a stone plinth with inscribed plaques, with the sea and a clear blue sky in the background. Two men walk along the promenade beside a blue streetlamp adorned with decorative lights." That's two sentences. That's fine.

Keywords: 10-18 unique, comma-separated. Must cover relevant context and visible details. So we can include: Felixstowe, Seafront, War memorial, Column, Eagle statue, Stone plinth, Inscriptions, Sea, Sky, Blue pole, Decorative lights, Rocky breakwater, Promenade, East Suffolk, England, United Kingdom, Europe, Adobe Stock, Any Vision. That's 18? Let's count: Felixstowe (1), Seafront (2), War memorial (3), Column (4), Eagle statue (5), Stone plinth (6), Inscriptions (7), Sea (8), Sky (9), Blue pole (10), Decorative lights (11), Rocky breakwater (12), Promenade (13), East Suffolk (14), England (15), United Kingdom (16), Europe (17), Adobe Stock (18), Any Vision (19). That's 19, too many. We need 10-18. So we need to limit to 10-18. Let's pick 14 maybe. Let's list: Felixstowe, Seafront, War memorial, Column, Eagle statue, Stone plinth, Inscriptions, Sea, Sky, Decorative lights, Rocky breakwater, Promenade, East Suffolk, United Kingdom. That's 14. That
</pre>

</details>

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Response appears cut off at the token limit; Keyword list
  has 52 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 5.15s
- *Generation time:* 19.27s
- *Total time:* 24.43s
- *Input validation time:* 7.68e-05
- *Prompt preparation time:* 0.00849
- *First-token latency:* 1.6
- *Cleanup time:* 0.191
- *Prompt tokens:* 1,584
- *Generation tokens:* 1,000
- *Total tokens:* 2,584
- *Prompt throughput (raw):* 991 tok/s
- *Generation throughput (raw):* 59.1 tok/s
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
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 800,
  "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Alright, let's get to work. I've got an image of a seaside scene, and I need to generate some metadata for it. First things first, I need to break down the request. The core task is to create a title, description, and keywords for this image, all in British English. The image itself is a seaside view, and I have some hints to work with, but I need to be careful. The hints are a bit jumbled, so I'll have to sift through them.

The hints mention "Albion," "St. Mary's," and "Suffolk," which is a good starting point. The GPS coordinates are crucial, and the date/time are important too. I'll use the GPS to pinpoint the location. The hints are a bit of a mess, but I can see the key elements: a seaside view, a statue, and some people.

Now, let's analyze the image itself. It's a seaside view, and the main subject is a statue. It's a tall, white, classical-style monument, and it has a statue on top. The base is made of stone, and there's a plaque. The setting is a beach or promenade, with the sea in the background. The lighting is bright, so it's probably daytime. There are a couple of people in the foreground.

Okay, time to synthesize. The location is definitely Felixstowe, Suffolk, England. The statue is the focal point. The description needs to be concise and factual. I'll start with the location and the main subject. I'll mention the sea, the statue, and the people. I'll also include the date and time.

The title needs to be short and descriptive. I'll go with "Felixstowe Seafront with Altar of Peace Monument."

The description will be a bit longer. I'll start with the location and date, then describe the scene. I'll mention the statue, the sea, and the people.

Finally, the keywords. I need to cover the location, the subject, the setting, and the details. I'll include "Felixstowe," "Suffolk," "England," "UK," "GBR," "seafront," "beach," "monument," "statue," "altar," "peace," "memorial," "people," "walkers," "sunny day," "bright," "blue sky," "sea," "ocean," "coast," "cliff," "rock," "stone," "plaque," "bronze," "golden," "sunlight," "shadow," "summer," "August," "2026," "pavilion," "pier," "harbour," "harbor," "promenade," "walkway," "path," "grass," "rocky," "sandy," "blue," "clear," "calm," "peaceful," "serene," "scenic," "landscape," "view," "outdoor," "recreation," "leisure," "tourism," "history," "commemorative," "memorial," "war," "remembrance," "remembrance day," "memorial day," "Albion," "St. Mary's," "East Suffolk," "Europe," "GBR," "UK," "England," "Suffolk," "Felixstowe," "Suffolk," "UK," "GBR," "Europe," "seafront," "beach," "monument," "statue," "altar," "peace," "memorial," "people," "walkers," "sunny day," "bright,"
&lt;/think&gt;

Title: Felixstowe Seafront with Altar of Peace Monument

Description: A sunny day in August 2026 at the Felixstowe seafront in Suffolk, England, UK. A tall white classical monument with a golden eagle statue stands on a rocky shore, with a few people walking nearby.

Keywords: Felixstowe, Suffolk, England, UK, GBR, Europe, seafront, beach, monument, statue, altar, peace, memorial, people, walkers, sunny day, bright, blue sky, sea, ocean, coast, cliff, rock, stone, plaque, bronze, golden, sunlight, shadow, summer, August, 2026, pavilion, pier, harbour, harbor, promenade, walkway, path, grass, rocky, sandy, blue, clear, calm, peaceful, serene, scenic, landscape, view, outdoor, recreation,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get to work. I've got an image of a seaside scene, and I need to generate some metadata for it. First things first, I need to break down the request. The core task is to create a title, description, and keywords for this image, all in British English. The image itself is a seaside view, and I have some hints to work with, but I need to be careful. The hints are a bit jumbled, so I'll have to sift through them.

The hints mention "Albion," "St. Mary's," and "Suffolk," which is a good starting point. The GPS coordinates are crucial, and the date/time are important too. I'll use the GPS to pinpoint the location. The hints are a bit of a mess, but I can see the key elements: a seaside view, a statue, and some people.

Now, let's analyze the image itself. It's a seaside view, and the main subject is a statue. It's a tall, white, classical-style monument, and it has a statue on top. The base is made of stone, and there's a plaque. The setting is a beach or promenade, with the sea in the background. The lighting is bright, so it's probably daytime. There are a couple of people in the foreground.

Okay, time to synthesize. The location is definitely Felixstowe, Suffolk, England. The statue is the focal point. The description needs to be concise and factual. I'll start with the location and the main subject. I'll mention the sea, the statue, and the people. I'll also include the date and time.

The title needs to be short and descriptive. I'll go with "Felixstowe Seafront with Altar of Peace Monument."

The description will be a bit longer. I'll start with the location and date, then describe the scene. I'll mention the statue, the sea, and the people.

Finally, the keywords. I need to cover the location, the subject, the setting, and the details. I'll include "Felixstowe," "Suffolk," "England," "UK," "GBR," "seafront," "beach," "monument," "statue," "altar," "peace," "memorial," "people," "walkers," "sunny day," "bright," "blue sky," "sea," "ocean," "coast," "cliff," "rock," "stone," "plaque," "bronze," "golden," "sunlight," "shadow," "summer," "August," "2026," "pavilion," "pier," "harbour," "harbor," "promenade," "walkway," "path," "grass," "rocky," "sandy," "blue," "clear," "calm," "peaceful," "serene," "scenic," "landscape," "view," "outdoor," "recreation," "leisure," "tourism," "history," "commemorative," "memorial," "war," "remembrance," "remembrance day," "memorial day," "Albion," "St. Mary's," "East Suffolk," "Europe," "GBR," "UK," "England," "Suffolk," "Felixstowe," "Suffolk," "UK," "GBR," "Europe," "seafront," "beach," "monument," "statue," "altar," "peace," "memorial," "people," "walkers," "sunny day," "bright,"
</think>

Title: Felixstowe Seafront with Altar of Peace Monument

Description: A sunny day in August 2026 at the Felixstowe seafront in Suffolk, England, UK. A tall white classical monument with a golden eagle statue stands on a rocky shore, with a few people walking nearby.

Keywords: Felixstowe, Suffolk, England, UK, GBR, Europe, seafront, beach, monument, statue, altar, peace, memorial, people, walkers, sunny day, bright, blue sky, sea, ocean, coast, cliff, rock, stone, plaque, bronze, golden, sunlight, shadow, summer, August, 2026, pavilion, pier, harbour, harbor, promenade, walkway, path, grass, rocky, sandy, blue, clear, calm, peaceful, serene, scenic, landscape, view, outdoor, recreation,
```

</details>

</details>

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### mlx-community/FastVLM-0.5B-bf16

<details>
<summary>Complete evidence: mlx-community/FastVLM-0.5B-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Keywords; Response repeats
  the task instructions instead of only returning the requested fields; Extra
  text appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type llava_qwen2 via
  fastvlm)
- *Model load time:* 0.82s
- *Generation time:* 1.44s
- *Total time:* 2.27s
- *Input validation time:* 8.45e-05
- *Prompt preparation time:* 0.00173
- *First-token latency:* 0.117
- *Cleanup time:* 0.0779
- *Prompt tokens:* 308
- *Generation tokens:* 147
- *Total tokens:* 455
- *Prompt throughput (raw):* 2,631 tok/s
- *Generation throughput (raw):* 347 tok/s
- *Peak memory:* 2.0
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
A concrete 5-10-word title: "Felixstowe, UK, Seafront, GBR, East Suffolk, England, Europe, GBR, Seafront"
Description: The image features a prominent 5-10-word title describing a concrete 5-10-word description of a seafront in Felixstowe, UK, with a GBR (Great Britain) location, East Suffolk, England, and a GBR (Great Britain) context. The setting is a seafront with a 5-10-word title, a 1-2-sentence description of the main subject, and a 10-18 unique, comma-separated keywords covering relevant context and visible details.
</pre>

</details>

---

<a id="model-mlx-community-glm-41v-9b-thinking-8bit"></a>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

<details>
<summary>Complete evidence: mlx-community/GLM-4.1V-9B-Thinking-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title; Extra text appears before
  the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.74s
- *Generation time:* 26.72s
- *Total time:* 28.47s
- *Input validation time:* 7.42e-05
- *Prompt preparation time:* 0.00188
- *First-token latency:* 5.5
- *Cleanup time:* 0.095
- *Prompt tokens:* 6,307
- *Generation tokens:* 994
- *Total tokens:* 7,301
- *Prompt throughput (raw):* 1,146 tok/s
- *Generation throughput (raw):* 47.8 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.48
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
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
&lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. The image shows a monument by the sea, so maybe "Felixstowe Seafront Monument". Then the description should combine context (date, location) with the main subject (monument), setting (seafront), action (people walking), lighting (sunny), distinctive details (column, eagle statue, sea). Let's make sure it's factual. Then keywords need to cover context and visible details, like location, monument type, etc.

Title: Felixstowe Seafront Monument
Description: A monument with a column and eagle statue stands on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea.
Keywords: Felixstowe, seafront, monument, column, eagle statue, East Suffolk, England, UK, GBR, Europe, sea, coastal, memorial, stone structure, coastal walkway, sunny day, people walking, coastal rocks, memorial plaque

Wait, need to check the hints. The title hint was Seafront, Felixstowe... so maybe "Felixstowe Seafront Memorial" or similar. Let's make sure the description includes the capture date/time? Wait, the context says capture date/time is 2026-08-13 17:14:49 UTC+01:00, but maybe include that? Wait, the description should combine context with main subject, setting, action, lighting, distinctive details. So maybe: "A monument with a column and eagle statue is located on Felixstowe's seafront, with people walking nearby under clear, sunny lighting, overlooking the sea, captured on 2026-08-13."

Wait, let's structure:

Title: Felixstowe Seafront Monument
Description: A monument with a column and eagle statue stands on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea, captured on 2026-08-13.
Keywords: Felixstowe, seafront, monument, column, eagle statue, East Suffolk, England, UK, GBR, Europe, sea, coastal, memorial, stone structure, coastal rocks, sunny day, people walking, memorial plaque, coastal promenade

Wait, need to make sure keywords are unique, comma-separated, 10-18. Let's check the hints: keyword hints include Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. So include those. Also visible details: column, eagle statue, sea, rocks, people, lighting.

Finalizing:

Title: Felixstowe Seafront Monument
Description: A monument with a column and eagle statue is situated on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea, captured on 2026-08-13.
Keywords: Felixstowe, seafront, monument, column, eagle statue, East Suffolk, England, UK, GBR, Europe, sea, coastal, memorial, stone structure, coastal rocks, sunny day, people walking, memorial plaque, coastal promenade

Wait, check the length. Title is 5 words (Felixstowe Seafront Monument) – 5 words, good. Description is 2 sentences? Wait, the user said 1-2 sentence. Let's make it one sentence. "A monument with a column and eagle statue stands on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea, captured on 2026-08-13." That's one sentence. Then keywords need 10-18. Let's count: Felixstowe, seafront, monument, column, eagle statue, East Suffolk, England, UK, GBR, Europe, sea, coastal, memorial, stone structure, coastal rocks, sunny day, people walking – that's 16, which is good.

Wait, the context says "capture date/time" is authoritative, so include that in description. So description: "A monument with a column and eagle statue stands on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea, captured on 2026-08-13 UTC+01:00."

Yes, that's better. Then keywords include all the hints and visible details.&lt;/think&gt;&lt;answer&gt;Title: Felixstowe Seafront Monument
Description: A monument with a column and eagle statue stands on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea, captured on 2026-08-13 UTC+01:00.
Keywords: Felixstowe, seafront, monument, column, eagle statue, East Suffolk, England, UK, GBR, Europe, sea, coastal, memorial, stone structure, coastal rocks, sunny day, people walking, memorial plaque, coastal promenade
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words. The image shows a monument by the sea, so maybe "Felixstowe Seafront Monument". Then the description should combine context (date, location) with the main subject (monument), setting (seafront), action (people walking), lighting (sunny), distinctive details (column, eagle statue, sea). Let's make sure it's factual. Then keywords need to cover context and visible details, like location, monument type, etc.

Title: Felixstowe Seafront Monument
Description: A monument with a column and eagle statue stands on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea.
Keywords: Felixstowe, seafront, monument, column, eagle statue, East Suffolk, England, UK, GBR, Europe, sea, coastal, memorial, stone structure, coastal walkway, sunny day, people walking, coastal rocks, memorial plaque

Wait, need to check the hints. The title hint was Seafront, Felixstowe... so maybe "Felixstowe Seafront Memorial" or similar. Let's make sure the description includes the capture date/time? Wait, the context says capture date/time is 2026-08-13 17:14:49 UTC+01:00, but maybe include that? Wait, the description should combine context with main subject, setting, action, lighting, distinctive details. So maybe: "A monument with a column and eagle statue is located on Felixstowe's seafront, with people walking nearby under clear, sunny lighting, overlooking the sea, captured on 2026-08-13."

Wait, let's structure:

Title: Felixstowe Seafront Monument
Description: A monument with a column and eagle statue stands on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea, captured on 2026-08-13.
Keywords: Felixstowe, seafront, monument, column, eagle statue, East Suffolk, England, UK, GBR, Europe, sea, coastal, memorial, stone structure, coastal rocks, sunny day, people walking, memorial plaque, coastal promenade

Wait, need to make sure keywords are unique, comma-separated, 10-18. Let's check the hints: keyword hints include Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. So include those. Also visible details: column, eagle statue, sea, rocks, people, lighting.

Finalizing:

Title: Felixstowe Seafront Monument
Description: A monument with a column and eagle statue is situated on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea, captured on 2026-08-13.
Keywords: Felixstowe, seafront, monument, column, eagle statue, East Suffolk, England, UK, GBR, Europe, sea, coastal, memorial, stone structure, coastal rocks, sunny day, people walking, memorial plaque, coastal promenade

Wait, check the length. Title is 5 words (Felixstowe Seafront Monument) – 5 words, good. Description is 2 sentences? Wait, the user said 1-2 sentence. Let's make it one sentence. "A monument with a column and eagle statue stands on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea, captured on 2026-08-13." That's one sentence. Then keywords need 10-18. Let's count: Felixstowe, seafront, monument, column, eagle statue, East Suffolk, England, UK, GBR, Europe, sea, coastal, memorial, stone structure, coastal rocks, sunny day, people walking – that's 16, which is good.

Wait, the context says "capture date/time" is authoritative, so include that in description. So description: "A monument with a column and eagle statue stands on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea, captured on 2026-08-13 UTC+01:00."

Yes, that's better. Then keywords include all the hints and visible details.</think><answer>Title: Felixstowe Seafront Monument
Description: A monument with a column and eagle statue stands on Felixstowe's seafront, with people walking nearby under sunny lighting, overlooking the sea, captured on 2026-08-13 UTC+01:00.
Keywords: Felixstowe, seafront, monument, column, eagle statue, East Suffolk, England, UK, GBR, Europe, sea, coastal, memorial, stone structure, coastal rocks, sunny day, people walking, memorial plaque, coastal promenade
```

</details>

</details>

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### mlx-community/GLM-4.6V-Flash-mxfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-mxfp4</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Title has 3 words (requested 5-10); Keyword list has 382
  terms (requested 10-18); Duplicate keywords: seafront, uk, gbr, europe, war
  memorial, stone column, eagle statue, sea, coastal, memorial, remembrance,
  war, suffolk county
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.49s
- *Generation time:* 18.44s
- *Total time:* 19.95s
- *Input validation time:* 6.89e-05
- *Prompt preparation time:* 0.00696
- *First-token latency:* 5.24
- *Cleanup time:* 0.104
- *Prompt tokens:* 6,311
- *Generation tokens:* 1,000
- *Total tokens:* 7,311
- *Prompt throughput (raw):* 1,204 tok/s
- *Generation throughput (raw):* 78.2 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.44
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
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
Title: Felixstowe War Memorial
Description: The Felixstowe War Memorial, a tall stone column topped with an eagle, stands by the seafront in Felixstowe, England.
Keywords: Felixstowe War Memorial, seafront, Felixstowe, Suffolk, England, UK, gbr, Europe, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Felixstowe War Memorial
Description: The Felixstowe War Memorial, a tall stone column topped with an eagle, stands by the seafront in Felixstowe, England. 
Keywords: Felixstowe War Memorial, seafront, Felixstowe, Suffolk, England, UK, gbr, Europe, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK, gbr, Europe, seafront, coastal, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, Suffolk County, UK,
```

</details>

</details>

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

<details>
<summary>Complete evidence: mlx-community/Llama-3.2-11B-Vision-Instruct-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Title has 4 words (requested 5-10); Keyword list has 306
  terms (requested 10-18); Duplicate keywords: historical landmark, historical
  significance, cultural icon, historical icon
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 1.62s
- *Generation time:* 50.09s
- *Total time:* 51.71s
- *Input validation time:* 5.9e-05
- *Prompt preparation time:* 0.00155
- *First-token latency:* 1.43
- *Cleanup time:* 0.0952
- *Prompt tokens:* 275
- *Generation tokens:* 1,000
- *Total tokens:* 1,275
- *Prompt throughput (raw):* 192 tok/s
- *Generation throughput (raw):* 20.7 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.76
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00431
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
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
Title:
Felixstowe Seafront Memorial Column

Description:
The image features a prominent memorial column situated on the seafront in Felixstowe, England, with a columnar structure and a bird statue at the top, set against a clear blue sky and the calm sea in the background.

Keywords:
Felixstowe, England, UK, Seafront, Memorial, Column, Bird Statue, Blue Sky, Calm Sea, Coastal, Seaside, Public Monument, Historical Landmark, Stone Structure, Ornate Details, Seafront Promenade, Coastal Town, British Architecture, Coastal Scenery, Seaside Atmosphere, Peaceful Scene, Memorial Day, Remembrance, Commemoration, Historical Significance, Cultural Heritage, Architectural Landmark, Iconic Structure, Seafront Landmark, Coastal Icon, British History, Historical Monument, Cultural Significance, Architectural Style, Historical Landmark, Cultural Icon, British Culture, Historical Significance, Architectural Icon, Cultural Landmark, Historical Icon, Cultural Monument, Architectural Heritage, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: 
Felixstowe Seafront Memorial Column

Description:
The image features a prominent memorial column situated on the seafront in Felixstowe, England, with a columnar structure and a bird statue at the top, set against a clear blue sky and the calm sea in the background.

Keywords:
Felixstowe, England, UK, Seafront, Memorial, Column, Bird Statue, Blue Sky, Calm Sea, Coastal, Seaside, Public Monument, Historical Landmark, Stone Structure, Ornate Details, Seafront Promenade, Coastal Town, British Architecture, Coastal Scenery, Seaside Atmosphere, Peaceful Scene, Memorial Day, Remembrance, Commemoration, Historical Significance, Cultural Heritage, Architectural Landmark, Iconic Structure, Seafront Landmark, Coastal Icon, British History, Historical Monument, Cultural Significance, Architectural Style, Historical Landmark, Cultural Icon, British Culture, Historical Significance, Architectural Icon, Cultural Landmark, Historical Icon, Cultural Monument, Architectural Heritage, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon,
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
- *Model load time:* 2.45s
- *Generation time:* 26.81s
- *Total time:* 29.27s
- *Input validation time:* 7e-05
- *Prompt preparation time:* 0.00594
- *First-token latency:* 2.43
- *Cleanup time:* 0.103
- *Prompt tokens:* 3,119
- *Generation tokens:* 142
- *Total tokens:* 3,261
- *Prompt throughput (raw):* 1,285 tok/s
- *Generation throughput (raw):* 6.01 tok/s
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
Felixstowe War Memorial, East Suffolk Coast

The image shows a war memorial monument on a rocky seafront in Felixstowe, England. A tall white column with an eagle statue stands prominently, surrounded by a stone base with plaques. The memorial overlooks the ocean, with a clear blue sky and a few people visible in the distance. The scene captures a peaceful seaside setting with the memorial as the focal point.

Felixstowe, England, UK, GBR, East Suffolk, seafront, war memorial, white column, eagle statue, stone base, plaques, rocky coast, ocean, clear sky, people, memorial, monument, seaside, Suffolk, UK, gbr
</pre>

</details>

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Keyword list has 259 terms (requested 10-18); Duplicate
  keywords: stone column, bird statue, people walking, clear sky, calm sea,
  stone pathway, landmark, scenic beauty, seaside town
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.61s
- *Generation time:* 67.45s
- *Total time:* 68.07s
- *Input validation time:* 6.79e-05
- *Prompt preparation time:* 0.00356
- *First-token latency:* 62.2
- *Cleanup time:* 0.085
- *Prompt tokens:* 16,478
- *Generation tokens:* 1,000
- *Total tokens:* 17,478
- *Prompt throughput (raw):* 265 tok/s
- *Generation throughput (raw):* 223 tok/s
- *Peak memory:* 5.1
- *Active memory:* 1.3
- *Cache memory:* 0.83
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.00744
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
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
The Seafront of Felixstowe, England, UK, GBR, Europe, features a prominent seafront monument with a stone column and a bird statue on top. The monument is situated near the sea, with a clear blue sky and a calm sea in the background. The monument is surrounded by a stone pathway and a few people walking nearby. The monument is a notable landmark in the area, and its presence adds to the scenic beauty of the seaside town.

Keywords:
Seafront, Felixstowe, England, UK, GBR, Europe, Monument, Stone Column, Bird Statue, People Walking, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### mlx-community/Qwen3-VL-2B-Instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Title has 3 words (requested 5-10); Keyword list has 330
  terms (requested 10-18); Duplicate keywords: seafront, memorial, sea,
  england, uk, europe, 1939 1945, war, commemoration, plaques, lamppost, blue,
  sky, stone, column, bronze, eagle
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.92s
- *Generation time:* 28.01s
- *Total time:* 28.94s
- *Input validation time:* 5.97e-05
- *Prompt preparation time:* 0.000316
- *First-token latency:* 16.2
- *Cleanup time:* 0.0926
- *Prompt tokens:* 16,467
- *Generation tokens:* 1,000
- *Total tokens:* 17,467
- *Prompt throughput (raw):* 1,017 tok/s
- *Generation throughput (raw):* 91.0 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.4
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00775
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
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
Felixstowe Seafront Memorial

Description:
A stone memorial column stands at the seafront in Felixstowe, England, topped with a bronze eagle, commemorating the 1939-1945 war, with plaques listing names and a blue lamppost nearby under a clear blue sky.

Keywords: Felixstowe, seafront, memorial, war memorial, stone column, bronze eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, war, England, UK, Europe, sea, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue
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
- *Model load time:* 2.47s
- *Generation time:* 4.16s
- *Total time:* 6.64s
- *Input validation time:* 8.03e-05
- *Prompt preparation time:* 0.00348
- *First-token latency:* 0.378
- *Cleanup time:* 0.12
- *Prompt tokens:* 568
- *Generation tokens:* 162
- *Total tokens:* 730
- *Prompt throughput (raw):* 1,502 tok/s
- *Generation throughput (raw):* 48.5 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0119
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
A tall, light-colored stone monument stands prominently on a rocky coastline under a clear blue sky. The monument features a fluted column topped with a bronze eagle with outstretched wings. The base of the monument is multi-tiered and adorned with several dark plaques bearing inscriptions.

The monument is situated near a paved walkway and a rocky shoreline. In the background, the calm, light blue sea stretches out to the horizon. A few people are visible walking along the walkway to the left of the monument. Overhead, strings of colorful lights are strung between poles, adding a festive touch to the scene. A blue lamppost with decorative metalwork stands to the right of the monument.

The overall impression is one of a historical landmark in a coastal setting, captured on a bright, sunny day.
</pre>

<details>
<summary>Exact raw output</summary>

```text
A tall, light-colored stone monument stands prominently on a rocky coastline under a clear blue sky. The monument features a fluted column topped with a bronze eagle with outstretched wings. The base of the monument is multi-tiered and adorned with several dark plaques bearing inscriptions. 

The monument is situated near a paved walkway and a rocky shoreline. In the background, the calm, light blue sea stretches out to the horizon. A few people are visible walking along the walkway to the left of the monument. Overhead, strings of colorful lights are strung between poles, adding a festive touch to the scene. A blue lamppost with decorative metalwork stands to the right of the monument. 

The overall impression is one of a historical landmark in a coastal setting, captured on a bright, sunny day.
```

</details>

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
- *Model load time:* 1.07s
- *Generation time:* 3.25s
- *Total time:* 4.32s
- *Input validation time:* 6.93e-05
- *Prompt preparation time:* 0.00186
- *First-token latency:* 2.32
- *Cleanup time:* 0.0906
- *Prompt tokens:* 2,692
- *Generation tokens:* 18
- *Total tokens:* 2,710
- *Prompt throughput (raw):* 1,161 tok/s
- *Generation throughput (raw):* 64.4 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.87
- *Model-load active memory:* 8.05
- *Post-cleanup active memory:* 0.013
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
Seafront, Felixstowe, England, UK, GBR, Europe,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Seafront, Felixstowe, England, UK, GBR, Europe, 
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
- *Model load time:* 0.68s
- *Generation time:* 0.98s
- *Total time:* 1.67s
- *Input validation time:* 5.95e-05
- *Prompt preparation time:* 0.00169
- *First-token latency:* 0.0735
- *Cleanup time:* 0.0923
- *Prompt tokens:* 304
- *Generation tokens:* 151
- *Total tokens:* 455
- *Prompt throughput (raw):* 4,137 tok/s
- *Generation throughput (raw):* 369 tok/s
- *Peak memory:* 2.4
- *Active memory:* 0.61
- *Cache memory:* 0.30
- *Model-load active memory:* 0.612
- *Post-cleanup active memory:* 0.0136
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
Description: A concrete 5-10-word title, followed by a 1-2-sentence factual description that combines relevant context with the main visible subject, setting, action, lighting, and distinctive details. The description includes the title, the geographic location, the GBR code, and the name of the location. The keywords are provided in a comma-separated format, with the GBR code included in the second line. The context is provided in the third line, with the location being Felixstowe, England, UK, GBR, and the context being a photograph of the seafront in Felixstowe, England, UK, GBR.
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
- *Model load time:* 1.25s
- *Generation time:* 22.69s
- *Total time:* 23.95s
- *Input validation time:* 5.79e-05
- *Prompt preparation time:* 0.0002
- *First-token latency:* 1.13
- *Cleanup time:* 0.12
- *Prompt tokens:* 4,397
- *Generation tokens:* 1,000
- *Total tokens:* 5,397
- *Prompt throughput (raw):* 3,878 tok/s
- *Generation throughput (raw):* 48.0 tok/s
- *Peak memory:* 4.4
- *Active memory:* 1.7
- *Cache memory:* 2.1
- *Model-load active memory:* 1.73
- *Post-cleanup active memory:* 0.0146
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
- Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, GBR,
- Description hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, England, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, GBR
- Title hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felix
</pre>

</details>

---
