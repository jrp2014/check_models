# Model Output Gallery

Generated on: 2026-08-16 17:50:02 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS             | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                                 |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|---------------------|-----------------|---------|------------|---------|--------------------------------------------------------------------------------------------------------------|
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable`              | 1.36s   | 138 tok/s           | 0.09            | 5.2     | 409        | 82      | none                                                                                                         |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 4.60s   | 34.4 tok/s          | 1.20            | 17      | 3,435      | 60      | none                                                                                                         |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | 1.23s   | 185 tok/s           | 0.07            | 4.1     | 350        | 100     | none                                                                                                         |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 3.21s   | 69.5 tok/s          | 0.67            | 9.8     | 1,070      | 81      | none                                                                                                         |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              | 3.76s   | 64.6 tok/s          | 0.76            | 10      | 1,070      | 106     | none                                                                                                         |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 1.84s   | 193 tok/s           | 0.28            | 4.5     | 1,069      | 85      | none                                                                                                         |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable`              | 3.32s   | 53.7 tok/s          | 0.43            | 11      | 843        | 89      | none                                                                                                         |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `usable`              | 15.80s  | 6.21 tok/s          | 0.89            | 23      | 860        | 78      | none                                                                                                         |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | 2.36s   | 59.6 tok/s          | 0.21            | 9.3     | 883        | 68      | none                                                                                                         |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable`              | 1.48s   | 129 tok/s           | 0.09            | 5.3     | 409        | 82      | none                                                                                                         |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 3.60s   | 121 tok/s           | 0.35            | 21      | 421        | 89      | none                                                                                                         |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 2.53s   | 101 tok/s           | 0.28            | 7.1     | 421        | 86      | none                                                                                                         |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 9.52s   | 19.2 tok/s          | 0.84            | 30      | 421        | 107     | none                                                                                                         |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | 13.23s  | 44.2 tok/s          | 3.77            | 65      | 286        | 105     | none                                                                                                         |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable`              | 3.11s   | 66.1 tok/s          | 0.27            | 10      | 511        | 106     | none                                                                                                         |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | 5.91s   | 31.6 tok/s          | 0.76            | 18      | 379        | 90      | none                                                                                                         |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 3.28s   | 130 tok/s           | 0.37            | 16      | 390        | 73      | none                                                                                                         |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 6.28s   | 28.1 tok/s          | 0.83            | 19      | 390        | 80      | none                                                                                                         |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | 4.89s   | 40.5 tok/s          | 0.98            | 15      | 1,335      | 89      | none                                                                                                         |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable_with_caveats` | 0.64s   | 516 tok/s           | 0.08            | 1.3     | 350        | 63      | title/keyword constraints failed                                                                             |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | 6.33s   | 31.2 tok/s          | 0.98            | 20      | 537        | 100     | title/keyword constraints failed                                                                             |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | 2.33s   | 92.0 tok/s          | 0.32            | 7.8     | 499        | 61      | title/keyword constraints failed                                                                             |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | 13.99s  | 52.6 tok/s          | 4.69            | 63      | 499        | 89      | control tokens visible                                                                                       |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | 121.10s | 4.83 tok/s          | 1.17            | 40      | 489        | 562     | role tokens visible                                                                                          |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable_with_caveats` | 18.97s  | 66.2 tok/s          | 5.49            | 71      | 421        | 133     | title/keyword constraints failed                                                                             |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | 0.83s   | 307 tok/s           | 0.15            | 2.5     | 511        | 40      | title/keyword constraints failed                                                                             |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | 1.34s   | 125 tok/s           | 0.38            | 5.5     | 1,204      | 31      | title/keyword constraints failed                                                                             |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | 4.92s   | 60.8 tok/s          | 0.31            | 29      | 386        | 72      | control tokens visible; title/keyword constraints failed                                                     |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | 4.90s   | 59.3 tok/s          | 0.30            | 28      | 386        | 74      | control tokens visible                                                                                       |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `usable_with_caveats` | 4.08s   | 47.8 tok/s          | 0.35            | 17      | 378        | 63      | title/keyword constraints failed                                                                             |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 25.25s  | 44.0 tok/s          | 0.96            | 14      | 1,426      | 1,000   | missing required fields; echoes instructions; extra text before Title; cut off at token limit                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 27.62s  | 57.9 tok/s          | 2.68            | 60      | 545        | 944     | repeated text; extra text before Title; title/keyword constraints failed                                     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 1.03s   | 343 tok/s           | 0.30            | 2.2     | 120        | 32      | missing required fields                                                                                      |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 21.89s  | 50.4 tok/s          | 0.37            | 13      | 495        | 1,000   | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `unusable`            | 3.37s   | insufficient sample | 1.04            | 18      | 2,418      | 13      | missing required fields; role tokens visible                                                                 |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `unusable`            | 65.99s  | 15.9 tok/s          | 1.64            | 15      | 107        | 1,000   | repeated text; extra text before Title; cut off at token limit; title/keyword constraints failed             |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 1.22s   | 278 tok/s           | 0.10            | 3.0     | 330        | 58      | missing required fields; extra text before Title                                                             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | 7.62s   | 130 tok/s           | 0.08            | 5.3     | 411        | 872     | echoes instructions; extra text before Title                                                                 |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | 21.33s  | 54.7 tok/s          | 2.11            | 9.7     | 2,469      | 1,000   | repeated text; missing required fields; cut off at token limit                                               |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 0.83s   | 381 tok/s           | 0.07            | 2.1     | 116        | 97      | missing required fields                                                                                      |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 23.79s  | 47.0 tok/s          | 1.10            | 4.2     | 4,207      | 1,000   | repeated text; missing required fields; echoes instructions; extra text before Title; cut off at token limit |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `mlx-community/Ministral-3-3B-Instruct-2512-4bit` at 193 tok/s

Average clean-completion throughput: 79.7 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `mlx-community/LFM2.5-VL-1.6B-bf16` at 4.1 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Usability  | Observations                                                                                                 |
|--------------------------------------------------------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable` | missing required fields; echoes instructions; extra text before Title; cut off at token limit                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable` | repeated text; extra text before Title; title/keyword constraints failed                                     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable` | missing required fields                                                                                      |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable` | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                      | `unusable` | missing required fields; role tokens visible                                                                 |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit) | `unusable` | repeated text; extra text before Title; cut off at token limit; title/keyword constraints failed             |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                 | `unusable` | missing required fields; extra text before Title                                                             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable` | echoes instructions; extra text before Title                                                                 |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable` | repeated text; missing required fields; cut off at token limit                                               |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable` | missing required fields                                                                                      |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable` | repeated text; missing required fields; echoes instructions; extra text before Title; cut off at token limit |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                                                      |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable`              | Title: Two cats on a pink couch<br><br>Description: Two tabby cats are lying on a bright pink couch, sleeping peacefully with remote controls beside them. The scene is lit with soft, natural light, highlighting the cats' relaxed postures and the cozy setting.<br><br>Keywords: cats, couch,...                                                |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | Title: Two Cats Sleeping on Pink Blanket<br>Description: Two tabby cats are lying on a pink blanket with remote controls nearby, in a relaxed sleeping position.<br>Keywords: cats, sleeping, pink blanket, remote controls, tabby, relaxed, blanket, resting, pets, cozy, indoor                                                                   |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | Title: Two Cats Resting on a Pink Blanket<br><br>Description: Two cats are lying on a pink blanket, one is stretched out while the other is curled up. The cats appear to be sleeping peacefully, with one having a green collar. The lighting is soft and warm, highlighting the relaxed p...                                                      |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>Two Cats Relaxing on Pink Cushioned Surface<br><br>Description:<br>Two cats lie stretched out on a pink cushioned surface, positioned near remote controls. The lighting is natural, and the cats appear calm and at ease.<br><br>Keywords:<br>cats, pink cushioned surface, remote controls, re...                                       |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              | **Title:**<br>Relaxed Tabby Kittens on Pink Cushion<br><br>**Description:**<br>Two tabby kittens lie stretched out on a pink cushion, positioned near remote controls. Their relaxed postures and soft fur highlight a calm, indoor setting.<br><br>**Keywords:**<br>kittens, tabby, pink cushion, remote...                                        |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>Two Sleeping Tabby Kittens on Pink Couch<br><br>Description:<br>Two young tabby cats lie relaxed on a bright pink sofa cushion, each with remote controls beside them. Soft lighting casts gentle shadows across their fur.<br><br>Keywords:<br>pink sofa, remote controls, young tabbies, indoo...                                       |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable`              | Title: Two feline friends on a cozy sofa.<br><br>Description: Two cats, one larger and one smaller, rest on a pink blanket-covered sofa. One wears a green collar. Two remote controls lie between them. The scene is bathed in warm, artificial light.<br><br>Keywords: Pink, sofa, cats, remote...                                                |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `usable`              | Title: Two Cats Resting on Pink Blanket with Remote Controls<br><br>Description: Two cats are sleeping on a pink blanket. A pink couch is visible in the background. Two remote controls are placed on the blanket. The cats appear to be in a relaxed state.<br><br>Keywords: cats, pink blanket...                                                |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | Title: Two Cats Sleeping on Sofa<br><br>Description: Two cats are sleeping on a pink sofa with remote controls beside them.<br><br>Keywords: cats, sleeping, sofa, remote controls, pink, relaxed, comfortable, indoor, domestic, pet, resting                                                                                                      |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable`              | Title: Two cats on a pink couch<br><br>Description: Two tabby cats are lying on a bright pink couch, sleeping peacefully with remote controls beside them. The scene is lit with soft, natural light, highlighting the cats' relaxed postures and the cozy setting.<br><br>Keywords: cats, couch,...                                                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Two Tabby Cats Sleeping on Pink Sofa<br><br>Description: Two tabby cats lie asleep on a bright pink sofa, each positioned near a white remote control. The scene is lit by even indoor lighting, highlighting the cats’ striped fur and relaxed postures.<br><br>Keywords: tabby cats, pin...                                                |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | Title: Two Cats Sleeping on a Pink Couch<br><br>Description: Two tabby cats are lying on a pink couch, each with a remote control nearby. The scene is well-lit, highlighting the texture of the couch and the fur of the cats.<br><br>Keywords: cats, tabby, pink couch, remote control, sleepin...                                                |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | Title: Two Tabby Cats Resting on Pink Sofa<br><br>Description: Two tabby cats are lying on a pink sofa, one near a white remote control and the other beside a blue remote. The scene is brightly lit, highlighting the cats' striped fur and relaxed postures.<br><br>Keywords: tabby cats, pink...                                                |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | Title:<br>Two cats sleeping on a pink blanket<br><br>Description:<br>Two cats are lying on a pink blanket on a couch, with remote controls placed nearby. One cat is smaller with a green collar, while the other is larger with a striped coat. Both appear relaxed and asleep under indoor ligh...                                                |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable`              | Title: Cats Sleeping on Pink Couch with Remotes<br><br>Description: Two tabby cats lie on a pink couch, one curled up, the other stretched out, with a remote control between them. The setting is indoors, with soft lighting highlighting their relaxed postures. The couch's vibrant pin...                                                      |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | Title: Two Tabby Cats Resting on Pink Fabric<br><br>Description: Two tabby cats lie prone on a pink fabric surface, each with a remote control nearby. Both cats are asleep with paws extended, exhibiting striped fur patterns.<br><br>Keywords: cats, tabby, pink, fabric, remote control, slee...                                                |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: Two tabby cats lying on a pink surface<br><br>Description: Two tabby cats lie on a bright pink fabric surface next to two remote controls. The cats are positioned with their heads facing in opposite directions.<br><br>Keywords: tabby, cats, pink, fabric, remote control, lying, indo...                                                |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: Two tabby cats lying on a pink blanket<br>Description: Two tabby cats lie on a pink blanket atop a red sofa next to two remote controls. The scene is captured under bright, direct lighting.<br>Keywords: cats, tabby, pink blanket, red sofa, remote controls, lying down, pets, i...                                                      |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | ### Title:<br>Two Cats Lounging on Red Couch<br><br>### Description:<br>Two cats, one lying on its back and the other on its side, relax on a red couch. Remote controls are placed nearby.<br><br>### Keywords:<br>cats, lounging, red couch, remote controls, relaxed, indoor, comfort, feline, domestic...                                       |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable_with_caveats` | Title:<br>Cats on a Pink Sofa<br><br>Description:<br>Two cats are lying on a pink sofa, one is stretched out with its paws extended, while the other is curled up with its head resting on the sofa.<br><br>Keywords:<br>cats, sofa, pink, relaxation, fur, stripes, sleeping                                                                       |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | Title:<br>Two cats resting on a pink sofa<br><br>Description:<br>Two tabby cats are lying on a pink sofa, one on its back and the other on its side, with a remote control nearby. The lighting is natural, and the cats have distinctive striped fur patterns.<br><br>Keywords: cats, tabby, sofa, pin...                                          |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | Title: Two Cats on Pink Sofa<br>Description: Two tabby cats rest on a pink fabric sofa. One cat lies on its side with a remote control nearby. The other cat is positioned with its head down.<br>Keywords: two tabby cats, pink fabric sofa, remote control, head down                                                                             |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Two cats on a pink couch<br>Description: Two cats lie on a pink couch, each near a remote control, with one cat’s head resting on the fabric and the other stretched out with eyes closed.<br>Keywords: cats, pink couch, remote controls, sleeping, relaxed, stripe...                                                    |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | ◁think▷So, let's analyze the image. The image shows two tabby cats on a pink fabric surface. There are two remote controls, one near each cat. The cats are lying down, one facing left, the other facing down. The lighting is bright, making colors vivid. The scene is indoors, li...                                                            |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable_with_caveats` | Title: Two Tabby Cats Sleeping on Pink Sofa with Remotes<br><br>Description: Two tabby cats are lying side by side on a bright pink sofa, each near a white remote control. The cat on the left is stretched out with its tail curled, while the one on the right is curled up with its hea...                                                      |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | Title: Two Cats Sleeping on a Pink Couch<br>Description: Two tabby cats are sleeping on a pink couch with two remote controls nearby.<br>Keywords: cats, sleeping, pink couch, remote controls                                                                                                                                                      |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | Title: Two Cats on a Couch<br>Description: Two cats are sleeping on a pink couch.<br>Keywords: cats, couch, sleeping, pink                                                                                                                                                                                                                          |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Two tabby cats sleeping on a pink blanket<br>Description: Two striped tabby cats rest on a bright pink blanket atop a red sofa. Two remote controls lie near the sleeping animals.<br>Keywords: cats, tabby, sleeping, pink, blanket, sofa, remote, indo...                                                 |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Two tabby cats sleeping on a pink blanket<br>Description: Two tabby cats are lying down on a pink blanket on a red sofa. Two remote controls are visible near the cats.<br>Keywords: cats, tabby, sleeping, pink, blanket, sofa, red, indoor, remote, co...                                                 |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `usable_with_caveats` | **Title:** Cozy Naptime<br><br>**Description:** Two tabby cats are peacefully sleeping on a pink couch, each with a remote control nearby.<br><br>**Keywords:** cats, sleeping, couch, remote, tabby, pets, rest, relaxation, domestic, comfort, home, furry, striped, peaceful, indoors.                                                           |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>The user asks: "Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.<br><br>Write:<br>- a concrete 5-10-word title;<br>- a 1-2-sentence...                                              |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | Alright, let's get to work. I need to generate some metadata for this image, and the instructions are very specific. First, I need a title, a description, and a list of keywords. The title has to be between five and ten words, and the description needs to be a short, factual p...                                                            |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A serene moment of feline tranquility on a vibrant pink couch, with two cats in peaceful slumber, surrounded by the soft glow of ambient lighting.                                                                                                                                                                                                  |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words, concrete. Looking at the image: two cats on a pink blanket with remotes. So maybe "Cats on Pink Blanket with Remotes" or something similar. Then description: need to mention the main subject (two cats),...                                                      |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `unusable`            | Two cats laying on a pink blanket with remotes.<end_of_utterance>                                                                                                                                                                                                                                                                                   |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `unusable`            | Here is the information in the format you requested:<br><br>**Title:** Two Cats on a Pink Blanket<br><br>**Description:** Two tabby cats are sleeping on a pink blanket, with two remote controls nearby. The cats are lying on their backs, with their paws stretched out and their eyes closed....                                                |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | Two cats sleeping on pink blanket with remote controls<br><br>Description: Two tabby cats rest on a pink fabric, with remote controls nearby; soft lighting and cozy setting.<br><br>Keywords: cats, sleeping, pink blanket, remote controls, tabby, relaxation, comfort, domestic, soft lighting                                                   |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | Got it, let's tackle this. First, the title needs to be concrete and 5-10 words. The image has two cats on a pink couch with remotes. So maybe "Two tabby cats resting on pink couch with remotes" but that's 10 words. Wait, "Two tabby cats resting on pink couch with remote contr...                                                            |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | cat, sleeping, pink blanket, remote control, striped, tabby, kitten, cozy, domestic, indoor, relaxation, comfort, home, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, felin...                                                            |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | Title: "A Study in Stripes"<br>Description: A close-up of a person wearing a striped shirt, with a focus on the pattern and color of the fabric. The image is set against a pink background, and the person is lying on a bed covered with a pink blanket. The lighting is soft, and the...                                                         |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | Action:<br>Lighting:<br>Brand:<br>Species:<br>Period:<br>Event:<br>Date:<br>Name:<br>Intent:<br>Brand:<br>Date:<br>Name:<br>Description:<br>Action:<br>Lighting:<br>Keywords:<br>Return exactly these three sections and nothing else:<br>Title:<br>Description:<br>- 10-18 unique, comma-separated keywords.<br>- 10-18 unique, comma-separated... |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.14`
- `mlx`: `0.32.1.dev20260816+c2bcf47ee`
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

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Create British-English catalogue metadata using only clearly visible facts.
> Omit uncertain details and unsupported identity, location, event, brand,
> species, period, or intent.
>
> &#8203;Write:
> &#45; a concrete 5-10-word title;
> &#45; a 1-2-sentence factual description of the main subject, setting, action,
> lighting, and distinctive details;
> &#45; 10-18 unique, comma-separated keywords.
>
> &#8203;Return exactly these three sections and nothing else:
> &#8203;Title:
> &#8203;Description:
> &#8203;Keywords:
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Complete Per-model Evidence

Complete generated or crash evidence for every attempted model.

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### Qwen/Qwen3-VL-2B-Instruct

<details>
<summary>Complete evidence: Qwen/Qwen3-VL-2B-Instruct</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.64s
- *Generation time:* 0.71s
- *Total time:* 1.36s
- *Input validation time:* 7.71e-05
- *Prompt preparation time:* 0.00916
- *First-token latency:* 0.0861
- *Cleanup time:* 0.0688
- *Prompt tokens:* 409
- *Generation tokens:* 82
- *Total tokens:* 491
- *Prompt throughput (raw):* 4,748 tok/s
- *Generation throughput (raw):* 138 tok/s
- *Peak memory:* 5.2
- *Active memory:* 4.3
- *Cache memory:* 0.07
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 544
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
Title: Two cats on a pink couch

Description: Two tabby cats are lying on a bright pink couch, sleeping peacefully with remote controls beside them. The scene is lit with soft, natural light, highlighting the cats' relaxed postures and the cozy setting.

Keywords: cats, couch, sleeping, remote, pink, tabby, relaxation, soft light, comfort, pet, blanket, lounging
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
- *Model load time:* 1.61s
- *Generation time:* 2.98s
- *Total time:* 4.60s
- *Input validation time:* 8.75e-05
- *Prompt preparation time:* 0.00197
- *First-token latency:* 1.2
- *Cleanup time:* 0.098
- *Prompt tokens:* 3,435
- *Generation tokens:* 60
- *Total tokens:* 3,495
- *Prompt throughput (raw):* 2,864 tok/s
- *Generation throughput (raw):* 34.4 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.27
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.003
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 477
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
Title: Two Cats Sleeping on Pink Blanket
Description: Two tabby cats are lying on a pink blanket with remote controls nearby, in a relaxed sleeping position.
Keywords: cats, sleeping, pink blanket, remote controls, tabby, relaxed, blanket, resting, pets, cozy, indoor
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
- *Model load time:* 0.58s
- *Generation time:* 0.64s
- *Total time:* 1.23s
- *Input validation time:* 8.14e-05
- *Prompt preparation time:* 0.00365
- *First-token latency:* 0.0723
- *Cleanup time:* 0.0745
- *Prompt tokens:* 350
- *Generation tokens:* 100
- *Total tokens:* 450
- *Prompt throughput (raw):* 4,839 tok/s
- *Generation throughput (raw):* 185 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.01
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.00379
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 523
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
Title: Two Cats Resting on a Pink Blanket

Description: Two cats are lying on a pink blanket, one is stretched out while the other is curled up. The cats appear to be sleeping peacefully, with one having a green collar. The lighting is soft and warm, highlighting the relaxed posture of the cats.

Keywords: cats, pink blanket, sleeping, green collar, relaxed posture, soft lighting, warm tones, peaceful, curled up, stretched out, comfortable
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
- *Model load time:* 1.31s
- *Generation time:* 1.89s
- *Total time:* 3.21s
- *Input validation time:* 6.81e-05
- *Prompt preparation time:* 0.00673
- *First-token latency:* 0.672
- *Cleanup time:* 0.0867
- *Prompt tokens:* 1,070
- *Generation tokens:* 81
- *Total tokens:* 1,151
- *Prompt throughput (raw):* 1,592 tok/s
- *Generation throughput (raw):* 69.5 tok/s
- *Peak memory:* 9.8
- *Active memory:* 8.0
- *Cache memory:* 0.23
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00508
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 2910
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
Two Cats Relaxing on Pink Cushioned Surface

Description:
Two cats lie stretched out on a pink cushioned surface, positioned near remote controls. The lighting is natural, and the cats appear calm and at ease.

Keywords:
cats, pink cushioned surface, remote controls, relaxed posture, indoor setting, natural lighting, striped fur, resting, cushioned fabric, domestic scene
</pre>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.34s
- *Generation time:* 2.41s
- *Total time:* 3.76s
- *Input validation time:* 8.27e-05
- *Prompt preparation time:* 0.000349
- *First-token latency:* 0.759
- *Cleanup time:* 0.0864
- *Prompt tokens:* 1,070
- *Generation tokens:* 106
- *Total tokens:* 1,176
- *Prompt throughput (raw):* 1,410 tok/s
- *Generation throughput (raw):* 64.6 tok/s
- *Peak memory:* 10
- *Active memory:* 8.4
- *Cache memory:* 0.23
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00534
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 2910
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
Relaxed Tabby Kittens on Pink Cushion

**Description:**
Two tabby kittens lie stretched out on a pink cushion, positioned near remote controls. Their relaxed postures and soft fur highlight a calm, indoor setting.

**Keywords:**
kittens, tabby, pink cushion, remote controls, relaxed posture, indoor setting, soft fur, calm atmosphere, stretched out, pink fabric, resting, domestic scene, fur patterns, cushioned surface, laid-back pose, domestic pets
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
- *Model load time:* 1.05s
- *Generation time:* 0.77s
- *Total time:* 1.84s
- *Input validation time:* 7.06e-05
- *Prompt preparation time:* 0.0126
- *First-token latency:* 0.279
- *Cleanup time:* 0.125
- *Prompt tokens:* 1,069
- *Generation tokens:* 85
- *Total tokens:* 1,154
- *Prompt throughput (raw):* 3,837 tok/s
- *Generation throughput (raw):* 193 tok/s
- *Peak memory:* 4.5
- *Active memory:* 2.8
- *Cache memory:* 0.16
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.0056
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 2909
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
Two Sleeping Tabby Kittens on Pink Couch

Description:
Two young tabby cats lie relaxed on a bright pink sofa cushion, each with remote controls beside them. Soft lighting casts gentle shadows across their fur.

Keywords:
pink sofa, remote controls, young tabbies, indoor setting, cozy scene, pet photoshoot, domestic ambiance, soft lighting, pet accessories, resting cats
</pre>

</details>

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### mlx-community/Molmo-7B-D-0924-8bit

<details>
<summary>Complete evidence: mlx-community/Molmo-7B-D-0924-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.20s
- *Generation time:* 2.11s
- *Total time:* 3.32s
- *Input validation time:* 0.000111
- *Prompt preparation time:* 0.00161
- *First-token latency:* 0.427
- *Cleanup time:* 0.0827
- *Prompt tokens:* 843
- *Generation tokens:* 89
- *Total tokens:* 932
- *Prompt throughput (raw):* 1,972 tok/s
- *Generation throughput (raw):* 53.7 tok/s
- *Peak memory:* 11
- *Active memory:* 9.0
- *Cache memory:* 0.08
- *Model-load active memory:* 9.04
- *Post-cleanup active memory:* 0.00592
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 468
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
 Title: Two feline friends on a cozy sofa.

Description: Two cats, one larger and one smaller, rest on a pink blanket-covered sofa. One wears a green collar. Two remote controls lie between them. The scene is bathed in warm, artificial light.

Keywords: Pink, sofa, cats, remote controls, green collar, artificial light, cozy, rest, different sizes, close-up, domestic, feline, companionship
</pre>

</details>

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### mlx-community/MolmoPoint-8B-fp16

<details>
<summary>Complete evidence: mlx-community/MolmoPoint-8B-fp16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type molmo_point)
- *Model load time:* 2.21s
- *Generation time:* 13.57s
- *Total time:* 15.80s
- *Input validation time:* 9.64e-05
- *Prompt preparation time:* 0.00503
- *First-token latency:* 0.891
- *Cleanup time:* 0.168
- *Prompt tokens:* 860
- *Generation tokens:* 78
- *Total tokens:* 938
- *Prompt throughput (raw):* 965 tok/s
- *Generation throughput (raw):* 6.21 tok/s
- *Peak memory:* 23
- *Active memory:* 19
- *Cache memory:* 4.3
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.00654
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 510
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
Title: Two Cats Resting on Pink Blanket with Remote Controls

Description: Two cats are sleeping on a pink blanket. A pink couch is visible in the background. Two remote controls are placed on the blanket. The cats appear to be in a relaxed state.

Keywords: cats, pink blanket, remote controls, couch, sleeping, relaxed, feline, domestic, comfortable, rest
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
- *Model load time:* 0.93s
- *Generation time:* 1.42s
- *Total time:* 2.36s
- *Input validation time:* 0.000101
- *Prompt preparation time:* 0.00115
- *First-token latency:* 0.214
- *Cleanup time:* 0.0827
- *Prompt tokens:* 883
- *Generation tokens:* 68
- *Total tokens:* 951
- *Prompt throughput (raw):* 4,126 tok/s
- *Generation throughput (raw):* 59.6 tok/s
- *Peak memory:* 9.3
- *Active memory:* 8.3
- *Cache memory:* 0.42
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00711
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 493
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
Title: Two Cats Sleeping on Sofa

Description: Two cats are sleeping on a pink sofa with remote controls beside them.

Keywords: cats, sleeping, sofa, remote controls, pink, relaxed, comfortable, indoor, domestic, pet, resting
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### mlx-community/Qwen3-VL-2B-Instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.72s
- *Generation time:* 0.76s
- *Total time:* 1.48s
- *Input validation time:* 8.24e-05
- *Prompt preparation time:* 0.000337
- *First-token latency:* 0.0865
- *Cleanup time:* 0.0882
- *Prompt tokens:* 409
- *Generation tokens:* 82
- *Total tokens:* 491
- *Prompt throughput (raw):* 4,728 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 5.3
- *Active memory:* 4.3
- *Cache memory:* 0.07
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00773
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 544
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
Title: Two cats on a pink couch

Description: Two tabby cats are lying on a bright pink couch, sleeping peacefully with remote controls beside them. The scene is lit with soft, natural light, highlighting the cats' relaxed postures and the cozy setting.

Keywords: cats, couch, sleeping, remote, pink, tabby, relaxation, soft light, comfort, pet, blanket, lounging
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
- *Model load time:* 2.49s
- *Generation time:* 1.09s
- *Total time:* 3.60s
- *Input validation time:* 0.00012
- *Prompt preparation time:* 0.0144
- *First-token latency:* 0.348
- *Cleanup time:* 0.11
- *Prompt tokens:* 421
- *Generation tokens:* 89
- *Total tokens:* 510
- *Prompt throughput (raw):* 1,209 tok/s
- *Generation throughput (raw):* 121 tok/s
- *Peak memory:* 21
- *Active memory:* 21
- *Cache memory:* 0.10
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00855
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
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
Title: Two Tabby Cats Sleeping on Pink Sofa

Description: Two tabby cats lie asleep on a bright pink sofa, each positioned near a white remote control. The scene is lit by even indoor lighting, highlighting the cats’ striped fur and relaxed postures.

Keywords: tabby cats, pink sofa, sleeping cats, remote controls, domestic interior, feline relaxation, striped fur, pet photography, cozy setting, animal stillness
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
- *Model load time:* 1.37s
- *Generation time:* 1.15s
- *Total time:* 2.53s
- *Input validation time:* 0.000105
- *Prompt preparation time:* 0.000573
- *First-token latency:* 0.282
- *Cleanup time:* 0.0861
- *Prompt tokens:* 421
- *Generation tokens:* 86
- *Total tokens:* 507
- *Prompt throughput (raw):* 1,491 tok/s
- *Generation throughput (raw):* 101 tok/s
- *Peak memory:* 7.1
- *Active memory:* 6.6
- *Cache memory:* 0.11
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00906
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
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
Title: Two Cats Sleeping on a Pink Couch

Description: Two tabby cats are lying on a pink couch, each with a remote control nearby. The scene is well-lit, highlighting the texture of the couch and the fur of the cats.

Keywords: cats, tabby, pink couch, remote control, sleeping, fur, texture, lighting, comfort, relaxation, domestic, pet, rest, cozy, home
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
- *Model load time:* 3.05s
- *Generation time:* 6.44s
- *Total time:* 9.52s
- *Input validation time:* 8.1e-05
- *Prompt preparation time:* 0.0144
- *First-token latency:* 0.844
- *Cleanup time:* 0.132
- *Prompt tokens:* 421
- *Generation tokens:* 107
- *Total tokens:* 528
- *Prompt throughput (raw):* 499 tok/s
- *Generation throughput (raw):* 19.2 tok/s
- *Peak memory:* 30
- *Active memory:* 29
- *Cache memory:* 0.23
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.00957
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
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
Title: Two Tabby Cats Resting on Pink Sofa

Description: Two tabby cats are lying on a pink sofa, one near a white remote control and the other beside a blue remote. The scene is brightly lit, highlighting the cats' striped fur and relaxed postures.

Keywords: tabby cats, pink sofa, remote controls, striped fur, resting, bright lighting, relaxed posture, domestic setting, comfortable, feline, lounging, indoor, soft fabric, casual, pet, home environment, serene, cozy
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
- *Model load time:* 7.01s
- *Generation time:* 6.20s
- *Total time:* 13.23s
- *Input validation time:* 8.28e-05
- *Prompt preparation time:* 0.0163
- *First-token latency:* 3.77
- *Cleanup time:* 0.215
- *Prompt tokens:* 286
- *Generation tokens:* 105
- *Total tokens:* 391
- *Prompt throughput (raw):* 75.8 tok/s
- *Generation throughput (raw):* 44.2 tok/s
- *Peak memory:* 65
- *Active memory:* 63
- *Cache memory:* 0.11
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.00995
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 550
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
Two cats sleeping on a pink blanket

Description:
Two cats are lying on a pink blanket on a couch, with remote controls placed nearby. One cat is smaller with a green collar, while the other is larger with a striped coat. Both appear relaxed and asleep under indoor lighting.

Keywords:
cats, pink blanket, couch, remote controls, sleeping, striped coat, green collar, indoor, pets, relaxation, two animals, soft fabric, remote control, lying down, home setting, feline, nap, cozy
</pre>

</details>

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### mlx-community/X-Reasoner-7B-8bit

<details>
<summary>Complete evidence: mlx-community/X-Reasoner-7B-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.16s
- *Generation time:* 1.95s
- *Total time:* 3.11s
- *Input validation time:* 8.79e-05
- *Prompt preparation time:* 0.00047
- *First-token latency:* 0.271
- *Cleanup time:* 0.0975
- *Prompt tokens:* 511
- *Generation tokens:* 106
- *Total tokens:* 617
- *Prompt throughput (raw):* 1,884 tok/s
- *Generation throughput (raw):* 66.1 tok/s
- *Peak memory:* 10
- *Active memory:* 9.5
- *Cache memory:* 0.06
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 602
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
Title: Cats Sleeping on Pink Couch with Remotes

Description: Two tabby cats lie on a pink couch, one curled up, the other stretched out, with a remote control between them. The setting is indoors, with soft lighting highlighting their relaxed postures. The couch's vibrant pink fabric contrasts with the cats' striped fur.

Keywords: cats, sleeping, pink couch, tabby, remote control, indoor, relaxed, striped fur, soft lighting, domestic, feline, comfort, companionship, home setting.
</pre>

</details>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### mlx-community/gemma-3-27b-it-qat-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3-27b-it-qat-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 2.26s
- *Generation time:* 3.63s
- *Total time:* 5.91s
- *Input validation time:* 9.85e-05
- *Prompt preparation time:* 0.00286
- *First-token latency:* 0.764
- *Cleanup time:* 0.12
- *Prompt tokens:* 379
- *Generation tokens:* 90
- *Total tokens:* 469
- *Prompt throughput (raw):* 496 tok/s
- *Generation throughput (raw):* 31.6 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.33
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0108
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 527
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
Title: Two Tabby Cats Resting on Pink Fabric

Description: Two tabby cats lie prone on a pink fabric surface, each with a remote control nearby. Both cats are asleep with paws extended, exhibiting striped fur patterns.

Keywords: cats, tabby, pink, fabric, remote control, sleeping, lying down, indoors, domestic animals, striped, paws, feline, resting, horizontal, two animals, close-up, daylight, cushions.
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
- *Model load time:* 2.31s
- *Generation time:* 0.94s
- *Total time:* 3.28s
- *Input validation time:* 9.18e-05
- *Prompt preparation time:* 0.0204
- *First-token latency:* 0.37
- *Cleanup time:* 0.112
- *Prompt tokens:* 390
- *Generation tokens:* 73
- *Total tokens:* 463
- *Prompt throughput (raw):* 1,055 tok/s
- *Generation throughput (raw):* 130 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.15
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0119
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 526
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
Title: Two tabby cats lying on a pink surface

Description: Two tabby cats lie on a bright pink fabric surface next to two remote controls. The cats are positioned with their heads facing in opposite directions.

Keywords: tabby, cats, pink, fabric, remote control, lying, indoor, domestic, striped, fur, two, pets, bright, daylight
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
- *Model load time:* 2.57s
- *Generation time:* 3.70s
- *Total time:* 6.28s
- *Input validation time:* 7.53e-05
- *Prompt preparation time:* 0.000361
- *First-token latency:* 0.832
- *Cleanup time:* 0.129
- *Prompt tokens:* 390
- *Generation tokens:* 80
- *Total tokens:* 470
- *Prompt throughput (raw):* 469 tok/s
- *Generation throughput (raw):* 28.1 tok/s
- *Peak memory:* 19
- *Active memory:* 18
- *Cache memory:* 0.59
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0124
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 526
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
Title: Two tabby cats lying on a pink blanket
Description: Two tabby cats lie on a pink blanket atop a red sofa next to two remote controls. The scene is captured under bright, direct lighting.
Keywords: cats, tabby, pink blanket, red sofa, remote controls, lying down, pets, indoor, fabric, stripes, animals, furniture, home, resting, soft, textile
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
- *Model load time:* 1.69s
- *Generation time:* 3.19s
- *Total time:* 4.89s
- *Input validation time:* 8.82e-05
- *Prompt preparation time:* 0.00401
- *First-token latency:* 0.975
- *Cleanup time:* 0.097
- *Prompt tokens:* 1,335
- *Generation tokens:* 89
- *Total tokens:* 1,424
- *Prompt throughput (raw):* 1,369 tok/s
- *Generation throughput (raw):* 40.5 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.27
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0144
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 472
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
### Title:
Two Cats Lounging on Red Couch

### Description:
Two cats, one lying on its back and the other on its side, relax on a red couch. Remote controls are placed nearby.

### Keywords:
cats, lounging, red couch, remote controls, relaxed, indoor, comfort, feline, domestic, paws, stretch, lounge, resting, home, soft, purring, peaceful
</pre>

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

<details>
<summary>Complete evidence: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 7 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.41s
- *Generation time:* 0.21s
- *Total time:* 0.64s
- *Input validation time:* 0.00218
- *Prompt preparation time:* 0.00604
- *First-token latency:* 0.0831
- *Cleanup time:* 0.0598
- *Prompt tokens:* 350
- *Generation tokens:* 63
- *Total tokens:* 413
- *Prompt throughput (raw):* 4,212 tok/s
- *Generation throughput (raw):* 516 tok/s
- *Peak memory:* 1.3
- *Active memory:* 0.90
- *Cache memory:* 0.02
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 523
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
Cats on a Pink Sofa

Description:
Two cats are lying on a pink sofa, one is stretched out with its paws extended, while the other is curled up with its head resting on the sofa.

Keywords:
cats, sofa, pink, relaxation, fur, stripes, sleeping
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
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.11s
- *Generation time:* 4.20s
- *Total time:* 6.33s
- *Input validation time:* 7.54e-05
- *Prompt preparation time:* 0.0067
- *First-token latency:* 0.981
- *Cleanup time:* 0.101
- *Prompt tokens:* 537
- *Generation tokens:* 100
- *Total tokens:* 637
- *Prompt throughput (raw):* 548 tok/s
- *Generation throughput (raw):* 31.2 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.14
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.000968
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 472
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
Two cats resting on a pink sofa

Description:
Two tabby cats are lying on a pink sofa, one on its back and the other on its side, with a remote control nearby. The lighting is natural, and the cats have distinctive striped fur patterns.

Keywords: cats, tabby, sofa, pink, remote control, resting, lying, natural lighting, striped fur, indoor, furniture, pets, relaxation, home, domestic, fur, pattern, remote, control
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
- *Observations:* Keyword list has 4 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.32s
- *Generation time:* 0.99s
- *Total time:* 2.33s
- *Input validation time:* 8.25e-05
- *Prompt preparation time:* 0.00795
- *First-token latency:* 0.32
- *Cleanup time:* 0.0842
- *Prompt tokens:* 499
- *Generation tokens:* 61
- *Total tokens:* 560
- *Prompt throughput (raw):* 1,559 tok/s
- *Generation throughput (raw):* 92.0 tok/s
- *Peak memory:* 7.8
- *Active memory:* 6.8
- *Cache memory:* 0.05
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 553
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
Title: Two Cats on Pink Sofa
Description: Two tabby cats rest on a pink fabric sofa. One cat lies on its side with a remote control nearby. The other cat is positioned with its head down.
Keywords: two tabby cats, pink fabric sofa, remote control, head down
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Two Cats on Pink Sofa
Description: Two tabby cats rest on a pink fabric sofa. One cat lies on its side with a remote control nearby. The other cat is positioned with its head down. 
Keywords: two tabby cats, pink fabric sofa, remote control, head down
```

</details>

</details>

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### mlx-community/GLM-4.6V-nvfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 7.59s
- *Generation time:* 6.39s
- *Total time:* 13.99s
- *Input validation time:* 7.3e-05
- *Prompt preparation time:* 0.00709
- *First-token latency:* 4.69
- *Cleanup time:* 0.232
- *Prompt tokens:* 499
- *Generation tokens:* 89
- *Total tokens:* 588
- *Prompt throughput (raw):* 106 tok/s
- *Generation throughput (raw):* 52.6 tok/s
- *Peak memory:* 63
- *Active memory:* 62
- *Cache memory:* 0.16
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 553
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
&lt;|begin_of_box|&gt;Title: Two cats on a pink couch
Description: Two cats lie on a pink couch, each near a remote control, with one cat’s head resting on the fabric and the other stretched out with eyes closed.
Keywords: cats, pink couch, remote controls, sleeping, relaxed, striped fur, feline, domestic, indoor, pets, lounging, two animals, soft fabric, calm, cozy, household, companionship&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Two cats on a pink couch
Description: Two cats lie on a pink couch, each near a remote control, with one cat’s head resting on the fabric and the other stretched out with eyes closed.
Keywords: cats, pink couch, remote controls, sleeping, relaxed, striped fur, feline, domestic, indoor, pets, lounging, two animals, soft fabric, calm, cozy, household, companionship<|end_of_box|>
```

</details>

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
- *Model load time:* 3.23s
- *Generation time:* 117.87s
- *Total time:* 121.10s
- *Input validation time:* 0.000101
- *Prompt preparation time:* 0.00208
- *First-token latency:* 1.17
- *Cleanup time:* 0.15
- *Prompt tokens:* 489
- *Generation tokens:* 562
- *Total tokens:* 1,051
- *Prompt throughput (raw):* 417 tok/s
- *Generation throughput (raw):* 4.83 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.2
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00365
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 659
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
◁think▷So, let's analyze the image. The image shows two tabby cats on a pink fabric surface. There are two remote controls, one near each cat. The cats are lying down, one facing left, the other facing down. The lighting is bright, making colors vivid. The scene is indoors, likely a living room.

First, the title: need a 5-10 word title. Maybe "Two tabby cats resting on pink fabric with remotes." Wait, check word count. "Two tabby cats rest on pink fabric with remotes." That's 8 words. Good.

Description: 1-2 sentences. Describe main subject (two tabby cats), setting (pink fabric, likely couch), action (resting), lighting (bright, vivid colors), distinctive details (remotes, their positions). So: "Two tabby cats lie on a pink fabric surface, with two remote controls nearby, under bright lighting that enhances vivid colors." Wait, maybe shorter. "Two tabby cats rest on pink fabric; two remotes nearby, bright lighting highlights their fur patterns." Hmm, need to be concise. Let's refine: "Two tabby cats lie on pink fabric with two remote controls, under bright lighting emphasizing their striped fur." That's 2 sentences. Wait, the instruction says 1-2 sentences. Let's make it better: "Two tabby cats rest on a pink fabric surface, with two remote controls nearby, under bright lighting that highlights their striped patterns." That's 2 sentences.

Keywords: 10-18 unique, comma-separated. Keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment, relaxed cats. Let's count: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment. That's 10. Maybe add "two cats" but that's redundant. Wait, "two tabby cats" is already mentioned. So keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment. That's 10. Check if all are unique and visible. Yes.◁/think▷Title: Two tabby cats rest on pink fabric with remotes nearby
Description: Two tabby cats lie on a pink fabric surface, with two remote controls positioned near them, under bright lighting that highlights their striped fur patterns.
Keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment&lt;|im_assistant|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷So, let's analyze the image. The image shows two tabby cats on a pink fabric surface. There are two remote controls, one near each cat. The cats are lying down, one facing left, the other facing down. The lighting is bright, making colors vivid. The scene is indoors, likely a living room. 

First, the title: need a 5-10 word title. Maybe "Two tabby cats resting on pink fabric with remotes." Wait, check word count. "Two tabby cats rest on pink fabric with remotes." That's 8 words. Good.

Description: 1-2 sentences. Describe main subject (two tabby cats), setting (pink fabric, likely couch), action (resting), lighting (bright, vivid colors), distinctive details (remotes, their positions). So: "Two tabby cats lie on a pink fabric surface, with two remote controls nearby, under bright lighting that enhances vivid colors." Wait, maybe shorter. "Two tabby cats rest on pink fabric; two remotes nearby, bright lighting highlights their fur patterns." Hmm, need to be concise. Let's refine: "Two tabby cats lie on pink fabric with two remote controls, under bright lighting emphasizing their striped fur." That's 2 sentences. Wait, the instruction says 1-2 sentences. Let's make it better: "Two tabby cats rest on a pink fabric surface, with two remote controls nearby, under bright lighting that highlights their striped patterns." That's 2 sentences.

Keywords: 10-18 unique, comma-separated. Keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment, relaxed cats. Let's count: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment. That's 10. Maybe add "two cats" but that's redundant. Wait, "two tabby cats" is already mentioned. So keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment. That's 10. Check if all are unique and visible. Yes.◁/think▷Title: Two tabby cats rest on pink fabric with remotes nearby  
Description: Two tabby cats lie on a pink fabric surface, with two remote controls positioned near them, under bright lighting that highlights their striped fur patterns.  
Keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment<|im_assistant|>
```

</details>

</details>

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### mlx-community/Ornith-1.0-35B-bf16

<details>
<summary>Complete evidence: mlx-community/Ornith-1.0-35B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 11.18s
- *Generation time:* 7.76s
- *Total time:* 18.97s
- *Input validation time:* 9.02e-05
- *Prompt preparation time:* 0.0138
- *First-token latency:* 5.49
- *Cleanup time:* 0.255
- *Prompt tokens:* 421
- *Generation tokens:* 133
- *Total tokens:* 554
- *Prompt throughput (raw):* 76.7 tok/s
- *Generation throughput (raw):* 66.2 tok/s
- *Peak memory:* 71
- *Active memory:* 70
- *Cache memory:* 0.10
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00705
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
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
Title: Two Tabby Cats Sleeping on Pink Sofa with Remotes

Description: Two tabby cats are lying side by side on a bright pink sofa, each near a white remote control. The cat on the left is stretched out with its tail curled, while the one on the right is curled up with its head down.

Keywords: tabby cats, pink sofa, sleeping cats, remote controls, feline relaxation, striped fur, cozy setting, indoor scene, pet comfort, soft lighting, domestic life, animal companionship, resting pets, home environment, tranquil moment, feline behavior, sofa bedding, remote placement, cat napping, peaceful atmosphere
</pre>

</details>

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 4 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.52s
- *Generation time:* 0.30s
- *Total time:* 0.83s
- *Input validation time:* 8.89e-05
- *Prompt preparation time:* 0.00358
- *First-token latency:* 0.15
- *Cleanup time:* 0.0876
- *Prompt tokens:* 511
- *Generation tokens:* 40
- *Total tokens:* 551
- *Prompt throughput (raw):* 3,403 tok/s
- *Generation throughput (raw):* 307 tok/s
- *Peak memory:* 2.5
- *Active memory:* 1.3
- *Cache memory:* 0.03
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.00742
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 602
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
Title: Two Cats Sleeping on a Pink Couch
Description: Two tabby cats are sleeping on a pink couch with two remote controls nearby.
Keywords: cats, sleeping, pink couch, remote controls
</pre>

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

<details>
<summary>Complete evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 4 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.62s
- *Generation time:* 0.71s
- *Total time:* 1.34s
- *Input validation time:* 0.000107
- *Prompt preparation time:* 0.00239
- *First-token latency:* 0.379
- *Cleanup time:* 0.0809
- *Prompt tokens:* 1,204
- *Generation tokens:* 31
- *Total tokens:* 1,235
- *Prompt throughput (raw):* 3,180 tok/s
- *Generation throughput (raw):* 125 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.30
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.00968
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 504
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
 Title: Two Cats on a Couch
Description: Two cats are sleeping on a pink couch.
Keywords: cats, couch, sleeping, pink
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
- *Observations:* Unrecognised model control tokens remain visible; Duplicate
  keywords: domestic
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.24s
- *Generation time:* 1.65s
- *Total time:* 4.92s
- *Input validation time:* 8.29e-05
- *Prompt preparation time:* 0.0202
- *First-token latency:* 0.314
- *Cleanup time:* 0.135
- *Prompt tokens:* 386
- *Generation tokens:* 72
- *Total tokens:* 458
- *Prompt throughput (raw):* 1,231 tok/s
- *Generation throughput (raw):* 60.8 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 498
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
&lt;channel|&gt;Title: Two tabby cats sleeping on a pink blanket
Description: Two striped tabby cats rest on a bright pink blanket atop a red sofa. Two remote controls lie near the sleeping animals.
Keywords: cats, tabby, sleeping, pink, blanket, sofa, remote, indoor, pets, striped, resting, feline, domestic, domestic
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Two tabby cats sleeping on a pink blanket
Description: Two striped tabby cats rest on a bright pink blanket atop a red sofa. Two remote controls lie near the sleeping animals.
Keywords: cats, tabby, sleeping, pink, blanket, sofa, remote, indoor, pets, striped, resting, feline, domestic, domestic
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
- *Model load time:* 3.17s
- *Generation time:* 1.72s
- *Total time:* 4.90s
- *Input validation time:* 9.91e-05
- *Prompt preparation time:* 0.00038
- *First-token latency:* 0.302
- *Cleanup time:* 0.111
- *Prompt tokens:* 386
- *Generation tokens:* 74
- *Total tokens:* 460
- *Prompt throughput (raw):* 1,279 tok/s
- *Generation throughput (raw):* 59.3 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 498
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
&lt;channel|&gt;Title: Two tabby cats sleeping on a pink blanket
Description: Two tabby cats are lying down on a pink blanket on a red sofa. Two remote controls are visible near the cats.
Keywords: cats, tabby, sleeping, pink, blanket, sofa, red, indoor, remote, controls, feline, resting, domestic, fur, soft
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Two tabby cats sleeping on a pink blanket
Description: Two tabby cats are lying down on a pink blanket on a red sofa. Two remote controls are visible near the cats.
Keywords: cats, tabby, sleeping, pink, blanket, sofa, red, indoor, remote, controls, feline, resting, domestic, fur, soft
```

</details>

</details>

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### mlx-community/gemma-3n-E4B-it-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E4B-it-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 2 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 2.38s
- *Generation time:* 1.69s
- *Total time:* 4.08s
- *Input validation time:* 0.000105
- *Prompt preparation time:* 0.00292
- *First-token latency:* 0.355
- *Cleanup time:* 0.121
- *Prompt tokens:* 378
- *Generation tokens:* 63
- *Total tokens:* 441
- *Prompt throughput (raw):* 1,066 tok/s
- *Generation throughput (raw):* 47.8 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.04
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0113
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 529
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
**Title:** Cozy Naptime

**Description:** Two tabby cats are peacefully sleeping on a pink couch, each with a remote control nearby.

**Keywords:** cats, sleeping, couch, remote, tabby, pets, rest, relaxation, domestic, comfort, home, furry, striped, peaceful, indoors.
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
- *Observations:* Missing or empty fields: Title, Description; Response
  repeats the task instructions instead of only returning the requested
  fields; Extra text appears before the Title field; Response appears cut off
  at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.53s
- *Generation time:* 23.71s
- *Total time:* 25.25s
- *Input validation time:* 8.37e-05
- *Prompt preparation time:* 0.00667
- *First-token latency:* 0.959
- *Cleanup time:* 0.0869
- *Prompt tokens:* 1,426
- *Generation tokens:* 1,000
- *Total tokens:* 2,426
- *Prompt throughput (raw):* 1,488 tok/s
- *Generation throughput (raw):* 44.0 tok/s
- *Peak memory:* 14
- *Active memory:* 13
- *Cache memory:* 0.51
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.000706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 846
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
The user asks: "Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:"

We need to produce a title (5-10 words), a description (1-2 sentences), and a list of 10-18 unique keywords, separated by commas.

We must use only clearly visible facts from the image. The image shows two cats lying on a pink couch, sleeping. There's a remote control on the couch near them. The cats are tabby, with brown, black, and tan stripes. They appear to be sleeping, curled up. The couch is pink. The remote is a typical TV remote with buttons. The lighting appears to be natural or indoor, but we can say "soft indoor lighting" maybe. The cats are lying on their sides, one is curled up with its head on the couch, the other is lying on its side with its head near the remote. The remote is a standard TV remote, but we cannot assume brand. The couch is a pink fabric. The cats are domestic short-haired tabby. The image is presumably taken from above, showing the cats from a top-down perspective.

We need to avoid uncertain details: we cannot assume location (like "living room") unless it's clearly visible. The couch is pink, but we cannot be sure it's a living room. We can say "couch" or "sofa". The remote is present, but we cannot assume it's a TV remote? It looks like a remote control, but we can say "remote control". The cats are sleeping. The lighting is soft, maybe "ambient indoor lighting". The cats are tabby. The couch is pink. The remote is white with colored buttons. The cats are curled up. The cats are domestic. The cats are lying on a pink surface. The remote is placed near them.

We need to produce British English catalogue metadata. So we need to use British spelling, e.g., "colour" vs "color". But we can just use British English.

We need to produce a title of 5-10 words. Something like "Two Tabby Cats Napping on Pink Couch". That's 7 words: "Two Tabby Cats Napping on Pink Couch". That's 7 words. That works.

Description: 1-2 sentences. Something like: "Two domestic tabby cats are sleeping side‑by‑side on a bright pink sofa, one curled with its head on the fabric, the other stretched with its head near a white remote control. Soft indoor lighting highlights the cats' striped fur and the pink upholstery." That's two sentences. That includes main subject (two tabby cats), setting (bright pink sofa), action (sleeping side‑by‑side), lighting (soft indoor lighting), distinctive details (striped fur, remote control). That's fine.

Keywords: 10-18 unique, comma-separated. Something like: "tabby cats, sleeping, pink sofa, remote control, indoor lighting, striped fur, domestic cats, curled posture, side‑by‑side, soft illumination". That's 10 keywords. We can add a few more: "fabric upholstery, relaxed posture, top‑down view". That would bring to 13. Let's count: "tabby cats, sleeping, pink sofa, remote control, indoor lighting, striped fur, domestic cats, curled posture, side‑by‑side, soft illumination, fabric upholstery, relaxed posture, top‑down view". That's 13 keywords. Good.

We need to ensure we have exactly three sections: Title:, Description:, Keywords:. No extra text. The title must be 5-10 words. "Two Tabby Cats Napping on Pink Couch" is 7 words. Good.

Check for British English: "couch" is fine, but "sofa" might be more British. The title uses "couch". Could use "sofa". "Two Tabby Cats Napping on Pink Sofa". That's 7 words. "Two Tabby Cats Napping on Pink Sofa". That is British. Let's use "sofa". So title: "Two Tabby Cats Napping on Pink Sofa". That's 7 words.

Description: "Two domestic tabby cats are sleeping side‑by‑side on a
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
- *Observations:* Response repeats the same text; Extra text appears before
  the Title field; Keyword list has 33 terms (requested 10-18); Duplicate
  keywords: couch, pet, sofa, remote, television, television remote, blurry,
  indoor, home, furniture, domestic cat, animal sleeping
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 7.07s
- *Generation time:* 20.54s
- *Total time:* 27.62s
- *Input validation time:* 8.08e-05
- *Prompt preparation time:* 0.00798
- *First-token latency:* 2.68
- *Cleanup time:* 0.225
- *Prompt tokens:* 545
- *Generation tokens:* 944
- *Total tokens:* 1,489
- *Prompt throughput (raw):* 203 tok/s
- *Generation throughput (raw):* 57.9 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.10
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00118
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 662
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
Alright, let's get to work. I need to generate some metadata for this image, and the instructions are very specific. First, I need a title, a description, and a list of keywords. The title has to be between five and ten words, and the description needs to be a short, factual paragraph. The keywords are crucial; they have to be unique and relevant, and I can only use what's *clearly* visible. No guessing, no assumptions.

Okay, let's break down the image. I see two cats, definitely. One is a calico, and the other is a tabby. They're on a red couch, and there's a remote control nearby. The lighting seems to be indoor, probably from a lamp or the sun. The cats are sleeping, and the image is a bit blurry.

Now, the title. I need something short and descriptive. "Two cats sleeping on a red couch" is a good start, but I can do better. "Calico and tabby cats resting on a red couch" is more specific, but maybe too long. "Two cats sleeping on a red couch with a remote" is also good. I'll go with "Two cats sleeping on a red couch with a remote." It's concise and accurate.

Next, the description. I need to cover the main subject, setting, action, lighting, and any distinctive details. The main subjects are the two cats. The setting is a red couch. They're sleeping. The lighting is soft, and the image is a bit blurry. The distinctive details are the calico and tabby patterns, the red couch, and the remote. I'll write: "A calico cat and a tabby cat are sleeping side-by-side on a red couch. A remote control lies nearby. The image is slightly blurry, capturing a quiet moment indoors." That's two sentences, factual, and covers everything.

Finally, the keywords. I need to be very careful here. I'll list the obvious things: cat, cats, calico, tabby, couch, red couch, pet, animal, sleeping, resting, couch, sofa, remote, television, television remote, blurry, indoor, home, furniture, pet cat, domestic cat, animal sleeping, and couch. I'll make sure they are all unique and relevant. I'll remove "couch" and "sofa" since they are the same thing. I'll also remove "television" and "television remote" because the remote is the only thing that is related to a television. I'll also remove "blurry" because it is not a fact. I'll also remove "pet" and "domestic cat" because they are too general. I'll also remove "animal" because it is too general. I'll also remove "animal sleeping" because it is too general. I'll also remove "home" because it is too general. I'll also remove "furniture" because it is too general. I'll also remove "television" because it is too general. I'll also remove "television remote" because it is too general. I'll also remove "blurry" because it is not a fact. I'll also remove "indoor" because it is too general. I'll also remove "pet" because it is too general. I'll also remove "domestic cat" because it is too general. I'll also remove "animal" because it is too general. I'll also remove "animal sleeping" because it is too general. I'll also remove "home" because it is too general. I'll also remove "furniture" because it is too general. I'll also remove
&lt;/think&gt;

Title: Two cats sleeping on a red couch with a remote

Description: A calico cat and a tabby cat are sleeping side-by-side on a red couch. A remote control lies nearby. The image is slightly blurry, capturing a quiet moment indoors.

Keywords: cat, cats, calico, tabby, couch, red couch, pet, animal, sleeping, resting, sofa, remote, television, television remote, blurry, indoor, home, furniture, pet cat, domestic cat, animal sleeping, couch, sofa, remote, television, television remote, blurry, indoor, home, furniture, pet, domestic cat, animal sleeping
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get to work. I need to generate some metadata for this image, and the instructions are very specific. First, I need a title, a description, and a list of keywords. The title has to be between five and ten words, and the description needs to be a short, factual paragraph. The keywords are crucial; they have to be unique and relevant, and I can only use what's *clearly* visible. No guessing, no assumptions.

Okay, let's break down the image. I see two cats, definitely. One is a calico, and the other is a tabby. They're on a red couch, and there's a remote control nearby. The lighting seems to be indoor, probably from a lamp or the sun. The cats are sleeping, and the image is a bit blurry.

Now, the title. I need something short and descriptive. "Two cats sleeping on a red couch" is a good start, but I can do better. "Calico and tabby cats resting on a red couch" is more specific, but maybe too long. "Two cats sleeping on a red couch with a remote" is also good. I'll go with "Two cats sleeping on a red couch with a remote." It's concise and accurate.

Next, the description. I need to cover the main subject, setting, action, lighting, and any distinctive details. The main subjects are the two cats. The setting is a red couch. They're sleeping. The lighting is soft, and the image is a bit blurry. The distinctive details are the calico and tabby patterns, the red couch, and the remote. I'll write: "A calico cat and a tabby cat are sleeping side-by-side on a red couch. A remote control lies nearby. The image is slightly blurry, capturing a quiet moment indoors." That's two sentences, factual, and covers everything.

Finally, the keywords. I need to be very careful here. I'll list the obvious things: cat, cats, calico, tabby, couch, red couch, pet, animal, sleeping, resting, couch, sofa, remote, television, television remote, blurry, indoor, home, furniture, pet cat, domestic cat, animal sleeping, and couch. I'll make sure they are all unique and relevant. I'll remove "couch" and "sofa" since they are the same thing. I'll also remove "television" and "television remote" because the remote is the only thing that is related to a television. I'll also remove "blurry" because it is not a fact. I'll also remove "pet" and "domestic cat" because they are too general. I'll also remove "animal" because it is too general. I'll also remove "animal sleeping" because it is too general. I'll also remove "home" because it is too general. I'll also remove "furniture" because it is too general. I'll also remove "television" because it is too general. I'll also remove "television remote" because it is too general. I'll also remove "blurry" because it is not a fact. I'll also remove "indoor" because it is too general. I'll also remove "pet" because it is too general. I'll also remove "domestic cat" because it is too general. I'll also remove "animal" because it is too general. I'll also remove "animal sleeping" because it is too general. I'll also remove "home" because it is too general. I'll also remove "furniture" because it is too general. I'll also remove
</think>

Title: Two cats sleeping on a red couch with a remote

Description: A calico cat and a tabby cat are sleeping side-by-side on a red couch. A remote control lies nearby. The image is slightly blurry, capturing a quiet moment indoors.

Keywords: cat, cats, calico, tabby, couch, red couch, pet, animal, sleeping, resting, sofa, remote, television, television remote, blurry, indoor, home, furniture, pet cat, domestic cat, animal sleeping, couch, sofa, remote, television, television remote, blurry, indoor, home, furniture, pet, domestic cat, animal sleeping
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
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava_qwen2 via
  fastvlm)
- *Model load time:* 0.59s
- *Generation time:* 0.43s
- *Total time:* 1.03s
- *Input validation time:* 9.45e-05
- *Prompt preparation time:* 0.00177
- *First-token latency:* 0.299
- *Cleanup time:* 0.0746
- *Prompt tokens:* 120
- *Generation tokens:* 32
- *Total tokens:* 152
- *Prompt throughput (raw):* 401 tok/s
- *Generation throughput (raw):* 343 tok/s
- *Peak memory:* 2.2
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00149
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 567
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
A serene moment of feline tranquility on a vibrant pink couch, with two cats in peaceful slumber, surrounded by the soft glow of ambient lighting.
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
- *Observations:* Extra text appears before the Title field; Response appears
  cut off at the token limit; Internal reasoning block appears incomplete;
  Title has 21 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.65s
- *Generation time:* 20.23s
- *Total time:* 21.89s
- *Input validation time:* 6.81e-05
- *Prompt preparation time:* 0.0019
- *First-token latency:* 0.372
- *Cleanup time:* 0.0962
- *Prompt tokens:* 495
- *Generation tokens:* 1,000
- *Total tokens:* 1,495
- *Prompt throughput (raw):* 1,330 tok/s
- *Generation throughput (raw):* 50.4 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.08
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 529
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
&lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words, concrete. Looking at the image: two cats on a pink blanket with remotes. So maybe "Cats on Pink Blanket with Remotes" or something similar. Then description: need to mention the main subject (two cats), setting (pink blanket, red couch), action (lying down), lighting (bright), distinctive details (remotes, cat positions). Then keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home. Let's make sure to omit uncertain details.

Title: "Cats Resting on Pink Blanket with Remotes" – wait, 5-10 words. Let's check. "Two Cats on Pink Blanket with Remotes" is good.

Description: Two tabby cats lie on a pink blanket atop a red couch, with remote controls nearby, in bright lighting. – need to make it 1-2 sentences. So: "Two tabby cats rest on a pink blanket on a red couch, with remote controls nearby, under bright lighting."

Keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home. Let's check unique ones, comma-separated.

Now, let's structure each part:

Title: "Cats on Pink Blanket with Remotes" (5 words, concrete)

Description: Two tabby cats lie on a pink blanket on a red couch, with remote controls nearby, in bright lighting. (1 sentence, factual)

Keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home (10 words, unique)

Wait, need to make sure all are clearly visible. The image has two cats, pink blanket, red couch, remotes. So description should mention those. Let's adjust description: "Two tabby cats rest on a pink blanket on a red couch, with remote controls nearby, under bright lighting." That's factual, no unsupported details.

Keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home. Are these unique? Let's see. Maybe add "pet" or "cats" is there. Wait, need 10-18. Let's check: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home, couch, red (but red is part of couch, maybe not). Wait, the couch is red, but maybe include "red couch" as a keyword? But need unique. Let's do: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home, pet, indoor, resting, tabby, two. That's 12, which is between 10-18.

Wait, let's check the image again. Two tabby cats, so "tabby" can be a keyword. "resting" is action. "indoor" setting. "pink" blanket. "remotes" are there. "couch" is red. So keywords: cats, tabby, blanket, couch, remotes, pink, resting, indoor, relaxed, home, lying, domestic. That's 11, which is good.

Now, final check:

Title: "Cats on Pink Blanket with Remotes" (5 words, concrete)

Description: Two tabby cats rest on a pink blanket on a red couch, with remote controls nearby, in bright lighting. (1 sentence, factual)

Keywords: cats, tabby, blanket, couch, remotes, pink, resting, indoor, relaxed, home (or add more, but need unique). Wait, the image has two cats, so "two" could be a keyword. Let's do: cats, tabby, blanket, couch, remotes, pink, resting, indoor, relaxed, home, two. That's 11.

Wait, the user said "unique, comma-separated keywords" – so no duplicates. So "cats" once, "tabby" once, "blanket" once, "couch" once, "remotes" once, "pink" once, "resting" once, "indoor" once, "relaxed" once, "home" once, "two" once. That's 10. Wait, 10-18, so maybe add "feline" (same as cat), "lying" (same as resting). Hmm. Let's make sure each keyword is unique. Let's go with:

Title: "Cats on Pink Blanket with Remotes"

Description: Two tabby cats lie on a pink blanket on a red couch, with remote controls nearby, in bright lighting.

Keywords: cats, tabby, blanket,
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words, concrete. Looking at the image: two cats on a pink blanket with remotes. So maybe "Cats on Pink Blanket with Remotes" or something similar. Then description: need to mention the main subject (two cats), setting (pink blanket, red couch), action (lying down), lighting (bright), distinctive details (remotes, cat positions). Then keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home. Let's make sure to omit uncertain details. 

Title: "Cats Resting on Pink Blanket with Remotes" – wait, 5-10 words. Let's check. "Two Cats on Pink Blanket with Remotes" is good. 

Description: Two tabby cats lie on a pink blanket atop a red couch, with remote controls nearby, in bright lighting. – need to make it 1-2 sentences. So: "Two tabby cats rest on a pink blanket on a red couch, with remote controls nearby, under bright lighting." 

Keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home. Let's check unique ones, comma-separated. 

Now, let's structure each part:

Title: "Cats on Pink Blanket with Remotes" (5 words, concrete)

Description: Two tabby cats lie on a pink blanket on a red couch, with remote controls nearby, in bright lighting. (1 sentence, factual)

Keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home (10 words, unique)

Wait, need to make sure all are clearly visible. The image has two cats, pink blanket, red couch, remotes. So description should mention those. Let's adjust description: "Two tabby cats rest on a pink blanket on a red couch, with remote controls nearby, under bright lighting." That's factual, no unsupported details. 

Keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home. Are these unique? Let's see. Maybe add "pet" or "cats" is there. Wait, need 10-18. Let's check: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home, couch, red (but red is part of couch, maybe not). Wait, the couch is red, but maybe include "red couch" as a keyword? But need unique. Let's do: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home, pet, indoor, resting, tabby, two. That's 12, which is between 10-18. 

Wait, let's check the image again. Two tabby cats, so "tabby" can be a keyword. "resting" is action. "indoor" setting. "pink" blanket. "remotes" are there. "couch" is red. So keywords: cats, tabby, blanket, couch, remotes, pink, resting, indoor, relaxed, home, lying, domestic. That's 11, which is good. 

Now, final check:

Title: "Cats on Pink Blanket with Remotes" (5 words, concrete)

Description: Two tabby cats rest on a pink blanket on a red couch, with remote controls nearby, in bright lighting. (1 sentence, factual)

Keywords: cats, tabby, blanket, couch, remotes, pink, resting, indoor, relaxed, home (or add more, but need unique). Wait, the image has two cats, so "two" could be a keyword. Let's do: cats, tabby, blanket, couch, remotes, pink, resting, indoor, relaxed, home, two. That's 11. 

Wait, the user said "unique, comma-separated keywords" – so no duplicates. So "cats" once, "tabby" once, "blanket" once, "couch" once, "remotes" once, "pink" once, "resting" once, "indoor" once, "relaxed" once, "home" once, "two" once. That's 10. Wait, 10-18, so maybe add "feline" (same as cat), "lying" (same as resting). Hmm. Let's make sure each keyword is unique. Let's go with:

Title: "Cats on Pink Blanket with Remotes"

Description: Two tabby cats lie on a pink blanket on a red couch, with remote controls nearby, in bright lighting.

Keywords: cats, tabby, blanket,
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
- *Model load time:* 1.90s
- *Generation time:* 1.47s
- *Total time:* 3.37s
- *Input validation time:* 7.68e-05
- *Prompt preparation time:* 0.00147
- *First-token latency:* 1.04
- *Cleanup time:* 0.104
- *Prompt tokens:* 2,418
- *Generation tokens:* 13
- *Total tokens:* 2,431
- *Prompt throughput (raw):* 2,323 tok/s
- *Generation throughput (raw):* 34.8 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.39
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.00269
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 509
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
 Two cats laying on a pink blanket with remotes.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Two cats laying on a pink blanket with remotes.<end_of_utterance>
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
- *Observations:* Response repeats the same text; Extra text appears before
  the Title field; Response appears cut off at the token limit; Keyword list
  has 313 terms (requested 10-18); Duplicate keywords: household frames
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 1.55s
- *Generation time:* 64.44s
- *Total time:* 65.99s
- *Input validation time:* 7.83e-05
- *Prompt preparation time:* 0.00161
- *First-token latency:* 1.64
- *Cleanup time:* 0.0934
- *Prompt tokens:* 107
- *Generation tokens:* 1,000
- *Total tokens:* 1,107
- *Prompt throughput (raw):* 65.3 tok/s
- *Generation throughput (raw):* 15.9 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.76
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00431
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 576
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
Here is the information in the format you requested:

**Title:** Two Cats on a Pink Blanket

**Description:** Two tabby cats are sleeping on a pink blanket, with two remote controls nearby. The cats are lying on their backs, with their paws stretched out and their eyes closed.

**Keywords:** Tabby cats, pink blanket, remote controls, sleeping, feline, domestic, household, comfort, relaxation, domesticated, household pets, household items, household appliances, household furniture, household decor, household textiles, household linens, household blankets, household comforters, household pillows, household throws, household rugs, household mats, household carpets, household flooring, household walls, household ceilings, household windows, household doors, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames,
</pre>

</details>

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### mlx-community/MiniCPM-V-4.6-8bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.6-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title; Extra text appears before
  the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Model load time:* 0.86s
- *Generation time:* 0.34s
- *Total time:* 1.22s
- *Input validation time:* 8.22e-05
- *Prompt preparation time:* 0.00842
- *First-token latency:* 0.0978
- *Cleanup time:* 0.0841
- *Prompt tokens:* 330
- *Generation tokens:* 58
- *Total tokens:* 388
- *Prompt throughput (raw):* 3,375 tok/s
- *Generation throughput (raw):* 278 tok/s
- *Peak memory:* 3.0
- *Active memory:* 2.3
- *Cache memory:* 0.04
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00482
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 540
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
Two cats sleeping on pink blanket with remote controls

Description: Two tabby cats rest on a pink fabric, with remote controls nearby; soft lighting and cozy setting.

Keywords: cats, sleeping, pink blanket, remote controls, tabby, relaxation, comfort, domestic, soft lighting
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Response repeats the task instructions instead of only
  returning the requested fields; Extra text appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.78s
- *Generation time:* 6.81s
- *Total time:* 7.62s
- *Input validation time:* 8.3e-05
- *Prompt preparation time:* 0.0114
- *First-token latency:* 0.0843
- *Cleanup time:* 0.149
- *Prompt tokens:* 411
- *Generation tokens:* 872
- *Total tokens:* 1,283
- *Prompt throughput (raw):* 4,875 tok/s
- *Generation throughput (raw):* 130 tok/s
- *Peak memory:* 5.3
- *Active memory:* 4.3
- *Cache memory:* 0.19
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00805
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 552
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
Got it, let's tackle this. First, the title needs to be concrete and 5-10 words. The image has two cats on a pink couch with remotes. So maybe "Two tabby cats resting on pink couch with remotes" but that's 10 words. Wait, "Two tabby cats resting on pink couch with remote controls" but need to check. Wait, the problem says "British-English catalogue metadata", so maybe "Two tabby cats resting on pink couch with remote controls" but let's count: Two tabby cats resting on pink couch with remote controls is 10 words. That's good.

Next, the description. Need 1-2 sentences. Main subject: two cats. Setting: pink couch. Action: resting. Lighting: bright (since it's a photo, probably natural light). Distinctive details: tabby patterns, remotes. So "Two tabby cats rest on a bright pink couch, one with a green collar, while two remote controls lie nearby."

Wait, but the problem says "omit uncertain details and unsupported identity, location, event, brand, species, period, or intent." So species is cat, which is known, so okay. The remotes are visible, so "two remote controls" is okay. Lighting: the image is bright, so "bright lighting" or "natural light". Let's make it concise.

Description: "Two tabby cats rest on a pink couch, one with a green collar, while two remote controls lie nearby under the couch."

Wait, but the problem says "1-2 sentences". Let's check: "Two tabby cats rest on a bright pink couch, one with a green collar, while two remote controls lie nearby." That's two sentences. Wait, no, it's one sentence. Wait, the example in the problem says "1-2 sentences". So maybe "Two tabby cats rest on a pink couch, one with a green collar, while two remote controls lie nearby." That's one sentence. Wait, but the problem says "1-2 sentences", so that's okay.

Now keywords: 10-18 unique, comma-separated. Need to list distinct things. Let's see: cats, tabby, pink couch, remote controls, resting, two, couch, bright, collar, green, natural light. Wait, but need to be unique. Let's list: cats, tabby, pink, couch, remote, resting, collar, green, natural, light. Wait, but need 10-18. Let's check the image: two cats, pink couch, remotes, tabby pattern, resting, collar (green), bright lighting. So keywords: cats, tabby, pink, couch, remote, resting, collar, green, natural, light, two. Wait, but need to be unique. Let's make sure they are distinct. Let's list: cats, tabby, pink, couch, remote, resting, collar, green, natural, light, two. Wait, but "two" is a number, but maybe "two" is okay. Wait, the problem says "unique, comma-separated keywords". So "cats, tabby, pink, couch, remote, resting, collar, green, natural, light, two" but that's 11. Wait, maybe "cats, tabby, pink, couch, remote, resting, collar, green, natural, light, two" but let's check if they are unique. Yes, all are unique. Wait, but maybe "two" is redundant. Wait, the problem says "unique", so "two" is a count, but maybe it's okay. Wait, but let's see. Let's check the image: two cats, so "two" is a keyword. Let's make sure. Let's list: cats, tabby, pink
&lt;/think&gt;

Title:
Two tabby cats resting on pink couch with remotes

Description:
Two tabby cats rest on a bright pink couch, one with a green collar, while two remote controls lie nearby under the couch.

Keywords: cats, tabby, pink, couch, remote, resting, collar, green, natural, light, two
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this. First, the title needs to be concrete and 5-10 words. The image has two cats on a pink couch with remotes. So maybe "Two tabby cats resting on pink couch with remotes" but that's 10 words. Wait, "Two tabby cats resting on pink couch with remote controls" but need to check. Wait, the problem says "British-English catalogue metadata", so maybe "Two tabby cats resting on pink couch with remote controls" but let's count: Two tabby cats resting on pink couch with remote controls is 10 words. That's good.

Next, the description. Need 1-2 sentences. Main subject: two cats. Setting: pink couch. Action: resting. Lighting: bright (since it's a photo, probably natural light). Distinctive details: tabby patterns, remotes. So "Two tabby cats rest on a bright pink couch, one with a green collar, while two remote controls lie nearby."

Wait, but the problem says "omit uncertain details and unsupported identity, location, event, brand, species, period, or intent." So species is cat, which is known, so okay. The remotes are visible, so "two remote controls" is okay. Lighting: the image is bright, so "bright lighting" or "natural light". Let's make it concise.

Description: "Two tabby cats rest on a pink couch, one with a green collar, while two remote controls lie nearby under the couch."

Wait, but the problem says "1-2 sentences". Let's check: "Two tabby cats rest on a bright pink couch, one with a green collar, while two remote controls lie nearby." That's two sentences. Wait, no, it's one sentence. Wait, the example in the problem says "1-2 sentences". So maybe "Two tabby cats rest on a pink couch, one with a green collar, while two remote controls lie nearby." That's one sentence. Wait, but the problem says "1-2 sentences", so that's okay.

Now keywords: 10-18 unique, comma-separated. Need to list distinct things. Let's see: cats, tabby, pink couch, remote controls, resting, two, couch, bright, collar, green, natural light. Wait, but need to be unique. Let's list: cats, tabby, pink, couch, remote, resting, collar, green, natural, light. Wait, but need 10-18. Let's check the image: two cats, pink couch, remotes, tabby pattern, resting, collar (green), bright lighting. So keywords: cats, tabby, pink, couch, remote, resting, collar, green, natural, light, two. Wait, but need to be unique. Let's make sure they are distinct. Let's list: cats, tabby, pink, couch, remote, resting, collar, green, natural, light, two. Wait, but "two" is a number, but maybe "two" is okay. Wait, the problem says "unique, comma-separated keywords". So "cats, tabby, pink, couch, remote, resting, collar, green, natural, light, two" but that's 11. Wait, maybe "cats, tabby, pink, couch, remote, resting, collar, green, natural, light, two" but let's check if they are unique. Yes, all are unique. Wait, but maybe "two" is redundant. Wait, the problem says "unique", so "two" is a count, but maybe it's okay. Wait, but let's see. Let's check the image: two cats, so "two" is a keyword. Let's make sure. Let's list: cats, tabby, pink
</think>

Title:
Two tabby cats resting on pink couch with remotes

Description:
Two tabby cats rest on a bright pink couch, one with a green collar, while two remote controls lie nearby under the couch.

Keywords: cats, tabby, pink, couch, remote, resting, collar, green, natural, light, two
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
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Missing or empty fields:
  Title, Description, Keywords; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type llava_next)
- *Model load time:* 0.91s
- *Generation time:* 20.42s
- *Total time:* 21.33s
- *Input validation time:* 9.01e-05
- *Prompt preparation time:* 0.00187
- *First-token latency:* 2.11
- *Cleanup time:* 0.14
- *Prompt tokens:* 2,469
- *Generation tokens:* 1,000
- *Total tokens:* 3,469
- *Prompt throughput (raw):* 1,170 tok/s
- *Generation throughput (raw):* 54.7 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.92
- *Model-load active memory:* 8.05
- *Post-cleanup active memory:* 0.0125
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 474
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
cat, sleeping, pink blanket, remote control, striped, tabby, kitten, cozy, domestic, indoor, relaxation, comfort, home, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline
</pre>

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
- *Model load time:* 0.49s
- *Generation time:* 0.34s
- *Total time:* 0.83s
- *Input validation time:* 8.98e-05
- *Prompt preparation time:* 0.00172
- *First-token latency:* 0.072
- *Cleanup time:* 0.0873
- *Prompt tokens:* 116
- *Generation tokens:* 97
- *Total tokens:* 213
- *Prompt throughput (raw):* 1,611 tok/s
- *Generation throughput (raw):* 381 tok/s
- *Peak memory:* 2.1
- *Active memory:* 0.61
- *Cache memory:* 0.25
- *Model-load active memory:* 0.611
- *Post-cleanup active memory:* 0.0131
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 558
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
Title: "A Study in Stripes"
Description: A close-up of a person wearing a striped shirt, with a focus on the pattern and color of the fabric. The image is set against a pink background, and the person is lying on a bed covered with a pink blanket. The lighting is soft, and the person is facing away from the camera, giving a sense of intimacy. The image is a straightforward representation of the subject, with no additional details or context provided.
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
  Title; Response repeats the task instructions instead of only returning the
  requested fields; Extra text appears before the Title field; Response
  appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Model load time:* 1.15s
- *Generation time:* 22.62s
- *Total time:* 23.79s
- *Input validation time:* 8.77e-05
- *Prompt preparation time:* 0.000209
- *First-token latency:* 1.1
- *Cleanup time:* 0.136
- *Prompt tokens:* 4,207
- *Generation tokens:* 1,000
- *Total tokens:* 5,207
- *Prompt throughput (raw):* 3,810 tok/s
- *Generation throughput (raw):* 47.0 tok/s
- *Peak memory:* 4.2
- *Active memory:* 1.7
- *Cache memory:* 2.0
- *Model-load active memory:* 1.73
- *Post-cleanup active memory:* 0.0141
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 458
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
Action:
Lighting:
Brand:
Species:
Period:
Event:
Date:
Name:
Intent:
Brand:
Date:
Name:
Description:
Action:
Lighting:
Keywords:
Return exactly these three sections and nothing else:
Title:
Description:
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-1
</pre>

</details>

---
