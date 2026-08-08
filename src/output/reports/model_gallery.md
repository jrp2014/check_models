# Model Output Gallery

Generated on: 2026-08-09 00:14:54 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                                                      |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|------------|---------|-----------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | 13.10s  | 21.0 tok/s | 5.16            | 23      | 2,389      | 89      | none                                                                                                                              |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 6.85s   | 29.7 tok/s | 1.04            | 17      | 2,111      | 89      | none                                                                                                                              |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | 3.06s   | 158 tok/s  | 0.13            | 4.1     | 579        | 124     | none                                                                                                                              |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | 13.79s  | 11.2 tok/s | 2.77            | 15      | 300        | 90      | none                                                                                                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 9.86s   | 48.0 tok/s | 3.80            | 13      | 2,922      | 157     | none                                                                                                                              |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 4.47s   | 160 tok/s  | 1.20            | 7.8     | 2,921      | 128     | none                                                                                                                              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | 5.23s   | 55.7 tok/s | 0.29            | 9.4     | 1,125      | 168     | none                                                                                                                              |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | 27.04s  | 91.9 tok/s | 23.59           | 8.4     | 16,545     | 129     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | 11.84s  | 22.9 tok/s | 1.98            | 18      | 2,611      | 143     | role tokens visible                                                                                                               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | 9.54s   | 44.9 tok/s | 3.30            | 13      | 2,922      | 151     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | 6.23s   | 46.2 tok/s | 0.85            | 11      | 1,511      | 106     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | 27.36s  | 86.4 tok/s | 23.62           | 8.4     | 16,545     | 129     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | 4.55s   | 108 tok/s  | 0.41            | 5.5     | 424        | 113     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable_with_caveats` | 28.15s  | 47.5 tok/s | 21.58           | 13      | 16,556     | 195     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | 10.32s  | 25.7 tok/s | 1.12            | 18      | 590        | 148     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | 7.98s   | 37.7 tok/s | 2.03            | 16      | 3,112      | 118     | title/keyword constraints failed                                                                                                  |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `unusable`            | 3.29s   | 497 tok/s  | 0.04            | 1.2     | 389        | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 34.94s  | 33.4 tok/s | 2.32            | 15      | 3,203      | 1,000   | echoes instructions; extra text before Title; cut off at token limit; title/keyword constraints failed                            |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 47.18s  | 35.2 tok/s | 6.79            | 60      | 1,629      | 1,000   | repeated text; extra text before Title; cut off at token limit; wrapper tokens visible; title/keyword constraints failed          |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 3.60s   | 320 tok/s  | 0.11            | 1.9     | 332        | 173     | missing required fields                                                                                                           |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 40.77s  | 36.8 tok/s | 10.57           | 13      | 6,446      | 1,000   | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                      |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `unusable`            | 19.11s  | 58.9 tok/s | 8.77            | 8.4     | 6,446      | 420     | extra text before Title                                                                                                           |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `unusable`            | 79.79s  | 24.1 tok/s | 25.93           | 78      | 6,446      | 1,000   | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                      |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | 127.93s | 3.85 tok/s | 2.05            | 40      | 1,321      | 463     | missing required fields; extra text before Title; role tokens visible                                                             |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 4.74s   | 203 tok/s  | 0.27            | 3.7     | 930        | 412     | echoes instructions; extra text before Title; wrapper tokens visible                                                              |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | 35.35s  | 5.89 tok/s | 2.94            | 24      | 3,124      | 166     | missing required fields                                                                                                           |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `unusable`            | 91.28s  | 59.0 tok/s | 70.30           | 74      | 16,557     | 435     | extra text before Title; wrapper tokens visible                                                                                   |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `unusable`            | 88.24s  | 172 tok/s  | 85.53           | 5.1     | 16,556     | 148     | missing required fields                                                                                                           |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | 33.48s  | 56.5 tok/s | 23.46           | 8.4     | 16,547     | 428     | extra text before Title; wrapper tokens visible; title/keyword constraints failed                                                 |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `unusable`            | 93.44s  | 98.7 tok/s | 83.83           | 24      | 16,557     | 402     | extra text before Title; wrapper tokens visible                                                                                   |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `unusable`            | 86.56s  | 68.6 tok/s | 77.32           | 10.0    | 16,557     | 427     | extra text before Title; wrapper tokens visible                                                                                   |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `unusable`            | 151.03s | 12.1 tok/s | 108.65          | 35      | 16,557     | 450     | extra text before Title; wrapper tokens visible; title/keyword constraints failed                                                 |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `unusable`            | 50.49s  | 39.5 tok/s | 25.07           | 70      | 3,485      | 438     | extra text before Title; wrapper tokens visible                                                                                   |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `unusable`            | 17.60s  | 79.1 tok/s | 0.51            | 29      | 598        | 983     | repeated text; echoes instructions; extra text before Title                                                                       |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `unusable`            | 15.61s  | 86.0 tok/s | 0.47            | 28      | 598        | 915     | echoes instructions; extra text before Title; title/keyword constraints failed                                                    |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 6.85s   | 34.9 tok/s | 0.65            | 17      | 589        | 87      | missing required fields                                                                                                           |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `unusable`            | 16.87s  | 79.9 tok/s | 0.57            | 16      | 598        | 1,000   | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `unusable`            | 48.95s  | 19.4 tok/s | 1.14            | 20      | 598        | 857     | missing required fields; echoes instructions; extra text before Title                                                             |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | 6.75s   | 59.9 tok/s | 2.78            | 9.7     | 2,514      | 100     | missing required fields; extra text before Title                                                                                  |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 1.93s   | 330 tok/s  | 0.15            | 2.1     | 328        | 70      | missing required fields                                                                                                           |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 32.53s  | 34.7 tok/s | 1.16            | 4.5     | 4,416      | 1,000   | repeated text; missing required fields; echoes instructions; cut off at token limit                                               |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                                   | `not_evaluated`       | 1.07s   | -          | -               | -       | -          | -       | none                                                                                                                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `mlx-community/Ministral-3-3B-Instruct-2512-4bit` at 160 tok/s

Average clean-completion throughput: 69.1 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `mlx-community/LFM2.5-VL-1.6B-bf16` at 4.1 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                 | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | echoes instructions; extra text before Title; cut off at token limit; title/keyword constraints failed                            |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable`      | repeated text; extra text before Title; cut off at token limit; wrapper tokens visible; title/keyword constraints failed          |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable`      | missing required fields                                                                                                           |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable`      | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                      |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                             | `unusable`      | extra text before Title                                                                                                           |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                         | `unusable`      | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                      |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | `unusable`      | missing required fields; extra text before Title; role tokens visible                                                             |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                 | `unusable`      | echoes instructions; extra text before Title; wrapper tokens visible                                                              |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                | `unusable`      | missing required fields                                                                                                           |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                               | `unusable`      | extra text before Title; wrapper tokens visible                                                                                   |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                  | `unusable`      | missing required fields                                                                                                           |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable`      | extra text before Title; wrapper tokens visible; title/keyword constraints failed                                                 |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                             | `unusable`      | extra text before Title; wrapper tokens visible                                                                                   |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                               | `unusable`      | extra text before Title; wrapper tokens visible                                                                                   |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                   | `unusable`      | extra text before Title; wrapper tokens visible; title/keyword constraints failed                                                 |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                               | `unusable`      | extra text before Title; wrapper tokens visible                                                                                   |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | `unusable`      | repeated text; echoes instructions; extra text before Title                                                                       |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | `unusable`      | echoes instructions; extra text before Title; title/keyword constraints failed                                                    |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                            | `unusable`      | missing required fields                                                                                                           |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                      | `unusable`      | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                              | `unusable`      | missing required fields; echoes instructions; extra text before Title                                                             |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable`      | missing required fields; extra text before Title                                                                                  |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable`      | missing required fields                                                                                                           |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable`      | repeated text; missing required fields; echoes instructions; cut off at token limit                                               |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                        | `not_evaluated` | none                                                                                                                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | Title:<br>Akeman Street Baptist Church, Tring<br><br>Description:<br>A red-brick Baptist church in Tring, Hertfordshire, with arched windows and a central entrance, set against a partly cloudy sky.<br><br>Keywords:<br>Akeman Street Baptist Church, Tring, Hertfordshire, red-brick church, arched win... |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | Title: Akeman Street Baptist Church, Tring<br>Description: The image captures the Akeman Street Baptist Church in Tring, Hertfordshire, showcasing its red brick facade and arched windows under a partly cloudy sky.<br>Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, Engl...                |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | Title: Akeman Street Baptist Church, Tring, Hertfordshire, England<br><br>Description: This red brick church stands prominently on Akeman Street in Tring, Hertfordshire, England. Its distinctive red roof and arched windows are complemented by a small tower at the top. The church is...                 |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | Title: Red Brick Church in Tring's Town Centre<br>Description: The Akeman Street Baptist Church, a red brick building with a distinctive roof, stands in the heart of Tring, Hertfordshire, on a sunny day with a clear blue sky.<br>Keywords: Akeman Street Baptist Church, red brick, chu...                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**Akeman Street Baptist Church, Tring – Historic Red Brick Exterior**<br><br>Description:<br>A well-preserved red-brick Baptist church on Akeman Street, Tring, Hertfordshire, captured on a bright August afternoon under partly cloudy skies. The building features arched windows, a...          |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>Akeman Street Baptist Church, Tring – Historic Brick Chapel<br><br>Description:<br>The Akeman Street Baptist Church in Tring, Hertfordshire, stands prominently in a quiet town centre, constructed from traditional red brick with arched windows and a slate roof. Its elevated entra...          |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | Title: Akeman Street Baptist Church in Tring, Hertfordshire<br><br>Description: The Akeman Street Baptist Church, located in the heart of Tring, Hertfordshire, England, stands as a prominent red brick structure with a distinctive entrance and a steeply pitched roof. Surrounded by lu...                |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | Title: Akeman Street Baptist Church, Tring, Herts<br><br>Description: The Akeman Street Baptist Church in Tring, Hertfordshire, England, is a red brick building with a distinctive round roof and arched windows, surrounded by lush greenery and a stone wall, under a partly cloudy sky....                |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | Title: Akeman Street Baptist Church in Tring, Hertfordshire, England.<br>Description: The Akeman Street Baptist Church, a red-brick building with a grey slate roof and multiple arched windows, is situated in Tring, Hertfordshire, England. The church is surrounded by trees and has...                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | **Title:**<br>Akeman Street Baptist Church, Tring, Hertfordshire<br><br>**Description:**<br>A historic red-brick Baptist church on Akeman Street, Tring, captured on **August 8, 2026 (UTC+01:00)**, showcasing its arched windows, brickwork entrance, and well-maintained garden. The scene fea...          |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | Title: Akeman Street Baptist Church, Tring, Hertfordshire, England<br><br>Description: A red brick Baptist church stands prominently in Tring's town center, surrounded by bushes and a stone wall, with a red sign marking its entrance. The scene is bathed in sunlight, creating a warm...                 |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | Title: Akeman Street Baptist Church, Tring, Herts<br><br>Description: The Akeman Street Baptist Church in Tring, Hertfordshire, England, is a red brick building with a distinctive round roof and arched windows, surrounded by lush greenery and a stone wall, under a partly cloudy sky....                |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | Title: Akeman Street Baptist Church, Tring, Herts<br>Description: A red brick church with a large chimney and a sign stands in the town centre of Tring, Herts. The church is surrounded by trees and has a blue sky with clouds in the background.<br>Keywords: Adobe Stock, Akeman Street...                |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable_with_caveats` | Title: Akeman Street Baptist Church, Tring<br><br>Description: A historic red-brick Baptist church in Tring, Hertfordshire, England, stands under a partly cloudy sky, with lush greenery and a stone entrance, signifying its role as a community hub.<br><br>Keywords: Akeman Street Baptist Ch...          |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | Title: Akeman Street Baptist Church, Tring, Hertfordshire<br><br>Description: Captured on 8th August 2026 at 15:43, this photograph shows the red brick Akeman Street Baptist Church in Tring, Hertfordshire, bathed in afternoon sunlight; a stone wall and steps lead to the building’s c...                |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | Title:<br>Akeman Street Baptist Church, Tring, England<br><br>Description:<br>The Akeman Street Baptist Church in Tring, England, is a red-brick building with arched windows and a central entrance, surrounded by lush greenery under a partly cloudy sky.<br><br>Keywords:<br>Akeman Street Baptist Chu... |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `unusable`            | Title:<br>Akeman Street Baptist Church, Tring, England, UK, GBR, Europe<br><br>Description:<br>A brick-built church with a red door and arched windows stands in a serene town center setting, surrounded by lush greenery and a clear sky.<br><br>Keywords:<br>Akeman Street Baptist Church, Tring, Engla... |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The image shows a brick church building, presumably Akeman Street Baptist Church in Tring, Hertfordshire (though Tring is in Hertfordshire, not Buckinghamshi...                   |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | Okay, let's break this down. I've got an image of a church and need to create some metadata for it. First, I need a title. The hints are a bit messy, but I see "Akeman Street Baptist Church" and "Tring, Herts" in there. That's a good start. I'll go with "Akeman Street Baptist...                       |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A concrete 5-10-word title: Akeman Street Baptist Church, Tring, Herts, UK, GBR, Europe<br>A 1-2-sentence factual description: The image depicts the Akeman Street Baptist Church in Tring, Herts, UK, captured on 2026-08-08 15:43:55 UTC+01:00. The church is a 1-2-story red brick bu...                   |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. The main subject is the Akeman Street Baptist Church, so maybe "Akeman Street Baptist Church, Tring". Then the description should combine context (date, location) with the main subject, setting, etc. Le...                |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `unusable`            | &lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image, the main subject is Akeman Street Baptist Church. So a good title would be "Akeman Street Baptist Church, Tring".<br><br>Next, the description. It needs to be 1-2 sentences, combining co...          |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `unusable`            | &lt;think&gt;Got it, let's tackle this step by step. First, the title needs to be 5-10 words, concrete. The main subject is Akeman Street Baptist Church, so maybe "Akeman Street Baptist Church, Tring" but wait, need to be 5-10 words. Let's see, "Akeman Street Baptist Church, Tring,...                 |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | ◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the provided context and image. The capture date and GPS are authoritative, but I shouldn't claim they're visible. The title hint is "Town centre, Tring, England, UK, GBR, Eur...                      |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | We need to create the catalogue metadata. Let's break down the required sections:<br><br>1. Title: needs to be a concrete 5-10 word title.<br>2. Description: needs to be a 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting,...              |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | Akeman Street Baptist Church, Tring, Hertfordshire, England<br><br>The image shows the Akeman Street Baptist Church in Tring, Hertfordshire, England. Built of red brick with a distinctive arched doorway and three arched windows, the church stands behind a brick wall and gate. The sc...                |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `unusable`            | The user wants metadata for an image of a church.<br><br>**1. Analyze the Image:**<br>* **Subject:** A red brick building, clearly a church.<br>* **Signage:** A sign on the left reads "Akeman Street Baptist Church" with service times "Sunday Services 10:30am &amp; 6:00pm" and a website "www.a...      |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `unusable`            | Title:<br>Akeman Street Baptist Church, Tring, Herts, England, UK, GBR, Europe<br><br>Description:<br>Akeman Street Baptist Church is a historic building located in Tring, Herts, England. The church was built in 1808 and underwent a significant renovation in 1832. The building features a...           |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | Got it, let's tackle this step by step. First, the title. The context says it's Akeman Street Baptist Church in Tring, Herts. The descriptive hints mention that. So a title should be concise. Maybe "Akeman Street Baptist Church, Tring, Herts, England" but need to be 5-10 words...                      |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `unusable`            | The user wants metadata for an image of a church.<br><br>**1. Analyze the Image:**<br>* **Subject:** A red brick building, clearly a church.<br>* **Signage:** A red sign on the left reads "Akeman Street Baptist Church". It also lists Sunday Services times.<br>* **Architecture:** Red brick fa...       |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `unusable`            | The user wants British-English catalogue metadata for the provided image.<br><br>**1. Analyze the Image:**<br>* **Subject:** A red brick building, clearly a church.<br>* **Signage:** A red sign on the left reads "Akeman Street Baptist Church".<br>* **Architecture:**<br>* Red brick facade.<br>* Thr... |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `unusable`            | The user wants British-English catalogue metadata for the provided image.<br><br>**1. Analyze the Image:**<br>* **Subject:** A large, red brick building. It looks like a church or chapel.<br>* **Signage:** There is a red sign on the left that says "Akeman Street Baptist Church". It also l...          |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `unusable`            | Got it, let's tackle this step by step. First, the title needs to be 5-10 words, concrete. The subject is Akeman Street Baptist Church in Tring, England, right? Wait, let's make it accurate. Oh, "Akeman Street Baptist Church, Tring, Hertfordshire" that's good, count words: Ake...                      |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `unusable`            | <\|channel>thought<br>I need to create British-English catalogue metadata for the provided image and context.<br><br>- Image capture date/time: 2026-08-08 15:43:55 UTC+01:00<br>- GPS: 51.815915°N, 0.638706°W<br>- Title hint: Town centre, Tring, England, UK, GBR, Europe<br>- Description hint: Ake...   |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `unusable`            | <\|channel>thought<br>* Image: A two-story red brick building, labeled "Akeman Street Baptist Church".<br>* Capture date: 2026-08-08 15:43:55 UTC+01:00<br>* GPS: 51.815915°N, 0.638706°W<br>* Title hint: Town centre, Tring, England, UK, GBR, Europe<br>* Description hint: Akeman Street Baptist...       |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | A red brick building with arched windows and a dark red door stands on a sunny day in Tring, Hertfordshire, UK. The building is the Akeman Street Baptist Church, as indicated by a sign in the foreground. The church has a grey tiled roof and is surrounded by greenery, including...                      |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `unusable`            | <\|channel>thought<br>* Capture date/time: 2026-08-08 15:43:55 UTC+01:00<br>* GPS: 51.815915°N, 0.638706°W<br>* Title hint: Town centre, Tring, England, UK, GBR, Europe<br>* Description hint: Akeman Street Baptist Church, Tring, Herts<br>* Keywords: Adobe Stock, Akeman Street Baptist Church,...       |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `unusable`            | <\|channel>thought<br>* Image: A red brick building, identified as Akeman Street Baptist Church by a sign. It's a two-story structure with arched windows and a central arched door. There's a low stone wall in front with steps leading up to the entrance. Greenery (bushes, trees) su...                  |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | Akeman Street Baptist Church, Tring, Herts, England, UK, GBR, Europe<br><br>Title:<br>Description:<br>Keywords: [/Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations...          |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | Title: Akeman Street Baptist Church, Tring, England, UK, GBR, Europe<br>Description: A brick church in the town centre of Tring, England, UK, GBR, Europe. The church is a red brick building with a red roof and a sign on the front. It is a prominent landmark in the area.                                |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | - Title hint: Tring, England, UK, GBR, Europe<br>- Description hint: Tring, England, UK, GBR, Europe<br>- Keyword hints: Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe...                 |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                                   | `not_evaluated`       | Model loading failed: Received 362 parameters not in model:<br>audio_tower.encoder.biases,<br>audio_tower.encoder.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.biases,<br>language_model.model.layers.10.mlp.experts.down_proj.scales,<br>language_model.model.layers.10.mlp.expe...       |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.11`
- `mlx`: `0.32.1.dev20260808+8d6662986`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.27.0`
- *Python Version:* 3.13.13
- *OS:* Darwin 25.6.0
- *macOS Version:* 26.6
- *GPU/Chip:* Apple M5 Max
- *MLX Device:* Apple M5 Max
- *GPU Architecture:* applegpu_g17s
- *RAM:* 128.0 GB
- *Recommended Working Set:* 108 GB
- *Fused Attention:* Available

## Image Metadata

- *Title:* , Town centre, Tring, England, UK, GBR, Europe
- *Description:* Akeman Street Baptist Church, Tring, Herts
- *Keywords:* Adobe Stock, Akeman Street Baptist Church, Any Vision,
  Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds,
  England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick,
  Roof, Sign, Sky, Town, Town Centre, Trees, Tring, UK, Wall, arched windows,
  architecture, baptist, brick, british, building, christianity, day, doors,
  double doors, english, exterior, facade, gate, heritage, historic, nobody,
  old, outdoors, place of worship, religion, steps, street, summer,
  traditional, windows
- *Date:* 2026-08-08 15:43:55 UTC+01:00
- *Time:* 15:43:55
- *GPS:* 51.815915°N, 0.638706°W

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
> &#45; Capture date/time: 2026-08-08 15:43:55 UTC+01:00
> &#45; GPS: 51.815915°N, 0.638706°W
>
> &#8203;Descriptive hints:
> &#45; Title hint: Town centre, Tring, England, UK, GBR, Europe
> &#45; Description hint: Akeman Street Baptist Church, Tring, Herts
> &#45; Keyword hints: Adobe Stock, Akeman Street Baptist Church, Any Vision,
> Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds,
> England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick,
> Roof, Sign, Sky
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

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

<details>
<summary>Complete evidence: mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.32s
- *Generation time:* 10.07s
- *Total time:* 13.10s
- *Input validation time:* 0.689
- *Prompt preparation time:* 0.00712
- *First-token latency:* 5.16
- *Cleanup time:* 0.166
- *Prompt tokens:* 2,389
- *Generation tokens:* 89
- *Total tokens:* 2,478
- *Prompt throughput (raw):* 463 tok/s
- *Generation throughput (raw):* 21.0 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 0.47
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.000968
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1252
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Devstral-Small-2-24B-Instruct-2512-5bit/snapshots/0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Akeman Street Baptist Church, Tring

Description:
A red-brick Baptist church in Tring, Hertfordshire, with arched windows and a central entrance, set against a partly cloudy sky.

Keywords:
Akeman Street Baptist Church, Tring, Hertfordshire, red-brick church, arched windows, central entrance, partly cloudy sky, church building, religious architecture, historic church, UK, England, Europe
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
- *Generation time:* 4.77s
- *Total time:* 6.85s
- *Input validation time:* 0.46
- *Prompt preparation time:* 0.00176
- *First-token latency:* 1.04
- *Cleanup time:* 0.216
- *Prompt tokens:* 2,111
- *Generation tokens:* 89
- *Total tokens:* 2,200
- *Prompt throughput (raw):* 2,035 tok/s
- *Generation throughput (raw):* 29.7 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.17
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.003
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1257
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.internvl_chat.processor.InternVLChatProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3-8B-bf16/snapshots/e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Akeman Street Baptist Church, Tring
Description: The image captures the Akeman Street Baptist Church in Tring, Hertfordshire, showcasing its red brick facade and arched windows under a partly cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, arched windows, partly cloudy sky, entrance, bushes, chapel, sign, church, location, objects
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
- *Model load time:* 0.64s
- *Generation time:* 1.54s
- *Total time:* 3.06s
- *Input validation time:* 0.869
- *Prompt preparation time:* 0.0039
- *First-token latency:* 0.132
- *Cleanup time:* 0.0964
- *Prompt tokens:* 579
- *Generation tokens:* 124
- *Total tokens:* 703
- *Prompt throughput (raw):* 4,400 tok/s
- *Generation throughput (raw):* 158 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.02
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.00379
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1303
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-1.6B-bf16/snapshots/16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Akeman Street Baptist Church, Tring, Hertfordshire, England

Description: This red brick church stands prominently on Akeman Street in Tring, Hertfordshire, England. Its distinctive red roof and arched windows are complemented by a small tower at the top. The church is surrounded by greenery, with bushes and trees adding to its serene setting.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, arched windows, red roof, tower, greenery, bushes, church, churchyard, church entrance, European architecture
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
- *Model load time:* 1.68s
- *Generation time:* 11.61s
- *Total time:* 13.79s
- *Input validation time:* 0.499
- *Prompt preparation time:* 0.0014
- *First-token latency:* 2.77
- *Cleanup time:* 0.123
- *Prompt tokens:* 300
- *Generation tokens:* 90
- *Total tokens:* 390
- *Prompt throughput (raw):* 108 tok/s
- *Generation throughput (raw):* 11.2 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.57
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00431
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1356
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-11B-Vision-Instruct-8bit/snapshots/8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|eot_id\|>

*Readable output:*

<pre class="model-output-readable">
Title: Red Brick Church in Tring's Town Centre
Description: The Akeman Street Baptist Church, a red brick building with a distinctive roof, stands in the heart of Tring, Hertfordshire, on a sunny day with a clear blue sky.
Keywords: Akeman Street Baptist Church, red brick, church, Tring, Hertfordshire, England, UK, Europe, town centre, sunny day, blue sky.
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
- *Model load time:* 1.54s
- *Generation time:* 7.77s
- *Total time:* 9.86s
- *Input validation time:* 0.541
- *Prompt preparation time:* 0.0106
- *First-token latency:* 3.8
- *Cleanup time:* 0.136
- *Prompt tokens:* 2,922
- *Generation tokens:* 157
- *Total tokens:* 3,079
- *Prompt throughput (raw):* 769 tok/s
- *Generation throughput (raw):* 48.0 tok/s
- *Peak memory:* 13
- *Active memory:* 8.0
- *Cache memory:* 0.60
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00508
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3690
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 7c992876448fc5426841a85c6eb951b056fe38d2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-mxfp4/snapshots/7c992876448fc5426841a85c6eb951b056fe38d2
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
**Akeman Street Baptist Church, Tring – Historic Red Brick Exterior**

Description:
A well-preserved red-brick Baptist church on Akeman Street, Tring, Hertfordshire, captured on a bright August afternoon under partly cloudy skies. The building features arched windows, a central entrance with steps, and a small graveyard with memorials. A sign confirms its identity, and the surrounding greenery adds to the serene setting.

Keywords:
Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, red-brick church, arched windows, central entrance, graveyard, memorials, signpost, partly cloudy sky, August 2026, historic architecture, brick construction, church exterior, UK heritage site
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
- *Model load time:* 1.08s
- *Generation time:* 2.73s
- *Total time:* 4.47s
- *Input validation time:* 0.641
- *Prompt preparation time:* 0.00769
- *First-token latency:* 1.2
- *Cleanup time:* 0.103
- *Prompt tokens:* 2,921
- *Generation tokens:* 128
- *Total tokens:* 3,049
- *Prompt throughput (raw):* 2,441 tok/s
- *Generation throughput (raw):* 160 tok/s
- *Peak memory:* 7.8
- *Active memory:* 2.8
- *Cache memory:* 0.38
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.0056
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3689
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* a962dcb09eee4169c890e544c9eb938f1113fdee
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-3B-Instruct-2512-4bit/snapshots/a962dcb09eee4169c890e544c9eb938f1113fdee
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Akeman Street Baptist Church, Tring – Historic Brick Chapel

Description:
The Akeman Street Baptist Church in Tring, Hertfordshire, stands prominently in a quiet town centre, constructed from traditional red brick with arched windows and a slate roof. Its elevated entrance is framed by a small brick wall and flanked by stone monuments, bathed in soft daylight filtering through scattered clouds.

Keywords:
Akeman Street Baptist Church, brick chapel, red brick, arched windows, slate roof, stone monuments, brick wall, Tring, Hertfordshire, UK, historic building, chapel entrance, natural lighting, urban setting, heritage architecture
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
- *Model load time:* 0.91s
- *Generation time:* 3.87s
- *Total time:* 5.23s
- *Input validation time:* 0.46
- *Prompt preparation time:* 0.000873
- *First-token latency:* 0.289
- *Cleanup time:* 0.0834
- *Prompt tokens:* 1,125
- *Generation tokens:* 168
- *Total tokens:* 1,293
- *Prompt throughput (raw):* 3,894 tok/s
- *Generation throughput (raw):* 55.7 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.62
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00711
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1273
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Akeman Street Baptist Church in Tring, Hertfordshire

Description: The Akeman Street Baptist Church, located in the heart of Tring, Hertfordshire, England, stands as a prominent red brick structure with a distinctive entrance and a steeply pitched roof. Surrounded by lush greenery and a serene atmosphere, the church's entrance is marked by a stone wall and a signpost, inviting visitors to explore its historical and spiritual significance.

Keywords: Akeman Street, Baptist Church, Tring, Hertfordshire, England, red brick, entrance, stone wall, signpost, greenery, steeply pitched roof, historical, spiritual, serene, lush, inviting, historical significance, spiritual significance
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
- *Observations:* Keyword list has 23 terms (requested 10-18); Duplicate
  keywords: england, sky, entrance
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.68s
- *Generation time:* 25.94s
- *Total time:* 27.04s
- *Input validation time:* 0.413
- *Prompt preparation time:* 0.00958
- *First-token latency:* 23.6
- *Cleanup time:* 0.082
- *Prompt tokens:* 16,545
- *Generation tokens:* 129
- *Total tokens:* 16,674
- *Prompt throughput (raw):* 701 tok/s
- *Generation throughput (raw):* 91.9 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1324
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Akeman Street Baptist Church, Tring, Herts

Description: The Akeman Street Baptist Church in Tring, Hertfordshire, England, is a red brick building with a distinctive round roof and arched windows, surrounded by lush greenery and a stone wall, under a partly cloudy sky.

Keywords: Akeman Street Baptist Church, Tring, Herts, England, Church, Red brick, Roof, Arched windows, Greenery, Stone wall, Sky, Clouds, Entrance, Chimney, Bushes, Christian, Locations, Objects, Hertfordshire, Europe, England, Entrance, Sky
</pre>

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
- *Model load time:* 2.06s
- *Generation time:* 9.08s
- *Total time:* 11.84s
- *Input validation time:* 0.684
- *Prompt preparation time:* 0.00133
- *First-token latency:* 1.98
- *Cleanup time:* 0.293
- *Prompt tokens:* 2,611
- *Generation tokens:* 143
- *Total tokens:* 2,754
- *Prompt throughput (raw):* 1,317 tok/s
- *Generation throughput (raw):* 22.9 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.00269
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1289
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Idefics3-8B-Llama3-bf16/snapshots/8c2a30c48864f3251701b7bde40f601d25535098
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Akeman Street Baptist Church in Tring, Hertfordshire, England.
Description: The Akeman Street Baptist Church, a red-brick building with a grey slate roof and multiple arched windows, is situated in Tring, Hertfordshire, England. The church is surrounded by trees and has a brick wall with steps leading up to the entrance. A sign with the church's name is visible near the entrance.
Keywords: akeman street baptist church, tring, hertfordshire, england, red brick, grey slate roof, arched windows, trees, brick wall, steps, entrance, sign, church, building, europe, uk, gbr.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Title: Akeman Street Baptist Church in Tring, Hertfordshire, England.
Description: The Akeman Street Baptist Church, a red-brick building with a grey slate roof and multiple arched windows, is situated in Tring, Hertfordshire, England. The church is surrounded by trees and has a brick wall with steps leading up to the entrance. A sign with the church's name is visible near the entrance.
Keywords: akeman street baptist church, tring, hertfordshire, england, red brick, grey slate roof, arched windows, trees, brick wall, steps, entrance, sign, church, building, europe, uk, gbr.<end_of_utterance>
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
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.55s
- *Generation time:* 7.35s
- *Total time:* 9.54s
- *Input validation time:* 0.635
- *Prompt preparation time:* 0.000215
- *First-token latency:* 3.3
- *Cleanup time:* 0.157
- *Prompt tokens:* 2,922
- *Generation tokens:* 151
- *Total tokens:* 3,073
- *Prompt throughput (raw):* 886 tok/s
- *Generation throughput (raw):* 44.9 tok/s
- *Peak memory:* 13
- *Active memory:* 8.4
- *Cache memory:* 0.67
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00534
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3690
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 28777b889d841a86369c736175cb77258c8134b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-nvfp4/snapshots/28777b889d841a86369c736175cb77258c8134b2
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
**Title:**
Akeman Street Baptist Church, Tring, Hertfordshire

**Description:**
A historic red-brick Baptist church on Akeman Street, Tring, captured on **August 8, 2026 (UTC+01:00)**, showcasing its arched windows, brickwork entrance, and well-maintained garden. The scene features clear skies, adjacent greenery, and a prominent sign marking its location.

**Keywords:**
Akeman Street Baptist Church, Tring, Hertfordshire, red brick, brickwork, arched windows, historic church, Baptist church, UK, England, European architecture, church entrance, garden, stone sign, clear skies, stone pillars, brick boundary, town centre, Adobe Stock reference
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
- *Observations:* Duplicate keywords: england
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.32s
- *Generation time:* 4.40s
- *Total time:* 6.23s
- *Input validation time:* 0.498
- *Prompt preparation time:* 0.00134
- *First-token latency:* 0.854
- *Cleanup time:* 0.171
- *Prompt tokens:* 1,511
- *Generation tokens:* 106
- *Total tokens:* 1,617
- *Prompt throughput (raw):* 1,770 tok/s
- *Generation throughput (raw):* 46.2 tok/s
- *Peak memory:* 11
- *Active memory:* 9.0
- *Cache memory:* 0.14
- *Model-load active memory:* 9.04
- *Post-cleanup active memory:* 0.00592
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1248
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Molmo-7B-D-0924-8bit/snapshots/90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Akeman Street Baptist Church, Tring, Hertfordshire, England

Description: A red brick Baptist church stands prominently in Tring's town center, surrounded by bushes and a stone wall, with a red sign marking its entrance. The scene is bathed in sunlight, creating a warm and inviting atmosphere.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, Red brick, Church, Town center, Sunlight, Bushes, Stone wall, Sign, England, Europe
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
- *Observations:* Keyword list has 23 terms (requested 10-18); Duplicate
  keywords: england, sky, entrance
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.79s
- *Generation time:* 26.12s
- *Total time:* 27.36s
- *Input validation time:* 0.441
- *Prompt preparation time:* 0.0002
- *First-token latency:* 23.6
- *Cleanup time:* 0.124
- *Prompt tokens:* 16,545
- *Generation tokens:* 129
- *Total tokens:* 16,674
- *Prompt throughput (raw):* 700 tok/s
- *Generation throughput (raw):* 86.4 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00773
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1324
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Instruct-bf16/snapshots/c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Akeman Street Baptist Church, Tring, Herts

Description: The Akeman Street Baptist Church in Tring, Hertfordshire, England, is a red brick building with a distinctive round roof and arched windows, surrounded by lush greenery and a stone wall, under a partly cloudy sky.

Keywords: Akeman Street Baptist Church, Tring, Herts, England, Church, Red brick, Roof, Arched windows, Greenery, Stone wall, Sky, Clouds, Entrance, Chimney, Bushes, Christian, Locations, Objects, Hertfordshire, Europe, England, Entrance, Sky
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.84s
- *Generation time:* 2.82s
- *Total time:* 4.55s
- *Input validation time:* 0.884
- *Prompt preparation time:* 0.00425
- *First-token latency:* 0.411
- *Cleanup time:* 0.119
- *Prompt tokens:* 424
- *Generation tokens:* 113
- *Total tokens:* 537
- *Prompt throughput (raw):* 1,032 tok/s
- *Generation throughput (raw):* 108 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.19
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.0097
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1284
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--SmolVLM2-2.2B-Instruct-mlx/snapshots/844516024a1c4400d34489b89ee067d794e432ed
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <end_of_utterance>

*Readable output:*

<pre class="model-output-readable">
 Title: Akeman Street Baptist Church, Tring, Herts
Description: A red brick church with a large chimney and a sign stands in the town centre of Tring, Herts. The church is surrounded by trees and has a blue sky with clouds in the background.
Keywords: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky
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
- *Observations:* Keyword list has 26 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.14s
- *Generation time:* 26.58s
- *Total time:* 28.15s
- *Input validation time:* 0.428
- *Prompt preparation time:* 0.000238
- *First-token latency:* 21.6
- *Cleanup time:* 0.135
- *Prompt tokens:* 16,556
- *Generation tokens:* 195
- *Total tokens:* 16,751
- *Prompt throughput (raw):* 767 tok/s
- *Generation throughput (raw):* 47.5 tok/s
- *Peak memory:* 13
- *Active memory:* 9.5
- *Cache memory:* 1.4
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1382
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--X-Reasoner-7B-8bit/snapshots/21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Akeman Street Baptist Church, Tring

Description: A historic red-brick Baptist church in Tring, Hertfordshire, England, stands under a partly cloudy sky, with lush greenery and a stone entrance, signifying its role as a community hub.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Europe, Red brick, Chapel, Entrance, Sign, Clouds, Sky, Bushes, Chimney, Church, Christian, Historical, Architecture, 1808, 1852, Sunday services, 10:30am &amp; 6:00pm, www.akemanstreet.org, 2026-08-08 15:43:55 UTC+01:00, 51.815915°N, 0.638706°W
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Akeman Street Baptist Church, Tring

Description: A historic red-brick Baptist church in Tring, Hertfordshire, England, stands under a partly cloudy sky, with lush greenery and a stone entrance, signifying its role as a community hub.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Europe, Red brick, Chapel, Entrance, Sign, Clouds, Sky, Bushes, Chimney, Church, Christian, Historical, Architecture, 1808, 1852, Sunday services, 10:30am & 6:00pm, www.akemanstreet.org, 2026-08-08 15:43:55 UTC+01:00, 51.815915°N, 0.638706°W
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
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 2.37s
- *Generation time:* 7.50s
- *Total time:* 10.32s
- *Input validation time:* 0.438
- *Prompt preparation time:* 0.00339
- *First-token latency:* 1.12
- *Cleanup time:* 0.129
- *Prompt tokens:* 590
- *Generation tokens:* 148
- *Total tokens:* 738
- *Prompt throughput (raw):* 526 tok/s
- *Generation throughput (raw):* 25.7 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.44
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0108
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1307
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3.processing_gemma3.Gemma3Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3-27b-it-qat-4bit/snapshots/fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Akeman Street Baptist Church, Tring, Hertfordshire

Description: Captured on 8th August 2026 at 15:43, this photograph shows the red brick Akeman Street Baptist Church in Tring, Hertfordshire, bathed in afternoon sunlight; a stone wall and steps lead to the building’s central entrance, with a sign visible to the left. The church features a symmetrical facade, arched windows, and a prominent chimney.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, Church, Chapel, Christian, Red brick, Architecture, Facade, Chimney, Stone wall, Steps, Sign, Sunlight, Afternoon, Town centre, 2026, Exterior
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
- *Observations:* Keyword list has 23 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.76s
- *Generation time:* 5.76s
- *Total time:* 7.98s
- *Input validation time:* 0.447
- *Prompt preparation time:* 0.0043
- *First-token latency:* 2.03
- *Cleanup time:* 0.098
- *Prompt tokens:* 3,112
- *Generation tokens:* 118
- *Total tokens:* 3,230
- *Prompt throughput (raw):* 1,531 tok/s
- *Generation throughput (raw):* 37.7 tok/s
- *Peak memory:* 16
- *Active memory:* 13
- *Cache memory:* 0.59
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0144
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1252
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--pixtral-12b-8bit/snapshots/79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Akeman Street Baptist Church, Tring, England

Description:
The Akeman Street Baptist Church in Tring, England, is a red-brick building with arched windows and a central entrance, surrounded by lush greenery under a partly cloudy sky.

Keywords:
Akeman Street Baptist Church, Tring, England, red brick, arched windows, central entrance, greenery, partly cloudy sky, church, Christian, entrance, sign, roof, chimney, Buckinghamshire, Hertfordshire, Europe, objects, locations, bushes, chapel, sky, clouds
</pre>

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

<details>
<summary>Complete evidence: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Keyword list has 469 terms (requested 10-18); Duplicate
  keywords: church
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.35s
- *Generation time:* 2.52s
- *Total time:* 3.29s
- *Input validation time:* 0.404
- *Prompt preparation time:* 0.0065
- *First-token latency:* 0.0405
- *Cleanup time:* 0.0613
- *Prompt tokens:* 389
- *Generation tokens:* 1,000
- *Total tokens:* 1,389
- *Prompt throughput (raw):* 9,606 tok/s
- *Generation throughput (raw):* 497 tok/s
- *Peak memory:* 1.2
- *Active memory:* 0.90
- *Cache memory:* 0.04
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1303
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 6c33f49ebc0b50b75385f49ad3beddcb720d0c75
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-450M-MLX-bf16/snapshots/6c33f49ebc0b50b75385f49ad3beddcb720d0c75
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Akeman Street Baptist Church, Tring, England, UK, GBR, Europe

Description:
A brick-built church with a red door and arched windows stands in a serene town center setting, surrounded by lush greenery and a clear sky.

Keywords:
Akeman Street Baptist Church, Tring, England, UK, GBR, Europe, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church
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
  Response appears cut off at the token limit; Title has 77 words (requested
  5-10); Keyword list has 40 terms (requested 10-18); Duplicate keywords:
  tring, hertfordshire, red brick, white framed windows, central red door,
  cupola, partly cloudy sky, town centre, brick wall, stone pillars, gate
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.59s
- *Generation time:* 32.91s
- *Total time:* 34.94s
- *Input validation time:* 0.43
- *Prompt preparation time:* 0.00694
- *First-token latency:* 2.32
- *Cleanup time:* 0.134
- *Prompt tokens:* 3,203
- *Generation tokens:* 1,000
- *Total tokens:* 4,203
- *Prompt throughput (raw):* 1,378 tok/s
- *Generation throughput (raw):* 33.4 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.87
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.000706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1626
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Apriel-1.5-15b-Thinker-6bit-MLX/snapshots/24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Here are my reasoning steps:
We need to produce British-English catalogue metadata from the image and supplied context. The image shows a brick church building, presumably Akeman Street Baptist Church in Tring, Hertfordshire (though Tring is in Hertfordshire, not Buckinghamshire; but the hint says Buckinghamshire, but the context says Hertfordshire). The building is a red brick church with a central entrance, three windows on each floor (two on top floor, one on bottom floor?), actually there are three windows on each floor: top floor has three tall arched windows with white frames and multiple panes; bottom floor has three windows: left and right are rectangular with white frames, central is a red door. The building has a small cupola or steeple at the roof apex. The building is surrounded by a low brick wall with stone pillars and a gate. There's a sign on the left side: "Akeman Street Baptist Church" with some additional text. The sky is partly cloudy, with blue sky and some clouds. The lighting suggests mid-afternoon (sunlight from left side maybe). The capture date/time is 2026-08-08 15:43:55 UTC+01:00 (i.e., BST). GPS coordinates: 51.815915°N, 0.638706°W (Tring is around 51.82N, -0.63W). So we have location.

We need to produce:

- Title: concrete 5-10-word title. Should be British English. Something like "Akeman Street Baptist Church, Tring". That's 5 words? "Akeman Street Baptist Church, Tring" is 5 words? Actually "Akeman" (1), "Street" (2), "Baptist" (3), "Church," (4), "Tring" (5). That's 5 words. Could also include "Hertfordshire". But we need 5-10 words. So maybe "Akeman Street Baptist Church, Tring, Hertfordshire". That's 6 words? Let's count: "Akeman" (1), "Street" (2), "Baptist" (3), "Church," (4), "Tring," (5), "Hertfordshire" (6). That's 6 words. Good.

- Description: 1-2 sentences factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details. Should mention the building, its architecture, location, date/time maybe not needed but can mention capture date? Usually description includes location and details. Should mention red brick, white-framed windows, central red door, cupola, partly cloudy sky, mid-afternoon lighting, etc. Should be 1-2 sentences. Something like: "The red‑brick Akeman Street Baptist Church stands in Tring’s town centre, its symmetrical façade featuring three arched upper windows, a central red door, and a modest cupola under a partly cloudy sky. The building is framed by a low brick wall and stone pillars, with a sign indicating its name, captured on 8 August 2026 at 15:43 BST." That's two sentences. That includes date/time. Good.

- Keywords: 10-18 unique, comma-separated keywords covering relevant context and visible details. Should be British English. Should include: Akeman Street Baptist Church, Tring, Hertfordshire, red brick, white-framed windows, central red door, cupola, partly cloudy sky, British, church, Baptist, town centre, sign, brick wall, stone pillars, gate, Adobe Stock (maybe not needed but hint includes), Buckinghamshire (maybe not needed because location is Hertfordshire). But we need 10-18 unique keywords. Let's list maybe 14. Something like: "Akeman Street Baptist Church, Tring, Hertfordshire, red brick, white-framed windows, central red door, cupola, partly cloudy sky, brick wall, stone pillars, gate, town centre, Christian, Adobe Stock". That's 14. Ensure no duplicates. Good.

Now we need to output exactly three sections: Title: line, Description: line, Keywords: line. No extra text. Ensure proper formatting: Title: (the title). Description: (the description). Keywords: (the keywords). Use British English spelling (e.g., "centre" not "center").

Check the description: "The red‑brick Akeman Street Baptist Church stands in Tring’s town centre, its symmetrical façade featuring three arched upper windows, a central red door, and a modest cupola under a
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
  the Title field; Response appears cut off at the token limit; Expected model
  wrapper tokens remain visible; Keyword list has 223 terms (requested 10-18);
  Duplicate keywords: tring, hertfordshire, england, uk, europe, red brick,
  steeple, stone wall, gate, entrance, sign, sky, clouds, bushes, herts,
  baptist church, church
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 10.51s
- *Generation time:* 35.93s
- *Total time:* 47.18s
- *Input validation time:* 0.723
- *Prompt preparation time:* 0.0114
- *First-token latency:* 6.79
- *Cleanup time:* 0.316
- *Prompt tokens:* 1,629
- *Generation tokens:* 1,000
- *Total tokens:* 2,629
- *Prompt throughput (raw):* 240 tok/s
- *Generation throughput (raw):* 35.2 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.17
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00118
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1442
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--ERNIE-4.5-VL-28B-A3B-Thinking-bf16/snapshots/32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Okay, let's break this down. I've got an image of a church and need to create some metadata for it. First, I need a title. The hints are a bit messy, but I see "Akeman Street Baptist Church" and "Tring, Herts" in there. That's a good start. I'll go with "Akeman Street Baptist Church, Tring, Herts."

Next, the description. I need to be factual and descriptive. The image shows a red brick church with a tower and a steeple. The sky is partly cloudy, and there's a stone wall and gate in the foreground. I'll combine the location from the hints with the visual details. I'll say: "A red brick Akeman Street Baptist Church in Tring, Hertfordshire, England, with a steeple, stone wall, and gate under a partly cloudy sky."

Finally, keywords. I need to be comprehensive. I'll pull from the hints and the image itself. I'll include the name of the church, the location, the architectural style, the materials, the sky, and some general terms. I'll make sure to include "Hertfordshire" and "England" and "UK" for the location. I'll also add "Tring" and "Baptist Church" for specificity. I'll also add "Red brick," "
&lt;/think&gt;

Title: Akeman Street Baptist Church, Tring, Herts

Description: A red brick Akeman Street Baptist Church in Tring, Hertfordshire, England, with a steeple, stone wall, and gate under a partly cloudy sky.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, GBR, Europe, Red brick, Roof, Steeple, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign
</pre>

<details>
<summary>Exact raw output</summary>

```text
Okay, let's break this down. I've got an image of a church and need to create some metadata for it. First, I need a title. The hints are a bit messy, but I see "Akeman Street Baptist Church" and "Tring, Herts" in there. That's a good start. I'll go with "Akeman Street Baptist Church, Tring, Herts."

Next, the description. I need to be factual and descriptive. The image shows a red brick church with a tower and a steeple. The sky is partly cloudy, and there's a stone wall and gate in the foreground. I'll combine the location from the hints with the visual details. I'll say: "A red brick Akeman Street Baptist Church in Tring, Hertfordshire, England, with a steeple, stone wall, and gate under a partly cloudy sky."

Finally, keywords. I need to be comprehensive. I'll pull from the hints and the image itself. I'll include the name of the church, the location, the architectural style, the materials, the sky, and some general terms. I'll make sure to include "Hertfordshire" and "England" and "UK" for the location. I'll also add "Tring" and "Baptist Church" for specificity. I'll also add "Red brick," "
</think>

Title: Akeman Street Baptist Church, Tring, Herts

Description: A red brick Akeman Street Baptist Church in Tring, Hertfordshire, England, with a steeple, stone wall, and gate under a partly cloudy sky.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, GBR, Europe, Red brick, Roof, Steeple, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign
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
- *Model load time:* 0.72s
- *Generation time:* 2.21s
- *Total time:* 3.60s
- *Input validation time:* 0.672
- *Prompt preparation time:* 0.0023
- *First-token latency:* 0.107
- *Cleanup time:* 0.11
- *Prompt tokens:* 332
- *Generation tokens:* 173
- *Total tokens:* 505
- *Prompt throughput (raw):* 3,117 tok/s
- *Generation throughput (raw):* 320 tok/s
- *Peak memory:* 1.9
- *Active memory:* 1.2
- *Cache memory:* 0.02
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00149
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1347
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--FastVLM-0.5B-bf16/snapshots/81ffe929046666c43de53691147b1669ba0f3a4c
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
A concrete 5-10-word title: Akeman Street Baptist Church, Tring, Herts, UK, GBR, Europe
A 1-2-sentence factual description: The image depicts the Akeman Street Baptist Church in Tring, Herts, UK, captured on 2026-08-08 15:43:55 UTC+01:00. The church is a 1-2-story red brick building with a slate roof, featuring arched windows and a central wooden door. The church is surrounded by a low brick wall with stone pillars and a metal gate, and there is a sign indicating the name of the church. The setting is a sunny day with a clear blue sky, and the church is situated in a residential area with other houses visible in the background.
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
  Keyword list has 39 terms (requested 10-18); Duplicate keywords:
  hertfordshire, red brick, chapel, entrance, clouds, church, england, sign,
  bushes
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.85s
- *Generation time:* 38.38s
- *Total time:* 40.77s
- *Input validation time:* 0.533
- *Prompt preparation time:* 0.0018
- *First-token latency:* 10.6
- *Cleanup time:* 0.234
- *Prompt tokens:* 6,446
- *Generation tokens:* 1,000
- *Total tokens:* 7,446
- *Prompt throughput (raw):* 610 tok/s
- *Generation throughput (raw):* 36.8 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.44
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1309
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.1V-9B-Thinking-8bit/snapshots/9677807f106500eb7690391c27645d59f6855cfb
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. The main subject is the Akeman Street Baptist Church, so maybe "Akeman Street Baptist Church, Tring". Then the description should combine context (date, location) with the main subject, setting, etc. Let's check the image: red brick building, entrance with steps, sign, greenery. So description should mention the church, its location, the setting, maybe the time of day (since it's sunny). Then keywords need to cover the hints and visible details. Let's make sure each part is correct.

Title: "Akeman Street Baptist Church, Tring"
Description: "Akeman Street Baptist Church in Tring, Hertfordshire, captured on 2026-08-08, shows a red brick chapel with a central entrance, steps, and surrounding greenery under a partly cloudy sky."
Keywords: "Akeman Street Baptist Church, Tring, Hertfordshire, red brick, chapel, entrance, steps, greenery, clouds, sky, church, England, Hertfordshire, location, brick, building, sign, bushes"

Wait, need to make sure keywords are unique and cover the hints. Let's check the hints: Adobe Stock, Any Vision, Buckinghamshire (wait, Hertfordshire is the county, not Buckinghamshire), Bushes, Chapel, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky. So combine those. Let's list them:

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian

Wait, need 10-18. Let's count. Let's make sure each is unique. Let's do:

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian

Wait, maybe better to check the image details. The building is red brick, has a sign, entrance with steps, bushes, clouds in sky, etc. So keywords should include those. Let's finalize:

Title: "Akeman Street Baptist Church, Tring"
Description: "Akeman Street Baptist Church in Tring, Hertfordshire, photographed on 2026-08-08, features a red brick chapel with a central entrance, steps, and surrounding greenery under a partly cloudy sky."
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian

Wait, need to make sure the description is 1-2 sentences. Let's adjust:

Description: "Akeman Street Baptist Church in Tring, Hertfordshire, captured on 2026-08-08, is a red brick chapel with a central entrance, steps, and greenery, set under a partly cloudy sky."

Yes, that's 1-2 sentences. Now check the title is 5-10 words: "Akeman Street Baptist Church, Tring" is 6 words, good. Keywords need to be unique, comma-separated, 10-18. Let's list again:

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian. Wait, that's 16, which is within 10-18. Let's check if all hints are covered. The hints included Adobe Stock, Any Vision, Buckinghamshire (but we have Hertfordshire), Bushes, Chapel, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky. So we have most of them. Maybe add "red brick" as a keyword. Let's confirm each part.

Title: "Akeman Street Baptist Church, Tring" (5 words, good)
Description: "Akeman Street Baptist Church in Tring, Hertfordshire, photographed on 2026-08-08, is a red brick chapel with a central entrance, steps, and greenery, set under a partly cloudy sky." (1 sentence, covers context, subject, setting, lighting, details)
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertford
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words. The main subject is the Akeman Street Baptist Church, so maybe "Akeman Street Baptist Church, Tring". Then the description should combine context (date, location) with the main subject, setting, etc. Let's check the image: red brick building, entrance with steps, sign, greenery. So description should mention the church, its location, the setting, maybe the time of day (since it's sunny). Then keywords need to cover the hints and visible details. Let's make sure each part is correct.

Title: "Akeman Street Baptist Church, Tring"
Description: "Akeman Street Baptist Church in Tring, Hertfordshire, captured on 2026-08-08, shows a red brick chapel with a central entrance, steps, and surrounding greenery under a partly cloudy sky."
Keywords: "Akeman Street Baptist Church, Tring, Hertfordshire, red brick, chapel, entrance, steps, greenery, clouds, sky, church, England, Hertfordshire, location, brick, building, sign, bushes"

Wait, need to make sure keywords are unique and cover the hints. Let's check the hints: Adobe Stock, Any Vision, Buckinghamshire (wait, Hertfordshire is the county, not Buckinghamshire), Bushes, Chapel, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky. So combine those. Let's list them:

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian

Wait, need 10-18. Let's count. Let's make sure each is unique. Let's do:

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian

Wait, maybe better to check the image details. The building is red brick, has a sign, entrance with steps, bushes, clouds in sky, etc. So keywords should include those. Let's finalize:

Title: "Akeman Street Baptist Church, Tring"
Description: "Akeman Street Baptist Church in Tring, Hertfordshire, photographed on 2026-08-08, features a red brick chapel with a central entrance, steps, and surrounding greenery under a partly cloudy sky."
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian

Wait, need to make sure the description is 1-2 sentences. Let's adjust:

Description: "Akeman Street Baptist Church in Tring, Hertfordshire, captured on 2026-08-08, is a red brick chapel with a central entrance, steps, and greenery, set under a partly cloudy sky."

Yes, that's 1-2 sentences. Now check the title is 5-10 words: "Akeman Street Baptist Church, Tring" is 6 words, good. Keywords need to be unique, comma-separated, 10-18. Let's list again:

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian. Wait, that's 16, which is within 10-18. Let's check if all hints are covered. The hints included Adobe Stock, Any Vision, Buckinghamshire (but we have Hertfordshire), Bushes, Chapel, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky. So we have most of them. Maybe add "red brick" as a keyword. Let's confirm each part.

Title: "Akeman Street Baptist Church, Tring" (5 words, good)
Description: "Akeman Street Baptist Church in Tring, Hertfordshire, photographed on 2026-08-08, is a red brick chapel with a central entrance, steps, and greenery, set under a partly cloudy sky." (1 sentence, covers context, subject, setting, lighting, details)
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertford
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
- *Maintainer status:* none
- *Observations:* Extra text appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.67s
- *Generation time:* 16.51s
- *Total time:* 19.11s
- *Input validation time:* 0.912
- *Prompt preparation time:* 0.00752
- *First-token latency:* 8.77
- *Cleanup time:* 0.136
- *Prompt tokens:* 6,446
- *Generation tokens:* 420
- *Total tokens:* 6,866
- *Prompt throughput (raw):* 735 tok/s
- *Generation throughput (raw):* 58.9 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.47
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1309
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-Flash-mxfp4/snapshots/773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image, the main subject is Akeman Street Baptist Church. So a good title would be "Akeman Street Baptist Church, Tring".

Next, the description. It needs to be 1-2 sentences, combining context and visible details. Let's include the location (Tring, Hertfordshire), the subject (Akeman Street Baptist Church), visible details like red brick, arched windows, stone wall with pillars, and the sign. Also mention the setting (town centre, England).

So the description could be: "The Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a stone wall with pillars, and a sign indicating Sunday services."

Now keywords. Need 10-18 unique, comma-separated. Let's list them:

- Akeman Street Baptist Church
- Tring
- Hertfordshire
- Red brick
- Arched windows
- Stone wall
- Pillars
- Sunday services
- Town centre
- England
- UK
- Baptist
- Church

Let's pick the most relevant ones, ensuring 10-18. Let's check:

- Akeman Street Baptist Church
- Tring
- Hertfordshire
- Red brick
- Arched windows
- Stone wall
- Sunday services
- Town centre
- England

That's 11, which is within 10-18. Perfect.

Now let's put it all together.&lt;/think&gt;
Title: Akeman Street Baptist Church, Tring
Description: The Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a stone wall with pillars, and a sign indicating Sunday services.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, red brick, arched windows, stone wall, Sunday services, town centre, England, UK, Baptist, church
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image, the main subject is Akeman Street Baptist Church. So a good title would be "Akeman Street Baptist Church, Tring".

Next, the description. It needs to be 1-2 sentences, combining context and visible details. Let's include the location (Tring, Hertfordshire), the subject (Akeman Street Baptist Church), visible details like red brick, arched windows, stone wall with pillars, and the sign. Also mention the setting (town centre, England).

So the description could be: "The Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a stone wall with pillars, and a sign indicating Sunday services."

Now keywords. Need 10-18 unique, comma-separated. Let's list them:

- Akeman Street Baptist Church
- Tring
- Hertfordshire
- Red brick
- Arched windows
- Stone wall
- Pillars
- Sunday services
- Town centre
- England
- UK
- Baptist
- Church

Let's pick the most relevant ones, ensuring 10-18. Let's check:

- Akeman Street Baptist Church
- Tring
- Hertfordshire
- Red brick
- Arched windows
- Stone wall
- Sunday services
- Town centre
- England

That's 11, which is within 10-18. Perfect.

Now let's put it all together.</think>
Title: Akeman Street Baptist Church, Tring
Description: The Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a stone wall with pillars, and a sign indicating Sunday services.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, red brick, arched windows, stone wall, Sunday services, town centre, England, UK, Baptist, church
```

</details>

</details>

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### mlx-community/GLM-4.6V-nvfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-nvfp4</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Extra text appears before the Title field; Response appears
  cut off at the token limit; Internal reasoning block appears incomplete;
  Title has 15 words (requested 5-10); Keyword list has 66 terms (requested
  10-18); Duplicate keywords: hertfordshire, england, europe, red brick, sign,
  bushes, chapel, church, clouds, entrance, maroon door, arched windows,
  greenery
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 10.97s
- *Generation time:* 68.10s
- *Total time:* 79.79s
- *Input validation time:* 0.71
- *Prompt preparation time:* 0.008
- *First-token latency:* 25.9
- *Cleanup time:* 0.391
- *Prompt tokens:* 6,446
- *Generation tokens:* 1,000
- *Total tokens:* 7,446
- *Prompt throughput (raw):* 249 tok/s
- *Generation throughput (raw):* 24.1 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.6
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1309
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-nvfp4/snapshots/2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;Got it, let's tackle this step by step. First, the title needs to be 5-10 words, concrete. The main subject is Akeman Street Baptist Church, so maybe "Akeman Street Baptist Church, Tring" but wait, need to be 5-10 words. Let's see, "Akeman Street Baptist Church, Tring, England" is 7 words, that works.

Next, the description: need to combine context (Tring, Herts) with visible details. The church is red brick, has a steep roof, arched windows, a maroon door, steps, a sign, bushes around. Let's put that together: "Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a maroon door, and steps leading to its entrance, surrounded by greenery." Wait, does that cover the main points? Let's check: subject (church), setting (Tring, Herts), visible details (red brick, arched windows, maroon door, steps, greenery). Yes.

Keywords: Need 10-18 unique, comma-separated. Let's list the hints and add visible ones. Hints include Akeman Street Baptist Church, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof. Also, visible details: steps, maroon door, arched windows, greenery, brick wall, chimney (wait, do we see a chimney? The image has a chimney on the left house, but the church itself—wait, the church has a roof with a spire, but chimney? Maybe not. Let's stick to visible: Akeman Street Baptist Church, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery, Brick wall. Let's count: 1. Akeman Street Baptist Church, 2. Hertfordshire, 3. England, 4. Europe, 5. Red brick, 6. Sign, 7. Bushes, 8. Chapel, 9. Church, 10. Clouds, 11. Entrance, 12. Roof, 13. Steps, 14. Maroon door, 15. Arched windows, 16. Greenery, 17. Brick wall. That's 17, which is within range.

Wait, let's check the description again. The context says "Akeman Street Baptist Church, Tring, Herts" so the description should mention Tring and Herts. Let's adjust the description to include that: "Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a maroon door, and steps leading to its entrance, surrounded by greenery." Yes, that's good.

Title: Let's make sure it's 5-10 words. "Akeman Street Baptist Church, Tring" is 6 words, perfect.

Keywords: Let's make sure they are unique and cover all. Let's list again: Akeman Street Baptist Church, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery, Brick wall. That's 15, but maybe add "Tring" since it's in the context. Wait, the title has Tring, but keywords can include it. Let's replace one with Tring: Akeman Street Baptist Church, Tring, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery. That's 15, still good.

Wait, let's check the image: the sign says "Akeman Street Baptist Church" so that's a keyword. The church is red brick, has a roof with a spire, arched windows, maroon door, steps, a brick wall with pillars, bushes around. So keywords should include those. Let's finalize:

Title: Akeman Street Baptist Church, Tring

Description: Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a maroon door, and steps leading to its entrance, surrounded by greenery.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery, Brick wall

Wait, let's count keywords: 1
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this step by step. First, the title needs to be 5-10 words, concrete. The main subject is Akeman Street Baptist Church, so maybe "Akeman Street Baptist Church, Tring" but wait, need to be 5-10 words. Let's see, "Akeman Street Baptist Church, Tring, England" is 7 words, that works.

Next, the description: need to combine context (Tring, Herts) with visible details. The church is red brick, has a steep roof, arched windows, a maroon door, steps, a sign, bushes around. Let's put that together: "Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a maroon door, and steps leading to its entrance, surrounded by greenery." Wait, does that cover the main points? Let's check: subject (church), setting (Tring, Herts), visible details (red brick, arched windows, maroon door, steps, greenery). Yes.

Keywords: Need 10-18 unique, comma-separated. Let's list the hints and add visible ones. Hints include Akeman Street Baptist Church, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof. Also, visible details: steps, maroon door, arched windows, greenery, brick wall, chimney (wait, do we see a chimney? The image has a chimney on the left house, but the church itself—wait, the church has a roof with a spire, but chimney? Maybe not. Let's stick to visible: Akeman Street Baptist Church, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery, Brick wall. Let's count: 1. Akeman Street Baptist Church, 2. Hertfordshire, 3. England, 4. Europe, 5. Red brick, 6. Sign, 7. Bushes, 8. Chapel, 9. Church, 10. Clouds, 11. Entrance, 12. Roof, 13. Steps, 14. Maroon door, 15. Arched windows, 16. Greenery, 17. Brick wall. That's 17, which is within range.

Wait, let's check the description again. The context says "Akeman Street Baptist Church, Tring, Herts" so the description should mention Tring and Herts. Let's adjust the description to include that: "Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a maroon door, and steps leading to its entrance, surrounded by greenery." Yes, that's good.

Title: Let's make sure it's 5-10 words. "Akeman Street Baptist Church, Tring" is 6 words, perfect.

Keywords: Let's make sure they are unique and cover all. Let's list again: Akeman Street Baptist Church, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery, Brick wall. That's 15, but maybe add "Tring" since it's in the context. Wait, the title has Tring, but keywords can include it. Let's replace one with Tring: Akeman Street Baptist Church, Tring, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery. That's 15, still good.

Wait, let's check the image: the sign says "Akeman Street Baptist Church" so that's a keyword. The church is red brick, has a roof with a spire, arched windows, maroon door, steps, a brick wall with pillars, bushes around. So keywords should include those. Let's finalize:

Title: Akeman Street Baptist Church, Tring

Description: Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a maroon door, and steps leading to its entrance, surrounded by greenery.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery, Brick wall

Wait, let's count keywords: 1
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
- *Observations:* Missing or empty fields: Title; Extra text appears before
  the Title field; Conversation-role control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 3.83s
- *Generation time:* 123.28s
- *Total time:* 127.93s
- *Input validation time:* 0.806
- *Prompt preparation time:* 0.0019
- *First-token latency:* 2.05
- *Cleanup time:* 0.26
- *Prompt tokens:* 1,321
- *Generation tokens:* 463
- *Total tokens:* 1,784
- *Prompt throughput (raw):* 644 tok/s
- *Generation throughput (raw):* 3.85 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.5
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00365
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1439
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Kimi-VL-A3B-Thinking-2506-bf16/snapshots/fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* [EOS]

*Readable output:*

<pre class="model-output-readable">
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the provided context and image. The capture date and GPS are authoritative, but I shouldn't claim they're visible. The title hint is "Town centre, Tring, England, UK, GBR, Europe" and the description hint is "Akeman Street Baptist Church, Tring, Herts". The keywords include Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky.

Looking at the image, the main subject is the Akeman Street Baptist Church in Tring, a red-brick building with arched windows, a dark door, a sign, and surrounded by greenery. The sky is partly cloudy. The setting is a town centre in Tring, Hertfordshire, UK. The image is from Adobe Stock, so that's a keyword. The church is a chapel, has a chimney, and there's a sign. The lighting is bright, so "bright lighting" could be a detail. The keywords need to be unique and cover visible details. Let's structure the title first: "Akeman Street Baptist Church, Tring, Hertfordshire, UK". Then the description should mention the church, location, architectural details, surroundings, and lighting. Keywords should include all the hints and visible elements like red brick, chimney, sign, bushes, clouds, etc. Need to check for conflicts and ensure accuracy. Let's draft the title, description, and keywords.◁/think▷Title: Akeman Street Baptist Church, Tring, Hertfordshire, UK
Description: Akeman Street Baptist Church in Tring, Hertfordshire, UK, a red-brick chapel with arched windows, a dark wooden door, and a sign, set against a partly cloudy sky with lush bushes and adjacent buildings in a town centre.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, UK, Red brick, Chapel, Christian, Clouds, Bushes, Chimney, Church, Entrance, England, Europe, Locations, Objects, Sky, Bright lighting, Adobe Stock&lt;|im_assistant|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the provided context and image. The capture date and GPS are authoritative, but I shouldn't claim they're visible. The title hint is "Town centre, Tring, England, UK, GBR, Europe" and the description hint is "Akeman Street Baptist Church, Tring, Herts". The keywords include Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky.

Looking at the image, the main subject is the Akeman Street Baptist Church in Tring, a red-brick building with arched windows, a dark door, a sign, and surrounded by greenery. The sky is partly cloudy. The setting is a town centre in Tring, Hertfordshire, UK. The image is from Adobe Stock, so that's a keyword. The church is a chapel, has a chimney, and there's a sign. The lighting is bright, so "bright lighting" could be a detail. The keywords need to be unique and cover visible details. Let's structure the title first: "Akeman Street Baptist Church, Tring, Hertfordshire, UK". Then the description should mention the church, location, architectural details, surroundings, and lighting. Keywords should include all the hints and visible elements like red brick, chimney, sign, bushes, clouds, etc. Need to check for conflicts and ensure accuracy. Let's draft the title, description, and keywords.◁/think▷Title: Akeman Street Baptist Church, Tring, Hertfordshire, UK  
Description: Akeman Street Baptist Church in Tring, Hertfordshire, UK, a red-brick chapel with arched windows, a dark wooden door, and a sign, set against a partly cloudy sky with lush bushes and adjacent buildings in a town centre.  
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, UK, Red brick, Chapel, Christian, Clouds, Bushes, Chimney, Church, Entrance, England, Europe, Locations, Objects, Sky, Bright lighting, Adobe Stock<|im_assistant|>
```

</details>

</details>

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### mlx-community/MiniCPM-V-4.6-8bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.6-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the task instructions instead of only
  returning the requested fields; Extra text appears before the Title field;
  Expected model wrapper tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Model load time:* 1.01s
- *Generation time:* 3.17s
- *Total time:* 4.74s
- *Input validation time:* 0.546
- *Prompt preparation time:* 0.00808
- *First-token latency:* 0.274
- *Cleanup time:* 0.12
- *Prompt tokens:* 930
- *Generation tokens:* 412
- *Total tokens:* 1,342
- *Prompt throughput (raw):* 3,399 tok/s
- *Generation throughput (raw):* 203 tok/s
- *Peak memory:* 3.7
- *Active memory:* 2.3
- *Cache memory:* 0.05
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00482
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1309
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.6-8bit/snapshots/03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
We need to create the catalogue metadata. Let's break down the required sections:

1. Title: needs to be a concrete 5-10 word title.
2. Description: needs to be a 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, and distinctive details.
3. Keywords: 10-18 unique, covering relevant context and visible details.

First, extract the key elements from the image:

- The building is Akeman Street Baptist Church, Tring, England, UK, Europe.
- The building is a brick church with arched windows and a red door.
- There are trees and bushes, a sign for Akeman Street Baptist Church.
- The sky is partly cloudy.
- The setting is a town center, with a brick wall and gate.

Now, craft the title: need 5-10 words. Something like "Akeman Street Baptist Church, Tring, England". But need to make it concrete and descriptive.

Description: Combine the context and visible details. The church is in a town, with brick architecture, surrounded by greenery, under partly cloudy skies.

Keywords: need 10-18 unique, covering relevant context and visible details. Let's list: Akeman Street, Baptist Church, Tring, England, UK, Europe, brick, red door, arched windows, trees, bushes, sign, sky
&lt;/think&gt;

Title:
Akeman Street Baptist Church, Tring, England

Description:
Akeman Street Baptist Church is a brick church in Tring, featuring arched windows and a red door, set amidst greenery under partly cloudy skies.

Keywords: Akeman Street, Baptist Church, Tring, England, UK, Europe, brick, red door, arched windows, trees, bushes, sign, sky, Christian, church, entrance, Herts, Buckinghamshire, England, Europe, Tring, architecture, greenery
</pre>

<details>
<summary>Exact raw output</summary>

```text
We need to create the catalogue metadata. Let's break down the required sections:

1. Title: needs to be a concrete 5-10 word title.
2. Description: needs to be a 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, and distinctive details.
3. Keywords: 10-18 unique, covering relevant context and visible details.

First, extract the key elements from the image:

- The building is Akeman Street Baptist Church, Tring, England, UK, Europe.
- The building is a brick church with arched windows and a red door.
- There are trees and bushes, a sign for Akeman Street Baptist Church.
- The sky is partly cloudy.
- The setting is a town center, with a brick wall and gate.

Now, craft the title: need 5-10 words. Something like "Akeman Street Baptist Church, Tring, England". But need to make it concrete and descriptive.

Description: Combine the context and visible details. The church is in a town, with brick architecture, surrounded by greenery, under partly cloudy skies.

Keywords: need 10-18 unique, covering relevant context and visible details. Let's list: Akeman Street, Baptist Church, Tring, England, UK, Europe, brick, red door, arched windows, trees, bushes, sign, sky
</think>

Title:
Akeman Street Baptist Church, Tring, England

Description:
Akeman Street Baptist Church is a brick church in Tring, featuring arched windows and a red door, set amidst greenery under partly cloudy skies.

Keywords: Akeman Street, Baptist Church, Tring, England, UK, Europe, brick, red door, arched windows, trees, bushes, sign, sky, Christian, church, entrance, Herts, Buckinghamshire, England, Europe, Tring, architecture, greenery
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
- *Model load time:* 2.41s
- *Generation time:* 32.16s
- *Total time:* 35.35s
- *Input validation time:* 0.768
- *Prompt preparation time:* 0.00491
- *First-token latency:* 2.94
- *Cleanup time:* 0.137
- *Prompt tokens:* 3,124
- *Generation tokens:* 166
- *Total tokens:* 3,290
- *Prompt throughput (raw):* 1,061 tok/s
- *Generation throughput (raw):* 5.89 tok/s
- *Peak memory:* 24
- *Active memory:* 19
- *Cache memory:* 5.0
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.00654
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1290
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo_point.processing_molmo_point.MolmoPointProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MolmoPoint-8B-fp16/snapshots/0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Akeman Street Baptist Church, Tring, Hertfordshire, England

The image shows the Akeman Street Baptist Church in Tring, Hertfordshire, England. Built of red brick with a distinctive arched doorway and three arched windows, the church stands behind a brick wall and gate. The scene is set on a cloudy day with a blue sky and white clouds, surrounded by trees and bushes. The church's sign indicates it is open for worship on Sundays.

Akeman Street, Tring, Hertfordshire, England, Baptist Church, Red Brick, Arched Windows, Arched Doorway, Brick Wall, Gate, Trees, Bushes, Cloudy Sky, Blue Sky, White Clouds, Sunday Worship, Church Sign, Church Architecture, Church Exterior, Church Setting, Church Surroundings
</pre>

</details>

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### mlx-community/Ornith-1.0-35B-bf16

<details>
<summary>Complete evidence: mlx-community/Ornith-1.0-35B-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Extra text appears before the Title field; Expected model
  wrapper tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 12.09s
- *Generation time:* 78.75s
- *Total time:* 91.28s
- *Input validation time:* 0.421
- *Prompt preparation time:* 0.0137
- *First-token latency:* 70.3
- *Cleanup time:* 0.324
- *Prompt tokens:* 16,557
- *Generation tokens:* 435
- *Total tokens:* 16,992
- *Prompt throughput (raw):* 236 tok/s
- *Generation throughput (raw):* 59.0 tok/s
- *Peak memory:* 74
- *Active memory:* 70
- *Cache memory:* 0.89
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00705
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1332
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-bf16/snapshots/9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
The user wants metadata for an image of a church.

**1. Analyze the Image:**
*   **Subject:** A red brick building, clearly a church.
*   **Signage:** A sign on the left reads "Akeman Street Baptist Church" with service times "Sunday Services 10:30am &amp; 6:00pm" and a website "www.akemanstreet.org".
*   **Architecture:** Red brick facade, arched windows (three on the upper level, two on the lower level flanking the door), a central arched doorway with double red doors. There's a stone plaque above the door. The roof is a hipped roof with a small cupola/spire on top.
*   **Surroundings:**
    *   Left: Another brick building (residential?), trees, bushes.
    *   Right: Large green trees and bushes obscuring part of the building.
    *   Foreground: A low brick wall with stone pillars and black metal gates. Steps leading up to the church entrance. A paved path.
    *   Sky: Blue sky with scattered white clouds.
*   **Lighting:** Bright sunlight, casting shadows. The sun seems to be coming from the upper left/behind the camera, illuminating the front facade.
*   **Context clues:** The sign says "Akeman Street Baptist Church". The hint says "Tring, Herts".
&lt;/think&gt;

Title: Akeman Street Baptist Church in Tring, Hertfordshire
Description: A red brick Baptist church stands on Akeman Street in Tring, Hertfordshire, featuring arched windows, a central double door, and a stone plaque above the entrance. A low brick wall with stone pillars and black gates encloses the front garden, where a sign displays service times under a blue sky with scattered clouds.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Red brick, Church, Chapel, Sign, Entrance, Roof, Sky, Clouds, Gates, Steps, Garden, Architecture, Christian
</pre>

<details>
<summary>Exact raw output</summary>

```text
The user wants metadata for an image of a church.

**1. Analyze the Image:**
*   **Subject:** A red brick building, clearly a church.
*   **Signage:** A sign on the left reads "Akeman Street Baptist Church" with service times "Sunday Services 10:30am & 6:00pm" and a website "www.akemanstreet.org".
*   **Architecture:** Red brick facade, arched windows (three on the upper level, two on the lower level flanking the door), a central arched doorway with double red doors. There's a stone plaque above the door. The roof is a hipped roof with a small cupola/spire on top.
*   **Surroundings:**
    *   Left: Another brick building (residential?), trees, bushes.
    *   Right: Large green trees and bushes obscuring part of the building.
    *   Foreground: A low brick wall with stone pillars and black metal gates. Steps leading up to the church entrance. A paved path.
    *   Sky: Blue sky with scattered white clouds.
*   **Lighting:** Bright sunlight, casting shadows. The sun seems to be coming from the upper left/behind the camera, illuminating the front facade.
*   **Context clues:** The sign says "Akeman Street Baptist Church". The hint says "Tring, Herts".
</think>

Title: Akeman Street Baptist Church in Tring, Hertfordshire
Description: A red brick Baptist church stands on Akeman Street in Tring, Hertfordshire, featuring arched windows, a central double door, and a stone plaque above the entrance. A low brick wall with stone pillars and black gates encloses the front garden, where a sign displays service times under a blue sky with scattered clouds.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Red brick, Church, Chapel, Sign, Entrance, Roof, Sky, Clouds, Gates, Steps, Garden, Architecture, Christian
```

</details>

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
- *Model load time:* 0.53s
- *Generation time:* 87.28s
- *Total time:* 88.24s
- *Input validation time:* 0.419
- *Prompt preparation time:* 0.00324
- *First-token latency:* 85.5
- *Cleanup time:* 0.0948
- *Prompt tokens:* 16,556
- *Generation tokens:* 148
- *Total tokens:* 16,704
- *Prompt throughput (raw):* 194 tok/s
- *Generation throughput (raw):* 172 tok/s
- *Peak memory:* 5.1
- *Active memory:* 1.3
- *Cache memory:* 0.86
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.00742
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1382
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 01af461cdb9574acc09084a0ef94e216e142b085
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen2-VL-2B-Instruct-4bit/snapshots/01af461cdb9574acc09084a0ef94e216e142b085
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Akeman Street Baptist Church, Tring, Herts, England, UK, GBR, Europe

Description:
Akeman Street Baptist Church is a historic building located in Tring, Herts, England. The church was built in 1808 and underwent a significant renovation in 1832. The building features a red brick exterior and two arched windows on the front. The entrance is marked by a red door, and the church is surrounded by lush greenery, including bushes and trees. The sky above is partly cloudy, and the church is surrounded by a stone wall and gate. The church is open for Sunday services, and there is a sign indicating the time and date of the services.
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
- *Observations:* Extra text appears before the Title field; Expected model
  wrapper tokens remain visible; Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.78s
- *Generation time:* 32.02s
- *Total time:* 33.48s
- *Input validation time:* 0.658
- *Prompt preparation time:* 0.0113
- *First-token latency:* 23.5
- *Cleanup time:* 0.231
- *Prompt tokens:* 16,547
- *Generation tokens:* 428
- *Total tokens:* 16,975
- *Prompt throughput (raw):* 705 tok/s
- *Generation throughput (raw):* 56.5 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00805
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1332
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Thinking-bf16/snapshots/c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Got it, let's tackle this step by step. First, the title. The context says it's Akeman Street Baptist Church in Tring, Herts. The descriptive hints mention that. So a title should be concise. Maybe "Akeman Street Baptist Church, Tring, Herts, England" but need to be 5-10 words. Let's check: "Akeman Street Baptist Church, Tring, Herts, England" is 10 words. Wait, maybe shorter: "Akeman Street Baptist Church, Tring, Herts, England" is 10 words. Alternatively, "Tring Baptist Church, Akeman Street, Herts, England" but the context says "Akeman Street Baptist Church". So title: "Akeman Street Baptist Church, Tring, Herts, England" – that's 10 words.

Next, the description. Need to combine context with visible subject, setting, action, lighting, details. The image shows the church, red brick, arched windows, door, sign, trees, sky. Capture date is 2026-08-08, but we don't claim it's visible. The sign says "Akeman Street Baptist Church", so the main subject is the church. Setting is the town centre, Tring, Herts. Lighting: sunny, clouds in sky. Distinctive details: red brick, arched windows, dark red door, stone pillars,
&lt;/think&gt;

Title: Akeman Street Baptist Church, Tring, Herts, England

Description: Red-brick Akeman Street Baptist Church in Tring, Herts, England, featuring arched windows, a dark red door, stone pillars, and a sign, under a partly cloudy sky with sunlight illuminating the scene.

Keywords: Akeman Street Baptist Church, Tring, Herts, England, Red brick, Arched windows, Dark red door, Stone pillars, Sign, Clouds, Sunlight, Town centre, Church, Hertfordshire, Bushes, Chimney, Christian, Entrance, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this step by step. First, the title. The context says it's Akeman Street Baptist Church in Tring, Herts. The descriptive hints mention that. So a title should be concise. Maybe "Akeman Street Baptist Church, Tring, Herts, England" but need to be 5-10 words. Let's check: "Akeman Street Baptist Church, Tring, Herts, England" is 10 words. Wait, maybe shorter: "Akeman Street Baptist Church, Tring, Herts, England" is 10 words. Alternatively, "Tring Baptist Church, Akeman Street, Herts, England" but the context says "Akeman Street Baptist Church". So title: "Akeman Street Baptist Church, Tring, Herts, England" – that's 10 words.

Next, the description. Need to combine context with visible subject, setting, action, lighting, details. The image shows the church, red brick, arched windows, door, sign, trees, sky. Capture date is 2026-08-08, but we don't claim it's visible. The sign says "Akeman Street Baptist Church", so the main subject is the church. Setting is the town centre, Tring, Herts. Lighting: sunny, clouds in sky. Distinctive details: red brick, arched windows, dark red door, stone pillars,
</think>

Title: Akeman Street Baptist Church, Tring, Herts, England

Description: Red-brick Akeman Street Baptist Church in Tring, Herts, England, featuring arched windows, a dark red door, stone pillars, and a sign, under a partly cloudy sky with sunlight illuminating the scene.

Keywords: Akeman Street Baptist Church, Tring, Herts, England, Red brick, Arched windows, Dark red door, Stone pillars, Sign, Clouds, Sunlight, Town centre, Church, Hertfordshire, Bushes, Chimney, Christian, Entrance, Europe
```

</details>

</details>

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### mlx-community/Qwen3.5-35B-A3B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-35B-A3B-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Extra text appears before the Title field; Expected model
  wrapper tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 3.12s
- *Generation time:* 89.33s
- *Total time:* 93.44s
- *Input validation time:* 0.953
- *Prompt preparation time:* 0.0219
- *First-token latency:* 83.8
- *Cleanup time:* 0.146
- *Prompt tokens:* 16,557
- *Generation tokens:* 402
- *Total tokens:* 16,959
- *Prompt throughput (raw):* 197 tok/s
- *Generation throughput (raw):* 98.7 tok/s
- *Peak memory:* 24
- *Active memory:* 21
- *Cache memory:* 0.89
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00855
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1332
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 1e20fd8d42056f870933bf98ca6211024744f7ec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
The user wants metadata for an image of a church.

**1. Analyze the Image:**
*   **Subject:** A red brick building, clearly a church.
*   **Signage:** A red sign on the left reads "Akeman Street Baptist Church". It also lists Sunday Services times.
*   **Architecture:** Red brick facade, arched windows (some with white frames, some with leaded glass), a central arched doorway with double doors (dark red/maroon), a slate roof with a small cupola/spire on top. There are stone pillars and a low brick wall in the foreground.
*   **Setting:** Looks like a town street. There are other brick buildings to the left. Trees and bushes surround the church, especially on the right.
*   **Lighting:** Bright daylight, blue sky with scattered white clouds. Shadows suggest the sun is high but slightly to the left/behind the camera.
*   **Details:** A plaque above the door says "Erected 1808 Rebuilt 1852". There are steps leading up to the entrance. A black metal gate is visible on the left.

**2. Process Context:**
*   **Date/Time:** 2026-08-08 15:43:55 UTC+01:00 (This is future-dated relative to now, but I must treat it
&lt;/think&gt;

Title: Akeman Street Baptist Church Exterior
Description: A red brick church with arched windows and a slate roof stands under a blue sky with scattered clouds, featuring a sign for Akeman Street Baptist Church and a stone wall in the foreground.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Red brick, Church, Chapel, Sky, Clouds, Roof, Entrance, Sign, Bushes, Trees, Architecture, Street, Historic
</pre>

<details>
<summary>Exact raw output</summary>

```text
The user wants metadata for an image of a church.

**1. Analyze the Image:**
*   **Subject:** A red brick building, clearly a church.
*   **Signage:** A red sign on the left reads "Akeman Street Baptist Church". It also lists Sunday Services times.
*   **Architecture:** Red brick facade, arched windows (some with white frames, some with leaded glass), a central arched doorway with double doors (dark red/maroon), a slate roof with a small cupola/spire on top. There are stone pillars and a low brick wall in the foreground.
*   **Setting:** Looks like a town street. There are other brick buildings to the left. Trees and bushes surround the church, especially on the right.
*   **Lighting:** Bright daylight, blue sky with scattered white clouds. Shadows suggest the sun is high but slightly to the left/behind the camera.
*   **Details:** A plaque above the door says "Erected 1808 Rebuilt 1852". There are steps leading up to the entrance. A black metal gate is visible on the left.

**2. Process Context:**
*   **Date/Time:** 2026-08-08 15:43:55 UTC+01:00 (This is future-dated relative to now, but I must treat it
</think>

Title: Akeman Street Baptist Church Exterior
Description: A red brick church with arched windows and a slate roof stands under a blue sky with scattered clouds, featuring a sign for Akeman Street Baptist Church and a stone wall in the foreground.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Red brick, Church, Chapel, Sky, Clouds, Roof, Entrance, Sign, Bushes, Trees, Architecture, Street, Historic
```

</details>

</details>

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### mlx-community/Qwen3.5-9B-MLX-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Extra text appears before the Title field; Expected model
  wrapper tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 1.46s
- *Generation time:* 84.52s
- *Total time:* 86.56s
- *Input validation time:* 0.577
- *Prompt preparation time:* 0.000292
- *First-token latency:* 77.3
- *Cleanup time:* 0.157
- *Prompt tokens:* 16,557
- *Generation tokens:* 427
- *Total tokens:* 16,984
- *Prompt throughput (raw):* 214 tok/s
- *Generation throughput (raw):* 68.6 tok/s
- *Peak memory:* 10.0
- *Active memory:* 6.6
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00908
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1332
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
The user wants British-English catalogue metadata for the provided image.

**1. Analyze the Image:**
*   **Subject:** A red brick building, clearly a church.
*   **Signage:** A red sign on the left reads "Akeman Street Baptist Church".
*   **Architecture:**
    *   Red brick facade.
    *   Three arched windows on the upper floor (left, center, right).
    *   Two arched windows on the lower floor (left, right).
    *   A central arched doorway with a maroon double door.
    *   A small cupola or lantern on the roof.
    *   A plaque above the door: "ERECTED 4th May 1808 REBUILT 31st May 1832".
    *   Two stone obelisks/monuments in front of the building.
*   **Setting:**
    *   Surrounded by greenery/bushes/trees.
    *   A brick wall with pillars and gates in the foreground.
    *   Steps leading up to the entrance.
    *   Another brick building visible to the left.
*   **Lighting/Weather:**
    *   Daytime, partly cloudy sky.
    *   Sunlight hitting the roof and upper part of the building.
*   **Context:**
    *   Location: Tr
&lt;/think&gt;

Title: Akeman Street Baptist Church, Tring, Hertfordshire

Description: This red brick Baptist church in Tring, Hertfordshire, features a central maroon door, arched windows, and a small cupola, set against a partly cloudy sky. A brick wall with gates and stone monuments frame the entrance to the historic building.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, red brick, arched windows, cupola, maroon door, stone monuments, brick wall, gates, partly cloudy sky, historic church, Baptist, England, UK, architecture
</pre>

<details>
<summary>Exact raw output</summary>

```text
The user wants British-English catalogue metadata for the provided image.

**1. Analyze the Image:**
*   **Subject:** A red brick building, clearly a church.
*   **Signage:** A red sign on the left reads "Akeman Street Baptist Church".
*   **Architecture:**
    *   Red brick facade.
    *   Three arched windows on the upper floor (left, center, right).
    *   Two arched windows on the lower floor (left, right).
    *   A central arched doorway with a maroon double door.
    *   A small cupola or lantern on the roof.
    *   A plaque above the door: "ERECTED 4th May 1808 REBUILT 31st May 1832".
    *   Two stone obelisks/monuments in front of the building.
*   **Setting:**
    *   Surrounded by greenery/bushes/trees.
    *   A brick wall with pillars and gates in the foreground.
    *   Steps leading up to the entrance.
    *   Another brick building visible to the left.
*   **Lighting/Weather:**
    *   Daytime, partly cloudy sky.
    *   Sunlight hitting the roof and upper part of the building.
*   **Context:**
    *   Location: Tr
</think>

Title: Akeman Street Baptist Church, Tring, Hertfordshire

Description: This red brick Baptist church in Tring, Hertfordshire, features a central maroon door, arched windows, and a small cupola, set against a partly cloudy sky. A brick wall with gates and stone monuments frame the entrance to the historic building.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, red brick, arched windows, cupola, maroon door, stone monuments, brick wall, gates, partly cloudy sky, historic church, Baptist, England, UK, architecture
```

</details>

</details>

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### mlx-community/Qwen3.6-27B-mxfp8

<details>
<summary>Complete evidence: mlx-community/Qwen3.6-27B-mxfp8</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Extra text appears before the Title field; Expected model
  wrapper tokens remain visible; Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 3.35s
- *Generation time:* 146.85s
- *Total time:* 151.03s
- *Input validation time:* 0.804
- *Prompt preparation time:* 0.0148
- *First-token latency:* 109
- *Cleanup time:* 0.248
- *Prompt tokens:* 16,557
- *Generation tokens:* 450
- *Total tokens:* 17,007
- *Prompt throughput (raw):* 152 tok/s
- *Generation throughput (raw):* 12.1 tok/s
- *Peak memory:* 35
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.00959
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1332
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-mxfp8/snapshots/5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
The user wants British-English catalogue metadata for the provided image.

**1. Analyze the Image:**
*   **Subject:** A large, red brick building. It looks like a church or chapel.
*   **Signage:** There is a red sign on the left that says "Akeman Street Baptist Church". It also lists service times: "Sunday Services 10:30am &amp; 6:00pm".
*   **Architecture:**
    *   Red brick construction.
    *   Three large arched windows on the upper level.
    *   Two smaller rectangular windows on the lower level (flanking the door).
    *   A large arched double door (dark red/brown wood).
    *   A slate roof with a small cupola/spire on top.
    *   A stone plaque above the door (looks like a date, possibly 1808 or similar, though hard to read clearly, the sign confirms the name).
    *   Stone obelisks on either side of the entrance path.
*   **Setting:**
    *   Surrounded by greenery (bushes, trees).
    *   A brick wall and gate in the foreground.
    *   A paved path leading up to the church.
    *   Another brick building visible to the left (residential?).
    *   Sky is blue with scattered clouds.
*
&lt;/think&gt;

Title:
Akeman Street Baptist Church in Tring, Hertfordshire

Description:
This image shows the exterior of the Akeman Street Baptist Church in Tring, Hertfordshire, featuring a red brick facade with arched windows and a slate roof topped by a small cupola. A red sign identifies the building and lists service times, while stone obelisks flank the paved path leading to the main entrance.

Keywords:
Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, architecture, arched windows, slate roof, cupola, stone obelisk, entrance, sign, sky, clouds, brick wall, garden, religious building, UK
</pre>

<details>
<summary>Exact raw output</summary>

```text
The user wants British-English catalogue metadata for the provided image.

**1. Analyze the Image:**
*   **Subject:** A large, red brick building. It looks like a church or chapel.
*   **Signage:** There is a red sign on the left that says "Akeman Street Baptist Church". It also lists service times: "Sunday Services 10:30am & 6:00pm".
*   **Architecture:**
    *   Red brick construction.
    *   Three large arched windows on the upper level.
    *   Two smaller rectangular windows on the lower level (flanking the door).
    *   A large arched double door (dark red/brown wood).
    *   A slate roof with a small cupola/spire on top.
    *   A stone plaque above the door (looks like a date, possibly 1808 or similar, though hard to read clearly, the sign confirms the name).
    *   Stone obelisks on either side of the entrance path.
*   **Setting:**
    *   Surrounded by greenery (bushes, trees).
    *   A brick wall and gate in the foreground.
    *   A paved path leading up to the church.
    *   Another brick building visible to the left (residential?).
    *   Sky is blue with scattered clouds.
*  
</think>

Title:
Akeman Street Baptist Church in Tring, Hertfordshire

Description:
This image shows the exterior of the Akeman Street Baptist Church in Tring, Hertfordshire, featuring a red brick facade with arched windows and a slate roof topped by a small cupola. A red sign identifies the building and lists service times, while stone obelisks flank the paved path leading to the main entrance.

Keywords:
Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, architecture, arched windows, slate roof, cupola, stone obelisk, entrance, sign, sky, clouds, brick wall, garden, religious building, UK
```

</details>

</details>

---

<a id="model-mlx-community-step-37-flash-oq2e"></a>

### mlx-community/Step-3.7-Flash-oQ2e

<details>
<summary>Complete evidence: mlx-community/Step-3.7-Flash-oQ2e</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Extra text appears before the Title field; Expected model
  wrapper tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 12.73s
- *Generation time:* 37.08s
- *Total time:* 50.49s
- *Input validation time:* 0.653
- *Prompt preparation time:* 0.0179
- *First-token latency:* 25.1
- *Cleanup time:* 0.265
- *Prompt tokens:* 3,485
- *Generation tokens:* 438
- *Total tokens:* 3,923
- *Prompt throughput (raw):* 139 tok/s
- *Generation throughput (raw):* 39.5 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.42
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.00996
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1320
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.step3p7.processing_step3p7.Step3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Step-3.7-Flash-oQ2e/snapshots/3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Got it, let's tackle this step by step. First, the title needs to be 5-10 words, concrete. The subject is Akeman Street Baptist Church in Tring, England, right? Wait, let's make it accurate. Oh, "Akeman Street Baptist Church, Tring, Hertfordshire" that's good, count words: Akeman (1) Street (2) Baptist (3) Church, (4) Tring, (5) Hertfordshire (6) – that's 6 words, fits 5-10.

Next, the description. 1-2 sentences, factual. Let's include the context: it's the Akeman Street Baptist Church in Tring, Hertfordshire, UK, a red brick 19th-century chapel with arched windows, a central arched entrance, set against a partly cloudy sky, with surrounding residential buildings, a church sign, and landscaped grounds with a stone monument and gateposts. Wait, make it flow. Let's see: "The Akeman Street Baptist Church in Tring, Hertfordshire, UK, is a historic red brick chapel featuring arched windows, a central arched entrance, and a steeply pitched roof, set amid residential buildings, landscaped grounds with a stone monument, and a partly cloudy summer sky." Wait, check the details: the sign is there, the brick wall, the gateposts. Oh right, also mention the sign? Wait, let's make it accurate
&lt;/think&gt;
Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The Akeman Street Baptist Church in Tring, Hertfordshire, UK, is a historic red brick chapel with arched windows, a central arched entrance, and a steeply pitched roof, situated amid adjacent residential buildings, a church sign, landscaped grounds with a stone monument, and a partly cloudy summer sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Europe, red brick chapel, arched windows, church sign, stone monument, brick wall, gateposts, residential buildings, partly cloudy sky, landscaping, Christian place of worship
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this step by step. First, the title needs to be 5-10 words, concrete. The subject is Akeman Street Baptist Church in Tring, England, right? Wait, let's make it accurate. Oh, "Akeman Street Baptist Church, Tring, Hertfordshire" that's good, count words: Akeman (1) Street (2) Baptist (3) Church, (4) Tring, (5) Hertfordshire (6) – that's 6 words, fits 5-10.

Next, the description. 1-2 sentences, factual. Let's include the context: it's the Akeman Street Baptist Church in Tring, Hertfordshire, UK, a red brick 19th-century chapel with arched windows, a central arched entrance, set against a partly cloudy sky, with surrounding residential buildings, a church sign, and landscaped grounds with a stone monument and gateposts. Wait, make it flow. Let's see: "The Akeman Street Baptist Church in Tring, Hertfordshire, UK, is a historic red brick chapel featuring arched windows, a central arched entrance, and a steeply pitched roof, set amid residential buildings, landscaped grounds with a stone monument, and a partly cloudy summer sky." Wait, check the details: the sign is there, the brick wall, the gateposts. Oh right, also mention the sign? Wait, let's make it accurate
</think>
Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The Akeman Street Baptist Church in Tring, Hertfordshire, UK, is a historic red brick chapel with arched windows, a central arched entrance, and a steeply pitched roof, situated amid adjacent residential buildings, a church sign, landscaped grounds with a stone monument, and a partly cloudy summer sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Europe, red brick chapel, arched windows, church sign, stone monument, brick wall, gateposts, residential buildings, partly cloudy sky, landscaping, Christian place of worship
```

</details>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response repeats the task
  instructions instead of only returning the requested fields; Extra text
  appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.33s
- *Generation time:* 13.74s
- *Total time:* 17.60s
- *Input validation time:* 0.502
- *Prompt preparation time:* 0.0214
- *First-token latency:* 0.513
- *Cleanup time:* 0.143
- *Prompt tokens:* 598
- *Generation tokens:* 983
- *Total tokens:* 1,581
- *Prompt throughput (raw):* 1,167 tok/s
- *Generation throughput (raw):* 79.1 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1310
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-8bit/snapshots/7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
I need to create British-English catalogue metadata for the provided image and context.

- Image capture date/time: 2026-08-08 15:43:55 UTC+01:00
- GPS: 51.815915°N, 0.638706°W
- Title hint: Town centre, Tring, England, UK, GBR, Europe
- Description hint: Akeman Street Baptist Church, Tring, Herts
- Keyword hints: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky

- The building is a red brick church.
- It's a multi-story structure (likely a Baptist chapel).
- Arched windows on the upper and lower floors.
- A central arched doorway with dark red doors.
- A sign on the left that says "Akeman Street Baptist Church".
- There is a brick wall in front.
- There are bushes and trees around the building.
- The sky is cloudy with some sunlight.
- The location is Tring, Hertfordshire (based on hints and GPS). Note: One hint says Buckinghamshire, but another says Herts. Tring is in Hertfordshire.

- Title: Akeman Street Baptist Church in Tring, Hertfordshire (8 words)
- Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
- Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre.

- Title: Akeman Street Baptist Church in Tring, Hertfordshire
- Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
- Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town

Wait, let me double-check the location. Tring is in Hertfordshire. The GPS 51.815915°N, 0.638706°W corresponds to Tring, Herts.

Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre

Let's refine the keywords to be 10-18.
1. Akeman Street Baptist Church
2. Tring
3. Hertfordshire
4. England
5. red brick
6. church
7. chapel
8. Christian
9. entrance
10. windows
11. clouds
12. roof
13. architecture
14. sign
15. bushes
16. town centre
Total 16 keywords.

Title: Akeman Street Baptist Church, Tring, Hertfordshire (6 words)
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky. (2 sentences)
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre

Wait, the prompt asks for "British-English". "Town centre" is correct.

Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel's... no.&lt;channel|&gt;Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
I need to create British-English catalogue metadata for the provided image and context.

- Image capture date/time: 2026-08-08 15:43:55 UTC+01:00
- GPS: 51.815915°N, 0.638706°W
- Title hint: Town centre, Tring, England, UK, GBR, Europe
- Description hint: Akeman Street Baptist Church, Tring, Herts
- Keyword hints: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky

- The building is a red brick church.
- It's a multi-story structure (likely a Baptist chapel).
- Arched windows on the upper and lower floors.
- A central arched doorway with dark red doors.
- A sign on the left that says "Akeman Street Baptist Church".
- There is a brick wall in front.
- There are bushes and trees around the building.
- The sky is cloudy with some sunlight.
- The location is Tring, Hertfordshire (based on hints and GPS). Note: One hint says Buckinghamshire, but another says Herts. Tring is in Hertfordshire.

- Title: Akeman Street Baptist Church in Tring, Hertfordshire (8 words)
- Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
- Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre.

- Title: Akeman Street Baptist Church in Tring, Hertfordshire
- Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
- Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town

Wait, let me double-check the location. Tring is in Hertfordshire. The GPS 51.815915°N, 0.638706°W corresponds to Tring, Herts.

Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre

Let's refine the keywords to be 10-18.
1. Akeman Street Baptist Church
2. Tring
3. Hertfordshire
4. England
5. red brick
6. church
7. chapel
8. Christian
9. entrance
10. windows
11. clouds
12. roof
13. architecture
14. sign
15. bushes
16. town centre
Total 16 keywords.

Title: Akeman Street Baptist Church, Tring, Hertfordshire (6 words)
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky. (2 sentences)
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre

Wait, the prompt asks for "British-English". "Town centre" is correct.

Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel's... no.<channel|>Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre
```

</details>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-mxfp8</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Response repeats the task instructions instead of only
  returning the requested fields; Extra text appears before the Title field;
  Duplicate keywords: hertfordshire
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.23s
- *Generation time:* 11.94s
- *Total time:* 15.61s
- *Input validation time:* 0.435
- *Prompt preparation time:* 0.000311
- *First-token latency:* 0.467
- *Cleanup time:* 0.153
- *Prompt tokens:* 598
- *Generation tokens:* 915
- *Total tokens:* 1,513
- *Prompt throughput (raw):* 1,280 tok/s
- *Generation throughput (raw):* 86.0 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1310
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-mxfp8/snapshots/ded389e478f86d498ad9e7f47666e83b166a28f1
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
*   Image: A two-story red brick building, labeled "Akeman Street Baptist Church".
    *   Capture date: 2026-08-08 15:43:55 UTC+01:00
    *   GPS: 51.815915°N, 0.638706°W
    *   Title hint: Town centre, Tring, England, UK, GBR, Europe
    *   Description hint: Akeman Street Baptist Church, Tring, Herts
    *   Keyword hints: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky

    *   The building is Akeman Street Baptist Church.
    *   Location: Tring, Hertfordshire (Note: the hint says Buckinghamshire, but Tring is in Hertfordshire. The description hint says Herts). Let's check GPS: 51.815915°N, 0.638706°W. This is in Tring, Hertfordshire.
    *   Architecture: Red brick, two-story, arched windows, hipped roof with a small cupola/ventator.
    *   Setting: Town centre, wall, bushes, cloudy sky.

    *   *Constraint:* 5-10 words.
    *   *Draft 1:* Akeman Street Baptist Church in Tring, Hertfordshire (7 words)
    *   *Draft 2:* Red brick Akeman Street Baptist Church, Tring England (7 words)
    *   *Selection:* Akeman Street Baptist Church in Tring, Hertfordshire.

    *   *Constraint:* 1-2 sentences, factual, combining context, subject, setting, lighting, details.
    *   *Draft:* The Akeman Street Baptist Church is a two-story red brick building featuring large arched windows and a hipped roof. It is situated behind a low stone wall with lush green bushes under a partly cloudy sky.
    *   *Refinement:* The Akeman Street Baptist Church in Tring is a red-brick building with large arched windows and a hipped roof. The structure sits behind a low stone wall surrounded by lush greenery under a bright cloudy sky.

    *   *Constraint:* 10-18 unique, comma-separated keywords.
    *   *List:* Akeman Street Baptist Church, Tring, Hertfordshire, England, Church, Chapel, red brick, arched windows, entrance, roof, wall, bushes, sky, clouds, centre, Christian.
    *   *Checking count:*
        1. Akeman Street Baptist Church
        2. Tring
        3. Hertfordshire
        4. England
        5. Church
        6. Chapel
        7. red brick
        8. arched windows
        9. entrance
        10. roof
        11. wall
        12. bushes
        13. sky
        14. clouds
        15. centre
        16. Christian
    *   Total 16. Fits 10-18 range.

    *   Title: Akeman Street Baptist Church in Tring, Hertfordshire.
    *   Description: The Akeman Street Baptist Church in Tring is a red-brick building with large arched windows and a hipped roof. The structure sits behind a low stone wall surrounded by lush greenery under a bright cloudy sky.
    *   Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, Church, Chapel, red brick, arched windows, entrance, roof, wall, bushes, sky, clouds, centre, Christian.&lt;channel|&gt;Title: Akeman Street Baptist Church in Tring, Hertfordshire.
Description: The Akeman Street Baptist Church in Tring is a red-brick building with large arched windows and a hipped roof. The structure sits behind a low stone wall surrounded by much greenery under a bright cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, Church, Chapel, red brick, arched windows, entrance, roof, wall, bushes, sky, clouds, centre, Christian
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
*   Image: A two-story red brick building, labeled "Akeman Street Baptist Church".
    *   Capture date: 2026-08-08 15:43:55 UTC+01:00
    *   GPS: 51.815915°N, 0.638706°W
    *   Title hint: Town centre, Tring, England, UK, GBR, Europe
    *   Description hint: Akeman Street Baptist Church, Tring, Herts
    *   Keyword hints: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky

    *   The building is Akeman Street Baptist Church.
    *   Location: Tring, Hertfordshire (Note: the hint says Buckinghamshire, but Tring is in Hertfordshire. The description hint says Herts). Let's check GPS: 51.815915°N, 0.638706°W. This is in Tring, Hertfordshire.
    *   Architecture: Red brick, two-story, arched windows, hipped roof with a small cupola/ventator.
    *   Setting: Town centre, wall, bushes, cloudy sky.

    *   *Constraint:* 5-10 words.
    *   *Draft 1:* Akeman Street Baptist Church in Tring, Hertfordshire (7 words)
    *   *Draft 2:* Red brick Akeman Street Baptist Church, Tring England (7 words)
    *   *Selection:* Akeman Street Baptist Church in Tring, Hertfordshire.

    *   *Constraint:* 1-2 sentences, factual, combining context, subject, setting, lighting, details.
    *   *Draft:* The Akeman Street Baptist Church is a two-story red brick building featuring large arched windows and a hipped roof. It is situated behind a low stone wall with lush green bushes under a partly cloudy sky.
    *   *Refinement:* The Akeman Street Baptist Church in Tring is a red-brick building with large arched windows and a hipped roof. The structure sits behind a low stone wall surrounded by lush greenery under a bright cloudy sky.

    *   *Constraint:* 10-18 unique, comma-separated keywords.
    *   *List:* Akeman Street Baptist Church, Tring, Hertfordshire, England, Church, Chapel, red brick, arched windows, entrance, roof, wall, bushes, sky, clouds, centre, Christian.
    *   *Checking count:*
        1. Akeman Street Baptist Church
        2. Tring
        3. Hertfordshire
        4. England
        5. Church
        6. Chapel
        7. red brick
        8. arched windows
        9. entrance
        10. roof
        11. wall
        12. bushes
        13. sky
        14. clouds
        15. centre
        16. Christian
    *   Total 16. Fits 10-18 range.

    *   Title: Akeman Street Baptist Church in Tring, Hertfordshire.
    *   Description: The Akeman Street Baptist Church in Tring is a red-brick building with large arched windows and a hipped roof. The structure sits behind a low stone wall surrounded by lush greenery under a bright cloudy sky.
    *   Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, Church, Chapel, red brick, arched windows, entrance, roof, wall, bushes, sky, clouds, centre, Christian.<channel|>Title: Akeman Street Baptist Church in Tring, Hertfordshire.
Description: The Akeman Street Baptist Church in Tring is a red-brick building with large arched windows and a hipped roof. The structure sits behind a low stone wall surrounded by much greenery under a bright cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, Church, Chapel, red brick, arched windows, entrance, roof, wall, bushes, sky, clouds, centre, Christian
```

</details>

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
- *Model load time:* 2.55s
- *Generation time:* 3.84s
- *Total time:* 6.85s
- *Input validation time:* 0.434
- *Prompt preparation time:* 0.00345
- *First-token latency:* 0.647
- *Cleanup time:* 0.182
- *Prompt tokens:* 589
- *Generation tokens:* 87
- *Total tokens:* 676
- *Prompt throughput (raw):* 911 tok/s
- *Generation throughput (raw):* 34.9 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0114
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1309
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3n-E4B-it-bf16/snapshots/d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
A red brick building with arched windows and a dark red door stands on a sunny day in Tring, Hertfordshire, UK. The building is the Akeman Street Baptist Church, as indicated by a sign in the foreground. The church has a grey tiled roof and is surrounded by greenery, including bushes and trees. A brick wall with pillars encloses the property, and steps lead up to the entrance. The sky is partly cloudy.
</pre>

</details>

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### mlx-community/gemma-4-26b-a4b-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-26b-a4b-it-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the task instructions instead of only
  returning the requested fields; Extra text appears before the Title field;
  Response appears cut off at the token limit; Internal reasoning block
  appears incomplete; Keyword list has 36 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Model load time:* 2.38s
- *Generation time:* 13.77s
- *Total time:* 16.87s
- *Input validation time:* 0.69
- *Prompt preparation time:* 0.0212
- *First-token latency:* 0.571
- *Cleanup time:* 0.117
- *Prompt tokens:* 598
- *Generation tokens:* 1,000
- *Total tokens:* 1,598
- *Prompt throughput (raw):* 1,048 tok/s
- *Generation throughput (raw):* 79.9 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.26
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0119
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1310
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
*   Capture date/time: 2026-08-08 15:43:55 UTC+01:00
    *   GPS: 51.815915°N, 0.638706°W
    *   Title hint: Town centre, Tring, England, UK, GBR, Europe
    *   Description hint: Akeman Street Baptist Church, Tring, Herts
    *   Keywords: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky

    *   Subject: A two-story red brick building with arched windows.
    *   Signage: A sign on the left says "Akeman Street Baptist Church".
     least one sign on the building itself (small plaque above the door).
    *   Architecture: Red brick, arched windows (three on top, two on bottom flanking the door), central arched doorway with dark red doors, hipped roof with a small finial/vent.
    *   Setting: Street level, stone/brick wall in front, some greenery/bushes, cloudy sky.
    *   Location: Tring, Hertfordshire (based on description hint and GPS).
    *   Lighting: Daylight, slightly overcast/cloudy sky.

    *   *Title:* Akeman Street Baptist Church in Tring, England. (7 words) - Good.
    *   *Description:* A two-story red brick building housing the Akeman Street Baptist Church in Tring, Hertfordshire, under a cloudy sky. The structure features arched windows and a central dark red door.
    *   *Keywords:* Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, architecture, church, chapel, arched windows, historic building, Europe, UK, street level, cloudy sky, brick wall, entrance.

    *   *Title:* Akeman Street Baptist Church in Tring, England
    *   *Description:* The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central dark red door under a cloudy sky.
    *   *Keywords:* Akeman Street Baptist Church, Tolog, Hertfordshire, England, red brick, architecture, church, chapel, arched windows, historic building, Europe, UK, brick wall, entrance, sky, clouds, street level.

    *Wait, let's refine the title and description to be more professional.*

    *Title:* Akeman Street Baptist Church in Tring, England
    *Description:* The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central dark red door under a cloudy sky.
    *Keywords:* Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, architecture, church, chapel, arched windows, historic building, Europe, UK, brick wall, entrance, sky, clouds, street level.

    *Wait, the prompt says "a concrete 5-10-word title".*
    "Akeman Street Baptist Church in Tring, England" is 7 words. Perfect.

    *Description: 1-2 sentences.*
    "The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central dark red door under a cloudy sky." (21 words). Perfect.

    *Keywords: 10-18 unique, comma-separated.*
    1. Akeman Street Baptist Church
    2. Tring
    3. Hertfordshire
    4. England
    5. red brick
    6. architecture
    7. church
    8. chapel
    9. arched windows
    10. historic building
    11. Europe
            12. UK
    13. brick wall
    14. entrance
    15. sky
    16. clouds
    17. street level

    *Let's check the keywords against the hints.*
    Hints: Adobe Stock (omit), Akeman Street Baptist Church (keep), Any Vision (omit), Buckinghamshire (the hint says Tring, Herts, so Buckinghamshire is likely a mistake in the hint or refers to a different context, I will stick to Herts/Tring), Bushes (can add), Chapel (keep), Chimney (not visible), Christian (can add), Church (keep), Clouds (keep), England (keep), Entrance (keep), Europe (keep), Hertfordshire (keep), Locations (omit), Objects (omit), Red brick (keep), Roof (can add),
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
*   Capture date/time: 2026-08-08 15:43:55 UTC+01:00
    *   GPS: 51.815915°N, 0.638706°W
    *   Title hint: Town centre, Tring, England, UK, GBR, Europe
    *   Description hint: Akeman Street Baptist Church, Tring, Herts
    *   Keywords: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky

    *   Subject: A two-story red brick building with arched windows.
    *   Signage: A sign on the left says "Akeman Street Baptist Church".
     least one sign on the building itself (small plaque above the door).
    *   Architecture: Red brick, arched windows (three on top, two on bottom flanking the door), central arched doorway with dark red doors, hipped roof with a small finial/vent.
    *   Setting: Street level, stone/brick wall in front, some greenery/bushes, cloudy sky.
    *   Location: Tring, Hertfordshire (based on description hint and GPS).
    *   Lighting: Daylight, slightly overcast/cloudy sky.

    *   *Title:* Akeman Street Baptist Church in Tring, England. (7 words) - Good.
    *   *Description:* A two-story red brick building housing the Akeman Street Baptist Church in Tring, Hertfordshire, under a cloudy sky. The structure features arched windows and a central dark red door.
    *   *Keywords:* Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, architecture, church, chapel, arched windows, historic building, Europe, UK, street level, cloudy sky, brick wall, entrance.

    *   *Title:* Akeman Street Baptist Church in Tring, England
    *   *Description:* The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central dark red door under a cloudy sky.
    *   *Keywords:* Akeman Street Baptist Church, Tolog, Hertfordshire, England, red brick, architecture, church, chapel, arched windows, historic building, Europe, UK, brick wall, entrance, sky, clouds, street level.

    *Wait, let's refine the title and description to be more professional.*

    *Title:* Akeman Street Baptist Church in Tring, England
    *Description:* The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central dark red door under a cloudy sky.
    *Keywords:* Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, architecture, church, chapel, arched windows, historic building, Europe, UK, brick wall, entrance, sky, clouds, street level.

    *Wait, the prompt says "a concrete 5-10-word title".*
    "Akeman Street Baptist Church in Tring, England" is 7 words. Perfect.

    *Description: 1-2 sentences.*
    "The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central dark red door under a cloudy sky." (21 words). Perfect.

    *Keywords: 10-18 unique, comma-separated.*
    1. Akeman Street Baptist Church
    2. Tring
    3. Hertfordshire
    4. England
    5. red brick
    6. architecture
    7. church
    8. chapel
    9. arched windows
    10. historic building
    11. Europe
            12. UK
    13. brick wall
    14. entrance
    15. sky
    16. clouds
    17. street level

    *Let's check the keywords against the hints.*
    Hints: Adobe Stock (omit), Akeman Street Baptist Church (keep), Any Vision (omit), Buckinghamshire (the hint says Tring, Herts, so Buckinghamshire is likely a mistake in the hint or refers to a different context, I will stick to Herts/Tring), Bushes (can add), Chapel (keep), Chimney (not visible), Christian (can add), Church (keep), Clouds (keep), England (keep), Entrance (keep), Europe (keep), Hertfordshire (keep), Locations (omit), Objects (omit), Red brick (keep), Roof (can add),
```

</details>

</details>

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### mlx-community/gemma-4-31b-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-31b-it-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title; Response repeats the task
  instructions instead of only returning the requested fields; Extra text
  appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Model load time:* 2.64s
- *Generation time:* 45.87s
- *Total time:* 48.95s
- *Input validation time:* 0.428
- *Prompt preparation time:* 0.0003
- *First-token latency:* 1.14
- *Cleanup time:* 0.182
- *Prompt tokens:* 598
- *Generation tokens:* 857
- *Total tokens:* 1,455
- *Prompt throughput (raw):* 524 tok/s
- *Generation throughput (raw):* 19.4 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.99
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0124
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1310
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 696d436c404745a59f30e4939a658162b0a9e57f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/696d436c404745a59f30e4939a658162b0a9e57f
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
*   Image: A red brick building, identified as Akeman Street Baptist Church by a sign. It's a two-story structure with arched windows and a central arched door. There's a low stone wall in front with steps leading up to the entrance. Greenery (bushes, trees) surrounds the building. The sky is cloudy.
    *   Capture date/time: 2026-08-08 15:43:55 UTC+01:00 (Authoritative)
    *   GPS: 51.815915°N, 0.638706°W (Authoritative)
    *   Title hint: Town centre, Tring, England, UK, GBR, Europe
    *   Description hint: Akeman Street Baptist Church, Tring, Herts
    *   Keyword hints: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky.

    *   *Title:* Needs to be 5-10 words.
        *   Draft 1: Akeman Street Baptist Church in Tring, Hertfordshire, England. (8 words) - Good.
        *   Draft 2: The Akeman Street Baptist Church, Tring, England, UK. (8 words) - Good.
        *   Selection: Akeman Street Baptist Church in Tring, Hertfordshire, England.

    *   *Description:* 1-2 factual sentences. Combine context (Tring, Herts, Akeman Street Baptist Church) with visible details (red brick, arched windows, stone wall, cloudy sky).
        *   Draft 1: The red brick Akeman Street Baptist Church stands in Tring, Hertfordshire, featuring arched windows and a central entrance behind a low stone wall. The scene is captured under a cloudy sky in the afternoon.
        *   Refining: "The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central entrance behind a low stone wall. The building is set against a cloudy sky with surrounding greenery."

    *   *Keywords:* 10-18 unique, comma-separated.
        *   From hints: Akeman Street Baptist Church, Chapel, Christian, Church, England, Europe, Hertfordshire, Red brick, Roof, Sign, Sky, Clouds, Bushes.
        *   From image: Arched windows, stone wall, entrance, Tring.
        *   Removing irrelevant/generic hints: Adobe Stock, Any Vision, Buckinghamshire (it's in Herts), Locations, Objects.
        *   List: Akeman Street Baptist Church, Tring, Hertfordshire, England, church, chapel, red brick, arched windows, stone wall, entrance, Christian, cloudy sky, Europe, architecture. (14 keywords).

    *   Title: Akeman Street Baptist Church in Tring, Hertfordshire, England (8 words) - OK.
    *   Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central entrance behind a low stone wall. The building is set against a cloudy sky with surrounding greenery. (2 sentences) - OK.
    *   Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, church, chapel, red brick, arched windows, stone wall, entrance, Christian, cloudy sky, Europe, architecture. (14 keywords) - OK.&lt;channel|&gt;Title: Akeman Street Baptist Church in Tring, Hertfordshire, England
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central entrance behind a low stone wall. The building is set against a cloudy sky with surrounding greenery.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, church, chapel, red brick, arched windows, stone wall, entrance, Christian, cloudy sky, Europe, architecture
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
*   Image: A red brick building, identified as Akeman Street Baptist Church by a sign. It's a two-story structure with arched windows and a central arched door. There's a low stone wall in front with steps leading up to the entrance. Greenery (bushes, trees) surrounds the building. The sky is cloudy.
    *   Capture date/time: 2026-08-08 15:43:55 UTC+01:00 (Authoritative)
    *   GPS: 51.815915°N, 0.638706°W (Authoritative)
    *   Title hint: Town centre, Tring, England, UK, GBR, Europe
    *   Description hint: Akeman Street Baptist Church, Tring, Herts
    *   Keyword hints: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky.

    *   *Title:* Needs to be 5-10 words.
        *   Draft 1: Akeman Street Baptist Church in Tring, Hertfordshire, England. (8 words) - Good.
        *   Draft 2: The Akeman Street Baptist Church, Tring, England, UK. (8 words) - Good.
        *   Selection: Akeman Street Baptist Church in Tring, Hertfordshire, England.

    *   *Description:* 1-2 factual sentences. Combine context (Tring, Herts, Akeman Street Baptist Church) with visible details (red brick, arched windows, stone wall, cloudy sky).
        *   Draft 1: The red brick Akeman Street Baptist Church stands in Tring, Hertfordshire, featuring arched windows and a central entrance behind a low stone wall. The scene is captured under a cloudy sky in the afternoon.
        *   Refining: "The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central entrance behind a low stone wall. The building is set against a cloudy sky with surrounding greenery."

    *   *Keywords:* 10-18 unique, comma-separated.
        *   From hints: Akeman Street Baptist Church, Chapel, Christian, Church, England, Europe, Hertfordshire, Red brick, Roof, Sign, Sky, Clouds, Bushes.
        *   From image: Arched windows, stone wall, entrance, Tring.
        *   Removing irrelevant/generic hints: Adobe Stock, Any Vision, Buckinghamshire (it's in Herts), Locations, Objects.
        *   List: Akeman Street Baptist Church, Tring, Hertfordshire, England, church, chapel, red brick, arched windows, stone wall, entrance, Christian, cloudy sky, Europe, architecture. (14 keywords).

    *   Title: Akeman Street Baptist Church in Tring, Hertfordshire, England (8 words) - OK.
    *   Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central entrance behind a low stone wall. The building is set against a cloudy sky with surrounding greenery. (2 sentences) - OK.
    *   Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, church, chapel, red brick, arched windows, stone wall, entrance, Christian, cloudy sky, Europe, architecture. (14 keywords) - OK.<channel|>Title: Akeman Street Baptist Church in Tring, Hertfordshire, England
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central entrance behind a low stone wall. The building is set against a cloudy sky with surrounding greenery.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, church, chapel, red brick, arched windows, stone wall, entrance, Christian, cloudy sky, Europe, architecture
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
- *Observations:* Missing or empty fields: Title, Description; Extra text
  appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type llava_next)
- *Model load time:* 0.93s
- *Generation time:* 5.30s
- *Total time:* 6.75s
- *Input validation time:* 0.517
- *Prompt preparation time:* 0.00156
- *First-token latency:* 2.78
- *Cleanup time:* 0.0885
- *Prompt tokens:* 2,514
- *Generation tokens:* 100
- *Total tokens:* 2,614
- *Prompt throughput (raw):* 903 tok/s
- *Generation throughput (raw):* 59.9 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.90
- *Model-load active memory:* 8.05
- *Post-cleanup active memory:* 0.0125
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1254
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--llava-v1.6-mistral-7b-8bit/snapshots/b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Akeman Street Baptist Church, Tring, Herts, England, UK, GBR, Europe

Title:
Description:
Keywords: [/Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky]
</pre>

<details>
<summary>Exact raw output</summary>

```text
Akeman Street Baptist Church, Tring, Herts, England, UK, GBR, Europe

Title:
Description:
Keywords: [/Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky] 
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
- *Model load time:* 0.48s
- *Generation time:* 1.01s
- *Total time:* 1.93s
- *Input validation time:* 0.43
- *Prompt preparation time:* 0.00133
- *First-token latency:* 0.146
- *Cleanup time:* 0.084
- *Prompt tokens:* 328
- *Generation tokens:* 70
- *Total tokens:* 398
- *Prompt throughput (raw):* 2,243 tok/s
- *Generation throughput (raw):* 330 tok/s
- *Peak memory:* 2.1
- *Active memory:* 0.61
- *Cache memory:* 0.30
- *Model-load active memory:* 0.611
- *Post-cleanup active memory:* 0.0131
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1338
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--nanoLLaVA-1.5-4bit/snapshots/5240204744963d72823e5de933c528c4aa82dfca
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Akeman Street Baptist Church, Tring, England, UK, GBR, Europe
Description: A brick church in the town centre of Tring, England, UK, GBR, Europe. The church is a red brick building with a red roof and a sign on the front. It is a prominent landmark in the area.
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
- *Model load time:* 1.16s
- *Generation time:* 30.88s
- *Total time:* 32.53s
- *Input validation time:* 0.431
- *Prompt preparation time:* 0.000182
- *First-token latency:* 1.16
- *Cleanup time:* 0.23
- *Prompt tokens:* 4,416
- *Generation tokens:* 1,000
- *Total tokens:* 5,416
- *Prompt throughput (raw):* 3,801 tok/s
- *Generation throughput (raw):* 34.7 tok/s
- *Peak memory:* 4.5
- *Active memory:* 1.7
- *Cache memory:* 2.0
- *Model-load active memory:* 1.73
- *Post-cleanup active memory:* 0.0141
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1238
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--paligemma2-3b-pt-896-4bit/snapshots/a26bac48c7a661dfdafe1799c90177f818e79925
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 300,
  "thinking_end_token": "&lt;/think&gt;", "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
- Title hint: Tring, England, UK, GBR, Europe
- Description hint: Tring, England, UK, GBR, Europe
- Keyword hints: Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe
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
- *Model load time:* 0.20s
- *Generation time:* -
- *Total time:* 1.07s
- *Input validation time:* 0.868
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.102
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
- *Requested maximum tokens:* 1000
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11951, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11428, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 748, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 964, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 840, in load_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12381, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11966, in _run_model_generation
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
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 4404.15it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[23:57:20] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
```

</details>

---
