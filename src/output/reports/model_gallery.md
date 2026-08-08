# Model Output Gallery

Generated on: 2026-08-08 13:18:50 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS             | Prefill/first s | Peak GB | Gen tok | Observations                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|---------------------|-----------------|---------|---------|--------------------------------------------------------------------------------------------------------------------|
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `usable`              | 23.61s  | 5.06 tok/s          | 1.82            | 25      | 95      | none                                                                                                               |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `usable`              | 3.80s   | 56.8 tok/s          | 0.38            | 9.4     | 100     | none                                                                                                               |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `usable`              | 7.34s   | 32.2 tok/s          | 2.14            | 18      | 87      | none                                                                                                               |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 6.31s   | 33.9 tok/s          | 1.09            | 18      | 93      | none                                                                                                               |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `usable`              | 7.94s   | 21.1 tok/s          | 1.46            | 15      | 89      | none                                                                                                               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 5.80s   | 67.2 tok/s          | 1.81            | 12      | 121     | none                                                                                                               |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 3.25s   | 185 tok/s           | 0.84            | 6.4     | 117     | none                                                                                                               |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | 3.68s   | 56.7 tok/s          | 0.28            | 9.4     | 100     | none                                                                                                               |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `usable`              | 79.46s  | 30.5 tok/s          | 72.66           | 26      | 104     | none                                                                                                               |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `usable`              | 95.44s  | 14.9 tok/s          | 83.94           | 38      | 107     | none                                                                                                               |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 68.12s  | 110 tok/s           | 63.66           | 26      | 94      | none                                                                                                               |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `usable`              | 67.70s  | 91.8 tok/s          | 62.56           | 35      | 83      | none                                                                                                               |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `usable`              | 78.43s  | 65.9 tok/s          | 65.88           | 76      | 102     | none                                                                                                               |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | 28.19s  | 42.2 tok/s          | 17.89           | 70      | 119     | none                                                                                                               |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `usable`              | 12.97s  | 16.5 tok/s          | 1.22            | 32      | 122     | none                                                                                                               |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.38s   | 127 tok/s           | 0.44            | 16      | 96      | none                                                                                                               |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 7.74s   | 25.8 tok/s          | 1.07            | 20      | 83      | none                                                                                                               |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | 6.30s   | 39.6 tok/s          | 1.71            | 15      | 82      | none                                                                                                               |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `usable`              | 9.09s   | 20.1 tok/s          | 1.44            | 27      | 88      | none                                                                                                               |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `usable_with_caveats` | 2.76s   | 129 tok/s           | 0.39            | 5.5     | 104     | title/keyword constraints failed; draft hints copied unchanged                                                     |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable_with_caveats` | 1.20s   | 515 tok/s           | 0.03            | 1.3     | 107     | title/keyword constraints failed                                                                                   |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | 16.30s  | 91.4 tok/s          | 13.67           | 8.4     | 86      | title/keyword constraints failed                                                                                   |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | 9.79s   | 29.8 tok/s          | 2.88            | 22      | 110     | title/keyword constraints failed                                                                                   |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | 11.19s  | 74.1 tok/s          | 8.04            | 8.4     | 83      | title/keyword constraints failed                                                                                   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | 7.74s   | 32.1 tok/s          | 1.14            | 19      | 121     | role tokens visible                                                                                                |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `usable_with_caveats` | 1.50s   | 331 tok/s           | 0.09            | 3.0     | 80      | title/keyword constraints failed                                                                                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | 6.82s   | 64.0 tok/s          | 1.88            | 12      | 178     | title/keyword constraints failed                                                                                   |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | 5.03s   | 53.3 tok/s          | 0.67            | 11      | 105     | title/keyword constraints failed                                                                                   |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `usable_with_caveats` | 6.98s   | 30.8 tok/s          | 0.64            | 18      | 108     | title/keyword constraints failed                                                                                   |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable_with_caveats` | 74.12s  | 63.8 tok/s          | 61.60           | 76      | 107     | title/keyword constraints failed                                                                                   |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | 71.13s  | 212 tok/s           | 68.88           | 5.1     | 102     | title/keyword constraints failed; draft hints copied unchanged                                                     |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | 22.06s  | 91.1 tok/s          | 19.22           | 8.4     | 86      | title/keyword constraints failed                                                                                   |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable_with_caveats` | 66.35s  | 89.9 tok/s          | 62.61           | 11      | 112     | title/keyword constraints failed                                                                                   |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable_with_caveats` | 96.18s  | 15.9 tok/s          | 86.51           | 38      | 87      | title/keyword constraints failed                                                                                   |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `usable_with_caveats` | 2.81s   | 124 tok/s           | 0.42            | 5.5     | 104     | title/keyword constraints failed; draft hints copied unchanged                                                     |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | 2.81s   | 130 tok/s           | 0.33            | 5.5     | 125     | title/keyword constraints failed                                                                                   |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | 9.46s   | 29.5 tok/s          | 1.03            | 18      | 155     | title/keyword constraints failed                                                                                   |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `usable_with_caveats` | 6.17s   | 47.9 tok/s          | 0.39            | 17      | 130     | title/keyword constraints failed                                                                                   |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `unusable`            | 11.10s  | 73.3 tok/s          | 1.40            | 18      | 500     | repeated text; missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 16.03s  | 42.6 tok/s          | 1.90            | 14      | 500     | missing required fields; echoes instructions; cut off at token limit                                               |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 20.93s  | 56.5 tok/s          | 4.00            | 60      | 500     | repeated text; missing required fields; cut off at token limit                                                     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 2.55s   | 353 tok/s           | 0.10            | 2.2     | 130     | missing required fields; extra text before Title                                                                   |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 22.10s  | 40.0 tok/s          | 7.11            | 13      | 500     | missing required fields; cut off at token limit; incomplete thinking block                                         |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `unusable`            | 12.09s  | 51.9 tok/s          | 8.13            | 11      | 93      | control tokens visible; missing required fields; extra text before Title                                           |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `unusable`            | 31.24s  | 35.1 tok/s          | 19.65           | 78      | 106     | control tokens visible; missing required fields; extra text before Title                                           |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | 112.56s | 4.68 tok/s          | 1.52            | 40      | 500     | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block                |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `unusable`            | 11.59s  | 65.2 tok/s          | 0.89            | 22      | 500     | missing required fields; cut off at token limit; incomplete thinking block                                         |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `unusable`            | 1.84s   | 187 tok/s           | 0.10            | 4.1     | 93      | missing required fields                                                                                            |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 3.98s   | 275 tok/s           | 0.36            | 3.8     | 500     | missing required fields; cut off at token limit; incomplete thinking block                                         |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | 32.70s  | 5.93 tok/s          | 2.95            | 26      | 156     | missing required fields                                                                                            |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | 27.75s  | 84.6 tok/s          | 19.70           | 8.4     | 500     | missing required fields; cut off at token limit                                                                    |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | 31.61s  | 50.3 tok/s          | 19.42           | 14      | 500     | repeated text; cut off at token limit; title/keyword constraints failed                                            |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `unusable`            | 5.94s   | 61.4 tok/s          | 0.35            | 29      | 81      | missing required fields; extra text before Title                                                                   |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `unusable`            | 6.12s   | 53.5 tok/s          | 0.35            | 28      | 80      | missing required fields; extra text before Title                                                                   |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `unusable`            | 6.52s   | 126 tok/s           | 0.27            | 5.9     | 500     | repeated text; missing required fields; cut off at token limit                                                     |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `unusable`            | 11.94s  | insufficient sample | 4.01            | 64      | 1       | empty response; missing required fields                                                                            |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | 5.26s   | 61.8 tok/s          | 2.52            | 9.7     | 52      | missing required fields                                                                                            |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 1.60s   | 348 tok/s           | 0.08            | 2.3     | 71      | missing required fields                                                                                            |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `unusable`            | 6.52s   | 32.0 tok/s          | 0.97            | 12      | 97      | missing required fields                                                                                            |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `unusable`            | 24.81s  | 5.29 tok/s          | 1.33            | 26      | 107     | missing required fields                                                                                            |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `unusable`            | 10.56s  | 19.4 tok/s          | 0.43            | 11      | 153     | missing required fields                                                                                            |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 15.56s  | 41.0 tok/s          | 1.17            | 4.6     | 500     | repeated text; missing required fields; cut off at token limit                                                     |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `unusable`            | 1.59s   | 115 tok/s           | 0.09            | 4.5     | 19      | missing required fields                                                                                            |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                                   | `not_evaluated`       | 0.51s   | -                   | -               | -       | -       | none                                                                                                               |
<!-- markdownlint-enable MD034 MD049 -->

## Resource Highlights

Fastest valid generation: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 515 tok/s

Average valid generation throughput: 84.1 tok/s

Lowest captured peak memory: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.3 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                                                                       |
|--------------------------------------------------------------------------------------------------------------|-----------------|--------------------------------------------------------------------------------------------------------------------|
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                      | `unusable`      | repeated text; missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | missing required fields; echoes instructions; cut off at token limit                                               |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable`      | repeated text; missing required fields; cut off at token limit                                                     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable`      | missing required fields; extra text before Title                                                                   |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable`      | missing required fields; cut off at token limit; incomplete thinking block                                         |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                               | `unusable`      | control tokens visible; missing required fields; extra text before Title                                           |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                         | `unusable`      | control tokens visible; missing required fields; extra text before Title                                           |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | `unusable`      | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block                |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                  | `unusable`      | missing required fields; cut off at token limit; incomplete thinking block                                         |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                 | `unusable`      | missing required fields; cut off at token limit; incomplete thinking block                                         |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable`      | missing required fields; cut off at token limit                                                                    |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                            |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | `unusable`      | missing required fields; extra text before Title                                                                   |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | `unusable`      | missing required fields; extra text before Title                                                                   |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                  | `unusable`      | repeated text; missing required fields; cut off at token limit                                                     |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                    | `unusable`      | empty response; missing required fields                                                                            |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)    | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)    | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)      | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable`      | repeated text; missing required fields; cut off at token limit                                                     |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                            | `unusable`      | missing required fields                                                                                            |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                        | `not_evaluated` | none                                                                                                               |
<!-- markdownlint-enable MD034 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.11`
- `mlx`: `0.32.1.dev20260808+6539d1807`
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

- *Title:* , Seafront, Seaford, England, UK, GBR, Europe
- *Description:* Two inflatable boats with outboard motors are speeding across
  the ocean, leaving white wakes behind them, against a clear blue sky and a
  distinct horizon line.
- *Keywords:* Adobe Stock, Any Vision, Blue sky, Driver, England, Europe,
  Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing,
  Seaford, Sky, UK, Vehicles, Water, action, beautiful, blue, boat, boating,
  day, drive, extreme, fast, fun, landscape, leisure, lifestyle, luxury,
  marine, motor, nature, ocean, outdoors, passengers, power, powerboat,
  recreation, sea, seafront, ship, speed, speed boat, speedboat, splash,
  sport, summer, sun, sunny, tour, tourism, transport, transportation, travel,
  trip, vacation, view, wake, watercraft, wave, waves, white, yacht, zodiac
- *Date:* 2026-08-07 17:17:16 UTC+01:00
- *Time:* 17:17:16

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
> &#45; Capture date/time: 2026-08-07 17:17:16 UTC+01:00
>
> &#8203;Descriptive hints:
> &#45; Title hint: Seafront, Seaford, England, UK, GBR, Europe
> &#45; Description hint: Two inflatable boats with outboard motors are speeding
> across the ocean, leaving white wakes behind them, against a clear blue sky
> and a distinct horizon line.
> &#45; Keyword hints: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe,
> Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing,
> Seaford, Sky, UK, Vehicles, Water, action, beautiful
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

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### meta-llama/Llama-3.2-11B-Vision-Instruct

<details>
<summary>Complete evidence: meta-llama/Llama-3.2-11B-Vision-Instruct</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 2.20s
- *Generation time:* 21.10s
- *Total time:* 23.61s
- *Input validation time:* 0.302
- *Prompt preparation time:* 0.00537
- *First-token latency:* 1.82
- *Cleanup time:* 0.102
- *Prompt tokens:* 298
- *Generation tokens:* 95
- *Total tokens:* 393
- *Prompt throughput (raw):* 163 tok/s
- *Generation throughput (raw):* 5.06 tok/s
- *Peak memory:* 25
- *Active memory:* 21
- *Cache memory:* 3.5
- *Model-load active memory:* 21.3
- *Post-cleanup active memory:* 0.00175
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1412
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 9eb2daaa8597bf192a8b0e73f848f3a102794df5
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|eot_id\|>

*Readable output:*

<pre class="model-output-readable">
Title: Speeding Inflatable Boats on the Open Ocean

Description: Two inflatable boats with outboard motors speed across the ocean, leaving behind white wakes, under a clear blue sky with a distinct horizon line, on a sunny day in Seaford, England, UK.

Keywords: Ocean, Seaford, England, UK, Inflatable boats, Outboard motors, White wakes, Clear blue sky, Distinct horizon line, Sunny day.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Speeding Inflatable Boats on the Open Ocean

Description: Two inflatable boats with outboard motors speed across the ocean, leaving behind white wakes, under a clear blue sky with a distinct horizon line, on a sunny day in Seaford, England, UK. 

Keywords: Ocean, Seaford, England, UK, Inflatable boats, Outboard motors, White wakes, Clear blue sky, Distinct horizon line, Sunny day.
```

</details>

</details>

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### microsoft/Phi-3.5-vision-instruct

<details>
<summary>Complete evidence: microsoft/Phi-3.5-vision-instruct</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type phi3_v)
- *Model load time:* 0.92s
- *Generation time:* 2.57s
- *Total time:* 3.80s
- *Input validation time:* 0.304
- *Prompt preparation time:* 0.000893
- *First-token latency:* 0.385
- *Cleanup time:* 0.0718
- *Prompt tokens:* 1,112
- *Generation tokens:* 100
- *Total tokens:* 1,212
- *Prompt throughput (raw):* 2,889 tok/s
- *Generation throughput (raw):* 56.8 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00182
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1329
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 12b77fb40b63a2c73c68243d3f767aab688a1b2a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--microsoft--Phi-3.5-vision-instruct/snapshots/12b77fb40b63a2c73c68243d3f767aab688a1b2a
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: "Inflatable Boats Speed Across the Ocean"

Description: On August 7, 2026, two inflatable boats with outboard motors are seen speeding across the ocean in Seaford, England, leaving white wakes behind them under a clear blue sky.

Keywords: inflatable boats, outboard motors, ocean, Seaford, England, UK, Europe, clear blue sky, white wakes, speeding
</pre>

</details>

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### mlx-community/InternVL3-14B-8bit

<details>
<summary>Complete evidence: mlx-community/InternVL3-14B-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type internvl_chat)
- *Model load time:* 1.66s
- *Generation time:* 5.38s
- *Total time:* 7.34s
- *Input validation time:* 0.293
- *Prompt preparation time:* 0.00178
- *First-token latency:* 2.14
- *Cleanup time:* 0.108
- *Prompt tokens:* 2,610
- *Generation tokens:* 87
- *Total tokens:* 2,697
- *Prompt throughput (raw):* 1,222 tok/s
- *Generation throughput (raw):* 32.2 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.61
- *Model-load active memory:* 16.4
- *Post-cleanup active memory:* 0.00469
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1313
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.internvl_chat.processor.InternVLChatProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 50efc568c7dfd1b91569365f1e6eb65e752f4125
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3-14B-8bit/snapshots/50efc568c7dfd1b91569365f1e6eb65e752f4125
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Speeding Boats on the English Channel

Description:
Two inflatable boats with passengers speed across the ocean, leaving white wakes, under a clear blue sky on August 7, 2026.

Keywords:
Adobe Stock, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful
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
- *Model load time:* 1.62s
- *Generation time:* 4.38s
- *Total time:* 6.31s
- *Input validation time:* 0.298
- *Prompt preparation time:* 0.000193
- *First-token latency:* 1.09
- *Cleanup time:* 0.101
- *Prompt tokens:* 2,610
- *Generation tokens:* 93
- *Total tokens:* 2,703
- *Prompt throughput (raw):* 2,388 tok/s
- *Generation throughput (raw):* 33.9 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.21
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.005
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1313
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.internvl_chat.processor.InternVLChatProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3-8B-bf16/snapshots/e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Speeding Boats on Ocean Horizon

Description: Two inflatable boats with passengers speed across the ocean, leaving white wakes under a clear blue sky, captured at 17:17:16 UTC+01:00.

Keywords: Inflatable boat, Motorboat, People, Riding, Ocean, Horizon, Blue sky, Clear weather, Action, Water, Seaford, England, Europe, Holiday, Vehicles, Sailing
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
- *Model load time:* 1.47s
- *Generation time:* 6.15s
- *Total time:* 7.94s
- *Input validation time:* 0.303
- *Prompt preparation time:* 0.00122
- *First-token latency:* 1.46
- *Cleanup time:* 0.0907
- *Prompt tokens:* 297
- *Generation tokens:* 89
- *Total tokens:* 386
- *Prompt throughput (raw):* 203 tok/s
- *Generation throughput (raw):* 21.1 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.59
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.0071
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1412
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-11B-Vision-Instruct-8bit/snapshots/8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|eot_id\|>

*Readable output:*

<pre class="model-output-readable">
Title: Speeding Inflatable Boats on the Open Ocean

Description: Two inflatable boats with outboard motors speed across the ocean on a sunny day, leaving behind white wakes. The clear blue sky and distinct horizon line provide a picturesque backdrop.

Keywords: Ocean, Inflatable Boats, Outboard Motors, Speed, White Wakes, Sunny Day, Clear Blue Sky, Distinct Horizon Line, Speeding, Boating.
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
- *Model load time:* 1.43s
- *Generation time:* 4.06s
- *Total time:* 5.80s
- *Input validation time:* 0.302
- *Prompt preparation time:* 0.00711
- *First-token latency:* 1.81
- *Cleanup time:* 0.0816
- *Prompt tokens:* 2,578
- *Generation tokens:* 121
- *Total tokens:* 2,699
- *Prompt throughput (raw):* 1,426 tok/s
- *Generation throughput (raw):* 67.2 tok/s
- *Peak memory:* 12
- *Active memory:* 8.0
- *Cache memory:* 0.51
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00788
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3746
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 7c992876448fc5426841a85c6eb951b056fe38d2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-mxfp4/snapshots/7c992876448fc5426841a85c6eb951b056fe38d2
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
**Speeding Inflatable Boats Near Seaford, UK**

Description:
Two inflatable boats with outboard motors race across calm waters near Seaford, England, on a clear August evening in 2026, leaving pronounced white wakes under a vivid blue sky and distinct horizon.

Keywords:
Seaford, UK, inflatable boats, outboard motors, racing, ocean, blue sky, horizon, August 2026, clear waters, white wakes, coastal scene, holiday activity, motorboats, people on board, daylight, British coastline
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
- *Model load time:* 1.01s
- *Generation time:* 1.93s
- *Total time:* 3.25s
- *Input validation time:* 0.301
- *Prompt preparation time:* 0.00662
- *First-token latency:* 0.844
- *Cleanup time:* 0.0712
- *Prompt tokens:* 2,577
- *Generation tokens:* 117
- *Total tokens:* 2,694
- *Prompt throughput (raw):* 3,055 tok/s
- *Generation throughput (raw):* 185 tok/s
- *Peak memory:* 6.4
- *Active memory:* 2.8
- *Cache memory:* 0.35
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00841
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3745
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* a962dcb09eee4169c890e544c9eb938f1113fdee
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-3B-Instruct-2512-4bit/snapshots/a962dcb09eee4169c890e544c9eb938f1113fdee
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Two Speeding Motorboats on the Ocean Near Seaford

Description:
Two inflatable motorboats filled with passengers are racing across calm ocean waters under a clear blue sky, leaving pronounced white wakes. The bright sunlight and sharp horizon line enhance the dynamic scene of leisure activity.

Keywords:
inflatable-boat, motorboat, ocean, clear-blue-sky, white-wakes, passengers, Seaford, UK, Europe, coastal-England, holiday-activity, action-packed, water-sport, bright-lighting, outboard-motors
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
- *Model load time:* 0.89s
- *Generation time:* 2.47s
- *Total time:* 3.68s
- *Input validation time:* 0.31
- *Prompt preparation time:* 0.000839
- *First-token latency:* 0.282
- *Cleanup time:* 0.076
- *Prompt tokens:* 1,112
- *Generation tokens:* 100
- *Total tokens:* 1,212
- *Prompt throughput (raw):* 3,936 tok/s
- *Generation throughput (raw):* 56.7 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1329
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: "Inflatable Boats Speed Across the Ocean"

Description: On August 7, 2026, two inflatable boats with outboard motors are seen speeding across the ocean in Seaford, England, leaving white wakes behind them under a clear blue sky.

Keywords: inflatable boats, outboard motors, ocean, Seaford, England, UK, Europe, clear blue sky, white wakes, speeding
</pre>

</details>

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### mlx-community/Qwen3.5-27B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-27B-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 2.14s
- *Generation time:* 76.90s
- *Total time:* 79.46s
- *Input validation time:* 0.395
- *Prompt preparation time:* 0.014
- *First-token latency:* 72.7
- *Cleanup time:* 0.12
- *Prompt tokens:* 16,643
- *Generation tokens:* 104
- *Total tokens:* 16,747
- *Prompt throughput (raw):* 229 tok/s
- *Generation throughput (raw):* 30.5 tok/s
- *Peak memory:* 26
- *Active memory:* 18
- *Cache memory:* 1.9
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0117
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1399
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 45797d2985a12c55e6473686e9ea91b95e959553
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-27B-4bit/snapshots/45797d2985a12c55e6473686e9ea91b95e959553
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Rigid Inflatable Boats Speeding Off Seaford

Description: Two rigid inflatable boats carrying passengers speed across the blue sea off Seaford, England, leaving white wakes under a clear sky on 7 August 2026.

Keywords: Seaford, England, UK, rigid inflatable boat, RIB, motorboat, passengers, blue sea, horizon, clear sky, summer, wake, speed, coastal, watercraft, maritime, August 2026
</pre>

</details>

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### mlx-community/Qwen3.5-27B-mxfp8

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-27B-mxfp8</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 3.15s
- *Generation time:* 91.94s
- *Total time:* 95.44s
- *Input validation time:* 0.318
- *Prompt preparation time:* 0.0156
- *First-token latency:* 83.9
- *Cleanup time:* 0.134
- *Prompt tokens:* 16,643
- *Generation tokens:* 107
- *Total tokens:* 16,750
- *Prompt throughput (raw):* 198 tok/s
- *Generation throughput (raw):* 14.9 tok/s
- *Peak memory:* 38
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.0122
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1399
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 2d6caf2325c24e7dd3074e76a6608e9facaee36f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-27B-mxfp8/snapshots/2d6caf2325c24e7dd3074e76a6608e9facaee36f
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two RIBs Speeding Across Seaford Sea

Description:
Two rigid inflatable boats carrying passengers speed across the blue sea near Seaford, England, on 7 August 2026, leaving white wakes under a clear sky.

Keywords:
Seaford, England, RIB, inflatable boat, motorboat, passengers, sea, blue sky, horizon, wake, summer, 2026, water sports, coastal, UK, Europe, speed, outboard motor
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
- *Model load time:* 2.48s
- *Generation time:* 65.32s
- *Total time:* 68.12s
- *Input validation time:* 0.307
- *Prompt preparation time:* 0.000253
- *First-token latency:* 63.7
- *Cleanup time:* 0.114
- *Prompt tokens:* 16,643
- *Generation tokens:* 94
- *Total tokens:* 16,737
- *Prompt throughput (raw):* 261 tok/s
- *Generation throughput (raw):* 110 tok/s
- *Peak memory:* 26
- *Active memory:* 21
- *Cache memory:* 0.89
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.0127
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1399
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 1e20fd8d42056f870933bf98ca6211024744f7ec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Speeding Inflatable Boats on Blue Sea

Description: Two inflatable motorboats carrying passengers speed across the ocean under a clear blue sky, leaving white wakes behind them. The scene captures a sunny day on the water with a distinct horizon line.

Keywords: inflatable boat, motorboat, ocean, blue sky, horizon, passengers, wake, speed, water, action, sunny, sea, vehicles, people, driving, sailing, holiday, England
</pre>

</details>

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### mlx-community/Qwen3.5-35B-A3B-6bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-35B-A3B-6bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 3.10s
- *Generation time:* 64.28s
- *Total time:* 67.70s
- *Input validation time:* 0.304
- *Prompt preparation time:* 0.00025
- *First-token latency:* 62.6
- *Cleanup time:* 0.133
- *Prompt tokens:* 16,643
- *Generation tokens:* 83
- *Total tokens:* 16,726
- *Prompt throughput (raw):* 266 tok/s
- *Generation throughput (raw):* 91.8 tok/s
- *Peak memory:* 35
- *Active memory:* 30
- *Cache memory:* 0.89
- *Model-load active memory:* 29.1
- *Post-cleanup active memory:* 0.0132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1399
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* b729d115bb2cfea696e390dd6bb898528c66b6e9
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-6bit/snapshots/b729d115bb2cfea696e390dd6bb898528c66b6e9
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Speedboats on Seaford Seafront Waters

Description: Two inflatable motorboats speed across the blue sea near Seaford, England, leaving white wakes under a clear sky.

Keywords: Seaford, England, UK, inflatable boats, motorboats, speedboats, sea, ocean, water, wake, blue sky, horizon, people, passengers, driver, action, summer, holiday
</pre>

</details>

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### mlx-community/Qwen3.5-35B-A3B-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-35B-A3B-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 9.67s
- *Generation time:* 68.45s
- *Total time:* 78.43s
- *Input validation time:* 0.306
- *Prompt preparation time:* 0.000265
- *First-token latency:* 65.9
- *Cleanup time:* 0.208
- *Prompt tokens:* 16,643
- *Generation tokens:* 102
- *Total tokens:* 16,745
- *Prompt throughput (raw):* 253 tok/s
- *Generation throughput (raw):* 65.9 tok/s
- *Peak memory:* 76
- *Active memory:* 70
- *Cache memory:* 0.89
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.0137
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1399
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 731d09ba3597261e84c28881116558364bb8b97c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-bf16/snapshots/731d09ba3597261e84c28881116558364bb8b97c
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Speedboats Racing on Seaford Seafront

Description: Two inflatable motorboats carrying passengers speed across the blue sea under a clear sky, leaving white wakes behind them. The scene captures a sunny day at Seaford, England, with the horizon clearly visible.

Keywords: Seaford, England, UK, inflatable boats, motorboats, speedboats, ocean, sea, blue sky, horizon, white wake, passengers, sunny day, action, vehicles, water, holiday, Europe
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
- *Model load time:* 6.39s
- *Generation time:* 21.47s
- *Total time:* 28.19s
- *Input validation time:* 0.303
- *Prompt preparation time:* 0.0134
- *First-token latency:* 17.9
- *Cleanup time:* 0.225
- *Prompt tokens:* 3,478
- *Generation tokens:* 119
- *Total tokens:* 3,597
- *Prompt throughput (raw):* 194 tok/s
- *Generation throughput (raw):* 42.2 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.41
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.0152
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1386
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.step3p7.processing_step3p7.Step3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Step-3.7-Flash-oQ2e/snapshots/3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two Inflatable Boats Speeding Across the Ocean Near Seaford

Description:
On 7 August 2026 at 17:17 UTC+1, two inflatable motorboats with outboard engines speed across the ocean off Seaford, England, leaving white wakes behind them under a clear blue sky and a distinct horizon line.

Keywords:
Inflatable boat, Motorboat, Seaford, England, UK, Europe, Blue sky, Horizon, Ocean, Water, People, Action, Holiday, Sailing, Driving, Vehicles, Adobe Stock, Any Vision
</pre>

</details>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### mlx-community/gemma-3-27b-it-qat-8bit

<details>
<summary>Complete evidence: mlx-community/gemma-3-27b-it-qat-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 3.49s
- *Generation time:* 9.15s
- *Total time:* 12.97s
- *Input validation time:* 0.326
- *Prompt preparation time:* 0.000206
- *First-token latency:* 1.22
- *Cleanup time:* 0.148
- *Prompt tokens:* 579
- *Generation tokens:* 122
- *Total tokens:* 701
- *Prompt throughput (raw):* 473 tok/s
- *Generation throughput (raw):* 16.5 tok/s
- *Peak memory:* 32
- *Active memory:* 30
- *Cache memory:* 0.44
- *Model-load active memory:* 29.7
- *Post-cleanup active memory:* 0.0166
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1363
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3.processing_gemma3.Gemma3Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c408904bc9a0b7e996173d3cbe1bfe30e04e678d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3-27b-it-qat-8bit/snapshots/c408904bc9a0b7e996173d3cbe1bfe30e04e678d
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Inflatable boats speeding off Seaford seafront, August 2026
Description: Two black inflatable boats with passengers are travelling at speed on the sea off Seaford, England, captured on 7th August 2026 in bright sunlight. White wakes trail behind the vessels against a clear blue sky and a distinct horizon.
Keywords: Seaford, England, UK, inflatable boat, motorboat, sea, ocean, water, horizon, blue sky, passengers, speed, bright sunlight, August 2026, vehicles, travel, leisure, coastal
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
- *Model load time:* 2.37s
- *Generation time:* 1.68s
- *Total time:* 4.38s
- *Input validation time:* 0.307
- *Prompt preparation time:* 0.0215
- *First-token latency:* 0.443
- *Cleanup time:* 0.117
- *Prompt tokens:* 588
- *Generation tokens:* 96
- *Total tokens:* 684
- *Prompt throughput (raw):* 1,328 tok/s
- *Generation throughput (raw):* 127 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.20
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0182
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1362
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Two motorboats speeding across the blue sea
Description: Two inflatable motorboats speed across the blue ocean, leaving white wakes behind them under a clear sky. The scene captures a sense of action against a sharp horizon line near Seaford, England.
Keywords: blue sky, England, Europe, holiday, horizon, inflatable boat, motorboat, people, riding, sailing, Seaford, sky, UK, vehicles, water, action, sea, ocean
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
- *Model load time:* 2.65s
- *Generation time:* 4.77s
- *Total time:* 7.74s
- *Input validation time:* 0.309
- *Prompt preparation time:* 0.000283
- *First-token latency:* 1.07
- *Cleanup time:* 0.132
- *Prompt tokens:* 588
- *Generation tokens:* 83
- *Total tokens:* 671
- *Prompt throughput (raw):* 552 tok/s
- *Generation throughput (raw):* 25.8 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.77
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0193
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1362
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 696d436c404745a59f30e4939a658162b0a9e57f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/696d436c404745a59f30e4939a658162b0a9e57f
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Inflatable boats speeding off the coast of Seaford
Description: Two inflatable motorboats carrying passengers speed across the blue waters of the English Channel near Seaford, England, under a clear sky.
Keywords: Seaford, England, UK, inflatable boat, motorboat, ocean, water, wake, speed, passengers, horizon, blue sky, sailing, maritime, Europe, action
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
- *Model load time:* 1.73s
- *Generation time:* 4.24s
- *Total time:* 6.30s
- *Input validation time:* 0.317
- *Prompt preparation time:* 0.00853
- *First-token latency:* 1.71
- *Cleanup time:* 0.0973
- *Prompt tokens:* 2,649
- *Generation tokens:* 82
- *Total tokens:* 2,731
- *Prompt throughput (raw):* 1,546 tok/s
- *Generation throughput (raw):* 39.6 tok/s
- *Peak memory:* 15
- *Active memory:* 14
- *Cache memory:* 0.50
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1308
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--pixtral-12b-8bit/snapshots/79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Two Inflatable Boats Speeding Across the Sea

Description:
Two inflatable boats with outboard motors speed across the sea, leaving white wakes behind them under a clear blue sky with a distinct horizon line.

Keywords:
Inflatable boat, Motorboat, Sea, Blue sky, Horizon, Water, Speeding, Outboard motor, People, White wake, Clear sky, Distinct horizon
</pre>

</details>

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### mlx-community/pixtral-12b-bf16

<details>
<summary>Complete evidence: mlx-community/pixtral-12b-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 2.51s
- *Generation time:* 6.27s
- *Total time:* 9.09s
- *Input validation time:* 0.304
- *Prompt preparation time:* 0.000211
- *First-token latency:* 1.44
- *Cleanup time:* 0.115
- *Prompt tokens:* 2,649
- *Generation tokens:* 88
- *Total tokens:* 2,737
- *Prompt throughput (raw):* 1,843 tok/s
- *Generation throughput (raw):* 20.1 tok/s
- *Peak memory:* 27
- *Active memory:* 25
- *Cache memory:* 0.50
- *Model-load active memory:* 25.4
- *Post-cleanup active memory:* 0.0246
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1308
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 378cf65efd79647fc324a079c141fd92cb29916e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--pixtral-12b-bf16/snapshots/378cf65efd79647fc324a079c141fd92cb29916e
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Two Inflatable Boats Speeding Across the Sea

Description:
Two inflatable boats with outboard motors speed across the sea, leaving white wakes behind them under a clear blue sky with a distinct horizon line.

Keywords:
Inflatable boat, Motorboat, Sea, Blue sky, Horizon, Water, Speeding, Outboard motor, People, Riding, Vehicles, Clear sky, Distinct horizon, White wake
</pre>

</details>

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### HuggingFaceTB/SmolVLM-Instruct

<details>
<summary>Complete evidence: HuggingFaceTB/SmolVLM-Instruct</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 6 words (requested 5-10); Keyword list has 20
  terms (requested 10-18); Title, Description and Keywords copy all supplied
  hints unchanged
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 0.71s
- *Generation time:* 1.76s
- *Total time:* 2.76s
- *Input validation time:* 0.286
- *Prompt preparation time:* 0.0018
- *First-token latency:* 0.388
- *Cleanup time:* 0.0642
- *Prompt tokens:* 1,510
- *Generation tokens:* 104
- *Total tokens:* 1,614
- *Prompt throughput (raw):* 3,895 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.40
- *Model-load active memory:* 4.49
- *Post-cleanup active memory:* 0.000116
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1340
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 81cd9a775a4d644f2faf4e7becff4559b46b14c7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-Instruct/snapshots/81cd9a775a4d644f2faf4e7becff4559b46b14c7
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <end_of_utterance>

*Readable output:*

<pre class="model-output-readable">
 Title: Seafront, Seaford, England, UK, GBR, Europe.
Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.
Keywords: adobe stock, any vision, blue sky, driver, england, europe, holiday, horizon, inflatable boat, motorboat, people, riding, sailing, seaford, sky, uk, vehicles, water, action, beautiful.
</pre>

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

<details>
<summary>Complete evidence: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 6 words (requested 5-10); Keyword list has 18
  terms (requested 10-18); Duplicate keywords: seaford, uk
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.33s
- *Generation time:* 0.58s
- *Total time:* 1.20s
- *Input validation time:* 0.28
- *Prompt preparation time:* 0.00547
- *First-token latency:* 0.0251
- *Cleanup time:* 0.0536
- *Prompt tokens:* 385
- *Generation tokens:* 107
- *Total tokens:* 492
- *Prompt throughput (raw):* 15,357 tok/s
- *Generation throughput (raw):* 515 tok/s
- *Peak memory:* 1.3
- *Active memory:* 0.90
- *Cache memory:* 0.02
- *Model-load active memory:* 0.898
- *Post-cleanup active memory:* 0.000263
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1359
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 6c33f49ebc0b50b75385f49ad3beddcb720d0c75
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-450M-MLX-bf16/snapshots/6c33f49ebc0b50b75385f49ad3beddcb720d0c75
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Seafront, Seaford, England, UK, GBR, Europe

Description:
Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.

Keywords:
Seafront, Seaford, England, UK, GBR, Europe, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, Action, Beautiful
</pre>

</details>

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### Qwen/Qwen3-VL-2B-Instruct

<details>
<summary>Complete evidence: Qwen/Qwen3-VL-2B-Instruct</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 5 words (requested 5-10); Keyword list has 18
  terms (requested 10-18); Duplicate keywords: action, vehicles
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.66s
- *Generation time:* 15.35s
- *Total time:* 16.30s
- *Input validation time:* 0.28
- *Prompt preparation time:* 0.00893
- *First-token latency:* 13.7
- *Cleanup time:* 0.0949
- *Prompt tokens:* 16,628
- *Generation tokens:* 86
- *Total tokens:* 16,714
- *Prompt throughput (raw):* 1,216 tok/s
- *Generation throughput (raw):* 91.4 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000575
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1380
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two boats on the sea

Description: Two inflatable motorboats speed across the ocean, leaving white wakes, under a clear blue sky with a distinct horizon line, in Seaford, England, UK.

Keywords: inflatable boat, motorboat, sea, ocean, people, sky, horizon, action, water, England, UK, Seaford, vehicles, holiday, blue sky, vehicles, action, beautiful
</pre>

</details>

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

<details>
<summary>Complete evidence: mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 7 words (requested 5-10); Keyword list has 24
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.41s
- *Generation time:* 7.04s
- *Total time:* 9.79s
- *Input validation time:* 0.321
- *Prompt preparation time:* 0.00679
- *First-token latency:* 2.88
- *Cleanup time:* 0.0994
- *Prompt tokens:* 2,045
- *Generation tokens:* 110
- *Total tokens:* 2,155
- *Prompt throughput (raw):* 711 tok/s
- *Generation throughput (raw):* 29.8 tok/s
- *Peak memory:* 22
- *Active memory:* 18
- *Cache memory:* 0.42
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.00234
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1308
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Devstral-Small-2-24B-Instruct-2512-5bit/snapshots/0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Two Inflatable Boats Racing Across the Ocean

Description:
Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, under a clear blue sky with a distinct horizon line.

Keywords:
Adobe Stock, inflatable boats, outboard motors, ocean, blue sky, horizon, speeding, white wakes, clear sky, racing, water, vehicles, action, holiday, sailing, people, riding, driver, beautiful, seafront, Seaford, England, UK, Europe
</pre>

</details>

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### mlx-community/GLM-4.6V-Flash-mxfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-mxfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 3 words (requested 5-10); Keyword list has 13
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.30s
- *Generation time:* 9.59s
- *Total time:* 11.19s
- *Input validation time:* 0.297
- *Prompt preparation time:* 0.000238
- *First-token latency:* 8.04
- *Cleanup time:* 0.0774
- *Prompt tokens:* 6,330
- *Generation tokens:* 83
- *Total tokens:* 6,413
- *Prompt throughput (raw):* 787 tok/s
- *Generation throughput (raw):* 74.1 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.46
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.0038
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1389
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-Flash-mxfp4/snapshots/773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Inflatable Boats
Description: Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, under a clear blue sky with a distinct horizon line.
Keywords: Two Inflatable Boats, outboard motors, ocean, white wakes, clear blue sky, distinct horizon line, Seaford, England, UK, Europe, holiday, action, beautiful.
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
- *Model load time:* 1.91s
- *Generation time:* 5.50s
- *Total time:* 7.74s
- *Input validation time:* 0.324
- *Prompt preparation time:* 0.00136
- *First-token latency:* 1.14
- *Cleanup time:* 0.0979
- *Prompt tokens:* 2,608
- *Generation tokens:* 121
- *Total tokens:* 2,729
- *Prompt throughput (raw):* 2,280 tok/s
- *Generation throughput (raw):* 32.1 tok/s
- *Peak memory:* 19
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.00438
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1345
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Idefics3-8B-Llama3-bf16/snapshots/8c2a30c48864f3251701b7bde40f601d25535098
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Inflatable Boats Speed Across the Ocean on a Clear Day.

Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. The boats are carrying several people, and the water is calm with no visible waves. The sky is clear with no clouds, and the sun is not visible, but the lighting suggests it is daytime.

Keywords: inflatable boats, ocean, outboard motors, white wakes, clear blue sky, horizon line, people, calm water, clear sky, daytime.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Title: Inflatable Boats Speed Across the Ocean on a Clear Day.

Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. The boats are carrying several people, and the water is calm with no visible waves. The sky is clear with no clouds, and the sun is not visible, but the lighting suggests it is daytime.

Keywords: inflatable boats, ocean, outboard motors, white wakes, clear blue sky, horizon line, people, calm water, clear sky, daytime.<end_of_utterance>
```

</details>

</details>

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### mlx-community/LFM2-VL-1.6B-8bit

<details>
<summary>Complete evidence: mlx-community/LFM2-VL-1.6B-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 2 words (requested 5-10); Keyword list has 14
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.49s
- *Generation time:* 0.70s
- *Total time:* 1.50s
- *Input validation time:* 0.303
- *Prompt preparation time:* 0.00114
- *First-token latency:* 0.0921
- *Cleanup time:* 0.0678
- *Prompt tokens:* 571
- *Generation tokens:* 80
- *Total tokens:* 651
- *Prompt throughput (raw):* 6,198 tok/s
- *Generation throughput (raw):* 331 tok/s
- *Peak memory:* 3.0
- *Active memory:* 2.1
- *Cache memory:* 0.03
- *Model-load active memory:* 2.06
- *Post-cleanup active memory:* 0.00644
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1359
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 294b90e5ae2389ecb61a9427b4572975eef614fe
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2-VL-1.6B-8bit/snapshots/294b90e5ae2389ecb61a9427b4572975eef614fe
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Seafront Adventure
Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.
Keywords: Seaford, England, UK, GBR, Europe, Inflatable boat, Motorboat, People, Horizon, Sky, Vehicles, Water, Action, Beautiful
</pre>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 7 words (requested 5-10); Keyword list has 20
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.40s
- *Generation time:* 5.11s
- *Total time:* 6.82s
- *Input validation time:* 0.299
- *Prompt preparation time:* 0.000229
- *First-token latency:* 1.88
- *Cleanup time:* 0.0846
- *Prompt tokens:* 2,578
- *Generation tokens:* 178
- *Total tokens:* 2,756
- *Prompt throughput (raw):* 1,370 tok/s
- *Generation throughput (raw):* 64.0 tok/s
- *Peak memory:* 12
- *Active memory:* 8.4
- *Cache memory:* 0.51
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00814
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3746
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 28777b889d841a86369c736175cb77258c8134b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-nvfp4/snapshots/28777b889d841a86369c736175cb77258c8134b2
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
**Title:**
*Inflatable Boats Racing Near Seaford, UK (2026-08-07)*

**Description:**
On **2026-08-07 at 17:17 UTC+01:00**, two inflatable motorboats speed across calm waters near **Seaford, England (UK)**, leaving pronounced white wakes. Against a clear blue sky and distinct horizon, passengers appear engaged, showcasing dynamic movement and bright daylight conditions.

**Keywords:**
inflatable motorboats, Seaford, England, UK, Europe, 2026-08-07, racing, ocean, white wakes, clear blue sky, horizon, passengers, bright daylight, outboard motors, coastal waters, dynamic movement, summer scene, horizon line, coastal holiday, motorboat action
</pre>

</details>

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### mlx-community/Molmo-7B-D-0924-8bit

<details>
<summary>Complete evidence: mlx-community/Molmo-7B-D-0924-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 5 words (requested 5-10); Keyword list has 19
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.23s
- *Generation time:* 3.50s
- *Total time:* 5.03s
- *Input validation time:* 0.301
- *Prompt preparation time:* 0.00134
- *First-token latency:* 0.67
- *Cleanup time:* 0.0744
- *Prompt tokens:* 1,498
- *Generation tokens:* 105
- *Total tokens:* 1,603
- *Prompt throughput (raw):* 2,236 tok/s
- *Generation throughput (raw):* 53.3 tok/s
- *Peak memory:* 11
- *Active memory:* 9.0
- *Cache memory:* 0.14
- *Model-load active memory:* 9.05
- *Post-cleanup active memory:* 0.00872
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1304
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Molmo-7B-D-0924-8bit/snapshots/90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Speedboats Raging on Seaford's Shores

Description: Two inflatable motorboats race across the ocean, leaving white wakes behind them against a clear blue sky and distinct horizon, with passengers enjoying the thrilling ride on a sunny day in Seaford, England.

Keywords: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Seaford, Sky, UK, Vehicles, Water, action, beautiful
</pre>

</details>

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### mlx-community/Molmo-7B-D-0924-bf16

<details>
<summary>Complete evidence: mlx-community/Molmo-7B-D-0924-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 5 words (requested 5-10); Keyword list has 19
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.68s
- *Generation time:* 4.97s
- *Total time:* 6.98s
- *Input validation time:* 0.317
- *Prompt preparation time:* 0.00123
- *First-token latency:* 0.638
- *Cleanup time:* 0.0999
- *Prompt tokens:* 1,498
- *Generation tokens:* 108
- *Total tokens:* 1,606
- *Prompt throughput (raw):* 2,349 tok/s
- *Generation throughput (raw):* 30.8 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.14
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.00903
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1304
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* d871cbdb87a49b8071003098d6dbfd2a0f5a5b84
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Molmo-7B-D-0924-bf16/snapshots/d871cbdb87a49b8071003098d6dbfd2a0f5a5b84
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Speedboats Raging on Seaford's Shores

Description: Two inflatable motorboats race across the calm ocean, leaving white wakes in their path, against a clear blue sky and horizon. Passengers enjoy the thrilling ride, with the sun casting a warm glow on the scene.

Keywords: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Seaford, Sky, UK, Vehicles, Water, action, beautiful
</pre>

</details>

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### mlx-community/Ornith-1.0-35B-bf16

<details>
<summary>Complete evidence: mlx-community/Ornith-1.0-35B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 8 words (requested 5-10); Keyword list has 19
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 9.27s
- *Generation time:* 64.52s
- *Total time:* 74.12s
- *Input validation time:* 0.303
- *Prompt preparation time:* 0.0136
- *First-token latency:* 61.6
- *Cleanup time:* 0.223
- *Prompt tokens:* 16,643
- *Generation tokens:* 107
- *Total tokens:* 16,750
- *Prompt throughput (raw):* 270 tok/s
- *Generation throughput (raw):* 63.8 tok/s
- *Peak memory:* 76
- *Active memory:* 70
- *Cache memory:* 0.89
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1399
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-bf16/snapshots/9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two inflatable motorboats speeding across calm blue sea

Description: Two rigid inflatable boats carrying passengers cruise across the open sea under a clear sky, leaving white wakes behind them. The scene captures a bright, sunny day on the water with a distinct horizon line separating the deep blue sea from the pale sky.

Keywords: inflatable boat, motorboat, sea, ocean, people, passengers, wake, speed, blue sky, horizon, sunny, vessels, water, leisure, boating, England, UK, Europe, adventure
</pre>

</details>

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 6 words (requested 5-10); Keyword list has 20
  terms (requested 10-18); Title, Description and Keywords copy all supplied
  hints unchanged
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.66s
- *Generation time:* 70.15s
- *Total time:* 71.13s
- *Input validation time:* 0.306
- *Prompt preparation time:* 0.0033
- *First-token latency:* 68.9
- *Cleanup time:* 0.0777
- *Prompt tokens:* 16,639
- *Generation tokens:* 102
- *Total tokens:* 16,741
- *Prompt throughput (raw):* 242 tok/s
- *Generation throughput (raw):* 212 tok/s
- *Peak memory:* 5.1
- *Active memory:* 1.3
- *Cache memory:* 0.87
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.0105
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1438
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 01af461cdb9574acc09084a0ef94e216e142b085
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen2-VL-2B-Instruct-4bit/snapshots/01af461cdb9574acc09084a0ef94e216e142b085
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Seafront, Seaford, England, UK, GBR, Europe

Description:
Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.

Keywords:
Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### mlx-community/Qwen3-VL-2B-Instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 5 words (requested 5-10); Keyword list has 18
  terms (requested 10-18); Duplicate keywords: action, vehicles
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.77s
- *Generation time:* 20.98s
- *Total time:* 22.06s
- *Input validation time:* 0.306
- *Prompt preparation time:* 0.000221
- *First-token latency:* 19.2
- *Cleanup time:* 0.082
- *Prompt tokens:* 16,628
- *Generation tokens:* 86
- *Total tokens:* 16,714
- *Prompt throughput (raw):* 865 tok/s
- *Generation throughput (raw):* 91.1 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.27
- *Post-cleanup active memory:* 0.0108
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1380
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Instruct-bf16/snapshots/c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two boats on the sea

Description: Two inflatable motorboats speed across the ocean, leaving white wakes, under a clear blue sky with a distinct horizon line, in Seaford, England, UK.

Keywords: inflatable boat, motorboat, sea, ocean, people, sky, horizon, action, water, England, UK, Seaford, vehicles, holiday, blue sky, vehicles, action, beautiful
</pre>

</details>

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### mlx-community/Qwen3.5-9B-MLX-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 10 words (requested 5-10); Keyword list has 20
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 1.37s
- *Generation time:* 64.66s
- *Total time:* 66.35s
- *Input validation time:* 0.312
- *Prompt preparation time:* 0.000261
- *First-token latency:* 62.6
- *Cleanup time:* 0.0864
- *Prompt tokens:* 16,643
- *Generation tokens:* 112
- *Total tokens:* 16,755
- *Prompt throughput (raw):* 266 tok/s
- *Generation throughput (raw):* 89.9 tok/s
- *Peak memory:* 11
- *Active memory:* 6.7
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.0142
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1399
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two inflatable motorboats speeding across the sea at Seaford, UK.

Description:
Two inflatable motorboats with outboard motors speed across calm blue waters under a clear sky, leaving white wakes behind them, with people on board enjoying the ride near the horizon line.

Keywords:
Inflatable boat, motorboat, outboard motor, people, sea, ocean, wake, horizon, blue sky, clear sky, Seaford, UK, England, Europe, holiday, action, riding, vehicles, water, beautiful
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:  
Two inflatable motorboats speeding across the sea at Seaford, UK.

Description:  
Two inflatable motorboats with outboard motors speed across calm blue waters under a clear sky, leaving white wakes behind them, with people on board enjoying the ride near the horizon line.

Keywords:  
Inflatable boat, motorboat, outboard motor, people, sea, ocean, wake, horizon, blue sky, clear sky, Seaford, UK, England, Europe, holiday, action, riding, vehicles, water, beautiful
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
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 6 words (requested 5-10); Keyword list has 19
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 3.07s
- *Generation time:* 92.78s
- *Total time:* 96.18s
- *Input validation time:* 0.303
- *Prompt preparation time:* 0.0139
- *First-token latency:* 86.5
- *Cleanup time:* 0.133
- *Prompt tokens:* 16,643
- *Generation tokens:* 87
- *Total tokens:* 16,730
- *Prompt throughput (raw):* 192 tok/s
- *Generation throughput (raw):* 15.9 tok/s
- *Peak memory:* 38
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.0147
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1399
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-mxfp8/snapshots/5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Speedboats Racing Across the Ocean

Description: Two motorised inflatable boats carrying passengers speed across the blue sea, leaving white wakes behind them under a clear sky.

Keywords: Sea, Ocean, Boats, Motorboats, Inflatable boats, People, Passengers, Speeding, Wakes, Blue water, Clear sky, Horizon, Summer, Daytime, England, UK, Europe, Action, Water sports
</pre>

</details>

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### mlx-community/SmolVLM-Instruct-bf16

<details>
<summary>Complete evidence: mlx-community/SmolVLM-Instruct-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 6 words (requested 5-10); Keyword list has 20
  terms (requested 10-18); Title, Description and Keywords copy all supplied
  hints unchanged
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 0.64s
- *Generation time:* 1.86s
- *Total time:* 2.81s
- *Input validation time:* 0.305
- *Prompt preparation time:* 0.00119
- *First-token latency:* 0.415
- *Cleanup time:* 0.0762
- *Prompt tokens:* 1,510
- *Generation tokens:* 104
- *Total tokens:* 1,614
- *Prompt throughput (raw):* 3,636 tok/s
- *Generation throughput (raw):* 124 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.40
- *Model-load active memory:* 4.51
- *Post-cleanup active memory:* 0.0148
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1340
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* cae61cdedd0602419b43b6102dc33cd9f1e929a6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--SmolVLM-Instruct-bf16/snapshots/cae61cdedd0602419b43b6102dc33cd9f1e929a6
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <end_of_utterance>

*Readable output:*

<pre class="model-output-readable">
 Title: Seafront, Seaford, England, UK, GBR, Europe.
Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.
Keywords: adobe stock, any vision, blue sky, driver, england, europe, holiday, horizon, inflatable boat, motorboat, people, riding, sailing, seaford, sky, uk, vehicles, water, action, beautiful.
</pre>

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

<details>
<summary>Complete evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 2 words (requested 5-10); Keyword list has 20
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.61s
- *Generation time:* 1.89s
- *Total time:* 2.81s
- *Input validation time:* 0.305
- *Prompt preparation time:* 0.00095
- *First-token latency:* 0.326
- *Cleanup time:* 0.0736
- *Prompt tokens:* 411
- *Generation tokens:* 125
- *Total tokens:* 536
- *Prompt throughput (raw):* 1,262 tok/s
- *Generation throughput (raw):* 130 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.19
- *Model-load active memory:* 4.51
- *Post-cleanup active memory:* 0.015
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1340
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--SmolVLM2-2.2B-Instruct-mlx/snapshots/844516024a1c4400d34489b89ee067d794e432ed
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <end_of_utterance>

*Readable output:*

<pre class="model-output-readable">
 Title: "Seafront Adventure"
Description: "Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. The boats are being ridden by people enjoying a holiday in England, Europe, with a beautiful view of the horizon and clear blue sky."
Keywords: "Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful"
</pre>

</details>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### mlx-community/gemma-3-27b-it-qat-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3-27b-it-qat-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 8 words (requested 5-10); Keyword list has 24
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 2.32s
- *Generation time:* 6.79s
- *Total time:* 9.46s
- *Input validation time:* 0.336
- *Prompt preparation time:* 0.0031
- *First-token latency:* 1.03
- *Cleanup time:* 0.14
- *Prompt tokens:* 579
- *Generation tokens:* 155
- *Total tokens:* 734
- *Prompt throughput (raw):* 562 tok/s
- *Generation throughput (raw):* 29.5 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.44
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0161
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1363
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3.processing_gemma3.Gemma3Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3-27b-it-qat-4bit/snapshots/fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Seaford Ribs Speeding Across the Channel, August 2026

Description: Two black rigid inflatable boats (RIBs) with passengers are travelling at speed on the sea off Seaford, England, captured on 7th August 2026 at 17:17 UTC+01:00 under clear, bright conditions. Distinct white wakes trail behind the vessels against a flat, calm horizon.

Keywords: Seaford, England, UK, Channel, Rib, Inflatable boat, Motorboat, Speedboat, Water, Sea, Ocean, August, 2026, Passengers, Vehicles, Action, Horizon, Blue sky, Daytime, Summer, Transport, Recreation, Coastal, Rigid Hull
</pre>

</details>

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### mlx-community/gemma-3n-E4B-it-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E4B-it-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Title has 7 words (requested 5-10); Keyword list has 19
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 2.31s
- *Generation time:* 3.54s
- *Total time:* 6.17s
- *Input validation time:* 0.305
- *Prompt preparation time:* 0.00265
- *First-token latency:* 0.393
- *Cleanup time:* 0.12
- *Prompt tokens:* 578
- *Generation tokens:* 130
- *Total tokens:* 708
- *Prompt throughput (raw):* 1,470 tok/s
- *Generation throughput (raw):* 47.9 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0177
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1365
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3n-E4B-it-bf16/snapshots/d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
**Title:** Two Inflatable Boats Speed Across the Ocean

**Description:** Two blue inflatable boats, each with an outboard motor, are speeding across a vast, choppy blue ocean. White wakes trail behind them, contrasting against the deep blue water. The sky is a clear, pale blue, meeting the horizon line in the distance. Several people are visible in each boat, appearing to be enjoying a fast ride.

**Keywords:** Inflatable boat, motorboat, ocean, water, speed, blue, sky, people, riding, action, waves, sea, travel, adventure, summer, coastal, UK, England, Seaford.
</pre>

</details>

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### jqlive/Kimi-VL-A3B-Thinking-2506-6bit

<details>
<summary>Complete evidence: jqlive/Kimi-VL-A3B-Thinking-2506-6bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Missing or empty fields:
  Title; Extra text appears before the Title field; Response appears cut off
  at the token limit; Internal reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 1.98s
- *Generation time:* 8.78s
- *Total time:* 11.10s
- *Input validation time:* 0.33
- *Prompt preparation time:* 0.00189
- *First-token latency:* 1.4
- *Cleanup time:* 0.089
- *Prompt tokens:* 1,313
- *Generation tokens:* 500
- *Total tokens:* 1,813
- *Prompt throughput (raw):* 941 tok/s
- *Generation throughput (raw):* 73.3 tok/s
- *Peak memory:* 18
- *Active memory:* 14
- *Cache memory:* 1.4
- *Model-load active memory:* 13.8
- *Post-cleanup active memory:* 0.00123
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1495
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 4b3b11ce0874c36a99e13e17e355049042f8620a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--jqlive--Kimi-VL-A3B-Thinking-2506-6bit/snapshots/4b3b11ce0874c36a99e13e17e355049042f8620a
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* [EOS]

*Readable output:*

<pre class="model-output-readable">
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and context provided. Let's start by analyzing the image and the hints.

The context says the capture date/time is 2026-08-07 17:17:16 UTC+01:00, but we don't need to claim it's visible. The descriptive hints include "Seafront, Seaford, England, UK, GBR, Europe" so location is Seaford, UK. The image shows two inflatable boats with outboard motors speeding across the ocean, leaving white wakes, clear blue sky, distinct horizon. Keywords: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful.

First, the title: needs to be 5-10 words. Let's see, "Two inflatable motorboats speed across Seaford's sea with people aboard, clear blue sky and horizon." That's 10 words.

Description: Combine context and main elements. "Two inflatable motorboats with outboard motors speed across the ocean in Seaford, UK, under a clear blue sky, creating white wakes, with people aboard, against a distinct horizon line." That's a factual description.

Keywords: Need 10-18 unique keywords. Let's list them: Seaford, UK, England, Europe, Blue sky, Horizon, Inflatable boat, Motorboat, People, Water, Action, Holiday, Sky, Vehicles, Adobe Stock, Any Vision, Driving, Speed, Waves, Ocean, Clear sky, Distinct horizon. Wait, check for uniqueness and relevance. Let's make sure they are comma-separated and unique. Let's adjust: Seaford, UK, England, Europe, Blue sky, Horizon, Inflatable boat, Motorboat, People, Water, Action, Sky, Vehicles, Adobe Stock, Any Vision, Driving, Speed, Waves, Ocean, Clear sky, Distinct horizon. That's 18 keywords. Wait, maybe "Driving" instead of "Driving" but "Speed" is better. Let's check: Seaford, UK, England, Europe, Blue sky, Horizon, Inflatable boat, Motorboat, People, Water, Action, Sky, Vehicles, Adobe Stock, Any Vision, Speed, Waves, Ocean, Clear
</pre>

</details>

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

<details>
<summary>Complete evidence: mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response repeats the task instructions instead of only returning the
  requested fields; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.62s
- *Generation time:* 14.09s
- *Total time:* 16.03s
- *Input validation time:* 0.307
- *Prompt preparation time:* 0.00745
- *First-token latency:* 1.9
- *Cleanup time:* 0.093
- *Prompt tokens:* 2,740
- *Generation tokens:* 500
- *Total tokens:* 3,240
- *Prompt throughput (raw):* 1,442 tok/s
- *Generation throughput (raw):* 42.6 tok/s
- *Peak memory:* 14
- *Active memory:* 13
- *Cache memory:* 0.67
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.00208
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1682
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Apriel-1.5-15b-Thinker-6bit-MLX/snapshots/24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Here are my reasoning steps:
We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time: 2026-08-07 17:17:16 UTC+01:00. The descriptive hints: Title hint: Seafront, Seaford, England, UK, GBR, Europe. Description hint: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. Keyword hints: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful.

We need to produce:

- a concrete 5-10-word title (so 5-10 words). Must be concrete, not hint. Should reflect the image: maybe "Inflatable boats racing on Seaford's horizon". That's 7 words. Or "Two speedboats carving wakes on Seaford sea". That's 9 words. Must be British English.

- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details. Must be 1-2 sentences. Should mention capture date/time? The instruction: "Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible." So we can incorporate the date/time in description if relevant. But description should combine relevant context with main visible subject, setting, action, lighting, distinctive details. So we can say: "On 7 August 2026, two inflatable motorboats speed across the calm blue sea off Seaford, leaving white wakes behind them under a clear sky and a sharp horizon." That's one sentence. Or two sentences: "Two inflatable motorboats race across the sea off Seaford on 7 August 2026, leaving white wakes. The scene is set against a clear blue sky and a distinct horizon." That's two sentences.

- 10-18 unique, comma-separated keywords covering relevant context and visible details. Must be 10-18 unique keywords. Use British English. Should include: Seaford
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
  Title, Description, Keywords; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 7.21s
- *Generation time:* 13.40s
- *Total time:* 20.93s
- *Input validation time:* 0.313
- *Prompt preparation time:* 0.00739
- *First-token latency:* 4
- *Cleanup time:* 0.216
- *Prompt tokens:* 1,590
- *Generation tokens:* 500
- *Total tokens:* 2,090
- *Prompt throughput (raw):* 397 tok/s
- *Generation throughput (raw):* 56.5 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.14
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00256
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1498
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--ERNIE-4.5-VL-28B-A3B-Thinking-bf16/snapshots/32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Alright, let's get to work. I've got the image and the hints, and the task is to create some metadata. First, the title. "Two inflatable boats speeding across the sea" is a good start, but I need something more evocative. "Boats on the Seafront at Seaford" is a bit bland. "Speedboats on the Seafront at Seaford" is better, but I think I'll go with "Two inflatable boats speeding across the seafront at Seaford." That's descriptive and accurate.

Now, the description. I need to weave in the context and the hints. The date and time are important, so I'll start with that. "Captured on August 7th, 2026, at 17:17 UTC+01:00, this image shows two inflatable motorboats speeding across the seafront at Seaford, England." Then, I'll add the details from the hints: "Against a clear blue sky, the boats leave white wakes as they race across the water, with people visible on board." That's a solid, factual description.

Finally, the keywords. I need to cover everything. I'll start with the obvious: "boats, water, sea, seafront, Seaford, England, UK, Europe, inflatable boat, motorboat, speeding, wake, white wake, blue sky, horizon, people, action, summer, holiday, outboard motor, racing, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat
</pre>

</details>

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### mlx-community/FastVLM-0.5B-bf16

<details>
<summary>Complete evidence: mlx-community/FastVLM-0.5B-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description; Extra text
  appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type llava_qwen2 via
  fastvlm)
- *Model load time:* 0.69s
- *Generation time:* 1.54s
- *Total time:* 2.55s
- *Input validation time:* 0.309
- *Prompt preparation time:* 0.00161
- *First-token latency:* 0.097
- *Cleanup time:* 0.0741
- *Prompt tokens:* 319
- *Generation tokens:* 130
- *Total tokens:* 449
- *Prompt throughput (raw):* 3,288 tok/s
- *Generation throughput (raw):* 353 tok/s
- *Peak memory:* 2.2
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00287
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1403
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--FastVLM-0.5B-bf16/snapshots/81ffe929046666c43de53691147b1669ba0f3a4c
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
A concrete 5-10-word title: "Two inflatable boats speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line."
A 1-2-sentence factual description: "Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line."
Keywords: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful.
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
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response appears cut off at the token limit; Internal reasoning block
  appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.73s
- *Generation time:* 20.06s
- *Total time:* 22.10s
- *Input validation time:* 0.299
- *Prompt preparation time:* 0.00163
- *First-token latency:* 7.11
- *Cleanup time:* 0.156
- *Prompt tokens:* 6,326
- *Generation tokens:* 500
- *Total tokens:* 6,826
- *Prompt throughput (raw):* 890 tok/s
- *Generation throughput (raw):* 40.0 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.41
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.00318
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1365
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.1V-9B-Thinking-8bit/snapshots/9677807f106500eb7690391c27645d59f6855cfb
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image: two inflatable boats, sea, action. So maybe "Inflatable Boats Speeding at Seafront". Wait, need to make it concise. Let's check the hints. The title hint was Seafront, Seaford, England... but the image shows two inflatable boats. So maybe "Speeding Inflatable Boats at Seafront". Wait, 5-10 words. Let's go with "Inflatable Boats Racing at Seaford Seafront" maybe, but need to be concise. Alternatively, "Two Inflatable Boats Speeding on the Sea". Hmm, let's pick a title that's clear. Maybe "Speeding Inflatable Boats at Sea".

Next, the description: need 1-2 sentences. Combine context (capture date, setting) with main subject, setting, action, lighting, distinctive details. The image has two inflatable boats, speeding, white wakes, clear blue sky, distinct horizon. So: "Two inflatable boats speed across the seafront waters of Seaford, England, leaving white wakes behind them under a clear blue sky with a distinct horizon line, captured on 2026-08-07." Wait, need to make it factual. Let's structure: "Two inflatable motorboats speed across the sea, creating white wakes, under a clear blue sky with a distinct horizon, photographed on 2026-08-07." That's two sentences? Wait, 1-2 sentences. So maybe: "Two inflatable motorboats speed across the sea, leaving white wakes, under a clear blue sky with a distinct horizon, captured on 2026-08-07." That's one sentence. Wait, need to include relevant context (date) and main visible subject (inflatable boats), setting (sea, clear sky, horizon), action (speeding, leaving wakes), lighting (clear blue sky), distinctive details (two boats, people on board). So: "Two inflatable motorboats speed across the sea, creating white wakes, under a clear blue sky with a distinct horizon, photographed on 2026-08-07." That works.

Then keywords: need 10-18 unique, comma-separated. From hints and image: Adobe Stock (maybe not, but hints include Adobe Stock), Any Vision (maybe
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image: two inflatable boats, sea, action. So maybe "Inflatable Boats Speeding at Seafront". Wait, need to make it concise. Let's check the hints. The title hint was Seafront, Seaford, England... but the image shows two inflatable boats. So maybe "Speeding Inflatable Boats at Seafront". Wait, 5-10 words. Let's go with "Inflatable Boats Racing at Seaford Seafront" maybe, but need to be concise. Alternatively, "Two Inflatable Boats Speeding on the Sea". Hmm, let's pick a title that's clear. Maybe "Speeding Inflatable Boats at Sea".

Next, the description: need 1-2 sentences. Combine context (capture date, setting) with main subject, setting, action, lighting, distinctive details. The image has two inflatable boats, speeding, white wakes, clear blue sky, distinct horizon. So: "Two inflatable boats speed across the seafront waters of Seaford, England, leaving white wakes behind them under a clear blue sky with a distinct horizon line, captured on 2026-08-07." Wait, need to make it factual. Let's structure: "Two inflatable motorboats speed across the sea, creating white wakes, under a clear blue sky with a distinct horizon, photographed on 2026-08-07." That's two sentences? Wait, 1-2 sentences. So maybe: "Two inflatable motorboats speed across the sea, leaving white wakes, under a clear blue sky with a distinct horizon, captured on 2026-08-07." That's one sentence. Wait, need to include relevant context (date) and main visible subject (inflatable boats), setting (sea, clear sky, horizon), action (speeding, leaving wakes), lighting (clear blue sky), distinctive details (two boats, people on board). So: "Two inflatable motorboats speed across the sea, creating white wakes, under a clear blue sky with a distinct horizon, photographed on 2026-08-07." That works.

Then keywords: need 10-18 unique, comma-separated. From hints and image: Adobe Stock (maybe not, but hints include Adobe Stock), Any Vision (maybe
```

</details>

</details>

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### mlx-community/GLM-4.6V-Flash-6bit

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-6bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible; Missing or
  empty fields: Title; Extra text appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.40s
- *Generation time:* 10.36s
- *Total time:* 12.09s
- *Input validation time:* 0.319
- *Prompt preparation time:* 0.00673
- *First-token latency:* 8.13
- *Cleanup time:* 0.0819
- *Prompt tokens:* 6,330
- *Generation tokens:* 93
- *Total tokens:* 6,423
- *Prompt throughput (raw):* 778 tok/s
- *Generation throughput (raw):* 51.9 tok/s
- *Peak memory:* 11
- *Active memory:* 9.4
- *Cache memory:* 0.45
- *Model-load active memory:* 9.43
- *Post-cleanup active memory:* 0.00349
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1389
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* df9464782d3452e0dfd86afe0984f1c9eca75ca1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-Flash-6bit/snapshots/df9464782d3452e0dfd86afe0984f1c9eca75ca1
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;|begin_of_box|&gt;Title: Two Inflatable Boats on the Sea
Description: Two inflatable motorboats with passengers speed across the blue ocean, leaving white wakes behind them under a clear sky with a distinct horizon line, captured on 2026-08-07.
Keywords: inflatable boats, motorboats, sea, passengers, wakes, clear sky, horizon line, 2026-08-07, England, UK, Europe, holiday, action, water&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Two Inflatable Boats on the Sea
Description: Two inflatable motorboats with passengers speed across the blue ocean, leaving white wakes behind them under a clear sky with a distinct horizon line, captured on 2026-08-07.
Keywords: inflatable boats, motorboats, sea, passengers, wakes, clear sky, horizon line, 2026-08-07, England, UK, Europe, holiday, action, water<|end_of_box|>
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
- *Observations:* Unrecognised model control tokens remain visible; Missing or
  empty fields: Title; Extra text appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 7.80s
- *Generation time:* 23.12s
- *Total time:* 31.24s
- *Input validation time:* 0.297
- *Prompt preparation time:* 0.00672
- *First-token latency:* 19.6
- *Cleanup time:* 0.256
- *Prompt tokens:* 6,330
- *Generation tokens:* 106
- *Total tokens:* 6,436
- *Prompt throughput (raw):* 322 tok/s
- *Generation throughput (raw):* 35.1 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00411
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1389
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-nvfp4/snapshots/2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;|begin_of_box|&gt;Title: Two inflatable boats speeding across the ocean
Description: Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line on 2026-08-07 at 17:17:16 UTC+01:00.
Keywords: inflatable boat, motorboat, ocean, speed, white wake, clear blue sky, horizon line, people, England, UK, Europe, seafront, Seaford, action, water, vehicles&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Two inflatable boats speeding across the ocean
Description: Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line on 2026-08-07 at 17:17:16 UTC+01:00.
Keywords: inflatable boat, motorboat, ocean, speed, white wake, clear blue sky, horizon line, people, England, UK, Europe, seafront, Seaford, action, water, vehicles<|end_of_box|>
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
- *Observations:* Missing or empty fields: Title, Description; Extra text
  appears before the Title field; Response appears cut off at the token limit;
  Internal reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 3.24s
- *Generation time:* 109.01s
- *Total time:* 112.56s
- *Input validation time:* 0.298
- *Prompt preparation time:* 0.00156
- *First-token latency:* 1.52
- *Cleanup time:* 0.136
- *Prompt tokens:* 1,313
- *Generation tokens:* 500
- *Total tokens:* 1,813
- *Prompt throughput (raw):* 861 tok/s
- *Generation throughput (raw):* 4.68 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.6
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00565
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1495
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Kimi-VL-A3B-Thinking-2506-bf16/snapshots/fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* [EOS]

*Readable output:*

<pre class="model-output-readable">
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.

First, the context says the capture date/time is 2026-08-07 17:17:16 UTC+01:00, but we don't need to claim it's visible. The title hint is "Seafront, Seaford, England, UK, GBR, Europe". The description hint mentions two inflatable boats with outboard motors, white wakes, clear blue sky, distinct horizon. Keywords include Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful.

Now, the image shows two motorboats (inflatable boats with outboard motors) on the sea, people on board, creating white wakes. The setting is Seaford, England, UK, Europe. The sky is clear blue, horizon is distinct. So the title should be concise, 5-10 words. Let's think: "Two inflatable motorboats speed across Seaford's sea with white wakes under a clear blue sky."

Description needs to be 1-2 sentences, combining context and main elements: "Two inflatable motorboats with outboard motors create white wakes as they speed across the sea at Seaford, England, under a clear blue sky with a distinct horizon."

Keywords: need 10-18 unique, comma-separated. Let's list them: Seaford, England, UK, Europe, Two inflatable motorboats, outboard motors, white wakes, clear blue sky, distinct horizon, people, motorboats, sea, action, holiday, vehicles, water, blue sky, horizon line, speed, ocean, boats, passengers, water splashes.

Wait, check the hints: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful. But the image might not show "Driver" or "Sailing" as the boats are moving, maybe "Riding" is better. Also, "beautiful" is a hint but maybe not visible. Let's focus on what's visible. So keywords
</pre>

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### mlx-community/Kimi-VL-A3B-Thinking-8bit

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response appears cut off at the token limit; Internal reasoning block
  appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 2.16s
- *Generation time:* 9.13s
- *Total time:* 11.59s
- *Input validation time:* 0.298
- *Prompt preparation time:* 0.00162
- *First-token latency:* 0.886
- *Cleanup time:* 0.132
- *Prompt tokens:* 1,313
- *Generation tokens:* 500
- *Total tokens:* 1,813
- *Prompt throughput (raw):* 1,481 tok/s
- *Generation throughput (raw):* 65.2 tok/s
- *Peak memory:* 22
- *Active memory:* 18
- *Cache memory:* 1.3
- *Model-load active memory:* 17.8
- *Post-cleanup active memory:* 0.00631
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1495
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 85daf3dc2490c0f824143338f08ba45f475c9ce4
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Kimi-VL-A3B-Thinking-8bit/snapshots/85daf3dc2490c0f824143338f08ba45f475c9ce4
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* [EOS]

*Readable output:*

<pre class="model-output-readable">
◁think▷Okay, let's tackle this query. The user wants me to create British-English catalogue metadata based on the image and the provided context. First, I need to parse the instructions carefully. The key points are:

1. **Title**: A concrete 5-10 word title.
2. **Description**: 1-2 sentences combining context, subject, setting, action, lighting, and details.
3. **Keywords**: 10-18 unique keywords, covering context and visible details.

The context given is:
- Capture date/time: 2026-08-07 17:17:16 UTC+01:00
- Location: Seafront, Seaford, England, UK, GBR, Europe
- Descriptive hints include seafront, Seaford, England, UK, Europe, etc.
- Keywords hints include inflatable boats, motorboats, people, etc.

The image shows two motorboats on the ocean with people on board, moving at speed, creating wakes. The sky is clear blue, horizon visible. The image is likely taken at Seaford, England, as per the context.

First, the title needs to be concise but descriptive. Maybe something like "Two Speedboats Racing Across Seaford's Clear Blue Waters" – that's 10 words. It includes the key elements: two boats, speed, location (Seaford), and the clear blue water.

For the description, I need to combine the context and visible details. The date/time is given, so maybe mention the time of day. The boats are inflatable with outboard motors, people are riding, against a clear sky. So something like "Two motorboats with outboard engines speed across the ocean near Seaford, England, under a clear blue sky on 7th August 2026 at 17:17 UTC+01:00, creating white wakes as they race against the horizon." That covers the date, location, action, and key elements.

Keywords should include all relevant terms from the context and image. From the context: Seafront, Seaford, England, UK, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Sky, UK, Vehicles, Water, action, beautiful. From the image: two motorboats, outboard motors, clear blue sky, white wakes, ocean, people on board. Need to make sure they are unique and relevant. Maybe
</pre>

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### mlx-community/LFM2.5-VL-1.6B-bf16

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.58s
- *Generation time:* 0.95s
- *Total time:* 1.84s
- *Input validation time:* 0.297
- *Prompt preparation time:* 0.00384
- *First-token latency:* 0.0973
- *Cleanup time:* 0.0694
- *Prompt tokens:* 571
- *Generation tokens:* 93
- *Total tokens:* 664
- *Prompt throughput (raw):* 5,867 tok/s
- *Generation throughput (raw):* 187 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.02
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.00657
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1359
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-1.6B-bf16/snapshots/16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Seafoam Adventure, Two Inflatable Boats, Clear Sky, Horizon, People, Motorboat, Adventure, Seaford, England, Europe, Action, Water, Adventure, Inflatable Boat, Motorboat, People, Riding, Sailing, Seafoam, Blue Sky, UK, Vehicles, Water, Action, Beautiful, 2026-08-07 17:17:16 UTC+01:00
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
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response appears cut off at the token limit; Internal reasoning block
  appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Model load time:* 0.91s
- *Generation time:* 2.75s
- *Total time:* 3.98s
- *Input validation time:* 0.299
- *Prompt preparation time:* 0.00792
- *First-token latency:* 0.36
- *Cleanup time:* 0.0789
- *Prompt tokens:* 920
- *Generation tokens:* 500
- *Total tokens:* 1,420
- *Prompt throughput (raw):* 2,557 tok/s
- *Generation throughput (raw):* 275 tok/s
- *Peak memory:* 3.8
- *Active memory:* 2.3
- *Cache memory:* 0.06
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00762
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1376
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.6-8bit/snapshots/03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;
We need to create the catalogue metadata. First, the title should be concrete and 5-10 words. Then the description should combine relevant context with main visible subjects, setting, action, lighting, and distinctive details. Then the keywords should be 10-18 unique, covering relevant context and visible details.

Let's extract info:

- The image shows two inflatable boats (inflatable boats) with people, moving across the ocean, leaving wakes. The setting is seafront, Seaford, England, UK, Europe. The time is 2026-08-07 17:17:16 UTC+01:00.

We need to describe: two inflatable boats with people, speeding across the ocean, clear sky, horizon, action (speeding, leaving wakes). Lighting seems sunny.

Now, title: need 5-10 words. Something like "Two Speed Inflatable Boats Across Ocean". But let's count: two, speed, inflatable, boats, across, ocean. That's 6 words. Maybe "Action Scenes of Inflatable Boats on Ocean". But better to be concrete.

Let's think: "Two Speed Inflatable Boats Across Blue Ocean". That's 7 words.

Now description: combine context and visible details. The context says capture date/time is 2026-08-07 17:17:16 UTC+01:00. So we can mention the time. The setting is Seaford, England, UK, Europe. The action is the boats speeding, leaving white wakes, under clear blue sky. So description: Two inflatable boats are speeding across the ocean, creating wakes, under a clear sky, in Seaford, England.

Now keywords: need 10-18 unique, covering relevant context and visible details. Let's list:

1. inflatable boats
2. speed, speeding
3. ocean, sea, water
4. blue sky, clear, sunny
5. horizon, Seaford, England, UK, Europe
6. people, passengers, riders
7. outboard motors, motorboats
8. white wakes, wakes, splashes
9. action, action scenes
10. holiday, leisure, travel
11. water, aquatic, marine
12. boats, vessels
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>
We need to create the catalogue metadata. First, the title should be concrete and 5-10 words. Then the description should combine relevant context with main visible subjects, setting, action, lighting, and distinctive details. Then the keywords should be 10-18 unique, covering relevant context and visible details.

Let's extract info:

- The image shows two inflatable boats (inflatable boats) with people, moving across the ocean, leaving wakes. The setting is seafront, Seaford, England, UK, Europe. The time is 2026-08-07 17:17:16 UTC+01:00.

We need to describe: two inflatable boats with people, speeding across the ocean, clear sky, horizon, action (speeding, leaving wakes). Lighting seems sunny.

Now, title: need 5-10 words. Something like "Two Speed Inflatable Boats Across Ocean". But let's count: two, speed, inflatable, boats, across, ocean. That's 6 words. Maybe "Action Scenes of Inflatable Boats on Ocean". But better to be concrete.

Let's think: "Two Speed Inflatable Boats Across Blue Ocean". That's 7 words.

Now description: combine context and visible details. The context says capture date/time is 2026-08-07 17:17:16 UTC+01:00. So we can mention the time. The setting is Seaford, England, UK, Europe. The action is the boats speeding, leaving white wakes, under clear blue sky. So description: Two inflatable boats are speeding across the ocean, creating wakes, under a clear sky, in Seaford, England.

Now keywords: need 10-18 unique, covering relevant context and visible details. Let's list:

1. inflatable boats
2. speed, speeding
3. ocean, sea, water
4. blue sky, clear, sunny
5. horizon, Seaford, England, UK, Europe
6. people, passengers, riders
7. outboard motors, motorboats
8. white wakes, wakes, splashes
9. action, action scenes
10. holiday, leisure, travel
11. water, aquatic, marine
12. boats, vessels
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
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type molmo_point)
- *Model load time:* 2.34s
- *Generation time:* 30.05s
- *Total time:* 32.70s
- *Input validation time:* 0.301
- *Prompt preparation time:* 0.0048
- *First-token latency:* 2.95
- *Cleanup time:* 0.11
- *Prompt tokens:* 3,111
- *Generation tokens:* 156
- *Total tokens:* 3,267
- *Prompt throughput (raw):* 1,055 tok/s
- *Generation throughput (raw):* 5.93 tok/s
- *Peak memory:* 26
- *Active memory:* 19
- *Cache memory:* 5.0
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.00965
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1346
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo_point.processing_molmo_point.MolmoPointProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MolmoPoint-8B-fp16/snapshots/0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Inflatable Boats Speeding Across Open Sea Near Seaford

Two inflatable motorboats with outboard engines race across a vast, dark blue ocean, leaving white wakes behind them. The boats carry multiple passengers and feature distinctive blue and white designs, with one displaying a blue and white flag. The scene is set against a clear blue sky with a distinct horizon line, captured on August 7, 2026, at 17:17:16 UTC+01:00.

Inflatable boat, motorboat, Seaford, England, UK, Europe, open sea, horizon, blue sky, white wake, passengers, outboard motor, action, vehicles, water, sailing, holiday, Any Vision, Adobe Stock
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
  Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 1.03s
- *Generation time:* 26.40s
- *Total time:* 27.75s
- *Input validation time:* 0.305
- *Prompt preparation time:* 0.0109
- *First-token latency:* 19.7
- *Cleanup time:* 0.135
- *Prompt tokens:* 16,630
- *Generation tokens:* 500
- *Total tokens:* 17,130
- *Prompt throughput (raw):* 844 tok/s
- *Generation throughput (raw):* 84.6 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.4
- *Model-load active memory:* 4.27
- *Post-cleanup active memory:* 0.0112
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1388
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Thinking-bf16/snapshots/c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows two inflatable boats (motorboats) on the ocean, so maybe "Two motorboats speeding across the sea" but need to be concise. Let's see: "Two inflatable motorboats racing on the ocean" – that's 8 words, good.

Next, the description. It has to combine context with visible subject, setting, action, lighting, details. Context is capture date/time, but we don't claim that. The main subject is two inflatable boats with outboard motors, speeding, leaving wakes, clear blue sky, horizon line. So: "Two inflatable motorboats with outboard motors speed across the ocean, leaving white wakes, under a clear blue sky with distinct horizon line, on a calm sea."

Wait, need to check if the context is included. The context says capture date/time is 2026-08-07 17:17:16 UTC+01:00, but we don't include that in the description. The description should be factual with visible details. The hints mention "Seafront, Seaford, England, UK, GBR, Europe" but the image doesn't show Seaford, so maybe omit that. The description should focus on what's visible: two inflatable boats (motorboats), speeding, wakes, blue sky, horizon. So: "Two inflatable motorboats with outboard motors speed across the ocean, leaving white wakes, under a clear blue sky with distinct horizon line, on a calm sea."

Now keywords. Need 10-18 unique, comma-separated. From context: UK, Europe, England, Horizon, Water, action, Motorboat, Inflatable boat, People, Sailing, Holiday, Sky, Vehicles, Sea. Let's list: Inflatable boat, Motorboat, People, Sea, Ocean, Horizon, Blue sky, Action, UK, Europe, Holiday, Sailing, Vehicles, Water. Wait, need to check if they are visible. The image has two boats, people on them, blue sky, ocean. So keywords: Inflatable boat, Motorboat, People, Sea, Ocean, Horizon, Blue sky, Action, UK, Europe, Holiday, Sailing, Vehicles, Water. Let's count: 14. Let's make sure they are unique and
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
  the token limit; Title has 6 words (requested 5-10); Keyword list has 166
  terms (requested 10-18); Duplicate keywords: seaford, england, uk, europe,
  inflatable boat, motorboat, people, horizon, blue sky, clear, water, action,
  holiday, white wake, driver, yamaha, rigid hull, lady maverick, yamaha
  engine, clear day, open sea, summer, adventure, travel, tourism, seaside,
  gbr, blue, horizon line, inflatable, outboard motor, speed boat
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.19s
- *Generation time:* 30.11s
- *Total time:* 31.61s
- *Input validation time:* 0.318
- *Prompt preparation time:* 0.000262
- *First-token latency:* 19.4
- *Cleanup time:* 0.0954
- *Prompt tokens:* 16,639
- *Generation tokens:* 500
- *Total tokens:* 17,139
- *Prompt throughput (raw):* 857 tok/s
- *Generation throughput (raw):* 50.3 tok/s
- *Peak memory:* 14
- *Active memory:* 9.5
- *Cache memory:* 1.3
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0155
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1438
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--X-Reasoner-7B-8bit/snapshots/21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Inflatable Boats Speeding Across the Ocean

Description:
Two inflatable motorboats, one labeled "Rapid," cruise across the open sea, leaving white wakes as they move, under a clear blue sky with a distinct horizon line, capturing a vibrant holiday scene.

Keywords: Seaford, England, UK, Europe, Inflatable boat, Motorboat, People, Horizon, Blue sky, Clear, Ocean, Water, Action, Holiday, Vehicles, Speeding, White wake, Driver, Yamaha, Rigid Hull, Lady Maverick, Yamaha engine, Clear day, Open sea, Summer, Adventure, Travel, Tourism, Seaside, England, UK, Europe, GBR, Blue, Horizon line, Inflatable, Outboard motor, Speed boat, Seaford, England, UK, Europe, GBR, Blue sky, Clear, Open sea, Water, Action, Holiday, People, Horizon, Inflatable boat, Motorboat, Yamaha, Rigid Hull, Lady Maverick, White wake, Driver, Yamaha engine, Clear day, Open sea, Summer, Adventure, Travel, Tourism, Seaside, England, UK, Europe, GBR, Blue, Horizon line, Inflatable, Outboard motor, Speed boat, Seaford, England, UK, Europe, GBR, Blue sky, Clear, Open sea, Water, Action, Holiday, People, Horizon, Inflatable boat, Motorboat, Yamaha, Rigid Hull, Lady Maverick, White wake, Driver, Yamaha engine, Clear day, Open sea, Summer, Adventure, Travel, Tourism, Seaside, England, UK, Europe, GBR, Blue, Horizon line, Inflatable, Outboard motor, Speed boat, Seaford, England, UK, Europe, GBR, Blue sky, Clear, Open sea, Water, Action, Holiday, People, Horizon, Inflatable boat, Motorboat, Yamaha, Rigid Hull, Lady Maverick, White wake, Driver, Yamaha engine, Clear day, Open sea, Summer, Adventure, Travel, Tourism, Seaside, England, UK, Europe, GBR, Blue, Horizon line, Inflatable, Outboard motor, Speed boat, Seaford, England, UK, Europe, GBR, Blue sky, Clear, Open sea, Water, Action, Holiday, People, Horizon, Inflatable boat, Motorboat, Yamaha, Rigid Hull,
</pre>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title; Extra text appears before
  the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.29s
- *Generation time:* 2.31s
- *Total time:* 5.94s
- *Input validation time:* 0.304
- *Prompt preparation time:* 0.0206
- *First-token latency:* 0.346
- *Cleanup time:* 0.157
- *Prompt tokens:* 584
- *Generation tokens:* 81
- *Total tokens:* 665
- *Prompt throughput (raw):* 1,689 tok/s
- *Generation throughput (raw):* 61.4 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0155
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1334
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-8bit/snapshots/7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
&lt;channel|&gt;Title: Two Inflatable Boats Speeding Across Seaford Coast
Description: Two inflatable motorboats with outboard engines speed across the deep blue sea, leaving white wakes under a clear sky.
Keywords: Seaford, England, UK, inflatable boat, motorboat, sea, water, action, people, people, horizon, sky, blue, vehicles, holiday, transport
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Two Inflatable Boats Speeding Across Seaford Coast
Description: Two inflatable motorboats with outboard engines speed across the deep blue sea, leaving white wakes under a clear sky.
Keywords: Seaford, England, UK, inflatable boat, motorboat, sea, water, action, people, people, horizon, sky, blue, vehicles, holiday, transport
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
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title; Extra text appears before
  the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.24s
- *Generation time:* 2.55s
- *Total time:* 6.12s
- *Input validation time:* 0.327
- *Prompt preparation time:* 0.000299
- *First-token latency:* 0.345
- *Cleanup time:* 0.117
- *Prompt tokens:* 584
- *Generation tokens:* 80
- *Total tokens:* 664
- *Prompt throughput (raw):* 1,691 tok/s
- *Generation throughput (raw):* 53.5 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0155
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1334
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-mxfp8/snapshots/ded389e478f86d498ad9e7f47666e83b166a28f1
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
&lt;channel|&gt;Title: Two Inflatable Boats Speeding Across Seaford Sea
Description: Two inflatable motorboats with outboard engines speed across the blue ocean water, leaving white wakes under a clear sky.
Keywords: Seaford, England, UK, inflatable boat, motorboat, action, ocean, water, horizon, sky, people, holiday, vehicles, speed, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Two Inflatable Boats Speeding Across Seaford Sea
Description: Two inflatable motorboats with outboard engines speed across the blue ocean water, leaving white wakes under a clear sky. 
Keywords: Seaford, England, UK, inflatable boat, motorboat, action, ocean, water, horizon, sky, people, holiday, vehicles, speed, Europe
```

</details>

</details>

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### mlx-community/gemma-3n-E2B-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E2B-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Missing or empty fields:
  Title, Description, Keywords; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 1.50s
- *Generation time:* 4.70s
- *Total time:* 6.52s
- *Input validation time:* 0.307
- *Prompt preparation time:* 0.000177
- *First-token latency:* 0.275
- *Cleanup time:* 0.0951
- *Prompt tokens:* 570
- *Generation tokens:* 500
- *Total tokens:* 1,070
- *Prompt throughput (raw):* 2,073 tok/s
- *Generation throughput (raw):* 126 tok/s
- *Peak memory:* 5.9
- *Active memory:* 4.5
- *Cache memory:* 0.04
- *Model-load active memory:* 4.48
- *Post-cleanup active memory:* 0.0172
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1306
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* ec68dc186276e20e4bed30b96a2b5c667e0a81e3
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3n-E2B-4bit/snapshots/ec68dc186276e20e4bed30b96a2b5c667e0a81e3
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">

- 10-18-16 2016-08-07 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16
</pre>

<details>
<summary>Exact raw output</summary>

```text

- 10-18-16 2016-08-07 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 17:17:16 
```

</details>

</details>

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### mlx-community/gemma-4-31b-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-4-31b-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* No response text was returned; Missing or empty fields:
  Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Model load time:* 7.06s
- *Generation time:* 4.57s
- *Total time:* 11.94s
- *Input validation time:* 0.308
- *Prompt preparation time:* 0.000145
- *First-token latency:* 4.01
- *Cleanup time:* 0.234
- *Prompt tokens:* 576
- *Generation tokens:* 1
- *Total tokens:* 577
- *Prompt throughput (raw):* 144 tok/s
- *Generation throughput (raw):* 49,080 tok/s
- *Peak memory:* 64
- *Active memory:* 63
- *Cache memory:* 0.77
- *Model-load active memory:* 62.6
- *Post-cleanup active memory:* 0.0187
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1297
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 19f0f1af698c51edaf1e93b3a3a5435b282de30f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-bf16/snapshots/19f0f1af698c51edaf1e93b3a3a5435b282de30f
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

#### Complete generated output

empty output

</details>

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### mlx-community/llava-v1.6-mistral-7b-8bit

<details>
<summary>Complete evidence: mlx-community/llava-v1.6-mistral-7b-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava_next)
- *Model load time:* 0.98s
- *Generation time:* 3.98s
- *Total time:* 5.26s
- *Input validation time:* 0.304
- *Prompt preparation time:* 0.00156
- *First-token latency:* 2.52
- *Cleanup time:* 0.0854
- *Prompt tokens:* 2,305
- *Generation tokens:* 52
- *Total tokens:* 2,357
- *Prompt throughput (raw):* 916 tok/s
- *Generation throughput (raw):* 61.8 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.90
- *Model-load active memory:* 8.06
- *Post-cleanup active memory:* 0.0193
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1310
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--llava-v1.6-mistral-7b-8bit/snapshots/b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Seafront, Seaford, England, UK, GBR, Europe, Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Seafront, Seaford, England, UK, GBR, Europe, Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. 
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
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava-qwen2 via
  llava_bunny)
- *Model load time:* 0.52s
- *Generation time:* 0.77s
- *Total time:* 1.60s
- *Input validation time:* 0.301
- *Prompt preparation time:* 0.00136
- *First-token latency:* 0.0798
- *Cleanup time:* 0.0799
- *Prompt tokens:* 315
- *Generation tokens:* 71
- *Total tokens:* 386
- *Prompt throughput (raw):* 3,945 tok/s
- *Generation throughput (raw):* 348 tok/s
- *Peak memory:* 2.3
- *Active memory:* 0.62
- *Cache memory:* 0.30
- *Model-load active memory:* 0.618
- *Post-cleanup active memory:* 0.02
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1394
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--nanoLLaVA-1.5-4bit/snapshots/5240204744963d72823e5de933c528c4aa82dfca
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: "Seaford in Action"
Description: A serene scene of two inflatable boats, one driving away from the other, captured in motion on a clear blue sky. The boats are driving on a calm sea, with a distinct horizon line in the background. The image is a testament to the beauty of nature and the joy of outdoor activities.
</pre>

</details>

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### mlx-community/paligemma2-10b-ft-docci-448-6bit

<details>
<summary>Complete evidence: mlx-community/paligemma2-10b-ft-docci-448-6bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Model load time:* 1.65s
- *Generation time:* 4.56s
- *Total time:* 6.52s
- *Input validation time:* 0.302
- *Prompt preparation time:* 0.000165
- *First-token latency:* 0.969
- *Cleanup time:* 0.121
- *Prompt tokens:* 1,332
- *Generation tokens:* 97
- *Total tokens:* 1,429
- *Prompt throughput (raw):* 1,375 tok/s
- *Generation throughput (raw):* 32.0 tok/s
- *Peak memory:* 12
- *Active memory:* 7.9
- *Cache memory:* 1.7
- *Model-load active memory:* 7.88
- *Post-cleanup active memory:* 0.021
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1294
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 1485fa9b3c7adb360cd354a29a401f0d441ec728
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--paligemma2-10b-ft-docci-448-6bit/snapshots/1485fa9b3c7adb360cd354a29a401f0d441ec728
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
A long shot of two inflatable boats speeding across the ocean, leaving white waves behind them. The boats are facing to the right, with the boat on the left slightly ahead of the boat on the right. The boats are both blue with white tops, and they both have people riding in them. The sky is clear and light blue, with no visible cloud coverage. The water is dark blue with small ripples on the surface. The boats are casting shadows on the water behind them.
</pre>

</details>

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### mlx-community/paligemma2-10b-ft-docci-448-bf16

<details>
<summary>Complete evidence: mlx-community/paligemma2-10b-ft-docci-448-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Model load time:* 2.44s
- *Generation time:* 22.06s
- *Total time:* 24.81s
- *Input validation time:* 0.303
- *Prompt preparation time:* 0.000127
- *First-token latency:* 1.33
- *Cleanup time:* 0.136
- *Prompt tokens:* 1,332
- *Generation tokens:* 107
- *Total tokens:* 1,439
- *Prompt throughput (raw):* 1,004 tok/s
- *Generation throughput (raw):* 5.29 tok/s
- *Peak memory:* 26
- *Active memory:* 19
- *Cache memory:* 6.2
- *Model-load active memory:* 19.3
- *Post-cleanup active memory:* 0.022
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1294
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 7c412694b919432784c730b62fadafb1c2e15d0d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--paligemma2-10b-ft-docci-448-bf16/snapshots/7c412694b919432784c730b62fadafb1c2e15d0d
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
A long shot view of two inflatable boats speeding across the ocean, leaving white waves behind them. The boats are facing to the right, with the boat on the left slightly ahead of the boat on the right. The boats are both blue with white tops, and they both have people riding on them. The boat on the left has a person standing up at the front, and the boat on the right has a person standing up at the back. The water is blue, and the sky is clear blue with no clouds in sight.
</pre>

</details>

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### mlx-community/paligemma2-3b-ft-docci-448-bf16

<details>
<summary>Complete evidence: mlx-community/paligemma2-3b-ft-docci-448-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Model load time:* 1.46s
- *Generation time:* 8.80s
- *Total time:* 10.56s
- *Input validation time:* 0.302
- *Prompt preparation time:* 0.000102
- *First-token latency:* 0.43
- *Cleanup time:* 0.1
- *Prompt tokens:* 1,332
- *Generation tokens:* 153
- *Total tokens:* 1,485
- *Prompt throughput (raw):* 3,101 tok/s
- *Generation throughput (raw):* 19.4 tok/s
- *Peak memory:* 11
- *Active memory:* 6.1
- *Cache memory:* 3.7
- *Model-load active memory:* 6.09
- *Post-cleanup active memory:* 0.0231
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1294
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* f66333527ce75342b09d4df81873f65272ec2f30
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--paligemma2-3b-ft-docci-448-bf16/snapshots/f66333527ce75342b09d4df81873f65272ec2f30
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
A medium-close-up view of two white boats that are speeding through the ocean. The boat on the left is facing towards the right, and along the front portion of the boat there is a black engine that is running. Along the top portion of the boat, there are two black poles that run vertically, and along these poles there are people who are sitting. The boat on the right is facing towards the left, and along the front portion of the boat there is a black engine that is running. Along the top portion of the boat, there are two poles that run vertically, and along these poles there are people who are sitting. The ocean is a dark blue, and along the ocean there are white waves that are being created by the boats.
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
- *Model load time:* 1.15s
- *Generation time:* 14.09s
- *Total time:* 15.56s
- *Input validation time:* 0.304
- *Prompt preparation time:* 0.00011
- *First-token latency:* 1.17
- *Cleanup time:* 0.19
- *Prompt tokens:* 4,404
- *Generation tokens:* 500
- *Total tokens:* 4,904
- *Prompt throughput (raw):* 3,751 tok/s
- *Generation throughput (raw):* 41.0 tok/s
- *Peak memory:* 4.6
- *Active memory:* 1.7
- *Cache memory:* 1.9
- *Model-load active memory:* 1.74
- *Post-cleanup active memory:* 0.0241
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1294
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--paligemma2-3b-pt-896-4bit/snapshots/a26bac48c7a661dfdafe1799c90177f818e79925
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
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
- Author
</pre>

</details>

---

<a id="model-qnguyen3-nanollava"></a>

### qnguyen3/nanoLLaVA

<details>
<summary>Complete evidence: qnguyen3/nanoLLaVA</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava-qwen2 via
  llava_bunny)
- *Model load time:* 0.53s
- *Generation time:* 0.76s
- *Total time:* 1.59s
- *Input validation time:* 0.302
- *Prompt preparation time:* 0.000563
- *First-token latency:* 0.0932
- *Cleanup time:* 0.0911
- *Prompt tokens:* 315
- *Generation tokens:* 19
- *Total tokens:* 334
- *Prompt throughput (raw):* 3,378 tok/s
- *Generation throughput (raw):* 115 tok/s
- *Peak memory:* 4.5
- *Active memory:* 2.1
- *Cache memory:* 1.6
- *Model-load active memory:* 2.13
- *Post-cleanup active memory:* 0.0252
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1394
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 13d60cec183a86755afed64da495fcc2c382ea80
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--qnguyen3--nanoLLaVA/snapshots/13d60cec183a86755afed64da495fcc2c382ea80
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 4096,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
A boat with outboard motors is driving across the ocean, leaving a wake behind it.
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
- *Error code:* MLX_MODEL_LOAD_MODEL
- *Error type:* ValueError
- *Error package:* mlx
- *Error message:* Model loading failed: Received 362 parameters not in model:
  <br>audio_tower.encoder.biases,<br>audio_tower.encoder.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.biases,<br>language_model.model.layers.10.mlp.experts.down_proj.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.weight,<br>language_model.model.layers.10.mlp.experts.gate_proj.biases,<br>language_model.model.layers.10.mlp.experts.gate_proj.scales,<br>language_model.model.layers.10.mlp.experts.gate_proj.weight,<br>language_model.model.layers.10.mlp.experts.up_proj.biases,<br>language_model.model.layers.10.mlp.experts.up_proj.scales,<br>language_model.model.layers.10.mlp.experts.up_proj.weight,<br>language_model.model.layers.11.mlp.experts.down_proj.biases,<br>language_model.model.layers.11.mlp.experts.down_proj.scales,<br>language_model.model.layers.11.mlp.experts.down_proj.weight,<br>language_model.model.layers.11.mlp.experts.gate_proj.biases,<br>language_model.model.layers.11.mlp.experts.gate_proj.scales,<br>language_model.model.layers.11.mlp.experts.gate_proj.weight,<br>language_model.model.layers.11.mlp.experts.up_proj.biases,<br>language_model.model.layers.11.mlp.experts.up_proj.scales,<br>language_model.model.layers.11.mlp.experts.up_proj.weight,<br>language_model.model.layers.12.mlp.experts.down_proj.biases,<br>language_model.model.layers.12.mlp.experts.down_proj.scales,<br>language_model.model.layers.12.mlp.experts.down_proj.weight,<br>language_model.model.layers.12.mlp.experts.gate_proj.biases,<br>language_model.model.layers.12.mlp.experts.gate_proj.scales,<br>language_model.model.layers.12.mlp.experts.gate_proj.weight,<br>language_model.model.layers.12.mlp.experts.up_proj.biases,<br>language_model.model.layers.12.mlp.experts.up_proj.scales,<br>language_model.model.layers.12.mlp.experts.up_proj.weight,<br>language_model.model.layers.13.mlp.experts.down_proj.biases,<br>language_model.model.layers.13.mlp.experts.down_proj.scales,<br>language_model.model.layers.13.mlp.experts.down_proj.weight,<br>language_model.model.layers.13.mlp.experts.gate_proj.biases,<br>language_model.model.layers.13.mlp.experts.gate_proj.scales,<br>language_model.model.layers.13.mlp.experts.gate_proj.weight,<br>language_model.model.layers.13.mlp.experts.up_proj.biases,<br>language_model.model.layers.13.mlp.experts.up_proj.scales,<br>language_model.model.layers.13.mlp.experts.up_proj.weight,<br>language_model.model.layers.14.mlp.experts.down_proj.biases,<br>language_model.model.layers.14.mlp.experts.down_proj.scales,<br>language_model.model.layers.14.mlp.experts.down_proj.weight,<br>language_model.model.layers.14.mlp.experts.gate_proj.biases,<br>language_model.model.layers.14.mlp.experts.gate_proj.scales,<br>language_model.model.layers.14.mlp.experts.gate_proj.weight,<br>language_model.model.layers.14.mlp.experts.up_proj.biases,<br>language_model.model.layers.14.mlp.experts.up_proj.scales,<br>language_model.model.layers.14.mlp.experts.up_proj.weight,<br>language_model.model.layers.15.mlp.experts.down_proj.biases,<br>language_model.model.layers.15.mlp.experts.down_proj.scales,<br>language_model.model.layers.15.mlp.experts.down_proj.weight,<br>language_model.model.layers.15.mlp.experts.gate_proj.biases,<br>language_model.model.layers.15.mlp.experts.gate_proj.scales,<br>language_model.model.layers.15.mlp.experts.gate_proj.weight,<br>language_model.model.layers.15.mlp.experts.up_proj.biases,<br>language_model.model.layers.15.mlp.experts.up_proj.scales,<br>language_model.model.layers.15.mlp.experts.up_proj.weight,<br>language_model.model.layers.16.mlp.experts.down_proj.biases,<br>language_model.model.layers.16.mlp.experts.down_proj.scales,<br>language_model.model.layers.16.mlp.experts.down_proj.weight,<br>language_model.model.layers.16.mlp.experts.gate_proj.biases,<br>language_model.model.layers.16.mlp.experts.gate_proj.scales,<br>language_model.model.layers.16.mlp.experts.gate_proj.weight,<br>language_model.model.layers.16.mlp.experts.up_proj.biases,<br>language_model.model.layers.16.mlp.experts.up_proj.scales,<br>language_model.model.layers.16.mlp.experts.up_proj.weight,<br>language_model.model.layers.17.mlp.experts.down_proj.biases,<br>language_model.model.layers.17.mlp.experts.down_proj.scales,<br>language_model.model.layers.17.mlp.experts.down_proj.weight,<br>language_model.model.layers.17.mlp.experts.gate_proj.biases,<br>language_model.model.layers.17.mlp.experts.gate_proj.scales,<br>language_model.model.layers.17.mlp.experts.gate_proj.weight,<br>language_model.model.layers.17.mlp.experts.up_proj.biases,<br>language_model.model.layers.17.mlp.experts.up_proj.scales,<br>language_model.model.layers.17.mlp.experts.up_proj.weight,<br>language_model.model.layers.18.mlp.experts.down_proj.biases,<br>language_model.model.layers.18.mlp.experts.down_proj.scales,<br>language_model.model.layers.18.mlp.experts.down_proj.weight,<br>language_model.model.layers.18.mlp.experts.gate_proj.biases,<br>language_model.model.layers.18.mlp.experts.gate_proj.scales,<br>language_model.model.layers.18.mlp.experts.gate_proj.weight,<br>language_model.model.layers.18.mlp.experts.up_proj.biases,<br>language_model.model.layers.18.mlp.experts.up_proj.scales,<br>language_model.model.layers.18.mlp.experts.up_proj.weight,<br>language_model.model.layers.19.mlp.experts.down_proj.biases,<br>language_model.model.layers.19.mlp.experts.down_proj.scales,<br>language_model.model.layers.19.mlp.experts.down_proj.weight,<br>language_model.model.layers.19.mlp.experts.gate_proj.biases,<br>language_model.model.layers.19.mlp.experts.gate_proj.scales,<br>language_model.model.layers.19.mlp.experts.gate_proj.weight,<br>language_model.model.layers.19.mlp.experts.up_proj.biases,<br>language_model.model.layers.19.mlp.experts.up_proj.scales,<br>language_model.model.layers.19.mlp.experts.up_proj.weight,<br>language_model.model.layers.2.mlp.experts.down_proj.biases,<br>language_model.model.layers.2.mlp.experts.down_proj.scales,<br>language_model.model.layers.2.mlp.experts.down_proj.weight,<br>language_model.model.layers.2.mlp.experts.gate_proj.biases,<br>language_model.model.layers.2.mlp.experts.gate_proj.scales,<br>language_model.model.layers.2.mlp.experts.gate_proj.weight,<br>language_model.model.layers.2.mlp.experts.up_proj.biases,<br>language_model.model.layers.2.mlp.experts.up_proj.scales,<br>language_model.model.layers.2.mlp.experts.up_proj.weight,<br>language_model.model.layers.20.mlp.experts.down_proj.biases,<br>language_model.model.layers.20.mlp.experts.down_proj.scales,<br>language_model.model.layers.20.mlp.experts.down_proj.weight,<br>language_model.model.layers.20.mlp.experts.gate_proj.biases,<br>language_model.model.layers.20.mlp.experts.gate_proj.scales,<br>language_model.model.layers.20.mlp.experts.gate_proj.weight,<br>language_model.model.layers.20.mlp.experts.up_proj.biases,<br>language_model.model.layers.20.mlp.experts.up_proj.scales,<br>language_model.model.layers.20.mlp.experts.up_proj.weight,<br>language_model.model.layers.21.mlp.experts.down_proj.biases,<br>language_model.model.layers.21.mlp.experts.down_proj.scales,<br>language_model.model.layers.21.mlp.experts.down_proj.weight,<br>language_model.model.layers.21.mlp.experts.gate_proj.biases,<br>language_model.model.layers.21.mlp.experts.gate_proj.scales,<br>language_model.model.layers.21.mlp.experts.gate_proj.weight,<br>language_model.model.layers.21.mlp.experts.up_proj.biases,<br>language_model.model.layers.21.mlp.experts.up_proj.scales,<br>language_model.model.layers.21.mlp.experts.up_proj.weight,<br>language_model.model.layers.22.mlp.experts.down_proj.biases,<br>language_model.model.layers.22.mlp.experts.down_proj.scales,<br>language_model.model.layers.22.mlp.experts.down_proj.weight,<br>language_model.model.layers.22.mlp.experts.gate_proj.biases,<br>language_model.model.layers.22.mlp.experts.gate_proj.scales,<br>language_model.model.layers.22.mlp.experts.gate_proj.weight,<br>language_model.model.layers.22.mlp.experts.up_proj.biases,<br>language_model.model.layers.22.mlp.experts.up_proj.scales,<br>language_model.model.layers.22.mlp.experts.up_proj.weight,<br>language_model.model.layers.23.mlp.experts.down_proj.biases,<br>language_model.model.layers.23.mlp.experts.down_proj.scales,<br>language_model.model.layers.23.mlp.experts.down_proj.weight,<br>language_model.model.layers.23.mlp.experts.gate_proj.biases,<br>language_model.model.layers.23.mlp.experts.gate_proj.scales,<br>language_model.model.layers.23.mlp.experts.gate_proj.weight,<br>language_model.model.layers.23.mlp.experts.up_proj.biases,<br>language_model.model.layers.23.mlp.experts.up_proj.scales,<br>language_model.model.layers.23.mlp.experts.up_proj.weight,<br>language_model.model.layers.24.mlp.experts.down_proj.biases,<br>language_model.model.layers.24.mlp.experts.down_proj.scales,<br>language_model.model.layers.24.mlp.experts.down_proj.weight,<br>language_model.model.layers.24.mlp.experts.gate_proj.biases,<br>language_model.model.layers.24.mlp.experts.gate_proj.scales,<br>language_model.model.layers.24.mlp.experts.gate_proj.weight,<br>language_model.model.layers.24.mlp.experts.up_proj.biases,<br>language_model.model.layers.24.mlp.experts.up_proj.scales,<br>language_model.model.layers.24.mlp.experts.up_proj.weight,<br>language_model.model.layers.25.mlp.experts.down_proj.biases,<br>language_model.model.layers.25.mlp.experts.down_proj.scales,<br>language_model.model.layers.25.mlp.experts.down_proj.weight,<br>language_model.model.layers.25.mlp.experts.gate_proj.biases,<br>language_model.model.layers.25.mlp.experts.gate_proj.scales,<br>language_model.model.layers.25.mlp.experts.gate_proj.weight,<br>language_model.model.layers.25.mlp.experts.up_proj.biases,<br>language_model.model.layers.25.mlp.experts.up_proj.scales,<br>language_model.model.layers.25.mlp.experts.up_proj.weight,<br>language_model.model.layers.26.mlp.experts.down_proj.biases,<br>language_model.model.layers.26.mlp.experts.down_proj.scales,<br>language_model.model.layers.26.mlp.experts.down_proj.weight,<br>language_model.model.layers.26.mlp.experts.gate_proj.biases,<br>language_model.model.layers.26.mlp.experts.gate_proj.scales,<br>language_model.model.layers.26.mlp.experts.gate_proj.weight,<br>language_model.model.layers.26.mlp.experts.up_proj.biases,<br>language_model.model.layers.26.mlp.experts.up_proj.scales,<br>language_model.model.layers.26.mlp.experts.up_proj.weight,<br>language_model.model.layers.27.mlp.experts.down_proj.biases,<br>language_model.model.layers.27.mlp.experts.down_proj.scales,<br>language_model.model.layers.27.mlp.experts.down_proj.weight,<br>language_model.model.layers.27.mlp.experts.gate_proj.biases,<br>language_model.model.layers.27.mlp.experts.gate_proj.scales,<br>language_model.model.layers.27.mlp.experts.gate_proj.weight,<br>language_model.model.layers.27.mlp.experts.up_proj.biases,<br>language_model.model.layers.27.mlp.experts.up_proj.scales,<br>language_model.model.layers.27.mlp.experts.up_proj.weight,<br>language_model.model.layers.28.mlp.experts.down_proj.biases,<br>language_model.model.layers.28.mlp.experts.down_proj.scales,<br>language_model.model.layers.28.mlp.experts.down_proj.weight,<br>language_model.model.layers.28.mlp.experts.gate_proj.biases,<br>language_model.model.layers.28.mlp.experts.gate_proj.scales,<br>language_model.model.layers.28.mlp.experts.gate_proj.weight,<br>language_model.model.layers.28.mlp.experts.up_proj.biases,<br>language_model.model.layers.28.mlp.experts.up_proj.scales,<br>language_model.model.layers.28.mlp.experts.up_proj.weight,<br>language_model.model.layers.29.mlp.experts.down_proj.biases,<br>language_model.model.layers.29.mlp.experts.down_proj.scales,<br>language_model.model.layers.29.mlp.experts.down_proj.weight,<br>language_model.model.layers.29.mlp.experts.gate_proj.biases,<br>language_model.model.layers.29.mlp.experts.gate_proj.scales,<br>language_model.model.layers.29.mlp.experts.gate_proj.weight,<br>language_model.model.layers.29.mlp.experts.up_proj.biases,<br>language_model.model.layers.29.mlp.experts.up_proj.scales,<br>language_model.model.layers.29.mlp.experts.up_proj.weight,<br>language_model.model.layers.3.mlp.experts.down_proj.biases,<br>language_model.model.layers.3.mlp.experts.down_proj.scales,<br>language_model.model.layers.3.mlp.experts.down_proj.weight,<br>language_model.model.layers.3.mlp.experts.gate_proj.biases,<br>language_model.model.layers.3.mlp.experts.gate_proj.scales,<br>language_model.model.layers.3.mlp.experts.gate_proj.weight,<br>language_model.model.layers.3.mlp.experts.up_proj.biases,<br>language_model.model.layers.3.mlp.experts.up_proj.scales,<br>language_model.model.layers.3.mlp.experts.up_proj.weight,<br>language_model.model.layers.30.mlp.experts.down_proj.biases,<br>language_model.model.layers.30.mlp.experts.down_proj.scales,<br>language_model.model.layers.30.mlp.experts.down_proj.weight,<br>language_model.model.layers.30.mlp.experts.gate_proj.biases,<br>language_model.model.layers.30.mlp.experts.gate_proj.scales,<br>language_model.model.layers.30.mlp.experts.gate_proj.weight,<br>language_model.model.layers.30.mlp.experts.up_proj.biases,<br>language_model.model.layers.30.mlp.experts.up_proj.scales,<br>language_model.model.layers.30.mlp.experts.up_proj.weight,<br>language_model.model.layers.31.mlp.experts.down_proj.biases,<br>language_model.model.layers.31.mlp.experts.down_proj.scales,<br>language_model.model.layers.31.mlp.experts.down_proj.weight,<br>language_model.model.layers.31.mlp.experts.gate_proj.biases,<br>language_model.model.layers.31.mlp.experts.gate_proj.scales,<br>language_model.model.layers.31.mlp.experts.gate_proj.weight,<br>language_model.model.layers.31.mlp.experts.up_proj.biases,<br>language_model.model.layers.31.mlp.experts.up_proj.scales,<br>language_model.model.layers.31.mlp.experts.up_proj.weight,<br>language_model.model.layers.32.mlp.experts.down_proj.biases,<br>language_model.model.layers.32.mlp.experts.down_proj.scales,<br>language_model.model.layers.32.mlp.experts.down_proj.weight,<br>language_model.model.layers.32.mlp.experts.gate_proj.biases,<br>language_model.model.layers.32.mlp.experts.gate_proj.scales,<br>language_model.model.layers.32.mlp.experts.gate_proj.weight,<br>language_model.model.layers.32.mlp.experts.up_proj.biases,<br>language_model.model.layers.32.mlp.experts.up_proj.scales,<br>language_model.model.layers.32.mlp.experts.up_proj.weight,<br>language_model.model.layers.33.mlp.experts.down_proj.biases,<br>language_model.model.layers.33.mlp.experts.down_proj.scales,<br>language_model.model.layers.33.mlp.experts.down_proj.weight,<br>language_model.model.layers.33.mlp.experts.gate_proj.biases,<br>language_model.model.layers.33.mlp.experts.gate_proj.scales,<br>language_model.model.layers.33.mlp.experts.gate_proj.weight,<br>language_model.model.layers.33.mlp.experts.up_proj.biases,<br>language_model.model.layers.33.mlp.experts.up_proj.scales,<br>language_model.model.layers.33.mlp.experts.up_proj.weight,<br>language_model.model.layers.34.mlp.experts.down_proj.biases,<br>language_model.model.layers.34.mlp.experts.down_proj.scales,<br>language_model.model.layers.34.mlp.experts.down_proj.weight,<br>language_model.model.layers.34.mlp.experts.gate_proj.biases,<br>language_model.model.layers.34.mlp.experts.gate_proj.scales,<br>language_model.model.layers.34.mlp.experts.gate_proj.weight,<br>language_model.model.layers.34.mlp.experts.up_proj.biases,<br>language_model.model.layers.34.mlp.experts.up_proj.scales,<br>language_model.model.layers.34.mlp.experts.up_proj.weight,<br>language_model.model.layers.35.mlp.experts.down_proj.biases,<br>language_model.model.layers.35.mlp.experts.down_proj.scales,<br>language_model.model.layers.35.mlp.experts.down_proj.weight,<br>language_model.model.layers.35.mlp.experts.gate_proj.biases,<br>language_model.model.layers.35.mlp.experts.gate_proj.scales,<br>language_model.model.layers.35.mlp.experts.gate_proj.weight,<br>language_model.model.layers.35.mlp.experts.up_proj.biases,<br>language_model.model.layers.35.mlp.experts.up_proj.scales,<br>language_model.model.layers.35.mlp.experts.up_proj.weight,<br>language_model.model.layers.36.mlp.experts.down_proj.biases,<br>language_model.model.layers.36.mlp.experts.down_proj.scales,<br>language_model.model.layers.36.mlp.experts.down_proj.weight,<br>language_model.model.layers.36.mlp.experts.gate_proj.biases,<br>language_model.model.layers.36.mlp.experts.gate_proj.scales,<br>language_model.model.layers.36.mlp.experts.gate_proj.weight,<br>language_model.model.layers.36.mlp.experts.up_proj.biases,<br>language_model.model.layers.36.mlp.experts.up_proj.scales,<br>language_model.model.layers.36.mlp.experts.up_proj.weight,<br>language_model.model.layers.37.mlp.experts.down_proj.biases,<br>language_model.model.layers.37.mlp.experts.down_proj.scales,<br>language_model.model.layers.37.mlp.experts.down_proj.weight,<br>language_model.model.layers.37.mlp.experts.gate_proj.biases,<br>language_model.model.layers.37.mlp.experts.gate_proj.scales,<br>language_model.model.layers.37.mlp.experts.gate_proj.weight,<br>language_model.model.layers.37.mlp.experts.up_proj.biases,<br>language_model.model.layers.37.mlp.experts.up_proj.scales,<br>language_model.model.layers.37.mlp.experts.up_proj.weight,<br>language_model.model.layers.38.mlp.experts.down_proj.biases,<br>language_model.model.layers.38.mlp.experts.down_proj.scales,<br>language_model.model.layers.38.mlp.experts.down_proj.weight,<br>language_model.model.layers.38.mlp.experts.gate_proj.biases,<br>language_model.model.layers.38.mlp.experts.gate_proj.scales,<br>language_model.model.layers.38.mlp.experts.gate_proj.weight,<br>language_model.model.layers.38.mlp.experts.up_proj.biases,<br>language_model.model.layers.38.mlp.experts.up_proj.scales,<br>language_model.model.layers.38.mlp.experts.up_proj.weight,<br>language_model.model.layers.39.mlp.experts.down_proj.biases,<br>language_model.model.layers.39.mlp.experts.down_proj.scales,<br>language_model.model.layers.39.mlp.experts.down_proj.weight,<br>language_model.model.layers.39.mlp.experts.gate_proj.biases,<br>language_model.model.layers.39.mlp.experts.gate_proj.scales,<br>language_model.model.layers.39.mlp.experts.gate_proj.weight,<br>language_model.model.layers.39.mlp.experts.up_proj.biases,<br>language_model.model.layers.39.mlp.experts.up_proj.scales,<br>language_model.model.layers.39.mlp.experts.up_proj.weight,<br>language_model.model.layers.4.mlp.experts.down_proj.biases,<br>language_model.model.layers.4.mlp.experts.down_proj.scales,<br>language_model.model.layers.4.mlp.experts.down_proj.weight,<br>language_model.model.layers.4.mlp.experts.gate_proj.biases,<br>language_model.model.layers.4.mlp.experts.gate_proj.scales,<br>language_model.model.layers.4.mlp.experts.gate_proj.weight,<br>language_model.model.layers.4.mlp.experts.up_proj.biases,<br>language_model.model.layers.4.mlp.experts.up_proj.scales,<br>language_model.model.layers.4.mlp.experts.up_proj.weight,<br>language_model.model.layers.40.mlp.experts.down_proj.biases,<br>language_model.model.layers.40.mlp.experts.down_proj.scales,<br>language_model.model.layers.40.mlp.experts.down_proj.weight,<br>language_model.model.layers.40.mlp.experts.gate_proj.biases,<br>language_model.model.layers.40.mlp.experts.gate_proj.scales,<br>language_model.model.layers.40.mlp.experts.gate_proj.weight,<br>language_model.model.layers.40.mlp.experts.up_proj.biases,<br>language_model.model.layers.40.mlp.experts.up_proj.scales,<br>language_model.model.layers.40.mlp.experts.up_proj.weight,<br>language_model.model.layers.41.mlp.experts.down_proj.biases,<br>language_model.model.layers.41.mlp.experts.down_proj.scales,<br>language_model.model.layers.41.mlp.experts.down_proj.weight,<br>language_model.model.layers.41.mlp.experts.gate_proj.biases,<br>language_model.model.layers.41.mlp.experts.gate_proj.scales,<br>language_model.model.layers.41.mlp.experts.gate_proj.weight,<br>language_model.model.layers.41.mlp.experts.up_proj.biases,<br>language_model.model.layers.41.mlp.experts.up_proj.scales,<br>language_model.model.layers.41.mlp.experts.up_proj.weight,<br>language_model.model.layers.5.mlp.experts.down_proj.biases,<br>language_model.model.layers.5.mlp.experts.down_proj.scales,<br>language_model.model.layers.5.mlp.experts.down_proj.weight,<br>language_model.model.layers.5.mlp.experts.gate_proj.biases,<br>language_model.model.layers.5.mlp.experts.gate_proj.scales,<br>language_model.model.layers.5.mlp.experts.gate_proj.weight,<br>language_model.model.layers.5.mlp.experts.up_proj.biases,<br>language_model.model.layers.5.mlp.experts.up_proj.scales,<br>language_model.model.layers.5.mlp.experts.up_proj.weight,<br>language_model.model.layers.6.mlp.experts.down_proj.biases,<br>language_model.model.layers.6.mlp.experts.down_proj.scales,<br>language_model.model.layers.6.mlp.experts.down_proj.weight,<br>language_model.model.layers.6.mlp.experts.gate_proj.biases,<br>language_model.model.layers.6.mlp.experts.gate_proj.scales,<br>language_model.model.layers.6.mlp.experts.gate_proj.weight,<br>language_model.model.layers.6.mlp.experts.up_proj.biases,<br>language_model.model.layers.6.mlp.experts.up_proj.scales,<br>language_model.model.layers.6.mlp.experts.up_proj.weight,<br>language_model.model.layers.7.mlp.experts.down_proj.biases,<br>language_model.model.layers.7.mlp.experts.down_proj.scales,<br>language_model.model.layers.7.mlp.experts.down_proj.weight,<br>language_model.model.layers.7.mlp.experts.gate_proj.biases,<br>language_model.model.layers.7.mlp.experts.gate_proj.scales,<br>language_model.model.layers.7.mlp.experts.gate_proj.weight,<br>language_model.model.layers.7.mlp.experts.up_proj.biases,<br>language_model.model.layers.7.mlp.experts.up_proj.scales,<br>language_model.model.layers.7.mlp.experts.up_proj.weight,<br>language_model.model.layers.8.mlp.experts.down_proj.biases,<br>language_model.model.layers.8.mlp.experts.down_proj.scales,<br>language_model.model.layers.8.mlp.experts.down_proj.weight,<br>language_model.model.layers.8.mlp.experts.gate_proj.biases,<br>language_model.model.layers.8.mlp.experts.gate_proj.scales,<br>language_model.model.layers.8.mlp.experts.gate_proj.weight,<br>language_model.model.layers.8.mlp.experts.up_proj.biases,<br>language_model.model.layers.8.mlp.experts.up_proj.scales,<br>language_model.model.layers.8.mlp.experts.up_proj.weight,<br>language_model.model.layers.9.mlp.experts.down_proj.biases,<br>language_model.model.layers.9.mlp.experts.down_proj.scales,<br>language_model.model.layers.9.mlp.experts.down_proj.weight,<br>language_model.model.layers.9.mlp.experts.gate_proj.biases,<br>language_model.model.layers.9.mlp.experts.gate_proj.scales,<br>language_model.model.layers.9.mlp.experts.gate_proj.weight,<br>language_model.model.layers.9.mlp.experts.up_proj.biases,<br>language_model.model.layers.9.mlp.experts.up_proj.scales,<br>language_model.model.layers.9.mlp.experts.up_proj.weight.
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* Received 362 parameters not in model:
  <br>audio_tower.encoder.biases,<br>audio_tower.encoder.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.biases,<br>language_model.model.layers.10.mlp.experts.down_proj.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.weight,<br>language_model.model.layers.10.mlp.experts.gate_proj.biases,<br>language_model.model.layers.10.mlp.experts.gate_proj.scales,<br>language_model.model.layers.10.mlp.experts.gate_proj.weight,<br>language_model.model.layers.10.mlp.experts.up_proj.biases,<br>language_model.model.layers.10.mlp.experts.up_proj.scales,<br>language_model.model.layers.10.mlp.experts.up_proj.weight,<br>language_model.model.layers.11.mlp.experts.down_proj.biases,<br>language_model.model.layers.11.mlp.experts.down_proj.scales,<br>language_model.model.layers.11.mlp.experts.down_proj.weight,<br>language_model.model.layers.11.mlp.experts.gate_proj.biases,<br>language_model.model.layers.11.mlp.experts.gate_proj.scales,<br>language_model.model.layers.11.mlp.experts.gate_proj.weight,<br>language_model.model.layers.11.mlp.experts.up_proj.biases,<br>language_model.model.layers.11.mlp.experts.up_proj.scales,<br>language_model.model.layers.11.mlp.experts.up_proj.weight,<br>language_model.model.layers.12.mlp.experts.down_proj.biases,<br>language_model.model.layers.12.mlp.experts.down_proj.scales,<br>language_model.model.layers.12.mlp.experts.down_proj.weight,<br>language_model.model.layers.12.mlp.experts.gate_proj.biases,<br>language_model.model.layers.12.mlp.experts.gate_proj.scales,<br>language_model.model.layers.12.mlp.experts.gate_proj.weight,<br>language_model.model.layers.12.mlp.experts.up_proj.biases,<br>language_model.model.layers.12.mlp.experts.up_proj.scales,<br>language_model.model.layers.12.mlp.experts.up_proj.weight,<br>language_model.model.layers.13.mlp.experts.down_proj.biases,<br>language_model.model.layers.13.mlp.experts.down_proj.scales,<br>language_model.model.layers.13.mlp.experts.down_proj.weight,<br>language_model.model.layers.13.mlp.experts.gate_proj.biases,<br>language_model.model.layers.13.mlp.experts.gate_proj.scales,<br>language_model.model.layers.13.mlp.experts.gate_proj.weight,<br>language_model.model.layers.13.mlp.experts.up_proj.biases,<br>language_model.model.layers.13.mlp.experts.up_proj.scales,<br>language_model.model.layers.13.mlp.experts.up_proj.weight,<br>language_model.model.layers.14.mlp.experts.down_proj.biases,<br>language_model.model.layers.14.mlp.experts.down_proj.scales,<br>language_model.model.layers.14.mlp.experts.down_proj.weight,<br>language_model.model.layers.14.mlp.experts.gate_proj.biases,<br>language_model.model.layers.14.mlp.experts.gate_proj.scales,<br>language_model.model.layers.14.mlp.experts.gate_proj.weight,<br>language_model.model.layers.14.mlp.experts.up_proj.biases,<br>language_model.model.layers.14.mlp.experts.up_proj.scales,<br>language_model.model.layers.14.mlp.experts.up_proj.weight,<br>language_model.model.layers.15.mlp.experts.down_proj.biases,<br>language_model.model.layers.15.mlp.experts.down_proj.scales,<br>language_model.model.layers.15.mlp.experts.down_proj.weight,<br>language_model.model.layers.15.mlp.experts.gate_proj.biases,<br>language_model.model.layers.15.mlp.experts.gate_proj.scales,<br>language_model.model.layers.15.mlp.experts.gate_proj.weight,<br>language_model.model.layers.15.mlp.experts.up_proj.biases,<br>language_model.model.layers.15.mlp.experts.up_proj.scales,<br>language_model.model.layers.15.mlp.experts.up_proj.weight,<br>language_model.model.layers.16.mlp.experts.down_proj.biases,<br>language_model.model.layers.16.mlp.experts.down_proj.scales,<br>language_model.model.layers.16.mlp.experts.down_proj.weight,<br>language_model.model.layers.16.mlp.experts.gate_proj.biases,<br>language_model.model.layers.16.mlp.experts.gate_proj.scales,<br>language_model.model.layers.16.mlp.experts.gate_proj.weight,<br>language_model.model.layers.16.mlp.experts.up_proj.biases,<br>language_model.model.layers.16.mlp.experts.up_proj.scales,<br>language_model.model.layers.16.mlp.experts.up_proj.weight,<br>language_model.model.layers.17.mlp.experts.down_proj.biases,<br>language_model.model.layers.17.mlp.experts.down_proj.scales,<br>language_model.model.layers.17.mlp.experts.down_proj.weight,<br>language_model.model.layers.17.mlp.experts.gate_proj.biases,<br>language_model.model.layers.17.mlp.experts.gate_proj.scales,<br>language_model.model.layers.17.mlp.experts.gate_proj.weight,<br>language_model.model.layers.17.mlp.experts.up_proj.biases,<br>language_model.model.layers.17.mlp.experts.up_proj.scales,<br>language_model.model.layers.17.mlp.experts.up_proj.weight,<br>language_model.model.layers.18.mlp.experts.down_proj.biases,<br>language_model.model.layers.18.mlp.experts.down_proj.scales,<br>language_model.model.layers.18.mlp.experts.down_proj.weight,<br>language_model.model.layers.18.mlp.experts.gate_proj.biases,<br>language_model.model.layers.18.mlp.experts.gate_proj.scales,<br>language_model.model.layers.18.mlp.experts.gate_proj.weight,<br>language_model.model.layers.18.mlp.experts.up_proj.biases,<br>language_model.model.layers.18.mlp.experts.up_proj.scales,<br>language_model.model.layers.18.mlp.experts.up_proj.weight,<br>language_model.model.layers.19.mlp.experts.down_proj.biases,<br>language_model.model.layers.19.mlp.experts.down_proj.scales,<br>language_model.model.layers.19.mlp.experts.down_proj.weight,<br>language_model.model.layers.19.mlp.experts.gate_proj.biases,<br>language_model.model.layers.19.mlp.experts.gate_proj.scales,<br>language_model.model.layers.19.mlp.experts.gate_proj.weight,<br>language_model.model.layers.19.mlp.experts.up_proj.biases,<br>language_model.model.layers.19.mlp.experts.up_proj.scales,<br>language_model.model.layers.19.mlp.experts.up_proj.weight,<br>language_model.model.layers.2.mlp.experts.down_proj.biases,<br>language_model.model.layers.2.mlp.experts.down_proj.scales,<br>language_model.model.layers.2.mlp.experts.down_proj.weight,<br>language_model.model.layers.2.mlp.experts.gate_proj.biases,<br>language_model.model.layers.2.mlp.experts.gate_proj.scales,<br>language_model.model.layers.2.mlp.experts.gate_proj.weight,<br>language_model.model.layers.2.mlp.experts.up_proj.biases,<br>language_model.model.layers.2.mlp.experts.up_proj.scales,<br>language_model.model.layers.2.mlp.experts.up_proj.weight,<br>language_model.model.layers.20.mlp.experts.down_proj.biases,<br>language_model.model.layers.20.mlp.experts.down_proj.scales,<br>language_model.model.layers.20.mlp.experts.down_proj.weight,<br>language_model.model.layers.20.mlp.experts.gate_proj.biases,<br>language_model.model.layers.20.mlp.experts.gate_proj.scales,<br>language_model.model.layers.20.mlp.experts.gate_proj.weight,<br>language_model.model.layers.20.mlp.experts.up_proj.biases,<br>language_model.model.layers.20.mlp.experts.up_proj.scales,<br>language_model.model.layers.20.mlp.experts.up_proj.weight,<br>language_model.model.layers.21.mlp.experts.down_proj.biases,<br>language_model.model.layers.21.mlp.experts.down_proj.scales,<br>language_model.model.layers.21.mlp.experts.down_proj.weight,<br>language_model.model.layers.21.mlp.experts.gate_proj.biases,<br>language_model.model.layers.21.mlp.experts.gate_proj.scales,<br>language_model.model.layers.21.mlp.experts.gate_proj.weight,<br>language_model.model.layers.21.mlp.experts.up_proj.biases,<br>language_model.model.layers.21.mlp.experts.up_proj.scales,<br>language_model.model.layers.21.mlp.experts.up_proj.weight,<br>language_model.model.layers.22.mlp.experts.down_proj.biases,<br>language_model.model.layers.22.mlp.experts.down_proj.scales,<br>language_model.model.layers.22.mlp.experts.down_proj.weight,<br>language_model.model.layers.22.mlp.experts.gate_proj.biases,<br>language_model.model.layers.22.mlp.experts.gate_proj.scales,<br>language_model.model.layers.22.mlp.experts.gate_proj.weight,<br>language_model.model.layers.22.mlp.experts.up_proj.biases,<br>language_model.model.layers.22.mlp.experts.up_proj.scales,<br>language_model.model.layers.22.mlp.experts.up_proj.weight,<br>language_model.model.layers.23.mlp.experts.down_proj.biases,<br>language_model.model.layers.23.mlp.experts.down_proj.scales,<br>language_model.model.layers.23.mlp.experts.down_proj.weight,<br>language_model.model.layers.23.mlp.experts.gate_proj.biases,<br>language_model.model.layers.23.mlp.experts.gate_proj.scales,<br>language_model.model.layers.23.mlp.experts.gate_proj.weight,<br>language_model.model.layers.23.mlp.experts.up_proj.biases,<br>language_model.model.layers.23.mlp.experts.up_proj.scales,<br>language_model.model.layers.23.mlp.experts.up_proj.weight,<br>language_model.model.layers.24.mlp.experts.down_proj.biases,<br>language_model.model.layers.24.mlp.experts.down_proj.scales,<br>language_model.model.layers.24.mlp.experts.down_proj.weight,<br>language_model.model.layers.24.mlp.experts.gate_proj.biases,<br>language_model.model.layers.24.mlp.experts.gate_proj.scales,<br>language_model.model.layers.24.mlp.experts.gate_proj.weight,<br>language_model.model.layers.24.mlp.experts.up_proj.biases,<br>language_model.model.layers.24.mlp.experts.up_proj.scales,<br>language_model.model.layers.24.mlp.experts.up_proj.weight,<br>language_model.model.layers.25.mlp.experts.down_proj.biases,<br>language_model.model.layers.25.mlp.experts.down_proj.scales,<br>language_model.model.layers.25.mlp.experts.down_proj.weight,<br>language_model.model.layers.25.mlp.experts.gate_proj.biases,<br>language_model.model.layers.25.mlp.experts.gate_proj.scales,<br>language_model.model.layers.25.mlp.experts.gate_proj.weight,<br>language_model.model.layers.25.mlp.experts.up_proj.biases,<br>language_model.model.layers.25.mlp.experts.up_proj.scales,<br>language_model.model.layers.25.mlp.experts.up_proj.weight,<br>language_model.model.layers.26.mlp.experts.down_proj.biases,<br>language_model.model.layers.26.mlp.experts.down_proj.scales,<br>language_model.model.layers.26.mlp.experts.down_proj.weight,<br>language_model.model.layers.26.mlp.experts.gate_proj.biases,<br>language_model.model.layers.26.mlp.experts.gate_proj.scales,<br>language_model.model.layers.26.mlp.experts.gate_proj.weight,<br>language_model.model.layers.26.mlp.experts.up_proj.biases,<br>language_model.model.layers.26.mlp.experts.up_proj.scales,<br>language_model.model.layers.26.mlp.experts.up_proj.weight,<br>language_model.model.layers.27.mlp.experts.down_proj.biases,<br>language_model.model.layers.27.mlp.experts.down_proj.scales,<br>language_model.model.layers.27.mlp.experts.down_proj.weight,<br>language_model.model.layers.27.mlp.experts.gate_proj.biases,<br>language_model.model.layers.27.mlp.experts.gate_proj.scales,<br>language_model.model.layers.27.mlp.experts.gate_proj.weight,<br>language_model.model.layers.27.mlp.experts.up_proj.biases,<br>language_model.model.layers.27.mlp.experts.up_proj.scales,<br>language_model.model.layers.27.mlp.experts.up_proj.weight,<br>language_model.model.layers.28.mlp.experts.down_proj.biases,<br>language_model.model.layers.28.mlp.experts.down_proj.scales,<br>language_model.model.layers.28.mlp.experts.down_proj.weight,<br>language_model.model.layers.28.mlp.experts.gate_proj.biases,<br>language_model.model.layers.28.mlp.experts.gate_proj.scales,<br>language_model.model.layers.28.mlp.experts.gate_proj.weight,<br>language_model.model.layers.28.mlp.experts.up_proj.biases,<br>language_model.model.layers.28.mlp.experts.up_proj.scales,<br>language_model.model.layers.28.mlp.experts.up_proj.weight,<br>language_model.model.layers.29.mlp.experts.down_proj.biases,<br>language_model.model.layers.29.mlp.experts.down_proj.scales,<br>language_model.model.layers.29.mlp.experts.down_proj.weight,<br>language_model.model.layers.29.mlp.experts.gate_proj.biases,<br>language_model.model.layers.29.mlp.experts.gate_proj.scales,<br>language_model.model.layers.29.mlp.experts.gate_proj.weight,<br>language_model.model.layers.29.mlp.experts.up_proj.biases,<br>language_model.model.layers.29.mlp.experts.up_proj.scales,<br>language_model.model.layers.29.mlp.experts.up_proj.weight,<br>language_model.model.layers.3.mlp.experts.down_proj.biases,<br>language_model.model.layers.3.mlp.experts.down_proj.scales,<br>language_model.model.layers.3.mlp.experts.down_proj.weight,<br>language_model.model.layers.3.mlp.experts.gate_proj.biases,<br>language_model.model.layers.3.mlp.experts.gate_proj.scales,<br>language_model.model.layers.3.mlp.experts.gate_proj.weight,<br>language_model.model.layers.3.mlp.experts.up_proj.biases,<br>language_model.model.layers.3.mlp.experts.up_proj.scales,<br>language_model.model.layers.3.mlp.experts.up_proj.weight,<br>language_model.model.layers.30.mlp.experts.down_proj.biases,<br>language_model.model.layers.30.mlp.experts.down_proj.scales,<br>language_model.model.layers.30.mlp.experts.down_proj.weight,<br>language_model.model.layers.30.mlp.experts.gate_proj.biases,<br>language_model.model.layers.30.mlp.experts.gate_proj.scales,<br>language_model.model.layers.30.mlp.experts.gate_proj.weight,<br>language_model.model.layers.30.mlp.experts.up_proj.biases,<br>language_model.model.layers.30.mlp.experts.up_proj.scales,<br>language_model.model.layers.30.mlp.experts.up_proj.weight,<br>language_model.model.layers.31.mlp.experts.down_proj.biases,<br>language_model.model.layers.31.mlp.experts.down_proj.scales,<br>language_model.model.layers.31.mlp.experts.down_proj.weight,<br>language_model.model.layers.31.mlp.experts.gate_proj.biases,<br>language_model.model.layers.31.mlp.experts.gate_proj.scales,<br>language_model.model.layers.31.mlp.experts.gate_proj.weight,<br>language_model.model.layers.31.mlp.experts.up_proj.biases,<br>language_model.model.layers.31.mlp.experts.up_proj.scales,<br>language_model.model.layers.31.mlp.experts.up_proj.weight,<br>language_model.model.layers.32.mlp.experts.down_proj.biases,<br>language_model.model.layers.32.mlp.experts.down_proj.scales,<br>language_model.model.layers.32.mlp.experts.down_proj.weight,<br>language_model.model.layers.32.mlp.experts.gate_proj.biases,<br>language_model.model.layers.32.mlp.experts.gate_proj.scales,<br>language_model.model.layers.32.mlp.experts.gate_proj.weight,<br>language_model.model.layers.32.mlp.experts.up_proj.biases,<br>language_model.model.layers.32.mlp.experts.up_proj.scales,<br>language_model.model.layers.32.mlp.experts.up_proj.weight,<br>language_model.model.layers.33.mlp.experts.down_proj.biases,<br>language_model.model.layers.33.mlp.experts.down_proj.scales,<br>language_model.model.layers.33.mlp.experts.down_proj.weight,<br>language_model.model.layers.33.mlp.experts.gate_proj.biases,<br>language_model.model.layers.33.mlp.experts.gate_proj.scales,<br>language_model.model.layers.33.mlp.experts.gate_proj.weight,<br>language_model.model.layers.33.mlp.experts.up_proj.biases,<br>language_model.model.layers.33.mlp.experts.up_proj.scales,<br>language_model.model.layers.33.mlp.experts.up_proj.weight,<br>language_model.model.layers.34.mlp.experts.down_proj.biases,<br>language_model.model.layers.34.mlp.experts.down_proj.scales,<br>language_model.model.layers.34.mlp.experts.down_proj.weight,<br>language_model.model.layers.34.mlp.experts.gate_proj.biases,<br>language_model.model.layers.34.mlp.experts.gate_proj.scales,<br>language_model.model.layers.34.mlp.experts.gate_proj.weight,<br>language_model.model.layers.34.mlp.experts.up_proj.biases,<br>language_model.model.layers.34.mlp.experts.up_proj.scales,<br>language_model.model.layers.34.mlp.experts.up_proj.weight,<br>language_model.model.layers.35.mlp.experts.down_proj.biases,<br>language_model.model.layers.35.mlp.experts.down_proj.scales,<br>language_model.model.layers.35.mlp.experts.down_proj.weight,<br>language_model.model.layers.35.mlp.experts.gate_proj.biases,<br>language_model.model.layers.35.mlp.experts.gate_proj.scales,<br>language_model.model.layers.35.mlp.experts.gate_proj.weight,<br>language_model.model.layers.35.mlp.experts.up_proj.biases,<br>language_model.model.layers.35.mlp.experts.up_proj.scales,<br>language_model.model.layers.35.mlp.experts.up_proj.weight,<br>language_model.model.layers.36.mlp.experts.down_proj.biases,<br>language_model.model.layers.36.mlp.experts.down_proj.scales,<br>language_model.model.layers.36.mlp.experts.down_proj.weight,<br>language_model.model.layers.36.mlp.experts.gate_proj.biases,<br>language_model.model.layers.36.mlp.experts.gate_proj.scales,<br>language_model.model.layers.36.mlp.experts.gate_proj.weight,<br>language_model.model.layers.36.mlp.experts.up_proj.biases,<br>language_model.model.layers.36.mlp.experts.up_proj.scales,<br>language_model.model.layers.36.mlp.experts.up_proj.weight,<br>language_model.model.layers.37.mlp.experts.down_proj.biases,<br>language_model.model.layers.37.mlp.experts.down_proj.scales,<br>language_model.model.layers.37.mlp.experts.down_proj.weight,<br>language_model.model.layers.37.mlp.experts.gate_proj.biases,<br>language_model.model.layers.37.mlp.experts.gate_proj.scales,<br>language_model.model.layers.37.mlp.experts.gate_proj.weight,<br>language_model.model.layers.37.mlp.experts.up_proj.biases,<br>language_model.model.layers.37.mlp.experts.up_proj.scales,<br>language_model.model.layers.37.mlp.experts.up_proj.weight,<br>language_model.model.layers.38.mlp.experts.down_proj.biases,<br>language_model.model.layers.38.mlp.experts.down_proj.scales,<br>language_model.model.layers.38.mlp.experts.down_proj.weight,<br>language_model.model.layers.38.mlp.experts.gate_proj.biases,<br>language_model.model.layers.38.mlp.experts.gate_proj.scales,<br>language_model.model.layers.38.mlp.experts.gate_proj.weight,<br>language_model.model.layers.38.mlp.experts.up_proj.biases,<br>language_model.model.layers.38.mlp.experts.up_proj.scales,<br>language_model.model.layers.38.mlp.experts.up_proj.weight,<br>language_model.model.layers.39.mlp.experts.down_proj.biases,<br>language_model.model.layers.39.mlp.experts.down_proj.scales,<br>language_model.model.layers.39.mlp.experts.down_proj.weight,<br>language_model.model.layers.39.mlp.experts.gate_proj.biases,<br>language_model.model.layers.39.mlp.experts.gate_proj.scales,<br>language_model.model.layers.39.mlp.experts.gate_proj.weight,<br>language_model.model.layers.39.mlp.experts.up_proj.biases,<br>language_model.model.layers.39.mlp.experts.up_proj.scales,<br>language_model.model.layers.39.mlp.experts.up_proj.weight,<br>language_model.model.layers.4.mlp.experts.down_proj.biases,<br>language_model.model.layers.4.mlp.experts.down_proj.scales,<br>language_model.model.layers.4.mlp.experts.down_proj.weight,<br>language_model.model.layers.4.mlp.experts.gate_proj.biases,<br>language_model.model.layers.4.mlp.experts.gate_proj.scales,<br>language_model.model.layers.4.mlp.experts.gate_proj.weight,<br>language_model.model.layers.4.mlp.experts.up_proj.biases,<br>language_model.model.layers.4.mlp.experts.up_proj.scales,<br>language_model.model.layers.4.mlp.experts.up_proj.weight,<br>language_model.model.layers.40.mlp.experts.down_proj.biases,<br>language_model.model.layers.40.mlp.experts.down_proj.scales,<br>language_model.model.layers.40.mlp.experts.down_proj.weight,<br>language_model.model.layers.40.mlp.experts.gate_proj.biases,<br>language_model.model.layers.40.mlp.experts.gate_proj.scales,<br>language_model.model.layers.40.mlp.experts.gate_proj.weight,<br>language_model.model.layers.40.mlp.experts.up_proj.biases,<br>language_model.model.layers.40.mlp.experts.up_proj.scales,<br>language_model.model.layers.40.mlp.experts.up_proj.weight,<br>language_model.model.layers.41.mlp.experts.down_proj.biases,<br>language_model.model.layers.41.mlp.experts.down_proj.scales,<br>language_model.model.layers.41.mlp.experts.down_proj.weight,<br>language_model.model.layers.41.mlp.experts.gate_proj.biases,<br>language_model.model.layers.41.mlp.experts.gate_proj.scales,<br>language_model.model.layers.41.mlp.experts.gate_proj.weight,<br>language_model.model.layers.41.mlp.experts.up_proj.biases,<br>language_model.model.layers.41.mlp.experts.up_proj.scales,<br>language_model.model.layers.41.mlp.experts.up_proj.weight,<br>language_model.model.layers.5.mlp.experts.down_proj.biases,<br>language_model.model.layers.5.mlp.experts.down_proj.scales,<br>language_model.model.layers.5.mlp.experts.down_proj.weight,<br>language_model.model.layers.5.mlp.experts.gate_proj.biases,<br>language_model.model.layers.5.mlp.experts.gate_proj.scales,<br>language_model.model.layers.5.mlp.experts.gate_proj.weight,<br>language_model.model.layers.5.mlp.experts.up_proj.biases,<br>language_model.model.layers.5.mlp.experts.up_proj.scales,<br>language_model.model.layers.5.mlp.experts.up_proj.weight,<br>language_model.model.layers.6.mlp.experts.down_proj.biases,<br>language_model.model.layers.6.mlp.experts.down_proj.scales,<br>language_model.model.layers.6.mlp.experts.down_proj.weight,<br>language_model.model.layers.6.mlp.experts.gate_proj.biases,<br>language_model.model.layers.6.mlp.experts.gate_proj.scales,<br>language_model.model.layers.6.mlp.experts.gate_proj.weight,<br>language_model.model.layers.6.mlp.experts.up_proj.biases,<br>language_model.model.layers.6.mlp.experts.up_proj.scales,<br>language_model.model.layers.6.mlp.experts.up_proj.weight,<br>language_model.model.layers.7.mlp.experts.down_proj.biases,<br>language_model.model.layers.7.mlp.experts.down_proj.scales,<br>language_model.model.layers.7.mlp.experts.down_proj.weight,<br>language_model.model.layers.7.mlp.experts.gate_proj.biases,<br>language_model.model.layers.7.mlp.experts.gate_proj.scales,<br>language_model.model.layers.7.mlp.experts.gate_proj.weight,<br>language_model.model.layers.7.mlp.experts.up_proj.biases,<br>language_model.model.layers.7.mlp.experts.up_proj.scales,<br>language_model.model.layers.7.mlp.experts.up_proj.weight,<br>language_model.model.layers.8.mlp.experts.down_proj.biases,<br>language_model.model.layers.8.mlp.experts.down_proj.scales,<br>language_model.model.layers.8.mlp.experts.down_proj.weight,<br>language_model.model.layers.8.mlp.experts.gate_proj.biases,<br>language_model.model.layers.8.mlp.experts.gate_proj.scales,<br>language_model.model.layers.8.mlp.experts.gate_proj.weight,<br>language_model.model.layers.8.mlp.experts.up_proj.biases,<br>language_model.model.layers.8.mlp.experts.up_proj.scales,<br>language_model.model.layers.8.mlp.experts.up_proj.weight,<br>language_model.model.layers.9.mlp.experts.down_proj.biases,<br>language_model.model.layers.9.mlp.experts.down_proj.scales,<br>language_model.model.layers.9.mlp.experts.down_proj.weight,<br>language_model.model.layers.9.mlp.experts.gate_proj.biases,<br>language_model.model.layers.9.mlp.experts.gate_proj.scales,<br>language_model.model.layers.9.mlp.experts.gate_proj.weight,<br>language_model.model.layers.9.mlp.experts.up_proj.biases,<br>language_model.model.layers.9.mlp.experts.up_proj.scales,<br>language_model.model.layers.9.mlp.experts.up_proj.weight.
- *Arch supported by installed mlx-vlm:* yes (model_type inkling_mm_model via
  inkling)
- *Model load time:* 0.21s
- *Generation time:* -
- *Total time:* 0.51s
- *Input validation time:* 0.3
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0615
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.00438
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11995, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11349, in _load_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12419, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12010, in _run_model_generation
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
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 3995.00it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[12:57:22] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
```

</details>

---
